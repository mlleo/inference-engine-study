"""
오프라인 지표 계산 및 비교표 출력. GPU 를 전혀 쓰지 않으므로 몇 번이든 재실행 가능.

    python -m bench.metrics 'results/agent__*.json'
    python -m bench.metrics 'results/mixed__*.json' --by slo_class    # 테넌트별
    python -m bench.metrics 'results/agent__*.json' --by turn_idx     # 턴별
    python -m bench.metrics 'results/structured__*.json' --by schema_name

지표 정의:
  TTFT   요청 제출 → 첫 토큰 수신. prefill + 큐 대기 시간을 모두 포함한다.
  TPOT   첫 토큰 이후 토큰당 평균 시간. decode 속도.
  maxITL 한 요청 안에서 겪은 최악의 토큰 간 정체. 사용자가 "멈췄다"고 느끼는 순간.
  goodput SLO 를 만족한 요청 수 / 초.  <-- 이게 헤드라인 지표다.

throughput 이 아니라 goodput 을 보고하라. "처리량 40% 향상"인데 P99 TTFT 가
400ms → 9s 가 되었다면 그건 개선이 아니라 회귀다. 아래 표는 그걸 드러낸다.
"""

from __future__ import annotations

import argparse
import glob
import json


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def per_request(rec: dict) -> dict:
    out = {"rid": rec["rid"], "error": rec["error"]}
    if rec["error"] or rec["first_token_t"] is None:
        return out
    out["ttft_ms"] = (rec["first_token_t"] - rec["submit_t"]) * 1000
    out["e2e_ms"] = (rec["end_t"] - rec["submit_t"]) * 1000
    n = rec["completion_tokens"]
    out["completion_tokens"] = n
    out["prompt_tokens"] = rec["prompt_tokens"]
    out["cached_tokens"] = rec["cached_tokens"]
    if n > 1:
        out["tpot_ms"] = (rec["end_t"] - rec["first_token_t"]) * 1000 / (n - 1)
    gaps = [(b - a) * 1000 for a, b in zip(rec["chunk_times"], rec["chunk_times"][1:])]
    if gaps:
        out["max_itl_ms"] = max(gaps)
    ok = True
    if rec["ttft_slo_ms"] is not None:
        ok = ok and out["ttft_ms"] <= rec["ttft_slo_ms"]
    if rec["tpot_slo_ms"] is not None and "tpot_ms" in out:
        ok = ok and out["tpot_ms"] <= rec["tpot_slo_ms"]
    out["slo_ok"] = ok
    out["slo_class"] = rec["slo_class"]
    return out


def summarize(blob: dict, filt=None) -> dict:
    recs = [r for r in blob["records"] if not filt or filt(r)]
    rows = [per_request(r) for r in recs]
    good = [r for r in rows if not r["error"] and "ttft_ms" in r]
    if not good:
        return {"tag": blob["tag"], "n": len(rows), "n_ok": 0}

    ends = [r["end_t"] for r in recs if r["end_t"]]
    subs = [r["submit_t"] for r in recs if r["submit_t"]]
    span = (max(ends) - min(subs)) if ends and subs else 0.0
    out_tok = sum(r["completion_tokens"] for r in good)
    in_tok = sum(r["prompt_tokens"] for r in good)
    cached = sum(r["cached_tokens"] for r in good)

    return {
        "tag": blob["tag"], "n": len(rows), "n_ok": len(good),
        "n_err": sum(1 for r in rows if r["error"]),
        "wall_s": round(span, 1),
        "ttft_p50": pct([r["ttft_ms"] for r in good], 50),
        "ttft_p99": pct([r["ttft_ms"] for r in good], 99),
        "tpot_p50": pct([r["tpot_ms"] for r in good if "tpot_ms" in r], 50),
        "tpot_p99": pct([r["tpot_ms"] for r in good if "tpot_ms" in r], 99),
        "maxitl_p99": pct([r["max_itl_ms"] for r in good if "max_itl_ms" in r], 99),
        "e2e_p99": pct([r["e2e_ms"] for r in good], 99),
        "out_tok_s": out_tok / span if span else 0,
        # 캐시 작업의 핵심 지표: 전체 프롬프트 토큰 중 재사용된 비율
        "cache_hit_%": 100.0 * cached / in_tok if in_tok else 0.0,
        "slo_%": 100.0 * sum(1 for r in good if r["slo_ok"]) / len(good),
        "goodput_rps": sum(1 for r in good if r["slo_ok"]) / span if span else 0,
    }


COLS = ["tag", "n_ok", "n_err", "wall_s", "ttft_p50", "ttft_p99", "tpot_p50",
        "tpot_p99", "maxitl_p99", "e2e_p99", "out_tok_s", "cache_hit_%",
        "slo_%", "goodput_rps"]


def table(rows: list[dict]) -> str:
    if not rows:
        return "(데이터 없음)"
    def cell(r, c):
        v = r.get(c, "")
        return f"{v:.1f}" if isinstance(v, float) else str(v)
    w = {c: max(len(c), max(len(cell(r, c)) for r in rows)) for c in COLS}
    lines = [" | ".join(c.rjust(w[c]) for c in COLS),
             "-+-".join("-" * w[c] for c in COLS)]
    for r in rows:
        lines.append(" | ".join(cell(r, c).rjust(w[c]) for c in COLS))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="오프라인 지표 계산")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--by", default=None,
                    help="분해 기준 필드. 예: slo_class, turn_idx, schema_name")
    ap.add_argument("--csv", default=None, help="CSV 로도 저장")
    args = ap.parse_args()

    paths = [p for pat in args.files for p in sorted(glob.glob(pat))]
    if not paths:
        print("파일을 찾지 못했다. 패턴을 따옴표로 감쌌는지 확인할 것.")
        return
    blobs = [json.load(open(p)) for p in paths]

    all_rows = []
    if not args.by:
        rows = [summarize(b) for b in blobs]
        print(table(rows))
        all_rows = rows
    else:
        def key(r):
            return r.get(args.by, (r.get("tags") or {}).get(args.by))
        vals = sorted({str(key(r)) for b in blobs for r in b["records"]},
                      key=lambda x: (len(x), x))
        for v in vals:
            rows = [summarize(b, filt=lambda r, v=v: str(key(r)) == v) for b in blobs]
            rows = [r for r in rows if r.get("n_ok")]
            if not rows:
                continue
            print(f"\n=== {args.by} = {v} ===")
            print(table(rows))
            for r in rows:
                r[args.by] = v
            all_rows += rows

    if args.csv:
        import csv
        cols = ([args.by] if args.by else []) + COLS
        with open(args.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(all_rows)
        print(f"\nCSV 저장: {args.csv}")


if __name__ == "__main__":
    main()
