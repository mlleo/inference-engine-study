"""
병목 진단 리포트. 2주차 '병목 분석' 단계의 주력 도구.

metrics.py 가 "얼마나 느린가"를 알려준다면, 이 도구는 "왜 느린가"를 알려준다.
GPU 없이 결과 파일만으로 동작한다.

    python -m bench.analyze results/rag__default.json

출력 5종:
  1. E2E 시간 분해        prefill(TTFT) 지배 vs decode 지배 판정
  2. 시간대별 동시 요청 수  큐 적체가 언제 시작되는지
  3. 도착률 vs 완료율      서버가 유입을 따라가고 있는가
  4. 프롬프트 길이별 TTFT   prefill 비용이 길이에 어떻게 반응하는가
  5. 캐시 적중 분해        턴/스키마별 재사용 실태
"""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from bench.metrics import per_request, pct


def bar(v: float, vmax: float, width: int = 40) -> str:
    if vmax <= 0:
        return ""
    return "█" * max(0, min(width, round(width * v / vmax)))


def sec_1_breakdown(rows: list[dict]) -> None:
    print("\n[1] E2E 시간 분해 — 이 워크로드는 무엇에 지배되는가")
    print("-" * 68)
    fr = [r["ttft_ms"] / r["e2e_ms"] for r in rows if r.get("e2e_ms", 0) > 0]
    if not fr:
        return
    med = pct(fr, 50)
    print(f"  TTFT 가 E2E 에서 차지하는 비중  p50={med:.1%}  "
          f"p90={pct(fr, 90):.1%}")
    print(f"  TTFT  p50={pct([r['ttft_ms'] for r in rows], 50):8.1f} ms   "
          f"p99={pct([r['ttft_ms'] for r in rows], 99):9.1f} ms")
    tp = [r["tpot_ms"] for r in rows if "tpot_ms" in r]
    if tp:
        print(f"  TPOT  p50={pct(tp, 50):8.1f} ms   p99={pct(tp, 99):9.1f} ms")
    verdict = ("PREFILL 지배 — prefill 경로(청킹, 캐시, KV 재사용)를 파라"
               if med > 0.5 else
               "DECODE 지배 — 배치 크기·KV 용량·어텐션 커널을 파라"
               if med < 0.2 else
               "혼재 — 요청 종류별로 나눠서 다시 보라 (--by 사용)")
    print(f"  => 판정: {verdict}")


def sec_2_concurrency(recs: list[dict]) -> None:
    print("\n[2] 시간대별 서버 내 동시 요청 수 — 큐 적체 시점")
    print("-" * 68)
    ev = []
    for r in recs:
        if r["submit_t"] is None or r["end_t"] is None:
            continue
        ev.append((r["submit_t"], 1))
        ev.append((r["end_t"], -1))
    if not ev:
        return
    ev.sort()
    T = ev[-1][0]
    nb = 30
    peak = [0] * nb
    cur = 0
    for t, d in ev:
        cur += d
        b = min(nb - 1, int(nb * t / T)) if T > 0 else 0
        peak[b] = max(peak[b], cur)
    vmax = max(peak) or 1
    for i, v in enumerate(peak):
        print(f"  {T*i/nb:6.0f}s {v:5d} {bar(v, vmax)}")
    print(f"  => 최대 동시 요청 {vmax}. 후반으로 갈수록 계속 커지면 서버가 "
          f"유입을 못 따라가는 것이다.")


def sec_3_arrival_vs_completion(recs: list[dict]) -> None:
    print("\n[3] 도착률 vs 완료율 (요청/초)")
    print("-" * 68)
    subs = sorted(r["submit_t"] for r in recs if r["submit_t"] is not None)
    ends = sorted(r["end_t"] for r in recs if r["end_t"] is not None)
    if not subs or not ends:
        return
    T = max(ends)
    nb = 12
    ain = [0] * nb
    aout = [0] * nb
    for t in subs:
        ain[min(nb - 1, int(nb * t / T))] += 1
    for t in ends:
        aout[min(nb - 1, int(nb * t / T))] += 1
    dt = T / nb
    print(f"  {'구간':>10} {'도착/s':>8} {'완료/s':>8}   누적 미완료")
    backlog = 0
    for i in range(nb):
        backlog += ain[i] - aout[i]
        print(f"  {T*i/nb:8.0f}s {ain[i]/dt:8.2f} {aout[i]/dt:8.2f}   "
              f"{backlog:6d} {bar(max(backlog,0), max(1,len(recs)//4), 24)}")
    print("  => 누적 미완료가 단조 증가하면 시스템이 포화(saturated)된 것이다.")


def sec_4_ttft_vs_len(rows: list[dict]) -> None:
    print("\n[4] 프롬프트 길이 구간별 TTFT / 캐시 적중")
    print("-" * 68)
    have = [r for r in rows if r.get("prompt_tokens")]
    if not have:
        return
    have.sort(key=lambda r: r["prompt_tokens"])
    nb = min(8, max(2, len(have) // 20))
    size = len(have) // nb
    print(f"  {'프롬프트토큰':>14} {'건수':>6} {'TTFT p50':>10} {'TTFT p99':>10} "
          f"{'캐시적중':>8}")
    for i in range(nb):
        g = have[i * size:(i + 1) * size] if i < nb - 1 else have[i * size:]
        if not g:
            continue
        pt = sum(r["prompt_tokens"] for r in g)
        ch = sum(r["cached_tokens"] for r in g)
        print(f"  {g[0]['prompt_tokens']:6d}-{g[-1]['prompt_tokens']:<7d} "
              f"{len(g):6d} {pct([r['ttft_ms'] for r in g], 50):10.1f} "
              f"{pct([r['ttft_ms'] for r in g], 99):10.1f} "
              f"{100*ch/pt if pt else 0:7.1f}%")
    print("  => TTFT 가 길이에 선형 비례하면 순수 prefill 비용이다. 길이와 무관하게"
          "\n     전 구간이 높으면 큐 대기가 섞인 것이다.")


def sec_5_cache(recs: list[dict], rows: list[dict]) -> None:
    by = {}
    for rec, row in zip(recs, rows):
        if "ttft_ms" not in row:
            continue
        tags = rec.get("tags") or {}
        k = tags.get("turn", tags.get("schema_name", rec.get("turn_idx")))
        if k is None:
            continue
        d = by.setdefault(str(k), [0, 0, [], 0])
        d[0] += rec["prompt_tokens"]
        d[1] += rec["cached_tokens"]
        d[2].append(row["ttft_ms"])
        d[3] += 1
    if len(by) < 2:
        return
    print("\n[5] 그룹별 캐시 적중률 (턴 / 스키마)")
    print("-" * 68)
    print(f"  {'그룹':>8} {'건수':>6} {'평균프롬프트':>12} {'캐시적중':>9} {'TTFT p50':>10}")
    for k in sorted(by, key=lambda x: (len(x), x)):
        pt, ch, tt, n = by[k]
        print(f"  {k:>8} {n:6d} {pt/n:12.0f} {100*ch/pt if pt else 0:8.1f}% "
              f"{pct(tt, 50):10.1f}")
    print("  => 턴이 올라가는데 적중률이 오르지 않으면 프리픽스가 evict 되고 있다.")


def main() -> None:
    ap = argparse.ArgumentParser(description="병목 진단 리포트")
    ap.add_argument("file")
    args = ap.parse_args()
    blob = json.load(open(args.file))
    recs = [r for r in blob["records"] if not r["error"]]
    rows = [per_request(r) for r in recs]
    rows = [r for r in rows if "ttft_ms" in r]

    print("=" * 68)
    print(f"병목 진단  트레이스={blob['trace']}  실험={blob['tag']}  "
          f"유효요청={len(rows)}")
    print(f"메타: {blob.get('meta', {}).get('bottleneck', '')}")
    print("=" * 68)
    sec_1_breakdown(rows)
    sec_2_concurrency(recs)
    sec_3_arrival_vs_completion(recs)
    sec_4_ttft_vs_len(rows)
    sec_5_cache(recs, [per_request(r) for r in recs])
    print("\n주의: 클라이언트 측 관점만 담긴 리포트다. 원인 확정은 반드시 서버 로그"
          "\n(running-req, token usage, #retracted, cache hit rate)와 대조할 것.")


if __name__ == "__main__":
    main()
