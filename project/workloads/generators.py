"""
5개 워크로드 생성기. 각 워크로드는 학생 1명이 담당한다.

각 생성기의 파라미터는 **담당자가 자기 병목을 만드는 축을 스윕(sweep)해서 곡선을
그릴 수 있도록** 노출되어 있다. 보고서에는 숫자 하나가 아니라 곡선이 들어가야 한다.

사용법:
    python -m workloads.generators all --out traces/ --model Qwen/Qwen3-4B
    python -m workloads.generators agent --out traces/ --think-mean 30
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.trace import Request, Trace
from common.textgen import make_text, lognormal_int, poisson_arrivals

SYS_PROMPT = (
    "You are a precise assistant. Answer using only the provided context. "
    "If the context is insufficient, say so explicitly. Be concise.\n\n"
)


# =============================================================================
# W1 — 롱컨텍스트 RAG              병목: PREFILL + KV 메모리
# =============================================================================
# 발견해야 할 현상:
#   prefill 이 E2E 지연의 대부분을 차지한다. 그리고 같은 청크가 반복 검색되는데도
#   radix cache 적중률이 거의 0 에 머문다. 이유는 아래 rng.shuffle 한 줄이다.
#   검색된 청크의 **순서**가 요청마다 달라서, 접두사(prefix) 매칭이 첫 청크에서
#   끊기기 때문이다. --reuse-alpha 를 올려 재사용률을 높여도 적중률은 오르지 않는다.
#   이 간극이 곧 과제다.
# =============================================================================
def gen_rag(n=200, pool=400, chunks_per_req=6, chunk_tokens=1200,
            qps=0.6, reuse_alpha=1.1, model=None, seed=1) -> Trace:
    rng = random.Random(seed)
    docs = {i: make_text(chunk_tokens, seed=100000 + i, model=model)
            for i in range(pool)}

    # Zipf 형태의 인기도: 소수의 청크가 계속 검색된다 (실제 RAG 의 특성)
    w = [1.0 / ((i + 1) ** reuse_alpha) for i in range(pool)]
    s = sum(w)
    cum, acc = [], 0.0
    for x in w:
        acc += x / s
        cum.append(acc)

    def pick() -> int:
        x, lo, hi = rng.random(), 0, pool - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        return lo

    arrivals = poisson_arrivals(rng, qps, n)
    reqs = []
    for i in range(n):
        ids, seen = [], set()
        while len(ids) < chunks_per_req:
            d = pick()
            if d not in seen:
                seen.add(d)
                ids.append(d)
        rng.shuffle(ids)          # <<<< 검색 순서가 매번 다르다. 이 한 줄이 과제의 핵심.
        ctx = "\n\n".join(f"[{j+1}] {docs[d]}" for j, d in enumerate(ids))
        q = make_text(40, seed=900000 + i, model=model)
        reqs.append(Request(
            rid=f"rag-{i:05d}", session_id=f"rag-s{i:05d}",
            arrival_time=arrivals[i],
            prompt=f"{SYS_PROMPT}Context:\n{ctx}\n\nQuestion: {q}\nAnswer:",
            max_new_tokens=lognormal_int(rng, 140, 0.4, 48, 400),
            temperature=0.0, ttft_slo_ms=4000,
            tags={"workload": "rag", "chunk_ids": ids,
                  "ctx_tokens": chunks_per_req * chunk_tokens}))
    return Trace("rag", reqs, meta={
        "bottleneck": "prefill 지배적 / 위치 의존적 KV 재사용 실패",
        "sweep": "reuse_alpha, chunks_per_req, chunk_tokens",
        "pool": pool, "chunks_per_req": chunks_per_req,
        "chunk_tokens": chunk_tokens, "qps": qps, "reuse_alpha": reuse_alpha})


# =============================================================================
# W2 — AI 에이전트                 병목: 캐시 유지(retention) / 스케줄러
# =============================================================================
# 발견해야 할 현상:
#   세션의 프리픽스는 턴이 진행될수록 단조 증가한다(1.7k → 6k 토큰). 그런데 툴
#   호출로 20초를 쉬는 동안 LRU 로 evict 되고, 다음 턴에서 6k 토큰을 통째로 다시
#   prefill 한다. --think-mean 을 1초에서 60초까지 스윕하면서 cache_hit_pct 가
#   절벽처럼 떨어지는 지점을 찾아라. 그 지점이 캐시의 유효 보존 시간이다.
# =============================================================================
def gen_agent(sessions=40, turns=6, tool_defs_tokens=1500, tool_result_tokens=600,
              think_mean=8.0, think_sigma=0.9, session_qps=0.5,
              model=None, seed=2) -> Trace:
    rng = random.Random(seed)
    starts = poisson_arrivals(rng, session_qps, sessions)
    reqs = []
    for s in range(sessions):
        # 툴 정의 블록은 5종뿐 -> 세션 간에도 공유 프리픽스가 존재한다
        tools = make_text(tool_defs_tokens, seed=200000 + s % 5, model=model)
        goal = make_text(60, seed=300000 + s, model=model)
        head = (f"{SYS_PROMPT}Available tools:\n{tools}\n\n"
                f"User goal: {goal}\n\nAssistant:")
        prev = None
        for t in range(turns):
            rid = f"agent-{s:04d}-t{t}"
            if t == 0:
                r = Request(rid=rid, session_id=f"agent-{s:04d}", turn_idx=0,
                            arrival_time=starts[s], prompt=head)
            else:
                obs = make_text(
                    lognormal_int(rng, tool_result_tokens, 0.7, 100, 4000),
                    seed=400000 + s * 100 + t, model=model)
                # think_time: 툴 실행 / 사용자 확인에 걸리는 시간. evict 의 원인.
                tt = rng.lognormvariate(
                    math.log(think_mean) - 0.5 * think_sigma ** 2, think_sigma)
                r = Request(rid=rid, session_id=f"agent-{s:04d}", turn_idx=t,
                            depends_on=prev, think_time=round(tt, 3),
                            suffix=f"\n\nTool result:\n{obs}\n\nAssistant:")
            r.max_new_tokens = lognormal_int(rng, 160, 0.5, 40, 500)
            r.temperature, r.ttft_slo_ms = 0.0, 1500
            r.tags = {"workload": "agent", "session_idx": s, "turn": t}
            reqs.append(r)
            prev = rid
    return Trace("agent", reqs, meta={
        "bottleneck": "think time 동안의 프리픽스 evict / 세션 인지 캐시 유지",
        "sweep": "think_mean, sessions, turns",
        "sessions": sessions, "turns": turns, "think_mean": think_mean,
        "tool_defs_tokens": tool_defs_tokens})


# =============================================================================
# W3 — 장문 추론(Long CoT)          병목: DECODE / KV 증가
# =============================================================================
# 발견해야 할 현상:
#   prefill 은 극히 짧고 decode 가 수천 토큰 이어진다. 시퀀스당 KV 가 수 분 동안
#   계속 커진다. 메모리가 차면 running batch 가 붕괴하고 스케줄러가 요청을
#   retract(선점) 하기 시작한다. 서버 로그에서 `#retracted` 를 찾아라. 이 값이
#   0 이 아니게 되는 QPS 지점이 이 과제의 출발점이다.
# =============================================================================
def gen_reasoning(n=120, prompt_tokens=350, out_mean=3000, out_sigma=0.6,
                  qps=0.35, model=None, seed=3) -> Trace:
    rng = random.Random(seed)
    arrivals = poisson_arrivals(rng, qps, n)
    reqs = []
    for i in range(n):
        p = make_text(prompt_tokens, seed=500000 + i, model=model)
        reqs.append(Request(
            rid=f"reason-{i:05d}", session_id=f"reason-s{i:05d}",
            arrival_time=arrivals[i],
            prompt=f"{SYS_PROMPT}Problem: {p}\n\nThink step by step, then answer.\n",
            # 출력 길이 분포가 롱테일이다. 스케줄러는 이 길이를 미리 알 수 없다.
            max_new_tokens=lognormal_int(rng, out_mean, out_sigma, 256, 16000),
            temperature=0.0, ignore_eos=True, tpot_slo_ms=60,
            tags={"workload": "reasoning"}))
    return Trace("reasoning", reqs, meta={
        "bottleneck": "decode 지배적 / KV 증가, 배치 붕괴, retraction",
        "sweep": "qps, out_mean, out_sigma",
        "out_mean": out_mean, "qps": qps})


# =============================================================================
# W4 — 구조화 출력(JSON 추출)       병목: 요청당 CPU 오버헤드
# =============================================================================
# 발견해야 할 현상:
#   모든 요청이 작다. GPU 는 놀고 있다. 비용은 전부 CPU 쪽이다 — 스케줄러 루프,
#   detokenize, 문법(grammar) 컴파일, HTTP. 스키마 분포를 일부러 치우치게 했다
#   (invoice/ticket 이 85%). 문법 캐시가 동작한다면 희귀 스키마의 첫 요청에서만
#   컴파일 비용이 보여야 한다. 실제로 그런지 rid 순서대로 TTFT 를 찍어보라.
# =============================================================================
SCHEMAS = {
    "invoice": {"type": "object", "properties": {
        "vendor": {"type": "string"}, "total": {"type": "number"},
        "date": {"type": "string"}, "line_items": {"type": "array", "items": {
            "type": "object", "properties": {
                "sku": {"type": "string"}, "qty": {"type": "integer"}}}}},
        "required": ["vendor", "total", "date"]},
    "resume": {"type": "object", "properties": {
        "name": {"type": "string"}, "years_experience": {"type": "integer"},
        "skills": {"type": "array", "items": {"type": "string"}}},
        "required": ["name", "skills"]},
    "ticket": {"type": "object", "properties": {
        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
        "component": {"type": "string"}, "summary": {"type": "string"}},
        "required": ["severity", "component"]},
    "contract": {"type": "object", "properties": {
        "parties": {"type": "array", "items": {"type": "string"}},
        "effective_date": {"type": "string"},
        "termination_days": {"type": "integer"}},
        "required": ["parties"]},
}


def gen_structured(n=1500, doc_tokens=450, qps=12.0, model=None, seed=4) -> Trace:
    rng = random.Random(seed)
    arrivals = poisson_arrivals(rng, qps, n)
    names = list(SCHEMAS)
    weights = [0.45, 0.08, 0.4, 0.07]   # 치우친 스키마 분포
    reqs = []
    for i in range(n):
        name = rng.choices(names, weights=weights)[0]
        doc = make_text(doc_tokens, seed=600000 + i, model=model)
        reqs.append(Request(
            rid=f"struct-{i:05d}", session_id=f"struct-s{i:05d}",
            arrival_time=arrivals[i],
            prompt=(f"{SYS_PROMPT}Extract fields as JSON matching the "
                    f"'{name}' schema.\n\nDocument:\n{doc}\n\nJSON:"),
            max_new_tokens=lognormal_int(rng, 110, 0.35, 32, 300),
            temperature=0.0, ttft_slo_ms=500, tpot_slo_ms=25,
            tags={"workload": "structured", "schema_name": name,
                  "json_schema": SCHEMAS[name]}))
    return Trace("structured", reqs, meta={
        "bottleneck": "CPU/요청당 오버헤드 / 문법 컴파일 + 캐시",
        "sweep": "qps, doc_tokens, 스키마 분포",
        "qps": qps, "schemas": list(SCHEMAS)})


# =============================================================================
# W5 — 혼합 SLO / 멀티테넌트        병목: 스케줄링 정책
# =============================================================================
# 발견해야 할 현상:
#   12k 토큰짜리 배치 요청이 버스트로 몰려오면 대화형 요청의 P99 TTFT 가 폭발한다
#   (head-of-line blocking). SGLang 기본 정책인 LPM(longest-prefix-match)은
#   처리량에는 최적이지만 이 상황을 악화시킨다. 반드시 `--by slo_class` 로 나눠
#   측정하라. 전체 평균만 보면 아무 문제도 없어 보인다.
# =============================================================================
def gen_mixed(inter_n=900, inter_qps=8.0, batch_n=60, burst_size=12,
              batch_prompt_tokens=12000, model=None, seed=5) -> Trace:
    rng = random.Random(seed)
    reqs = []

    arrivals = poisson_arrivals(rng, inter_qps, inter_n)
    for i in range(inter_n):
        p = make_text(lognormal_int(rng, 320, 0.6, 64, 2000),
                      seed=700000 + i, model=model)
        reqs.append(Request(
            rid=f"inter-{i:05d}", session_id=f"inter-s{i:05d}",
            arrival_time=arrivals[i],
            prompt=f"{SYS_PROMPT}{p}\n\nAnswer:",
            max_new_tokens=lognormal_int(rng, 120, 0.5, 32, 400),
            temperature=0.0, slo_class="interactive",
            ttft_slo_ms=500, tpot_slo_ms=40, tags={"workload": "mixed"}))

    horizon = arrivals[-1] if arrivals else 60.0
    n_bursts = max(1, batch_n // burst_size)
    for b in range(n_bursts):
        t0 = round(horizon * (b + 0.5) / n_bursts, 3)
        for j in range(burst_size):
            p = make_text(batch_prompt_tokens, seed=800000 + b * 100 + j, model=model)
            reqs.append(Request(
                rid=f"batch-{b:03d}-{j:03d}", session_id=f"batch-{b:03d}",
                arrival_time=round(t0 + j * 0.05, 3),
                prompt=f"{SYS_PROMPT}Summarize:\n{p}\n\nSummary:",
                max_new_tokens=lognormal_int(rng, 700, 0.4, 200, 2000),
                temperature=0.0, slo_class="batch", tags={"workload": "mixed"}))

    reqs.sort(key=lambda r: r.arrival_time or 0.0)
    return Trace("mixed", reqs, meta={
        "bottleneck": "head-of-line blocking / SLO 인지 스케줄링·admission control",
        "sweep": "inter_qps, burst_size, batch_prompt_tokens",
        "inter_qps": inter_qps, "burst_size": burst_size,
        "batch_prompt_tokens": batch_prompt_tokens})


GENERATORS = {"rag": gen_rag, "agent": gen_agent, "reasoning": gen_reasoning,
              "structured": gen_structured, "mixed": gen_mixed}


def main() -> None:
    ap = argparse.ArgumentParser(description="워크로드 트레이스 생성기")
    ap.add_argument("which", choices=list(GENERATORS) + ["all"])
    ap.add_argument("--out", default="traces")
    ap.add_argument("--model", default=None,
                    help="HF 모델 ID. 지정하면 토큰 길이가 정확해짐. 제출용은 필수.")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="요청 수를 줄여 저비용 스모크 테스트용 트레이스 생성")
    ap.add_argument("--suffix", default="", help="파일명 접미사 (스윕 실험용)")
    ap.add_argument("--set", nargs="*", default=[], metavar="K=V",
                    help="생성기 파라미터 오버라이드. 예: --set think_mean=30 qps=1.2")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = float(v) if "." in v else int(v)

    for name in (list(GENERATORS) if args.which == "all" else [args.which]):
        fn = GENERATORS[name]
        sig = inspect.signature(fn).parameters
        kw = {"model": args.model}
        if args.scale != 1.0:
            for p in ("n", "sessions", "inter_n", "batch_n"):
                if p in sig:
                    kw[p] = max(2, int(sig[p].default * args.scale))
        for k, v in overrides.items():
            if k in sig:
                kw[k] = v
        tr = fn(**kw)
        path = os.path.join(args.out, f"{name}{args.suffix}.jsonl")
        tr.save(path)
        print(f"{path}\n  {tr.describe()}\n  병목: {tr.meta.get('bottleneck')}")


if __name__ == "__main__":
    main()
