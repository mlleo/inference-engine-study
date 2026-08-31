"""
트레이스 재생 하니스(클라이언트).

반드시 지켜야 할 설계 원칙 3가지:

1. OPEN LOOP (개방 루프)
   독립 요청은 서버가 밀리든 말든 벽시계 기준으로 발사한다. 고정 동시성(concurrency)
   방식의 폐쇄 루프 클라이언트는 서버가 느려지면 스스로도 느려져서, 정확히 우리가
   보려는 큐잉 붕괴 현상을 감춰버린다. W3/W5 과제는 이걸 틀리면 아무것도 못 본다.

2. 의존성 인지
   연쇄 요청은 부모가 **완료된 뒤** think_time 만큼 기다렸다가 발사되고, 프롬프트는
   실제로 부모_프롬프트 + 부모_출력 + suffix 로 조립된다. 공유 프리픽스가 진짜로
   생기므로 cached_tokens 가 의미 있는 값이 된다.

3. 원본 기록 저장
   요청별 원시 타임스탬프를 전부 파일로 남긴다. 지표 계산은 bench/metrics.py 에서
   오프라인으로 한다. **GPU 시간이 비싸다. 백분위 하나 바꾸려고 트레이스를 재실행
   하는 일은 절대 없어야 한다.**

사용법:
    python -m bench.replay --trace traces/agent.jsonl \\
        --url http://127.0.0.1:30000 --tag baseline --out results/ --keep-output
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.trace import Request, Trace


class Result:
    __slots__ = ("rid", "session_id", "turn_idx", "slo_class", "tags",
                 "submit_t", "first_token_t", "end_t", "chunk_times",
                 "chunk_tokens", "prompt_tokens", "cached_tokens",
                 "completion_tokens", "output", "error", "ttft_slo_ms",
                 "tpot_slo_ms")

    def __init__(self, r: Request):
        self.rid, self.session_id = r.rid, r.session_id
        self.turn_idx, self.slo_class, self.tags = r.turn_idx, r.slo_class, r.tags
        self.ttft_slo_ms, self.tpot_slo_ms = r.ttft_slo_ms, r.tpot_slo_ms
        self.submit_t = self.first_token_t = self.end_t = None
        self.chunk_times, self.chunk_tokens = [], []
        self.prompt_tokens = self.cached_tokens = self.completion_tokens = 0
        self.output, self.error = "", None

    def to_dict(self, t0: float, keep_output: bool) -> dict:
        d = {k: getattr(self, k) for k in self.__slots__ if k != "output"}
        for k in ("submit_t", "first_token_t", "end_t"):
            if d[k] is not None:
                d[k] = round(d[k] - t0, 6)
        d["chunk_times"] = [round(t - t0, 6) for t in self.chunk_times]
        if keep_output:
            d["output"] = self.output
        return d


async def _one(session, url: str, r: Request, prompt: str, res: Result,
               timeout: float, use_grammar: bool) -> None:
    sp = {
        "max_new_tokens": r.max_new_tokens,
        "temperature": r.temperature,
        "top_p": r.top_p,
        "ignore_eos": r.ignore_eos,
    }
    if r.stop:
        sp["stop"] = r.stop
    # W4 담당자용: --grammar 플래그를 켜면 JSON 스키마 제약이 활성화된다
    if use_grammar and "json_schema" in (r.tags or {}):
        sp["json_schema"] = json.dumps(r.tags["json_schema"])
        sp["ignore_eos"] = False   # 문법 제약 시 EOS 를 막으면 안 됨

    payload = {"text": prompt, "sampling_params": sp, "stream": True}
    res.submit_t = time.perf_counter()
    try:
        import aiohttp
        async with session.post(f"{url}/generate", json=payload,
                                timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                res.error = f"http {resp.status}: {(await resp.text())[:200]}"
                res.end_t = time.perf_counter()
                return
            prev = 0
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue
                now = time.perf_counter()
                meta = obj.get("meta_info") or {}
                n = meta.get("completion_tokens", prev + 1)
                if n <= prev:
                    continue
                if res.first_token_t is None:
                    res.first_token_t = now
                res.chunk_times.append(now)
                res.chunk_tokens.append(n - prev)
                prev = n
                res.output = obj.get("text", res.output)
                res.prompt_tokens = meta.get("prompt_tokens", res.prompt_tokens)
                res.cached_tokens = meta.get("cached_tokens", res.cached_tokens)
                res.completion_tokens = n
    except asyncio.TimeoutError:
        res.error = "timeout"
    except Exception as e:  # noqa: BLE001
        res.error = f"{type(e).__name__}: {e}"
    res.end_t = time.perf_counter()


async def run(trace: Trace, url: str, timeout: float, keep_output: bool,
              max_inflight: int, use_grammar: bool = False) -> list[dict]:
    import aiohttp

    by_rid = {r.rid: r for r in trace.requests}
    children: dict[str, list[str]] = {}
    for r in trace.requests:
        if r.depends_on:
            children.setdefault(r.depends_on, []).append(r.rid)

    results: dict[str, Result] = {}
    full_text: dict[str, str] = {}
    sem = asyncio.Semaphore(max_inflight)
    t0 = time.perf_counter()
    tasks: list[asyncio.Task] = []

    conn = aiohttp.TCPConnector(limit=0, force_close=False)
    async with aiohttp.ClientSession(connector=conn) as session:

        async def fire(rid: str, prompt: str) -> None:
            r = by_rid[rid]
            res = Result(r)
            results[rid] = res
            async with sem:
                await _one(session, url, r, prompt, res, timeout, use_grammar)
            full_text[rid] = prompt + (res.output or "")
            for c in children.get(rid, []):
                child = by_rid[c]
                await asyncio.sleep(child.think_time)   # 툴 실행 시간 모사
                tasks.append(asyncio.create_task(
                    fire(c, full_text[rid] + child.suffix)))

        async def at(t: float, rid: str) -> None:
            d = t - (time.perf_counter() - t0)
            if d > 0:
                await asyncio.sleep(d)
            await fire(rid, by_rid[rid].prompt)

        for r in trace.requests:
            if r.arrival_time is not None:
                tasks.append(asyncio.create_task(at(r.arrival_time, r.rid)))

        done = 0
        while tasks:
            batch, tasks = tasks, []
            await asyncio.gather(*batch)
            if len(results) > done:
                done = len(results)
                print(f"  ... {done}/{len(trace.requests)} 완료 "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

    return [results[r.rid].to_dict(t0, keep_output)
            for r in trace.requests if r.rid in results]


def main() -> None:
    ap = argparse.ArgumentParser(description="트레이스 재생 클라이언트")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--tag", required=True,
                    help="실험 이름. 예: default / ablated / mine")
    ap.add_argument("--out", default="results")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--max-inflight", type=int, default=100000,
                    help="기본은 무제한. 큐잉은 클라이언트가 아니라 서버가 해야 한다.")
    ap.add_argument("--keep-output", action="store_true",
                    help="verify.py 로 정확성 검증하려면 필수")
    ap.add_argument("--grammar", action="store_true",
                    help="W4: json_schema 제약 활성화")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="서버 없이 스케줄만 출력")
    args = ap.parse_args()

    trace = Trace.load(args.trace)
    print(trace.describe())
    if args.dry_run:
        for r in trace.requests[:15]:
            print(f"  {r.rid} 도착={r.arrival_time} 부모={r.depends_on} "
                  f"think={r.think_time} 출력={r.max_new_tokens}")
        return

    if args.warmup:
        # CUDA graph 캡처 / JIT / 가중치 로딩이 첫 요청에 섞여 들어가지 않도록
        print(f"워밍업: {args.warmup} 요청")
        warm = Trace("warmup", [
            Request(rid=f"w{i}", session_id="w", arrival_time=0.0,
                    prompt=trace.requests[i].prompt, max_new_tokens=8)
            for i in range(min(args.warmup, len(trace.requests)))])
        asyncio.run(run(warm, args.url, 300, False, 8))
        time.sleep(2)

    print(f"재생 시작 -> {args.url}")
    recs = asyncio.run(run(trace, args.url, args.timeout, args.keep_output,
                           args.max_inflight, args.grammar))

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{trace.name}__{args.tag}.json")
    with open(path, "w") as f:
        json.dump({"trace": trace.name, "tag": args.tag, "url": args.url,
                   "meta": trace.meta, "records": recs}, f, ensure_ascii=False)
    errs = sum(1 for r in recs if r["error"])
    print(f"저장 완료: {path}  (기록 {len(recs)}건, 에러 {errs}건)")
    if errs:
        print("  !! 에러가 있다. 고치기 전에는 어떤 수치도 보고하지 말 것.")


if __name__ == "__main__":
    main()
