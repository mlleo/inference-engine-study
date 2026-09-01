#!/usr/bin/env bash
# 3-bar 프로토콜. 보고하는 모든 결과는 반드시 이 세 줄이 함께 있어야 한다.
#
#   A. default  순정 SGLang, 기능 전부 켬        -> 네가 이겨야 하는 대상
#   B. ablated  관련 기능만 끔                    -> 그 기능이 이미 주고 있던 이득
#   C. mine     네 수정본                        -> 네 기여분
#
# B 를 빼먹으면, radix cache 가 원래 내주던 3배 향상이 네 기여로 둔갑한다.
# 이 스터디에서 가장 흔한 실수이고, 발표 때 가장 먼저 지적당할 지점이다.
set -euo pipefail

TRACE=${1:?사용법: run_matrix.sh <trace.jsonl> [ablation 플래그...]}
shift
ABLATE=("$@")
MODEL=${MODEL:-Qwen/Qwen3-4B}
PORT=${PORT:-30000}
URL="http://127.0.0.1:${PORT}"
MINE_FLAGS=${MINE_FLAGS:-}     # 네 최적화를 켜는 플래그를 여기에

mkdir -p results logs
COMMON=(--model-path "$MODEL" --port "$PORT" --context-length 32768
        --mem-fraction-static 0.85 --random-seed 42 --log-level info --enable-metrics)

run_bar() {
  local TAG=$1; shift
  echo "=== bar: $TAG ==="
  python -m sglang.launch_server "${COMMON[@]}" "$@" > "logs/server_${TAG}.log" 2>&1 &
  local PID=$!
  for _ in $(seq 1 180); do
    curl -sf "${URL}/health_generate" >/dev/null 2>&1 && break; sleep 2
  done
  python -m bench.replay --trace "$TRACE" --url "$URL" --tag "$TAG" \
    --out results/ --keep-output
  grep -E "cache hit rate|#retracted|token usage|#running-req" \
    "logs/server_${TAG}.log" | tail -30 > "logs/sched_${TAG}.txt" || true
  kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; sleep 5
}

run_bar default
run_bar ablated "${ABLATE[@]}"
run_bar mine    ${MINE_FLAGS}

NAME=$(basename "$TRACE" .jsonl)
echo; echo "########## 3-bar 비교 ##########"
python -m bench.metrics "results/${NAME}__default.json" \
                        "results/${NAME}__ablated.json" \
                        "results/${NAME}__mine.json" --csv "results/${NAME}_3bar.csv"
echo; echo "########## 정확성 게이트 ##########"
python -m bench.verify "results/${NAME}__default.json" "results/${NAME}__mine.json" || true

# 워크로드별 ablation 플래그 예시:
#   rag / agent : --disable-radix-cache
#   reasoning   : --chunked-prefill-size -1
#   structured  : --grammar-backend outlines
#   mixed       : --schedule-policy fcfs
#   공통        : --disable-cuda-graph  --disable-overlap-schedule
