#!/usr/bin/env bash
# 5주차 교차 검증. 내 엔진을 **다른 사람의 5개 워크로드 전부**에 돌린다.
# 이 5x5 행렬이 스터디의 진짜 결과물이다. 대각선이 아닌 칸들 —
# 즉 내 최적화가 지는 워크로드 — 이 각자 보고서의 트레이드오프 절이 된다.
set -euo pipefail

TAG=${1:?사용법: cross_replay.sh <내-엔진-태그>}
MODEL=${MODEL:-Qwen/Qwen3-4B}
PORT=${PORT:-30000}
URL="http://127.0.0.1:${PORT}"
MINE_FLAGS=${MINE_FLAGS:-}

mkdir -p results logs
python -m sglang.launch_server --model-path "$MODEL" --port "$PORT" \
  --context-length 32768 --mem-fraction-static 0.85 --random-seed 42 \
  --enable-metrics ${MINE_FLAGS} > "logs/server_cross_${TAG}.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do curl -sf "${URL}/health_generate" >/dev/null 2>&1 && break; sleep 2; done

# 비용 절감: 교차 검증은 축소 트레이스(--scale 0.3)로 충분하다
for W in rag agent reasoning structured mixed; do
  [ -f "traces/${W}_x.jsonl" ] || \
    python -m workloads.generators "$W" --out traces/ --model "$MODEL" \
      --scale 0.3 --suffix _x
  python -m bench.replay --trace "traces/${W}_x.jsonl" --url "$URL" \
    --tag "$TAG" --out results/cross/
done

echo; echo "########## 교차 결과 ##########"
for W in rag agent reasoning structured mixed; do
  echo "--- 워크로드: $W ---"
  python -m bench.metrics "results/cross/${W}_x__*.json"
done
