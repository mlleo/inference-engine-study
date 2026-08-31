#!/usr/bin/env bash
# 1주차용: 베이스라인 1회 실행 + 병목 진단 리포트까지.
# 학생이 처음 돌려보는 스크립트다.
set -euo pipefail

WORKLOAD=${1:?사용법: run_baseline.sh <rag|agent|reasoning|structured|mixed>}
MODEL=${MODEL:-Qwen/Qwen3-4B}
PORT=${PORT:-30000}
URL="http://127.0.0.1:${PORT}"
EXTRA=("${@:2}")

mkdir -p traces results logs

[ -f "traces/${WORKLOAD}.jsonl" ] || \
  python -m workloads.generators "$WORKLOAD" --out traces/ --model "$MODEL"

python -m sglang.launch_server \
  --model-path "$MODEL" --port "$PORT" \
  --context-length 32768 --mem-fraction-static 0.85 \
  --random-seed 42 --log-level info --enable-metrics \
  "${EXTRA[@]}" > "logs/server_default.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT

echo "서버 기동 대기..."
for _ in $(seq 1 180); do
  curl -sf "${URL}/health_generate" >/dev/null 2>&1 && break
  sleep 2
done

python -m bench.replay --trace "traces/${WORKLOAD}.jsonl" --url "$URL" \
  --tag default --out results/ --keep-output

# 스케줄러 관점의 근거. 클라이언트 지표만으로 원인을 단정하지 말 것.
grep -E "cache hit rate|#retracted|token usage|#running-req" \
  "logs/server_default.log" | tail -30 > "logs/sched_default.txt" || true

echo; echo "########## 지표 ##########"
python -m bench.metrics "results/${WORKLOAD}__default.json"
echo; echo "########## 병목 진단 ##########"
python -m bench.analyze "results/${WORKLOAD}__default.json"
echo; echo "########## 서버 로그 발췌 (logs/sched_default.txt) ##########"
tail -10 logs/sched_default.txt || true
