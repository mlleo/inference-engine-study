#!/usr/bin/env bash
# RunPod 파드 초기 세팅. 파드를 새로 띄울 때마다 한 번씩 실행.
#
# 중요: 모델 가중치는 반드시 네트워크 볼륨에 캐싱한다. 파드를 띄울 때마다
# 8GB 를 다시 받는 것이 이런 스터디에서 가장 크고 가장 피하기 쉬운 낭비다.
set -euo pipefail

export HF_HOME=${HF_HOME:-/workspace/hf}      # <- 네트워크 볼륨 경로
mkdir -p "$HF_HOME"

# 버전 고정은 협상 대상이 아니다. 5명이 서로 다른 커밋을 쓰면 비교가 무의미해진다.
# 기본값 0.5.18 로 고정. 다른 버전을 쓰려면 SGLANG_VERSION 을 명시적으로 덮어쓸 것.
SGLANG_VERSION=${SGLANG_VERSION:-0.5.18}
MODEL=${MODEL:-Qwen/Qwen3-4B}

pip install --upgrade pip
pip install "sglang[all]==${SGLANG_VERSION}"
pip install aiohttp transformers

# 가중치 선다운로드 (측정 시간에 다운로드가 섞이지 않도록)
python - <<PY
import os
from huggingface_hub import snapshot_download
snapshot_download("${MODEL}", cache_dir=os.environ["HF_HOME"])
print("모델 캐시 완료:", os.environ["HF_HOME"])
PY

echo
echo "==== 환경 고정 정보 (보고서 부록에 그대로 붙여넣을 것) ===="
python -c "import torch,sglang;print('sglang',sglang.__version__);print('torch',torch.__version__);print('gpu',torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "model=${MODEL}"
