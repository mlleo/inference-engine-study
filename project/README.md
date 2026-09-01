# SGLang 워크로드 최적화 스터디 프로젝트

이 저장소는 학생들이 SGLang 추론 엔진의 워크로드별 병목을 분석하고
최적화하는 5주차 프로젝트를 위한 것이다.

**전체 안내는 [`docs/README.md`](docs/README.md) 를 볼 것.**

## 빠른 시작

```bash
pip install -r requirements.txt

# (1) GPU 파드에서 환경 세팅
SGLANG_VERSION=<확정된 버전> MODEL=Qwen/Qwen3-4B bash scripts/setup_pod.sh

# (2) 배정받은 워크로드의 베이스라인 실행 (예: agent)
bash scripts/run_baseline.sh agent
```

자세한 내용:
- [프로젝트 안내 (README)](docs/README.md)
- [환경 설정 가이드](docs/01_환경설정.md)
- [수행 5단계](docs/02_수행단계.md)
- [실험 프로토콜](docs/03_실험프로토콜.md)
- [보고서 템플릿](docs/04_보고서_템플릿.md)
- [평가 루브릭](docs/05_평가루브릭.md)
- [워크로드 배정 카드](docs/workloads/)
