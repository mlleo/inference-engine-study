# SGLang 워크로드 최적화 스터디 프로젝트

## 1. 프로젝트 목표

각자에게 **서로 다른 워크로드 하나**가 무작위로 배정된다. 목표는 그 워크로드에서
LLM 추론 엔진이 왜 느린지를 **증거로 규명**하고, SGLang 을 실제로 수정해서
개선한 뒤, **무엇이 나빠졌는지까지** 정직하게 보고하는 것이다.

이 프로젝트의 핵심 명제는 하나다.

> **모든 추론 최적화는 특정 트래픽 패턴에 대한 베팅이다.**
> 어떤 워크로드에서 2배 빨라지는 기법은 다른 워크로드에서 반드시 손해를 본다.

그래서 마지막 주에 5명의 엔진을 5개 워크로드 전부에 교차 실행한다.
그 **5×5 행렬이 이 스터디의 진짜 결과물**이다.

### 자주 나오는 오해 하나

"내가 하려는 최적화가 이미 SGLang 에 구현되어 있는데요?"

맞다. RadixAttention, chunked prefill, CUDA graph, overlap scheduler,
xgrammar jump-forward, EAGLE, hierarchical KV offload — 대부분 이미 들어 있다.
**그래서 이 프로젝트의 무게중심은 '새 기법 발명'이 아니라 '왜 그 기법이
존재하는지를 측정으로 재발견하는 것'에 있다.** 기능을 끄고 켜서 그 기능이
막고 있던 재앙을 직접 목격한 사람만이, 그 기능이 여전히 못 막는 지점을 찾아낼 수
있다. 배정 카드(`docs/workloads/`)에 각 워크로드에서 **SGLang 이 아직 못 하는
것**을 명시해 두었다.

---

## 2. 산출물 (제출물)

| # | 제출물 | 형식 | 마감 |
|---|--------|------|------|
| 1 | 베이스라인 실행 결과 + 병목 가설 | 1페이지 요약 | 2주차 |
| 2 | 최종 보고서 | `docs/보고서_템플릿.md` 형식 | 5주차 |
| 3 | 코드 | SGLang fork 의 diff + 재현 스크립트 | 5주차 |
| 4 | PoC 발표 | 15분 발표 + 10분 Q&A | 5주차 |
| 5 | 원시 결과 파일 | `results/*.json` 전체 | 5주차 |

**원시 결과 파일을 반드시 함께 제출한다.** 표에 적힌 숫자가 실제 측정에서 나온
것인지 확인할 수 있어야 한다.

---

## 3. 수행 5단계

각 단계의 상세는 [`docs/02_수행단계.md`](docs/02_수행단계.md) 참고.

| 단계 | 이름 | 핵심 질문 | 산출물 |
|------|------|-----------|--------|
| 1 | 워크로드 분석 | 이 워크로드는 **무엇이** 어려운가? | 요청 길이 분포, 재사용 구조, SLO 정의 |
| 2 | 병목 규명 | prefill / decode / KV / 스케줄러 중 **어디**인가? | `analyze.py` 리포트 + 서버 로그 근거 |
| 3 | 엔진 수정 | SGLang 에 최적화 **하나**를 구현 | git diff, 켜고 끌 수 있는 플래그 |
| 4 | 평가 | 공통 베이스라인 대비 **얼마나**? | 3-bar 비교표 + 정확성 게이트 통과 |
| 5 | 트레이드오프 | **무엇이 나빠졌는가?** | 자기 최적화가 지는 워크로드 |

> **5단계가 이 프로젝트에서 가장 배점이 높다.** 자기 최적화가 손해를 보는
> 워크로드를 스스로 찾아내지 못한 보고서는 완성된 것이 아니다.

---

## 4. 5개 워크로드

| # | 워크로드 | 주 병목 | 배정 카드 |
|---|----------|---------|-----------|
| W1 | 롱컨텍스트 RAG | Prefill, KV 용량 | [W1_RAG.md](docs/workloads/W1_RAG.md) |
| W2 | AI 에이전트 (멀티턴 + 툴) | 캐시 유지(retention) | [W2_Agent.md](docs/workloads/W2_Agent.md) |
| W3 | 장문 추론 (Long CoT) | Decode, KV 증가 | [W3_Reasoning.md](docs/workloads/W3_Reasoning.md) |
| W4 | 구조화 출력 (JSON 추출) | 요청당 CPU 오버헤드 | [W4_Structured.md](docs/workloads/W4_Structured.md) |
| W5 | 혼합 SLO 멀티테넌트 | 스케줄링 정책 | [W5_Mixed.md](docs/workloads/W5_Mixed.md) |

---

## 5. 저장소 구조

```
common/trace.py            공용 트레이스 스키마  ← 1주차에 확정, 이후 수정 금지
common/textgen.py          결정적 합성 텍스트 생성
workloads/generators.py    5개 워크로드 생성기
bench/replay.py            open-loop + 의존성 인지 재생 클라이언트
bench/metrics.py           오프라인 지표 계산 (GPU 불필요)
bench/analyze.py           병목 진단 리포트  ← 2단계 주력 도구
bench/verify.py            greedy 출력 정확성 게이트
scripts/setup_pod.sh       RunPod 세팅 + 모델 캐싱
scripts/run_baseline.sh    베이스라인 1회 실행 + 진단   ← 제일 먼저 돌릴 것
scripts/run_matrix.sh      3-bar 실험 (default/ablated/mine)
scripts/cross_replay.sh    5주차 교차 검증
docs/                      문서 일체
```

---

## 6. 첫날에 할 일 (30분)

```bash
git clone <이 저장소> && cd sglang-study
pip install -r requirements.txt

# (1) GPU 파드에서 환경 세팅
SGLANG_VERSION=<확정된 버전> MODEL=Qwen/Qwen3-4B bash scripts/setup_pod.sh

# (2) 배정받은 워크로드의 베이스라인 실행 (예: agent)
bash scripts/run_baseline.sh agent

# (3) 출력된 [병목 진단] 섹션을 읽고, 아래 질문에 답을 적어보기
#     - E2E 시간의 몇 %가 TTFT 인가? prefill 지배인가 decode 지배인가?
#     - 동시 요청 수가 시간에 따라 계속 커지는가? (= 서버 포화)
#     - 캐시 적중률이 몇 %인가? 예상보다 낮다면 왜인가?
```

먼저 저비용으로 파이프라인만 확인하고 싶으면 `--scale 0.05` 로 축소 트레이스를
만들어 돌린다.

---

## 7. 반드시 지킬 규칙 5가지

1. **모든 것을 고정한다.** SGLang 버전 태그, 모델 리비전, GPU 종류,
   `--random-seed 42`, `--mem-fraction-static`. SGLang `main` 은 3주면 다른
   엔진이 된다.
2. **Python 만 수정한다.** `sgl-kernel` 을 다시 빌드해야 하는 순간 며칠이
   날아간다. 이 제약은 자연스럽게 스케줄러/캐시/정책 쪽으로 주제를 몰아주는데,
   그게 이 스터디의 적정 범위다.
3. **3-bar 없이는 어떤 숫자도 보고하지 않는다.** ([`docs/03_실험프로토콜.md`](docs/03_실험프로토콜.md))
4. **정확성 게이트를 통과하지 못한 속도 향상은 없는 것으로 친다.**
5. **throughput 이 아니라 goodput 을 보고한다.**

---

## 8. 일정

| 주차 | 내용 |
|------|------|
| 1주차 | SGLang 코드 함께 읽기 → `srt/managers/scheduler.py`, `srt/mem_cache/radix_cache.py`, `srt/managers/schedule_policy.py`, attention backend. 트레이스 스키마 확정. 전원 베이스라인 실행 성공. |
| 2주차 | **코드 작성 금지.** 각자 병목 가설을 증거와 함께 발표. |
| 3–4주차 | 구현. |
| 5주차 | 교차 검증(5×5), 트레이드오프 분석, PoC 발표. |

**2주차를 반드시 지킨다.** 어떤 지표를 가리키며 "이것이 병목이다"라고 말하지
못하는 상태에서 코드를 시작하면, 십중팔구 엉뚱한 곳을 최적화하게 된다.

---

## 9. 비용 관리

- 모델은 **전원 동일**하게 쓴다 (`Qwen/Qwen3-4B`, bf16, 24GB 카드 기준 KV 여유
  약 15GB → W1/W5 의 32k 컨텍스트 수용 가능). 각자 다른 모델을 쓰면 교차 검증이
  그 자리에서 무의미해진다.
- **모델 가중치는 네트워크 볼륨에 캐싱한다.** 파드 재시작마다 8GB 재다운로드가
  이 스터디에서 가장 큰 낭비다.
- **개발은 로컬 CPU, 측정만 GPU.** 트레이스 생성·지표 계산·진단·정확성 검증은
  전부 GPU 없이 돌아간다. 파드는 `run_matrix.sh` 돌릴 때만 켠다.
- 3-bar 1회가 10~15분에 끝나도록 트레이스 크기를 조정한다.
- 커뮤니티 클라우드 스팟 인스턴스가 시큐어 클라우드보다 싸고, 우리 실행은 짧고
  재시작이 쉬우므로 중단되어도 손해가 작다. 가격은 자주 바뀌니 직접 확인할 것.
