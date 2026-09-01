# W5 — 혼합 SLO / 멀티테넌트

> **배정자:** ____________

## 시나리오

하나의 엔진에 두 종류 트래픽이 섞여 들어온다.

| 클래스 | 특성 | SLO |
|--------|------|-----|
| **interactive** | 입력 320 토큰, 출력 120 토큰, QPS 8 (포아송) | TTFT ≤ 500ms, TPOT ≤ 40ms |
| **batch** | 입력 **12,000 토큰**, 출력 700 토큰, 12개씩 버스트 | 없음 |

```bash
python -m workloads.generators mixed --out traces/ --model Qwen/Qwen3-4B
bash scripts/run_baseline.sh mixed
```

## 주 병목

**스케줄링 정책.** 하드웨어가 아니라 정책 문제다.

## 이 워크로드의 진짜 함정

**전체 평균을 보면 아무 문제도 없어 보인다.**

12k 토큰짜리 배치 요청이 버스트로 몰려오면 대화형 요청의 P99 TTFT 가 폭발한다
(head-of-line blocking). 그런데 배치 요청 수가 적어서 전체 평균은 멀쩡하다.

**반드시 클래스별로 나눠서 볼 것. 이 한 줄이 이 과제의 전부다.**

```bash
python -m bench.metrics 'results/mixed__*.json' --by slo_class
```

SGLang 의 기본 정책 `lpm`(longest-prefix-match)은 처리량에는 최적이지만
이 상황을 **악화시킨다.** `--schedule-policy fcfs` 와 비교해보면 정책이 만드는
차이를 볼 수 있다.

**1차 목표:** interactive 의 P99 TTFT 가 SLO(500ms)를 깨는 시점이 배치 버스트
도착 시점과 정확히 일치함을 시계열로 보여라. `analyze.py` 의 [2], [3] 섹션이
이 그림을 만들어준다.

## SGLang 이 이미 하는 것

- LPM / FCFS 스케줄 정책
- Chunked prefill (긴 prefill 이 decode 를 완전히 굶기는 것은 이미 막고 있다)
- 연속 배칭

## SGLang 이 아직 못 하는 것 = 과제 후보

1. **SLO 인지 우선순위 스케줄링** — 요청에 우선순위/데드라인을 부여하고,
   데드라인이 임박한 순으로 배치를 구성. 기아(starvation) 방지를 위한 aging 필수.
   난이도 중간. **추천.**
2. **Admission control** — interactive SLO 가 위태로우면 배치 요청 수락을
   일시 지연. 가장 단순하면서 효과가 확실한 방향.
3. **클래스별 KV 쿼터** — batch 가 쓸 수 있는 KV 상한을 두어 interactive 용
   여유를 보장. 구현이 명확하고 트레이드오프도 명확하다.
4. **적응적 chunked prefill 크기** — interactive 대기열 길이에 따라 청크 크기를
   동적으로 조절.

## 스윕할 축

```bash
--set inter_qps=4 / 8 / 16 / 32          # ★ 대화형 부하
--set burst_size=4 / 12 / 32             # ★ 버스트 강도
--set batch_prompt_tokens=4000 / 12000 / 24000
```

가장 중요한 그림: **x축 = 배치 부하, y축 = interactive goodput.** 여기서
default / fcfs / mine 세 곡선이 갈라지는 것을 보여라.

## 확인할 서버 로그

```
#queue-req        버스트 도착 시 대기열 급증
#running-req      배치 요청이 슬롯을 얼마나 점유하는가
token usage       12k 프롬프트가 KV 를 얼마나 먹는가
```

## 예상되는 트레이드오프 (5단계에서 확인할 것)

- **공정성 vs 처리량은 진짜 교환관계다.** interactive 를 우선하면 전체 토큰
  처리량은 거의 확실히 떨어진다. 두 지표를 나란히 보고하라.
- 우선순위 스케줄링은 **batch 기아**를 만들 수 있다. batch 의 최대 대기 시간을
  반드시 측정해서 보고하라. aging 을 넣었다면 그 효과도.
- LPM 을 버리면 **프리픽스 공유 이득을 잃는다.** W1/W2 처럼 프리픽스 공유가
  중요한 워크로드에서 이 손해가 크게 나타날 것이다. 교차 검증의 핵심 관전 포인트.
- 단일 클래스 워크로드에서는 우선순위 계산이 순수 오버헤드다.
