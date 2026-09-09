# W3 Reasoning — 새 Step 1·2 분석

2026-09-09에 기본 trace부터 새로 생성·재생했다.
환경은 [공통 실험 설명](../README.md), 이전 오류는 [감사 기록](audit.md)을 참고한다.

## Step 1: 전체 작업량과 순간 KV를 구분한다

원시 trace: `artifacts/reasoning/reasoning_baseline.jsonl`.
실측: `artifacts/reasoning/baseline_default_r1/reasoning_baseline__default_r1.json`.

| 항목 | 새 측정/정의 |
|---|---|
| 요청 | 독립120개, seed=3 |
| 입력 p50/p90/p99 | 387 / 387 / 387토큰. 이전 보고서의 703/710과 다름 |
| 출력 p50/p90/p99/max | 2338 / 4744.7 / 8927.2 / 10541토큰 |
| 입력:출력 중앙값 비율 | 약 1:6.04. 모델 연산량 비율은 아님 |
| 출력 분포 | lognormal 평균 파라미터3000, sigma=.6, [256,16000] clip |
| 도착 | Poisson .35 requests/s, 도착 구간315.643초 |
| 공유 | 문제 본문은 다르지만 시스템 안내 등 공통 prefix가 있음 |
| SLO | 평균 TPOT≤60ms만 지정. TTFT·maxITL 제한은 없음 |
| 계획 출력 합 | 337433토큰. 누적 생성 작업량이며 동시 KV가 아님 |
| 동시성 | client in-flight 최대14, 서버 running 최대14 |
| KV 크기 감각 | 입력387+출력2338=2725토큰≈.374GiB/요청. 중앙값 길이로 peak를 단정하지 않음 |

이 trace는 ignore_eos=True이므로 정상 완료 시 출력 길이는 max_new_tokens와 같다.
이를 사용하는 admission은 실서비스의 길이 예측과 다르다. 토큰 ID는 유지한 채
retraction된 KV를 재prefill하는 비용과, 아직 시작 못한 요청의 큐 대기를 구분한다.

## Step 2: baseline에 없던 메모리 병목을 통제해 만든다

기본 trace 1회에서 TTFT p50/p99=17.55/24.12ms, TPOT p50=5.29ms,
goodput=.3541 req/s, 출력 처리량995.65 tok/s, SLO 만족100%였다.
cache hit=9.20%이므로 “공유 prefix가 전혀 없다”는 설명도 맞지 않는다.

서버 로그는 active usage 최대 .07, running 최대14, queue=0, retract=0이다.
max stream gap p99=185.94ms지만 retract는 없다. 큰 gap 하나만으로 retraction을 추론하지 않는다.
기본 부하에서는 capacity 병목을 관측하지 못했다는 것이 새 분석의 출발점이다.

pressure trace는 요청48개, out_mean6000, out_sigma=.2, QPS4로 고정하고
KV 풀만65536으로 제한한다. QPS2/4/8을 같은 trace 구조에서 비교한다.
원래 큰 풀에서 같은 pressure trace를 재생하는 용량 대조도 분리한다.
단지 출력 총합이 풀보다 크다는 계산 대신 실제 retraction 로그·active usage·queue를 확인한다.

프로토콜 지정 ablation은 chunked prefill OFF다. 이것이 효과 없으면 prefill 간섭이
현재 주된 원인이 아니라는 제한적인 증거다. **capacity 가설의 직접 대조는 풀 크기**다.
재prefill에는 이전에 생성한 긴 이력도 포함될 수 있으므로 처음 prompt가 짧다고
청킹의 영향이 항상 없다고 단정하지 않는다.

최종 반복/대조는 [controlled 측정 표](../../artifacts_deterministic/measurements.md)에 보관한다.

## Step 2 최종 대조 결과

모든 bar에 deterministic FA3를 적용했다. ordinary baseline과 직접 비교하지 않는다.
pressure(QPS4)는 각3회 중앙값, load endpoint와 큰 풀 control은 각1회다.

| QPS | default goodput | chunk OFF goodput | default gap p99 ms | chunk OFF gap p99 ms |
|---|---|---|---|---|
| 2 | .250 | .249 | 77902.86 | 81915.39 |
| 4 | .261 | .248 | 105381.42 | 104673.21 |
| 8 | .269 | .249 | 101036.37 | 76525.83 |

QPS4 default는 active usage1.00, running 최대 중앙값46, queue 최대31,
retracted request count34–35(중앙값35)였다. 이 count는 로그의 합이며 같은 요청이
여러 번 retract될 수 있으므로 고유 요청 수라고 부르지 않는다.
chunk OFF도35회와 약104.7초 gap이 남는다. 청킹 해제가 capacity 문제를 없애지 않는다.

**같은 pressure trace에서 KV 풀만430067로 늘리면** retraction0, queue0,
gap p99=38.75ms, TTFT p99=43.96ms, goodput=.4015였다.
따라서 이 통제한 작은 풀에서의 **KV capacity/retraction 압박**은 실측 대조로 지지된다.
이는 작은 GPU의 연산 속도나 기본 trace에서의 메모리 절벽을 재현했다는 뜻은 아니다.

QPS8 default의 TTFT p99는38.96ms로 작지만 gap p99는약101초이고 retraction38회다.
처음 토큰이 빨랐다는 사실은 이후 decode 정체가 없다는 뜻이 아니다.
3개 default load 모두 TPOT-only SLO 만족100%라서 goodput만 보고 UX가 좋다고
판정해서도 안 된다. [부하 곡선](../../artifacts_deterministic/sweep.png)과
[출력 검사](../../artifacts_deterministic/verification.json)를 함께 제공한다.
Step 3에서는 알려진 출력 예산으로 peak KV를 보수적으로 예약해 본다.
retraction과 stream gap이 줄더라도 TTFT나 총 완료시간이 나빠질 수 있다.
원래 TPOT-only goodput과 TTFT/maxITL/E2E를 함께 보고하여 큐로 이동한 지연을 드러낸다.
