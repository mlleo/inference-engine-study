# W2 Agent — 새 Step 1·2 분석

2026-09-09에 trace를 새로 생성하고 재측정했다. 기존 수치와 병목 결론을 이어 쓰지 않았다.
환경은 [공통 실험 설명](../README.md), 기존 오류는 [감사 기록](audit.md)을 참고한다.

## Step 1: 세 종류의 동시성을 분리한다

원시 trace: `artifacts/agent/agent_baseline.jsonl`.
실측: `artifacts/agent/baseline_default_r1/agent_baseline__default_r1.json`.

| 항목 | 새 측정/정의 |
|---|---|
| 요청/세션 | 40세션×6턴=240개, root 40개와 종속 200개, seed=2 |
| 전체 입력 p50/p90/p99 | 3411 / 5904.8 / 7991.9토큰 |
| 출력 p50/p90/p99/max | 150.5 / 251.1 / 382.18 / 429토큰 |
| 입력:출력 중앙값 비율 | 약 22.7:1. 캐시와 시간 비중은 별도로 계산 |
| 턴별 입력 p50 | 1596 → 2162 → 2925 → 3754.5 → 4454 → 5476.5 |
| 세션 도착 | Poisson .5 sessions/s, root 도착 구간 77.3632초 |
| think time | 평균 파라미터 8초, sigma=.9; 실측 p50=5.476, p90=15.5084초 |
| 세션 연결 | 부모 prompt+실제 output+suffix. 후속 턴은 부모 완료 후 think time만큼 지연 |
| 공유 | 5종 tool definition 및 세션 내 이력. 토큰 경계 매칭은 실측으로 확인 |
| SLO | TTFT≤1500ms, 교육용 턴 응답성 목표 |
| 동시성 | 살아 있는 세션 최대 23, client in-flight 최대 8, 서버 running 최대 7 |
| KV 크기 감각 | 최종 턴 입력 중앙값 5476.5토큰≈.75GiB/세션(공유 차감 전). 총 40세션을 동시에 더하지 않음 |

살아 있는 세션은 첫 요청 제출~최종 턴 완료 구간이며 think 중인 시간도 포함한다.
client in-flight에는 서버 큐가 포함된다. server running은 실제 배치 관측치다.
서로 다른 세 값을 곱셈에 섞으면 캐시 압박 추정이 잘못된다.

## Step 2: baseline에서는 여유, 통제한 작은 풀에서는 압박

아래 값은 각각 첫 실행의 관찰값이다. 주요 pressure 비교는 별도로 3회 반복한다.

| 조건 | TTFT p50/p99 ms | TPOT p50 ms | hit | SLO 만족 | goodput req/s |
|---|---|---|---|---|---|
| 기본 40세션, 풀430067 | 24.68 / 96.14 | 4.87 | 83.69% | 100% | 1.8356 |
| 80세션, q4, think_mean4, 풀65536 | 76.39 / 2505.60 | 12.17 | 48.67% | 81.46% | 4.8972 |

기본 로그: active usage 최대 .06, running 7, queue 0, retract 0.
종료 후 metrics에는 active KV=0과 evictable KV=179068이 함께 나타난다.
“실행 중 usage가 낮다”와 “radix에 저장된 토큰이 적다”는 다른 말이다.

pressure 첫 실행: active usage 1.00, running 최대32, queue 최대24,
retracted request 1. 살아 있는 세션 최대80, client in-flight 최대43이다.
턴별 hit은 90.58→70.10→57.04→46.14→38.56→32.55%다.
실행 KV와 캐시 경쟁, 큐가 동시에 나타났으므로 **순수 idle eviction만의 병목**이라고
부르지 않는다. 기본/pressure 행은 workload와 capacity가 함께 달라 인과 비교 그 자체는 아니다.

인과 실험은 pressure trace를 고정하고 radix OFF, 원래 풀 크기, cache eviction 정책을
각각 분리해 비교한다. 부하 sweep은 같은 80세션/think_mean4/풀65536에서
session_qps 2/4/8이다. QPS는 턴 요청률이 아니라 root 세션 도착률이다.
세 bar가 후속 턴을 같은 절대 시간에 제출하지 않는 것은 부모 완료 의존성의 결과다.

전체 반복과 대조 결과는 [controlled 측정 표](../../artifacts_deterministic/measurements.md)에 보관한다.

## Step 2 최종 대조 결과

모든 bar를 deterministic FA3로 통제한 별도 series다. 위 ordinary 표와 혼합하지 않는다.
같은80세션·think_mean4를 유지한다. 중앙 pressure는 각3회 중앙값, 나머지는1회다.

| sessions/s | default goodput | radix OFF goodput | default TTFT p99 ms | radix OFF TTFT p99 ms |
|---|---|---|---|---|
| 2 | 4.637 | 1.551 | 2048.74 | 5989.81 |
| 4 | 3.631 | 1.447 | 4204.35 | 9298.38 |
| 8 | 3.827 | 1.805 | 4992.91 | 10863.72 |

pressure의 기본65536토큰 풀에서는 hit47.27%, TTFT p50/p99=279.80/4204.35ms,
queue 최대 중앙값29, active usage1.00이다. **동일 pressure trace**의 풀만430067로
늘린 control에서는 hit84.77%, TTFT37.73/124.37ms, queue0, goodput6.374였다.
캐시 OFF 및 용량 대조는 캐시 재사용과 제한된 KV 용량이 중요한 원인임을 지지한다.
그러나 active KV와 eviction이 공간을 공유하므로 순수 idle retention만 분리한 증명은 아니다.

도착률8에서4보다 goodput이 높다는 이유로 포화가 사라졌다고 해석하지 않는다.
이 값은 유한 chained trace의 SLO 만족 요청 수/makespan이며 think time과
완료 의존 도착이 함께 바뀐다. TTFT tail은 오히려 악화됐다.
새 eviction policy는 hit47.22%, goodput3.587로 기본보다 좋아지지 않았다.
‘캐시가 중요하다’와 ‘깊은 prefix를 보존하면 개선된다’는 별개의 가설이다.

반복 범위, [부하 곡선](../../artifacts_deterministic/sweep.png),
[전체 출력·후속 prompt 검사](../../artifacts_deterministic/verification.json)를 함께 제공한다.
Step 3 가설은 제한된 cache 공간에서 재계산이 비싼 prefix의 보존 순서를 바꾸면
hit/TTFT를 개선할 수 있다는 것이다. 새 정책은 잠금이나 KV 계산을 바꾸지 않고,
작은 세션과 CPU eviction 비용이 악화될 수 있음을 평가한다.
