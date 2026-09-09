# W1 RAG — 새 Step 1·2 분석

2026-09-09에 trace를 새로 생성하고 재측정했다. 기존 보고서의 표를 재사용하지 않았다.
환경과 재현 방법은 [공통 실험 설명](../README.md), 기존 오류는 [감사 기록](audit.md)을 참고한다.

## Step 1: 데이터가 말하는 구조

원시 trace: `artifacts/rag/rag_baseline.jsonl`.
실측: `artifacts/rag/baseline_default_r1/rag_baseline__default_r1.json`.

| 항목 | 새 측정/정의 |
|---|---|
| 요청 | 독립 200개, seed=1 |
| 입력 토큰 p50/p90/p99 | 7297 / 7297 / 7297 |
| 출력 토큰 p50/p90/p99/max | 127.5 / 211.3 / 290.46 / 400 |
| 입력:출력 중앙값 비율 | 57.2:1. 시간이나 FLOP 비율은 아님 |
| 문서 구성 | 400개 pool에서 6개씩, 청크당 1200토큰, reuse_alpha=1.1 |
| 중복 | 1200회 등장 중 고유 240개: 등장 중복률 80%. 토큰 cache hit과 다른 지표 |
| 도착 | Poisson 목표 .6 requests/s, 실제 첫~마지막 도착 356.2948초 |
| SLO | TTFT≤4000ms. 생성기가 고정한 교육용 응답성 목표 |
| 출력 예산 | 합계 27660토큰, ignore_eos=True |
| 관측 동시성 | client in-flight 최대 4, 서버 로그 running 최대 4 |
| KV 추정 | 입력만 4×7297≈29.2k토큰=4.01GiB. 출력·캐시 잔존·공유는 별도 |

정확히 같은 첫 청크와 그 이전 토큰들이 같으면 긴 prefix가 재사용될 수 있다.
문서 ID가 동일해도 선행 문맥이 달라지면 같은 KV라고 할 수 없다.
현재 병목 가설은 “긴 uncached prefill이 TTFT와 함께 실행 중인 decode의 정체를 만든다”이다.
낮은 cache hit을 곧바로 해결 가능한 낭비 또는 손실 없는 재사용의 상한으로 해석하지 않는다.

## Step 2: 새 baseline과 검증할 인과

아래 baseline은 1회 관찰 결과이며 반복 신뢰구간을 뜻하지 않는다.

| TTFT p50/p99 ms | TPOT p50 ms | max stream gap p99 ms | cache hit | SLO 만족 | goodput req/s |
|---|---|---|---|---|---|
| 133.86 / 216.14 | 4.73 | 123.83 | 9.76% | 100% | .5599 |

서버 로그는 active usage 최대 .07, queue 최대 0, retraction 0이다.
이 부하에서 서비스 포화나 실행 중 KV 부족은 관측되지 않았다.
active usage는 evictable cache 전체가 아니므로 “캐시 용량은 무관”이라고 단정하지 않는다.
첫 토큰 이후 생성이 수백 ms 이어지므로 긴 입력만 보고 E2E도 prefill 지배라고 부르지 않는다.

재현 실험은 동일한 120요청 trace 구조에서 QPS 4/8/16을 사용한다.
각 점에서 default와 radix OFF를 비교하고, QPS 8의 주요 비교는 3회 반복한다.
도착 시각만 다른 sweep이며 각 점 내 세 bar의 trace hash는 같다.
서버 running/queue, uncached token 수, TTFT와 stream gap을 함께 본다.
캐시 OFF는 기존 radix 효과를 분리하며, prefill budget 변경은 decode 간섭 가설을 검증한다.
부하별 실제 결과는 [controlled 측정 표](../../artifacts_deterministic/measurements.md)에 보관한다.

## Step 2 최종 대조 결과

아래는 모든 bar에 deterministic FA3를 적용한 별도 series다. 위 ordinary baseline과
성능을 직접 합치지 않는다. QPS8은 각 3회 중앙값이며 QPS4/16은 각 1회 탐색이다.

| QPS | default goodput | radix OFF goodput | default TTFT p99 ms | radix OFF TTFT p99 ms |
|---|---|---|---|---|
| 4 | 3.650 | 3.610 | 643.09 | 802.37 |
| 8 | 3.098 | 2.402 | 7632.72 | 9267.82 |
| 16 | 2.058 | 1.755 | 14464.28 | 16157.02 |

QPS8 default의 sampled active usage 최대 중앙값은 .99, running 최대64,
queue 최대36이며 retraction은 세 실행 모두0이다. radix를 끄면 goodput이22.4% 감소한다.
따라서 약9.32%의 hit을 ‘아무 효과 없음’으로 해석할 수 없다. 높은 입력률에서
queue와 prefill/decode 서비스 간섭이 나타나며 낮은 부하의 TTFT만으로 이를 예측할 수 없다.

같은 QPS8에서 chunk 크기2048만 지정한 control의 gap p99는4178.79ms로,
default 중앙값4173.51ms와 비슷하다. 반면 decode 서비스 기회를 주는 fair는977.10ms,
기존 mixed-chunk control은206.18ms다. KV 계산·prompt를 바꾸지 않고 gap이 크게
변하는 것은 **스케줄링 간섭** 가설을 지지한다. 캐시 효과, capacity admission과 queue가
함께 있으므로 ‘순수 prefill FLOP만의 병목’이나 ‘GPU 대역폭 병목’으로 확정하지 않는다.
GPU kernel profiler에 의한 시간/대역폭 분해는 이번 검증 범위에 포함하지 않았다.

전체 3회 범위와 [부하 곡선](../../artifacts_deterministic/sweep.png),
[출력 gate](../../artifacts_deterministic/verification.json)를 함께 읽는다.

이 분석의 결론은 관측 범위에 한정한다. 순서 정규화/위치 독립 KV 재사용은
별도 의미·정확성 문제를 가지므로 이번 lossless 엔진 비교에 포함하지 않는다.
Step 3에서는 원본 prompt를 보존한 채 prefill 스케줄의 반응성/처리량 교환을 시험한다.
