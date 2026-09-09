# W1 풀이 — chunking만으로 decode가 서비스되지는 않는다

## 근거에서 구현까지

새 baseline은 낮은 부하에서 queue/retraction이 없었지만 QPS8 pressure에서는
prefill 대기와 수초 stream gap이 나타났다. 기존 “RAG라서 prefill만 고치면 된다”는
설명보다 실제 scheduler가 decode를 언제 선택하는지가 중요했다.

첫 구현(`STUDY_OPT=rag`)은 running decode≥8일 때 PrefillAdder의 input/chunk
budget을2048로 줄였다. 로그에서 실제2048토큰 prefill이 반복되는 것은 확인했으나
성능 차이는 작았고 큰 stream gap도 남았다. 코드 경로를 다시 읽으면 이유가 명확하다.
`get_next_batch`는 새 prefill이 있으면 그것을 먼저 선택한다. **한 번의 prefill을
짧게 만드는 것과 decode 사이의 대기시간을 제한하는 것은 다른 정책**이다.

수정 구현(`STUDY_OPT=rag_fair`)은 decode≥8일 때 prefill batch를 한 번 선택한 뒤
다음 기회에는 빈 prefill plan을 반환한다. 그러면 기존 scheduler가 native decode
경로를 실행한다. 다음 번에는 다시 prefill이 가능하다. original chunk 크기는 유지한다.
대기열이 없거나 decode가 적으면 native 동작을 따른다.

## 구현 불변 조건

- prompt 순서·token IDs·KV 내용·출력 길이는 바뀌지 않는다.
- KV allocation이나 refcount를 직접 수정하지 않는다.
- unfinished chunk는 scheduler의 기존 상태에 남고 다음 prefill 기회에 계속 처리한다.
- OFF에서는 기존 NextBatchPlan과 native scheduling을 사용한다.
- 검증 범위는 Qwen3-4B BF16, TP1, classic radix, 일반 autoregressive 실행이다.
  speculative decoding, disaggregation, multimodal/grammar의 별도 조합은 검증하지 않았다.

## 대조와 결과 읽기

주요3-bar는 default / radix OFF(ablated) / fair다.
최초 budget 축소(mine)는 실패한 시도로 따로 표시한다.
`--chunked-prefill-size 2048`(tuned)와 `--enable-mixed-chunk`(mixed)는 기존 기능 대조다.
mixed가 더 좋으면 그것을 우선 사용해야 하며 새 코드의 기여를 과장하지 않는다.

QPS4/8/16에서 goodput, TTFT p99, max stream gap을 함께 비교한다.
TTFT SLO는4000ms로 고정했고 gap 개선을 SLO 변경으로 goodput 개선처럼 만들지 않았다.
각 수치와 반복 범위는 [controlled 측정표](../../artifacts_deterministic/measurements.md),
출력 비교는 [검증 로그](../../artifacts_deterministic/verification.json)에 있다.

## 최종 실측 요약

QPS8,120요청, deterministic FA3. default/fair는3회 중앙값, control은1회다.

| 정책 | 반복 | goodput req/s | TTFT p50 ms | TTFT p99 ms | gap p99 ms |
|---|---|---|---|---|---|
| default | 3 | 3.098 | 1606.30 | 7632.72 | 4173.51 |
| 최초 budget 변경(mine) | 2 | 3.096 | 1533.12 | 7636.58 | 4187.98 |
| 수정 fair | 3 | 3.144 | 1942.10 | 7551.81 | 977.10 |
| 기존 chunk size2048 | 1 | 3.104 | 1519.71 | 7629.47 | 4178.79 |
| 기존 mixed chunk | 1 | 3.266 | 1850.96 | 7342.09 | 206.18 |
| 패치 설치, switch OFF | 1 | 3.094 | 1632.93 | 7654.53 | 4190.34 |

fair는 gap을76.6% 줄였지만 TTFT p50은20.9% 악화됐다. goodput은1.5%만 증가했다.
QPS4에서는 goodput3.650→3.651로 거의 같고 gap1703.23→310.00ms,
QPS16에서는 goodput2.058→1.866으로9.3% 감소하지만 gap8622.62→1000.84ms다.
즉 부하에 걸쳐 streaming 반응성 교환은 나타나지만 범용 goodput 개선은 아니다.

**단일 mixed-chunk control이 fair보다 goodput과 gap 모두 좋았다.** 3회 반복의
우월성 검정은 아니므로 확정 순위를 주장하지 않되, 실용적인 다음 선택은 기존 mixed
기능의 반복 검증이다. custom fair가 upstream보다 새로운 최선이라고 제출하지 않는다.
모든 수집된 W1 비교는 동일 prompt/길이와100% 출력 일치를 통과했다.
정확한 수정본은 [fair.patch](../../experiments/fair.patch), 최초본은
[study.patch](../../experiments/study.patch)다.

## 비용과 적용 판단

decode를 더 자주 서비스하면 prefill이 기다리는 시간이 늘 수 있다. 낮은 부하에서는
효과가 없고, GPU의 큰 prefill 효율을 희생할 수 있다. queue를 없애는 정책이 아니다.
TTFT와 stream gap 중 어떤 SLO가 실제로 중요한지 먼저 정해야 한다.
이 풀이의 핵심은 최초 시도를 성공으로 포장하지 않고, 로그와 scheduler 분기를
대조해 다음 가설을 만든 과정이다.
