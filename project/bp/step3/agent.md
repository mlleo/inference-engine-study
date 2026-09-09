# W2 풀이 — 비싼 prefix에 제한된 보존 credit

## 가설과 작은 변경

새 pressure 분석에서는 캐시 miss뿐 아니라 active KV 포화와 queue가 함께 나타났다.
radix OFF는 명확히 악화되지만, 그것이 곧 새로운 eviction 정책이 크게 이긴다는 뜻은 아니다.

`STUDY_OPT=agent`는 classic radix의 evictable leaf 순서만 바꾼다.
LRU의 마지막 접근 시각 대신 다음 score를 사용하며 작은 값부터 회수한다.

```text
score = last_access_time + min(16 seconds, prefix_depth_tokens / 512)
```

prefix depth는 leaf부터 root까지 key 길이를 더한 값이다. 긴 prefix를 다시 계산하는
비용이 크다는 휴리스틱이며 실제 GPU 비용 모델이나 다음 턴 예측기는 아니다.
16초 상한은 오래된 항목의 우선권이 무한히 커지는 것을 방지한다.
이는 session_id 기반 upstream session-aware radix와도 다른 정책이다.

## 보존한 불변 조건

native `evictable_leaves`, lock_ref, allocator.free_segment, leaf 삭제,
부모가 새 leaf가 되는 처리, eviction metrics는 그대로 사용한다.
보호가 hard pin으로 바뀌지 않으며 공간이 필요하면 회수할 수 있다.
CPU invariant test는 공유 prefix split 후 한 branch를 lock하고 다른 branch만
회수하는지, unlock 후 나머지를 정확히 한 번 회수하는지를 OFF/ON에서 검증한다.

## 평가와 한계

80세션×6턴, think_mean4, KV65536을 고정하고 session_qps2/4/8을 비교한다.
주요3-bar는 LRU default / radix OFF / mine이며, 기존 LFU는 추가 대조다.
동일 series 안의 동일 trace hash, sampling, 모든 후속 prompt hash를 비교한다.

처음 ordinary mode 결과는 후속 prompt가 달라 엄격한 gate를 통과하지 못했다.
FA3 deterministic mode를 공통 적용한 새 비교에서는 원래 출력 gate를 유지한다.
측정값은 [controlled 표](../../artifacts_deterministic/measurements.md)에 있고,
[verification.json](../../artifacts_deterministic/verification.json)에 비교별 로그가 있다.

## 최종 실측 요약

session_qps4,80세션×6턴,KV65536, deterministic FA3. 주요 bar는3회 중앙값이다.

| 정책 | 반복 | goodput req/s | TTFT p50 ms | TTFT p99 ms | gap p99 ms | hit % |
|---|---|---|---|---|---|---|
| default LRU | 3 | 3.631 | 279.80 | 4204.35 | 281.44 | 47.269 |
| bounded credit | 3 | 3.587 | 229.23 | 4072.77 | 345.13 | 47.223 |
| radix OFF | 3 | 1.447 | 3620.38 | 9298.38 | 561.80 | 0 |
| 기존 LFU | 1 | 3.641 | 228.92 | 4085.38 | 213.59 | 47.785 |
| 패치 설치, switch OFF | 1 | 3.575 | 225.58 | 4084.12 | 294.28 | 47.318 |

새 정책은 TTFT p50을18.1% 줄였지만 goodput은1.2% 감소하고 gap은22.6% 증가했다.
goodput 범위도 default3.543–3.667, mine3.540–3.632로 겹친다.
OFF control의 TTFT p50도225.58ms이므로 중앙값 TTFT 감소 전체를 새 정책의
인과 효과로 주장하지 않는다. hit 향상도 없었다. **기본 채택을 뒷받침하지 못한 후보**다.
기존 LFU 단일 control은 새 코드보다 좋은 goodput/gap을 보였으나 반복 검증 전의 신호다.

session_qps2의 goodput은4.637→4.757(+2.6%), q8에서는3.827→3.651(−4.6%)였다.
한 점의 이득을 일반화하지 않는다. 모든 비교에서480개 전체 출력과 후속 prompt hash가
일치했다. 큰 풀 control은 hit84.77%, goodput6.374로 개선되므로 cache/capacity 병목
가설은 지지되지만, 이번 보존 휴리스틱의 타당성은 지지되지 않는다.

bounded credit이 cache hit을 반드시 올린다는 보장은 없다. 최종 턴이 끝난 긴
prefix는 비싸지만 다시 쓰이지 않을 수 있다. 이 정책에는 종료 의도 정보가 없으므로
그 항목을 보존하며 짧은 세션을 희생할 수 있다. 또한 eviction 시 후보마다
부모 경로를 순회하므로 기존 LRU보다 CPU 비용이 크다.

개선이 반복 편차보다 작거나 goodput이 떨어지면 기본으로 켜지 않는다.
현재 가설이 실패하면 다음 과제는 단순히 credit을 키우는 것이 아니라,
재사용 확률과 재계산 비용을 분리하고 기존 session-aware cache/LFU와 비교하는 것이다.
기능을 켠 것만으로 신규 기여라고 하지 않는 것이 이 풀이의 주요 교훈이다.
