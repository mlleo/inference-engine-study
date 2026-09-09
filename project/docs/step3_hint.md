# Step 3 — 측정에서 엔진 변경으로

대상: W1 RAG, W2 Agent, W3 Reasoning 담당 학생 각 1명.
먼저 새 [Step 1·2 보고서](../bp/step1_2/)를 읽는다. 강사의 H100 수치는
학생 GPU의 목표 성능이 아니다. 학생은 자기 환경에서 같은 가설을 검증한다.
이 문서는 힌트이며 완성 구현/결과는 강사용 `bp/step3/`에 별도로 둔다.

## 1. 시작 전에 한 장으로 확정할 것

다음 네 문장을 자신의 실측값으로 채운다.

1. “부하 ___에서 ___ 지표가 나빠지고, 서버에서 ___를 관측했다.”
2. “___를 끈/줄인 대조 실험은 ___였으므로, 현재 가설은 ___다.”
3. “엔진의 ___ 결정만 바꾸면 ___가 줄어들 것이다.”
4. “대신 ___ 요청의 ___가 나빠질 수 있다.”

원하는 현상이 나오지 않으면 병목을 선언하지 않는다. SLO가 모두 만족되면
TTFT 개선과 goodput 개선을 구분하고, 포화점을 찾지 못했으면 탐색 범위를 쓴다.
입출력 토큰 비율만으로 GPU 계산 시간 비중을 결론 내리지 않는다.

## 2. RunPod 예산을 아끼는 실행 순서

| 단계 | 실행 | 넘어가는 조건 |
|---|---|---|
| CPU 준비 | 코드 읽기, trace 검증, 패치 OFF 확인, 분석 스크립트 준비 | GPU 없이 잡을 오류를 해결 |
| GPU 연결 | 모델/토크나이저 확보, 버전·GPU·KV 풀 기록, 짧은 요청 1–4개 | OOM·API 오류 없음 |
| 탐색 | 작은 trace로 default/ablated/mine 각 1회, 부하 2–3점 | 병목이 관측되고 패치가 실제 경로에 적용됨 |
| 검증 | 선택한 동일 trace에서 3-bar 각 3회, 순서 교대 | 출력 검증 통과, 분산까지 기록 |
| 손해 탐색 | 낮은 부하 또는 다른 워크로드의 작은 trace | 이득이 사라지거나 손해가 생기는 조건 설명 |
| 종료 | 결과·로그·trace·diff를 로컬로 다운로드한 뒤 pod 종료 | 파일을 로컬에서 읽을 수 있음 |

탐색 trace 예시: W1 요청 24–48개, W2 세션 8–16개 × 3–4턴,
W3 요청 12–24개 × 출력 평균 1000–3000토큰. 이는 실행 점검용이며
기본 trace의 성능 결과라고 부르지 않는다. 샘플이 작으면 p99는 거의 최대값이고
꼬리 확률 추정이 불안정하므로 요청 수와 반복별 값을 반드시 표시한다.

예산은 실제 계정의 시간당 가격 P와 측정한 실행시간으로 계산한다.
예상 비용 = P × (다운로드/설치 + 서버 시작 + 모든 재생 + 결과 복사 시간)/3600.
실행 전에 총 허용 시간을 정한다. W3는 QPS를 올려도 마지막 긴 요청을 기다리는
시간이 줄지 않을 수 있다. 반복할 trace를 줄일 때는 세 bar에 똑같이 적용한다.
원시 기록을 저장하면 표·분위수·그림을 다시 만들 때 GPU가 필요 없다.

## 3. H100과 학생 GPU를 비교하는 올바른 방법

강사 환경은 H100 80GB, Qwen3-4B BF16, TP=1이다. 모델의 full-attention KV는
토큰당 `2 × 36 layers × 8 KV heads × 128 head_dim × 2 bytes = 147456 bytes`
(144 KiB)다. allocator 정렬과 부가 메모리는 별도다.

24GB 전체를 이 값으로 나누면 안 된다. 가중치, CUDA graph, attention 작업 공간,
런타임 메모리를 제외한 **서버가 출력한 KV token capacity**를 사용한다.
같은 24GB라도 GPU 종류와 서버 설정에 따라 가용 풀이 다르다.

비교할 값은 다음과 같다.

- W1: 초당 uncached prompt tokens, 실행 중 decode 수, TTFT·최대 stream gap.
- W2: 완료되지 않은 세션 수, 다음 턴까지의 간격, 재사용 가능한 prefix 크기,
  free/evictable/active KV의 구분. `#running-req`는 think 중인 세션을 포함하지 않는다.
- W3: 동시에 생성 중인 시퀀스들의 KV 성장, queue, retracted requests, maxITL·TTFT.

강사는 H100에서 `--max-total-tokens 65536`으로 용량을 통제한 실험도 한다.
이는 KV 용량을 제한하는 실험이며 24GB GPU의 연산/대역폭 모사가 아니다.
학생은 먼저 기본 설정에서 실제 용량을 기록하고, 필요하면 더 작은 풀에서 원인을
확인한다. H100과 동일 QPS를 강제로 맞추거나 성능을 VRAM 비율로 환산하지 않는다.

## 4. W1 힌트 — exact prefix와 긴 prefill을 분리하라

관찰할 것: 검색된 문서의 중복률과 실제 토큰 prefix 적중률은 다르다.
동일 문서라도 앞 문서가 달라지면 causal attention의 KV가 달라질 수 있다.
문서 순서를 바꾸거나 chunk 경계를 정렬한다고 정확한 KV가 자동 재사용되지 않는다.

낮은 힌트: 긴 prefill이 들어오는 순간 이미 스트리밍 중인 요청들의 gap이 커지는가?
TTFT만 보지 말고 decode 요청 수와 maxITL을 같은 시간축에서 보라.

구현 방향 힌트: decode가 거의 없을 때는 큰 prefill이 효율적일 수 있고,
decode가 많을 때는 작은 prefill budget이 반응성을 개선할 수 있다.
`PrefillAdder`의 현재 batch budget이 어디서 정해지고 소모되는지 따라가라.
기존 chunk 크기 고정 플래그를 먼저 시험한 뒤 동적 정책의 추가 기여를 비교한다.
chunk 크기는 줄었는데 gap이 그대로라면 scheduler가 다음에도 prefill을 선택하는지
따라가라. 작은 batch와 decode에 실제 서비스 기회를 주는 정책은 다르다.

불변 조건: prompt bytes, token IDs, retrieval order, sampling, 출력 길이를 바꾸지 않는다.
budget은 새 요청뿐 아니라 chunk continuation 경로에서도 어떤 의미인지 확인한다.
부하가 높을 때 TTFT가 나빠지고 gap만 좋아질 수 있다. 이것을 숨기지 않는다.

피할 결론: “정규화로 2배 빨라졌으니 lossless 엔진 최적화다.”
정규화는 별도 serving-pipeline 실험이며 순서·인용·정답 품질 평가가 필요하다.

## 5. W2 힌트 — 재사용 가치와 보존 비용을 함께 계산하라

관찰할 것: root 요청과 후속 턴을 나누고, `cached_tokens / prompt_tokens`와
uncached token 수를 함께 표시한다. 접미사가 길어지면 eviction 없이도 적중 비율이
낮아질 수 있다. output의 재토큰화 경계도 점검한다.

낮은 힌트: 최근에 접근한 짧은 prefix와 조금 오래된 긴 prefix 중 무엇을 먼저
evict해야 하는가? “오래된 것”과 “다시 계산하기 싼 것”은 같은 기준이 아니다.

구현 방향 힌트: 기존 radix cache의 **evict 가능한 노드 사이 순서**만 바꾸면
KV 값 계산과 allocator를 건드리지 않고 가설을 시험할 수 있다.
비용 credit에 상한을 두어 오래된 항목이 영구히 우선권을 갖지 않게 하라.
원래 LRU, 기존 LFU/다른 정책, 자신의 정책을 구분해 비교한다.

불변 조건: 실행 중 prefix의 lock/refcount를 무시하지 않는다. 필요할 때 충분한
토큰을 회수할 수 있어야 한다. 노드 split·부모가 leaf가 되는 경우도 확인한다.
추가 순회 비용과 짧은 세션의 손해를 측정한다.

업스트림에 session-aware cache가 이미 있을 수 있다. 설치 commit에서 확인하라.
`session_id` 전달은 대화 이력을 자동으로 만드는 기능이 아니며,
해당 API를 쓰면 정상·실패·취소 경로의 session close와 만료 전략까지 필요하다.
기존 기능 활성화와 새 정책 구현은 보고서에서 구분한다.

## 6. W3 힌트 — retraction을 줄인 대가를 찾아라

관찰할 것: 전체 trace 출력 합을 KV 수요로 쓰지 않는다. 실제 동시 요청의 KV가
풀에 가까워지는지, 직접 retraction 로그가 있는지 먼저 확인한다.
retraction에서 기존 토큰 ID를 다시 생성하는 것과 KV를 다시 계산하는 것은 다르다.

낮은 힌트: 처음부터 너무 많은 긴 요청을 admit한 뒤 선점하는 대신,
실행 중 요청이 앞으로 쓸 공간을 일부 예약하면 어떤 일이 일어나는가?

구현 방향 힌트: running + 이번에 admit할 요청들의 peak KV를 보수적으로
추정하고 새 요청의 입장을 늦춘다. 기본 구현의 max_new_tokens clipping,
new_token_ratio, evictable 공간 취급을 먼저 읽는다.

이 toy trace는 `ignore_eos=True`라 출력 예산이 실제 길이와 거의 같다.
그 값을 사용하는 정책은 **합성 trace의 알려진 길이를 이용한 예약**이다.
실서비스의 출력 길이 예측기를 개발했다고 주장하지 않는다.

불변 조건: 요청 하나도 시작 못 하는 deadlock, chunk continuation 차단,
재진입 요청의 영구 대기, 공유 prefix의 이중 계산, page 정렬을 점검한다.
입장을 늦추면 retraction/maxITL이 줄어도 TTFT가 크게 늘 수 있다.
기존 `--max-running-requests` 고정 제한보다 나은지도 확인하라.

원래 W3 SLO는 평균 TPOT≤60ms뿐이다. 10초 정체도 많은 토큰으로 나뉘면 통과할 수 있다.
원래 goodput을 그대로 보고하고, TTFT/maxITL/E2E를 별도로 공개한다.
추가 SLO를 보고 싶으면 결과를 보기 전에 기준을 정하고 보조 지표라고 표시한다.

## 7. 공정한 3-bar와 정확성

`default`는 실제 설치 upstream 설정, `ablated`는 관련 기능만 제거,
`mine`은 동일 조건에서 내 플래그만 ON이다. OFF 복사본도 upstream과 비교한다.
‘default=모든 기능 ON’은 아니다. 실제 기본값과 resolved settings를 저장한다.

탐색은 1회, 최종 선택 조건은 세 bar 각각 3회. 측정 순서를 교대하고
중앙값과 min/max 또는 각 반복 값을 공개한다. 기여는 mine/default로 계산한다.
ablated/default는 기존 기능의 효과를 설명한다. 기존 튜닝이 있다면 4번째 control을 둔다.

서버는 bar마다 새로 시작한다. 워밍업 후 캐시 시작 상태를 동일하게 맞춘다.
서버 준비 실패·이전 프로세스 재사용·오류 응답이 있으면 해당 bar는 성능 결과에서 제외한다.
클라이언트 semaphore에서 기다린 시간을 숨기지 않도록 scheduled→submit 지연을 확인한다.
W2는 세션 시작만 open loop이며 후속 턴은 부모 완료에 종속된다.

```bash
# project/에서 실행. 결과 파일의 실제 이름을 사용한다.
python -m bench.verify results/default.json results/mine.json
python -m bench.metrics 'results/*.json' --csv summary.csv
python -m bench.metrics 'results/*.json' --by turn_idx
```

`--keep-output`이 필요하다. 새 replay는 입력 hash도 남긴다. W2에서 이전 출력이
달라지면 다음 prompt도 달라지므로, 최초 분기 턴부터 조사한다.
95% 기준은 자동 screening이지 정확성 증명이 아니다. 초반 불일치라고 곧바로
KV 버그로 단정하지 말고 greedy 단일 요청, OFF/ON, tokenizer, backend를 확인한다.
강사 실험에서는 ordinary upstream 반복도95%를 통과하지 못했다. 이런 경우
기준을 낮추지 말고 지원되는 deterministic mode를 모든 bar에 공통 적용한 별도
비교를 만든다. FA3의 radix 호환성을 설치 버전에서 확인하고, 일반 실행 성능과 분리한다.
각 SSE chunk에 여러 토큰이 있으면 maxITL은 개별 토큰 간격이 아니라 stream gap이다.

## 8. 제출 체크

- 한 가지 가설, engine diff, ON/OFF 방법, 실제 설치 commit과 모든 플래그.
- trace hash, tokenizer/model 정보, GPU와 실제 KV capacity.
- 3-bar × 3회 원시 JSON, 서버 로그, 정확성 결과, 반복별 표.
- 부하 곡선 또는 탐색 범위와 “무릎 미관측”이라는 정직한 결론.
- 기존 튜닝과의 차이, 손해/무효 조건, RunPod 실행시간·예산 소모 기록.

성능 향상 폭 자체보다 원인을 분리하고 실패를 설명하는 것이 평가의 핵심이다.
원하는 결과를 얻기 위해 prompt·출력 길이·SLO를 mine에서만 바꾸지 않는다.
