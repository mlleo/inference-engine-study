# W4 — 구조화 출력 (JSON 스키마 추출)

> **배정자:** ____________

## 시나리오

문서에서 필드를 뽑아 JSON 으로 반환하는 추출 API. 입력 450 토큰, 출력 110 토큰.
**QPS 12** 로 높다. 스키마는 4종인데 분포가 치우쳐 있다
(invoice 45%, ticket 40%, resume 8%, contract 7%).

```bash
python -m workloads.generators structured --out traces/ --model Qwen/Qwen3-4B
bash scripts/run_baseline.sh structured
# 문법 제약을 켜고 재생하려면
python -m bench.replay --trace traces/structured.jsonl --tag grammar --grammar ...
```

## 주 병목

**요청당 CPU 오버헤드.** 이 워크로드에서 GPU 는 놀고 있다.

## 이 워크로드의 진짜 함정

**여기서는 GPU 최적화가 아무 의미가 없다.**

모든 요청이 작다. 배치는 금방 채워진다. 비용은 전부 CPU 쪽에 있다:
스케줄러 루프, detokenize, 문법(grammar) FSM 컴파일, HTTP 파싱, Python 오버헤드.

`--disable-cuda-graph` 와 `--disable-overlap-schedule` 을 각각 켜보면 이게
드러난다. 다른 워크로드에서는 미미한 이 플래그들이 여기서는 크게 움직인다.
**이것이 이 워크로드의 정체를 보여주는 첫 실험이다.**

**1차 목표:** 스키마 분포가 치우쳐 있으므로, 문법 캐시가 동작한다면 **희귀
스키마(resume/contract)의 첫 요청에서만** 컴파일 비용이 보여야 한다.
실제로 그런지 확인하라.

```bash
python -m bench.metrics 'results/structured__*.json' --by schema_name
```

resume/contract 의 TTFT p99 가 invoice/ticket 보다 유의미하게 높다면 문법 캐시가
제대로 안 먹고 있는 것이다.

## SGLang 이 이미 하는 것

- xgrammar 백엔드 + 문법 캐시
- Jump-forward decoding (스키마상 확정된 토큰은 건너뜀)
- Overlap scheduler (CPU 스케줄링과 GPU 실행 중첩)
- CUDA graph

## SGLang 이 아직 못 하는 것 = 과제 후보

1. **문법 캐시 정책 개선** — 현재 캐시 키·수명·워밍업 전략을 분석하고, 치우친
   스키마 분포에 맞게 개선. 콜드 스타트 컴파일을 요청 경로 밖으로 빼내는 것도 포함.
   난이도 낮음~중간, 측정이 깔끔하다. **추천.**
2. **FSM 전진의 배치화** — 현재 요청별로 CPU 에서 마스크를 만든다. 같은 스키마
   요청들을 묶어 한 번에 처리하면 CPU 시간이 줄어든다. 치우친 분포에서 특히 유리.
3. **스케줄러 루프 프로파일링 후 최적화** — `py-spy` 로 스케줄러 스텝의 CPU
   시간을 분해하고, 짧은 요청 다수 상황의 hot path 를 줄인다. 발견 자체가 성과.

**이 워크로드는 프로파일러를 쓰는 것이 필수다.**
```bash
py-spy top --pid <scheduler pid>
py-spy record -o profile.svg --pid <scheduler pid> --duration 60
```

## 스윕할 축

```bash
--set qps=4 / 12 / 24 / 48        # ★ CPU 포화 지점 찾기
--set doc_tokens=200 / 450 / 900  # GPU 쪽 비중 조절
# 스키마 분포는 generators.py 의 weights 를 수정해 실험
```

## 확인할 서버 로그

```
#running-req      배치는 큰데 처리량이 안 나오면 CPU 병목
gen throughput    GPU 활용도 대비 실효 처리량
```
`nvidia-smi dmon` 으로 GPU 사용률이 낮게 유지되는 것을 함께 보여주면 강력한 증거다.

## 예상되는 트레이드오프 (5단계에서 확인할 것)

- 문법 캐시를 크게 잡으면 **메모리를 먹고**, 스키마가 다양한 환경에서는
  적중률이 낮아 순수 손해가 된다. 스키마 종류 수를 늘려가며 손익분기점을 찾아라.
- FSM 배치화는 **묶기를 기다리는 지연**을 추가한다. QPS 가 낮으면 손해다.
  낮은 QPS 구간을 반드시 곡선에 포함시켜라.
- 긴 요청 워크로드(W1/W3)에서는 CPU 최적화의 효과가 측정 오차에 묻힌다.
  교차 검증에서 "효과 없음"이 나오는 것도 정당한 결과다. 그대로 보고하라.
