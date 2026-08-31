# W3 — 장문 추론 (Long CoT)

> **배정자:** ____________

## 시나리오

추론 모델 서빙. 입력은 350 토큰으로 짧지만 출력이 **중앙값 3,000 토큰,
최대 16,000 토큰**까지 간다(롱테일 lognormal). 시퀀스당 KV 가 수 분 동안 계속
커진다.

```bash
python -m workloads.generators reasoning --out traces/ --model Qwen/Qwen3-4B
bash scripts/run_baseline.sh reasoning
```

## 주 병목

**Decode 지배 + KV 증가.** prefill 은 무시할 만하다. 문제는 메모리다.

## 이 워크로드의 진짜 함정

**스케줄러는 요청이 얼마나 길게 생성할지 미리 알 수 없다.**

그래서 낙관적으로 요청을 받아들이고(admit), KV 가 차면 실행 중인 요청을
**선점(retract)** 해서 뱉어낸다. 뱉어낸 요청은 나중에 **처음부터 다시**
prefill 된다. 이미 생성한 토큰이 통째로 버려진다.

**1차 목표:** `#retracted` 가 0 이 아니게 되는 QPS 지점을 찾아라.

```bash
grep -E "#retracted|token usage|#running-req" logs/server_default.log | tail -40
```

`token usage` 가 1.0 근처에 붙어 있고 `#running-req` 가 톱니처럼 오르내리면
retraction 이 일어나고 있는 것이다. **이 로그 발췌가 2주차 발표의 핵심 증거다.**

`analyze.py` 의 [2] 동시 요청 수 그래프에서 배치가 커졌다 무너지는 패턴도 함께 볼 것.

## SGLang 이 이미 하는 것

- 연속 배칭(continuous batching)
- Retraction / 선점 기반 메모리 회수
- Chunked prefill
- Hierarchical KV 오프로딩

## SGLang 이 아직 못 하는 것 = 과제 후보

1. **출력 길이 예측 기반 admission control** — 프롬프트로부터 출력 길이를
   대략 예측(작은 분류기 또는 휴리스틱)하고, KV 여유가 없으면 아예 받지 않는다.
   retraction 을 사전에 막는 쪽이 사후에 처리하는 것보다 항상 싸다.
   난이도 중간. **추천.**
2. **KV 헤드룸 예약** — 실행 중인 요청들이 앞으로 소비할 KV 를 추정해 그만큼
   비워두고 admit 한다. 예측 없이도 가능한 보수적 버전.
3. **선점 정책 개선** — 현재는 대체로 후순위 요청을 뱉는다. "이미 생성한 토큰이
   가장 적은 요청"을 뱉으면 버려지는 작업량이 최소화된다. 난이도 낮음, 효과 명확.

## 스윕할 축

```bash
--set qps=0.2 / 0.35 / 0.6 / 1.0 / 2.0   # ★ retraction 발생 지점 찾기
--set out_mean=1000 / 3000 / 8000        # KV 압박 강도
--set out_sigma=0.3 / 0.6 / 1.0          # 꼬리 두께 (예측 난이도)
```

`out_sigma` 가 특히 중요하다. 꼬리가 두꺼울수록 길이 예측이 어려워지고,
예측 기반 방법의 이득이 줄어든다. 이 곡선이 보고서의 핵심 그림이 된다.

## 확인할 서버 로그

```
#retracted        ★ 0 이 아니면 그 자체가 이야기다
token usage       1.0 근처에 붙어 있는가
#running-req      톱니 패턴이면 배치 붕괴
gen throughput    실효 생성 처리량
```

## 예상되는 트레이드오프 (5단계에서 확인할 것)

- **보수적 admission 은 GPU 를 놀린다.** retraction 은 줄지만 처리량도 준다.
  이 교환비를 정량화하는 것이 이 과제의 본체다.
- 길이 예측이 틀리면(특히 과소 예측) 상황이 더 나빠진다. 예측 오차 분포와
  최종 goodput 의 관계를 보여라.
- 짧은 요청만 오는 워크로드(W4)에서는 예측 오버헤드가 순수 비용이다.
- 예약 헤드룸은 KV 를 놀리는 것이므로 W1/W2 처럼 KV 를 많이 쓰는 워크로드에서
  손해가 난다.
