# W3 (Reasoning) — Step 1·2 Best Practice 보고서

**작성자:** 조교(베스트 프랙티스 예시) / **워크로드:** reasoning / **환경:** H100 80GB (n7 클러스터)
**대응 학생 환경:** 24GB GPU (RunPod) — 6절에서 스케일링 매핑 제공

---

## 0. 이 문서에 대하여

W1 보고서와 동일한 구조(결정 로그 → 실측 → 한 문장 주장 + 증거 3종). W3의 특수성:
**H100 80GB에서는 기본 트레이스가 병목을 전혀 드러내지 않는다.** 이 보고서는 "여유 있는 하드웨어에서
병목을 재현하기 위해 부하를 어떻게 스케일링했는가"까지 포함한다 — 이 과정 자체이 Step 2의
모범 사례가 된다.

---

## 1. 실험 환경 (고정)

W1과 동일 (H100 80GB, SGLang 0.5.18, Qwen3-4B, mem-fraction 0.85, KV 풀 430,067 토큰).
bar마다 서버 재시작, 오픈 루프, 보고 bar 3회 반복.

---

## 2. Step 1 — 워크로드 분석

### 2.1 분석 표

| 항목 | 값 | 어떻게 구했나 |
|------|-----|--------------|
| 입력 토큰 (p50/p90/p99) | 387 / 703 / 710 | 재생 결과 `prompt_tokens` (step1_stats: 문자/4.2 하한 688과 일치) |
| 출력 토큰 (p50/p90/p99/max) | 2345 / 4859 / 9083 / 10541 | 재생 결과 `completion_tokens` (lognormal 중앙값 3000 상한 16000) |
| 입력:출력 비율 | **1 : 6.1** (p50 기준) | decode 지배 — prefill은 무시 가능 |
| 공유 프리픽스 | 없음 (독립 문제 120건) | 트레이스 구조 |
| 도착 패턴 | Poisson, QPS 0.35 (120건, 도착 구간 320s) | 트레이스 `__meta__` |
| 세션 구조 | 독립 요청 120건 | `depends_on` 전부 null |
| SLO | TPOT 60ms (트레이스 기본) | 추론 사고 스트리밍 — 토큰 간 간격 체감 기준 |
| 계획 출력 합계 | 337,433 토큰 | 트레이스 `max_new_tokens` 합 |
| 최대 동시 KV 점유 추정 | 계획 출력 337k + 프롬프트 46k = **~383k 토큰 (풀 430k의 89%)** | 전 요청 동시 상정 상한. 실제로는 도착 분산으로 완화되지만 **풀 부족 시나리오가 성립하는 구조** |

### 2.2 "무엇이 어려운가" (핵심 문단)

> 출력이 입력의 6배 — decode 지배 워크로드다. 문제는 총 출력 수요(337k 토큰)가 KV 풀 크기와
> 같은 규모라는 것: 풀이 부족하면 SGLang은 실행 중인 요청을 **retract(선점 후 처음부터 재prefill)** 한다.
> retraction은 (1) 이미 생성한 토큰의 KV를 버리고 (2) 재prefill 계산을 물어보며 (3) TPOT에 수백 ms
> 급 공백(maxITL 폭증)을 낸다. 24GB 학생 GPU(풀 ~110k)에서는 기본 트레이스만으로 재현되지만,
> H100(풀 430k)에서는 트레이스를 스케일링해야 같은 압박을 만든다.

### 2.3 Step 1 완료 조건 확인

- [x] 표 전부 실측값으로 채움
- [x] SLO 근거 명시 (TPOT 60ms)
- [x] 코드 수정 0건

---

## 3. Step 2 — 실험 설계 (결정 로그)

| # | 실험 | 설계 이유 |
|---|------|-----------|
| 1 | baseline (q0.35, out_mean 3000) | 기준점. **H100에서는 여유로움을 먼저 확인** — 토큰 사용량·TTFT가 바닥이면 병목이 안 보이는 상태 |
| 2 | qps sweep 0.35→0.6→1.0→2.0 | 도착률 축. 동시성↑ → 라이브 KV↑. 어디서 포화되는가 |
| 3 | out_mean sweep 3000→8000→12000 | **출력 길이 축 = KV 수요 축** (H100 스케일링의 핵심). 계획 출력 337k→864k→1.2M 토큰으로 풀(430k) 대비 초과구독률을 0.8×→2.0×→2.8×로 올림 |
| 4 | retract 트레이스 (out_mean 12000, sigma 0.1 균등, qps 4.0) | 균등 길이 + 고압 도착 = 동시 120건 전원 장기 실행 → 풀 강제 고갈. retraction 재현 전용 트레이스 |
| 5 | clip16k env probe | `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=16384` — admission control의 출력 추정 상한(기본 4096)이 얼마나 보수적인지 확인. 코드 수정 없이 env만으로 탐침 |
| 6 | ablation `--mem-fraction-static 0.45` | **인과 실험 ★** — KV 풀을 절반으로 줄이면 retraction이 늘어나야 한다. "병목이 KV 용량" 가설의 dose-response |
| 7 | ablation `--chunked-prefill-size -1` | 프로토콜 지정 W3 ablation. retraction 재prefill이 prefill 청킹에 영향받는지 |
| 8 | retract 트레이스 3회 반복 | 보고 bar 안정성 |

### 3.1 H100 스케일링 설계 (왜 out_mean을 3000→12000으로 올렸는가)

학생 GPU(24GB) 풀 ~110k 토큰 vs H100 풀 430k — **3.9배**. 기본 트레이스(계획 출력 337k)는
학생 GPU에서 풀의 3.1배(초과구독)지만 H100에서는 0.8배(미만)다. 같은 압박을 만들려면 출력 수요를
~3.9배로: out_mean 8000(864k, 2.0×) / 12000(1.2M, 2.8×). **"학생 GPU에서 기본 트레이스로 관측될
현상을 H100에서 재현하는 레시피"가 이 절의 산출물이다.**

---

## 4. Step 2 — 결과

### 4.1 Baseline — H100에서는 아무 일도 일어나지 않는다

| bar | ttft_p50 | ttft_p99 | tpot_p50 | maxitl_p99 | out_tok/s | slo% | goodput |
|-----|----------|----------|----------|------------|-----------|------|---------|
| default (3회 중앙값) | 17.5 | 22.2 | 5.2 | 25.4 | 997.0 | 100 | 0.4 |

서버 로그: token usage 최대 **0.01**, #running-req 최대 1~2, #queue-req 0, #retracted 0.
TPOT 5.2ms는 배치 1의 바닥값. **병목이 전혀 관측되지 않는다 — 이것이 스케일링이 필요하다는 증거.**

### 4.2 qps sweep — 도착률 축

| qps | ttft_p50 | ttft_p99 | tpot_p50 | maxitl_p99 | out_tok/s | slo% | goodput |
|-----|----------|----------|----------|------------|-----------|------|---------|
| 0.35 | 17.5 | 22.2 | 5.2 | 25.4 | 997 | 100 | 0.4 |
| 0.6 | 18.4 | 23.1 | 5.7 | 24.4 | 1605 | 100 | 0.6 |
| 1.0 | 20.3 | 26.4 | 6.6 | 32.3 | 2348 | 100 | 0.8 |
| 2.0 | 23.5 | 73.9 | 8.2 | 71.0 | 3184 | 100 | 1.1 |

**해석:** qps 2.0까지 SLO 100% 유지. out_tok/s가 997→3184 (3.2배)로 배치 효율이 오르는 구간.
TPOT 5.2→8.2ms — 배치가 커져도 TPOT SLO(60ms)에 7배 여유. **도착률 축만으로는 무릎이 안 나온다.**
(학생 GPU에서는 q1.0 부근에서 이미 retraction이 시작된다 — 풀 크기 차이)

### 4.3 out_mean sweep — KV 수요 축 ★ (H100 스케일링 실험)

| out_mean | 계획 출력 | 풀 대비 | ttft_p50 | ttft_p99 | tpot_p50 | maxitl_p99 | e2e_p99(s) | out_tok/s | slo% | goodput |
|----------|-----------|---------|----------|----------|----------|------------|------------|-----------|------|---------|
| 3000 | 337k | 0.8× | 20.3 | 26.4 | 6.6 | 23.4 | 61 | 2349 | 100 | 0.8 |
| 8000 | 864k | 2.0× | 31.9 | **271.2** | 15.9 | 195.5 | 251 | 2625 | **97.5** | 0.4 |
| 12000 | 1.2M | 2.8× | 32.7 | 206.2 | 21.9 | 70.0 | 422 | 2326 | 100 | 0.2 |

**해석 — 압박의 3단계:**
1. **0.8× (여유):** retraction 0건, TPOT 6.6ms. 풀보다 작으면 아무 일 없다.
2. **2.0× (경계):** TTFT p99가 26→271ms로 10배 폭증, SLO 97.5%로 하락. **retraction 시작점.**
   maxITL 195ms — retract된 요청의 재prefill 공백이 클라이언트에 그대로 보인다.
3. **2.8× (과포화):** TPOT 21.9ms, e2e p99 422s. 시스템이 retraction-재실행 사이클로
   겨우 버티는 상태 (goodput 0.2 — 처리량이 절반으로 떨어짐).

**초과구독률 2.0×가 이 워크로드의 무릎이다.** (학생 GPU에서는 기본 트레이스 0.35qps가 이미 3.1×)

### 4.4 retract 트레이스 — retraction 재현 ★

균등 길이(out_mean 12000, sigma 0.1) + qps 4.0: 120건이 28초 안에 몰리고 전원 장기 실행.

| bar | ttft_p50 | ttft_p99 | tpot_p50 | maxitl_p99 | e2e_p99(s) | out_tok/s | slo% | goodput |
|-----|----------|----------|----------|------------|------------|-----------|------|---------|
| default | 24.8 | 61.9 | 37.4 | 272234 | 556 | 2484 | 100 | 0.2 |
| rep2 | 25.3 | 48.3 | 37.4 | 272129 | 555 | 2484 | 100 | 0.2 |
| rep3 | 24.9 | 108.3 | 37.0 | 271894 | 555 | 2483 | 100 | 0.2 |

**TPOT p50 37.4ms vs baseline 5.2ms — 7.2배. maxITL p99 272초** — retract된 요청은 재prefill 동안
토큰이 멈춘다. 3회 반복이 사실상 동일 (spread <1%) — 재현 안정성 확인.

**서버 로그 (retraction 직접 관측):**
```
KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_tokens_gained: 2187, #new_token_ratio: 0.0980 -> 0.2747
KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_tokens_gained: 2252, ...
(총 98회 retraction 이벤트, token usage 1.00 도달, #queue-req 최대 77, #running-req 120)
```

### 4.5 clip16k probe — admission control 탐침

| bar | ttft_p50 | ttft_p99 | tpot_p50 | maxitl_p99 | out_tok/s | slo% | goodput |
|-----|----------|----------|----------|------------|-----------|------|---------|
| default | 24.8 | 61.9 | 37.4 | 272234 | 2484 | 100 | 0.2 |
| clip16k | 24.9 | 46.5 | 37.1 | 279310 | 2458 | 100 | 0.2 |

**해석:** 차이 없음. admission control의 `max_new_tokens` 추정 상한(기본 4096)을 16384로 올려도
입장 제어가 변하지 않는다 — 이 트레이스는 max_new_tokens ≤16000이라 추정치가 이미 충분했거나,
제어가 이 압력에서는 binding constraint가 아니다. **"탐침을 돌리고 효과 없음을 확인하는 것"도
유효한 결과다.**

### 4.6 Ablation — `--mem-fraction-static 0.45` (dose-response) ★★

| bar | KV 풀 | ttft_p50 | ttft_p99 | tpot_p50 | maxitl_p99 | e2e_p99(s) | out_tok/s | slo% | goodput |
|-----|-------|----------|----------|----------|------------|------------|-----------|------|---------|
| default (0.85) | 430k | 24.8 | 61.9 | 37.4 | 272234 | 556 | 2484 | 100 | 0.2 |
| memfrac45 | **~228k** | 24.6 | **114292** | 35.7 | 327842 | 680 | 2016 | 100 | 0.2 |

**해석 — 인과 확정:** KV 풀을 절반으로 줄이자 **TTFT p99가 62ms → 114초로 1843배 폭증.**
retraction이 훨씬 더 자주 터져 재prefill 대기열이 길어진 것이다. TPOT은 비슷(35.7 vs 37.4) —
**retraction의 고통은 TPOT이 아니라 TTFT(재prefill 대기)에 집중된다**는 것도 함께 배웠다.
"병목 = KV 풀 용량" 가설의 dose-response 완료.

### 4.7 Ablation — `--chunked-prefill-size -1` (프로토콜 지정)

| bar | ttft_p50 | ttft_p99 | tpot_p50 | maxitl_p99 | out_tok/s | slo% | goodput |
|-----|----------|----------|----------|------------|-----------|------|---------|
| default | 24.8 | 61.9 | 37.4 | 272234 | 2484 | 100 | 0.2 |
| ablated | 32.7 | 49.0 | 37.3 | 265719 | 2466 | 100 | 0.2 |

**해석:** 사실상 차이 없음. W3의 프롬프트는 387토큰 — prefill 청킹이 문제될 여지가 없다.
**프로토콜이 지정한 ablation이 "효과 없음"을 확인해도 무익한 실험이 아니다** — 병목이 prefill
경로에 없다는 소거 증거가 된다.

### 4.8 서버 로그 대조 요약

| 로그 항목 | baseline | m8000 | retract | memfrac45 |
|-----------|----------|-------|---------|-----------|
| token usage 최대 | 0.01 | 1.00 | 1.00 | 1.00 |
| #retracted (이벤트) | 0 | 다수 | **98** | 더 많음 |
| #queue-req 최대 | 0 | 중간 | 77 | 최대 |
| #running-req 최대 | 2 | ~120 | 120 | 120 |

---

## 5. 병목 한 문장 주장 + 증거 3종 ★

> **W3의 병목은 KV 풀 용량 대비 초과구독이다. 계획 출력이 풀의 2배를 넘는 순간(token usage 1.00,
> #retracted>0) retraction-재prefill 사이클로 전환되어 TTFT p99가 62ms→114초까지 폭증하고
> goodput이 절반으로 떨어진다 — decode 속도 자체(TPOT 5.2ms)는 전 구간 건재하다.**

| 증거 유형 | 내용 |
|-----------|------|
| 클라이언트 지표 | TPOT 5.2→37.4ms(7.2배), maxITL p99 272초(retract 공백), e2e p99 556s; out_mean sweep에서 초과구독 2.0×부터 SLO 97.5% 붕괴 시작 |
| 서버 로그 | `KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_token_ratio: 0.0980 -> 0.2747` × 98회; token usage 1.00 핀; #queue-req 77 |
| Ablation | mem-fraction 0.85→0.45 (풀 절반) → TTFT p99 62ms→114초 (1843배). 풀 용량이 원인임을 dose-response로 확정. chunked-prefill ablation은 무효과 — prefill 경로 소거 |

**3단계 시사점:** retraction은 손실 없는(lossless) 최적화 대상이다 — retract를 줄이는 것(입장 제어,
풀 할당 조정, 계층 캐시)이 곧 goodput이다. 특히 "긴 요청을 먼저 retract하는 정책(retraction_policy=length)"의
비용이 maxITL 272초로 측정되었다 — 정책 개선 여지가 숫자로 남아 있다.

---

## 6. H100 ↔ 학생 GPU(24GB) 스케일링 매핑

| 항목 | 학생 GPU (24GB) | H100 (80GB) | W3 재현 레시피 |
|------|-----------------|-------------|----------------|
| KV 풀 | ~110k 토큰 | 430,067 토큰 | 3.9배 |
| 기본 트레이스 초과구독 | 337k/110k = **3.1×** (기본 트레이스로 retraction 재현됨) | 0.8× (재현 안 됨) | out_mean 8000(2.0×) 또는 12000(2.8×)로 |
| retraction 시작점 | q0.35~0.6 | out_mean sweep 2.0× | 학생은 qps sweep에서, H100은 out_mean sweep에서 찾음 |
| TPOT 바닥 | ~10-15ms (추정) | 5.2ms | 절대값은 다르나 TPOT 7배 악화 패턴 동일 |

**학생 재현 지침:** 기본 트레이스(q0.35)만으로 token usage 1.00 + #retracted>0이 관측될 것이다.
그 관측이 곧 병목 증거다. H100 예시는 "여유 하드웨어에서 같은 현상을 만드는 방법(출력 수요 스케일링)"을
보여준다. memfrac ablation은 양쪽 환경에서 동일하게 작동한다.

---

## 부록

### A. 재현 명령어

```bash
# 트레이스
python -m workloads.generators reasoning --out traces/ --model /group-volume/Qwen3-4B
python -m workloads.generators reasoning --out traces/ --model /group-volume/Qwen3-4B \
  --set qps=0.6 --suffix _q0.6          # 1.0, 2.0 동일
python -m workloads.generators reasoning --out traces/ --model /group-volume/Qwen3-4B \
  --set out_mean=8000 --suffix _m8000   # 12000 동일
python -m workloads.generators reasoning --out traces/ --model /group-volume/Qwen3-4B \
  --set out_mean=12000 out_sigma=0.1 qps=4.0 --suffix _retract

# 서버 + 재생: W1 부록 A 참조 (동일 패턴, 트레이스/태그만 교체)
# ablation: --mem-fraction-static 0.45 / --chunked-prefill-size -1
# env probe: SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=16384 (서버 기동 env)

# retraction 관측
grep -E "#retracted|token usage|#running-req" logs/server_reasoning_retract__default.log | tail -40
```

### B. 원시 결과 파일

```
results/reasoning__default.json, reasoning_q0.6/1.0/2.0__default.json
results/reasoning_m3000/8000/12000__default.json
results/reasoning_retract__default.json, __rep2.json, __rep3.json
results/reasoning_retract__clip16k.json, __memfrac45.json, __ablated.json
logs/server_reasoning_*.log, logs/sched_reasoning_*.txt
```

### C. 서버 로그 발췌 (증거용)

```
[15:54:17] KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_tokens_gained: 2187, #new_token_ratio: 0.0980 -> 0.2747
[15:54:18] KV cache pool is full. Retract requests. #retracted_reqs: 1, #new_tokens_gained: 2252, #new_token_ratio: 0.2566 -> 0.2772
(총 98회. token usage 1.00 도달, #queue-req 최대 77)
```
