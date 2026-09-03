# W1 (RAG) — Step 1·2 Best Practice 보고서

**작성자:** 조교(베스트 프랙티스 예시) / **워크로드:** rag / **환경:** H100 80GB (n7 클러스터)
**대응 학생 환경:** 24GB GPU (RunPod) — 6절에서 스케일링 매핑 제공

---

## 0. 이 문서에 대하여

학생들이 Step 1(워크로드 분석)과 Step 2(병목 규명)를 수행할 때 참고할 모범 사례다.
**어떤 실험을 왜 했는지(결정 로그)** → **무엇을 쟀는지(실측값)** → **무엇을 결론 내렸는지(한 문장 병목 주장 + 증거 3종)** 순서로 쓰여 있다.
숫자는 전부 실측값이며 원시 파일은 부록 C에 있다.

---

## 1. 실험 환경 (고정)

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA H100 80GB HBM3 (driver 550.x) |
| SGLang | 0.5.18 (sglang-kernel 0.4.6.post1+cu129) |
| 모델 | Qwen3-4B (로컬 경로, bf16) |
| 서버 플래그 | `--context-length 32768 --mem-fraction-static 0.85 --random-seed 42 --log-level info --enable-metrics` |
| KV 풀 | **430,067 토큰** (K 29.53GB + V 29.53GB, bf16) |
| 측정 프로토콜 | bar마다 서버 재시작, 오픈 루프, 보고 bar는 3회 반복의 중앙값 (03_실험프로토콜.md 규칙 2·5) |

> 서버 로그 발췌: `KV cache pool ... #tokens: 430067` / `max_total_num_tokens=430067, chunked_prefill_size=8192`

---

## 2. Step 1 — 워크로드 분석

### 2.1 분석 표 (전부 실측/계산값)

| 항목 | 값 | 어떻게 구했나 |
|------|-----|--------------|
| 입력 토큰 (p50/p90/p99) | 7297 / 7297 / 7297 | 재생 결과 `prompt_tokens` (200건 전부 동일 — 6청크×1200토큰 고정) |
| 출력 토큰 (p50/p90/p99/max) | 128 / 211 / 290 / 400 | 재생 결과 `completion_tokens` (트레이스 고정값) |
| 입력:출력 비율 | **57 : 1** | 7297 / 128 — prefill 계산량이 압도적 |
| 공유 프리픽스 구조 | 청크 등장 1200회 / 고유 청크 240개 → **청크 수준 재사용률 80%**. 그러나 요청마다 청크 순서가 셔플 → 접두사 매칭은 첫 청크에서 끊김. 시스템 프롬프트 ~32토큰 전 요청 공유. 같은 청크로 시작하는 요청 최대 32건 | `tags.chunk_ids` 교집합 계산 (step1_stats.py) |
| 도착 패턴 | Poisson, QPS 0.6 (200건, 도착 구간 359.6s) | 트레이스 `__meta__` |
| 세션 구조 | 독립 요청 200건 (연쇄 0건) | 트레이스 `depends_on` |
| SLO | TTFT 4000ms (트레이스 기본) | RAG 검색 API 시나리오 — 사용자 대화형 대기 기준 |
| 최대 동시 KV 점유 추정 | ~30k 토큰 (동시 4건 × 7.3k) — **풀 430k의 7%** | 실측 최대 동시 4건 × 7297 |

### 2.2 "무엇이 어려운가" (핵심 문단)

> 요청당 입력 7297토큰, 출력 128토큰으로 prefill:decode 계산량 비가 57:1이다. 청크 수준 재사용률이
> 80%임에도(1200회 등장 / 240개 고유), 검색 청크의 **순서가 요청마다 셔플**되어 radix cache의
> 접두사 매칭이 첫 청크(1200토큰)에서 끊긴다. 즉 "재사용 가능한 토큰은 많은데 접두사 형태가
> 아니라서 쓸 수 없다"는 것이 이 워크로드의 구조적 난점이다. 메모리 압박은 없다(점유 7%).

### 2.3 Step 1 완료 조건 확인

- [x] 표 전부 실측값으로 채움
- [x] SLO 근거 명시 (TTFT 4000ms)
- [x] 코드 수정 0건

---

## 3. Step 2 — 실험 설계 (결정 로그)

각 bar를 **왜** 돌렸는지. 이 표가 보고서의 뼈대다.

| # | 실험 | 설계 이유 |
|---|------|-----------|
| 1 | baseline (rag, q0.6) ×3회 | 기준점. 3회 반복으로 측정 노이즈 확인(프로토콜 규칙 5) |
| 2 | reuse_alpha=2.0 probe | 청크 인기도를 극단적으로 치우치게 하면(유사 중복 요청) 재사용률 93%까지 올라간다. **그래도 cache_hit가 낮게 유지되면** 원인이 청크 '수'가 아니라 청크 '순서'라는 인과 확인 |
| 3 | qps sweep 0.6→1.2→2.4→4.8 | goodput이 꺾이는 지점(knee) 탐색. H100은 여유가 커서 학생 GPU보다 높은 qps까지 늘림 |
| 4 | chunks_per_req 3 / 12 | 프롬프트 길이(=prefill 비용) 축을 3685 / 14524토큰으로 분리 — TTFT가 길이에 비례하는지, cache_hit가 청크 수에 어떻게 의존하는지 확인 |
| 5 | ablation `--disable-radix-cache` | "radix cache가 이 워크로드에서 거의 아무것도 주지 않는다"는 가설의 인과 확인. 끄고도 거의 안 나빠지면(→ 나아지면) 가설 성립 |

---

## 4. Step 2 — 결과

### 4.1 Baseline (3회 반복)

| bar | ttft_p50 | ttft_p99 | tpot_p50 | maxitl_p99 | out_tok/s | cache_hit% | slo% | goodput |
|-----|----------|----------|----------|------------|-----------|------------|------|---------|
| default (1회차) | 134.8 | 209.3 | 4.6 | 126.8 | 77.4 | 11.3 | 100.0 | 0.6 |
| rep2 | 135.8 | 212.3 | 4.7 | 127.0 | 77.4 | 11.3 | 100.0 | 0.6 |
| rep3 | 137.2 | 217.0 | 4.7 | 127.1 | 77.4 | 11.3 | 100.0 | 0.6 |

3회 spread가 TTFT p50 기준 1.8% — 측정이 안정적이므로 이후 bar는 1회 + 필요시 반복.

**읽는 법:** TTFT p50 134.8ms는 7297토큰 prefill 계산(≈54k tok/s)이 전부. TPOT 4.6ms는 저배치 decode의 바닥값.
cache_hit 11.3% = 요청당 평균 825토큰 적중 — **시스템 프롬프트(32) + 첫 청크(1200)가 간신히 걸치는 수준**.

### 4.2 재사용 탐침(reuse_alpha=2.0) — 인과 실험 ★

| bar | 청크 재사용률(트레이스) | cache_hit%(실측) | ttft_p50 | 비고 |
|-----|------------------------|------------------|----------|------|
| default (alpha=1.0) | 80.0% (고유 240개) | **11.3%** | 134.8 | |
| reuse2 (alpha=2.0) | 93.0% (고유 84개) | **21.8%** | 118.5 | 유사 중복 요청 |

**해석:** 청크 재사용률을 80→93%로 올려도(고유 청크 1/3로 감소) 실제 접두사 적중은 11→22%에 그친다.
적중분의 중앙값 1243토큰 = 시스템 프롬프트 + 청크 1개. **재사용 가능성이 아무리 높아도 접두사(앞부분)가
같아야 매칭이 이어지는데, 셔플 때문에 두 번째 청크부터 항상 어긋난다.** 이것이 "위치 의존적 KV 재사용 실패"의 직접 증거다.

### 4.3 부하 스윕 (knee 탐색)

| qps | ttft_p50 | ttft_p99 | tpot_p50 | maxitl_p99 | out_tok/s | slo% | goodput |
|-----|----------|----------|----------|------------|-----------|------|---------|
| 0.6 | 134.8 | 209.3 | 4.6 | 126.8 | 77.4 | 100 | 0.6 |
| 1.2 | 137.9 | 238.6 | 5.5 | 143.9 | 154.5 | 100 | 1.1 |
| 2.4 | 142.2 | 295.5 | 7.5 | 348.3 | 307.2 | 100 | 2.2 |
| 4.8 | 168.2 | 512.6 | 21.0 | 904.7 | 603.2 | 100 | 4.4 |

**knee 판정:** H100에서는 qps 4.8까지 SLO 위반이 0건 — goodput이 qps에 거의 선형(0.6→4.4, 7.3배)으로
따라간다. **무릎은 4.8보다 위에 있다.** TTFT p99가 209→513ms로 2.4배 늘었지만 SLO 4000ms에 8배 여유.
TPOT p50 4.6→21.0ms (배치 커짐)가 가장 먼저 반응하는 지표다.

### 4.4 프롬프트 길이 축 (chunks_per_req 3 / 12)

| bar | 프롬프트 | cache_hit% | cached p50(토큰) | ttft_p50 | ttft_p99 |
|-----|----------|------------|------------------|----------|----------|
| c3 | 3685 | 24.7 | 1242 | 53.4 | 70.9 |
| 기본(6청크) | 7297 | 11.3 | 1241 | 134.8 | 209.3 |
| c12 | 14524 | 4.2 | **39** | 320.3 | 611.1 |

**해석 (두 효과가 분리됨):**
1. **TTFT는 프롬프트 길이에 정확히 선형** — 3685→7297→14524토큰에서 TTFT p50 53→135→320ms.
   큐 대기가 아니라 순수 prefill 계산 비용이다 (analyze 리포트 [4]와 일치).
2. **cache_hit는 청크 수에 반비례** — 청크가 많을수록(12개) 첫 청크가 같을 확률이 낮아져 적중이
   시스템 프롬프트(39토큰)만 남는다. c3는 프롬프트 자체가 짧아 첫 청크 비중이 커서 24.7%.

### 4.5 Ablation — `--disable-radix-cache` ★

| bar | ttft_p50 | ttft_p99 | tpot_p50 | cache_hit% | slo% | goodput |
|-----|----------|----------|----------|------------|------|---------|
| default | 134.8 | 209.3 | 4.6 | 11.3 | 100 | 0.6 |
| ablated | 137.6 | 237.8 | 4.8 | **0.0** | 100 | 0.6 |

**해석:** radix cache를 꺼도 TTFT p50이 2%만 나빠진다. **radix cache가 이 워크로드에 주는 이득이
사실상 0이라는 인과 확인.** 11.3%의 적중(시스템 프롬프트+첫청크)이 주는 이득이 2%뿐이다.
→ "접두사 캐시를 더 잘 쓰게" 하는 방향의 최적화는 현재 구조(셔플)에서는 뽑아낼 것이 없고,
**재사용을 접두사가 아닌 형태로 만드는 것**(예: 청크 순서 정규화, 위치 독립 캐시)이 3단계 과제가 된다.

### 4.6 서버 로그 대조 (클라이언트 지표 ↔ 서버)

| 로그 항목 | 관측값 | 해석 |
|-----------|--------|------|
| token usage (최대) | 0.07 | 메모리 압박 없음 — 병목은 메모리가 아님 |
| #retracted | 0 | 선점 없음 |
| #queue-req | 0 | 큐 적체 없음 (q4.8에서도) |
| prefill 입력 처리율 | ~54k tok/s (7297토큰 / 135ms) | TTFT의 정체는 prefill 계산 |
| 서버측 cached 비율 | 11.1% (new 1,316,871 / cached 164,377) | 클라이언트 cache_hit 11.3%와 일치 — 상호 검증 완료 |

### 4.7 analyze 리포트 [1]의 "DECODE 지배" 판정에 대한 주석 (교육적 주의)

`bench.analyze`의 E2E 분해는 TTFT/E2E 비율 16.4%로 "DECODE 지배" 판정을 내린다. 이는
**E2E 시간의 비율**이 맞다(저부하 배치 1~4에서 decode가 토큰당 비싸다). 그러나 병목 논의는 다르다:
- TPOT 4.6ms는 이미 바닥이며 SLO 여유가 크다. 최적화 레버가 아니다.
- TTFT 135ms는 **prefill 계산 그 자체**이고, 입력:출력 57:1에서 서버 총 연산량의 대부분이 prefill이다.
- 4.4절: TTFT는 길이에 선형 → prefill 비용. 4.2절: 재사용 가능한데 못 쓰는 토큰이 89%.
→ "어느 단계가 E2E를 오래 쓰는가"와 "어디에 낭비가 있어 최적화 레버가 되는가"는 다른 질문이다.

---

## 5. 병목 한 문장 주장 + 증거 3종 ★

> **W1의 병목은 메모리도 decode도 아닌 "도달 불가능한 접두사 재사용"이다. 청크 수준 재사용률이
> 80%인데도 접두사 적중은 11.3%뿐이고(요청마다 청크가 셔플되어 매칭이 첫 청크에서 끊김),
> 요청당 ~6.5k 토큰이 매번 새로 prefill된다 — TTFT p50 135ms의 정체다.**

| 증거 유형 | 내용 |
|-----------|------|
| 클라이언트 지표 | cache_hit 11.3% (재사용률 80% 대비); TTFT p50 135ms ≈ 7297토큰 prefill 계산; TPOT 4.6ms·SLO 100% (decode·메모리 무고장) |
| 서버 로그 | token usage ≤0.07, #retracted 0, #queue-req 0 — 메모리·스케줄러 무결. prefill 처리율 ~54k tok/s. 서버측 cached 비율 11.1% (클라이언트와 일치) |
| Ablation | radix cache OFF → TTFT +2%만 (radix가 주는 것이 없음 = 적중이 구조적으로 막혀 있음). reuse_alpha=2.0 → 재사용률 93%에도 적중 21.8% (순서가 원인임을 분리) |

**3단계 시사점:** 접두사 기반 캐시로는 뽑을 수 없다. 청크 순서를 정규화하거나(워크로드 변경 —
실서비스에서는 불가능할 수 있음), 위치 독립적 KV 재사용 구조가 필요하다. 이 격차(80% vs 11%)가
프로젝트의 손실 이득 상한선이다.

---

## 6. H100 ↔ 학생 GPU(24GB) 스케일링 매핑

| 항목 | 학생 GPU (24GB) | H100 (80GB) | W1에 미치는 영향 |
|------|-----------------|-------------|------------------|
| KV 풀 | ~110k 토큰 | 430,067 토큰 | **없음** — W1 최대 점유 ~30k로 어느 쪽도 메모리 압박 아님. 트레이스 스케일링 불필요 (원본 트레이스 그대로 사용) |
| prefill 처리율 | 낮음 (GPU 세대 의존) | ~54k tok/s | TTFT 절대값은 학생 쪽이 큼. **TTFT는 길이에 선형**이믧 현상 동일 |
| qps knee | 더 낮은 qps에서 SLO 위반 시작 | 4.8까지 무릎 없음 | 학생은 0.6~2.4 사이에서 무릎을 찾게 됨. H100 예시는 "무릎이 안 보인다"도 결과임을 보여줌 |

**학생 재현 지침:** 트레이스 변경 없이 그대로 재생. qps 스윕은 0.6/1.2/2.4에서 SLO가 깨지기 시작하는지
관찰. cache_hit 11%와 "cached p50 = 시스템 프롬프트+첫청크" 패턴은 동일하게 재현될 것.

---

## 부록

### A. 재현 명령어 (pod 내, /group-volume/jeongho/study/project 기준)

```bash
# 트레이스 (결정적 — 학생과 바이트 동일)
python -m workloads.generators rag --out traces/ --model /group-volume/Qwen3-4B
python -m workloads.generators rag --out traces/ --model /group-volume/Qwen3-4B \
  --set reuse_alpha=2.0 --suffix _reuse2
python -m workloads.generators rag --out traces/ --model /group-volume/Qwen3-4B \
  --set qps=1.2 --suffix _q1.2   # 2.4, 4.8 동일
python -m workloads.generators rag --out traces/ --model /group-volume/Qwen3-4B \
  --set chunks_per_req=3 --suffix _c3   # 12 동일

# 서버 + 재생 (bar마다 재시작)
python -m sglang.launch_server --model-path /group-volume/Qwen3-4B --port 30000 \
  --context-length 32768 --mem-fraction-static 0.85 --random-seed 42 \
  --log-level info --enable-metrics > logs/server_rag__default.log 2>&1 &
python -m bench.replay --trace traces/rag.jsonl --url http://127.0.0.1:30000 \
  --tag default --out results/ --keep-output

# ablation: 위 서버 플래그에 --disable-radix-cache 추가, --tag ablated

# 지표/분석
python -m bench.metrics results/rag__*.json
python -m bench.analyze results/rag__default.json
```

### B. 원시 결과 파일

```
results/rag__default.json, rag__rep2.json, rag__rep3.json     # baseline ×3
results/rag_reuse2__default.json                              # 재사용 탐침
results/rag_q1.2/2.4/4.8__default.json                         # 부하 스윕
results/rag_c3__default.json, rag_c12__default.json            # 길이 축
results/rag__ablated.json                                      # ablation
logs/server_rag__*.log, logs/sched_rag__*.txt                 # 서버 로그/발췌
```

### C. 서버 로그 발췌 (증거용)

```
KV Cache is allocated. dtype: torch.bfloat16, #tokens: 430067, K size: 29.53 GB, V size: 29.53 GB
max_total_num_tokens=430067, chunked_prefill_size=8192, max_prefill_tokens=16384
# baseline 전체 로그에서 retraction/queue 이벤트 0건, token usage 최대 0.07
```
