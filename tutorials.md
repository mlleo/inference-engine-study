# SGLang 추론엔진 실습 가이드
> RadixAttention · Continuous Batching 실측 실습 (1~2시간 / RunPod / 저토큰 예산)

## 공통 준비 (약 10분)

### 서버 실행

토큰 예산이 적으므로 1.5B급 소형 모델을 사용합니다. 실험 목적(캐시/배칭 관찰)에는 모델 크기가 전혀 중요하지 않습니다.

```bash
pip install "sglang[all]"

python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --port 30000 \
  --log-level info
```

서버 로그 창은 실습 내내 켜두세요. **프리필 배치마다 찍히는 로그가 오늘 실습의 핵심 관찰 대상**입니다:

```
Prefill batch. #new-seq: 1, #new-token: 512, #cached-token: 0, token usage: 0.00, ...
```

- `#new-token`: 이번에 실제로 프리필 연산을 수행한 토큰 수
- `#cached-token`: **radix tree에서 매칭되어 연산을 건너뛴 토큰 수**

### 유용한 엔드포인트 3개

```bash
# 캐시 완전 초기화 (대조 실험용)
curl -X POST http://localhost:30000/flush_cache

# 서버 상태 조회
curl http://localhost:30000/get_server_info

# 네이티브 generate — 응답의 meta_info에 캐시 히트 수치가 들어있음
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 8}}'
```

`/generate` 응답의 `meta_info`에 `prompt_tokens`, `completion_tokens`, `cached_tokens` 필드가 있습니다. 로그를 눈으로 좇지 않아도 **요청 단위로 캐시 히트를 프로그래밍적으로 수집**할 수 있어서 이걸 기본 계측 수단으로 씁니다.

> 참고: SGLang은 버전에 따라 플래그/필드명이 조금씩 바뀝니다. 안 맞으면 `python -m sglang.launch_server --help`와 응답 JSON을 직접 확인하세요.

### 공통 계측 헬퍼

모든 실습에서 재사용할 스크립트입니다. `lab_common.py`로 저장하세요.

```python
# lab_common.py
import requests, time

URL = "http://localhost:30000"

def gen(text, max_new_tokens=16):
    """요청을 보내고 (지연시간, cached_tokens, prompt_tokens)를 반환"""
    t0 = time.perf_counter()
    r = requests.post(f"{URL}/generate", json={
        "text": text,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
    })
    latency = time.perf_counter() - t0
    meta = r.json()["meta_info"]
    return {
        "latency": latency,
        "cached": meta.get("cached_tokens", 0),
        "prompt": meta.get("prompt_tokens", 0),
    }

def flush():
    requests.post(f"{URL}/flush_cache")
```

`max_new_tokens=16`, `temperature=0`으로 고정하는 이유: 생성 토큰을 최소화해 **토큰 예산을 아끼고**, 프리필(캐시) 효과만 깔끔하게 분리해서 보기 위함입니다.

---

## 실습 1 — RadixAttention 캐시 히트 관찰 (약 30분) ⭐

**목표**: 같은 prefix를 공유하는 요청에서 캐시 히트가 실제로 발생하는지, TTFT/지연시간이 얼마나 줄어드는지 수치로 확인한다.

### 1-1. 실험 설계

긴 시스템 프롬프트(공유 prefix) + 서로 다른 유저 질문 구조로 요청을 보냅니다.

```python
# lab1_cache_hit.py
from lab_common import gen, flush

# 공유 prefix를 일부러 길게 (~500 토큰). 내용은 아무거나 OK.
SYSTEM = (
    "You are a world-class culinary expert with 30 years of experience "
    "in French, Italian, Japanese, and Korean cuisine. "
) * 20  # 반복으로 길이 확보

def ask(question):
    prompt = SYSTEM + "\n\nUser: " + question + "\nAssistant:"
    return gen(prompt)

flush()  # 깨끗한 상태에서 시작

r1 = ask("How do I make carbonara?")
r2 = ask("How do I make kimchi stew?")
r3 = ask("How do I make sushi rice?")

for i, r in enumerate([r1, r2, r3], 1):
    print(f"req{i}: latency={r['latency']*1000:.0f}ms  "
          f"prompt={r['prompt']}  cached={r['cached']}  "
          f"hit_rate={r['cached']/r['prompt']:.1%}")
```

### 1-2. 예상 결과와 해석

| 요청 | cached_tokens | 해석 |
|---|---|---|
| req1 | 0 (또는 극소수) | 콜드 스타트 — tree가 비어 있음 |
| req2 | ≈ SYSTEM 길이 | req1이 삽입해둔 SYSTEM prefix에 히트 |
| req3 | ≈ SYSTEM 길이 | 동일 |

req2, req3의 latency가 req1 대비 눈에 띄게 짧아야 합니다(프리필 연산량이 시스템 프롬프트만큼 줄었으므로). 동시에 서버 로그에서 `#cached-token`이 같은 수치로 찍히는 것을 교차 확인하세요.

### 1-3. 대조 실험 (반드시 할 것)

```python
flush()          # 캐시 비우기
r4 = ask("How do I make carbonara?")   # req1과 완전히 같은 요청
print(r4)        # cached=0으로 돌아감 → 빨랐던 이유가 정말 캐시였음을 증명
```

### 1-4. 토론 포인트

- `cached_tokens`가 SYSTEM 길이와 정확히 같지 않고 조금 다를 수 있습니다. 왜일까요? (힌트: radix tree의 노드는 토큰 단위지만, 매칭은 token id 시퀀스 기준 → 토크나이저가 경계에서 SYSTEM 끝 + User 시작을 다르게 병합할 수 있음. 또한 페이지/블록 단위 정렬의 영향)
- 오늘 코드에서 본 `match_prefix()`가 정확히 이 순간 호출됩니다. req2가 들어왔을 때 tree 안에서 무슨 일이 일어났는지 말로 설명해보기.

---

## 실습 2 — Radix Tree 분기(branch) 추적 및 시각화 (약 20분)

**목표**: 요청 순서를 설계해서 tree가 언제 분기하고, 어디까지 재사용되는지 요청별 `cached_tokens`로 역추적한다.

### 2-1. 실험 설계

prefix 공유 구조가 서로 다른 4개의 요청을 순서대로 보냅니다.

```python
# lab2_tree_branch.py
from lab_common import gen, flush

P = "You are a helpful chef assistant. Answer briefly.\n"  # 공통 루트
A = P + "User: Give me a pasta recipe.\nAssistant:"
B = P + "User: Give me a pizza recipe.\nAssistant:"

flush()

r1 = gen(A)                      # tree: 루트 생성
r2 = gen(B)                      # P까지만 히트 → 분기 발생 지점 확인
r3 = gen(A)                      # A 경로 전체 + r1의 "생성 결과"까지 히트?
# 멀티턴: r1의 출력을 이어붙여 후속 질문
followup = A + " (some answer...)" + "\nUser: Make it vegetarian.\nAssistant:"
r4 = gen(followup)               # A 경로를 얼마나 연장 재사용하는지

for name, r in zip("r1 r2 r3 r4".split(), [r1, r2, r3, r4]):
    print(f"{name}: prompt={r['prompt']}  cached={r['cached']}")
```

### 2-2. 관찰 포인트 & 손그림 과제

각 요청의 `cached` 값으로 tree 모양을 **화이트보드에 직접 그려보세요**:

```
                [P: 공통 시스템 프롬프트]
                /                    \
   [User: pasta... + 생성결과]   [User: pizza... + 생성결과]
          |
   [User: vegetarian...]   ← r4가 연장
```

- r2의 `cached` ≈ P의 토큰 수 → **P 끝에서 노드가 split되며 분기**했다는 증거
- r3의 `cached`가 r1의 `prompt`와 거의 같음 → 프롬프트뿐 아니라 **디코드된 KV도 tree에 남는다**는 것 확인 (RadixAttention의 중요한 특징: 생성이 끝난 시퀀스의 KV cache도 즉시 버리지 않고 tree에 유지)
- r4에서 followup에 넣은 "(some answer...)" 부분이 r1의 실제 생성 결과와 다르면 그 지점부터 캐시 미스가 납니다. **r1의 실제 출력 텍스트를 그대로 이어붙이면** cached가 훨씬 커지는 것도 실험해보세요 — 이것이 멀티턴 챗봇에서 RadixAttention이 강력한 이유입니다.

### 2-3. 토론 포인트

- 오늘 본 코드의 insert / split 로직과 대응시키기: r2가 도착했을 때 `_split_node()`가 호출되는 위치는?
- 메모리가 꽉 차면 어떤 노드부터 evict될까? (LRU + leaf부터) → `--mem-fraction-static`을 낮게 줘서 eviction 로그를 유도해보는 것도 심화 실험으로 가능

---

## 실습 3 — Continuous Batching 체감 (약 30분)

**목표**: 동시 요청 수를 늘려도 총 시간이 비례해서 늘지 않음(= 배칭으로 흡수됨)을 측정하고, throughput vs latency 트레이드오프 곡선을 그린다.

### 3-1. 방법 A: 직접 측정 (개념 이해에 좋음)

캐시 효과가 섞이지 않도록 **요청마다 prefix를 다르게** 만듭니다.

```python
# lab3_batching.py
import asyncio, aiohttp, time, random, string

URL = "http://localhost:30000/generate"

def unique_prompt(i):
    # 캐시 히트 방지용 랜덤 prefix
    salt = "".join(random.choices(string.ascii_letters, k=32))
    return f"[{salt}] Write a one-line fun fact about the number {i}."

async def one(session, i):
    t0 = time.perf_counter()
    async with session.post(URL, json={
        "text": unique_prompt(i),
        "sampling_params": {"max_new_tokens": 32, "temperature": 0},
    }) as resp:
        await resp.json()
    return time.perf_counter() - t0

async def run(concurrency, n=32):
    sem = asyncio.Semaphore(concurrency)
    async def guarded(session, i):
        async with sem:
            return await one(session, i)
    t0 = time.perf_counter()
    async with aiohttp.ClientSession() as s:
        lats = await asyncio.gather(*[guarded(s, i) for i in range(n)])
    total = time.perf_counter() - t0
    print(f"concurrency={concurrency:>2}  total={total:6.2f}s  "
          f"throughput={n/total:5.2f} req/s  "
          f"avg_latency={sum(lats)/len(lats)*1000:6.0f}ms")

for c in [1, 4, 16, 32]:
    asyncio.run(run(c))
```

### 3-2. 예상 결과와 해석

| concurrency | total | throughput | avg latency |
|---|---|---|---|
| 1 | 기준 | 기준 | 최소 |
| 4 | 기준의 ~1.n배 (4배 아님!) | 크게 증가 | 소폭 증가 |
| 16 | 완만히 증가 | 계속 증가 | 증가 |
| 32 | — | 증가폭 둔화 (포화) | 뚜렷이 증가 |

핵심 해석 3가지:

1. **static batching이었다면** concurrency=1 대비 32일 때 총 시간이 훨씬 길어야 하지만, continuous batching은 매 스텝(iteration)마다 끝난 요청을 빼고 새 요청을 끼워 넣으므로 GPU가 놀지 않음 → 총 시간이 거의 안 늘어남
2. throughput은 올라가지만 개별 latency는 나빠짐 → **서빙 시스템의 근본 트레이드오프**를 숫자로 확인
3. 어느 지점부터 throughput 증가가 멈추는가 = GPU 연산/메모리 포화점. 실행 중 서버 로그의 `#running-req`, `token usage` 수치가 올라가는 것을 같이 관찰하세요.

### 3-3. 방법 B: 공식 벤치마크 도구 (시간 절약용)

```bash
python -m sglang.bench_serving --backend sglang \
  --dataset-name random \
  --num-prompts 64 \
  --random-input-len 256 --random-output-len 32 \
  --request-rate 8
```

`--request-rate`를 1 → 8 → 32로 바꿔가며 리포트의 throughput / mean TTFT / mean latency 비교. output-len을 32로 짧게 잡으면 토큰 소모가 적습니다.

---

## 실습 4 — Cache-aware 스케줄링: LPM vs FCFS (시간 남으면, 약 20분)

**목표**: SGLang의 기본 스케줄 정책인 LPM(Longest Prefix Match — 캐시에 이미 있는 prefix가 긴 요청부터 처리)이 FCFS 대비 캐시 히트율을 얼마나 올리는지 재현한다. RadixAttention 논문의 핵심 주장 중 하나입니다.

### 4-1. 실험 설계

서로 다른 시스템 프롬프트를 가진 3개 그룹의 요청을 **일부러 섞어서(interleave)** 한꺼번에 던집니다.

```python
# lab4_scheduling.py — 실습 3의 asyncio 구조 재사용
SYSTEMS = [("You are a chef. " * 30), ("You are a lawyer. " * 30), ("You are a poet. " * 30)]
prompts = []
for i in range(10):
    for s in SYSTEMS:                      # A,B,C,A,B,C,... 순으로 섞임
        prompts.append(s + f"\nUser: question {i}\nAssistant:")
# 이 30개를 concurrency 30으로 동시에 투척 → 총 cached_tokens 합계 집계
```

### 4-2. 비교 절차

1. 기본 설정(LPM)으로 실행 → 각 응답의 `cached_tokens` 합계 기록 → `/flush_cache`
2. 서버를 `--schedule-policy fcfs`로 재시작 후 동일 실행 → 합계 기록
3. 두 정책의 총 캐시 히트 토큰 수 / 총 소요 시간 비교

### 4-3. 예상 결과와 해석

대기 큐가 충분히 쌓이는 상황에서 LPM은 같은 prefix 그룹을 **모아서** 처리하는 효과를 내므로 캐시 히트율이 높게 나옵니다. FCFS는 도착 순서(A,B,C,A,B,C...)대로 처리하다가 메모리 압박으로 다른 그룹의 캐시가 evict되면 히트율이 떨어집니다.

> 주의: GPU 메모리가 넉넉하고 요청 수가 적으면 eviction이 안 일어나 두 정책의 차이가 안 보일 수 있습니다. 차이가 안 보이면 (1) 요청 수를 늘리거나 (2) `--mem-fraction-static`을 낮춰 캐시 공간을 좁혀서 eviction을 유도하세요. "차이가 안 나는 조건"을 찾는 것 자체도 좋은 학습입니다.

---

## 시간 배분 요약

| 순서 | 실습 | 시간 | 우선순위 |
|---|---|---|---|
| 0 | 서버 셋업 + 헬퍼 | 10분 | 필수 |
| 1 | 캐시 히트 관찰 | 30분 | ⭐ 필수 |
| 2 | Tree 분기 추적 | 20분 | ⭐ 필수 |
| 3 | Continuous batching 측정 | 30분 | 권장 |
| 4 | LPM vs FCFS | 20분 | 선택 |

## 토큰 절약 팁

- `max_new_tokens`는 8~32로 고정 — 캐시/배칭 실험은 프리필 관찰이 핵심이라 생성은 짧아도 됨
- `temperature=0` — 재현 가능한 결과 + 멀티턴 실험(실습 2-2)에서 출력 재사용 용이
- 실습 1~2는 요청 수가 총 10개 안팎이라 토큰 소모 미미. 토큰 대부분은 실습 3~4에서 쓰이므로 예산이 빠듯하면 3-1의 `n`과 `max_new_tokens`를 줄일 것
