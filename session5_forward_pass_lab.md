# SGLang 추론엔진 실습 가이드 — 5회차
> ForwardBatch · Attention Backend · CUDA Graph 실측 실습 (1~2시간 / RunPod / 저예산)

지난 4회차까지는 **"무엇을 계산할지 결정하는"** 부분(스케줄러, radix tree, HiCache)을 봤습니다.
오늘은 그 결정이 **실제 GPU 텐서로 변환되어 커널이 도는** 구간을 직접 열어봅니다.

---

## 공통 준비 (약 15분)

### 0-1. 소스 체크아웃 + 커밋 고정 ⚠️ 필수

오늘부터는 서버를 띄우는 것으로 끝나지 않고 **소스를 직접 읽고 수정**합니다. `srt/` 디렉토리는 리팩토링이 잦으므로 스터디 전원이 **같은 커밋**을 봐야 합니다.

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git log -1 --format=%H          # 이 해시를 스터디 채널에 공지
git checkout -b study5          # 실습 중 코드 수정용 브랜치

pip install -e "python[all]"    # 소스 수정이 바로 반영되도록 editable 설치
```

> 이미 `pip install sglang[all]`로 설치했다면 실습 2·3에서 코드를 고쳐도 반영이 안 됩니다. 반드시 editable 설치로 바꾸거나, `pip show -f sglang`으로 실제 설치 경로를 찾아 그쪽을 수정하세요.

### 0-2. 오늘의 주력 도구: `bench_one_batch`

지금까지는 `launch_server` + HTTP로 실습했습니다. 오늘은 다릅니다.

| | `launch_server` | `bench_one_batch` |
|---|---|---|
| 프로세스 | Tokenizer / Scheduler / Detokenizer 분리 | **단일 프로세스** (TP=1일 때) |
| ForwardBatch 생성 위치 | 자식 프로세스 | 내 프로세스 |
| 디버거 / monkey patch | 매우 번거로움 | **바로 됨** |
| HTTP·스케줄링 노이즈 | 있음 | 없음 |

`bench_one_batch`는 `ModelRunner`와 `ForwardBatch`를 **직접** 만들어 forward만 돌립니다. 오늘 배울 경로를 최소 재현하는 데 이보다 좋은 진입점이 없습니다.

```bash
python -m sglang.bench_one_batch \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --batch-size 4 \
  --input-len 256 \
  --output-len 8
```

> 플래그 이름은 버전마다 다릅니다. 안 맞으면 `python -m sglang.bench_one_batch --help`를 먼저 확인하세요. `--batch-size`는 여러 값을 한 번에 받을 수 있는 버전도 있습니다.

출력에서 오늘 볼 숫자는 세 개입니다.

- **Prefill latency / throughput** — EXTEND 한 번의 비용
- **Decode median latency** — DECODE 한 스텝의 비용
- **Total** — 전체

프리필 256토큰(4배치 = 1024토큰)과 디코드 1스텝(4토큰)의 **토큰당 비용 차이**를 먼저 계산해보세요. 이게 이론 자료 §3.1에서 말한 EXTEND/DECODE 비대칭입니다.

### 0-3. 공통 계측 헬퍼

`lab5_dump.py`로 저장하세요. `ForwardBatch.init_new`를 가로채서 텐서를 찍어보는 스크립트입니다. **소스 수정 없이** 동작합니다.

```python
# lab5_dump.py
import runpy, sys, torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_orig = ForwardBatch.init_new.__func__     # classmethod 언랩
_seen = {}
MAX_PRINT_PER_MODE = 2                     # 모드별로 처음 2번만 출력

def _fmt(x):
    if x is None:
        return "None"
    if isinstance(x, torch.Tensor):
        head = x.flatten()[:8].tolist() if x.numel() else []
        return f"shape={tuple(x.shape):<12} dtype={str(x.dtype):<13} head={head}"
    return repr(x)[:70]

FIELDS = [
    "input_ids", "positions", "req_pool_indices", "seq_lens",
    "extend_prefix_lens", "extend_seq_lens", "out_cache_loc",
]

def _patched(cls, batch, model_runner):
    fb = _orig(cls, batch, model_runner)
    mode = str(fb.forward_mode)
    n = _seen.get(mode, 0) + 1
    _seen[mode] = n
    if n <= MAX_PRINT_PER_MODE:
        print("\n" + "=" * 78)
        print(f"[ForwardBatch #{n}] mode={mode}  batch_size={fb.batch_size}")
        print("=" * 78)
        for f in FIELDS:
            print(f"  {f:<20}{_fmt(getattr(fb, f, None))}")

        # --- req_to_token 페이지 테이블 엿보기 (실습 3에서 사용) ---
        try:
            pool = fb.req_to_token_pool.req_to_token
            slot = int(fb.req_pool_indices[0])
            n_tok = int(fb.seq_lens[0])
            row = pool[slot, :n_tok]
            print(f"  {'req_to_token[0]':<20}len={n_tok} "
                  f"first8={row[:8].tolist()} last8={row[-8:].tolist()}")
        except Exception as e:
            print(f"  (req_to_token 조회 실패: {e})")
    return fb

ForwardBatch.init_new = classmethod(_patched)

# 이 스크립트에 넘긴 인자를 그대로 bench_one_batch로 전달
sys.argv = ["sglang.bench_one_batch"] + sys.argv[1:]
runpy.run_module("sglang.bench_one_batch", run_name="__main__")
```

실행:

```bash
python lab5_dump.py \
  --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --batch-size 4 --input-len 256 --output-len 8 \
  --disable-cuda-graph
```

> `--disable-cuda-graph`를 붙이는 이유: CUDA graph가 켜져 있으면 캡처 단계에서 더미 `ForwardBatch`가 잔뜩 만들어져 출력이 지저분해집니다. 실습 4에서 다시 켭니다.

**패치가 안 먹는 경우** — 버전에 따라 `bench_one_batch`가 `ForwardBatch.init_new`를 안 거칠 수 있습니다. 이럴 땐 fallback:

```bash
grep -rn "ForwardBatch.init_new\|ForwardBatch(" python/sglang/ | head
```

로 실제 생성 지점을 찾아 그 함수 안에 직접 `print`를 넣고, 실습 끝나면 `git checkout .`으로 되돌리세요.

---

## 실습 1 — EXTEND vs DECODE 텐서 모양 예측하기 (약 30분) ⭐

**목표**: 같은 배치가 프리필과 디코드에서 완전히 다른 텐서로 표현된다는 것을 눈으로 확인하고, `#cached-token`이 텐서 어디에 들어있는지 찾는다.

### 1-1. 먼저 예측 (실행 전에 반드시!)

화이트보드에 각자 적고 시작하세요. `batch_size=4`, `input_len=256`, 캐시 히트 없음:

| 필드 | EXTEND | DECODE |
|---|---|---|
| `input_ids` | ? | ? |
| `positions` | ? | ? |
| `seq_lens` | ? | ? |
| `extend_seq_lens` | ? | ? |
| `out_cache_loc` | ? | ? |
| `req_pool_indices` | ? | ? |

### 1-2. 정답

```
EXTEND                              DECODE (첫 스텝)
input_ids        (1024,)            (4,)
positions        (1024,)            (4,)
req_pool_indices (4,)               (4,)
seq_lens         (4,)  = [256]*4    (4,)  = [257]*4
extend_prefix_lens (4,) = [0]*4     None (또는 미사용)
extend_seq_lens  (4,)  = [256]*4    None
out_cache_loc    (1024,)            (4,)
```

여기서 나와야 하는 반응 세 가지:

1. **`input_ids`가 2차원이 아니다.** `(4, 256)`이 아니라 `(1024,)`입니다. 패딩이 아예 없고, 시퀀스 경계는 `extend_seq_lens`(→ `cu_seqlens`)로만 표현됩니다. 길이가 10배 차이 나는 요청들을 패딩해서 배치하면 얼마나 낭비인지 계산해보세요.
2. **`seq_lens`는 캐시된 prefix를 포함하고 `extend_seq_lens`는 안 한다.** 이 둘을 헷갈리는 게 attention backend를 짤 때 가장 흔한 버그입니다.
3. **`out_cache_loc`의 길이가 이번 스텝에 KV를 새로 쓰는 토큰 수다.** DECODE는 4개. EXTEND는 1024개.

### 1-3. 캐시 히트를 만들어서 다시 보기

`--input-len`을 두 번 다르게 주는 대신, `bench_one_batch`에 `--batch-size 1`을 주고 두 번 연속 돌리는 것으로는 캐시가 안 걸립니다(매번 새 프로세스). 캐시 히트가 걸린 `ForwardBatch`를 보려면 **서버 쪽**을 봐야 합니다:

```bash
# 터미널 1
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-1.5B-Instruct --port 30000 --log-level info

# 터미널 2 — 2회차 실습의 lab_common.py 재사용
python lab1_cache_hit.py
```

서버 로그의 `#new-token` / `#cached-token`을 보면서 이론 자료의 등식을 확인하세요:

```
#cached-token  ==  sum(extend_prefix_lens)
#new-token     ==  sum(extend_seq_lens)  ==  len(out_cache_loc)
```

**2회차에서 본 `cached_tokens` 숫자가 이 텐서의 어느 필드였는지**를 말로 설명할 수 있으면 이번 실습의 절반은 성공입니다.

### 1-4. 토론 포인트

- `positions`가 `input_ids`와 같은 shape인 이유는? 캐시 히트가 500토큰 있었다면 `positions`의 첫 값은 0일까 500일까?
- `req_pool_indices`가 `[0,1,2,3]`이 아니라 `[3,0,7,1]`처럼 나올 수 있습니다. 왜일까요? (힌트: 슬롯 풀은 재사용됨)

---

## 실습 2 — KV 슬롯 번호 추적: radix tree와 연결하기 (약 25분) ⭐

**목표**: 2회차에서 tree node의 `value`에 들어있던 "KV cache indices"가 실제로 어디를 가리키는지 확인한다.

### 2-1. 페이지 테이블 읽기

`lab5_dump.py`가 이미 `req_to_token[slot, :seq_len]`을 출력하고 있습니다. EXTEND와 DECODE에서 각각 관찰하세요.

```
EXTEND 직후:  req_to_token[0] = [0, 1, 2, ..., 255]          (연속일 수도 있음)
DECODE 직후:  req_to_token[0] = [0, 1, 2, ..., 255, 1031]    ← 새 슬롯이 뒤에 붙음
```

확인할 것:

1. DECODE의 `out_cache_loc` 값 4개가 `req_to_token` 각 행의 **맨 끝**에 그대로 나타나는가?
2. 슬롯 번호가 연속인가, 흩어져 있는가? 여러 스텝을 돌리면 어떻게 되는가?
3. 흩어져 있다면 → attention 커널은 K/V를 **gather**해서 읽어야 한다는 뜻입니다. 이게 "paged attention"이라고 부르는 것의 실체입니다.

### 2-2. page_size 바꿔보기

```bash
python lab5_dump.py --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --batch-size 4 --input-len 256 --output-len 8 \
  --disable-cuda-graph --page-size 16
```

`out_cache_loc`와 `req_to_token` 값의 **정렬 패턴**이 어떻게 바뀌는지 보세요. 16의 배수 경계가 보이기 시작합니다.

그리고 서버 쪽에서 2회차 실습 1을 `--page-size 1`과 `--page-size 32`로 각각 재실행해 `cached_tokens`를 비교하세요.

| page_size | 예상 cached_tokens | 이유 |
|---|---|---|
| 1 | 최대 | 토큰 단위 매칭 |
| 32 | 32의 배수로 내림 | 페이지가 꽉 안 차면 매칭 불가 |

2회차에서 봤던 `RadixKey.page_aligned(page_size)`가 바로 이 동작입니다. **캐시 재사용률 ↔ 커널 효율**의 트레이드오프를 처음으로 숫자로 만지는 지점입니다.

### 2-3. 토론 포인트

- `--page-size 64`인데 프롬프트가 32토큰이면 캐시 히트가 0입니다. 실제 서비스에서 이게 문제가 되는 워크로드는? 안 되는 워크로드는?
- 두 요청이 같은 prefix를 공유할 때, `req_to_token`의 두 행에 **같은 슬롯 번호**가 들어있는 것을 확인할 수 있을까요? (서버 모드 + 프리필 배치 안에 두 요청을 같이 넣어야 함)

---

## 실습 3 — CUDA Graph의 효과 측정 (약 25분) ⭐

**목표**: CUDA graph가 무엇을 없애주는지, 언제 효과가 크고 언제 작은지를 배치 크기별로 측정한다.

### 3-1. 측정

```bash
for BS in 1 4 16 64; do
  echo "===== batch_size=$BS (graph ON) ====="
  python -m sglang.bench_one_batch --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --batch-size $BS --input-len 128 --output-len 32 2>&1 | grep -i "decode"

  echo "===== batch_size=$BS (graph OFF) ====="
  python -m sglang.bench_one_batch --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --batch-size $BS --input-len 128 --output-len 32 --disable-cuda-graph 2>&1 | grep -i "decode"
done
```

표로 정리:

| batch_size | decode latency (graph ON) | (graph OFF) | 절대 차이 | 상대 차이 |
|---|---|---|---|---|
| 1 | | | | |
| 4 | | | | |
| 16 | | | | |
| 64 | | | | |

### 3-2. 예상 결과와 해석

- **절대 차이(ms)는 배치 크기와 거의 무관하게 일정**해야 합니다. 커널 런치 오버헤드는 배치 크기가 아니라 **레이어 수 × 연산 수**에 비례하기 때문입니다.
- **상대 차이(%)는 배치가 커질수록 줄어듭니다.** GPU가 실제로 하는 일이 늘어나면서 런치 오버헤드가 묻히기 때문입니다.
- 즉 **CUDA graph는 저부하·저지연 구간에서 가장 크게 이깁니다.** 이게 "zero-overhead scheduler"류 최적화가 왜 중요한지의 전조입니다(6회차 예고).

작은 모델(1.5B)을 쓰는 게 오히려 유리합니다. 모델이 작을수록 커널당 GPU 시간이 짧아서 런치 오버헤드 비중이 커지고, 차이가 극적으로 보입니다.

### 3-3. 패딩 관찰

```bash
python -m sglang.bench_one_batch --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --batch-size 5 --input-len 128 --output-len 32 --log-level debug 2>&1 | grep -i "capture\|graph"
```

- 시작할 때 어떤 배치 크기들을 캡처하는지 로그로 확인
- `--cuda-graph-max-bs 4`로 낮추면 `batch-size 8`은 어떻게 되는지 (캡처 범위를 벗어나 eager로 떨어짐)
- `--cuda-graph-max-bs`를 크게 주면 startup 시간과 GPU 메모리가 어떻게 변하는지 (`nvidia-smi`로 확인)

### 3-4. 토론 포인트

- 배치 5가 캡처된 8로 패딩된다면, 낭비되는 연산은 얼마인가? 그런데도 왜 이득인가?
- 왜 프리필은 캡처하지 않는가? "토큰 수를 버킷팅하면 되지 않나?"에 대한 반박을 각자 만들어보기.
- CUDA graph를 켠 상태로 `pdb`를 걸면 왜 스택이 이상하게 보일까?

---

## 실습 4 — Attention Backend 교체 및 코드 읽기 (약 25분)

**목표**: `RadixAttention`이 실제 커널이 아니라 **교체 가능한 이음새**임을 확인하고, backend가 준비하는 메타데이터를 읽는다.

### 4-1. 성능 비교

```bash
for B in triton flashinfer fa3; do
  echo "===== $B ====="
  python -m sglang.bench_one_batch --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --batch-size 16 --input-len 512 --output-len 32 \
    --attention-backend $B 2>&1 | tail -20
done
```

> 하드웨어에 따라 지원되지 않는 backend가 있습니다(`fa3`는 Hopper+CUDA 12.3 이상 등). 에러가 나면 그 자체가 좋은 관찰거리이니 로그를 읽고 넘어가세요. 사용 가능한 목록은 `--help`와 `docs/advanced_features/attention_backend.md`에서 확인.

prefill과 decode를 따로 지정할 수도 있습니다 — 왜 이런 옵션이 존재하는지 생각해보세요:

```bash
--prefill-attention-backend fa3 --decode-attention-backend triton
```

### 4-2. 코드 읽기 (여기가 본론)

`python/sglang/srt/layers/attention/triton_backend.py`를 엽니다. 세 함수만 봅니다.

```bash
grep -n "def init_forward_metadata\|def forward_extend\|def forward_decode" \
  python/sglang/srt/layers/attention/triton_backend.py
```

읽으면서 답할 것:

1. `init_forward_metadata`는 **forward당 몇 번** 호출되는가? `forward_decode`는? (힌트: 레이어 수)
2. → 그래서 무거운 준비 작업(page table gather, `cu_seqlens` 계산, plan)은 어디에 있어야 하는가?
3. `forward_decode` 안에서 `req_to_token`을 어떻게 쓰는가? 실습 2에서 본 그 테이블이 맞는가?
4. `forward_extend`에서 prefix가 있을 때와 없을 때 경로가 갈리는가?

그다음 `python/sglang/srt/layers/radix_attention.py`를 엽니다. **20줄 남짓**입니다. 이름과 달리 radix tree 로직이 하나도 없다는 것을 확인하고, 왜 이 파일이 이렇게 얇은지 토론하세요.

마지막으로 `python/sglang/srt/models/qwen2.py`에서 `RadixAttention`이 어디서 생성되는지 찾아보세요:

```bash
grep -n "RadixAttention" python/sglang/srt/models/qwen2.py
```

**"새 모델을 추가할 때 attention 커널을 건드릴 필요가 없다"**는 게 무슨 뜻인지 여기서 체감됩니다.

---

## 실습 5 — 프로파일러로 한 스텝 뜯어보기 (시간 남으면, 약 20분)

**목표**: decode 한 스텝의 시간이 어디로 가는지 trace로 확인한다.

```bash
python -m sglang.bench_one_batch --model-path Qwen/Qwen2.5-1.5B-Instruct \
  --batch-size 8 --input-len 256 --output-len 16 \
  --disable-cuda-graph --profile
```

생성된 trace 파일을 다운로드해서 <https://ui.perfetto.dev> 에 드래그하면 됩니다.

볼 것:

1. **CPU 타임라인과 GPU 타임라인 사이의 빈틈** — CUDA graph OFF 상태라면 GPU 커널 사이사이에 공백이 보여야 합니다. 이게 3-2에서 측정한 그 오버헤드의 정체입니다.
2. **레이어 하나의 커널 구성** — qkv projection GEMM → attention 커널 → o_proj → MLP GEMM 2~3개 → RMSNorm. 몇 개인지 세어보고 × 레이어 수를 하면 스텝당 커널 수가 나옵니다.
3. **attention 커널이 전체에서 차지하는 비중** — `--input-len`을 256 → 2048로 늘리면 이 비중이 어떻게 변하는가? (KV cache 읽는 양이 늘어남)
4. `--profile` 없이 `--disable-cuda-graph`를 뺀 버전과 비교하면 replay 하나로 뭉쳐진 것을 볼 수 있습니다.

---

## 실습 6 — LogitsProcessor와 Sampler (선택, 약 15분)

**목표**: 모델이 뱉은 hidden state가 왜 전부 다 쓰이지 않는지 확인한다.

### 6-1. 마지막 토큰만 살아남는다

```bash
grep -n "def forward" python/sglang/srt/layers/logits_processor.py
```

`extend_seq_lens`를 이용해 시퀀스별 **마지막 hidden state만 슬라이싱**하는 부분을 찾으세요. 프롬프트가 4096토큰이면 hidden state는 4096개인데 LM head를 통과하는 건 1개입니다.

- `(4096, d) × (d, vocab)`와 `(1, d) × (d, vocab)`의 FLOPs 차이를 vocab=150k로 계산해보세요.

### 6-2. return_logprob의 비용

```bash
# 서버 모드
curl -X POST http://localhost:30000/generate -H "Content-Type: application/json" -d '{
  "text": "The capital of France is",
  "sampling_params": {"max_new_tokens": 8, "temperature": 0},
  "return_logprob": true, "logprob_start_len": 0
}' | python -m json.tool | head -40
```

`return_logprob: true/false`로 각각 여러 번 던져 지연시간을 비교하세요. 6-1에서 계산한 "생략된 연산"이 여기서는 생략되지 않기 때문에 차이가 납니다.

### 6-3. Sampler

```bash
grep -n "temperature\|top_p\|top_k\|multinomial\|argmax" python/sglang/srt/layers/sampler.py | head -30
```

`SamplingBatchInfo`가 요청별 파라미터를 `(bs, 1)` 텐서로 배치한다는 것을 확인하세요. **배치 안의 요청마다 temperature가 달라도 배치를 쪼갤 필요가 없다**는 게 핵심입니다.

---

## 시간 배분 요약

| 순서 | 실습 | 시간 | 우선순위 |
|---|---|---|---|
| 0 | 소스 체크아웃 + `bench_one_batch` + 헬퍼 | 15분 | 필수 |
| 1 | EXTEND vs DECODE 텐서 모양 | 30분 | ⭐ 필수 |
| 2 | KV 슬롯 추적 + page_size | 25분 | ⭐ 필수 |
| 3 | CUDA Graph 측정 | 25분 | ⭐ 필수 |
| 4 | Attention backend 교체 + 코드 읽기 | 25분 | 권장 |
| 5 | 프로파일러 | 20분 | 선택 |
| 6 | LogitsProcessor / Sampler | 15분 | 선택 |

2시간이면 0~3 + 4의 코드 읽기 부분까지가 현실적입니다. 5·6은 숙제로 돌려도 좋습니다.

---

## 예산·시간 절약 팁

- **모델은 계속 1.5B로.** 오늘 보는 것(텐서 shape, 슬롯 번호, 런치 오버헤드)은 모델 크기와 무관하고, 오히려 작은 모델일수록 CUDA graph 효과가 선명합니다.
- `bench_one_batch`는 **HTTP 토큰 소비가 없습니다.** 합성 입력으로 로컬에서 도는 것이라 GPU 시간만 씁니다.
- 실습 3의 이중 루프가 가장 오래 걸립니다. 시간이 빠듯하면 `batch_size`를 `[1, 16]` 두 점만 재도 추세는 보입니다.
- 매 실행마다 모델 로딩(~20초)이 반복됩니다. 로딩 시간을 빼고 비교하려면 `bench_one_batch`가 출력하는 **decode latency**만 보세요(로딩 포함 total이 아니라).

---

## 트러블슈팅

**심볼을 못 찾겠을 때** — 파일이 옮겨졌을 가능성이 높습니다.

```bash
grep -rn "def forward_batch_generation" python/sglang/srt/
grep -rn "class ForwardBatch"          python/sglang/srt/
grep -rn "class ForwardMode"           python/sglang/srt/
grep -rn "out_cache_loc"               python/sglang/srt/model_executor/
grep -rn "class CudaGraphRunner"       python/sglang/srt/
grep -rn "class RadixAttention"        python/sglang/srt/
```

**monkey patch가 안 먹을 때** — TP > 1이면 자식 프로세스가 뜨면서 패치가 전파되지 않습니다. 오늘은 반드시 **TP=1**로 하세요.

**OOM** — `--mem-fraction-static 0.7`로 낮추거나 `--cuda-graph-max-bs`를 줄이세요. 실습 3-3에서 이 둘이 같은 메모리를 놓고 경쟁한다는 걸 보게 됩니다.

**출력이 너무 많을 때** — `lab5_dump.py`의 `MAX_PRINT_PER_MODE`를 1로 낮추거나, `--disable-cuda-graph`를 확인하세요.

**코드를 되돌리고 싶을 때**

```bash
git checkout .          # 실습 중 넣은 print 전부 제거
git status              # 깨끗한지 확인
```

---

## 다음 회차로 넘어가는 질문

실습 3에서 CUDA graph를 켜서 GPU 커널 런치 오버헤드는 없앴습니다. 그런데 **스텝 N의 GPU 연산이 도는 동안 CPU는 뭘 하고 있나요?** 스텝 N+1의 `ScheduleBatch`를 만들고 있어야 정상이지만, 만약 CPU가 스텝 N의 결과를 기다린 다음에 시작한다면 GPU는 그만큼 놉니다.

→ 6회차: overlap scheduler와 CPU 오버헤드, 그리고 `bench_serving`으로 하는 본격적인 성능 측정.
