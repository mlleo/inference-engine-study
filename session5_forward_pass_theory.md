# SGLang Model Execution: From ScheduleBatch to Logits

A code-level walkthrough of what happens **after** the scheduler picks a batch — how a batch becomes GPU tensors, how attention reads the KV cache, why CUDA graphs exist, and how logits become the next token.

> **A note on line numbers.** The RadixTree material from Session 2 cited exact line numbers. This document deliberately cites **file paths and symbol names only**, because `python/sglang/srt/` is refactored frequently (`srt/utils/` was split into a subpackage, `input_buffers.py` and `piecewise_cuda_graph_runner.py` were added, etc.). Pin one commit for the whole study group, then run the `grep` recipes in §12 to resolve every symbol to a line number in *your* checkout. Code blocks below are **simplified for reading** — they show structure, not verbatim source.

---

## 1. Where We Are

Sessions 1–4 covered the request's journey up to the moment it is scheduled:

```
HTTP request
  → TokenizerManager        (session 1)
  → Scheduler waiting_queue (session 3)
  → match_prefix / RadixTree(session 2)  ← how many tokens can we skip?
  → HiCache                 (session 4)  ← where does evicted KV go?
  → [ ??? ]                 ← TODAY
  → DetokenizerManager → HTTP response
```

Everything so far has been about **deciding what to compute**. Today is about **actually computing it**. The question this session answers:

> The scheduler produced a `ScheduleBatch` with 8 requests and a list of KV cache indices. What has to happen before a GPU kernel can run, and what happens after it finishes?

---

## 2. Why the Batch Is Transformed Three Times

A batch changes identity three times on its way to the GPU. Understanding *why* explains a large part of SGLang's design.

| Stage | Class | File | Lives where | Contains |
|---|---|---|---|---|
| 1 | `ScheduleBatch` | `managers/schedule_batch.py` | Scheduler process, CPU | `List[Req]` objects, tree-cache nodes, sampling params, Python state |
| 2 | `ModelWorkerBatch` | `managers/schedule_batch.py` | Boundary object, CPU | Only the fields the model actually needs — flat lists/arrays |
| 3 | `ForwardBatch` | `model_executor/forward_batch_info.py` | Model worker, **GPU** | `torch.Tensor` on device + attention metadata |

```python
# Conceptually, in the scheduler loop:
batch: ScheduleBatch = self.get_next_batch_to_run()
model_worker_batch = batch.get_model_worker_batch()      # 1 → 2
result = self.tp_worker.forward_batch_generation(model_worker_batch)

# Inside TpModelWorker (managers/tp_worker.py):
forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)  # 2 → 3
logits_output, _ = self.model_runner.forward(forward_batch)
next_token_ids = self.model_runner.sample(logits_output, model_worker_batch)
```

**Why not one class?** Three reasons, all worth discussing in the session:

1. **A `Req` object is not sendable.** `Req` holds a reference to `last_node`, a live `TreeNode` in the radix tree. Under TP > 1 the batch must be broadcast to every TP rank, and under a pipeline/overlap setup it crosses a thread or process boundary. `ModelWorkerBatch` is the serializable subset.
2. **CPU/GPU separation.** `ForwardBatch` construction is where `.to(device)` happens. Keeping that in one place makes it possible to overlap the CPU-side work of step *N+1* with the GPU-side work of step *N* (the "zero-overhead scheduler" — Session 6 material).
3. **Different consumers.** The scheduler cares about priority, retraction and eviction. The attention kernel cares about `cu_seqlens`. Neither wants the other's fields.

---

## 3. ForwardMode — One Batch, Two Very Different Computations

`ForwardMode` is defined in `model_executor/forward_batch_info.py`. The important members:

| Mode | Meaning |
|---|---|
| `EXTEND` | Prefill — process many new tokens per sequence |
| `DECODE` | One new token per sequence |
| `IDLE` | Nothing to run (needed so all DP/TP ranks stay in lockstep) |
| `TARGET_VERIFY`, `DRAFT_EXTEND` | Speculative decoding (Session 7) |

Note there is no separate "prefill" mode: a fresh prompt and a chunked continuation are both `EXTEND`. `EXTEND` means "this sequence already has `prefix_len` tokens of KV cache, now add `extend_len` more" — and a cold request is simply the `prefix_len = 0` case. This is exactly the abstraction that makes the radix cache from Session 2 free to integrate: a cache hit just raises `prefix_len`.

### 3.1 Why the two modes behave so differently

For a single sequence with hidden size `d`:

| | EXTEND (L new tokens) | DECODE (1 new token) |
|---|---|---|
| GEMM shapes | `(L, d) × (d, d)` — tall | `(1, d) × (d, d)` — a matrix-vector product |
| Weights read per step | once | once |
| Arithmetic intensity | high | **very low** |
| Bottleneck | GPU FLOPs | **memory bandwidth** (weights + KV cache) |
| Batching helps because | already saturated | more rows amortize the same weight read |

This asymmetry drives almost everything downstream:

- Batching is enormously valuable in decode and only mildly so in prefill → **continuous batching**.
- Decode kernels are launched thousands of times with identical shapes → **CUDA graphs** (§8).
- Prefill and decode compete for the same GPU and hurt each other's latency → **PD disaggregation** (a later session).

---

## 4. The Memory Pools — Where "KV Cache Index" Actually Points

In Session 2 the radix tree stored `value = torch.Tensor` of "KV cache indices". This session is where those integers get dereferenced.

There are two levels of indirection, both set up in `ModelRunner` and defined in `mem_cache/memory_pool.py` (with the allocator often split into `mem_cache/allocator.py`):

```
req_pool_indices ──►  ReqToTokenPool
                      req_to_token[req_slot, position] = kv_index
                                                          │
                                                          ▼
                                        MHATokenToKVPool / MLATokenToKVPool
                                        k_buffer[layer][kv_index] = key vector
                                        v_buffer[layer][kv_index] = value vector
```

- **`ReqToTokenPool`** — a fixed `(max_running_requests, max_context_len)` int32 table. Row = a running request's slot; entry at column *p* = which KV slot holds position *p* of that sequence. This is the **page table**.
- **`TokenToKVPool`** — the actual tensors, allocated once at startup, sized by `--mem-fraction-static`. Layout is `[layer][num_slots, num_kv_heads, head_dim]` for MHA; MLA stores a single compressed latent per token instead.
- **`TokenToKVPoolAllocator`** — the free-list. `alloc(n)` returns `n` slot indices; the radix cache calls `free()` on eviction.

**Key insight for the study group:** the KV cache is not per-request storage. It is one global slab of numbered slots. A request "owns" a scattered set of slot numbers, and the radix tree is what allows two requests to own the *same* slot numbers. Prefix sharing is index sharing — no tensor is copied.

### 4.1 `out_cache_loc`

The single most important field to understand in `ForwardBatch`:

```python
out_cache_loc: torch.Tensor   # int, len == number of tokens computed this step
```

It is the answer to "**where should the K/V produced by this forward pass be written?**" Every attention layer writes into `token_to_kv_pool` at exactly these slots, and `req_to_token` is updated so future steps can find them.

- `EXTEND`: `len(out_cache_loc) == sum(extend_seq_lens)` — only the *uncached* tail. The prefix's slots are already populated and are read, not written.
- `DECODE`: `len(out_cache_loc) == batch_size` — one slot per sequence.

Cache hit, expressed in one line: *a longer matched prefix means a shorter `out_cache_loc`.*

### 4.2 `page_size`

`--page-size` groups tokens into blocks of contiguous KV slots.

- `page_size = 1` — token-level allocation and token-level prefix matching. Maximum reuse.
- `page_size > 1` — fewer, larger blocks; better attention-kernel performance and smaller page tables, but a prefix can only be matched in whole pages, so a partial page is not reusable.

This is why Session 2's `RadixKey.page_aligned(page_size)` existed. It is also the first real trade-off knob the group can measure.

---

## 5. Anatomy of a ForwardBatch

Fields worth reading aloud in the session (`model_executor/forward_batch_info.py`):

```python
@dataclass
class ForwardBatch:
    forward_mode: ForwardMode
    batch_size: int

    input_ids: torch.Tensor        # EXTEND: (total_tokens,)   DECODE: (bs,)
    positions: torch.Tensor        # RoPE positions, same shape as input_ids
    req_pool_indices: torch.Tensor # (bs,) rows into ReqToTokenPool
    seq_lens: torch.Tensor         # (bs,) TOTAL length incl. cached prefix
    out_cache_loc: torch.Tensor    # where to write new KV (see §4.1)

    # EXTEND only
    extend_seq_lens: torch.Tensor      # (bs,) new tokens per sequence
    extend_prefix_lens: torch.Tensor   # (bs,) cached tokens per sequence

    # references, not data
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: KVCache
    attn_backend: AttentionBackend

    sampling_info: SamplingBatchInfo
```

Two things surprise people the first time:

1. **`input_ids` is flat, not `(batch, seq)`.** There is no padding anywhere. Variable-length sequences are concatenated and boundaries are described by `extend_seq_lens` / `cu_seqlens`. Padding a `(bs, max_len)` tensor would waste both compute and memory when lengths differ by 10×.
2. **`seq_lens` includes the cached prefix, `extend_seq_lens` does not.** Attention must read `seq_lens` tokens of K/V while only `extend_seq_lens` queries are computed. Getting these two confused is the most common bug when writing an attention backend.

---

## 6. The Attention Layer and Its Backends

### 6.1 `RadixAttention` is a thin shim

`layers/radix_attention.py` defines the module every model file instantiates. It holds configuration (`num_heads`, `head_dim`, `scaling`, `layer_id`, `sliding_window_size`) and delegates:

```python
class RadixAttention(nn.Module):
    def forward(self, q, k, v, forward_batch, ...):
        return forward_batch.attn_backend.forward(q, k, v, self, forward_batch)
```

The name is historical and slightly misleading — it does not contain radix tree logic. It is the seam where the model graph meets a swappable kernel. Any model in `srt/models/` uses it; that is why adding a model rarely requires touching attention.

### 6.2 The backend contract

`layers/attention/base_attn_backend.py`:

```python
class AttentionBackend:
    def init_forward_metadata(self, forward_batch): ...      # once per forward
    def init_cuda_graph_state(self, max_bs): ...             # capture-time buffers
    def init_forward_metadata_capture_cuda_graph(...): ...
    def init_forward_metadata_replay_cuda_graph(...): ...
    def forward_extend(self, q, k, v, layer, forward_batch): ...
    def forward_decode(self, q, k, v, layer, forward_batch): ...
```

The critical split: **`init_forward_metadata` runs once per forward pass; `forward_*` runs once per layer.** A 32-layer model calls the metadata prep once and the kernel 32 times. Anything that can be hoisted out of the per-layer path belongs in metadata — this is where FlashInfer's `plan()` calls, page-table gathering, and `cu_seqlens` construction happen.

### 6.3 Ragged vs paged

Two shapes of attention appear in one server:

- **Ragged** — Q, K and V are all fresh, contiguous, variable-length. Used for `EXTEND` with no cached prefix. Described by `cu_seqlens_q` / `cu_seqlens_k` (cumulative-sum offsets marking sequence boundaries in the flat tensor).
- **Paged** — Q is fresh, K/V must be gathered from scattered slots in `token_to_kv_pool` via `req_to_token`. Used for `DECODE`, and for `EXTEND` with a cache hit.

FlashInfer exposes these as separate wrappers (`BatchPrefillWithRaggedKVCacheWrapper`, `BatchPrefillWithPagedKVCacheWrapper`, `BatchDecodeWithPagedKVCacheWrapper`). A prefill with a partial cache hit may use both and merge the results.

### 6.4 Which backend

`--attention-backend`, or `--prefill-attention-backend` / `--decode-attention-backend` separately. Common values: `fa3`, `flashinfer`, `triton`, `trtllm_mha`, `trtllm_mla`, `torch_native`. If unspecified, SGLang picks by hardware and model architecture — for example Hopper defaults to `fa3` under supported configurations, and Blackwell to `trtllm_mha`. Check `docs/advanced_features/attention_backend.md` in your pinned commit for the current support matrix.

For **reading**, start with `triton_backend.py`. It is pure Python + Triton, so the kernel is right there. `flashinfer_backend.py` is mostly plumbing around an external library.

---

## 7. Sampling: From Hidden States to a Token

Two stages, both easy to overlook.

### 7.1 `LogitsProcessor` (`layers/logits_processor.py`)

The model produces a hidden state for **every** token in `input_ids`. In `EXTEND` with 4096 prompt tokens, projecting all of them through a `(d, vocab)` matrix would be a massive waste — only the last position of each sequence predicts anything.

So `LogitsProcessor` first **slices out the last hidden state of each sequence** (using `extend_seq_lens` to find the boundaries), and only then applies the LM head. The exception is `return_logprob`, where the caller explicitly wants per-token logprobs and the full projection is unavoidable — a good explanation for why `--return-logprob` costs real throughput.

Output is a `LogitsProcessorOutput` carrying `next_token_logits` plus optional logprob fields.

### 7.2 `Sampler` (`layers/sampler.py`, with `sampling/sampling_batch_info.py`)

`SamplingBatchInfo` batches per-request sampling parameters into tensors — `temperatures (bs, 1)`, `top_ps`, `top_ks`, penalty state. The sampler then applies, roughly in order: penalties → temperature scaling → softmax → top-k/top-p filtering → multinomial draw. Greedy (`temperature = 0`) short-circuits to `argmax`.

Two design points worth noting: every request in the batch can have different sampling parameters without splitting the batch, and grammar-constrained decoding (a later session) plugs in here as a logit mask.

---

## 8. CUDA Graphs

### 8.1 The problem

A decode step on a 7B model might take ~10 ms of GPU time and consist of several hundred kernel launches. Each launch costs the CPU roughly 5–10 µs of Python + PyTorch dispatch + driver work. At small batch sizes the CPU cannot issue work fast enough and the GPU idles between kernels. The GPU is memory-bound *and* launch-bound.

### 8.2 The fix

A CUDA graph records a sequence of kernel launches once and replays the whole thing with a single call. `model_executor/cuda_graph_runner.py`:

- **Capture** happens at startup, for a fixed list of batch sizes (`--cuda-graph-max-bs`, `--cuda-graph-bs`). This is why server startup pauses for several seconds after the weights load.
- **Replay** requires the input tensors to live at the *same addresses* as at capture time. So the runner keeps pre-allocated static buffers, copies the real inputs into them, and replays.
- A batch of size 5 with captured sizes `[1, 2, 4, 8, ...]` is **padded up to 8**. Padding wastes a little compute and is still much cheaper than launching eagerly.

### 8.3 Why decode and not prefill

Graphs require static shapes. Decode shapes depend only on `batch_size` — a small, enumerable set. Prefill shapes depend on total token count, which is effectively unbounded. Hence: decode is always captured; prefill is not, except in specific configurations (e.g. speculative decoding with `--speculative-attention-mode prefill`, or the newer piecewise-graph path in `piecewise_cuda_graph_runner.py`).

Consequences the group should be able to predict before measuring:

- CUDA graphs help **most at small batch sizes** and taper off as batch grows (GPU work per step grows, launch overhead stays constant).
- They cost GPU memory (captured graphs and static buffers), which competes with KV cache.
- `--disable-cuda-graph` is the first thing to try when debugging, because graph replay makes stack traces useless.

---

## 9. Complete Call Chain

### 9.1 A DECODE step (steady state, CUDA graph active)

```
Scheduler.run_batch()                                   managers/scheduler.py
→ batch.get_model_worker_batch()                        managers/schedule_batch.py
   ├─ collects input_ids (1 per req), seq_lens, req_pool_indices
   └─ out_cache_loc = allocator.alloc(batch_size)       mem_cache/allocator.py
↓
TpModelWorker.forward_batch_generation(mwb)             managers/tp_worker.py
↓
ForwardBatch.init_new(mwb, model_runner)                model_executor/forward_batch_info.py
   ├─ .to(device) for every tensor
   ├─ attaches req_to_token_pool / token_to_kv_pool / attn_backend
   └─ attn_backend.init_forward_metadata(fb)            layers/attention/*_backend.py
↓
ModelRunner.forward(fb)                                 model_executor/model_runner.py
→ CudaGraphRunner.can_run(fb)?                          model_executor/cuda_graph_runner.py
   ├─ YES → copy inputs into static buffers → graph.replay()
   └─ NO  → forward_decode(fb) eagerly
↓
   model.forward(input_ids, positions, forward_batch)   models/qwen2.py
   └─ for each decoder layer:
        qkv_proj → RadixAttention.forward()             layers/radix_attention.py
          → attn_backend.forward_decode()
              ├─ write new K/V to token_to_kv_pool[out_cache_loc]
              ├─ gather old K/V via req_to_token[req_pool_indices, :seq_lens]
              └─ kernel
        → o_proj → MLP → residual
↓
LogitsProcessor(hidden_states, lm_head, fb)             layers/logits_processor.py
→ LogitsProcessorOutput.next_token_logits  (bs, vocab)
↓
ModelRunner.sample(logits_output, mwb)                  layers/sampler.py
→ next_token_ids  (bs,)
↓
Scheduler.process_batch_result_decode()                 managers/scheduler_components/...
→ append token to each Req, check stop conditions
→ stream out via DetokenizerManager
```

### 9.2 An EXTEND step with a cache hit

Same skeleton, with these differences:

```
match_prefix() returned prefix_len = 500 for a 512-token prompt   (session 2)
↓
get_model_worker_batch():
   seq_lens          = [512]        ← total
   extend_prefix_lens= [500]        ← already in KV cache, NOT recomputed
   extend_seq_lens   = [12]         ← actually computed now
   input_ids         = 12 token ids ← not 512!
   out_cache_loc     = alloc(12)    ← only 12 new slots
↓
ForwardBatch.init_new → attn_backend.init_forward_metadata
   → builds a paged prefill plan: 12 queries attend over 512 keys
↓
model.forward → RadixAttention → forward_extend
   → writes 12 new K/V entries, reads 500 cached + 12 new
↓
LogitsProcessor slices the LAST of the 12 hidden states → 1 logit row
↓
after the step: tree_cache.cache_unfinished_req(req) inserts the 12 new
                indices into the radix tree                (session 2)
```

The `#cached-token` number you measured in Session 2's lab is `sum(extend_prefix_lens)`. The lab for this session makes that identity visible in the tensors themselves.

---

## 10. Where the Time Goes in One Decode Step

A useful mental budget to sanity-check any measurement:

| Component | Scales with | Reduced by |
|---|---|---|
| Weight reads (GEMMs) | model size; **not** batch size | quantization, TP |
| KV cache reads (attention) | batch size × sequence length | MLA/GQA, paged kernels, shorter contexts |
| Kernel launch / Python overhead | number of layers × number of ops | CUDA graphs, overlap scheduler, `torch.compile` |
| Sampling | batch size × vocab | cheap unless `return_logprob` |

Predict before measuring: at `batch_size = 1`, which row dominates? At `batch_size = 64` with 4k contexts? The lab measures both.

---

## 11. Discussion Questions

1. Why does `ForwardBatch` need `seq_lens` *and* `extend_seq_lens`? Construct a case where using the wrong one produces a plausible-looking but wrong output.
2. `out_cache_loc` for a decode batch of 8 is 8 integers, and they are usually **not** contiguous. Where do they come from, and what does that imply about the attention kernel's memory access pattern?
3. Why can't we CUDA-graph the prefill path by simply bucketing token counts the way we bucket batch sizes?
4. If two requests share a 500-token prefix and both are in the same `EXTEND` batch, is the prefix's KV read once or twice? Does the answer change with `page_size`?
5. `LogitsProcessor` throws away all but the last hidden state per sequence. Speculative decoding needs more than the last one — where would you change this? (Foreshadows `CaptureHiddenMode`.)
6. `--mem-fraction-static` sets the KV pool size. CUDA graphs also consume memory. Sketch the failure mode when both are set aggressively.

---

## 12. Key File Reference

| File | What's in it |
|---|---|
| `python/sglang/srt/managers/schedule_batch.py` | `ScheduleBatch`, `Req`, `ModelWorkerBatch`, `get_model_worker_batch()` |
| `python/sglang/srt/managers/tp_worker.py` | `TpModelWorker.forward_batch_generation` — the 2→3 handoff |
| `python/sglang/srt/model_executor/forward_batch_info.py` | `ForwardMode`, `ForwardBatch`, `init_new`, `CaptureHiddenMode` |
| `python/sglang/srt/model_executor/model_runner.py` | `ModelRunner` — weight loading, memory pool init, backend init, `forward`, `sample` |
| `python/sglang/srt/model_executor/cuda_graph_runner.py` | `CudaGraphRunner` — `capture`, `can_run`, `replay`, static buffers |
| `python/sglang/srt/mem_cache/memory_pool.py` | `ReqToTokenPool`, `MHATokenToKVPool`, `MLATokenToKVPool` |
| `python/sglang/srt/mem_cache/allocator.py` | `TokenToKVPoolAllocator` — `alloc` / `free` |
| `python/sglang/srt/layers/radix_attention.py` | `RadixAttention` — the model↔backend seam |
| `python/sglang/srt/layers/attention/base_attn_backend.py` | The backend interface every kernel implements |
| `python/sglang/srt/layers/attention/triton_backend.py` | Most readable backend — start here |
| `python/sglang/srt/layers/attention/flashinfer_backend.py` | Wrapper dispatch, `plan()`, ragged vs paged |
| `python/sglang/srt/layers/logits_processor.py` | Last-token slicing, LM head, logprobs |
| `python/sglang/srt/layers/sampler.py` | Penalties, temperature, top-k/top-p, multinomial |
| `python/sglang/srt/sampling/sampling_batch_info.py` | Per-request sampling params as batched tensors |
| `python/sglang/srt/models/qwen2.py`, `llama.py` | A model file — short, worth reading end to end |
| `python/sglang/bench_one_batch.py` | Runs the whole path with no server. Today's lab vehicle. |

### If a symbol has moved in your checkout

```bash
cd sglang
grep -rn "def forward_batch_generation" python/sglang/srt/
grep -rn "class ForwardMode"           python/sglang/srt/
grep -rn "class ForwardBatch"          python/sglang/srt/
grep -rn "out_cache_loc"               python/sglang/srt/model_executor/
grep -rn "class RadixAttention"        python/sglang/srt/
grep -rn "def init_forward_metadata"   python/sglang/srt/layers/attention/
grep -rn "class CudaGraphRunner"       python/sglang/srt/
```

---

## 13. Summary

1. A batch is transformed **three times** — `ScheduleBatch` (CPU, rich) → `ModelWorkerBatch` (serializable subset) → `ForwardBatch` (GPU tensors) — because of process boundaries, the CPU/GPU boundary, and differing consumers.
2. `ForwardMode` splits execution into `EXTEND` (compute-bound, many tokens) and `DECODE` (memory-bound, one token). Nearly every optimization in SGLang targets one or the other, rarely both.
3. The KV cache is a **global numbered slab**, addressed through `req_to_token` → `token_to_kv_pool`. Prefix sharing is index sharing; nothing is copied. `out_cache_loc` is where this step writes, and a cache hit is literally a shorter `out_cache_loc`.
4. `RadixAttention` is a **seam**, not an algorithm. The backend does the real work, and it splits into per-forward metadata preparation and per-layer kernel invocation.
5. **CUDA graphs** eliminate kernel-launch overhead for the fixed shapes of decode, at the cost of memory, padding, and debuggability.
6. `LogitsProcessor` slices out only the last hidden state per sequence before the LM head; `Sampler` applies per-request parameters batched into tensors.

**Next session hook:** everything above assumed the CPU finishes preparing step *N+1* before the GPU finishes step *N*. When it doesn't, the GPU idles. That's the overlap scheduler — and it's how we start measuring instead of reading.
