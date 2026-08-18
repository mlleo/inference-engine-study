# PagedAttention in vLLM — A Beginner's Guide

A code-level explanation of PagedAttention, following the SOSP 2023 paper (arXiv:2309.06180)
and the vLLM codebase (github.com/vllm-project/vllm, v1 architecture).

Every claim references exact file paths and line numbers from the vLLM codebase.

---

## Table of Contents

1. The Problem: KV Cache Memory Waste
2. The OS Analogy: Virtual Memory with Paging
3. PagedAttention: The Attention Algorithm
4. The KV Cache Manager: Logical vs Physical Blocks
5. Block Tables: Mapping Logical to Physical
6. The Block Pool: Allocation, Freeing, and Eviction
7. Prefix Caching: Hash-Based Block Reuse
8. Copy-on-Write: Sharing Blocks Across Sequences
9. The Scheduler: Tying It All Together
10. The Attention Kernel: How block_table Reaches the GPU
11. Scheduling and Preemption: Swapping vs Recomputation
12. Complete Request Lifecycle: From Arrival to Completion
13. Comparison with SGLang's RadixAttention

---

## 1. The Problem: KV Cache Memory Waste

### What is KV Cache?

When an LLM generates tokens autoregressively, it stores the key (K) and value (V)
vectors for every previous token. This is the **KV cache**. Without it, the model
would need to recompute attention over all previous tokens at every step.

For a 13B parameter model (OPT-13B), a single token's KV cache requires ~800 KB
of GPU memory. A request generating 2048 tokens needs up to 1.6 GB.

### Three Types of Memory Waste (Paper Section 3.1, Figure 3)

Existing systems (before vLLM) store each request's KV cache in a **contiguous**
memory chunk, pre-allocated to the maximum possible sequence length. This causes:

```
|======== Reserved ========|== Internal Frag ==|  External Frag  |
| (future tokens)          | (over-provision)  | (allocator gap) |
```

1. **Reserved** — memory set aside for future tokens that can't be used by others
2. **Internal fragmentation** — the actual sequence is shorter than the max length
3. **External fragmentation** — gaps between differently-sized allocations

The paper's profiling (Figure 2) shows only **20.4%–38.2%** of KV cache memory
actually stores useful token states. The rest is wasted.

### Why Not Just Allocate Exactly What's Needed?

Deep learning frameworks require tensors to be stored in **contiguous memory**.
The KV cache grows dynamically — you don't know the output length in advance.
You can't easily resize a contiguous tensor without copying it.

---

## 2. The OS Analogy: Virtual Memory with Paging

The key insight from the paper (Section 4.2):

> "vLLM uses the ideas behind virtual memory to manage the KV cache in an LLM service."

The analogy is exact:

| OS Virtual Memory | vLLM KV Cache |
|---|---|
| Process | Request |
| Logical pages | Logical KV blocks |
| Physical pages | Physical KV blocks |
| Page table | Block table |
| Page size | Block size (e.g., 16 tokens) |
| Bytes | Tokens |
| Page fault | Block allocation on demand |
| Copy-on-write (fork) | Copy-on-write (parallel sampling) |
| Swapping to disk | Swapping KV cache to CPU RAM |

In an OS, a process sees a contiguous virtual address space, but the physical
pages can be scattered anywhere in RAM. The page table maps virtual → physical.

In vLLM, a request sees a contiguous logical KV cache, but the physical blocks
can be scattered anywhere in GPU memory. The block table maps logical → physical.

---

## 3. PagedAttention: The Attention Algorithm

### The Core Idea (Paper Section 4.1, Figure 5)

Traditional attention computes: `output = softmax(Q · K^T) · V` over all
previous tokens, assuming K and V are stored contiguously.

**PagedAttention** does the same computation but fetches K and V **block by block**,
following the block table. The blocks don't need to be contiguous in physical memory.

```
Query token "forth"
    |
    v
+-------+    +-------+    +-------+
| Block |    | Block |    | Block |
|   0   |    |   1   |    |   2   |
| K,V   |    | K,V   |    | K,V   |
+-------+    +-------+    +-------+
    |            |            |
    v            v            v
  scores[0]   scores[1]   scores[2]
    |            |            |
    +------+-----+-----+------+
           |
           v
    weighted sum of V blocks
           |
           v
        output
```

The attention kernel reads each block's K vectors, computes attention scores,
then reads the V vectors to produce the output — all through the block table
indirection.

### Block Size

The paper studies block size tradeoffs (Section 7.2):
- Larger block → more parallelism in the kernel, lower latency
- Smaller block → less internal fragmentation (waste in the last, partially-filled block)

vLLM typically uses block_size = 16.

---

## 4. The KV Cache Manager: Logical vs Physical Blocks

### Data Structure: KVCacheBlock (kv_cache_utils.py:118)

Each physical block is represented by a `KVCacheBlock` dataclass:

```python
# vllm/v1/core/kv_cache_utils.py:118
@dataclass(slots=True)
class KVCacheBlock:
    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count — how many requests share this block.
    ref_cnt: int = 0
    # The hash key of the block, only available when the block is full and cached.
    _block_hash: BlockHashWithGroupId | None = None
    # Number of prefix tokens covered by _block_hash.
    _block_hash_num_tokens: int | None = None

    # Doubly linked list pointers for the free block queue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    # Whether the block is a null block (placeholder, never cached).
    is_null: bool = False
```

Key fields:
- `block_id` — the physical block index (0 to num_gpu_blocks-1)
- `ref_cnt` — reference count for sharing (like OS page table reference count)
- `_block_hash` — content hash for prefix caching (set when block is full)
- `prev/next_free_block` — linked list pointers for the free list

### Data Structure: FreeKVCacheBlockQueue (kv_cache_utils.py:185)

All free blocks are organized in a **doubly linked list** for O(1) allocation
and removal:

```python
# vllm/v1/core/kv_cache_utils.py:185
class FreeKVCacheBlockQueue:
    """Organizes KVCacheBlock objects to a doubly linked list of free blocks.

    The queue is ordered by block ID initially. When a block is allocated
    and then freed, it is appended back with the eviction order:
    1. The least recently used block is at the front (LRU).
    2. If two blocks have the same last accessed time, the one with more
       hash tokens (tail of a block chain) is at the front.
    """
```

This is a **LRU eviction queue**. The front has the least-recently-used block.
Allocation pops from the front; freeing appends to the back (or front, depending
on whether the block was cached).

### Data Structure: BlockPool (block_pool.py:143)

The `BlockPool` owns all physical blocks and manages allocation:

```python
# vllm/v1/core/block_pool.py:143
class BlockPool:
    def __init__(self, num_gpu_blocks, enable_caching, hash_block_size, ...):
        # All KV-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        # Free block queue (doubly linked list).
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        # Hash table for prefix caching: block_hash -> block
        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
        # Placeholder block (block_id=0) for skipped positions.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
```

### Data Structure: SingleTypeKVCacheManager (single_type_kv_cache_manager.py:36)

Each request's blocks are tracked per attention type:

```python
# vllm/v1/core/single_type_kv_cache_manager.py:36
class SingleTypeKVCacheManager(ABC):
    def __init__(self, kv_cache_spec, block_pool, enable_caching, ...):
        self.block_size = kv_cache_spec.block_size
        self.block_pool = block_pool

        # request_id -> list of KVCacheBlock (the request's block table)
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)

        # request_id -> number of already-cached blocks
        self.num_cached_block: dict[str, int] = {}
```

The `req_to_blocks` dict is the **logical block table** — it maps each request
to its ordered list of physical blocks. This is the in-memory representation of
the block table that gets sent to the GPU.

### Data Structure: KVCacheManager (kv_cache_manager.py:118)

The top-level manager coordinates across attention types:

```python
# vllm/v1/core/kv_cache_manager.py:118
class KVCacheManager:
    def __init__(self, kv_cache_config, max_model_len, ...):
        self.coordinator = get_kv_cache_coordinator(...)
        self.block_pool = self.coordinator.block_pool
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )
```

---

## 5. Block Tables: Mapping Logical to Physical

### The Concept (Paper Section 4.2, Figure 6)

Each request has a **block table** — an array that maps logical block indices
to physical block IDs:

```
Request A's Block Table:
+--------+--------+--------+
| Log 0  | Log 1  | Log 2  |
| → Phy 7| → Phy 1| → Phy 3|
+--------+--------+--------+

Request B's Block Table:
+--------+--------+
| Log 0  | Log 1  |
| → Phy 7| → Phy 5|  (shares block 7 with A!)
+--------+--------+
```

Logical blocks are filled left to right. The last block may be partially filled.
The block table also tracks how many positions are filled in the last block.

### How the Block Table Reaches the GPU

The block table is converted to a tensor and passed as metadata to the attention
kernel:

```python
# vllm/v1/attention/backends/flash_attn.py:293
class FlashAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor       # <-- the block table!
    slot_mapping: torch.Tensor      # <-- where to write new KV in the cache
```

The `block_table` tensor has shape `[num_requests, max_num_blocks_per_request]`.
Each entry is a physical block ID. The attention kernel uses this to look up
where K and V vectors are stored.

The `slot_mapping` tensor maps each new token to its physical slot in the KV
cache: `slot = block_id * block_size + offset_within_block`.

---

## 6. The Block Pool: Allocation, Freeing, and Eviction

### Allocating New Blocks (block_pool.py:647)

```python
# vllm/v1/core/block_pool.py:647
def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    if num_blocks > self.get_num_free_blocks():
        raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

    # Pop from the front of the free list (LRU order)
    ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

    if self.enable_caching:
        for block in ret:
            # If this block was cached, evict it from the hash table
            self._maybe_evict_cached_block(block)
            assert block.ref_cnt == 0
            block.ref_cnt += 1
    else:
        for block in ret:
            assert block.ref_cnt == 0
            block.ref_cnt += 1
    return ret
```

When a block is allocated:
1. Pop from the front of the free list (least recently used)
2. If the block had a cached hash, **evict it** from the prefix cache hash table
3. Set `ref_cnt = 1` (this request now owns it)

### Evicting a Cached Block (block_pool.py:679)

```python
# vllm/v1/core/block_pool.py:679
def _maybe_evict_cached_block(self, block: KVCacheBlock) -> bool:
    if self.metrics_collector:
        self.metrics_collector.on_block_evicted(block)

    evicted_hashes = self._remove_cached_block_hashes(block)
    if not evicted_hashes:
        # The block doesn't have a hash, eviction is not needed
        return False

    self._emit_block_removed_events(evicted_hashes)
    return True
```

Eviction removes the block's hash from `cached_block_hash_to_block`. The physical
memory is reused — the old KV data is simply overwritten.

### Touching Blocks for Prefix Cache Hits (block_pool.py:702)

When a prefix cache hit is found, the block's reference count is incremented
to prevent eviction:

```python
# vllm/v1/core/block_pool.py:702
def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
    for block in blocks:
        # ref_cnt=0 means this block is in the free list, so remove it.
        if block.ref_cnt == 0 and not block.is_null:
            self.free_block_queue.remove(block)
        block.ref_cnt += 1
```

### Freeing Blocks (block_pool.py:719)

```python
# vllm/v1/core/block_pool.py:719
def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
    blocks_to_evict_last = []   # cached blocks → back of free list (FIFO)
    blocks_to_evict_first = []  # uncached blocks → front of free list (LIFO)

    for block in ordered_blocks:
        block.ref_cnt -= 1
        if block.ref_cnt == 0 and not block.is_null:
            if block.block_hash is None or not self.enable_caching:
                # No hash: LIFO reuse for better GPU locality
                blocks_to_evict_first.append(block)
            else:
                # Has hash: FIFO reuse for LRU eviction behavior
                blocks_to_evict_last.append(block)

    # Uncached blocks go to the front (reused first)
    self.free_block_queue.prepend_n(blocks_to_evict_first)
    # Cached blocks go to the back (reused last, preserving cache)
    self.free_block_queue.append_n(blocks_to_evict_last)
```

When a block is freed:
1. Decrement `ref_cnt`
2. If `ref_cnt` reaches 0, the block goes back to the free list
3. **Cached** blocks go to the **back** (FIFO — they stay in the cache longer)
4. **Uncached** blocks go to the **front** (LIFO — reused immediately for locality)

This dual strategy means cached prefix blocks survive longer in the free list,
giving future requests a chance to hit them.

---

## 7. Prefix Caching: Hash-Based Block Reuse

### The Concept (Paper Section 4.4, Figure 10)

When multiple requests share a common prefix (e.g., the same system prompt),
their KV cache for that prefix can be shared. vLLM implements this through
**content-addressed blocks**: each full block is identified by a hash of its
token content.

### Block Hashing (kv_cache_utils.py)

Each block's hash is a **chained hash** — it depends on all previous tokens:

```
hash(block_0) = hash(token_0, token_1, ..., token_{block_size-1})
hash(block_1) = hash(hash(block_0), token_{block_size}, ..., token_{2*block_size-1})
hash(block_2) = hash(hash(block_1), token_{2*block_size}, ...)
```

This means:
- If two requests share the first N blocks, they have the **same hashes** for those blocks
- The hash chain breaks at the first divergence point
- Lookup is O(1) per block via a hash table

### Finding the Longest Cache Hit (single_type_kv_cache_manager.py:684)

```python
# vllm/v1/core/single_type_kv_cache_manager.py:684 (FullAttentionManager)
@classmethod
def find_longest_cache_hit(cls, block_hashes, max_length, kv_cache_group_ids,
                           block_pool, ...):
    block_size = kv_cache_spec.block_size

    computed_blocks = tuple([] for _ in range(len(kv_cache_group_ids)))

    # Phase 1: Walk the hash chain from the start, finding consecutive hits.
    for block_hash in itertools.islice(full_block_hashes, max_length // block_size):
        cached_block = block_pool.get_cached_block(block_hash, kv_cache_group_ids)
        if not cached_block:
            break  # Miss — chain breaks, stop searching
        for computed, cached in zip(computed_blocks, cached_block):
            computed.append(cached)

    hit_length = len(computed_blocks[0]) * block_size

    # Phase 2 (fine-grained only): probe interior hash boundaries
    # for sub-block hits (used when hash_block_size < block_size).
    ...

    return computed_blocks, hit_length
```

The algorithm:
1. Start from block 0, look up its hash in `cached_block_hash_to_block`
2. If found, append to the hit list and continue to the next block
3. If not found, stop — the chain is broken (chained hashes guarantee no later block can match)
4. Return the list of matched blocks and the total hit length

### Caching Full Blocks (block_pool.py:225)

After a block is filled (all `block_size` tokens are computed), it gets cached:

```python
# vllm/v1/core/block_pool.py:225
def cache_full_blocks(self, request, blocks, num_cached_blocks,
                      num_full_blocks, block_size, kv_cache_group_id, ...):
    new_full_blocks = blocks[num_cached_blocks:num_full_blocks]
    block_hashes = resolve_block_hashes(request.block_hashes, ...)

    for i, blk in enumerate(new_full_blocks):
        if blk.is_null:
            continue
        block_hash = new_block_hashes[i]
        block_hash_with_group_id = make_block_hash_with_group_id(
            block_hash, kv_cache_group_id
        )
        # Store the hash in the block and in the hash table
        blk.set_block_hash(block_hash_with_group_id, num_tokens=num_hash_tokens)
        self._insert_block_hash(block_hash_with_group_id, blk, ...)
```

### The Prefix Cache Lookup Flow

```
Request arrives with tokens: [t0, t1, t2, ..., t99]
                              ^^^^^^^^^^^^^^^^^^^^^^^^
                              block_size = 16

Step 1: Compute block hashes
  hash(block_0) = hash(t0..t15)
  hash(block_1) = hash(hash(block_0), t16..t31)
  hash(block_2) = hash(hash(block_1), t32..t47)
  ...

Step 2: find_longest_cache_hit walks the hash chain
  block_0 → HIT (found in cached_block_hash_to_block)
  block_1 → HIT
  block_2 → MISS (stop here)

Step 3: Return [block_0, block_1] as computed blocks
  → 32 tokens of KV cache are reused, no recomputation needed
  → Only tokens 32..99 need to be computed
```

---

## 8. Copy-on-Write: Sharing Blocks Across Sequences

### The Concept (Paper Section 4.4, Figure 8)

When two sequences share a prefix (e.g., parallel sampling from the same prompt),
they share the same physical blocks. The shared blocks have `ref_cnt > 1`.

When one sequence needs to **write** to a shared block (because it's generating
new tokens that go into a partially-filled shared block), vLLM performs
**copy-on-write (CoW)**:

1. Allocate a new physical block
2. Copy the KV data from the shared block to the new block
3. Decrement the shared block's ref_cnt
4. Update the request's block table to point to the new block

### Implementation (single_type_kv_cache_manager.py:402)

```python
# vllm/v1/core/single_type_kv_cache_manager.py:402
def _apply_cow(self, request_id, block_idx, source_block, cow_block):
    """Redirect a partial prefix-cache hit to a private CoW block."""
    req_blocks = self.req_to_blocks[request_id]
    assert block_idx < len(req_blocks)
    assert req_blocks[block_idx] is source_block
    assert not source_block.is_null and source_block.ref_cnt > 0

    # Replace the shared block with the private copy in the block table
    req_blocks[block_idx] = cow_block

    # Record the copy for the worker to execute on GPU
    self._pending_cow_copies.append((source_block, cow_block))

    # The new block gets an extra ref (held until the copy completes)
    cow_block.ref_cnt += 1
```

The actual GPU copy is deferred — the scheduler records `(source, destination)`
pairs, and the worker executes them before the attention kernel runs.

### When CoW is Triggered (single_type_kv_cache_manager.py:327)

```python
# vllm/v1/core/single_type_kv_cache_manager.py:327
def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
    cow_blocks: list[KVCacheBlock] = []
    if request_id in self._partial_hit_reqs:
        # Partial hit: the last prefix-cache hit block is shared and partially
        # filled. Redirect it to a private CoW block.
        block_idx, source_block = self._partial_hit_reqs.pop(request_id)
        cow_block = self.block_pool.get_new_blocks(1)[0]
        self._apply_cow(request_id, block_idx, source_block, cow_block)
        cow_blocks.append(cow_block)

    # Allocate remaining new blocks normally
    req_blocks = self.req_to_blocks[request_id]
    num_required_blocks = cdiv(num_tokens, self.block_size)
    num_new_blocks = num_required_blocks - len(req_blocks)
    if num_new_blocks > 0:
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)
    return cow_blocks + new_blocks
```

CoW is triggered when a prefix cache hit ends **inside** a block (a "partial hit").
The shared block is only partially filled, and the new request needs to write
new tokens into it. Since the block is shared (ref_cnt > 1), writing directly
would corrupt the other sequence's KV cache. So a private copy is made.

---

## 9. The Scheduler: Tying It All Together

### The Main Scheduling Loop (scheduler.py:476)

```python
# vllm/v1/core/sched/scheduler.py:476
def schedule(self, throttle_prefills=False) -> SchedulerOutput:
    # Step 1: Schedule RUNNING requests first
    while req_index < len(self.running) and token_budget > 0:
        request = self.running[req_index]
        num_new_tokens = request.num_tokens_with_spec - request.num_computed_tokens

        # Allocate KV cache slots for the new tokens
        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_new_tokens, ...
        )
        if new_blocks is None:
            # Not enough memory — preempt a request
            preempted_req = self.running.pop()
            self._preempt_request(preempted_req, ...)
            continue

        scheduled_running_reqs.append(request)
        token_budget -= num_new_tokens
        req_index += 1

    # Step 2: Schedule WAITING requests (new arrivals)
    while self.waiting and token_budget > 0:
        request = request_queue.peek_request()

        # Prefix cache lookup
        (new_computed_blocks, num_new_local_computed_tokens, ...) = \
            self._get_local_prefix_cache_hit(request)

        # Allocate slots (reuses cached blocks + allocates new ones)
        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_new_tokens,
            num_new_computed_tokens=num_new_local_computed_tokens,
            new_computed_blocks=new_computed_blocks, ...
        )

        scheduled_new_reqs.append(request)
```

### The Full allocate_slots Flow (kv_cache_manager.py:347)

```python
# vllm/v1/core/kv_cache_manager.py:347
def allocate_slots(self, request, num_new_tokens,
                   num_new_computed_tokens=0, new_computed_blocks=None, ...):
    # 1. Free blocks outside the sliding window (if applicable)
    self.coordinator.remove_skipped_blocks(...)

    # 2. Touch the prefix-cache hit blocks (increment ref_cnt)
    if new_computed_block_list:
        self.coordinator.allocate_new_computed_blocks(
            request_id=request.request_id,
            new_computed_blocks=new_computed_block_list, ...
        )

    # 3. Allocate new blocks for the new tokens
    new_blocks = self.coordinator.allocate_new_blocks(
        request.request_id, num_tokens_need_slot, ...
    )

    # 4. Cache full blocks (compute hashes, store in hash table)
    if self.enable_caching and not delay_cache_blocks:
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

    return self.create_kv_cache_blocks(new_blocks)
```

The `allocate_slots` method does four things in order:
1. **Free** blocks that are no longer needed (sliding window)
2. **Touch** prefix-cache hit blocks (reuse shared blocks)
3. **Allocate** new blocks for new tokens (may trigger CoW)
4. **Cache** full blocks (compute hashes, add to hash table)

---

## 10. The Attention Kernel: How block_table Reaches the GPU

### Building the Metadata (flash_attn.py:382)

The `FlashAttentionMetadataBuilder` constructs the `block_table` tensor from
the per-request block tables:

```python
# vllm/v1/attention/backends/flash_attn.py:597 (simplified)
block_table_tensor = common_attn_metadata.block_table_tensor
slot_mapping = common_attn_metadata.slot_mapping

metadata = FlashAttentionMetadata(
    block_table=block_table_tensor,   # [num_reqs, max_blocks] → physical block IDs
    slot_mapping=slot_mapping,         # [num_tokens] → physical slot indices
    ...
)
```

### The Forward Pass (flash_attn.py:970)

```python
# vllm/v1/attention/backends/flash_attn.py:970
def forward(self, layer, query, key, value, kv_cache, attn_metadata, output):
    # kv_cache shape: [num_blocks, num_kv_heads, block_size, 2 * head_size]
    # Split into key_cache and value_cache
    key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

    block_table = attn_metadata.block_table

    # The flash attention kernel uses block_table to look up KV blocks
    flash_attn_varlen_func(
        q=query[:num_actual_tokens],
        k=key_cache,           # paged KV cache
        v=value_cache,
        out=output[:num_actual_tokens],
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        block_table=block_table,   # <-- passed to the kernel!
        ...
    )
```

### Writing New KV Cache (flash_attn.py:1233)

After attention, new K and V vectors are written into the paged cache using
`slot_mapping`:

```python
# vllm/v1/attention/backends/flash_attn.py:1233
def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
    key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

    # Scatter write: slot_mapping maps each token to its physical slot
    # slot = block_id * block_size + offset_within_block
    reshape_and_cache_flash(
        key, value,
        key_cache, value_cache,
        slot_mapping,           # <-- where to write in the paged cache
        self.kv_cache_dtype,
        layer._k_scale, layer._v_scale,
    )
```

The `slot_mapping` tensor tells the kernel exactly where in the paged KV cache
to write each new token's K and V vectors. It's computed as:

```
slot_mapping[i] = block_table[req_idx, logical_block_idx] * block_size + offset
```

---

## 11. Scheduling and Preemption: Swapping vs Recomputation

### The Problem (Paper Section 4.5)

When GPU memory is full and a new request arrives, vLLM must **preempt** an
existing request. The paper describes two recovery strategies:

**Swapping**: Copy the evicted request's KV cache to CPU RAM. When the request
is rescheduled, copy it back.

**Recomputation**: Simply drop the KV cache. When the request is rescheduled,
recompute the KV cache from scratch (treating all generated tokens as a new
prompt). This is fast because prefill is parallelizable.

### Implementation in the Scheduler

The scheduler uses an **all-or-nothing** eviction policy — either evict all
blocks of a request or none. This is because all blocks of a sequence are
accessed together during attention.

```python
# vllm/v1/core/sched/scheduler.py:640 (simplified)
# When allocate_slots returns None (not enough memory):
if new_blocks is None:
    # Preempt the lowest-priority request
    preempted_req = self.running.pop()
    self._preempt_request(preempted_req, ...)
    # The preempted request's blocks are freed
    # It goes back to the waiting queue for rescheduling
```

In the v1 architecture, recomputation is the default strategy. The preempted
request's blocks are freed, and when it's rescheduled, its KV cache is
recomputed. The prefix cache may still have some of the blocks cached, so
recomputation can be partial.

---

## 12. Complete Request Lifecycle: From Arrival to Completion

### Step-by-Step Walkthrough

```
┌─────────────────────────────────────────────────────────────────────┐
│                    REQUEST LIFECYCLE IN vLLM                        │
└─────────────────────────────────────────────────────────────────────┘

1. REQUEST ARRIVES
   └─→ Added to the waiting queue

2. SCHEDULER PICKS UP THE REQUEST (scheduler.py:751)
   └─→ _get_local_prefix_cache_hit() (scheduler.py:443)
       └─→ KVCacheManager.get_computed_blocks() (kv_cache_manager.py:232)
           └─→ coordinator.find_longest_cache_hit() (kv_cache_coordinator.py:404)
               └─→ FullAttentionManager.find_longest_cache_hit() (single_type_kv_cache_manager.py:684)
                   └─→ Walks block hash chain, returns matched blocks
           └─→ Returns (computed_blocks, num_computed_tokens, ...)

3. ALLOCATE SLOTS (scheduler.py:1033)
   └─→ KVCacheManager.allocate_slots() (kv_cache_manager.py:347)
       ├─→ coordinator.allocate_new_computed_blocks() (kv_cache_coordinator.py:240)
       │   └─→ manager.add_local_computed_blocks() (single_type_kv_cache_manager.py:229)
       │       └─→ block_pool.touch() — increment ref_cnt on cached blocks
       ├─→ coordinator.allocate_new_blocks() (kv_cache_coordinator.py:268)
       │   └─→ manager.allocate_new_blocks() (single_type_kv_cache_manager.py:327)
       │       ├─→ If partial hit: _apply_cow() — copy-on-write
       │       └─→ block_pool.get_new_blocks() — allocate from free list
       └─→ coordinator.cache_blocks() (kv_cache_coordinator.py:303)
           └─→ manager.cache_blocks() (single_type_kv_cache_manager.py:424)
               └─→ block_pool.cache_full_blocks() — hash and store full blocks

4. BUILD METADATA
   └─→ FlashAttentionMetadataBuilder.build()
       └─→ Constructs block_table tensor and slot_mapping tensor

5. GPU WORKER EXECUTES
   ├─→ do_kv_cache_update() — writes new K,V via slot_mapping
   └─→ forward() — attention via block_table

6. REQUEST COMPLETES
   └─→ KVCacheManager.free() (kv_cache_manager.py:570)
       └─→ coordinator.free() (kv_cache_coordinator.py:325)
           └─→ manager.free() (single_type_kv_cache_manager.py:516)
               └─→ block_pool.free_blocks() — decrement ref_cnt, return to free list
```

### Memory Layout Example

```
GPU Memory (Physical KV Blocks):
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
| 0| 1| 2| 3| 4| 5| 6| 7| 8| 9|10|11|12|13|14|15|
+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
   ↑     ↑        ↑  ↑              ↑
   null  Req A    Req A  Req B      Free
         block 0  block 1 block 0   (in free list)
         (shared)         (CoW copy)

Request A Block Table: [7, 1, 3]     (7 tokens, block_size=4)
Request B Block Table: [7, 8]       (shares block 7 with A, then CoW'd to 8)

Block 7: ref_cnt = 2 (shared by A and B)
Block 1: ref_cnt = 1 (A only)
Block 3: ref_cnt = 1 (A only)
Block 8: ref_cnt = 1 (B only, CoW copy of block 1's content)
```

---

## 13. Comparison with SGLang's RadixAttention

| Aspect | vLLM PagedAttention | SGLang RadixAttention |
|---|---|---|
| **Data structure** | Flat block array + hash table | Radix tree (tree of token sequences) |
| **Block size** | Fixed (e.g., 16 tokens) | Variable (token-level granularity) |
| **Prefix matching** | Hash chain lookup (O(1) per block) | Tree traversal (O(matched length)) |
| **Memory unit** | Physical blocks (fixed size) | Tree nodes (variable length) |
| **Sharing mechanism** | ref_cnt on blocks + CoW | lock_ref on tree nodes |
| **Eviction** | LRU free list (doubly linked list) | Heap-based (LRU/LFU/FIFO) |
| **Cache granularity** | Block-level (must fill a full block) | Token-level (any prefix length) |
| **CoW** | Block-level copy | Node split + new child |
| **Block table** | Tensor passed to GPU kernel | req_to_token_pool (2D tensor) |
| **Prefix cache lookup** | find_longest_cache_hit (hash chain) | match_prefix (tree walk) |

### Key Architectural Difference

**vLLM** uses a **hash table** approach: each block is identified by a content
hash. Prefix matching is a hash chain walk — fast (O(1) per block) but requires
blocks to be full before they can be cached. Partial blocks can't be shared.

**SGLang** uses a **radix tree** approach: the tree stores token sequences as
paths. Prefix matching is a tree walk — O(matched length) but can match at any
token boundary, not just block boundaries. This gives finer-grained reuse but
requires tree traversal.

### When Each Shines

- **vLLM's approach** is better when block_size-aligned prefixes are common
  (e.g., system prompts that are multiples of block_size). The hash lookup is
  very fast and the data structure is simple.

- **SGLang's approach** is better when prefixes don't align to block boundaries.
  The radix tree can match any prefix length, avoiding the "last partial block"
  waste that vLLM has.

---

## Appendix: Key File Locations in vLLM v1

| File | Purpose |
|---|---|
| `vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock`, `FreeKVCacheBlockQueue`, block hashing |
| `vllm/v1/core/block_pool.py` | `BlockPool` — allocation, freeing, eviction, prefix caching |
| `vllm/v1/core/single_type_kv_cache_manager.py` | `SingleTypeKVCacheManager` — per-request block tracking, CoW |
| `vllm/v1/core/kv_cache_coordinator.py` | `KVCacheCoordinator` — multi-attention-type coordination |
| `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager` — top-level API for the scheduler |
| `vllm/v1/core/sched/scheduler.py` | `Scheduler` — main scheduling loop |
| `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionImpl` — attention kernel with block_table |
| `vllm/v1/request.py` | `Request` — request data structure with block_hashes |

---

## Appendix: Paper-to-Code Mapping

| Paper Concept | Paper Section | Code Location |
|---|---|---|
| PagedAttention algorithm | §4.1, Fig. 5 | `flash_attn.py:970` (forward with block_table) |
| KV Cache Manager | §4.2 | `kv_cache_manager.py:118`, `block_pool.py:143` |
| Block table | §4.2, Fig. 6 | `FlashAttentionMetadata.block_table` (`flash_attn.py:307`) |
| Logical vs physical blocks | §4.2 | `KVCacheBlock` (`kv_cache_utils.py:118`) |
| Reference counting | §4.4 | `KVCacheBlock.ref_cnt`, `BlockPool.touch()` (`block_pool.py:702`) |
| Copy-on-write | §4.4, Fig. 8 | `SingleTypeKVCacheManager._apply_cow()` (`single_type_kv_cache_manager.py:402`) |
| Shared prefix caching | §4.4, Fig. 10 | `BlockPool.cache_full_blocks()` (`block_pool.py:225`), `find_longest_cache_hit()` (`single_type_kv_cache_manager.py:684`) |
| Preemption (swapping) | §4.5 | `Scheduler._preempt_request()` (scheduler.py) |
| Preemption (recomputation) | §4.5 | Default in v1: free blocks, recompute on reschedule |
| FCFS scheduling | §4.5 | `Scheduler.schedule()` (`scheduler.py:476`) |
| Distributed execution | §4.6 | Centralized scheduler broadcasts block tables to workers |
