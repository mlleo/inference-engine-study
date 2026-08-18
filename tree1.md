# SGLang RadixTree Construction: From Initial Request to Followup Request

A code-level walkthrough of how the RadixTree is built and reused in SGLang.
Every claim below references exact file paths and line numbers from the codebase.

---

## 1. What Is the RadixTree and Why Does It Exist?

When an LLM processes a prompt, it computes "KV cache" — intermediate key/value tensors for every token. If a second request shares a prefix with the first (e.g., the same system prompt), we can **reuse** the already-computed KV cache instead of recomputing it. This saves enormous compute.

The **RadixTree** is the data structure that tracks which token sequences have KV cache stored, so we can:

1. **Match**: find the longest shared prefix when a new request arrives (cache hit)
2. **Insert**: store new KV cache after computing it (cache fill)
3. **Evict**: free old KV cache when GPU memory is full (cache replacement)

The tree is a **radix tree** (compressed trie): each node stores a segment of tokens and the KV cache indices for that segment. Shared prefixes are shared nodes.

---

## 2. The Key Data Structures

### 2.1 TreeNode — One Node in the Tree

**File**: `python/sglang/srt/mem_cache/radix_cache.py:217-277`

```python
class TreeNode:
    def __init__(self, id: Optional[int] = None, priority: int = 0):
        self.children = defaultdict(TreeNode)   # child nodes, keyed by first token(s)
        self.parent: TreeNode = None            # back-pointer for lock/evict walks
        self.key: RadixKey = None               # the token IDs this node represents
        self.value: Optional[torch.Tensor] = None  # KV cache indices for these tokens
        self.lock_ref = 0                       # how many in-flight requests use this node
        self.last_access_time = time.monotonic()  # for LRU eviction
        self.creation_time = time.monotonic()     # for FIFO eviction
        self.hit_count = 0                        # for LFU eviction
        self.priority = priority                  # for priority-aware eviction
        # ... (host_value, hash_value, etc. for hierarchical cache)
```

**Key idea**: `key` is the token sequence, `value` is the GPU memory indices pointing to the actual KV cache tensors. `lock_ref` > 0 means a running request is using this node, so it cannot be evicted.

### 2.2 RadixKey — The Token Sequence Wrapper

**File**: `python/sglang/srt/mem_cache/radix_cache.py:60-214`

```python
class RadixKey:
    __slots__ = ("token_ids", "extra_key", "is_bigram", "limit")

    def __init__(self, token_ids, extra_key=None, is_bigram=False, limit=None):
        self.token_ids = token_ids   # the actual token IDs (array("q"))
        self.extra_key = extra_key   # namespace tag (e.g., lora_id) to isolate cache
        self.is_bigram = is_bigram   # EAGLE speculative decoding uses bigram keys
        self.limit = limit           # virtual cap to avoid O(n) slicing
```

Two important methods on RadixKey:

- **`match(other, page_size)`** (line 162): compares two token sequences using exponential search + binary search to find the longest shared prefix length. This is how we determine how much of the tree to traverse.

- **`child_key(page_size)`** (line 198): extracts the first `page_size` tokens as a hashable key (e.g., a tuple like `(5,)` or `(5, 9)`). This is the dictionary key used in `node.children` to find the right child.

### 2.3 RadixCache — The Tree Manager

**File**: `python/sglang/srt/mem_cache/radix_cache.py:280-309`

```python
class RadixCache(SessionRadixCacheMixin, KVCacheEventMixin, BasePrefixCache):
    def __init__(self, params: CacheInitParams):
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size
        self.eviction_strategy = get_eviction_strategy(params.eviction_policy)
        self.evictable_leaves = set()
        self.reset()
```

The `reset()` method (line 331) creates the **root node**:

```python
def reset(self):
    self.root_node = TreeNode(priority=-sys.maxsize)
    self.root_node.key = RadixKey(token_ids=array("q"), extra_key=None)
    self.root_node.value = []
    self.root_node.lock_ref = 1  # root is permanently locked
    self.evictable_size_ = 0
    self.protected_size_ = 0
    self.evictable_leaves.clear()
```

The root node has empty tokens and is permanently locked (`lock_ref = 1`). All real data lives in the root's children.

---

## 3. The Complete Flow: Initial Request

### 3.1 Step 1: Request Arrives at the Scheduler

When a user sends a request, the scheduler's `_add_request_to_queue` puts it into the `waiting_queue`:

**File**: `python/sglang/srt/managers/scheduler.py:2391-2398`

```python
def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
    if not self._set_or_validate_priority(req):
        return
    if self.disaggregation_mode == DisaggregationMode.NULL:
        if self._abort_on_queued_limit(req):
            return
        self._prefetch_kvcache(req)
        self.waiting_queue.append(req)
```

At this point, the request has token IDs but no KV cache has been computed yet.

### 3.2 Step 2: Scheduling — Prefix Match Against the Tree

When the scheduler is ready to run the request, it calls `init_next_round_input`:

**File**: `python/sglang/srt/managers/scheduler.py:2985`

```python
req.init_next_round_input(self.tree_cache)
```

This calls into `Req.init_next_round_input`:

**File**: `python/sglang/srt/managers/schedule_batch.py:1180-1251`

```python
def init_next_round_input(self, tree_cache=None, cow_mamba=None):
    # ... prepare fill IDs ...
    token_ids_to_match = self.full_untruncated_fill_ids

    if tree_cache is not None:
        match_result = tree_cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(
                    token_ids=token_ids_to_match,
                    extra_key=self.extra_key,
                    limit=key_limit,
                ),
                req=self,
                cow_mamba=cow_mamba,
            )
        )
        # Store the matched prefix indices and tree node on the request
        self.prefix_indices = match_result.device_indices
        self.last_node = match_result.last_device_node
        self.last_host_node = match_result.last_host_node
```

**What happens here**: The request's token IDs are wrapped in a `RadixKey` and passed to `tree_cache.match_prefix()`. The result gives us:
- `device_indices`: the GPU KV cache indices for the matched prefix (can be empty if no match)
- `last_device_node`: the tree node where the match ended (we'll lock this to prevent eviction during prefill)

### 3.3 Step 3: match_prefix — Walking the Tree

**File**: `python/sglang/srt/mem_cache/radix_cache.py:355-413`

```python
def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
    key = params.key
    key, _ = key.maybe_to_bigram_view(self.is_eagle)

    if self.disable or len(key) == 0:
        return self._empty_match_result

    key = key.page_aligned(self.page_size)  # truncate to page boundary

    value, last_node = self._match_prefix_helper(self.root_node, key)
    if value:
        value = torch.cat(value)
    else:
        value = self._empty_match_result.device_indices
    return MatchResult(
        device_indices=value,
        last_device_node=last_node,
        last_host_node=last_node,
        best_match_node=last_node,
    )
```

### 3.4 Step 4: _match_prefix_helper — The Core Tree Walk

**File**: `python/sglang/srt/mem_cache/radix_cache.py:650-674`

```python
def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
    access_time = time.monotonic()
    node.last_access_time = access_time

    child_key = key.child_key(self.page_size)  # first token(s) as dict key

    value = []
    while len(key) > 0 and child_key in node.children.keys():
        child = node.children[child_key]
        child.last_access_time = access_time
        prefix_len = child.key.match(key, page_size=self.page_size)

        if prefix_len < len(child.key):
            # Partial match: the key diverges in the middle of this child node.
            # Split the child node at the divergence point.
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            # Full match: this child's entire key is a prefix of our search key.
            # Take its value and continue deeper.
            value.append(child.value)
            node = child
            key = key[prefix_len:]  # advance past the matched portion

            if len(key):
                child_key = key.child_key(self.page_size)  # next child key

    return value, node
```

**Let's break this down for beginners**:

1. Start at the root node.
2. Look at the first token(s) of the search key (`child_key`).
3. Is there a child node under that key? If no, we're done — the match ends here.
4. If yes, compare the child's stored tokens with our search key using `child.key.match(key)`.
- This returns `prefix_len`: how many tokens they share from the start.
5. If `prefix_len < len(child.key)`: the search key diverges in the middle of this child. We **split** the child node into two nodes at the divergence point, take the first half, and stop.
6. If `prefix_len == len(child.key)`: the entire child matches. We collect its KV indices (`child.value`) and advance the search key past this segment, then continue to the next child.
7. Repeat until we can't find a child or run out of key.

### 3.5 Step 5: _split_node — Splitting a Node at a Divergence Point

**File**: `python/sglang/srt/mem_cache/radix_cache.py:676-696`

```python
def _split_node(self, key: RadixKey, child: TreeNode, split_len: int):
    # Before:  parent -> child [tokens A B C D]
    # After:   parent -> new_node [tokens A B] -> child [tokens C D]

    new_node = TreeNode(priority=child.priority)
    new_node.hit_count = child.hit_count
    new_node.children = {key[split_len:].child_key(self.page_size): child}
    new_node.parent = child.parent
    new_node.lock_ref = child.lock_ref
    new_node.key = child.key[:split_len]           # first half
    new_node.value = child.value[:split_len].clone()
    child.parent = new_node
    child.key = child.key[split_len:]              # second half
    child.value = child.value[split_len:].clone()
    new_node.parent.children[key.child_key(self.page_size)] = new_node

    new_node.hash_value, child.hash_value = split_node_hash_value(
        child.hash_value, split_len, self.page_size
    )
    return new_node
```

**Why split?** Imagine the tree has a node storing tokens `[1, 2, 3, 4]`. A new request comes with tokens `[1, 2, 9, 9]`. They share the prefix `[1, 2]` but diverge at position 2. We split the existing node into:
- `[1, 2]` (shared prefix, becomes the new parent)
- `[3, 4]` (the remainder, becomes a child of the new node)

Now the tree can later add `[9, 9]` as another child of `[1, 2]`.

### 3.6 Step 6: After Prefill — Inserting KV Cache into the Tree

After the model finishes prefill (computing KV cache), the scheduler decides what to do:

**File**: `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:238-244`

```python
if req.finished():
    release_kv_cache(req, self.tree_cache)       # request is done → final insert
elif not batch.decoding_reqs or req not in batch.decoding_reqs:
    maybe_cache_unfinished_req(req, self.tree_cache)  # prefill done, will decode → partial insert
```

#### Case A: Request Finished (cache_finished_req)

**File**: `python/sglang/srt/mem_cache/common.py:131-152`

```python
def release_kv_cache(req: Req, tree_cache: BasePrefixCache, is_insert: bool = True):
    effective_kv_committed_len = req.effective_kv_committed_len()
    tree_cache.cache_finished_req(
        req,
        is_insert=is_insert and not getattr(req, "skip_radix_cache_insert", False),
        kv_len_to_handle=effective_kv_committed_len,
    )
```

**File**: `python/sglang/srt/mem_cache/radix_cache.py:437-484`

```python
def cache_finished_req(self, req, is_insert=True, *, kv_len_to_handle):
    token_ids = (req.origin_input_ids + req.output_ids)[:kv_len_to_handle]
    kv_indices = self.req_to_token_pool.req_to_token[
        req.req_pool_idx, : len(token_ids)
    ]

    radix_key = RadixKey(token_ids, req.extra_key, is_bigram=self.is_eagle)
    radix_key = radix_key.page_aligned(self.page_size)
    values = kv_indices[: len(radix_key)].to(dtype=torch.int64, copy=True)

    if is_insert:
        result = self.insert(InsertParams(key=radix_key, value=values, priority=priority))
        session_leaf = result.last_device_node
        # Free duplicate indices that were already in the tree
        self.token_to_kv_pool_allocator.free(
            kv_indices[req.cache_protected_len : result.prefix_len]
        )

    # Free the unaligned tail
    self.token_to_kv_pool_allocator.free(kv_indices[len(radix_key):])

    # Release the lock from the old match point
    if req.last_node is not None:
        self.dec_lock_ref(req.last_node)
```

#### Case B: Request Unfinished (cache_unfinished_req)

**File**: `python/sglang/srt/mem_cache/radix_cache.py:490-556`

```python
def cache_unfinished_req(self, req, chunked=False):
    token_ids = req.get_fill_ids()
    kv_indices = self.req_to_token_pool.req_to_token[
        req.req_pool_idx, : len(token_ids)
    ]

    radix_key = RadixKey(token_ids, req.extra_key, is_bigram=self.is_eagle)
    radix_key = radix_key.page_aligned(self.page_size)
    values = kv_indices[: len(radix_key)].to(dtype=torch.int64, copy=True)

    result = self.insert(InsertParams(key=radix_key, value=values, chunked=chunked, ...))
    new_prefix_len = result.prefix_len

    # Free duplicate indices
    self.token_to_kv_pool_allocator.free(
        kv_indices[req.cache_protected_len : new_prefix_len]
    )

    # Re-match to get updated indices (the tree may have been restructured)
    match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
    new_indices = match_result.device_indices
    new_last_node = match_result.last_device_node

    # Update the request's lock: release old node, lock new node
    self.dec_lock_ref(req.last_node)
    self.inc_lock_ref(new_last_node)

    req.last_node = new_last_node
```

### 3.7 Step 7: insert — The Public Insert API

**File**: `python/sglang/srt/mem_cache/radix_cache.py:415-435`

```python
def insert(self, params: InsertParams) -> InsertResult:
    if self.disable:
        return InsertResult(prefix_len=0)

    key = params.key
    value = params.value
    key, value = key.maybe_to_bigram_view(self.is_eagle, value)
    key = key.page_aligned(self.page_size)
    if value is not None:
        value = value[: len(key)]

    prefix_len, last_node = self._insert_helper(
        self.root_node, key, value, params.priority, params.chunked
    )
    return InsertResult(prefix_len=prefix_len, last_device_node=last_node)
```

### 3.8 Step 8: _insert_helper — The Core Tree Construction

**File**: `python/sglang/srt/mem_cache/radix_cache.py:706-759`

```python
def _insert_helper(self, node, key, value, priority=0, chunked=False):
    access_time = time.monotonic()
    node.last_access_time = access_time
    node.priority = max(node.priority, priority)

    if len(key) == 0:
        return 0, node  # nothing to insert

    child_key = key.child_key(self.page_size)
    total_prefix_length = 0

    # Phase 1: Walk down existing nodes that match
    while len(key) > 0 and child_key in node.children.keys():
        node = node.children[child_key]
        node.last_access_time = access_time
        prefix_len = node.key.match(key, page_size=self.page_size)
        total_prefix_length += prefix_len
        key = key[prefix_len:]       # advance past matched portion
        value = value[prefix_len:]   # advance value too

        if prefix_len < len(node.key):
            # Partial match: split the node, just like in _match_prefix_helper
            new_node = self._split_node(node.key, node, prefix_len)
            new_node.priority = max(new_node.priority, priority)
            self._inc_hit_count(new_node, chunked)
            node = new_node
        else:
            node.priority = max(node.priority, priority)
            self._inc_hit_count(node, chunked)

        if len(key):
            child_key = key.child_key(self.page_size)

    # Phase 2: Create a new leaf node for the remaining unmatched tokens
    if len(key):
        new_node = TreeNode(priority=priority)
        new_node.parent = node
        new_node.key = key
        new_node.value = value.clone()
        self._inc_hit_count(new_node, chunked)
        node.children[child_key] = new_node
        self.evictable_size_ += len(key)
        self._update_leaf_status(node)
        self._update_leaf_status(new_node)
        node = new_node

    return total_prefix_length, node
```

**Let's break this down for beginners**:

**Phase 1 — Walk existing nodes**: This is almost identical to `_match_prefix_helper`. We walk down the tree as long as existing children match our key. If we hit a partial match, we split the node. The difference from `match_prefix` is that we also advance `value` alongside `key` — we're tracking which KV indices go where.

**Phase 2 — Create a new leaf**: After walking as far as existing nodes allow, if there are remaining tokens in the key, we create a brand new `TreeNode` as a child of the last matched node. This new node stores:
- `key`: the remaining unmatched tokens
- `value`: the KV cache indices for those tokens
- `parent`: the last matched node

We also update `evictable_size_` (the total number of tokens that can be evicted) and call `_update_leaf_status` to maintain the `evictable_leaves` set used by the eviction logic.

---

## 4. The Complete Flow: Followup Request

### 4.1 What Happens When a Second Request Shares a Prefix?

When a followup request arrives, it goes through the exact same path:

1. `_add_request_to_queue` → `waiting_queue`
2. Scheduler calls `init_next_round_input` → `match_prefix`
3. `_match_prefix_helper` walks the tree

But this time, the tree already has nodes from the first request. The walk finds the shared prefix and returns the cached KV indices. The request only needs to compute KV cache for the **new** (non-matched) tokens.

### 4.2 Concrete Example

Let's trace a concrete example through the code.

**Initial state**: Empty tree (just root node with `key=[]`, `value=[]`).

#### Request 1: tokens = [1, 2, 3, 4, 5]

**Match phase** (`match_prefix` → `_match_prefix_helper`):
- Start at root. `child_key = (1,)`. Root has no children. Match returns empty.
- `prefix_indices = []`, `last_node = root_node`.

**Prefill**: The model computes KV cache for all 5 tokens. GPU allocates indices, say `[100, 101, 102, 103, 104]`.

**Insert phase** (`cache_finished_req` → `insert` → `_insert_helper`):
- Start at root. `child_key = (1,)`. No child exists.
- Phase 2: Create new node with `key = [1, 2, 3, 4, 5]`, `value = [100, 101, 102, 103, 104]`.
- Tree: `root → node_A{key=[1,2,3,4,5], value=[100,101,102,103,104]}`

#### Request 2: tokens = [1, 2, 3, 8, 9]

**Match phase** (`match_prefix` → `_match_prefix_helper`):
- Start at root. `child_key = (1,)`. Root has child `node_A`.
- `child.key = [1, 2, 3, 4, 5]`. `child.key.match([1,2,3,8,9])` returns `prefix_len = 3` (tokens 1,2,3 match).
- `prefix_len (3) < len(child.key) (5)`: **partial match → split!**
- `_split_node(node_A, split_len=3)`:
- Creates `new_node` with `key = [1, 2, 3]`, `value = [100, 101, 102]`
- `node_A` becomes `key = [4, 5]`, `value = [103, 104]`
- Tree: `root → new_node{[1,2,3]} → node_A{[4,5]}`
- Match returns `value = [100, 101, 102]`, `last_node = new_node`.
- `prefix_indices = [100, 101, 102]` — the GPU KV cache for tokens [1,2,3] is reused!

**Prefill**: Only tokens [8, 9] need new KV cache computation. GPU allocates indices `[200, 201]`.

**Insert phase** (`_insert_helper`):
- Start at root. `child_key = (1,)`. Root has child `new_node{[1,2,3]}`.
- `new_node.key.match([1,2,3,8,9])` returns `prefix_len = 3`. Full match (`3 == 3`).
- Advance: `key = [8, 9]`, `value = [200, 201]`.
- `child_key = (8,)`. `new_node` has no child with key `(8,)`.
- Phase 2: Create `node_B` with `key = [8, 9]`, `value = [200, 201]`.
- Tree:
```
root → new_node{[1,2,3], value=[100,101,102]}
            ├── node_A{[4,5], value=[103,104]}
            └── node_B{[8,9], value=[200,201]}
```

#### Request 3: tokens = [1, 2, 3, 4, 5, 10]

**Match phase**:
- Root → `new_node{[1,2,3]}`: full match (3 == 3).
- `child_key = (4,)`. `new_node` has child `node_A{[4,5]}`.
- `node_A.key.match([4,5,10])` returns `prefix_len = 2`. Full match (2 == 2).
- `child_key = (10,)`. `node_A` has no child with key `(10,)`.
- Match returns `value = [100, 101, 102, 103, 104]`, `last_node = node_A`.
- **All 5 tokens' KV cache is reused!** Only token [10] needs new computation.

---

## 5. Locking: Preventing Eviction During Use

### 5.1 inc_lock_ref — Locking a Path

**File**: `python/sglang/srt/mem_cache/radix_cache.py:594-607`

```python
def inc_lock_ref(self, node: TreeNode) -> IncLockRefResult:
    delta = 0
    while node != self.root_node:
        if node.lock_ref == 0:
            self.evictable_size_ -= len(node.key)
            self.protected_size_ += len(node.key)
            delta -= len(node.key)
        node.lock_ref += 1
        self._update_leaf_status(node)
        node = node.parent  # walk up to root, locking every ancestor
    return IncLockRefResult(delta=delta)
```

When a request starts using a prefix, it calls `inc_lock_ref(last_node)`. This walks **up** from the matched node to the root, incrementing `lock_ref` on every node along the path. Locked nodes (`lock_ref > 0`) cannot be evicted.

### 5.2 dec_lock_ref — Unlocking

**File**: `python/sglang/srt/mem_cache/radix_cache.py:609-628`

```python
def dec_lock_ref(self, node, params=None) -> DecLockRefResult:
    delta = 0
    while node != self.root_node:
        if node.lock_ref == 1:
            self.evictable_size_ += len(node.key)
            self.protected_size_ -= len(node.key)
            delta += len(node.key)
        node.lock_ref -= 1
        self._update_leaf_status(node)
        node = node.parent
    return DecLockRefResult(delta=delta)
```

When the request finishes or moves to a new prefix, it calls `dec_lock_ref(old_node)` to release the lock, making those nodes evictable again.

---

## 6. Eviction: Freeing Memory When Full

### 6.1 The Eviction Strategy

**File**: `python/sglang/srt/mem_cache/evict_policy.py`

SGLang supports multiple eviction policies. Each provides a `get_priority(node)` that returns a sortable key — lower values are evicted first:

| Policy | get_priority returns | Meaning |
|--------|---------------------|---------|
| LRU (default) | `node.last_access_time` | Oldest access evicted first |
| LFU | `(node.hit_count, node.last_access_time)` | Fewest hits evicted first |
| FIFO | `node.creation_time` | Oldest creation evicted first |
| MRU | `-node.last_access_time` | Most recent access evicted first |
| Priority | `(node.priority, node.last_access_time)` | Lower priority evicted first |

### 6.2 The Evict Method

**File**: `python/sglang/srt/mem_cache/radix_cache.py:565-592`

```python
def evict(self, params: EvictParams) -> EvictResult:
    num_tokens = params.num_tokens
    leaves = list(self.evictable_leaves)  # only leaf nodes can be evicted
    eviction_heap = [
        (self.eviction_strategy.get_priority(node), node) for node in leaves
    ]
    heapq.heapify(eviction_heap)

    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)

        self.token_to_kv_pool_allocator.free(x.value)  # free GPU memory
        num_evicted += len(x.value)
        self._delete_leaf(x)  # remove from tree

        # If the parent is now a leaf and unlocked, it becomes evictable
        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
            new_priority = self.eviction_strategy.get_priority(x.parent)
            heapq.heappush(eviction_heap, (new_priority, x.parent))

    return EvictResult(num_tokens_evicted=num_evicted)
```

**Key points**:
- Only **leaf nodes** (nodes with no children) can be evicted. This is tracked by `evictable_leaves`.
- We use a **min-heap** ordered by the eviction strategy's priority.
- After evicting a leaf, if its parent becomes a leaf (no more children) and is unlocked, it becomes evictable too.
- `lock_ref > 0` nodes are never in `evictable_leaves` (enforced by `_update_leaf_status`).

### 6.3 _update_leaf_status — Maintaining the Evictable Set

**File**: `python/sglang/srt/mem_cache/radix_cache.py:790-803`

```python
def _update_leaf_status(self, node: TreeNode):
    if node.evicted or node.lock_ref > 0:
        if node in self.evictable_leaves:
            self.evictable_leaves.remove(node)
        return

    for child in node.children.values():
        if not child.evicted:
            if node in self.evictable_leaves:
                self.evictable_leaves.remove(node)
            return  # has live children → not a leaf

    if node not in self.evictable_leaves:
        self.evictable_leaves.add(node)
```

A node is added to `evictable_leaves` only if:
1. It has a value (not evicted)
2. `lock_ref == 0` (no request is using it)
3. It has no non-evicted children (it's a leaf)

---

## 7. The In-Batch Prefix Cache (waiting_queue_radix_tree)

There's a second, smaller radix tree used for **in-batch prefix caching** — detecting when multiple requests in the waiting queue share a prefix with each other (not just with the main tree).

**File**: `python/sglang/srt/managers/schedule_policy.py:182`

```python
self.waiting_queue_radix_tree = RadixCache.create_simulated()
```

This is a **simulated** tree (no real KV cache, just token tracking). During scheduling:

**File**: `python/sglang/srt/managers/schedule_policy.py:271-308`

```python
for r in waiting_queue:
    prefix_ids = r.origin_input_ids + r.output_ids
    match_result = match_prefix_for_req(self.tree_cache, r, prefix_ids, include_req=True)

    if len(r.prefix_indices) <= IN_BATCH_PREFIX_CACHING_CHECK_THRESHOLD:
        match_result = self.waiting_queue_radix_tree.match_prefix(...)
        in_batch_matching_prefixes = match_result.device_indices
        if len(in_batch_matching_prefixes) >= IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD:
            temporary_deprioritized.add(r.rid)
        else:
            self.waiting_queue_radix_tree.insert(
                InsertParams(key=RadixKey(token_ids=prefix_ids, extra_key=extra_key),
                            value=torch.empty(len(prefix_ids), dtype=torch.bool))
            )
```

This helps the scheduler prioritize requests that share prefixes with each other, so they can be batched together for more efficient prefill.

---

## 8. Complete Call Chain Summary

### 8.1 Initial Request (First Time, No Cache)

```
User sends request
↓
Scheduler._add_request_to_queue()                    [scheduler.py:2391]
→ waiting_queue.append(req)
↓
Scheduler event loop → calc_priority()               [schedule_policy.py:184]
→ match_prefix_for_req(tree_cache, r)               [schedule_policy.py:92]
    → tree_cache.match_prefix(MatchPrefixParams)      [schedule_batch.py:1241]
    → RadixCache.match_prefix()                     [radix_cache.py:355]
        → _match_prefix_helper(root_node, key)        [radix_cache.py:650]
        → walks tree, returns (value=[], last_node=root)
→ req.prefix_indices = []  (no match)
→ req.last_node = root
↓
Scheduler → req.init_next_round_input(tree_cache)     [schedule_batch.py:1180]
→ tree_cache.match_prefix() → same as above
→ inc_lock_ref(req.last_node)  [locks root, no-op]
↓
Model runs prefill → computes KV cache for all tokens
↓
BatchResultProcessor.process_batch_result_prefill()   [batch_result_processor.py:238]
→ if req.finished():
    release_kv_cache(req, tree_cache)               [common.py:131]
        → tree_cache.cache_finished_req(req)           [radix_cache.py:437]
        → self.insert(InsertParams(key, value))     [radix_cache.py:415]
            → _insert_helper(root_node, key, value)    [radix_cache.py:706]
            → Phase 1: no existing children to walk
            → Phase 2: create new TreeNode with key+value
            → Tree: root → node{key, value}
        → dec_lock_ref(req.last_node)  [unlock old match point]
    → elif not finished:
    maybe_cache_unfinished_req(req, tree_cache)     [common.py:98]
        → tree_cache.cache_unfinished_req(req)         [radix_cache.py:490]
        → self.insert(...)  [same as above]
        → self.match_prefix()  [re-match to get updated node]
        → dec_lock_ref(old_node) + inc_lock_ref(new_node)
```

### 8.2 Followup Request (Cache Hit)

```
User sends request with shared prefix
↓
Scheduler._add_request_to_queue()                    [scheduler.py:2391]
→ waiting_queue.append(req)
↓
Scheduler → req.init_next_round_input(tree_cache)     [schedule_batch.py:1180]
→ tree_cache.match_prefix(MatchPrefixParams)        [radix_cache.py:355]
    → _match_prefix_helper(root_node, key)            [radix_cache.py:650]
    → child_key = key.child_key(page_size)          [radix_cache.py:654]
    → while child_key in node.children:
        → child = node.children[child_key]
        → prefix_len = child.key.match(key)         [radix_cache.py:660]
        → if prefix_len < len(child.key):
            → _split_node(child, prefix_len)        [radix_cache.py:676]
            → break
        → else:
            → value.append(child.value)  ← REUSED KV CACHE!
            → key = key[prefix_len:]
            → continue deeper
    → return (value, last_node)
→ req.prefix_indices = [cached KV indices]  ← CACHE HIT!
→ req.last_node = last_node
→ inc_lock_ref(last_node)  [prevent eviction during prefill]
↓
Model runs prefill → only computes KV for NON-matched tokens
↓
BatchResultProcessor → cache_finished_req or cache_unfinished_req
→ insert() → _insert_helper()
    → Phase 1: walks existing nodes (the matched prefix)
    → Phase 2: creates new leaf for the new tokens
→ dec_lock_ref(old_last_node) + inc_lock_ref(new_last_node)
```

---

## 9. Visual Example: Tree Evolution

### After Request 1: tokens = [1, 2, 3, 4, 5]

```
root (key=[], lock_ref=1)
└── node_A (key=[1,2,3,4,5], value=[100,101,102,103,104], lock_ref=0)
```

### After Request 2: tokens = [1, 2, 3, 8, 9]

Match phase splits node_A at position 3:

```
root (key=[], lock_ref=1)
└── node_C (key=[1,2,3], value=[100,101,102], lock_ref=0)
    ├── node_A (key=[4,5], value=[103,104], lock_ref=0)
    └── node_B (key=[8,9], value=[200,201], lock_ref=0)  ← NEW
```

### After Request 3: tokens = [1, 2, 3, 4, 5, 10]

Match phase walks root → node_C (full match) → node_A (full match). No split needed.
Insert phase adds a new child to node_A:

```
root (key=[], lock_ref=1)
└── node_C (key=[1,2,3], value=[100,101,102], lock_ref=0)
    ├── node_A (key=[4,5], value=[103,104], lock_ref=0)
    │    └── node_D (key=[10], value=[300], lock_ref=0)  ← NEW
    └── node_B (key=[8,9], value=[200,201], lock_ref=0)
```

### After Request 4: tokens = [1, 2, 6, 7]

Match phase: root → node_C. `node_C.key = [1,2,3]`. `match([1,2,6,7])` returns 2.
Split node_C at position 2:

```
root (key=[], lock_ref=1)
└── node_E (key=[1,2], value=[100,101], lock_ref=0)  ← SPLIT from node_C
    ├── node_C (key=[3], value=[102], lock_ref=0)   ← REMAINDER
    │    ├── node_A (key=[4,5], value=[103,104])
    │    │    └── node_D (key=[10], value=[300])
    │    └── node_B (key=[8,9], value=[200,201])
    └── node_F (key=[6,7], value=[400,401], lock_ref=0)  ← NEW
```

---

## 10. Key File Reference

| File | What's in it |
|------|-------------|
| `python/sglang/srt/mem_cache/radix_cache.py` | RadixCache, TreeNode, RadixKey — the core tree data structure |
| `python/sglang/srt/mem_cache/base_prefix_cache.py` | BasePrefixCache ABC, MatchPrefixParams, InsertParams, MatchResult, InsertResult |
| `python/sglang/srt/mem_cache/evict_policy.py` | Eviction strategies (LRU, LFU, FIFO, MRU, etc.) |
| `python/sglang/srt/mem_cache/utils.py` | get_eviction_strategy, hash utilities, split_node_hash_value |
| `python/sglang/srt/mem_cache/common.py` | release_kv_cache, maybe_cache_unfinished_req — scheduler-facing helpers |
| `python/sglang/srt/managers/schedule_batch.py:1180` | Req.init_next_round_input — where match_prefix is called per-request |
| `python/sglang/srt/managers/schedule_policy.py:92` | match_prefix_for_req — scheduling-time prefix match |
| `python/sglang/srt/managers/scheduler.py:2391` | _add_request_to_queue — request entry point |
| `python/sglang/srt/managers/scheduler.py:2985` | Scheduler calls init_next_round_input before running |
| `python/sglang/srt/managers/scheduler_components/batch_result_processor.py:238` | Post-prefill: decides cache_finished_req vs cache_unfinished_req |

---

## 11. Summary

The RadixTree in SGLang is built through two fundamental operations:

1. **match_prefix** (read): Walks the tree from the root, comparing token sequences at each node. When a partial match is found, the node is **split** to expose the shared prefix as a separate node. Returns the KV cache indices for the matched prefix.

2. **insert** (write): After prefill, walks the tree the same way (potentially splitting nodes), then creates a **new leaf node** for the unmatched tail tokens. The new node stores the KV cache indices.

The tree grows organically: every request that shares a prefix with an existing node deepens the tree by adding new branches. Every divergence causes a split, creating a shared parent. This is why it's called a **radix tree** — it's a trie where nodes with single children are compressed into multi-token segments, reducing depth and memory overhead.

The locking mechanism (`inc_lock_ref` / `dec_lock_ref`) ensures that nodes being used by in-flight requests are never evicted. The eviction mechanism (heap-based, strategy-driven) frees the least valuable leaf nodes when GPU memory is needed.
