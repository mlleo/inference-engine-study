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
