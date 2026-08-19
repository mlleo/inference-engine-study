SESSION 3 PROPOSAL: "The Scheduler Engine — How Continuous Batching
    Actually Runs"
    
    This is the natural next step. Session 2 covered RadixAttention (the
    cache) and continuous batching (the concept) and traced the radix tree
    construction. Session 3 should open the hood on the Scheduler itself —
    the engine that decides what runs on the GPU every single iteration.
    
    Here is the detailed breakdown:
    
    ==========================================================
    PART 1: THEORETICAL PART (~40 min)
    ==========================================================
    
    1. The Scheduling Problem in LLM Serving
       - Why scheduling matters: prefill is compute-bound, decode is memory-bound
       - The fundamental tension: TTFT (time to first token) vs throughput
       - Why prefill gets priority over decode (keeps GPU busy with
    compute-heavy work)
       - The three queues: waiting_queue, running_batch, last_batch
    
    2. Scheduling Policies
       - FCFS (default) — First Come First Serve
       - LPM — Longest Prefix Match (cache-aware, maximizes radix tree hits)
       - LOF — Longest Output First
       - RANDOM, DFS-weight, routing-key
       - When to use each: LPM for shared system prompts, FCFS for fairness
       - Trade-offs: LPM starves short-prefix requests under high load
    
    3. Admission Control
       - Why you can't just admit every request: GPU memory is finite
       - Token budget calculation: input tokens + estimated output tokens
       - new_token_ratio: how the system estimates future decode memory
       - batch_is_full: when the system says "stop admitting"
    
    4. Memory Pressure: Eviction vs Retraction
       - Eviction: free cached prefix blocks in the radix tree (doesn't
    affect running requests)
       - Retraction: kick a running request out of the decode batch, free its
    KV cache, put it back in waiting_queue
       - Why eviction is always tried first, retraction is last resort
       - Retracted requests keep their generated tokens and may hit the radix
    tree on re-prefill
    
    5. Chunked Prefill
       - Long prompts (e.g. 32K tokens) would block the GPU for seconds
       - Solution: split into chunks, process one chunk per iteration alongside decode
       - Mixed chunked prefill: prefill + decode tokens in the same batch
       - Dynamic chunking: adapt chunk size based on history
    
    6. Overlap Scheduling (CPU/GPU Pipeline)
       - Normal mode: launch batch → wait for GPU → process results → repeat
       - Overlap mode: launch batch N on GPU, while CPU processes batch N-1 results
       - 1-batch pipeline: hides CPU overhead (tokenization, sampling,
    detokenization) behind GPU compute
       - When overlap is disabled: consecutive prefill batches (to improve
    TTFT of first prefill)
    
    ==========================================================
    PART 2: CODEBASE TUTORIAL (~50 min)
    ==========================================================
    
    All file:line references are from the actual codebase on your branch.
    
    1. The Event Loop — the heartbeat of the scheduler
       File: python/sglang/srt/managers/scheduler.py
    
       - event_loop_normal (line 1520): the simple version
         while True:
           recv_reqs → process_input_requests
           plan = get_next_batch_to_run(running_batch, last_batch)
           if batch: run_batch → process_batch_result
           else: on_idle
           last_batch = batch
    
       - event_loop_overlap (line 1554): the pipelined version
         Show the result_queue deque, pop_and_process pattern
         Show is_disable_overlap_for_batch (line 1627) — when overlap is
    turned off
    
       Key teaching point: the entire server is this one while loop. Every
       "step" is one iteration. Understanding this loop is understanding SGLang.
    
    2. get_next_batch_to_run (line 2687) — the decision function
       - Step 1: Merge last prefill batch into running_batch (line 2739-2764)
         - filter_batch: remove finished requests
         - merge_batch: combine prefill batch into running decode batch
       - Step 2: Try get_new_batch_prefill (line 2779)
         - If a prefill batch is created → run it (prefill has priority)
       - Step 3: If no prefill → update_running_batch → run decode (line 2801)
       - Step 4: If nothing to run → return None (idle)
    
       Trace a concrete example:
       - 3 requests decoding, 2 new requests arrive
       - Show what get_next_batch_to_run returns at each iteration
    
    3. PrefillAdder — admission control (schedule_policy.py:441)
       - The constructor (line 442): rem_total_tokens, new_token_ratio, page_size
       - add_one_req (line ~1001): the per-request admission check
         - Calculate total_tokens = input_len + max_new_tokens +
    page_alignment
         - Check against rem_total_tokens
         - Lock prefix cache node (prevent eviction during prefill)
         - Re-check budget after locking
         - Append to can_run_list or return NO_TOKEN
       - The admission loop (scheduler.py:2955): iterate waiting_queue,
    add_one_req each
    
       Trace a concrete example:
       - waiting_queue = [req_A (500 tokens), req_B (2000 tokens)]
       - rem_total_tokens = 4096
       - Show which requests get admitted, which get deferred
    
    4. Decode Scheduling & Retraction
       - update_running_batch (scheduler.py:3140)
       - check_decode_mem: is there enough KV cache for the next decode step?
       - evict_from_tree_cache: try freeing cached prefixes first
       - retract_decode (schedule_batch.py:2599): the last resort
         - Sort by priority (least-preferred retracted first)
         - Pop requests, free KV, put back in waiting_queue
         - Always keep at least 1 request
       - Show the retraction order logic and how retracted requests resume
    
    5. process_batch_result (scheduler.py:3562)
       - Dispatch by forward_mode: decode vs extend vs idle
       - process_batch_result_prefill (batch_result_processor.py:180):
         - Append output token, check finished
         - If finished: release_kv_cache → cache_finished_req → radix tree insert
         - If not: maybe_cache_unfinished_req
       - process_batch_result_decode (batch_result_processor.py:651):
         - Append output token, check finished
         - If finished: release_kv_cache
    
       This connects back to Session 2: the radix tree insert happens here,
       in process_batch_result, after the forward pass completes.
    
    6. Chunked Prefill in code
       - self.chunked_req: the in-progress chunked request
       - add_chunked_req in PrefillAdder
       - init_next_round_input for the next chunk
       - Dynamic chunk size: predict_next_chunk_size
    
    7. Schedule Policies in code
       - SchedulePolicy class (schedule_policy.py:163)
       - calc_priority: how waiting_queue is reordered
       - Show FCFS vs LPM ordering with a concrete example
    
    ==========================================================
    PART 3: LIVE DEMO (~10 min)
    ==========================================================
    
    - Launch server with --schedule-policy lpm vs default fcfs
    - Send requests with shared prefix, show log output:
      Prefill batch. #new-seq: 5, #new-token: 1234, ...
      Decode batch. #running-reqs: 10, ...
    - Show the scheduler log format and what each field means
    - Demonstrate retraction: overload the server, show requests being
    retracted and re-prefilled
    
    ==========================================================
    KEY FILES REFERENCE TABLE
    ==========================================================
    
    scheduler.py              — Scheduler class, event loop,
    get_next_batch_to_run, run_batch
    schedule_policy.py        — PrefillAdder (admission), SchedulePolicy (policies)
    schedule_batch.py         — ScheduleBatch, filter_batch, merge_batch, retract_decode
    batch_result_processor.py — process_batch_result_prefill/decode
    server_args.py            — schedule_policy default (fcfs),
    chunked_prefill_size defaults
    
    ==========================================================
    WHY THIS IS THE RIGHT SESSION 3
    ==========================================================
    
    Session 1: What is LLM inference + how to run SGLang (the "what")
    Session 2: RadixAttention + continuous batching concept + radix tree (the
    "cache")
    Session 3: The Scheduler engine (the "how" — how batches are formed,
    admitted, scheduled, retracted)
    
    This gives a complete picture: the cache (Session 2) feeds into the
    scheduler (Session 3), which decides what runs on the GPU. After Session
    3, the natural Session 4 would be the forward pass / model worker side —
    how the batch actually flows through the model, attention backends, and
    KV cache memory layout.
