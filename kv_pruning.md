WHY Pruning Causes Almost Zero Quality Degradation
    
    There are five distinct structural reasons, each discovered by a different paper. Together they explain why you can throw away 50-70% of the KV cache and barely notice.
    
    Reason 1: Attention is naturally heavy-tailed (H2O, 2023)
    
    H2O (arXiv:2306.14048) made the foundational observation: at any given decoding step, the attention distribution across all cached positions is extremely concentrated. A tiny
    fraction of tokens receives the vast majority of the attention weight. The rest receive weights close to zero.
    
    Concretely, if you have 10,000 cached tokens, the top 20% might account for 95%+ of the total attention mass. The bottom 80% collectively contribute less than 5% to the output.
    When you compute the weighted sum of values, those bottom 80% are multiplied by near-zero weights, so their contribution to the output vector is negligible. Removing them
    changes the output by less than the floating-point noise already present in the computation.
    
    H2O also found that this heavy-hitter property is persistent — tokens that were heavy hitters in the past tend to remain heavy hitters. So the eviction decision is stable, not
    random.
    
    Reason 2: Softmax creates attention sinks (StreamingLLM + "When Attention Sink Emerges", 2023-2024)
    
    StreamingLLM (arXiv:2309.17453) discovered that the first 1-4 tokens of any sequence receive disproportionately high attention regardless of their semantic content. If you evict
    these "sink" tokens, the model output collapses entirely. If you keep them, you can evict almost everything else.
    
    The paper "When Attention Sink Emerges in Language Models" (Gu et al., 2024, arXiv:2410.10781) explained WHY sinks exist. The root cause is the softmax function. Softmax must
    sum to 1.0 across all positions. When the current query has low similarity to ALL cached keys (which happens frequently), softmax still has to put the weight somewhere. It
    "dumps" the excess attention onto the initial tokens, which become sinks.
    
    The key finding: attention sinks act as "key biases" — they store extra attention scores but do NOT contribute meaningfully to the value computation. They are essentially a
    softmax normalization artifact, not a semantic signal. This means the high attention scores on sink tokens are misleading — they look important but are actually just absorbing
    overflow.
    
    A follow-up paper, "On the Existence and Behavior of Secondary Attention Sinks" (Wong et al., 2025, arXiv:2512.22213), discovered that sinks are not just at position 0. There
    are "secondary sinks" that emerge in middle layers, created by specific MLP modules that map token representations to align with the primary sink direction. These secondary
    sinks persist for variable numbers of layers and draw smaller but still significant attention mass. This means the sink phenomenon is more complex than "keep the first 4 tokens"
    — it's a multi-layer pattern.
    
    And "Attention Sink Forges Native MoE in Attention Layers" (Fu et al., 2026, arXiv:2602.01203) showed that the sink mechanism naturally creates a Mixture-of-Experts structure
    within attention layers, explaining the "head collapse" phenomenon where only a fixed subset of attention heads contributes to generation.
    
    Reason 3: Attention is pyramidal across layers (PyramidKV, 2024)
    
    PyramidKV (arXiv:2406.02069) discovered that attention distribution is layer-dependent in a pyramidal pattern:
    
    - Shallow layers (layers 1-10): attention is spread broadly across many positions. These layers capture syntax and local patterns, needing broad context.
    - Middle layers (layers 10-20): attention starts concentrating on fewer positions.
    - Deep layers (layers 20-32): attention focuses on very few critical tokens (massive activation / attention sinks).
    
    This means a uniform pruning budget across all layers is suboptimal. PyramidKV allocates a decreasing budget — shallow layers keep more KV entries, deep layers keep fewer. By
    matching the natural attention distribution, you evict positions that were already unimportant for that layer. PyramidKV showed that retaining only 12% of KV cache matches
    full-cache performance, and even at 0.7% retention, it outperforms other methods.
    
    Reason 4: Prefill attention predicts generation attention (SnapKV, 2024)
    
    SnapKV (arXiv:2404.14469) discovered that the attention pattern during prefill (when the model processes the input prompt) is a strong predictor of which positions will be
    important during generation. Specifically, the last few tokens of the prompt (the "observation window") attend to specific earlier positions, and those same positions remain important during generation.
    
    This means you can make the pruning decision ONCE, at prefill time, with zero overhead during generation. You look at where the observation window tokens attend, keep those
    positions, evict the rest. The pruned cache is then fixed for the entire generation.
    
    SnapKV achieved 3.6x speedup and 8.2x memory efficiency improvement with "comparable performance to baseline models across 16 long sequence datasets" and "only a negligible
    accuracy drop in the Needle-in-a-Haystack test" even at 380K context tokens.
    
    Reason 5: Pruning can actually IMPROVE quality (Make Each Token Count, 2026)
    
    This is the most surprising finding. "Make Each Token Count" (Bui et al., 2026, arXiv:2605.09649) showed that full-cache attention is NOT always optimal. In long contexts,
    irrelevant tokens dilute attention away from useful evidence. Selective eviction can actually improve generation quality rather than merely approximating the full cache.
    
    The paper provides theoretical analysis showing that preferentially retaining useful tokens reduces attention dilution. Their method "substantially reduces KV memory while
    matching or surpassing full-cache inference" across diverse long-context benchmarks.
    
    This means the zero-degradation claim is not just "we lose almost nothing" — in some cases, pruning actively helps.
    
    The Quantitative Evidence
    
    Here is what the papers actually claim:
    
    - H2O: 20% of KV cache retained, throughput up to 29x, quality "comparable" to full cache
    - StreamingLLM: 4 sink tokens + sliding window, stable generation up to 4 million tokens
    - SnapKV: 12% of KV cache retained, "comparable performance" across 16 datasets, "negligible accuracy drop" on needle-in-haystack at 380K tokens
    - PyramidKV: 12% retained, matches full cache; 0.7% retained, still outperforms other methods
    - Mustafar (arXiv:2505.22913): 70% sparsity (only 30% retained), "without compromising accuracy or requiring fine-tuning"
    - SparK (arXiv:2508.15212): 80% channel pruning, "less than 5% degradation compared to baseline"
    - Near-Oracle KV Selection (arXiv:2602.08329): "under 1% average degradation on LongBench", 9.9x attention speedup
    
    The Latest Famous Papers (2025-2026)
    
    Here are the most important recent papers, organized by what they contribute:
    
    Benchmarking / Survey:
    - Benchmarking KV-Cache Optimizations (Agrawal & Mayer, 2026, arXiv:2607.05399) — The definitive comparison paper. Benchmarks KIVI, TurboQuant, SnapKV, CaM on Llama-3.1-8B and
    Mistral-7B across LongBench tasks. Key finding: "compression ratio alone is a poor predictor of end-to-end performance" and "KIVI4 provides the most stable quality, SnapKV
    delivers the strongest long-context throughput." This is the paper to cite for deployment decisions.
    - How Query Visibility Changes KV-Cache Compression Rankings (Luo et al., 2026, arXiv:2607.11942) — Critical finding: most methods are evaluated query-aware (query visible
    during compression), but in production (cache reuse), compression must be query-agnostic. Under query-agnostic evaluation, SnapKV actually loses to "keep the start and recent
    window." This paper challenges the evaluation methodology of the entire field.
    
    Pruning for Reasoning Models:
    - Value-Aware Stochastic KV Cache Eviction (VaSE) (Chang et al., 2026, arXiv:2606.03928) — Specifically targets reasoning models (o1-style chain-of-thought). Key insight: a
    small fraction of value states have abnormally large magnitudes, and evicting them causes "catastrophic failure where models enter repetitive reasoning loops." VaSE protects
    these large-magnitude values and adds stochasticity for cache diversity. 4x compression with higher accuracy than selection-based methods.
    - ForesightKV (Dong et al., 2026, arXiv:2602.03203) — Uses supervised learning + RL (GRPO) to learn which KV pairs to evict for reasoning models. Introduces "Golden Eviction"
    algorithm using future attention scores. Outperforms prior methods with only half the cache budget on AIME2024/2025.
    - Epiphany-Aware KV Cache Eviction (Kolawole & Smith, 2026, arXiv:2606.26472) — Scores tokens by "epiphany score" (change in model's internal representation) instead of
    attention weights. Works without materializing the attention matrix, compatible with FlashAttention. Scales to 16x longer context than attention-based methods. 72% on MATH-500
    at 4096-token cache.
    
    Advanced Pruning Methods:
    - Mustafar (Joo et al., 2025, arXiv:2505.22913) — Unstructured sparsity up to 70% without fine-tuning. Uses per-token magnitude-based pruning with a custom bitmap-based sparse
    attention kernel. 2.23x throughput improvement.
    - SparK (Liao et al., 2025, arXiv:2508.15212) — Channel-level pruning (not token-level). Prunes KV at the channel dimension, dynamically restores pruned entries during
    attention. Orthogonal to existing token-level methods. 80% pruning with <5% degradation.
    - MosaicKV (Qiang et al., 2026, arXiv:2607.00760) — Dynamic 2D compression (both sequence and channel dimensions). Up to 16x attention speedup, 4.8x lower decode latency, only
    1.76% average accuracy loss on LongBench/RULER.
    - Near-Oracle KV Selection via Pre-hoc Sparsity (Gao et al., 2026, arXiv:2602.08329) — Selects KV entries BEFORE attention scoring (pre-hoc), not after (posterior). Provides
    theoretical mutual-information bound. Under 1% degradation on LongBench, 9.9x attention speedup.
    
    Robustness Improvements:
    - ReST-KV (An et al., 2026, arXiv:2605.08840) — Addresses the problem that existing eviction methods ignore attention redistribution after token removal. Formulates eviction as
    an optimization problem minimizing output discrepancies. 10.61x decoding latency reduction at 128K context.
    - Make Each Token Count (Bui et al., 2026, arXiv:2605.09649) — Global retention-based eviction with learnable retention gates. Shows pruning can IMPROVE quality by reducing
    attention dilution. Matches or surpasses full-cache inference.
    - G-KV (Liao et al., 2025, arXiv:2512.00504) — Global attention scoring combining local and historical attention scores, with RL + distillation post-training. From Microsoft,
    targets reasoning models.
    
    Attention Sink Analysis:
    - When Attention Sink Emerges (Gu et al., 2024, arXiv:2410.10781) — Empirical study of WHY sinks exist. Shows sinks emerge during pretraining, act as key biases, and stem from
    softmax normalization. Sinks disappear with sigmoid attention (no normalization).
    - Secondary Attention Sinks (Wong et al., 2025, arXiv:2512.22213) — Discovers secondary sinks in middle layers created by MLP modules. Identifies 3 sink levels in QwQ-32B, 6 in
    Qwen3-14B.
    - Attention Sink Forges Native MoE (Fu et al., 2026, arXiv:2602.01203) — Shows sinks naturally create MoE structure in attention layers, explaining head collapse.
    
    The Bottom Line
    
    The reason pruning works with near-zero degradation is not magic — it is because:
    
    1. Attention is inherently sparse (heavy-tailed), so most KV entries contribute near-zero to the output
    2. The high-attention positions are predictable (sinks + heavy hitters + prefill patterns)
    3. The importance distribution is layer-dependent (pyramidal), so you can prune aggressively where attention is concentrated
    4. The decision can be made at prefill time with no generation overhead
    5. In some cases, removing irrelevant tokens actually helps by reducing attention dilution
    
    The safe operating point is 30-50% pruning with methods like SnapKV or PyramidKV, which multiple papers have validated as near-zero degradation across diverse tasks. Pushing to
    70% (Mustafar) or 80% (SparK) is possible but requires more careful method selection and task-specific evaluation.
