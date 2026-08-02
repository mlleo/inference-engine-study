# Research Collaboration Proposal: KV Cache Optimization for Quality-Cost Trade-off in LLM Serving

---

## 1. Background and Motivation

### 1.1 Current Landscape

Within our organization, we operate a hybrid LLM inference infrastructure that serves both self-hosted open-source models (on internal GPU clusters) and third-party API-based models (e.g., GPT, Claude, Gemini). This dual-track approach provides flexibility but introduces significant cost variability: third-party APIs offer high quality at premium per-token pricing, while self-hosted models reduce marginal cost but require substantial GPU capital expenditure and operational overhead.

To optimize the cost-efficiency of this hybrid setup, we are developing an **auto-routing layer** that dynamically dispatches incoming requests to the most cost-effective model capable of meeting the required quality bar. The router must make real-time decisions considering query complexity, latency requirements, and current system load.

### 1.2 The KV Cache Bottleneck

A critical bottleneck in this architecture is the **Key-Value (KV) cache**, which stores the key and value tensors from the attention mechanism to avoid recomputing them for previously processed tokens. As context windows grow (128K+ tokens) and batch sizes increase, the KV cache becomes the dominant consumer of GPU memory and a primary driver of both prefill latency and infrastructure cost. Specifically:

- **Prefill cost**: Processing a long prompt requires computing KV cache for all tokens, which is compute-bound and dominates Time-To-First-Token (TTFT).
- **Memory cost**: The KV cache for a single request can consume tens of gigabytes, limiting batch sizes and throughput.
- **Redundancy**: Many requests share overlapping context (system prompts, retrieved documents, few-shot examples), yet the KV cache is typically recomputed from scratch for each request.

### 1.3 Why KV Cache Optimization Matters for the Routing Layer

The auto-routing layer's effectiveness depends on minimizing the cost gap between self-hosted and API-based models. KV cache optimization directly impacts this gap in several ways:

1. **Reducing prefill cost** through cache reuse and partial prefill makes self-hosted models more competitive for long-context queries.
2. **Reducing memory footprint** through pruning and compaction increases batch sizes and throughput, lowering per-request cost.
3. **Reducing latency** through pre-warming and disaggregation improves the quality-of-service of self-hosted models, expanding the range of queries that can be routed locally.

### 1.4 Collaboration Rationale

While our team has strong systems engineering and production deployment expertise, we seek to collaborate with [University Lab Name] to leverage their deep expertise in attention mechanism analysis, compression theory, and efficient algorithm design. We believe a joint effort combining industrial-scale infrastructure with academic rigor will produce both practically deployable solutions and publishable scientific contributions.

---

## 2. Research Topics

We propose five interrelated research topics, each addressing a different facet of KV cache optimization. These topics are not fixed — we welcome the university's input on prioritization, refinement, or additional directions.

---

### Topic 1: KV Partial Prefill — Recompute Partial KV Cache Instead of Full Recompute

#### 2.1 Problem Statement

When a new request shares a partially overlapping prefix with a previously processed request (e.g., same system prompt but different user query, or same retrieved document with slight reordering), the standard approach either (a) recomputes the entire KV cache from scratch, or (b) only reuses the exact prefix match. We investigate methods to **reuse KV cache from近似-prefix contexts** by recomputing only the divergent portions, even when the prompt is not 100% identical.

#### 2.2 Related Work

- **SGLang / RadixAttention** (Zheng et al., 2023, arXiv:2312.07104): Introduces a radix tree structure to automatically identify and reuse shared prefixes across requests. This is the foundation of automatic prefix caching (APC) in modern serving systems, but it requires exact token-level prefix matches.

- **vLLM / PagedAttention** (Kwon et al., 2023, arXiv:2309.06180): Proposes paged memory management for KV cache inspired by OS virtual memory, enabling efficient sharing of identical prefix KV cache blocks across requests. Does not address approximate/partial reuse.

- **CacheBlend** (Yao et al., 2024, arXiv:2405.16444): Addresses the case where multiple text chunks are pre-cached independently and then combined in a new request. It identifies that directly concatenating cached KV leads to quality degradation due to cross-chunk attention, and proposes a selective recomputation strategy that re-computes attention only for tokens whose cached KV is most affected by the new context. This is the closest existing work to our vision of partial prefill.

- **KVLink** (Yang et al., 2025, arXiv:2502.16002): Accelerates LLMs by reusing KV cache when different inputs share overlapping context (e.g., same retrieved document in multiple queries). Proposes methods to handle position-ID mismatches when reusing cached KV from different contexts.

- **Attention Store** (Li et al., 2025, arXiv:2503.14647): Proposes an economic framework for reusing stored KV cache across different LLM inputs, analyzing when reuse is beneficial and when recomputation is more economical.

- **CacheClip** (Yang et al., 2025, arXiv:2510.10129): Accelerates RAG by reusing KV cache with a method to handle the fundamental trade-off between prefix reuse and quality degradation from context changes.

- **KV Packet** (Chen et al., 2026, arXiv:2604.13226): Proposes recomputation-free, context-independent KV caching — the idea that KV states can be packaged and reused across different contexts without recomputation, challenging the assumption that KV cache is inherently context-dependent.

- **C2KV** (Du et al., 2026, arXiv:2607.17715): Introduces compressed and composable KV cache reuse, where cached KV from different sources can be combined with minimal recomputation.

- **Decoupled Attention Fusion** (Wu et al., 2026, arXiv:2607.21599): Accelerates RAG by decoupling the attention computation when reusing KV cache, allowing partial reuse with quality preservation.

- **ContiguousKV** (Zou et al., 2026, arXiv:2601.13631): Addresses the practical challenge of loading pre-computed prefix KV cache with granularity-aligned memory management to accelerate prefill.

- **MemServe** (Hu et al., 2024, arXiv:2406.17565): Proposes context caching for disaggregated LLM serving with an elastic memory pool, enabling KV cache reuse across serving instances.

#### 2.3 Proposed Research Directions

1. **Semantic prefix matching**: Go beyond exact token-level prefix matching to identify semantically equivalent prefixes (e.g., same document with minor formatting differences, paraphrased system prompts). Develop a similarity metric for KV cache reuse eligibility that predicts quality impact.

2. **Selective recomputation policy**: Given a partially matching context, determine which KV cache positions must be recomputed to maintain quality. Build on CacheBlend's approach but generalize it to arbitrary overlap patterns (not just chunk boundaries).

3. **Cross-request KV cache deduplication at scale**: Design a system-level cache manager that identifies and exploits KV cache sharing opportunities across a stream of requests in a production serving system, with bounded overhead for cache lookup and management.

4. **Quality-cost curve characterization**: For each partial reuse strategy, empirically map the trade-off between the fraction of KV cache reused and the output quality degradation, enabling the router to make informed decisions.

---

### Topic 2: KV Pre-warming — Prefill KV Cache in Advance to Reduce Latency

#### 2.1 Problem Statement

In many production scenarios, the set of likely upcoming prompts is predictable — e.g., a known set of documents that will be used in RAG, a system prompt template that will be combined with various user queries, or a multi-turn conversation that will continue. We investigate **pre-computing and storing KV cache for anticipated prompts** so that when the actual request arrives, the prefill phase is largely eliminated, dramatically reducing TTFT.

#### 2.2 Related Work

- **Mooncake** (Qin et al., 2024, arXiv:2407.00079): A KVCache-centric disaggregated architecture for LLM serving (deployed for Kimi/Moonshot AI). Separates prefill and decoding clusters and treats KV cache as a first-class resource that can be pre-computed, stored, and transferred. This is the closest production system to our pre-warming vision.

- **DistServe** (Zhong et al., 2024, arXiv:2401.09670): Disaggregates prefill and decoding onto separate GPU clusters to optimize goodput. While focused on resource allocation rather than pre-warming per se, the disaggregated architecture naturally enables pre-computing KV cache on prefill nodes before decoding nodes need it.

- **SmartGen** (Luo et al., 2026, arXiv:2607.28150): Addresses the KV cache transfer bottleneck in disaggregated serving by selectively transferring only the KV cache that is needed, reducing the overhead of pre-warming across nodes.

- **Cross-Family Speculative Prefill** (Upasani et al., 2026, arXiv:2603.02631): Uses small draft models to perform speculative prefill of long contexts, then transfers/refines the KV cache with the larger target model. This is directly relevant to pre-warming with cheaper compute.

- **Keeping the Cache Warm** (Khailo, 2026, arXiv:2607.19214): Analyzes the economics of keeping prefix caches warm for agentic workloads, where follow-up requests systematically share prefixes. Provides a cost-benefit framework for when pre-warming is economically justified.

- **CXL-SpecKV** (Liu & Yu, 2025, arXiv:2512.11920): Proposes a disaggregated FPGA-based speculative KV cache for datacenter LLM serving, using CXL interconnect for fast KV cache access.

- **TraCT** (Yoon et al., 2025, arXiv:2512.18194): Addresses disaggregated LLM serving with CXL shared memory for KV cache at rack-scale, enabling efficient pre-warming across nodes.

- **MemServe** (Hu et al., 2024, arXiv:2406.17565): Context caching for disaggregated LLM serving with an elastic memory pool, enabling pre-computed KV cache to be stored and retrieved on demand.

- **CacheGen** (Liu et al., 2023, arXiv:2310.07240): KV cache compression and streaming for fast LLM serving — directly relevant to the problem of efficiently storing and transferring pre-warmed KV cache.

#### 2.3 Proposed Research Directions

1. **Predictive pre-warming**: Develop models that predict which prompts are likely to arrive in the near future (based on conversation history, RAG document sets, application patterns) and pre-compute their KV cache during idle GPU cycles.

2. **Tiered KV cache storage**: Design a storage hierarchy for pre-warmed KV cache — GPU memory for imminent use, CPU memory for near-term, SSD/compressed storage for longer-term — with efficient promotion/demotion policies.

3. **Speculative prefill with small models**: Use a small, fast model to perform initial prefill, then transfer the KV cache (or a transformed version) to the larger target model. Investigate the quality implications and the transformation needed to make cross-model KV cache transfer viable.

4. **Pre-warming scheduling**: Given a budget of idle GPU cycles and a set of candidate prompts to pre-warm, develop a scheduling algorithm that maximizes the expected latency reduction (considering prompt arrival probability, cache hit rate, and storage cost).

5. **Integration with disaggregated serving**: Explore how pre-warming interacts with prefill-decode disaggregation architectures (Mooncake, DistServe), particularly the KV cache transfer overhead between prefill and decoding nodes.

---

### Topic 3: KV Pruning — Dynamically Compute Partial Attention for KV Cache

#### 2.1 Problem Statement

Not all tokens in the KV cache contribute equally to the output. Many tokens receive negligible attention weights and can be safely evicted without significant quality loss. We investigate **dynamic KV cache pruning** methods that identify and retain the most important KV entries while evicting the rest, trading a controlled amount of quality for reduced memory and computation.

#### 2.2 Related Work

- **H2O (Heavy-Hitter Oracle)** (Zhang et al., 2023, arXiv:2306.14048): A foundational work that observes a "heavy-hitter" phenomenon — a small set of tokens consistently receive high attention scores across layers and steps. H2O retains these heavy hitters and evicts the rest, achieving significant KV cache reduction with minimal quality loss. 844 citations as of 2026.

- **StreamingLLM / Attention Sinks** (Xiao et al., 2023, arXiv:2309.17453): Discovers that the initial tokens in a sequence act as "attention sinks" — they receive disproportionate attention regardless of their semantic relevance. StreamingLLM retains these sink tokens plus a sliding window of recent tokens, enabling efficient streaming inference.

- **SnapKV** (Li et al., 2024, arXiv:2404.14469): Observes that the KV cache positions that will be important during generation can be predicted from a small "observation window" of prefill attention patterns. SnapKV compresses the KV cache before generation based on these observations, achieving strong quality with minimal overhead.

- **PyramidKV** (Cai et al., 2024, arXiv:2406.02069): Discovers that attention is "pyramidal" — deeper layers concentrate attention on fewer tokens, while shallower layers attend more broadly. PyramidKV allocates a decreasing KV cache budget across layers, matching this pyramidal information flow.

- **InfiniGen** (Lee et al., 2024, arXiv:2406.19707): Efficient generative inference with dynamic KV cache management. Uses attention prediction to proactively manage which KV entries to keep, enabling long-context generation with bounded memory.

- **RetrievalAttention** (Liu et al., 2024, arXiv:2409.10516): Accelerates long-context inference by treating KV cache as a vector database and retrieving only the most relevant KV entries on-the-fly during attention computation, rather than attending over the full cache.

- **Mustafar** (Joo et al., 2025, arXiv:2505.22913): Demonstrates that unstructured sparsity significantly improves KV cache compression, enabling sparsity levels up to 70% without compromising accuracy or requiring fine-tuning. Systematically explores pruning strategies.

- **ReST-KV** (An et al., 2026, arXiv:2605.08840): Robust KV cache eviction with layer-wise output reconstruction and spatial-temporal smoothing, addressing the instability of attention-score-based eviction methods.

- **Make Each Token Count** (Bui et al., 2026, arXiv:2605.09649): Improves long-context performance with KV cache eviction, focusing on reducing the quality degradation that existing eviction methods cause relative to full-cache inference.

- **UniCAIM** (Xu et al., 2025, arXiv:2504.07479): A unified CAM/CIM architecture with static-dynamic KV cache pruning for efficient long-context LLM inference, combining hardware-aware design with algorithmic pruning.

- **Near-Oracle KV Selection** (Gao et al., 2026, arXiv:2602.08329): Achieves near-oracle top-k KV selection quality using pre-hoc sparsity, preserving dense attention quality while sharply reducing computation.

- **vAttention** (Prabhu et al., 2024, arXiv:2405.04437): Dynamic memory management for serving LLMs without PagedAttention, using virtual memory for efficient sparse attention. Also: vAttention: Verified Sparse Attention (Desai et al., 2025, arXiv:2510.05688) for reducing decoding latency.

#### 2.3 Proposed Research Directions

1. **Query-aware dynamic pruning**: Most existing methods prune the KV cache based on prefill attention patterns and keep the pruning fixed during generation. We propose dynamically adjusting the pruning based on the current decoding step's query, retaining different KV entries for different generated tokens.

2. **Layer-adaptive budget allocation**: Extend PyramidKV's insight with a learned policy that allocates KV cache budgets per layer based on the specific input, rather than using a fixed pyramid structure. Some inputs may need more cache in certain layers.

3. **Pruning-aware routing**: Integrate KV pruning into the auto-routing layer — when a request is routed to a self-hosted model, the router can specify the pruning aggressiveness based on the quality requirement, enabling a continuous quality-cost knob.

4. **Quality preservation analysis**: Conduct a systematic study of how different pruning methods affect various task types (reasoning, code generation, summarization, RAG), identifying which tasks are most sensitive to KV cache pruning and developing task-specific pruning policies.

5. **Hardware-aware pruning**: Co-design pruning algorithms with GPU memory hierarchy, ensuring that pruned attention computation maps efficiently to tensor core operations and memory access patterns.

---

### Topic 4: KV Compaction — Control the Bit-width of KV Cache as a Trade-off

#### 2.1 Problem Statement

KV cache is typically stored in FP16 or BF16 (16 bits per element). However, not all KV cache entries require full precision — some can be represented in 8-bit, 4-bit, 2-bit, or even 1-bit with minimal quality loss. We investigate **adaptive KV cache quantization** methods that dynamically control the bit-width of KV cache entries, using high precision for critical entries and low precision for less important ones, achieving a tunable quality-cost trade-off.

#### 2.2 Related Work

- **KIVI** (Liu et al., 2024, arXiv:2402.02750): A tuning-free asymmetric 2-bit quantization for KV cache. Key insight: keys should be quantized per-channel and values per-token, exploiting their different distributional properties. Achieves 2-bit KV cache with negligible quality loss.

- **KVQuant** (Hooper et al., 2024, arXiv:2401.18079): Towards 10 million context length LLM inference with KV cache quantization. Introduces several innovations including pre-RoPE key quantization, non-uniform quantization, and sparse attention score preservation, enabling sub-2-bit KV cache.

- **KV Cache is 1 Bit Per Channel** (Zhang et al., 2024, arXiv:2405.03917): Demonstrates that KV cache can be quantized to as low as 1 bit per channel using coupled quantization, pushing the extreme of KV cache compression.

- **CommVQ** (Li et al., 2025, arXiv:2506.18879): Commutative vector quantization for KV cache compression. Uses vector quantization codebooks to achieve flexible bit-widths, with a commutative property that enables efficient dequantization fused with attention.

- **XQuant** (Yang et al., 2025, arXiv:2510.11236): Achieves ultra-low bit KV cache quantization with cross-layer compression, exploiting inter-layer KV cache similarity to further reduce bit-width.

- **VecInfer** (Yao et al., 2025, arXiv:2510.06175): Efficient LLM inference with low-bit KV cache via outlier-suppressed vector quantization, addressing the outlier channels that plague low-bit quantization.

- **AnTKV** (Li et al., 2025, arXiv:2506.19505): Anchor token-aware sub-bit vector quantization for KV cache, identifying "anchor tokens" whose KV cache requires higher precision and quantizing the rest more aggressively.

- **KVSink** (Su & Yuan, 2025, arXiv:2508.04257): Studies how attention sinks should be preserved during KV cache quantization, showing that protecting sink tokens from aggressive quantization is critical for quality.

- **Plug-and-Play 1.x-Bit** (Tao et al., 2025, arXiv:2503.16257): Achieves 1.x-bit KV cache quantization for video LLMs, demonstrating that sub-2-bit is feasible even for multimodal models.

- **VQ-LLM** (Liu et al., 2025, arXiv:2503.02236): High-performance code generation for vector quantization augmented LLM inference, providing efficient GPU kernels for VQ-based KV cache.

- **WKVQuant** (Yue et al., 2024, arXiv:2402.12065): Joint quantization of both weights and KV cache, considering their interaction for optimal overall compression.

- **QServe** (Lin et al., 2024, arXiv:2405.04532): W4A8KV4 quantization and system co-design for efficient LLM serving, demonstrating a practical 4-bit KV cache deployment.

- **KV-CAR** (Roy et al., 2025, arXiv:2512.06727): Uses autoencoders to compress KV cache, learning a compact representation that can be decoded on-the-fly during attention.

#### 2.3 Proposed Research Directions

1. **Mixed-precision adaptive quantization**: Develop a method that assigns different bit-widths to different KV cache positions based on their importance (measured by attention scores, token type, or learned metrics). Critical positions get 8-bit, moderate positions get 4-bit, unimportant positions get 2-bit — all within the same request.

2. **Dynamic bit-width control via the router**: Integrate KV cache quantization level as a routing parameter. The router selects not just which model handles the request, but also the KV cache precision level, creating a two-dimensional quality-cost trade-off space.

3. **Quantization-aware prefill**: Co-design the prefill computation with the target quantization level. If the KV cache will be stored in 2-bit, the prefill computation can be optimized to produce KV values that are more amenable to low-bit quantization (e.g., by reducing outlier channels).

4. **Cross-method benchmarking**: Conduct a comprehensive empirical study comparing all major KV cache quantization methods (KIVI, KVQuant, CommVQ, etc.) under identical conditions (same models, same tasks, same hardware), producing a reproducible benchmark for the community.

5. **Interaction with pruning**: Study how KV cache quantization interacts with pruning — can they be composed for multiplicative savings? What is the optimal operating point when both are applied?

---

### Topic 5: Cross-KV Sharing — Share KV Cache Between Multiple Models

#### 2.1 Problem Statement

In a multi-model serving environment (as in our auto-routing setup), different models may process the same or similar prompts. Currently, each model maintains its own independent KV cache, leading to redundant prefill computation. We investigate methods to **share or transfer KV cache across models**, reducing total prefill cost. This is the most ambitious and least-explored topic, with significant research potential.

#### 2.2 Related Work

- **GQA (Grouped Query Attention)** (Ainslie et al., 2023, arXiv:2305.13245): Shares KV heads across query heads within a single model, reducing KV cache size by a factor of the group size. This is an architectural-level KV sharing that inspired cross-layer and cross-model approaches.

- **MLA (Multi-head Latent Attention)** (DeepSeek-V2, 2024): Projects KV into a compact latent space, drastically reducing KV cache size. The latent representation is decoded back to full KV during attention. This architectural innovation is directly relevant to cross-model sharing, as the latent representation may be more transferable than raw KV.

  - **MLA Hardware Analysis** (Geens & Verhelst, 2025, arXiv:2506.02523): Hardware-centric analysis of DeepSeek's MLA, providing insights into the efficiency and transferability of latent KV representations.

  - **YouZhi** (2026, arXiv:2606.05868): Adaptive GQA-to-MLA transition for high-concurrency financial LLMs, demonstrating practical migration between KV sharing architectures.

- **KVSharer** (Yang et al., 2024, arXiv:2410.18517): Shares KV cache across layers within a single model by identifying layer-wise dissimilarity — layers with similar attention patterns can share KV cache. This is the first explicit "KV sharing" work, though within a single model.

- **CommonKV** (Wang et al., 2025, arXiv:2508.16134): Compresses KV cache with cross-layer parameter sharing, extending KVSharer's approach with compression to further reduce memory.

- **PolyKV** (Patel & Joshi, 2026, arXiv:2604.24971): A shared asymmetrically-compressed KV cache pool for multi-agent LLM inference. Multiple concurrent inference agents share a single compressed KV cache pool, which is the closest existing work to cross-model KV sharing (though it assumes the same model architecture).

- **CacheGen** (Liu et al., 2023, arXiv:2310.07240): KV cache compression and streaming — relevant to cross-model sharing because compressed KV representations may be more architecture-agnostic.

- **Cross-Family Speculative Prefill** (Upasani et al., 2026, arXiv:2603.02631): Uses small draft models to perform speculative prefill and then transfers to larger target models. This is the closest work to cross-model KV transfer, though it focuses on same-family models (draft + target) rather than truly heterogeneous models.

#### 2.3 Proposed Research Directions

1. **Cross-model KV cache transformation**: Investigate whether KV cache from one model can be transformed (via a learned projection, adapter, or encoder-decoder) to be usable by another model. This is the core research question — is there a model-agnostic intermediate representation for KV cache?

2. **Shared latent KV space**: Inspired by MLA, explore whether multiple models can be trained or fine-tuned to project their KV into a shared latent space, enabling direct KV cache sharing. This would require coordinated training but could yield the largest savings.

3. **Model family KV transfer**: Start with models in the same family (e.g., Llama-3-8B and Llama-3-70B) that share architectural similarities, and investigate how much KV cache can be directly transferred vs. needs recomputation. Gradually extend to more heterogeneous model pairs.

4. **KV cache distillation**: Train a small "KV translator" network that takes KV cache from model A and produces approximate KV cache for model B. The translator could be trained on paired data where both models process the same prompts.

5. **Economic analysis of sharing**: Develop a cost model for when cross-model KV sharing is worthwhile — the overhead of transformation/translation vs. the savings from avoided prefill. This analysis will guide which model pairs in the routing system should enable sharing.

6. **Multi-model cache pool architecture**: Design a system architecture that maintains a shared KV cache pool across multiple models, with per-model metadata indicating which cached entries are reusable and which require recomputation.

---

## 3. Proposed Collaboration Structure

### 3.1 Phased Approach

**Phase 1 (Months 1-3): Benchmarking and Foundation**
- Set up a unified evaluation framework across all five topics
- Reproduce key existing methods (H2O, SnapKV, KIVI, CacheBlend, etc.) on shared model suite
- Identify quality-cost Pareto frontiers for each method
- Deliverable: Comprehensive benchmark report + reusable evaluation codebase

**Phase 2 (Months 4-9): Core Research**
- Focus on 2-3 highest-priority topics (to be jointly selected)
- Develop and validate new methods
- Conduct ablation studies and quality analysis
- Deliverable: Method papers + prototype implementations

**Phase 3 (Months 10-12): Integration and Production**
- Integrate promising methods into our production serving system
- Conduct end-to-end quality-cost evaluation under real workloads
- Prepare joint publications
- Deliverable: Production deployment + publications

### 3.2 Resource Contributions

**Industry Side (Our Team):**
- GPU compute resources for experiments and benchmarking
- Production-scale workload traces and evaluation data
- Engineering support for system integration
- Access to the auto-routing infrastructure for end-to-end evaluation

**University Side:**
- Algorithmic and theoretical innovation
- Rigorous experimental design and analysis
- Student researchers for implementation and experimentation
- Publication writing and academic dissemination

### 3.3 Expected Outcomes

1. Open-source benchmark suite for KV cache optimization methods
2. 2-3 top-tier publications (MLSys, OSDI, NeurIPS, ICML, or similar)
3. Production-deployed KV cache optimizations in our serving system
4. Patent filings for novel methods (jointly owned)

---

## 4. Evaluation Plan

### 4.1 Models
- Open-source: Llama-3 (8B/70B), Qwen-2.5 (7B/72B), Mistral (7B/Mixtral)
- Target: DeepSeek-V2 (with MLA) for cross-model sharing experiments

### 4.2 Tasks
- Long-context QA (Needle-in-haystack, LongBench)
- RAG (natural questions, multi-document reasoning)
- Code generation (HumanEval, MBPP)
- Multi-turn conversation
- Reasoning (GSM8K, MATH)

### 4.3 Metrics
- Quality: Task accuracy, BLEU/ROUGE, pass@k, human evaluation
- Cost: GPU-seconds per request, $/1M tokens
- Latency: TTFT, TPOT (time per output token), end-to-end latency
- Memory: Peak KV cache memory, GPU memory utilization
- System: Throughput (tokens/sec), max batch size, goodput

---

## 5. Key References Summary

| Topic | Paper | arXiv ID | Year |
|-------|-------|----------|------|
| **Partial Prefill** | SGLang / RadixAttention | 2312.07104 | 2023 |
| | vLLM / PagedAttention | 2309.06180 | 2023 |
| | CacheBlend | 2405.16444 | 2024 |
| | KVLink | 2502.16002 | 2025 |
| | Attention Store | 2503.14647 | 2025 |
| | CacheClip | 2510.10129 | 2025 |
| | KV Packet | 2604.13226 | 2026 |
| | C2KV | 2607.17715 | 2026 |
| | Decoupled Attention Fusion | 2607.21599 | 2026 |
| | ContiguousKV | 2601.13631 | 2026 |
| | MemServe | 2406.17565 | 2024 |
| **Pre-warming** | Mooncake | 2407.00079 | 2024 |
| | DistServe | 2401.09670 | 2024 |
| | SmartGen | 2607.28150 | 2026 |
| | Cross-Family Speculative Prefill | 2603.02631 | 2026 |
| | Keeping the Cache Warm | 2607.19214 | 2026 |
| | CXL-SpecKV | 2512.11920 | 2025 |
| | TraCT | 2512.18194 | 2025 |
| | CacheGen | 2310.07240 | 2023 |
| **Pruning** | H2O | 2306.14048 | 2023 |
| | StreamingLLM | 2309.17453 | 2023 |
| | SnapKV | 2404.14469 | 2024 |
| | PyramidKV | 2406.02069 | 2024 |
| | InfiniGen | 2406.19707 | 2024 |
| | RetrievalAttention | 2409.10516 | 2024 |
| | Mustafar | 2505.22913 | 2025 |
| | ReST-KV | 2605.08840 | 2026 |
| | Make Each Token Count | 2605.09649 | 2026 |
| | UniCAIM | 2504.07479 | 2025 |
| | Near-Oracle KV Selection | 2602.08329 | 2026 |
| | vAttention | 2405.04437 | 2024 |
| **Compaction** | KIVI | 2402.02750 | 2024 |
| | KVQuant | 2401.18079 | 2024 |
| | KV Cache 1 Bit/Channel | 2405.03917 | 2024 |
| | CommVQ | 2506.18879 | 2025 |
| | XQuant | 2510.11236 | 2025 |
| | VecInfer | 2510.06175 | 2025 |
| | AnTKV | 2506.19505 | 2025 |
| | KVSink | 2508.04257 | 2025 |
| | Plug-and-Play 1.x-Bit | 2503.16257 | 2025 |
| | VQ-LLM | 2503.02236 | 2025 |
| | WKVQuant | 2402.12065 | 2024 |
| | QServe | 2405.04532 | 2024 |
| | KV-CAR | 2512.06727 | 2025 |
| **Cross-Model Sharing** | GQA | 2305.13245 | 2023 |
| | MLA (DeepSeek-V2) | — | 2024 |
| | MLA Hardware Analysis | 2506.02523 | 2025 |
| | YouZhi (GQA-to-MLA) | 2606.05868 | 2026 |
| | KVSharer | 2410.18517 | 2024 |
| | CommonKV | 2508.16134 | 2025 |
| | PolyKV | 2604.24971 | 2026 |
| **LLM Routing** | FrugalGPT | 2305.05176 | 2023 |
| | Router-R1 | 2506.09033 | 2025 |
| | R2-Router | 2602.02823 | 2026 |
| | When Routing Collapses | 2602.03478 | 2026 |
