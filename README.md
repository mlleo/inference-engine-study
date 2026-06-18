# research

## environment
- nvidia h100 gpu & amd mi300x gpu
- pd disaggregation with homogeneous cluster

## Issue in sglang
- sglang does not support pp + speculative decoding -> amd shows better performance in decoding compared to nvidia
- sglang does not support heterogeneous gpu inference in pd disaggregation

## Promblem Statement
- Optimize cross-vendor PD disaggregation for LLM Serving
- Baseline A : H100-only PD, Baseline B : Mi300x-only PD, Proposed : H100 Prefill + Mi300x Decode cross-vendor PD
- H100 prefill gain + Mi300x decode gain > cross-vendor KV transfer cost + format conversion cost(kv layout, dtype conversion etc...) + scheduling overhead
- scheduling overhead
    - router -> H100 PD worker or Mi300x PD worker
    - router -> choose H100 prefill worker and Mi300x decode worker -> reserve kv memory of decode worker -> send metadata to prefill worker -> decide transfer path -> confirm completion of kv transfer -> decode run


## Solution

### Engineering
- implement the actual cross-vendor PD path
  - h100 prefill worker : export kv cache and metadata -> kv transfer bridge : transfer kv -> Mi300x decode worker : import kv cahce -> run decode -> if fail, fallback to h100 only or mi300x only path
- implement transfer path
  - direct RDMA path : H100 GPU -> NIC -> Mi300x GPU
  - pinned host staging path : H100 GPU -> Nvidia host pinned buffer -> network -> AMD host pinned buffer -> Mi300x GPU
  - compressed transfer path : H100 KV -> quantize / pack -> transfer -> dequantize / decode-side consume

### Research
- when cross vendor PD is better than H100-only or Mi300x-only?
- Tcross = Tprefill_h100 + Texport + Ttransfer + Timport + Tdecode_mi300x + Tqueue_cross
- Th100 = Tprefill_h100 + Tdecode_h100 + Tqueue_h100
- Tmi300x = Tprefill_mi300x + Tdecode_mi300x + Tqueue_mi300x
- Tcross < min(Th100, Tmi300x)

- maybe....
- short prompt + long output  → cross-vendor win
- long prompt + short output  → cross-vendor lose
- long prompt + long output   → transfer optimization dependency
- high decode pressure        → MI300X decode should be more used.
- high network congestion     → cross-vendor lose

- TTFT... cross vendor is not good.. -> prefill + kv transfer/import -> then decode starts to run
- TPOT... if output is long, then cross vendor can show better performance.

- Routing scheduling (Given heterogeneous prefill/decode workers, queue states, transfer bandwidth, and SLO constraints, choose a route that maximizes goodput)
  - TTFT-sensitive request -> homogeneous PD
  - TPOT-sensitive / long-generation request -> cross-vendor PD
  - decode queue low + network no pressure -> cross-vendor PD
  - network congested -> homogeneous PD
 
- KV transfer optimization
  - contiguous kv packing (prefill writes kv in export-friendly layout..) -> reduce small tons of copies / pack first then transfer
  - layer-wise pipelined kv transfer (streaming layer-wise kv blocks and start decode before full kv materialization)
    
- Multi-resource scheduling
  - H100 prefill compute
  - Mi300x decode HBM bandwidth
  - network bandwidth
  - host memory / PCIe bandwidth
  - How should an LLM serving router jointly schedule prefill compute, decode memory bandwidth, and cross-vendor KV transport under SLO contraints?
 
## Reference Paper
- Disaggregated Prefill and Decoding Inference System for Large Language Model Serving on Multi-Vendor GPUs
-> multi-vendor based gpu pd disaggregation inference
- FlowKV: Low-Latency KV Cache Transfer and Load-Aware Scheduling
-> kv cache transfer bottleneck : block-wise calling, discontinuout kv cache allocation
- TaiChi: Prefill-Decode Aggregation or Disaggregation? Unifying Both for Goodput-Optimized LLM Serving
-> always pd disaggreagtion good? no, need hybrid scheduling for goodput
