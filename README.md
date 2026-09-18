kv pre-warming
- kv cache를 미리 다른 모델에 prefill 해두어 추후에 요청이 올 때 처리 속도 향상
kv pruning
- trade-off에 따라 동적으로 일부 kv cache만 attention 계산하여 연산량 절감
kv compaction
- trade-off에 따라 kv cach4e 저장 bit 수를 조절
cross-kv sharing
- 여러 모델 간 kv cache sharing을 통한 prefill 절감
