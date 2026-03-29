## KVI ablation vs GraphRAG (relaxed EM)

- Dataset: **HotpotQA**
- N (limit): **8**
- Methods per run: `graphrag`, `kvi` only

| Variant | GraphRAG EM | KVI EM | KVI − GraphRAG |
|---|---:|---:|---:|
| Full KVI (default DRM / budget) | 62.5 | 37.5 | -25.0 |
| KVI + minimal prompt (no evidence block in user prompt) | 62.5 | 12.5 | -50.0 |
| KVI with max_kv_triples=0 (no KV injection) | 62.5 | 62.5 | +0.0 |
