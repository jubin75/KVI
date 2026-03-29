## KVI ablation vs GraphRAG (relaxed EM)

- Dataset: **HotpotQA**
- N (limit): **12**
- Methods per run: `graphrag`, `kvi` only

| Variant | GraphRAG EM | KVI EM | KVI − GraphRAG |
|---|---:|---:|---:|
| Full KVI (default DRM / budget) | 58.3 | 41.7 | -16.7 |
| KVI with max_kv_triples=0 (no KV injection) | 58.3 | 58.3 | +0.0 |
| KVI drm_threshold=0.0 (looser triple filter) | 58.3 | 41.7 | -16.7 |
| KVI drm_threshold=0.25 (stricter triple filter) | 58.3 | 41.7 | -16.7 |
| KVI with max_kv_triples=1 (smaller KV budget) | 58.3 | 41.7 | -16.7 |
