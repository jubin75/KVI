## Experiment 6 — Ablation Study

| Variant | Accuracy (%) | Hallucination (%) | Notes |
|---|---:|---:|---|
| Full KVI | 0.0 |  | graph + KV + prompt (dual-channel), openqa_mode |
| − graph retrieval (ANN-only baseline) |  |  | run modeA_rag |
| − KV injection |  |  | max_kv_triples 0 or GraphRAG |
| − DRM (threshold ablation) |  |  | drm_threshold sweep |
| − dual channel (minimal prompt) |  |  | kvi_minimal_prompt |
