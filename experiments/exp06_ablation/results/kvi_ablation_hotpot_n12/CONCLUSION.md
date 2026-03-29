## Conclusion (auto)

- **Baseline Full KVI** vs GraphRAG: KVI EM **41.7**, GraphRAG **58.3** (Δ **-16.7** pts).
- **KV off (`max_kv_triples=0`)**: KVI EM **58.3** vs GraphRAG **58.3**. If this tracks GraphRAG closely, injected KV (vs none) is the main differentiator for this run.

**Interpretation:** Positive Δ means KVI beat GraphRAG on this slice; negative means KV path underperformed. Synthetic `build_assets_from_dataset` graphs can distort Hotpot-style multi-hop behavior — treat this as pipeline-level evidence, not official benchmark claims.
