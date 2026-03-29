## Conclusion (auto)

- **Baseline Full KVI** vs GraphRAG: KVI EM **37.5**, GraphRAG **62.5** (Δ **-25.0** pts).
- **KV off (`max_kv_triples=0`)**: KVI EM **62.5** vs GraphRAG **62.5**. If this tracks GraphRAG closely, injected KV (vs none) is the main differentiator for this run.
- **Minimal prompt**: KVI EM **12.5** (vs full KVI **37.5**). Large swing suggests dual-channel prompt+KV interaction matters.

**Interpretation:** Positive Δ means KVI beat GraphRAG on this slice; negative means KV path underperformed. Synthetic `build_assets_from_dataset` graphs can distort Hotpot-style multi-hop behavior — treat this as pipeline-level evidence, not official benchmark claims.
