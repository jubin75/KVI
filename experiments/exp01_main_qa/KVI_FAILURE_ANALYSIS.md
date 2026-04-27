# Why KVI can score below RAG / GraphRAG (smoke100 observations)

## 1. Retrieval channel inconsistency (confounding factor)

| Method | Retrieval | Generation |
|------|------|------|
| **RAG** | **ANN** (dense) | Evidence in prompt (`modeA_rag`) |
| **GraphRAG** | **Graph + text fallback** | Graph-based reasoning, no KV |
| **KVI** | Graph + text fallback | **KV injection + prompt evidence** (dual-channel) |

Therefore "RAG > KVI" does **not necessarily** mean KV is harmful: it could also be that **ANN retrieval** on synthetic artifacts is better aligned with Hotpot's sentence distribution than **graph retrieval**.

## 2. Prompt domain mismatch (fixed)

`run_graph_inference.py` defaults to a **Chinese medical assistant** system prompt and "common medical knowledge" instructions, while Hotpot is **English open-domain**. When Graph/KVI share this script, they generate under incorrect priors; **KVI's additional KV injection** may amplify attention deflection in the wrong domain.

**Fix**: Enable `--openqa_mode` by default for Exp01 (`run_exp01.py`), using English open-domain instructions and baseline.

## 3. Dual-channel and synthetic graph

- Current triples/graph come from **QA synthetic artifacts**, with weak alignment to real Wiki entities; graph traversal may hit **noisy triples**, and pre-compiled KV entering attention manifests as "deflection."
- The dual-channel design in `docs/01_Article.md` assumes **short KV + no overlap with prompt wording**; this is easily violated on synthetic data.

## 4. Recommended experiment order (aligned with Exp3 / Exp6)

1. **Fix prompt**: Always use `--openqa_mode` before comparing KVI vs RAG.
2. **Ablate KV**: `--max_kv_triples 0` (no KV injection) or disable `--enable_kvi`, for a "graph-only channel" control.
3. **Ablate dual-channel prompt**: `--kvi_minimal_prompt` (remove long evidence list from prompt when injecting KV only), to test whether "pure KV + question" reduces deflection.
4. **Exp3**: Use `gold_supporting_sentences` and `experiments/exp03_retrieval_quality/code/run_exp03_retrieval.py` to report ANN vs Graph Recall@k / MRR.
5. **Exp6**: Tabulate ablations on `run_graph_inference` combinations of `drm_threshold`, `max_kv_triples`, `openqa_mode` (see `EXP03_EXP06.md`).
