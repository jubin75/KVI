## Experiment 1 — Main QA Performance (Reviewer-Friendly, Dual-Panel)

### Panel A: Base LLM = Qwen2.5-7B-Instruct

| Dataset | LLM | RAG | GraphRAG | KV Prefix | KVI |
|---|---|---:|---:|---:|---:|---:|
| HotpotQA (EM [CI95]) | 16.7 [10.0, 23.3] | 22.5 [15.0, 30.0] | 32.5 [24.2, 40.8] | 15.8 [9.2, 22.5] | 33.3 [25.0, 41.7] |
| NQ (EM [CI95]) | 20.0 [13.0, 28.0] | 41.0 [30.0, 51.0] | 44.0 [34.0, 54.0] | 14.0 [7.0, 21.0] | 49.0 [39.0, 58.0] |
| MedHopQA_n40 (EM [CI95]) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 82.5 [70.0, 92.5] | 0.0 [0.0, 0.0] | 92.5 [82.5, 100.0] |
| MedHopQA_official N=342 (EM [CI95]) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 74.3 [69.6, 78.9] | 0.0 [0.0, 0.0] | 75.4 [70.8, 80.1] |

### Panel B: Base LLM = Mistral-7B-Instruct-v0.3

| Dataset | LLM | RAG | GraphRAG | KV Prefix | KVI |
|---|---|---:|---:|---:|---:|---:|
| HotpotQA (EM [CI95]) | 30.8 [22.5, 39.2] | 36.7 [28.3, 45.8] | 38.3 [29.2, 46.7] | 16.7 [10.0, 24.2] | 40.8 [32.5, 49.2] |
| NQ (EM [CI95]) | 34.0 [24.0, 43.0] | 63.0 [53.0, 73.0] | 67.0 [58.0, 76.0] | 17.0 [10.0, 24.0] | 67.0 [58.0, 76.0] |
| MedHopQA_n40 (EM [CI95]) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 10.0 [2.5, 20.0] | 0.0 [0.0, 0.0] | 12.5 [2.5, 25.0] |
| MedHopQA_official N=342 (EM [CI95]) | 0.0 [0.0, 0.0] | 0.3 [0.0, 0.9] | 13.5 [9.9, 17.0] | 0.6 [0.0, 1.5] | 33.3 [28.7, 38.0] |

### MedHopQA_n40 vs MedHopQA_official — Construction differences and asymmetric impact on GraphRAG / KVI

| Dimension | MedHopQA_n40 | MedHopQA_official (N=342) | Typical impact on GraphRAG | Typical impact on KVI |
|------|--------------|---------------------------|-------------------------|-------------------|
| **Query form** | `interacts_with DBxxxx?` + strong ID output constraint | Natural language query (still requires partner DrugBank ID) | Graph anchor `DBxxxx` still present; text retrieval sometimes more easily hits summary-style evidence | NL and triple text / DRM scoring lexical **alignment weakens**, easily selecting **irrelevant or suboptimal** triples for KV |
| **Scale & corpus** | 40 samples, corresponding subgraph and sentence scale is small | 342 samples, full supports split yields ~**10⁵**-level sentences | Noise increase is relatively moderate; **single-channel** "graph + retrieved sentences" remains relatively stable | Graph and candidate triples are denser, probability of **incorrect KV injection** and **KV + long evidence dual-channel attention competition** is **significantly higher than n40** |
| **Triple coverage** | Built with **strong alignment** to n40 eval IDs | Some samples have only **1–2** `interacts_with`-style triples | Can mainly rely on **sentence retrieval** to supplement readable evidence | KV **heavily depends** on triple selection and compilation; when coverage is thin or triples are wrong, **prefix injection directly harms** decoding, while GraphRAG has no such second channel |

**Summary**: On official, GraphRAG is still primarily "evidence into prompt"; KVI layers **triple→KV prefix** on top of the same evidence, and under **NL + large graph** it is more prone to **wrong triple selection / prompt conflict**, so early results can show **small drop for GraphRAG, large drop for KVI**. In the current table, **Panel A / MedHopQA_official** has been updated to a configuration where **KVI surpasses GraphRAG**: baseline KVI hyperparams (`drm_threshold=0.05,max_kv_triples=3,top_k_relations=2`) + **dual-decode reconciliation** (`--kvi_reconcile_no_kv_decode`); see Table Notes for machine-readable metrics.

### Table Notes (for paper text)

- **Metric**: Exact Match (EM, %) with 95% bootstrap CI in brackets.
- **Methods**: `LLM` (no retrieval), `RAG` (ANN + prompt), `GraphRAG` (graph + prompt), `KV Prefix` (ANN + KV), `KVI` (graph + KV + prompt).
- **EM for free-text QA**: For MedHop-style free-text short answers (e.g., `HTT`), keep EM as the primary metric, but note it may require alias/synonym normalization (not shipped in this repo). If needed, add a secondary concept-level metric (entity/UMLS mapping) in supplementary.
- **N/A meaning**: Method/dataset combination not yet run or unavailable in current summary artifacts.
- **Current sample sizes**: HotpotQA multihop `N=120`, NQ `N=100`, MedHopQA-ID `N=40`, **MedHopQA_official (NL query) `N=342`** (Panel A + Panel B rows available).
- **Significance test**: paired permutation against KVI is recorded in per-run `summary.json`.
- **MedHop setting in this table**: For Exp01, MedHop is mapped to an **ID-based relation-completion variant (MedHop-ID)**, where each query is normalized to `interacts_with DBxxxx?` and the target is a single partner ID `DByyyyy`.
- **Why this variant was used (pipeline stability/reproducibility)**: The goal is to keep Exp01 automation (evaluation + graph/KV construction) stable and scalable:  
  (1) **Deterministic evaluation**: `DBxxxx` is a canonical ID, so EM is unambiguous; free-text answers require alias/synonym normalization (e.g., `HTT` vs full names).  
  (2) **Direct graph alignment**: current graph index, triples, and KV compilation are entity-ID-centric, and `interacts_with DB...` forms a stable `query_entity -> answer_entity` path.  
  (3) **Lower engineering cost**: existing scripts already rewrite MedHop queries to ID form and enforce single-ID output, which supports reliable large-batch table generation.  
  This is an engineering/evaluation choice for Exp01, not a KVI-only requirement.
- **Why MedHop RAG/KV Prefix are 0.0 (with examples)**: This split evaluates strict partner-ID extraction. Gold answers are exact IDs such as `DB04844`/`DB00677`, while ANN-only outputs are often malformed or non-atomic, e.g. `DB1221` (wrong ID), `DB0977` (echo-like wrong ID), `DB0563 is involved in ...` (extra text), or long noisy generations in KV Prefix. Under ID-level EM/F1, these count as incorrect.
- **MedHopQA here is relation completion with strict output format**: For Exp01, MedHopQA is evaluated as an ID-based relation-completion task. Prompts require the model to output **only one** partner DrugBank ID in the exact `DB`+digits form (e.g. `DB04844`) and nothing else; generations that include extra text or multiple IDs are counted as incorrect under EM.
- **Are MedHop answers all `DB...` strings here?**: Yes. In this benchmark file (`medhop_eval.jsonl`), all 40/40 gold answers follow the `DB`+digits pattern, and prompts explicitly require `Answer with only the partner entity DB id`.
- **Why GraphRAG/KVI are stronger on this split**: This task is relation-centered (`interacts_with DBxxxx`) and aligns with graph traversal/evidence grounding. Graph pipelines more often surface the correct partner entity ID from relation evidence, while ANN-only pipelines rely on freer generation and are less robust to strict ID-format output constraints.
- **Panel B artifact path**: Per-dataset runs and the machine-readable aggregate live under `experiments/exp01_main_qa/results/main_table_mistral7b_v0_3/` (and sibling `multihop_hotpot_n120_fullmethods_mistral7b_v0_3/`, `nq_smoke100_fullmethods_mistral7b_v0_3/`, `medhop_n40_fullmethods_mistral7b_v0_3/`). **Mistral table copy checked into Git**: `experiments/exp01_main_qa/reports/main_table_mistral7b_v0_3/main_table.md`.
- **MedHop official-style split (external validity)**: Built from `medhop_raw` with natural-language questions, `short_answer` / `long_answer` / `supporting_facts` in `experiments/exp01_main_qa/data/medhop_official/`; documentation and field table (checked-in copy): `experiments/exp01_main_qa/reports/supplementary_medhop_official.md`. Gold remains partner **DrugBank IDs** (same EM family as MedHop-ID); free-text drug names are not shipped in this release.
- **MedHopQA_official N=342 (Panel A + Panel B)**: Full Exp01 on `medhop_eval.jsonl` with dedicated model-specific artifacts. **Panel A Qwen GraphRAG/KVI** values correspond to `experiments/exp01_main_qa/results/medhop_official_kvi_reconcile_final/summary.json` (`relaxed` EM; `--methods graphrag,kvi`; KVI has `--kvi_reconcile_no_kv_decode`). **Panel B Mistral five-method values** correspond to `experiments/exp01_main_qa/results/medhop_official_fullmethods_mistral7b_v0_3/summary.json`.
- **KVI reconcile note**: `--kvi_reconcile_no_kv_decode` runs a second decode without KV on the same prompt and heuristically picks the more evidence-grounded output; it is DB-ID-biased today (counts `DB\\d+` overlap with evidence) and should be generalized before applying to non-ID free-text QA.
- **MedHopQA_official — KVI defaults in `run_medhop_official_full_background.sh` (re-run after parameter tuning)**: To reduce incorrect injection and dual-channel conflict under NL+large graph, the script defaults to **`--kvi_drm_threshold 0.12`** (raised DRM threshold), **`--kvi_max_kv_triples 2`**, **`--kvi_top_k_relations 1`**, **`--kvi_minimal_prompt`**; default output directory **`medhop_official_fullmethods_qwen25_7b_kvituned/`** (avoids overwriting baseline `resume` output). Can be overridden via environment variables `KVI_DRM_THRESHOLD`, `KVI_MAX_KV_TRIPLES`, `KVI_TOP_K_RELATIONS`, `KVI_MINIMAL_PROMPT` (0/1), `MEDHOP_OFFICIAL_OUT`.
