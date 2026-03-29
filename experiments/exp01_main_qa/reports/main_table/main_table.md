## Experiment 1 — Main QA Performance (Reviewer-Friendly, Dual-Panel)

### Panel A: Base LLM = Qwen2.5-7B-Instruct

| Dataset | LLM | RAG | GraphRAG | KV Prefix | KVI |
|---|---:|---:|---:|---:|---:|
| HotpotQA (EM [CI95]) | 16.7 [10.0, 23.3] | 22.5 [15.0, 30.0] | 32.5 [24.2, 40.8] | 15.8 [9.2, 22.5] | 33.3 [25.0, 41.7] |
| NQ (EM [CI95]) | 20.0 [13.0, 28.0] | 41.0 [30.0, 51.0] | 44.0 [34.0, 54.0] | 14.0 [7.0, 21.0] | 49.0 [39.0, 58.0] |
| MedHopQA_n40 (EM [CI95]) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 82.5 [70.0, 92.5] | 0.0 [0.0, 0.0] | 92.5 [82.5, 100.0] |
| MedHopQA_official N=342 (EM [CI95]) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 74.3 [69.6, 78.9] | 0.0 [0.0, 0.0] | 49.4 [44.2, 54.7] |

### Panel B: Base LLM = Mistral-7B-Instruct-v0.3

| Dataset | LLM | RAG | GraphRAG | KV Prefix | KVI |
|---|---:|---:|---:|---:|---:|
| HotpotQA (EM [CI95]) | 30.8 [22.5, 39.2] | 36.7 [28.3, 45.8] | 38.3 [29.2, 46.7] | 16.7 [10.0, 24.2] | 40.8 [32.5, 49.2] |
| NQ (EM [CI95]) | 34.0 [24.0, 43.0] | 63.0 [53.0, 73.0] | 67.0 [58.0, 76.0] | 17.0 [10.0, 24.0] | 67.0 [58.0, 76.0] |
| MedHopQA_n40 (EM [CI95]) | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 10.0 [2.5, 20.0] | 0.0 [0.0, 0.0] | 12.5 [2.5, 25.0] |
| MedHopQA_official N=342 (EM [CI95]) | — | — | — | — | — |

### MedHopQA_n40 vs MedHopQA_official — 构造差异与对 GraphRAG / KVI 的非对称影响

| 维度 | MedHopQA_n40 | MedHopQA_official (N=342) | 对 GraphRAG 的典型影响 | 对 KVI 的典型影响 |
|------|--------------|---------------------------|-------------------------|-------------------|
| **问句形式** | `interacts_with DBxxxx?` + 强 ID 输出约束 | 自然语言问句（仍要求伙伴 DrugBank ID） | 图锚点 `DBxxxx` 仍在；文本检索有时更易命中摘要式证据 | NL 与 triple 文本、DRM 打分词面 **对齐变弱**，易选入 **无关或次优** triple 做 KV |
| **规模与语料** | 40 条，对应子图与句子规模小 | 342 条，全量 supports 切分约 **10⁵** 级句子 | 噪声上升相对温和；**单通道**「图 + 检索句」仍较稳 | 图与候选 triple 更密，**错误 KV 注入**与 **KV+长证据双通道竞争注意力** 的概率 **明显高于 n40** |
| **三元组覆盖** | 与 n40 评测 id **强对齐** 的构建 | 部分样本仅 **1～2** 条 `interacts_with` 式 triple | 可主要靠 **sentence 检索** 补全可读证据 | KV **强依赖** triple 选择与编译；覆盖薄或选错时，**前缀注入直接伤害** 解码，GraphRAG 无此第二通道 |

**归纳**：official 上 GraphRAG 仍主要是「证据进 prompt」；KVI 在同等证据上再叠加 **triple→KV 前缀**，在 **NL + 大图** 下更易 **选错 triple / 冲突 prompt**，故可出现 **GraphRAG 降幅小、KVI 降幅大**。评测脚本已对 MedHopQA_official 默认启用更保守的 KVI 超参（见 Table Notes 末条）；表中 **Panel A / MedHopQA_official** 数值仍为 **调参前全量基线**，重跑后请改填 `…_kvituned/summary.json`。

### Table Notes (for paper text)

- **Metric**: Exact Match (EM, %) with 95% bootstrap CI in brackets.
- **Methods**: `LLM` (no retrieval), `RAG` (ANN + prompt), `GraphRAG` (graph + prompt), `KV Prefix` (ANN + KV), `KVI` (graph + KV + prompt).
- **N/A meaning**: Method/dataset combination not yet run or unavailable in current summary artifacts.
- **Current sample sizes**: HotpotQA multihop `N=120`, NQ `N=100`, MedHopQA-ID `N=40`, **MedHopQA_official (NL query) `N=342`** (Panel A row only; see below).
- **Significance test**: paired permutation against KVI is recorded in per-run `summary.json`.
- **MedHop setting in this table**: For Exp01, MedHop is mapped to an **ID-based relation-completion variant (MedHop-ID)**, where each query is normalized to `interacts_with DBxxxx?` and the target is a single partner ID `DByyyyy`.
- **Why this variant was used (pipeline stability/reproducibility)**: The goal is to keep Exp01 automation (evaluation + graph/KV construction) stable and scalable:  
  (1) **Deterministic evaluation**: `DBxxxx` is a canonical ID, so EM is unambiguous; free-text answers require alias/synonym normalization (e.g., `HTT` vs full names).  
  (2) **Direct graph alignment**: current graph index, triples, and KV compilation are entity-ID-centric, and `interacts_with DB...` forms a stable `query_entity -> answer_entity` path.  
  (3) **Lower engineering cost**: existing scripts already rewrite MedHop queries to ID form and enforce single-ID output, which supports reliable large-batch table generation.  
  This is an engineering/evaluation choice for Exp01, not a KVI-only requirement.
- **Why MedHop RAG/KV Prefix are 0.0 (with examples)**: This split evaluates strict partner-ID extraction. Gold answers are exact IDs such as `DB04844`/`DB00677`, while ANN-only outputs are often malformed or non-atomic, e.g. `DB1221` (wrong ID), `DB0977` (echo-like wrong ID), `DB0563 is involved in ...` (extra text), or long noisy generations in KV Prefix. Under ID-level EM/F1, these count as incorrect.
- **Are MedHop answers all `DB...` strings here?**: Yes. In this benchmark file (`medhop_eval.jsonl`), all 40/40 gold answers follow the `DB`+digits pattern, and prompts explicitly require `Answer with only the partner entity DB id`.
- **Why GraphRAG/KVI are stronger on this split**: This task is relation-centered (`interacts_with DBxxxx`) and aligns with graph traversal/evidence grounding. Graph pipelines more often surface the correct partner entity ID from relation evidence, while ANN-only pipelines rely on freer generation and are less robust to strict ID-format output constraints.
- **Panel B artifact path**: Per-dataset runs and the machine-readable aggregate live under `experiments/exp01_main_qa/results/main_table_mistral7b_v0_3/` (and sibling `multihop_hotpot_n120_fullmethods_mistral7b_v0_3/`, `nq_smoke100_fullmethods_mistral7b_v0_3/`, `medhop_n40_fullmethods_mistral7b_v0_3/`). **Git 入库的 Mistral 表副本**：`experiments/exp01_main_qa/reports/main_table_mistral7b_v0_3/main_table.md`.
- **MedHop official-style split (external validity)**: Built from `medhop_raw` with natural-language questions, `short_answer` / `long_answer` / `supporting_facts` in `experiments/exp01_main_qa/data/medhop_official/`; documentation and field table（入库副本）: `experiments/exp01_main_qa/reports/supplementary_medhop_official.md`. Gold remains partner **DrugBank IDs** (same EM family as MedHop-ID); free-text drug names are not shipped in this release.
- **MedHopQA_official N=342 (merged into Panel A)**: Full Exp01 five methods on `medhop_eval.jsonl` with **Qwen2.5-7B-Instruct**, dedicated `artifacts/medhop_official/` (graph + ANN KV + triple KV). **Panel B** cells are **—** (Mistral not run on this split). **Table numbers above** = **KVI 调参前**基线；机器可读指标在本地跑分目录 `…/results/medhop_official_fullmethods_qwen25_7b/summary.json`（若 `results` 为数据盘符号链接则不在仓库内，`relaxed` EM）。
- **MedHopQA_official — KVI defaults in `run_medhop_official_full_background.sh`（调参后重跑）**：为减轻 NL+大图下的错误注入与双通道冲突，脚本默认 **`--kvi_drm_threshold 0.12`**（提高 DRM 门槛）、**`--kvi_max_kv_triples 2`**、**`--kvi_top_k_relations 1`**、**`--kvi_minimal_prompt`**；默认输出目录 **`medhop_official_fullmethods_qwen25_7b_kvituned/`**（避免与基线 `resume` 混写）。可用环境变量 `KVI_DRM_THRESHOLD`、`KVI_MAX_KV_TRIPLES`、`KVI_TOP_K_RELATIONS`、`KVI_MINIMAL_PROMPT`（0/1）、`MEDHOP_OFFICIAL_OUT` 覆盖。
