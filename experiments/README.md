## Experiments

This directory contains **reproducible experiment code**, **small test datasets**, and **saved results** for the KVI paper. The layout below is a **main-line dependency tree** after multiple repo iterations: start from an experiment folder, then follow **calls into** `scripts/` and `src/` at repo root.

## 2026-04-28 Progressive Experiment Plan

This section is the new **execution mainline** for the next iteration of KVI experiments. It is written to address two confirmed issues:

- **Issue A: dual-channel conflict in Exp02**. In multi-hop QA, KV injection can help as a reasoning bias, but in hallucination control the same KV channel competes with evidence-grounded prompt generation and repeatedly fails to beat GraphRAG.
- **Issue B: unresolved large-graph retrieval**. The current KVI and GraphRAG pipelines both fall short on **query-relevant retrieval over large triple graphs**; current fallback behavior is suitable for small topic graphs, not for 10^5-scale triple stores.

The plan below replaces ad-hoc KVI sweeps with a staged protocol that separates:

- retrieval quality
- injected capsule quality
- answer generation quality under fixed evidence
- answer generation quality under noisy evidence

### New experiment policy

- **Exp01 remains the reasoning benchmark**: use it to test whether memory-level structure improves multi-hop and long-context QA.
- **Exp02 becomes a faithfulness benchmark**: use it to test whether the system avoids unsupported claims, abstains correctly, and stays evidence-consistent.
- **Do not require one KV strategy to optimize both tasks**.
- **For Exp02, schema-only KV is the default target architecture**.
- **Evidence and raw text are grounding channels, not injection channels**.
- **Triple-KV injection remains only as a legacy baseline and ablation target**.

### Method split for the next cycle

Use the following method definitions going forward.

| Method key | Retrieval | KV policy | Final answer writer | Primary use |
|---|---|---|---|---|
| `graphrag` | graph + sentence | no KV | evidence prompt only | baseline |
| `kvi_triple_legacy` | graph + sentence | triple KV into generator | same generator | legacy ablation |
| `kvi_schema_writer` | graph + sentence + schema retrieval | schema KV into generator | same generator | reasoning-oriented KVI |
| `kvi_schema_verifier` | graph + sentence + schema retrieval | schema KV only for plan / verify / abstain | evidence-only writer | new Exp02 mainline |
| `kvi_noinject_planner` | graph + sentence + schema retrieval | no generator KV, schema used only for rerank / routing | evidence-only writer | isolate planning value |

Short interpretation:

- `kvi_triple_legacy` tells us whether the old dual-channel triple-KV path still helps anywhere.
- `kvi_schema_writer` is the closest successor for Exp01.
- `kvi_schema_verifier` is the target method for Exp02.
- `kvi_noinject_planner` tests whether the real gain comes from better retrieval / planning rather than KV entering generation directly.

### Progressive phases

#### P0. Fix the measurement floor

Goal: stop optimizing on confounded or broken signals.

Required checks:

- Re-run and repair **Exp03 retrieval metrics**, because `Recall@k` and `MRR` cannot remain zero if the retrieval stack is to guide architecture choices.
- Re-check the TruthfulQA summary path where `MC1 proxy` becomes trivially saturated across methods.
- Freeze one canonical metric set for Exp02:
  - TruthfulQA: `MC1 proxy`, `MC2 proxy`, abstain / insufficient-evidence rate
  - FEVER: `fever_label_accuracy`
  - Shared: evidence consistency rate, contradiction rate, answer length stats

Files to inspect or change:

- `experiments/exp03_retrieval_quality/code/run_exp03_retrieval.py`
- `experiments/exp01_main_qa/code/metrics.py`
- `experiments/exp01_main_qa/code/run_exp01.py`
- `experiments/exp02_hallucination/results/summary.json`

Exit criteria:

- Exp03 metrics are non-degenerate and reproducible.
- TruthfulQA and FEVER summary rows are internally consistent.
- Every Exp02 run emits retrieval, evidence, and generation diagnostics in the same schema.

#### P1. Split Exp02 from legacy dual-channel generation

Goal: replace "KV writes first, evidence corrects later" with "evidence writes, schema verifies or constrains."

Protocol:

- Keep `graphrag` unchanged as the baseline.
- Keep `kvi_triple_legacy` only for comparison.
- Introduce a new Exp02 main method:
  - retrieve evidence sentences
  - retrieve schema capsules
  - use schema capsules only for slot / contradiction / abstention planning
  - render the final answer from evidence only

Required code changes:

- `scripts/run_graph_inference.py`
  - add an execution mode where KV does **not** enter `past_key_values` for the final writer
  - add a structured planning output:
    - resolved slots
    - contradiction flags
    - abstain recommendation
    - evidence coverage score
  - add an evidence-only final rendering branch
- `experiments/exp01_main_qa/code/run_exp01.py`
  - add method routing for `kvi_schema_verifier` and `kvi_noinject_planner`
  - keep `kvi` mapped to the legacy path until migration is complete

Exit criteria:

- Exp02 can run both `graphrag` and `kvi_schema_verifier` under the same artifacts.
- The final answer writer for `kvi_schema_verifier` receives no injected triple KV.
- Audit JSON clearly records whether KV was used for generation, planning, or neither.

#### P2. Build schema-first artifacts for Exp02

Goal: stop using answer-like QA triples as the only graph substrate for hallucination experiments.

Protocol:

- Keep current synthetic QA-derived artifacts only as legacy controls.
- Add a schema-first artifact path that produces:
  - evidence sentences
  - schema blocks
  - schema KV bank
  - sentence-level provenance links

Preferred artifact layering:

- `sentences.jsonl`: evidence-grounded sentence pool
- `graph_index.json`: provenance-linked graph
- `blocks.schema.jsonl`: schema capsules
- `kvbank_schema/`: only injection-eligible KV bank for Exp02-vNext

Required code changes:

- `experiments/exp01_main_qa/code/build_assets_from_dataset.py`
  - keep existing synthetic mode
  - add a second mode that does not force `question -> associated_with -> answer` triples
  - attach provenance-rich records and optional external evidence fields
- `experiments/exp02_hallucination/code/run_exp02_hallucination.py`
  - add a switch to choose `synthetic_qa_artifacts` vs `schema_first_artifacts`
- `scripts/build_schema_blocks_from_evidence_jsonl.py`
  - promote this into the Exp02 artifact pipeline

Exit criteria:

- Exp02 can be run in both legacy synthetic mode and schema-first mode.
- Schema-first mode produces `kvbank_schema` and uses it as the only injection-eligible bank.

#### P3. Replace large-graph fallback with staged retrieval

Goal: support 10^5-scale triple stores without global relation scans.

Protocol:

- Replace direct graph fallback with a staged retriever:
  - entity linker
  - triple / capsule retriever
  - sentence retriever
  - subgraph reranker
  - bounded graph walk on the reranked subgraph only

Target retrieval stack:

1. **Entity candidate generation**
2. **Triple / capsule ANN retrieval**
3. **Sentence ANN + lexical retrieval**
4. **Cross-source fusion**
5. **Small subgraph construction**
6. **Path scoring / reranking**
7. **Prompt evidence selection**
8. **Schema capsule selection**

Required code changes:

- `src/graph/graph_retriever.py`
  - deprecate topic-wide relation scans as the main fallback
  - move to candidate-first retrieval
- `src/graph/knowledge_graph.py`
  - add metadata fields that help subgraph scoring
- new modules recommended:
  - `src/graph/entity_linker.py`
  - `src/graph/triple_retriever.py`
  - `src/graph/subgraph_reranker.py`
  - `src/runtime/capsule_selector.py`

Exit criteria:

- No Exp02 mainline path depends on scanning all triples with a relation filter.
- Retrieval latency and candidate counts are bounded before graph walk.
- Retrieval evaluation reports can separate entity-link hit rate, triple recall, sentence recall, and final path hit rate.

#### P4. Separate reasoning gains from faithfulness gains

Goal: prove what each KVI variant is actually improving.

Required experiments:

- **Exp01 / reasoning**
  - `graphrag`
  - `kvi_triple_legacy`
  - `kvi_schema_writer`
  - `kvi_noinject_planner`
- **Exp02 / faithfulness**
  - `graphrag`
  - `kvi_triple_legacy`
  - `kvi_schema_verifier`
  - `kvi_noinject_planner`

Required ablations:

- fixed evidence, no retrieval noise
- retrieved evidence, real retrieval noise
- no KV
- schema KV only
- triple KV only
- planner-only, no generator KV

Exit criteria:

- We can identify whether a gain comes from retrieval, schema selection, KV generation bias, or evidence rendering.
- Exp01 and Exp02 no longer produce contradictory conclusions about the same mechanism.

### Exp02-vNext protocol

Use the following protocol for all new hallucination-facing runs.

#### Task definition

- TruthfulQA:
  - optimize `MC2 proxy`
  - monitor `MC1 proxy` but do not use it alone for ranking
  - add explicit insufficient-evidence / abstain accounting
- FEVER:
  - optimize `fever_label_accuracy`
  - require output normalization to exactly one label

#### Default compared methods

- `graphrag`
- `kvi_triple_legacy`
- `kvi_schema_verifier`
- `kvi_noinject_planner`

Do not use plain `kvi` as an ambiguous label in new result tables. When legacy compatibility is needed, explicitly state which implementation it maps to.

#### Required audits per run

Each Exp02 run must emit:

- retrieval hits
- evidence hits after rerank
- schema capsule hits
- contradiction flags
- abstain recommendation
- prompt evidence count
- whether final generation used injected KV
- answer length and sentence count

#### Required control settings

- fixed sample subsets for smoke and pre100 audits
- one resident model only
- serial runs only for protocol comparisons
- no mixing of old and new rows in the same `predictions.jsonl`

#### Pass / fail rules

- A new Exp02 method is only considered promising if it beats `graphrag` on:
  - TruthfulQA `MC2 proxy`
  - FEVER `fever_label_accuracy`
  - and does not increase unsupported long-form answers
- If a method helps one dataset but hurts the other, classify it as task-specific, not as the new default.

### Program improvement worklist

This is the required implementation order.

#### Work item 1. Add explicit KVI execution modes

Target files:

- `scripts/run_graph_inference.py`
- `experiments/exp01_main_qa/code/run_exp01.py`

Add modes:

- `legacy_triple_generate`
- `schema_generate`
- `schema_verify_then_evidence_write`
- `schema_plan_no_kv_generate`

Each mode should write its own debug block so later audits do not need to infer behavior from prompt text alone.

#### Work item 2. Add schema-selection diagnostics

Target files:

- `scripts/run_graph_inference.py`
- `src/runtime/slot_registry.py`
- `src/runtime/schema_answerability.py`
- `src/runtime/struct_slots.py`

Emit:

- selected schema ids
- selected slots
- unsupported slots
- contradiction slots
- abstain trigger reason

#### Work item 3. Add artifact-mode separation

Target files:

- `experiments/exp01_main_qa/code/build_assets_from_dataset.py`
- `experiments/exp02_hallucination/code/run_exp02_hallucination.py`
- `experiments/exp02_hallucination/code/prepare_exp02_datasets.py`

Requirements:

- artifact manifests must record whether they are `synthetic_qa` or `schema_first`
- result summaries must carry artifact mode
- no result table should compare methods across artifact modes without saying so

#### Work item 4. Add staged retrieval APIs

Target files:

- `src/graph/graph_retriever.py`
- new retriever helpers under `src/graph/`

Requirements:

- entity candidate recall metrics
- triple candidate recall metrics
- sentence candidate recall metrics
- subgraph size bounds
- removal of topic-wide relation scanning from the default mainline

#### Work item 5. Add evidence-only rendering path

Target files:

- `scripts/run_graph_inference.py`
- `experiments/exp01_main_qa/code/metrics.py`

Requirements:

- final answer text must be attributable to evidence sentences only
- schema outputs may shape abstention or slot resolution, but must not directly appear as unverified factual prose in the final answer

### Recommended command policy for the next two weeks

Use this rollout order.

1. P0 validation runs on existing artifacts and summaries.
2. P1 code changes with smoke tests on TruthfulQA `n=25` and FEVER `n=50`.
3. P2 artifact-mode split with `pre100` runs.
4. P3 staged retrieval prototype with retrieval-only evaluation first.
5. P4 full comparison tables only after P0-P3 pass.

### What should not be done anymore

- Do not treat longer or cleaner KVI output text as proof that KV helped.
- Do not use two-stage "KV draft then evidence correction" as the default Exp02 architecture.
- Do not use global relation-scan fallback as evidence that large-graph retrieval works.
- Do not merge synthetic QA-graph results and schema-first results into one unnamed `KVI` row.
- Do not optimize TruthfulQA alone and then claim a general hallucination reduction result.

### Experiment code tree (main pipeline)

```text
experiments/
├── README.md                          ← you are here
├── combine_experiment_results.py      → merges Exp01 + Exp03 + Exp06 → RESULTS_COMBINED.md
├── RESULTS_COMBINED.md
├── code/
│   ├── download_mirror_datasets.py    → HF mirror / local dataset resolution (Exp02 prep may call this)
│   └── run_exp02_exp07_cpu_nohup.sh   → batch driver (Exp02/Exp07)
│
├── exp01_main_qa/                     Experiment 1 — five-method QA (EM / relaxed EM / FEVER label / TQA MC proxy)
│   ├── code/
│   │   ├── run_exp01.py               ★ core runner: loads JSONL, per-method inference, metrics, summary.json
│   │   │       invokes (repo root):
│   │   │         scripts/run_graph_inference.py      … LLM, GraphRAG, KVI (graph + triple KV path)
│   │   │         scripts/run_kvi2_runtime_test.py    … RAG, KV Prefix (ANN / resident /infer/kvi)
│   │   ├── metrics.py                 … EM, FEVER label, TruthfulQA MC proxies, F1, etc.
│   │   ├── build_assets_from_dataset.py … dataset JSONL → artifacts/{sentences,triples}.jsonl + manifest
│   │   ├── prepare_hotpot_nq.py       … HotpotQA + NQ → unified JSONL (+ optional supporting sentences for Exp03)
│   │   ├── prepare_medhop_official_from_raw.py
│   │   ├── prepare_medhopqa_assets.py
│   │   ├── prepare_hotpot_multihop_assets.py
│   │   ├── sweep_kvi_vs_graphrag_medhop_official.py
│   │   ├── aggregate_exp01.py
│   │   ├── recalc_exp01_from_predictions.py
│   │   ├── collect_kvi_win_cases.py
│   │   ├── exp01_resident_infer_service.py … JSON HTTP wrapper for graph/ANN channels
│   │   └── *.sh                       … resident 18888, background full runs, MedHop, handoff, relocate data
│   ├── data/                          … benchmarks JSONL, manifests
│   ├── artifacts/                     … per-topic KV/graph/triple assets (large; often gitignored)
│   └── results/                       … main_table/, per-run predictions + md/csv/json
│
├── exp02_hallucination/               Experiment 2 — TruthfulQA + FEVER, unified "hallucination rate" summaries
│   ├── README.md                      … ops notes (resident, resume, fast_once, FEVER GPU)
│   ├── code/
│   │   ├── run_exp02_hallucination.py ★ orchestrator:
│   │   │       experiments/code/download_mirror_datasets.py (optional)
│   │   │       prepare_exp02_datasets.py
│   │   │       exp01_main_qa/code/build_assets_from_dataset.py
│   │   │       scripts/annotate_sentences_semantic_tags.py
│   │   │       scripts/build_kvbank_from_blocks_jsonl.py
│   │   │       scripts/build_knowledge_graph.py
│   │   │       src/graph/triple_kv_compiler.py
│   │   │       exp01_main_qa/code/run_exp01.py  (same five methods as Exp01)
│   │   ├── prepare_exp02_datasets.py
│   │   ├── plot_hallucination_proxy_bars.py
│   │   ├── plot_unified_hallucination_bars.py
│   │   └── *.sh                       … autoresume, fast_once, FEVER GPU/resume, KVI sweeps
│   ├── data/                          … truthfulqa_eval.jsonl, fever_eval.jsonl, dataset_manifest.json
│   ├── artifacts/{truthfulqa,fever}/  … graph_index, kvbank_sentences, triple_kvbank, …
│   └── results/                       … summary.md/json, per-dataset runs, figures (.svg/.html)
│
├── exp03_retrieval_quality/           Experiment 3 — retrieval metrics (Recall@k, MRR on Hotpot supporting sents)
│   ├── code/run_exp03_retrieval.py
│   ├── data/benchmarks/               … Hotpot JSONL with gold_supporting_sentences (from prepare_hotpot_nq)
│   └── results/                       … metrics.json, metrics.md
│
├── exp06_ablation/                    Experiment 6 — template / method ablations
│   ├── code/run_exp06_ablation.py
│   ├── code/run_kvi_ablation_suite.py
│   └── results/                     … ablation_table.md, ablation_table.json
│
└── exp07_clbench_longcontext/         Experiment 7 — long-context proxy (CL-Bench-style)
    ├── code/run_exp07_clbench_proxy.py
    └── code/run_exp07_autoresume.sh
```

### Repo-root scripts most often used by Exp01 / Exp02

```text
scripts/
├── run_graph_inference.py             … graph-side: LLM / GraphRAG / KVI (triple KV + prompt)
├── run_kvi2_runtime_test.py           … ANN-side: RAG / KV prefix injection
├── annotate_sentences_semantic_tags.py
├── build_kvbank_from_blocks_jsonl.py
└── build_knowledge_graph.py

src/graph/triple_kv_compiler.py        … graph_index + LLM → triple_kvbank (.pt + manifest)
```

Design docs that constrain what "injection" means (schema vs evidence) live under `docs/` (e.g. `00_overview.md`); they are **not** experiment entrypoints.

---

### Remote env (Linux) — network & model cache

- **HuggingFace connectivity**: Direct connection to `huggingface.co` is unreachable (timeout/Network unreachable). **Using a mirror works**: after `export HF_ENDPOINT=https://hf-mirror.com`, `curl https://hf-mirror.com` returns 200 (~0.5s), and `huggingface_hub` will use the mirror to generate download URLs (e.g. `https://hf-mirror.com/Qwen/Qwen2.5-7B-Instruct/...`). It is recommended to set this in the experiment terminal or `~/.bashrc` and `source ~/.bashrc` to pull models from the mirror.
- **Encoder local cache**: `sentence-transformers/all-MiniLM-L6-v2` already exists at **`/data/huggingface-cache/user/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2`**, readable by user `zd`. Usage: after `export HF_HOME=/data/huggingface-cache/user/huggingface`, transformers will load the encoder from that directory, **no internet required**.
- **BASE_LLM (Qwen2.5-7B-Instruct)**: `build.base_llm` in `config/topics/SFTSV/config.json` is already set to **`/home/zd/dev/KVI/models/Qwen2.5-7B-Instruct`** (local directory). For first use, download first: after `export HF_ENDPOINT=https://hf-mirror.com`, run `python scripts/download_qwen25_7b_local.py` (log: `experiments/logs/download_qwen25_7b.log`). After download, this path can be used directly as the local path for `from_pretrained`.

### Exp01 dataset construction (HotpotQA + NQ)

- **Source repos used in mirror**:
  - HotpotQA: `hotpot_qa` (config=`distractor`, split=`validation`)
  - NQ: `natural_questions` (split=`validation`)
- **Why these names**: in current mirror, `hotpotqa/hotpotqa` and `google/natural-questions` are unavailable; above names are reachable and verified.
- **Conversion script**: `experiments/exp01_main_qa/code/prepare_hotpot_nq.py`
- **Command used**:
  - `python experiments/exp01_main_qa/code/prepare_hotpot_nq.py --out_dir experiments/exp01_main_qa/data/benchmarks --hotpot_config distractor --hotpot_split validation --nq_split validation --hotpot_max 500 --nq_max 500 --streaming`
- **Output schema (paper-facing unified format)**:
  - each line: `{"id","question","answer","answers","dataset"}`
  - `answer`: first normalized answer used for compatibility
  - `answers`: all available gold aliases (used by EM via best-match)
- **Current local dataset size**:
  - `experiments/exp01_main_qa/data/benchmarks/hotpot_eval.jsonl`: 500
  - `experiments/exp01_main_qa/data/benchmarks/nq_eval.jsonl`: 500
  - manifest: `experiments/exp01_main_qa/data/benchmarks/dataset_manifest.json`

### Structure (short index)

- `exp01_main_qa/`: Experiment 1 — main QA performance (Exact Match / task-specific metrics).
- `exp02_hallucination/`: Experiment 2 — TruthfulQA + FEVER proxy hallucination summaries (`results/summary.md`).
- `exp03_retrieval_quality/`: Experiment 3 — ANN vs Graph retrieval (Recall@k, MRR on Hotpot supporting sentences).
- `exp06_ablation/`: Experiment 6 — ablation table (template + optional fill from Exp01).
- `exp07_clbench_longcontext/`: Experiment 7 — long-context proxy runs.
- `RESULTS_COMBINED.md`: merged Exp1 + Exp3 + Exp6 (run `python experiments/combine_experiment_results.py`).

### Medical "hallucination" vs TruthfulQA / FEVER / PubMedQA (brief)

- **TruthfulQA** is closest to "avoid popular **false** claims" (adversarial misconception style); MC proxies in Exp02 follow that spirit.
- **FEVER** is **evidence stance** (SUPPORTS / REFUTES / NEI) against a corpus: strong on **retrieval + attribution**, not the same construct as TQA's "myth busting."
- **PubMedQA** is **abstract-grounded MC** (yes/no/maybe): factual, but not primarily a **counter-misconception** benchmark.
- For a **medical analogue to TruthfulQA**, look for benchmarks built as **medical myth / unsafe false claim** discrimination or dedicated **medical hallucination** test suites (literature names evolve; search for "medical hallucination benchmark" / "Med-HALT"-style suites and cite the exact paper). PubMedQA can remain as a **separate axis** (reading + evidence in abstracts), not a drop-in replacement for TQA-style hallucination rate.
