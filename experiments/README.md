## Experiments

This directory contains **reproducible experiment code**, **small test datasets**, and **saved results** for the KVI paper. The layout below is a **main-line dependency tree** after multiple repo iterations: start from an experiment folder, then follow **calls into** `scripts/` and `src/` at repo root.

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
