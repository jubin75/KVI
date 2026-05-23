# Exp07 CL-Bench Long-Context Proxy

## Goal

Exp07 is the long-context reasoning track for KVI. It is rebuilt from the PRD goal:

- keep the original task answer unchanged
- enlarge context length with distractor evidence
- compare whether KVI-style grounding degrades more slowly than GraphRAG-style text grounding as context grows

This directory currently implements a **proxy benchmark** built on top of CL-Bench and the existing Exp01 evaluation stack.

## What This Program Does

`code/run_exp07_clbench_proxy.py` now runs the following pipeline:

1. Load CL-Bench mirror data from `experiments/_mirror_data_resolved/`.
2. Extract the original user prompt plus a proxy reference answer from `rubrics`.
3. Build explicit context-length regimes:
   - `short: 4K`
   - `medium: 16K`
   - `long: 32K`
   - `extreme: 64K`
4. Preserve the original CL-Bench prompt as the final `[Original Task]`.
5. Prepend sampled distractor documents from other CL-Bench examples until the target token budget is reached.
6. Emit a runnable JSONL benchmark split:
   - `data/clbench_longcontext_proxy_eval.jsonl`
7. Reuse the Exp01 artifact path to build:
   - `sentences.jsonl`
   - `triples.jsonl`
   - `graph_index.json`
   - `triple_kvbank/`
   - `kvbank_sentences/`
8. Run `experiments/exp01_main_qa/code/run_exp01.py` on the generated split.
9. Aggregate results by:
   - context-length regime
   - hop-count proxy

## Default Compared Methods

The rebuilt Exp07 defaults to:

- `llm`
- `rag`
- `graphrag`
- `kv_prefix`
- `kvi_triple_legacy`
- `kvi_schema_writer`
- `kvi_noinject_planner`

Rationale:

- `graphrag` is the text-grounding baseline.
- `kv_prefix` isolates KV-only effects without graph retrieval.
- `kvi_triple_legacy` preserves the historical KVI path for continuity.
- `kvi_schema_writer` is the current reasoning-oriented KVI successor.
- `kvi_noinject_planner` helps separate planning gains from direct KV-to-writer gains.

## PRD Alignment

This rebuild directly covers the strongest part of the PRD that is already compatible with the current codebase:

- Table 1 style main QA comparison
- Table 3 style performance by context-length regime
- length-scaling stress through distractor inflation
- unchanged task answer target
- standard HuggingFace/datasets-based loading

The script also records two proxy metadata axes per example:

- `hop_count_proxy`
- `evidence_sparsity_proxy`

Those proxies are heuristic, not gold annotations.

## Current Proxy Boundary

This is intentionally labeled a **proxy** because the current Exp07 still depends on the legacy Exp01 artifact path:

- artifact construction is still driven by `build_assets_from_dataset.py`
- it does **not** yet implement the final schema-first PRD artifact path
- it does **not** yet emit attention tensors, entropy curves, KV persistence curves, or KV-mask causal ablations
- it does **not** yet include the PRD-requested `GraphRAG + text triple prompting` baseline as a dedicated method key

So this rebuild is the correct executable bridge for the current repository state:

- good enough for long-context degradation comparisons now
- honest about what is still missing for the full paper-grade benchmark

## Outputs

Main outputs:

- `data/clbench_longcontext_proxy_eval.jsonl`
- `data/clbench_longcontext_proxy_manifest.json`
- `artifacts/clbench_proxy_v2/`
- `results/clbench_proxy_fullmethods_qwen25_7b/`
- `results/clbench_proxy_length_bucket_summary.json`
- `results/clbench_proxy_length_bucket_summary.csv`
- `results/clbench_proxy_length_bucket_summary.md`

## Run

Typical run:

```bash
/home/zd/dev/KVI/KVI/bin/python -u experiments/exp07_clbench_longcontext/code/run_exp07_clbench_proxy.py \
  --build_device cpu \
  --resident_url http://127.0.0.1:18888
```

Smoke run:

```bash
/home/zd/dev/KVI/KVI/bin/python -u experiments/exp07_clbench_longcontext/code/run_exp07_clbench_proxy.py \
  --build_device cpu \
  --resident_url http://127.0.0.1:18888 \
  --max_examples 40 \
  --limit 5
```

Prepare dataset and artifacts only:

```bash
/home/zd/dev/KVI/KVI/bin/python -u experiments/exp07_clbench_longcontext/code/run_exp07_clbench_proxy.py \
  --build_device cpu \
  --prepare_only
```

## Next Step Toward Full PRD

If Exp07 is upgraded from proxy to final benchmark, the next implementation step should be:

1. replace legacy QA-derived artifacts with schema-first evidence artifacts
2. add an explicit text-triple prompting baseline
3. export per-layer/per-head attention statistics
4. add KV-mask intervention evaluation
