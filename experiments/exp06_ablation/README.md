# Experiment 6 — Ablation Study

**Purpose**: Validate the contribution of each module (Full KVI vs −graph / −DRM / −KV / −dual-channel).

## Directory

| Path | Description |
|------|------|
| `code/run_exp06_ablation.py` | Generate ablation table template + optionally fill one row from Exp01 `summary.json` |
| `results/` | `ablation_table.md`, `ablation_table.json` |

## Mapping to `run_graph_inference.py`

| Variant | CLI Approach |
|--------|----------|
| Full KVI | `--enable_kvi --triple_kvbank_dir ... --openqa_mode` |
| − KV injection | `--max_kv_triples 0` or omit `--enable_kvi` (i.e. GraphRAG) |
| − DRM (relaxed) | Lower `--drm_threshold` (e.g., `0.0`) as proxy |
| − dual channel | `--enable_kvi --kvi_minimal_prompt --openqa_mode` |
| − graph retrieval | Use `run_kvi2_runtime_test.py --pipeline modeA_rag` (ANN only) |

**Accuracy / Hallucination** must be filled in from each run's `predictions.jsonl` + your hallucination judgment rules; this script initially produces table headers and placeholders.

## Run

```bash
python experiments/exp06_ablation/code/run_exp06_ablation.py \
  --exp01_summary experiments/exp01_main_qa/results/smoke100/hotpot/summary.json \
  --out_dir experiments/exp06_ablation/results
```

`--exp01_summary` is optional; if provided and contains the `kvi` method, the **Full KVI** row's EM will be filled in (relaxed EM consistent with Exp01).
