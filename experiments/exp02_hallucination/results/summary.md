## Experiment 2 — Unified Hallucination Rate Summary

- TruthfulQA (primary): `Hallucination Rate (%) = 100 - MC2 proxy (%)`
- TruthfulQA (aux): `Hallucination Rate (%) = 100 - MC1 proxy (%)`
- FEVER: `Hallucination Rate (%) = 100 - FEVER label accuracy (%)`

### Experimental Setup

| Item | Setting |
|---|---|
| Model | `/home/zd/dev/KVI/models/Qwen2.5-7B-Instruct` |
| Methods | `LLM`, `RAG`, `GraphRAG`, `KV Prefix`, `KVI` |
| TruthfulQA size | `100` |
| FEVER size | `100` |
| TruthfulQA evaluation source | `MC1/MC2 proxy` |
| FEVER evaluation source | `fever_label_accuracy` |
| Unified metric for plotting | `Hallucination Rate (%) = 100 - score(%)` |

### Metric Definition and Interpretation

- **Optimization direction**: lower `Hallucination Rate (%)` is better; `0%` is ideal and `100%` is worst.
- **TruthfulQA MC1 proxy (auxiliary)**: proportion of examples where the model's preferred option is labeled true (single-choice correctness proxy).
- **TruthfulQA MC2 proxy (primary)**: probability mass proxy assigned to true options (captures calibration beyond top-1 choice), less sensitive to top-1 tie effects.
- **FEVER label accuracy**: first parsed label in model output among `SUPPORTS/REFUTES/NOT ENOUGH INFO`, compared with gold label.
- **Unified conversion**: for all sources, `Hallucination Rate (%) = 100 - score(%)` so datasets can share one y-axis in figures.
- **Caveat**: TruthfulQA values here are `proxy` (not official leaderboard script), suitable for controlled internal comparison.

| Dataset | Method | Metric Source | Score (%) | Hallucination Rate (%) |
|---|---|---|---:|---:|
| fever | GraphRAG | fever_label_accuracy | 66.0 | 34.0 |
| fever | KVI Triple Legacy | fever_label_accuracy | 88.0 | 12.0 |
| fever | KVI Schema Writer | fever_label_accuracy | 77.0 | 23.0 |
| fever | KVI Schema Verifier | fever_label_accuracy | 67.0 | 33.0 |
| fever | KVI No-Inject Planner | fever_label_accuracy | 67.0 | 33.0 |
| truthfulqa | GraphRAG | mc1_proxy | 62.0 | 38.0 |
| truthfulqa | GraphRAG | mc2_proxy | 53.2 | 46.8 |
| truthfulqa | KVI Triple Legacy | mc1_proxy | 45.0 | 55.0 |
| truthfulqa | KVI Schema Writer | mc1_proxy | 52.0 | 48.0 |
| truthfulqa | KVI Schema Verifier | mc1_proxy | 49.0 | 51.0 |
| truthfulqa | KVI No-Inject Planner | mc1_proxy | 56.0 | 44.0 |
| truthfulqa | KVI Triple Legacy | mc2_proxy | 47.1 | 52.9 |
| truthfulqa | KVI Schema Writer | mc2_proxy | 50.9 | 49.1 |
| truthfulqa | KVI Schema Verifier | mc2_proxy | 49.7 | 50.3 |
| truthfulqa | KVI No-Inject Planner | mc2_proxy | 51.0 | 49.0 |

### Result Analysis (Primary metric: TruthfulQA MC2 proxy)

- On TruthfulQA `MC1 proxy`, the lowest hallucination rate is `38.0%` by **GraphRAG**.
- On TruthfulQA `MC2 proxy`, the lowest hallucination rate is `46.8%` by **GraphRAG**.
- On FEVER (`label accuracy` converted), the lowest hallucination rate is `12.0%` by **KVI Triple Legacy**.

**Note for paper writing**: TruthfulQA values here are proxy MC scores mapped into hallucination rate for unified plotting; they are suitable for internal comparison but should be explicitly labeled as proxy in final tables/figures.
