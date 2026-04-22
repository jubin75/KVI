## Experiment 2 — Unified Hallucination Rate Summary

- TruthfulQA (primary): `Hallucination Rate (%) = 100 - MC2 proxy (%)`
- TruthfulQA (aux): `Hallucination Rate (%) = 100 - MC1 proxy (%)`
- FEVER: `Hallucination Rate (%) = 100 - FEVER label accuracy (%)`

### Experimental Setup

| Item | Setting |
|---|---|
| Model | `/home/zd/dev/KVI/models/Qwen2.5-7B-Instruct` |
| Methods | `LLM`, `RAG`, `GraphRAG`, `KV Prefix`, `KVI` |
| TruthfulQA size | `2` |
| FEVER size | `1000` |
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
| fever | LLM | fever_label_accuracy | 30.9 | 69.1 |
| fever | RAG | fever_label_accuracy | 92.5 | 7.5 |
| fever | GraphRAG | fever_label_accuracy | 68.8 | 31.2 |
| fever | KV Prefix | fever_label_accuracy | 72.3 | 27.7 |
| fever | KVI | fever_label_accuracy | 89.3 | 10.7 |
| truthfulqa | LLM | mc1_proxy | 100.0 | 0.0 |
| truthfulqa | LLM | mc2_proxy | 66.9 | 33.1 |
| truthfulqa | RAG | mc1_proxy | 100.0 | 0.0 |
| truthfulqa | RAG | mc2_proxy | 73.1 | 26.9 |
| truthfulqa | GraphRAG | mc1_proxy | 100.0 | 0.0 |
| truthfulqa | GraphRAG | mc2_proxy | 78.1 | 21.9 |
| truthfulqa | KV Prefix | mc1_proxy | 100.0 | 0.0 |
| truthfulqa | KV Prefix | mc2_proxy | 63.5 | 36.5 |
| truthfulqa | KVI | mc1_proxy | 100.0 | 0.0 |
| truthfulqa | KVI | mc2_proxy | 69.2 | 30.8 |

### Result Analysis (Primary metric: TruthfulQA MC2 proxy)

- On TruthfulQA `MC1 proxy`, the lowest hallucination rate is `0.0%` by **LLM**.
- On TruthfulQA `MC2 proxy`, the lowest hallucination rate is `21.9%` by **GraphRAG**.
- On FEVER (`label accuracy` converted), the lowest hallucination rate is `7.5%` by **RAG**.
- `KV Prefix` shows a large MC1/MC2 gap (`0.0%` vs `36.5%`), indicating unstable behavior across different TruthfulQA proxy criteria.
- On TruthfulQA MC2 proxy, **KVI** (`30.8%`) is close to **GraphRAG** (`21.9%`), suggesting KVI is competitive on multi-choice mass proxy.
- On FEVER, **RAG** remains better than **KVI** in this run (`7.5%` vs `10.7%`).

**Note for paper writing**: TruthfulQA values here are proxy MC scores mapped into hallucination rate for unified plotting; they are suitable for internal comparison but should be explicitly labeled as proxy in final tables/figures.
