## Experiment 2 — Hallucination Reduction (Proxy)

TruthfulQA: Hallucination Rate (%) = `100 - relaxed EM`. FEVER: Hallucination Rate (%) = `100 - fever_label_accuracy`.

| Dataset | Method | Relaxed EM (%) | Hallucination Rate (%) |
|---|---|---:|---:|
| truthfulqa | GraphRAG | 9.0 | 91.0 |
| truthfulqa | KVI Triple Legacy | 0.0 | 100.0 |
| truthfulqa | KVI Schema Writer | 5.0 | 95.0 |
| truthfulqa | KVI Schema Verifier | 2.0 | 98.0 |
| truthfulqa | KVI No-Inject Planner | 6.0 | 94.0 |
| fever | GraphRAG | 67.0 | 34.0 |
| fever | KVI Triple Legacy | 88.0 | 12.0 |
| fever | KVI Schema Writer | 77.0 | 23.0 |
| fever | KVI Schema Verifier | 71.0 | 33.0 |
| fever | KVI No-Inject Planner | 71.0 | 33.0 |

### TruthfulQA MC1/MC2 (proxy)

> Proxy metrics from current TruthfulQA run (`summary.json`); not official TruthfulQA script scores.

| Method | MC1 Proxy (%) | MC2 Proxy (%) | Valid N |
|---|---:|---:|---:|
| GraphRAG | 62.0 | 53.2 | 100 |
| KVI Triple Legacy | 45.0 | 47.1 | 100 |
| KVI Schema Writer | 52.0 | 50.9 | 100 |
| KVI Schema Verifier | 49.0 | 49.7 | 100 |
| KVI No-Inject Planner | 56.0 | 51.0 | 100 |
