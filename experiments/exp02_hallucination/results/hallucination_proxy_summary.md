## Experiment 2 — Hallucination Reduction (Proxy)

TruthfulQA: Hallucination Rate (%) = `100 - relaxed EM`. FEVER: Hallucination Rate (%) = `100 - fever_label_accuracy`.

| Dataset | Method | Relaxed EM (%) | Hallucination Rate (%) |
|---|---|---:|---:|
| truthfulqa | LLM | 0.0 | 100.0 |
| truthfulqa | RAG | 0.0 | 100.0 |
| truthfulqa | GraphRAG | 0.0 | 100.0 |
| truthfulqa | KV Prefix | 0.0 | 100.0 |
| truthfulqa | KVI | 0.0 | 100.0 |
| fever | LLM | 38.3 | 69.1 |
| fever | RAG | 92.6 | 7.5 |
| fever | GraphRAG | 73.0 | 31.2 |
| fever | KV Prefix | 74.3 | 27.7 |
| fever | KVI | 89.3 | 10.7 |

### TruthfulQA MC1/MC2 (proxy)

> Proxy metrics from current TruthfulQA run (`summary.json`); not official TruthfulQA script scores.

| Method | MC1 Proxy (%) | MC2 Proxy (%) | Valid N |
|---|---:|---:|---:|
| LLM | 100.0 | 66.9 | 2 |
| RAG | 100.0 | 73.1 | 2 |
| GraphRAG | 100.0 | 78.1 | 2 |
| KV Prefix | 100.0 | 63.4 | 2 |
| KVI | 100.0 | 69.2 | 2 |
