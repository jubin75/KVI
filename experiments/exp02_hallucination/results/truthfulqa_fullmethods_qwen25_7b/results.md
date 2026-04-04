## Experiment 1 — Main QA Performance (single dataset)

- Dataset: TRUTHFULQA
- N: 5

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean |
|---|---|---|---:|---:|---:|
| LLM | none | none | 0.0 | [0.0, 0.0] | 0.232 |
| RAG | ANN | prompt | 0.0 | [0.0, 0.0] | 0.000 |
| GraphRAG | graph | prompt | 0.0 | [0.0, 0.0] | 0.286 |
| KV Prefix | ANN | KV | 0.0 | [0.0, 0.0] | 0.000 |
| KVI | graph | KV + prompt | 0.0 | [0.0, 0.0] | 0.261 |
