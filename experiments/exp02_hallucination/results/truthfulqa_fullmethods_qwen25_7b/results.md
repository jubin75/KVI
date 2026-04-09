## Experiment 1 — Main QA Performance (single dataset)

- Dataset: TRUTHFULQA
- N: 500

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean |
|---|---|---|---:|---:|---:|
| LLM | none | none | 5.0 | [3.2, 7.0] | 0.176 |
| RAG | ANN | prompt | 7.4 | [5.0, 9.6] | 0.135 |
| GraphRAG | graph | prompt | 17.8 | [14.6, 21.2] | 0.195 |
| KV Prefix | ANN | KV | 3.8 | [2.2, 5.6] | 0.016 |
| KVI | graph | KV + prompt | 11.8 | [9.0, 14.6] | 0.115 |
