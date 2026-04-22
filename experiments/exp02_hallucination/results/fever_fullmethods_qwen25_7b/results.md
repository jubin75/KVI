## Experiment 1 — Main QA Performance (single dataset)

- Dataset: FEVER
- N: 1000

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

- **FEVER label accuracy**: first occurrence in model text of `SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO` (see `metrics.parse_fever_label`) vs gold; closer to veracity label accuracy than substring relaxed EM.

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean | FEVER lbl % | FEVER CI |
|---|---|---|---:|---:|---:|---:|---:|
| LLM | none | none | 38.3 | [35.2, 41.2] | 0.173 | 30.9 | [28.0, 33.7] |
| RAG | ANN | prompt | 92.6 | [91.0, 94.1] | 0.288 | 92.5 | [90.9, 94.1] |
| GraphRAG | graph | prompt | 73.0 | [70.4, 76.0] | 0.576 | 68.8 | [66.1, 72.0] |
| KV Prefix | ANN | KV | 74.3 | [71.7, 76.8] | 0.312 | 72.3 | [69.6, 75.1] |
| KVI | graph | KV + prompt | 89.3 | [87.3, 91.2] | 0.893 | 89.3 | [87.3, 91.2] |
