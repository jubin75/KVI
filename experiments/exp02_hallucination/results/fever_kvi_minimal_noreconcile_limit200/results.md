## Experiment 1 — Main QA Performance (single dataset)

- Dataset: FEVER
- N: 200

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean |
|---|---|---|---:|---:|---:|
| GraphRAG | graph | prompt | 21.5 | [16.0, 28.0] | 0.024 |
| KVI | graph | KV + prompt | 32.0 | [25.0, 38.5] | 0.007 |
