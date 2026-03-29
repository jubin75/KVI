## Experiment 1 — Main QA Performance (single dataset)

- Dataset: HotpotQA
- N: 12

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean |
|---|---|---|---:|---:|---:|
| GraphRAG | graph | prompt | 58.3 | [33.3, 75.0] | 0.190 |
| KVI | graph | KV + prompt | 58.3 | [33.3, 83.3] | 0.024 |
