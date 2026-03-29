## Experiment 1 — Main QA Performance (single dataset)

- Dataset: HotpotQA
- N: 8

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean |
|---|---|---|---:|---:|---:|
| GraphRAG | graph | prompt | 62.5 | [25.0, 87.5] | 0.244 |
| KVI | graph | KV + prompt | 12.5 | [0.0, 37.5] | 0.017 |
