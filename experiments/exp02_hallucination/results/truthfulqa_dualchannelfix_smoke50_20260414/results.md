## Experiment 1 — Main QA Performance (single dataset)

- Dataset: TRUTHFULQA
- N: 50

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

- **TruthfulQA MC proxy**: from free-form output matched to MC options; **not** official MC1/MC2 (official requires option likelihood scoring).

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean | MC1 Proxy | MC2 Proxy |
|---|---|---|---:|---:|---:|---:|---:|
| GraphRAG | graph | prompt | 2.0 | [0.0, 6.0] | 0.223 | 0.660 | 0.539 |
| KV Prefix | ANN | KV | 4.0 | [0.0, 10.0] | 0.130 | 0.540 | 0.470 |
| KVI | graph | KV + prompt | 6.0 | [0.0, 14.0] | 0.366 | 0.600 | 0.507 |
