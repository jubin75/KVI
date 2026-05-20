## Experiment 1 — Main QA Performance (single dataset)

- Dataset: TRUTHFULQA
- N: 100

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

- **TruthfulQA MC proxy**: from free-form output matched to MC options; **not** official MC1/MC2 (official requires option likelihood scoring).

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean | MC1 Proxy | MC2 Proxy |
|---|---|---|---:|---:|---:|---:|---:|
| GraphRAG | graph | prompt | 9.0 | [4.0, 15.0] | 0.216 | 0.650 | 0.542 |
| KV Prefix | ANN | KV | 2.0 | [0.0, 5.0] | 0.113 | 0.490 | 0.460 |
| KVI | graph | KV + prompt | 3.0 | [0.0, 7.0] | 0.293 | 0.480 | 0.490 |
