## Experiment 1 — Main QA Performance (single dataset)

- Dataset: TRUTHFULQA
- N: 25

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

- **TruthfulQA MC proxy**: from free-form output matched to MC options; **not** official MC1/MC2 (official requires option likelihood scoring).

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean | MC1 Proxy | MC2 Proxy |
|---|---|---|---:|---:|---:|---:|---:|
| GraphRAG | graph | prompt | 12.0 | [0.0, 24.0] | 0.231 | 0.680 | 0.568 |
| KV Prefix | ANN | KV | 4.0 | [0.0, 12.0] | 0.141 | 0.600 | 0.507 |
| KVI | graph | KV + prompt | 8.0 | [0.0, 20.0] | 0.378 | 0.680 | 0.537 |
