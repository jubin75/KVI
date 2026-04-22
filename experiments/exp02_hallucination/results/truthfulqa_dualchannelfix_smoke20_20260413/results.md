## Experiment 1 — Main QA Performance (single dataset)

- Dataset: TRUTHFULQA
- N: 20

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

- **TruthfulQA MC proxy**: from free-form output matched to MC options; **not** official MC1/MC2 (official requires option likelihood scoring).

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean | MC1 Proxy | MC2 Proxy |
|---|---|---|---:|---:|---:|---:|---:|
| GraphRAG | graph | prompt | 0.0 | [0.0, 0.0] | 0.234 | 0.700 | 0.560 |
| KV Prefix | ANN | KV | 0.0 | [0.0, 0.0] | 0.137 | 0.600 | 0.499 |
| KVI | graph | KV + prompt | 5.0 | [0.0, 15.0] | 0.365 | 0.650 | 0.532 |
