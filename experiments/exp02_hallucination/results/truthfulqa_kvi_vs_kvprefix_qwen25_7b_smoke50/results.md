## Experiment 1 — Main QA Performance (single dataset)

- Dataset: TRUTHFULQA
- N: 50

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

- **TruthfulQA MC proxy**: from free-form output matched to MC options; **not** official MC1/MC2 (official requires option likelihood scoring).

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean | MC1 Proxy | MC2 Proxy |
|---|---|---|---:|---:|---:|---:|---:|
| KV Prefix | ANN | KV | 2.0 | [0.0, 6.0] | 0.067 | 0.680 | 0.532 |
| KVI | graph | KV + prompt | 2.0 | [0.0, 6.0] | 0.093 | 0.640 | 0.528 |
