## Experiment 1 — Main QA Performance (single dataset)

- Dataset: TRUTHFULQA
- N: 500

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

- **TruthfulQA MC proxy**: from free-form output matched to MC options; **not** official MC1/MC2 (official requires option likelihood scoring).

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean | MC1 Proxy | MC2 Proxy |
|---|---|---|---:|---:|---:|---:|---:|
| GraphRAG | graph | prompt | 18.4 | [15.2, 21.6] | 0.295 | 0.642 | 0.527 |
| KV Prefix | ANN | KV | 3.2 | [1.8, 4.8] | 0.116 | 0.462 | 0.435 |
| KVI | graph | KV + prompt | 10.2 | [7.6, 13.0] | 0.326 | 0.556 | 0.490 |
