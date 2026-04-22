## Experiment 1 — Main QA Performance (single dataset)

- Dataset: TRUTHFULQA
- N: 25

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

- **TruthfulQA MC proxy**: from free-form output matched to MC options; **not** official MC1/MC2 (official requires option likelihood scoring).

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean | MC1 Proxy | MC2 Proxy |
|---|---|---|---:|---:|---:|---:|---:|
| LLM | none | none | 8.0 | [0.0, 20.0] | 0.211 | 0.480 | 0.502 |
| RAG | ANN | prompt | 8.0 | [0.0, 20.0] | 0.093 | 0.600 | 0.514 |
| GraphRAG | graph | prompt | 8.0 | [0.0, 20.0] | 0.341 | 0.720 | 0.560 |
| KV Prefix | ANN | KV | 0.0 | [0.0, 0.0] | 0.055 | 0.720 | 0.548 |
| KVI | graph | KV + prompt | 8.0 | [0.0, 20.0] | 0.515 | 0.800 | 0.561 |
