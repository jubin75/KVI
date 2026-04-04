## Experiment 1 — Main QA Performance (single dataset)

- Dataset: FEVER
- N: 5

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean |
|---|---|---|---:|---:|---:|
| LLM | none | none | 0.0 | [0.0, 0.0] | 0.000 |
| RAG | ANN | prompt | 100.0 | [100.0, 100.0] | 0.323 |
| GraphRAG | graph | prompt | 60.0 | [20.0, 100.0] | 0.029 |
| KV Prefix | ANN | KV | 60.0 | [20.0, 100.0] | 0.147 |
| KVI | graph | KV + prompt | 20.0 | [0.0, 60.0] | 0.007 |
