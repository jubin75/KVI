## Experiment 1 — Main QA Performance (single dataset)

- Dataset: FEVER
- N: 50

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

- **FEVER label accuracy**: first occurrence in model text of `SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO` (see `metrics.parse_fever_label`) vs gold; closer to veracity label accuracy than substring relaxed EM.

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean | FEVER lbl % | FEVER CI |
|---|---|---|---:|---:|---:|---:|---:|
| GraphRAG | graph | prompt | 68.0 | [54.0, 80.0] | 0.643 | 68.0 | [54.0, 80.0] |
| KVI Triple Legacy | graph | KV + prompt (legacy) | 90.0 | [82.0, 98.0] | 0.900 | 90.0 | [82.0, 98.0] |
| KVI Schema Verifier | graph + schema | schema plan + evidence writer | 76.0 | [64.0, 86.0] | 0.702 | 70.0 | [56.0, 82.0] |
| KVI No-Inject Planner | graph + schema | planner only | 76.0 | [64.0, 86.0] | 0.702 | 70.0 | [56.0, 82.0] |
