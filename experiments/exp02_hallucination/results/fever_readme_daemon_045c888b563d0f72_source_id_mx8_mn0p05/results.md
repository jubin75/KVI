## Experiment 1 — Main QA Performance (single dataset)

- Dataset: FEVER
- N: 100

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

- **FEVER label accuracy**: first occurrence in model text of `SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO` (see `metrics.parse_fever_label`) vs gold; closer to veracity label accuracy than substring relaxed EM.

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean | FEVER lbl % | FEVER CI |
|---|---|---|---:|---:|---:|---:|---:|
| GraphRAG | graph | prompt | 67.0 | [58.0, 76.0] | 0.605 | 66.0 | [57.0, 75.0] |
| KVI Triple Legacy | graph | KV + prompt (legacy) | 88.0 | [82.0, 94.0] | 0.880 | 88.0 | [82.0, 94.0] |
| KVI Schema Writer | graph + schema | schema-guided KV + prompt | 77.0 | [68.0, 85.0] | 0.770 | 77.0 | [68.0, 85.0] |
| KVI Schema Verifier | graph + schema | schema plan + evidence writer | 71.0 | [62.0, 79.0] | 0.631 | 67.0 | [58.0, 75.0] |
| KVI No-Inject Planner | graph + schema | planner only | 71.0 | [62.0, 79.0] | 0.631 | 67.0 | [58.0, 75.0] |
