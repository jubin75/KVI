## Experiment 1 — Main QA Performance (single dataset)

- Dataset: FEVER
- N: 25

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

- **FEVER label accuracy**: first occurrence in model text of `SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO` (see `metrics.parse_fever_label`) vs gold; closer to veracity label accuracy than substring relaxed EM.

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean | FEVER lbl % | FEVER CI |
|---|---|---|---:|---:|---:|---:|---:|
| GraphRAG | graph | prompt | 60.0 | [40.0, 76.0] | 0.587 | 60.0 | [40.0, 76.0] |
| KVI Triple Legacy | graph | KV + prompt (legacy) | 88.0 | [76.0, 100.0] | 0.880 | 88.0 | [76.0, 100.0] |
| KVI Schema Verifier | graph + schema | schema plan + evidence writer | 68.0 | [48.0, 84.0] | 0.653 | 64.0 | [44.0, 80.0] |
| KVI No-Inject Planner | graph + schema | planner only | 68.0 | [48.0, 84.0] | 0.653 | 64.0 | [44.0, 80.0] |
