## Experiment 1 — Main QA Performance (single dataset)

- Dataset: TRUTHFULQA
- N: 100

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

- **TruthfulQA MC proxy**: from free-form output matched to MC options; **not** official MC1/MC2 (official requires option likelihood scoring).

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean | MC1 Proxy | MC2 Proxy |
|---|---|---|---:|---:|---:|---:|---:|
| GraphRAG | graph | prompt | 9.0 | [4.0, 15.0] | 0.246 | 0.620 | 0.532 |
| KVI Triple Legacy | graph | KV + prompt (legacy) | 0.0 | [0.0, 0.0] | 0.295 | 0.450 | 0.471 |
| KVI Schema Verifier | graph + schema | schema plan + evidence writer | 2.0 | [0.0, 5.0] | 0.266 | 0.490 | 0.497 |
| KVI No-Inject Planner | graph + schema | planner only | 6.0 | [2.0, 11.0] | 0.326 | 0.560 | 0.510 |
