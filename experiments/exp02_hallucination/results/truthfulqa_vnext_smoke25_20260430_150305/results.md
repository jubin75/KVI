## Experiment 1 — Main QA Performance (single dataset)

- Dataset: TRUTHFULQA
- N: 25

- EM mode: **relaxed** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)

- **TruthfulQA MC proxy**: from free-form output matched to MC options; **not** official MC1/MC2 (official requires option likelihood scoring).

| Method | Retrieval | Injection | EM | 95% CI | F1 Mean | MC1 Proxy | MC2 Proxy |
|---|---|---|---:|---:|---:|---:|---:|
| GraphRAG | graph | prompt | 8.0 | [0.0, 20.0] | 0.209 | 0.760 | 0.586 |
| KVI Triple Legacy | graph | KV + prompt (legacy) | 0.0 | [0.0, 0.0] | 0.340 | 0.600 | 0.506 |
| KVI Schema Verifier | graph + schema | schema plan + evidence writer | 4.0 | [0.0, 12.0] | 0.304 | 0.600 | 0.539 |
| KVI No-Inject Planner | graph + schema | planner only | 8.0 | [0.0, 20.0] | 0.383 | 0.680 | 0.542 |
