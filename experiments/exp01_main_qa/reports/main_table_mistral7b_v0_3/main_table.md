## Experiment 1 — Main QA Performance

| Method | Retrieval | Injection | HotpotQA EM | HotpotQA CI95 | MedHopQA EM | MedHopQA CI95 | NQ EM | NQ CI95 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| LLM | none | none | 30.8 | [22.5, 39.2] | 0.0 | [0.0, 0.0] | 34.0 | [24.0, 43.0] |
| RAG | ANN | prompt | 36.7 | [28.3, 45.8] | 0.0 | [0.0, 0.0] | 63.0 | [53.0, 73.0] |
| GraphRAG | graph | prompt | 38.3 | [29.2, 46.7] | 10.0 | [2.5, 20.0] | 67.0 | [58.0, 76.0] |
| KV Prefix | ANN | KV | 16.7 | [10.0, 24.2] | 0.0 | [0.0, 0.0] | 17.0 | [10.0, 24.0] |
| KVI | graph | KV + prompt | 40.8 | [32.5, 49.2] | 12.5 | [2.5, 25.0] | 67.0 | [58.0, 76.0] |
