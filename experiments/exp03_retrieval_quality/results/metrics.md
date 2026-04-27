## Experiment 3 — Retrieval Quality (HotpotQA)

- Evaluated: **50** (skipped no gold: 0)

| Retrieval | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| ANN | 0.0 | 0.0 | 0.0 |
| Graph | 0.0 | 0.0 | 0.0 |

> **Note:** `gold_supporting_sentences` come from Hotpot **Wiki original text**, while Exp01's `sentences.jsonl` is mostly a **QA synthetic sentence pool**, so substrings often don't match and Recall/MRR will be 0. For non-zero Exp3 results, build `sentences.jsonl` / graph using corpora aligned with Hotpot paragraphs, or switch to dense ANN (install `sentence-transformers`) on the full sentence pool.
