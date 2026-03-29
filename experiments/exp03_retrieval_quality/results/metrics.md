## Experiment 3 — Retrieval Quality (HotpotQA)

- Evaluated: **50** (skipped no gold: 0)

| Retrieval | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|
| ANN | 0.0 | 0.0 | 0.0 |
| Graph | 0.0 | 0.0 | 0.0 |

> **Note:** `gold_supporting_sentences` 来自 Hotpot **Wiki 原文**，而 Exp01 的 `sentences.jsonl` 多为 **QA 合成句池**，子串往往对不上，Recall/MRR 会为 0。若需非零 Exp3，请用与 Hotpot 段落对齐的语料构建 `sentences.jsonl` / 图，或改用 dense ANN（安装 `sentence-transformers`）在全句池上评测。
