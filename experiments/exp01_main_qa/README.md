## Experiment 1: Main QA Performance (EM)

Goal: verify whether KVI improves QA accuracy (Exact Match, EM) compared to baselines.

### Methods (implemented in this repo)

- **LLM**: no retrieval, no injection (prompt-only baseline)
- **GraphRAG**: graph retrieval + prompt evidence (KVI disabled)
- **KVI**: graph retrieval + prompt evidence + KV injection (KVI enabled)

Note: the design doc also mentions ANN retrieval and KV-prefix baselines; these are not wired in the current runtime CLI, so Exp01 focuses on the three methods above.

### Inputs you must provide

- A compiled `graph_index.json`
- (Optional, for KVI) a compiled `triple_kvbank/` directory
- A base LLM model path/name compatible with the runtime scripts

### Dataset format (JSONL)

One example per line:

```json
{"id":"ex1","question":"...","answer":"..."}
```

### How to run (toy)

From repo root:

```bash
python external_kv_injection/experiments/exp01_main_qa/code/run_exp01.py \
  --dataset external_kv_injection/experiments/exp01_main_qa/data/toy_hotpot.jsonl \
  --model /path/to/base_llm \
  --graph_index /path/to/graph_index.json \
  --triple_kvbank_dir /path/to/triple_kvbank \
  --out_dir external_kv_injection/experiments/exp01_main_qa/results/run_001
```

Outputs:
- `results.md`: EM table + counts
- `summary.json`: aggregate metrics
- `predictions.jsonl`: per-example predictions

