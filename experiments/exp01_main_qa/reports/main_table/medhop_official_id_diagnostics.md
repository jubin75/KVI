## MedHopQA_official ID Diagnostics (for explaining EM=0)

This table supplements EM with two diagnostics:
- **Valid-ID Rate (%)**: prediction is exactly one `DB` id (`^DB\\d+$`) with no extra text.
- **Extract-then-EM (%)**: extract the first `DB\\d+` from prediction, then compare to gold ID.

| Backbone | Method | N | Valid-ID Rate (%) | Extract-then-EM (%) |
|---|---|---:|---:|---:|
| Qwen2.5-7B-Instruct | LLM | 342 | 0.0 | 0.0 |
| Qwen2.5-7B-Instruct | RAG | 342 | 91.8 | 0.0 |
| Qwen2.5-7B-Instruct | GraphRAG | 342 | 0.0 | 36.8 |
| Qwen2.5-7B-Instruct | KV Prefix | 342 | 0.0 | 0.0 |
| Qwen2.5-7B-Instruct | KVI | 342 | 0.0 | 36.5 |
| Mistral-7B-Instruct-v0.3 | LLM | 342 | 0.0 | 0.0 |
| Mistral-7B-Instruct-v0.3 | RAG | 342 | 0.0 | 0.0 |
| Mistral-7B-Instruct-v0.3 | GraphRAG | 342 | 0.0 | 11.7 |
| Mistral-7B-Instruct-v0.3 | KV Prefix | 342 | 0.0 | 0.0 |
| Mistral-7B-Instruct-v0.3 | KVI | 342 | 63.7 | 33.3 |

### Source runs
- Qwen (`LLM/RAG/GraphRAG/KV Prefix`): `/home/zd/dev/KVI/experiments/exp01_main_qa/results/medhop_official_fullmethods_qwen25_7b/predictions.jsonl`
- Qwen (`KVI`): `/home/zd/dev/KVI/experiments/exp01_main_qa/results/medhop_official_kvi_reconcile_final/predictions.jsonl`
- Mistral (all five methods): `/home/zd/dev/KVI/experiments/exp01_main_qa/results/medhop_official_fullmethods_mistral7b_v0_3/predictions.jsonl`

Note: these diagnostics are for analysis only; the main table metric remains EM.
