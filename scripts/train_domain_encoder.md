# Script Description: Train DomainEncoder (Retrieval Encoder)

Purpose: Train/distill a domain retrieval encoder for chunk retrieval (Recall@k as primary metric).

## Input
- `--train_data`: retrieval pair data (query, positive, negatives)
- `--valid_data`: validation set

## Key Parameters (Suggested)
- `--model`: bert/scibert/mini-lm/... (chosen by implementation)
- `--batch_size`: tune based on VRAM and sequence length
- `--negatives`: in-batch | mined-hard
- `--max_length`: consistent with or slightly smaller than chunk token limit

## Output
- encoder weights
- Metrics report: Recall@k (k=8/16/32), MRR@k (optional)


