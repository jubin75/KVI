# Script Description: Train Projector / Gate

Purpose: Map domain embeddings to the target model's attention head_dim space, and learn/configure γ (gate).

## Input
- `--encoder_ckpt`: domain_encoder
- `--target_model`: HF model name or path (for reading head_dim/num_heads and other metadata)
- `--train_pairs`: training samples (can be derived from weak supervision or synthetic QA)

## Key Parameters (Suggested)
- `--projector`: linear | mlp | lora
- `--share_kv`: whether to share K/V projection (decided by implementation)
- `--gamma_mode`: constant | learned_layer | learned_head | learned_token
- `--gamma_clamp_max`: default 0.10

## Output
- projector weights
- gate parameters (if learned)
- Evaluation: post-injection QA metrics and stability regression items


