# Script Description: Build KV Bank + ANN Index

Purpose: Build a searchable KV Bank from ChunkStore and encoder/projector outputs, and generate an ANN index (faiss/hnsw).

## Input
- `--chunk_store`: ChunkStore path
- `--encoder_ckpt`: domain_encoder weights
- `--projector_ckpt`: projector weights (if used for generating K/V)

## Key Parameters (Suggested)
- `--index_type`: faiss | hnsw
- `--index_version`: v0/v1...
- `--top_k_default`: default retrieval k (suggested: 32)
- `--shard_size`: shard size (based on disk/memory constraints)
- `--compression`: none | pq | int8 (optional)

## Output
- KV Bank data (including citation metadata)
- ANN index file
- Build report: duration, vector count, shard info, version info


