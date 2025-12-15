# 脚本说明：构建 KV Bank + ANN 索引

目的：从 ChunkStore 与 encoder/projector 产物构建可检索的 KV Bank，并生成 ANN 索引（faiss/hnsw）。

## 输入
- `--chunk_store`：ChunkStore 路径
- `--encoder_ckpt`：domain_encoder 权重
- `--projector_ckpt`：projector 权重（若用于生成 K/V）

## 关键参数（建议）
- `--index_type`：faiss | hnsw
- `--index_version`：v0/v1...
- `--top_k_default`：默认检索 k（建议 32）
- `--shard_size`：分片大小（按磁盘/内存约束）
- `--compression`：none | pq | int8（可选）

## 输出
- KV Bank 数据（含引用元数据）
- ANN 索引文件
- 构建报告：耗时、向量数、分片信息、版本信息


