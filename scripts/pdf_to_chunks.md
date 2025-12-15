# 脚本说明：PDF → Chunks

目的：把 PDF/文档转换为 ChunkStore（见 `docs/10_data_spec.md`）。

## 输入
- `--input_dir`：PDF 目录
- `--output_path`：ChunkStore 输出（jsonl/parquet）

## 关键参数（建议）
- `--parser`：grobid | marker | unstructured | pymupdf（优先级由实现决定）
- `--ocr`：auto | on | off
- `--ocr_min_conf`：低于阈值降权或丢弃
- `--target_tokens`：200–500
- `--max_tokens`：800–1200
- `--overlap_ratio`：0.1–0.2
- `--dedupe`：on/off（simhash/minhash）
- `--lang_filter`：zh,en,...

## 输出
- ChunkStore 文件
- 统计报告：处理成功率、去重率、丢弃原因分布


