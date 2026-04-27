# Script Description: PDF → Chunks

Purpose: Convert PDF/documents into ChunkStore (see `docs/10_data_spec.md`).

## Input
- `--input_dir`: PDF directory
- `--output_path`: ChunkStore output (jsonl/parquet)

## Key Parameters (Suggested)
- `--parser`: grobid | marker | unstructured | pymupdf (priority determined by implementation)
- `--ocr`: auto | on | off
- `--ocr_min_conf`: below threshold downgrade or discard
- `--target_tokens`: 200–500
- `--max_tokens`: 800–1200
- `--overlap_ratio`: 0.1–0.2
- `--dedupe`: on/off (simhash/minhash)
- `--lang_filter`: zh,en,...

## Output
- ChunkStore file
- Statistics report: processing success rate, dedup rate, discard reason distribution


