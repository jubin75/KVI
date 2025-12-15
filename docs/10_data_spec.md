# 数据规范：PDF/文档 → ChunkStore

本文档定义数据层的输入输出与字段标准，面向 8×L40 48GB 资源约束下的可执行数据准备流程。

## 输入
- PDF（文本型/扫描型）
- HTML/网页
- DOCX/TXT

## 输出：ChunkStore（建议 Parquet/JSONL）
每条 chunk 记录必须包含以下字段（字段名可不完全一致，但语义必须一致）：

```
chunk_record = {
  doc_id: string,
  chunk_id: string,
  source_uri: string,            # 本地路径或 URL
  source_type: "pdf"|"html"|"txt"|"docx",
  page_range: [int,int] | null,  # 对 PDF 有效
  section_path: [string,...],    # 章节路径，至少包含顶层章节
  text: string,                  # 规范化文本/Markdown（实现自行选择）
  lang: string,                  # e.g. "zh"/"en"
  quality_score: float,          # 0~1
  ocr_used: bool,
  ocr_confidence: float | null,
  dedupe_hash: string,           # simhash/minhash
  created_at: string,
  dataset_version: string
}
```

## PDF 处理工具建议（选型参考）
- 结构化/版面：GROBID / marker / unstructured / PyMuPDF(fitz) / pdfplumber
- OCR：PaddleOCR（推荐中英）/ Tesseract（轻量）

## 切分策略（必须要求）
- 结构感知切分：标题/段落/表格/图注优先
- token-aware 合并：目标 200–500 tokens；上限 800–1200 tokens
- overlap：10%–20%（视任务与文本密度而定）
- 元数据贯穿：必须保留 page_range 与 section_path（或等价信息）

## 去重与质量控制（必须要求）
- chunk 级去重：simhash/minhash；并保留去重前后统计
- 噪声过滤：目录页/版权页/参考文献页可降权或剔除（策略由实现决定）
- OCR 置信度低：降权或剔除（阈值可配置）


