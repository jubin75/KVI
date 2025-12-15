# External KV Injection（可运行实现：数据构建 / 训练 / 多步注入）

本目录实现了基于 **HF Transformers** 的 External KV Injection（外部 KV 注入）工程：包含**可运行代码 + 文档 + 脚本**，用于把 PDF/文档知识构建为 **External KV Bank**，并在推理中以 **Multi-step Injection** 动态检索与注入 `K_ext/V_ext`（而不是做 RAG 文本拼接）。

## 核心数据流（见 `docs/00_overview.md`）
- **离线**：PDF/文档 → 抽取/OCR → 清洗/去重/打分 → 切分 raw_chunks（4096 tokens）→ 切分 blocks（256 tokens）→ 构建 KVBank（FAISS）
- **在线**：Query → Retriever top-k（从 KVBank 检索 blocks）→ 取回 `K_ext/V_ext` → 注入指定层 → 输出（可选导出引用与统计）

关键约束：
- **不训练主模型 layers**（可训练/使用外部模块：DomainEncoder / Projector / Gate）
- **KV cache 兼容**：外部 KV 以 `past_key_values` 的“静态前缀 KV”形式注入，prefill/decode 行为一致

## Raw context 与 KVBank 的职责边界（见 `docs/71_raw_context_pipeline.md`）
- **Raw context（`raw_chunks.jsonl`）**：仅用于建库，不进入 attention
- **KVBank（`kvbank_blocks/`）**：存 256-token memory blocks 的：
  - 检索用 embedding（ANN）
  - 注入用 `K_ext/V_ext`（默认来自 base LLM 的 `past_key_values`，常用 layers 0..3）
  - 结构化 `metadata`（含表格信息/页码/章节等，便于可解释与追溯）

医疗场景优化：
- **表格优先**（`pdfplumber` 抽取表格并转 markdown 追加到文本）
- **弱化噪声**（清洗时弱化/移除图例 Figure、引用标注、公式噪声）
- **DeepSeek 知识含量过滤（可选）**：段落级 KEEP/DROP，剔除前言泛背景、病例叙事、未来展望、方法学局限等低知识密度内容

## 多步注入（见 `docs/70_multistep_injection.md`）
- 每步检索/注入一次：基于当前状态动态更新 query
- 注入预算：单步 external KV tokens **≤1024（推荐）**，总注入 **≤2048**
- stopping policy：边际收益 + 冗余 + 安全上限，并可选加入 **external KV attention entropy** 收敛信号

## 快速开始（Linux：`/home/jb/KVI`）
请直接按 runbook 跑通全流程（PDF→KVBank→多步注入；可选 projector/gate 训练）：
- `docs/90_experiment_runbook_linux.md`

## 目录结构
- `docs/`：工程文档与可执行 runbook（建议从 `90_experiment_runbook_linux.md` 开始）
- `scripts/`：CLI 脚本（数据构建/训练/推理 demo）
- `src/`：核心实现（ingestion、cleaning、chunking、KVBank、runtime 注入、多步注入、过滤等）
- `configs/`：配置示例（部分脚本以 CLI 参数为主）

## 环境变量（如启用 DeepSeek 过滤）
- `DEEPSEEK_API_KEY`：DeepSeek API key（用于知识含量过滤）

## 与 PRD 的关系
- 总体 PRD：`PRD/PRD_Scheme_B_ExtKV_full.md`
- 多步注入约束：`PRD/多步注入的工程实现.md`
- raw context 构建流程：`PRD/raw context构建流程.md`


