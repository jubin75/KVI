## Authoring E2E (t5) — 一键跑通命令集

下面命令把整条链路跑通并输出可读检索结果：

- （可选但推荐）先重建 topic：生成 `docs.meta.jsonl` + 新版 `blocks.evidence.jsonl`
- Authoring DB（JSONL）准备
- 启动 Authoring UI（可选，但建议用来 approve）
- Doc 预览（按 doc meta 聚合查看 `blocks.evidence` 全量文本）
- 导出 approved-only runtime evidence
- base LLM forward 抽 K/V 并构建 evidence KVBank
- query 检索 top‑k 并打印可读结果

### 0) （推荐）重建 topic：生成 docs.meta + blocks.evidence（会调用 DeepSeek）

> 在 topic 工作目录（例如 `/home/jb/topics/SFTSV/work`）下生成：
> - `docs.meta.jsonl`（doc-level meta：title/doi/year/...）
> - `blocks.evidence.jsonl`（block-level evidence：含 block_type/kb_id）

```bash
python -u scripts/rebuild_topic_kvbank_from_config.py \
  --config config/topics/SFTSV/config.json
```

### 1) 设定工作目录（示例：临时目录）

```bash
WORK_DIR="/tmp/authoring_e2e"
mkdir -p "$WORK_DIR"
cp external_kv_injection/examples/authoring_evidence_units.sample.jsonl "$WORK_DIR/evidence_units.jsonl"
```

### 2) 启动 Authoring UI（新终端）

> `--host` 注意：
> - 本机/SSH 转发访问：用 `127.0.0.1`
> - 对外暴露（不推荐）：用 `0.0.0.0` 并放行端口

（示例：topic 目录下的真实 DB + 默认 kb_id= SFTSV）

```bash
python -u authoring_app/server.py \
  --db /home/jb/topics/SFTSV/work/evidence_units.jsonl \
  --default_kb_id SFTSV \
  --host 127.0.0.1 \
  --port 8765
```

浏览器打开：`http://127.0.0.1:8765`

> 你需要确保至少有 1 条 EvidenceUnit 被设置为 `approved`（否则不会进入 KVBank）。

### 3) Doc 预览（按 doc meta 聚合查看 blocks.evidence 全量文本）

在 UI 的 **Doc 预览** 区域填入：

- `/home/jb/topics/SFTSV/work/docs.meta.jsonl`
- `/home/jb/topics/SFTSV/work/blocks.evidence.jsonl`

点击 **Load Docs**，再点击某个 doc 卡片，即可在右侧看到该 doc 的 **blocks.evidence 全量文本**。

### 4) 一键导出 + 建库 + 检索（另一个终端）

把下面两个模型参数替换成你本机可用的模型（会由 transformers 下载/加载）：

- `BASE_LLM`: 用于 forward 抽 past_key_values（K/V）
- `ENCODER`: 用于语义检索向量（retrieval keys）

```bash
BASE_LLM="<你的base模型，如 Qwen/Qwen2.5-7B-Instruct>"
ENCODER="<你的encoder模型，如 sentence-transformers/all-MiniLM-L6-v2>"

python -u external_kv_injection/scripts/run_authoring_e2e_demo.py \
  --work_dir "$WORK_DIR" \
  --authoring_db_jsonl "$WORK_DIR/evidence_units.jsonl" \
  --base_llm_name_or_path "$BASE_LLM" \
  --retrieval_encoder_model "$ENCODER" \
  --query "common symptoms include fever headache myalgia" \
  --top_k 5
```

输出会按 STEP 0~3 打印，并在 STEP 3 列出 top‑k 结果（含 `evidence_id/score/semantic_type/schema_id/text`）。

### 5) （可选）拆开跑：导出 / 建库

导出 approved-only runtime jsonl：

```bash
python -u external_kv_injection/scripts/export_authoring_evidence_runtime_jsonl.py \
  --authoring_jsonl "$WORK_DIR/evidence_units.jsonl" \
  --out "$WORK_DIR/authoring.evidence.runtime.jsonl"
```

建 evidence KVBank：

```bash
python -u external_kv_injection/scripts/build_kvbank_from_authoring_evidence_jsonl.py \
  --evidence_jsonl "$WORK_DIR/authoring.evidence.runtime.jsonl" \
  --out_dir "$WORK_DIR/kvbank_evidence_authoring" \
  --base_llm_name_or_path "$BASE_LLM" \
  --retrieval_encoder_model "$ENCODER"
```

