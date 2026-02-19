# Build Graph 句子来源与抽取策略

## 1. 1374 条句子从哪来？是对单篇 document 还是全部？

**结论：当前是对「整个 topic 下全部 document」做抽取，不是单篇。**

Build Graph 的 Step1 逻辑（`server.py` 中 `_run_build_full_pipeline_background`）：

- 句子来源优先级：`blocks.evidence.jsonl` → `blocks.enriched.jsonl` → `blocks.jsonl`。
- 读取的是 **topic 的 work_dir** 下整份 blocks 文件：遍历文件中**每一条** block，把 `text`/`claim` 当作一条 sentence 加入列表。
- **没有按 `doc_id` 过滤**：所以 1374 = 该 topic（如 SFTSV）下**所有 PDF** 经 evidence 抽取后的 **block 总数**（约 110 篇 PDF × 每篇若干 block）。

因此：

- 若你期望「只对一篇 document 做三元组抽取」，当前实现**不满足**；需要后续加「按 doc 过滤」的选项（例如只选当前 View Details 的 doc_id 对应 blocks，或 UI 上勾选若干 doc 再 Build Graph）。
- 若你期望「只对当前打开的这篇建图」，需要在 pipeline 入口增加「仅使用指定 doc_id 的 blocks」的分支（从同一份 `blocks.evidence.jsonl` 里按 `doc_id` 过滤后再写 `sentences.jsonl`）。

---

## 2. 用 base LLM 抽取并顺便做 forward 语义对齐是否更好？

**结论：是更好的方向。** 用 base LLM 做抽取，并在同一模型里完成「短句生成 + forward 取 KV」，能同时解决「1374 次 DeepSeek 调用」和「语义空间不一致」两个问题。

### 2.1 当前流程

1. **抽取**：`extract_triples.py`  
   - DeepSeek API：逐条或小 batch 调用 → 输出 (S, R, O)。  
   - 或 base_llm：本地模型、可批量 → 同样只输出 (S, R, O)。
2. **建图**：`build_knowledge_graph.py` → `graph_index.json`。
3. **KV 编译**：`triple_kv_compiler.py`（**base_llm**）  
   - 用**规则**（`_PRED_VERB` 等）从 (S,R,O) 生成**短中文句**；  
   - 对该短句做 **forward**，取 `past_key_values` 存为 KV cache。

因此：

- 若用 **DeepSeek 抽取**：语义在 DeepSeek 空间，KV 在 base_llm 空间，两段语义不一致。
- 若用 **base_llm 抽取**：抽取与 KV 编译都在同一 base_llm 空间，推理时也是该模型，**语义对齐自然在同一个 forward 空间里完成**。

### 2.2 用 base LLM 抽取 + 顺便做「forward 语义对齐」的好处

1. **语义空间一致**  
   抽取、短句表达、KV 编译、推理全在 base_llm 内完成，避免「DeepSeek 理解 → 再交给 Qwen 做 KV」的跨模型偏差。

2. **省掉 1374 次 DeepSeek 调用**  
   本地 base_llm 可**批量**跑（如 `batch_size=8`），不依赖外网 API，无超时/限流，且 GPU 一次加载可复用。

3. **可把「短句」和「forward」更紧地绑在一起**  
   - **方案 A（最小改动）**：  
     - 抽取改用 base_llm（现有 `extract_triples.py --model` 已支持），适当调大 `batch_size`。  
     - triple_kv_compiler 仍用规则生成短句再 forward。  
     - 语义已对齐到 base_llm，只是短句仍由规则生成。  
   - **方案 B（更彻底）**：  
     - 扩展 extract 阶段：base_llm 一次输出 `(S, R, O, short_zh)`，其中 `short_zh` 为模型生成的、≤15 字的注入用短句。  
     - triple_kv_compiler 只负责：读 `short_zh`（或 triples + short_zh），对**该短句**做 forward 存 KV，不再用规则造句。  
     - 这样「语义对齐」在抽取阶段由 base_llm 一次完成，compiler 只做 KV 计算，风格也与推理模型完全一致。

### 2.3 实现上可采取的步骤

- **短期**：  
  - 在 topic 配置或 UI 中优先使用 **base_llm 抽取**（不设 DeepSeek 或显式关掉），并增大 `batch_size`（如 4–8），减少调用次数、避免长时间卡在 DeepSeek。  
  - 在 pipeline 或 UI 上增加说明：「Build Graph 当前使用本 topic 下**全部** blocks（所有 document），共 N 条句子。」  
- **中期**：  
  - 若需要「只对一篇 document 建图」：在 Build Graph 入口增加「按 doc_id 过滤 blocks」的选项（或仅对当前文档的 blocks 建图）。  
  - 设计并实现「extract 输出 (S,R,O,short_zh) + compiler 只 forward short_zh」的 B 方案，把 forward 语义对齐彻底收拢到 base_llm 一条链路里。

以上内容可直接作为「Build Graph 句子来源」和「用 base LLM 做抽取与语义对齐」的设计依据与实现备忘。
