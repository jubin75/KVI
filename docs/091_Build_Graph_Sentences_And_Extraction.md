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

---

## 3. 三元组质量评估示例（含 compiler 修复）

### 3.1 原始 sentence 与期望

- **原文**：`The molecular weight of the SFTSV Gn monomer was estimated to be approximately 48 kDa.`
- **期望**：三元组 (SFTSV Gn monomer, has_molecular_weight, approximately 48 kDa)，编译成短句如「SFTSV Gn monomer 分子量约为 approximately 48 kDa」或中文等价表述。

### 3.2 你看到的质量问题（修复前）

| 现象 | 原因 |
|------|------|
| `SFTSV Gn monomerhas_molecular_` 粘连、截断、缺宾语 | ① `has_molecular_weight` 未在 `_PRED_VERB` 中，直接用英文谓词；② `_build_triple_sentence` 用 `subject+verb+obj` 无分隔，且按 30 字符截断，导致宾语丢失。 |
| `immunoassaysutilizesantigens d`、`rGn-ELISAdetectsSFTSV Gn antib` | 同上：`utilizes`、`detects` 未映射为中文动词，且无分隔导致粘连，截断在词中间。 |
| `SFTSV导致unexpected SFTSV infect` | 抽取为英文片段，且「infect」被截断；设计上希望短句尽量中文，数值/单位可保留。 |

### 3.3 已做代码修改（triple_kv_compiler.py）

1. **补全谓词映射与层映射**  
   - `_PRED_VERB` 新增：`has_molecular_weight` → 「分子量约为」、`utilizes` → 「利用」、`detects` → 「检测」。  
   - `RELATION_LAYER_MAP` 为上述关系分配 (0, 7)。

2. **短句生成方式**  
   - 由 `subject+verb+obj` 无分隔改为用空格拼接：`" ".join([subject, verb, obj])`，避免「monomerhas_molecular_weight」式粘连。  
   - 截断由 30 字符改为 45 字符，减少在词中间截断、保留「48 kDa」等宾语。

重跑 Build Graph（当前文档）后，同一句应得到类似：  
`SFTSV Gn monomer 分子量约为 approximately 48 kDa`（或略截断在 45 字符内），质量会明显好于修复前。

---

## 4. 知识图谱格式是否为 (S, R, O, I)？

**结论：是。** 当前抽取与建图在语义上就是 **(S, R, O, I)**，其中 **I = 句子索引（sentence_id，可追溯来源句）**；只是存储时没有四个并列字段，而是把 I 放在 `provenance` 和图索引里。

### 4.1 各层存储方式

| 层级 | 格式 | I 的存放位置 |
|------|------|----------------|
| **Triple 结构**（`src/graph/schema.py`） | subject, predicate, object, triple_id, subject_type, object_type, confidence, **provenance** | `provenance["sentence_id"]`；provenance 还可含 sentence_text、source_block_id、source_doc_id |
| **triples.jsonl**（抽取输出） | 每行一个 Triple 的 `to_dict()` | 同左：`provenance.sentence_id` 即 I |
| **graph_index.json**（建图结果） | nodes, triples, entity_index, **sentence_index**, **triple_sentence_index** | `triple_sentence_index[triple_id] = [sentence_id]`；`sentence_index[sentence_id]` 存该句 text、source_block_id、triple_ids |

### 4.2 数据流

1. **抽取**（`triple_extractor.py`）：从句子列表得到 Triple，每个 Triple 带 `provenance = {sentence_id, sentence_text, source_block_id, source_doc_id}`，即 **(S, R, O) + I（及来源信息）**。
2. **写入 triples.jsonl**：`Triple.to_dict()` 含 subject, predicate, object, provenance，故 **I 在 provenance 里**。
3. **建图**（`knowledge_graph.py`）：读 triples，用 `triple.provenance["sentence_id"]` 建 `triple_sentence_index` 和 `sentence_index`，检索时可从 triple_id 反查 sentence_id 和原文。

因此可以确认：**抽取与知识图谱格式等价于 (S, R, O, I)**，I 用于句子级溯源与多跳检索时的证据回链。

---

## 5. 单文档建图：句子来源与实体过滤

### 5.1 当前文档的句子来源（含 Abstract 与 Key Notes）

当使用 **Build Graph（当前文档）**（即带 `doc_id`）时，Step1 的句子列表由两部分组成：

1. **blocks**：从 `blocks.evidence.jsonl`（或 enriched/jsonl）中按 `doc_id` 过滤得到的 block 文本。
2. **View Details 内容**：从该 topic 的 `docs.details.json` 中读取该 doc 的 **Abstract** 和 **Key Notes**，各作为一条/多条 sentence 追加进列表（Abstract 一条，每条 Key Note 一条），再一起参与后续 tagging 与三元组抽取。

因此「当前文档」建图时，三元组会来自：PDF 抽取得到的 blocks **以及** 你在 View Details 里保存的摘要与要点；若只看到 5 条 triple 而 Abstract 很长，多半是之前未把 Abstract 纳入句子来源——现已修复为会纳入。

### 5.2 只保留本文出现的实体（entities_from_triples_only）

当使用 **Build Graph（当前文档）** 时，建图会传入 `--entities_from_triples_only`：

- **行为**：只保留在**本批 triples** 中作为 subject 或 object 出现过的实体；aliases 里若某条记录的 `canonical` 不在这些实体中，整条记录会被跳过，**不会**为该 canonical 创建新节点。
- **效果**：例如 aliases 里配置了 安徽、山东、河南、湖北，但当前文档的 5 条 triples 里没有出现这些省份，则图中不会再有这 4 个节点，KV 列表里也不会出现这些 anchor。
- **实现**：`build_knowledge_graph.py` 支持 `--entities_from_triples_only`；`build_graph_from_triples_jsonl(..., entities_from_triples_only=True)` 时，加载 aliases 只处理 `canonical in builder._entities` 的记录。
