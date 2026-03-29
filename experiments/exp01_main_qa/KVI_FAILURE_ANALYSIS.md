# Why KVI can score below RAG / GraphRAG (smoke100 观察)

## 1. 检索通道不一致（混淆因素）

| 方法 | 检索 | 生成 |
|------|------|------|
| **RAG** | **ANN**（dense） | Prompt 里拼证据（`modeA_rag`） |
| **GraphRAG** | **Graph + 文本回退** | 同图推理，无 KV |
| **KVI** | Graph + 文本回退 | **KV 注入 + Prompt 证据**（双通道） |

因此「RAG > KVI」**不一定**说明 KV 有害：也可能是 **ANN 检索** 在 synthetic 工件上比 **图检索** 更贴 Hotpot 的句子分布。

## 2. 提示词域错配（已修复）

`run_graph_inference.py` 默认使用 **中文医学助手** system prompt 与「医学常识」指令，而 Hotpot 为 **英文开放域**。Graph/KVI 共用该脚本时，会在错误先验下生成；**KVI 额外注入 KV**，在错误域上可能放大注意力偏转。

**修复**：为 Exp01 默认开启 `--openqa_mode`（`run_exp01.py`），使用英文开放域指令与 baseline。

## 3. 双通道与 synthetic 图谱

- 当前 triple / graph 来自 **QA 合成工件**，实体与真实 Wiki 对齐弱；图游走可能命中 **噪声三元组**，预编译 KV 进入 attention 后表现为「偏转」。
- 文档 `docs/01_Article.md` 中的双通道设计假设 **短 KV + 与 prompt 措辞不重复**；synthetic 数据上易违反。

## 4. 建议实验顺序（与 Exp3 / Exp6 对齐）

1. **固定提示**：始终 `--openqa_mode` 再比较 KVI vs RAG。
2. **消融 KV**：`--max_kv_triples 0`（无 KV 注入）或 `--enable_kvi` 关闭，对照「仅 Graph 通道」。
3. **消融双通道 prompt**：`--kvi_minimal_prompt`（仅注入 KV 时去掉 prompt 里长证据列表），检验「纯 KV + 问题」是否减轻偏转。
4. **Exp3**：用 `gold_supporting_sentences` 与 `experiments/exp03_retrieval_quality/code/run_exp03_retrieval.py` 报告 ANN vs Graph 的 Recall@k / MRR。
5. **Exp6**：对 `run_graph_inference` 组合 `drm_threshold`、`max_kv_triples`、`openqa_mode` 做表格化消融（见 `EXP03_EXP06.md`）。
