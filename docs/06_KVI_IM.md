# RIM v0.3 —— KVI + Reasoning Introspection Module（单文件设计 Prompt）

## 0. 设计目标

在 **不修改 Base LLM 权重**、**不修改 KVI / KV Bank 实现** 的前提下，
补齐当前工程体系中**“中期推理状态无法重新对齐外部知识”**的结构性缺陷。

核心目标是让以下计算在推理中期真实发生：

q_t′ · K_ext

其中：
- q_t′ 来源于 **中期 reasoning state**
- 而非初始用户 query

---

## 1. 已知系统前提（不可违背）

- Base LLM：冻结参数
- KVI（Key-Value Injection）：已实现
- KV Bank：已存在，存储为 (K_ext, V_ext)
- 初始检索：仅发生一次（基于初始 query）
- 当前系统 **不存在** reasoning-aware re-alignment

---

## 2. 现有 KVI 的必然缺陷

现有流程为：

User Query
→ 初始 embedding
→ 检索 KV Bank
→ 注入 K_ext / V_ext
→ LLM 完整推理

若满足以下任一条件，则系统必然失败：

- 初始 query 与关键 evidence 语义不一致
- 关键概念仅在推理中期显现
- 推理路径发生语义转向（reframing）

此时即使模型“想到了正确方向”，
也**无法再次命中 KV Bank**。

---

## 3. RIM v0.3 的核心思想

> Knowledge relevance 是 **推理状态依赖的，而非 query 静态属性**

因此需要一个 **不生成 token 的内省模块**：

- 观察推理过程
- 判断是否发生 reasoning shift
- 用 reasoning state 构造新 query
- 重新对齐 KV Bank
- 通过 KVI 进行增量注入

---

## 4. RIM 在系统中的位置

LLM (decoding)
├─ hidden states
├─ generated tokens
↓
RIM.observe(...)
↓
RIM.should_realign()
↓
RIM.retrieve_additional_kv(q_t′)
↓
KVI.inject_incremental(K_ext_new, V_ext_new)
↓
LLM.continue_decoding()


---

## 5. 推理状态（Reasoning State）定义

### 5.1 采集位置

- Decoder 中后段层（如 L/2 或 L-1）
- 最近 N 个已生成 token

### 5.2 表示形式

H_i ∈ R^{N × d_model}

### 5.3 Reasoning Query 构造

允许方式：
- Mean Pooling
- Attention Pooling
- Last-token projection

输出：

q_t′ ∈ R^{d_model}

---

## 6. Reasoning Shift 判定（必须实现）

至少实现一种信号：

### 6.1 表征漂移

cosine_distance(q_0, q_t′) > τ

### 6.2 不确定性信号（可选）

- Next-token entropy
- Logit flattening

### 6.3 显式推理标志（可选）

- therefore / however / reconsider / indirectly 等

输出：

RETRIEVE_MORE ∈ {true, false}
confidence ∈ [0,1]

---

## 7. KV Re-alignment 机制

当 RETRIEVE_MORE == true：

- 使用 q_t′ 作为检索向量
- 在 KV Bank embedding 空间中进行 ANN / Exact Search
- 选取 top-k **未注入过**的 KV blocks

注意：
- 不是文本检索
- 不使用 prompt
- 不生成自然语言

---

## 8. 增量 KV 注入约束

- 仅注入新增 KV（去重）
- 沿 sequence 维度 concat
- 保持 head_dim / num_heads 一致
- 最大 re-injection 次数可配置（默认 ≤ 1）

---

## 9. RIM 接口定义（必须遵守）

```python
class RIM:
    def observe(self, hidden_states, generated_tokens): ...
    def should_realign(self) -> bool: ...
    def build_reasoning_query(self) -> Tensor: ...
    def retrieve_additional_kv(self, query_vec) -> KVBlocks: ...
    def inject(self, kv_blocks, past_key_values): ...
```

## 10. 参考伪代码
```python
if step == 0:
    K_ext, V_ext = KVI.initial_retrieval(query)

for t in decoding_steps:
    logits, hidden_states = model.step(...)

    RIM.observe(hidden_states, generated_tokens)

    if RIM.should_realign() and not RIM.exceeded_budget():
        q_t_prime = RIM.build_reasoning_query()
        new_kv = RIM.retrieve_additional_kv(q_t_prime)
        past_key_values = KVI.inject_incremental(
            past_key_values, new_kv
        )

    token = sample(logits)
```

---

## 11. 非目标（严禁实现）

❌ 模型参数编辑（ROME / MEMIT）

❌ 改写 attention 公式

❌ 强制 chain-of-thought 输出

❌ 文本级 RAG 替代 KV 注入

---

## 12. 一句话总结

KVI 决定“模型能看到什么记忆”
RIM 决定“模型在什么时候重新找记忆”

RIM 是 KVI 从“静态注入”走向“推理感知注入”的最小必需模块。

---

## 13. 落地到本仓库 external_kv_injection 的最小实现清单（代码映射）

> 目标：把本文档里的 RIM v0.3 设计，映射到当前仓库已有的“检索/注入/解码循环”实现上，给出最小改动路径与对接点（不引入文本级 RAG，不改 attention forward）。

### 13.1 模块与职责对应关系（直接对齐现有文件）

- **KV Bank（向量检索 + 存 K_ext/V_ext）**：`external_kv_injection/src/kv_bank.py`（统一入口） → `external_kv_injection/src/vector_store/faiss_kv_bank.py`（FAISS 后端，`search()` 返回 `KVItem(K_ext,V_ext,meta,score)`）
- **Retriever（薄封装 + debug）**：`external_kv_injection/src/retriever.py`（`Retriever.search(query_vec, top_k, filters, query_text)`）
- **KVI 注入器（past_key_values 前缀注入，不改 attention forward）**：
  - 低层拼接：`external_kv_injection/src/runtime/hf_cache_prefix_injection.py`（`build_past_key_values_prefix` / `stack_ext_kv_items_by_layer`）
  - 上层策略（concat / gate 近似）：`external_kv_injection/src/kv_injector.py`（`KVInjector.build_past_key_values(...)`）
- **解码循环（能拿到 hidden_states / 能接入 past_key_values）**：`external_kv_injection/src/runtime/multistep_injector.py`
  - 这里已经有“手写 greedy decode + prefix cache”路径（避免 HF `generate(past_key_values=...)` 的 past_length slicing 问题）

### 13.2 RIM 五个接口在本仓库里的最小对接点

把 RIM 看作一个“插在解码循环旁边的 controller”，在本仓库里最小可以这样落位（推荐直接复用 `multistep_injector.py` 的现成能力）：

- **`observe(hidden_states, generated_tokens)`**：
  - 位置：`runtime/multistep_injector.py` 的手写解码循环里（每步 forward 已经打开 `output_hidden_states=True`）。
  - 输入：优先用 `out.hidden_states[-1]`（last layer hidden），可选窗口为最近 N token。
- **`build_reasoning_query()`**：
  - 最小实现：直接复用 `multistep_injector.py` 里“从当前状态构造 query_vec”的逻辑：
    - 若调用方提供 `query_embed_fn(text)->np.ndarray`，用它（对应“中期状态→query”的可插拔编码器）。
    - 否则 fallback：对 `last_hidden` 做 mask mean-pool 得到向量（demo/近似版）。
- **`should_realign()`**（Reasoning shift 判定）：
  - 你文档主信号是 `cosine_distance(q_0, q'_t) > τ`；在本仓库中可对齐为：
    - 用 step=0 的 `query_vec` 作为 `q_0`，后续 step 的 `query_vec` 作为 `q'_t`；
    - 用余弦距离阈值决定是否触发一次“额外检索+注入”（预算默认 ≤1）。
  - 另外，本仓库 `MultiStepConfig` 已有一些 stop/稳定性阈值（`min_hidden_delta/min_logit_delta/...`），可作为“可选信号”来源，但不要把它们变成语义协议（保持纯控制信号）。
- **`retrieve_additional_kv(query_vec)`**：
  - 直接调用 `Retriever.search(query_vec, top_k=...)` 获取 `KVItem` 列表即可（`KVItem` 已自带 `K_ext/V_ext/meta/score`）。
  - 去重/“未注入过”：
    - 现成机制：`runtime/multistep_injector.py` 里已有 `self.used_block_ids: Set[str]`，会跳过已用 `block_id/chunk_id` 的条目。
- **`inject(kv_blocks, past_key_values)`**：
  - 最小实现：把本次新增的 `KVItem` 通过 `stack_ext_kv_items_by_layer` 组装成各层 `ExtKV`，再用 `build_past_key_values_prefix` 构造新的 prefix cache。
  - 若需要策略化（concat/gate）：走 `KVInjector.build_past_key_values(...)`。

### 13.3 “严格按 RIM v0.3（只允许 0→1 次再对齐）”的最小落地方式

当前仓库已有的 `MultiStepInjector` 是“每步都可检索/注入”的更强形式；要贴合本文档的 v0.3（初始检索一次 + 最多一次再对齐），最小可以做到：

- **配置上限制步数/预算**：
  - `max_steps = 2`（step=0 初始注入；step=1 仅在 `should_realign()==True` 时检索+注入）
  - 同时保留 `used_block_ids` 去重，确保 step=1 只注入“未注入过”的 blocks
- **把触发条件改为 shift-gated**：
  - step=1 的检索入口由 `should_realign()` 控制（而不是“无条件每步检索”）

### 13.4 最小需要补齐/显式化的配置项（建议写进 cfg）

- **shift 判定**：`tau_cos_dist`、窗口 `N`、（可选）熵/不确定性阈值
- **检索**：`top_k`、filters（可选）
- **注入**：`inject_layers`、`max_reinjections`（预算）、`min_kv_len_to_inject`、每步 token cap
- **去重键**：优先 `block_id`，fallback `chunk_id/id`（与现有代码一致）

---

## 14. Demo 场景（示例）：Self-Critique 触发一次再对齐（以 SFTSV 问题举例）

输入问题（示例，仅用于演示；**不代表系统只支持 SFTSV**）：

> 目前有哪些 FDA 已批准药物被研究用于 SFTSV？

期望流程（A/B 对比）：

1. base LLM **首轮生成回答**（无注入）
2. RIM Self-Critique 判断：**低 confidence + 医学事实** → 触发检索
3. RIM 检索 KV Bank（预存 SFTSV 文献 KV，返回可注入 `K_ext/V_ext`）
4. 注入 KV prefix（`past_key_values` 前缀注入，不改 attention forward）
5. **第二轮生成**
6. 输出对比：
   - 无 RIM
   - 有 RIM

### 14.1 对应到本仓库的最小可运行脚本

仓库已提供一个“2-pass + critique gating”的最小 demo 脚本：

- `external_kv_injection/scripts/run_rim_self_critique_demo.py`

它会打印：

- `=== 无 RIM（Base LLM）===`：首轮回答
- `=== RIM Self-Critique ===`：JSON（is_medical_fact / confidence / trigger）
- `=== RIM 检索（KV Bank）===`：检索 debug + top-k block ids
- `=== 有 RIM（注入 KV → 第二轮生成）===`：二轮回答

### 14.2 运行命令（topic mode / 或直接 --kv_dir）

说明：

- `--topic sftsv --topic_work_dir ...` 会自动在以下常见布局中寻找 `kvbank_blocks/manifest.json`：
  - `<topic_work_dir>/sftsv/kvbank_blocks`
  - `<topic_work_dir>/sftsv/work/kvbank_blocks`
  - `<topic_work_dir>/SFTSV/kvbank_blocks`
  - `<topic_work_dir>/SFTSV/work/kvbank_blocks`
- `--domain_encoder_model` **必须与构建该 KVBank 的 retrieval encoder 一致**，否则检索向量空间不匹配。

示例（topic mode）：

```bash
python external_kv_injection/scripts/run_rim_self_critique_demo.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --topic sftsv \
  --topic_work_dir /home/jb/KVI/topics \
  --domain_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --use_chat_template \
  --self_critique_mode llm_json \
  --critique_conf_threshold 0.55 \
  --top_k 8 \
  --layers 0,1,2,3 \
  --prompt "目前有哪些 FDA 已批准药物被研究用于 SFTSV？"

示例（直接指定任意 KVBank，不依赖 topic 名称）：

```bash
python external_kv_injection/scripts/run_rim_self_critique_demo.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --kv_dir /path/to/your/kvbank_blocks \
  --domain_encoder_model sentence-transformers/all-MiniLM-L6-v2 \
  --use_chat_template \
  --self_critique_mode llm_json \
  --critique_conf_threshold 0.55 \
  --top_k 8 \
  --layers 0,1,2,3 \
  --prompt "你的问题..."
```
```

### 14.3 关于“RIM 不生成 token”与 demo 的关系（重要）

本文档第 3 节定义的 RIM 理想形态是“不生成 token 的内省模块”。但为了满足 Demo 可复现与易用性：

- `run_rim_self_critique_demo.py` 允许用 `--self_critique_mode llm_json` 让 base LLM 生成一段 **JSON 评估**（仅用于触发检索的控制信号）
- 若你要保持“RIM 不生成 token”的严格形态，可改用 `--self_critique_mode heuristic`（脚本内置简单启发式），或把 Self-Critique 换成外部分类器/规则引擎（不影响后续检索/注入链路）
