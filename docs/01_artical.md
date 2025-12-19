附录 A：论文写作版本（System Motivation & Design Rationale）

本附录内容可直接用于论文中的 Motivation、System Overview 或 Design Rationale 章节，语言风格已从 PRD 转换为学术论文体。

A.1 Motivation

Despite the remarkable progress of large language models (LLMs), their deployment in knowledge-intensive and high-reliability scenarios remains fundamentally constrained by three structural limitations: bounded context windows, implicit and uncontrollable knowledge storage, and weak dependence on precise external evidence during reasoning.

Existing approaches such as prompt engineering and retrieval-augmented generation (RAG) attempt to alleviate these issues by concatenating retrieved text into the input context. However, this strategy treats external knowledge as optional textual hints, rather than as integral components of the model’s internal computation. As a result, LLMs may ignore, partially use, or hallucinate beyond retrieved evidence, especially in tasks requiring strict adherence to factual details, procedural steps, or numerical constraints.

To address these limitations, we propose External Key-Value Injection (KVI), a framework that introduces external knowledge directly into the attention mechanism of frozen LLMs, enabling reliable and controllable knowledge utilization without modifying model parameters.

A.2 Design Rationale
A.2.1 Knowledge as Attention-Level Memory

Transformer attention computes weighted aggregations over key-value pairs derived from previous tokens. From this perspective, long-term memory can be naturally modeled as an extension of the key-value space. Instead of expanding the textual context, KVI encodes external knowledge chunks into key-value representations that are directly consumable by the attention layers.

This design ensures that retrieved knowledge is not merely visible to the model, but actively participates in the attention computation, effectively becoming a first-class citizen in the reasoning process.

A.2.2 Decoupling Knowledge from Model Parameters

Unlike fine-tuning or continual training approaches, KVI explicitly separates knowledge storage from model parameters. The base LLM remains frozen, while domain-specific knowledge is maintained in an external KV Bank.

This decoupling provides three critical advantages:

Knowledge can be updated, removed, or corrected without retraining the model.

Multiple domain-specific knowledge banks can coexist and be dynamically selected.

System behavior becomes more interpretable and auditable.

A.2.3 Efficient Retrieval–Reasoning Integration

Rather than injecting retrieved text into the prompt, KVI employs an ANN-based retriever to select relevant chunks at inference time and projects them into the key-value space compatible with the target LLM.

By avoiding token-level concatenation, KVI significantly reduces context length consumption while improving the utilization efficiency of retrieved knowledge.

A.2.4 Compatibility with Autoregressive Inference

External key-value pairs are treated as static prefix memory, shared across both prefill and decode stages. This design maintains full compatibility with standard KV caching mechanisms while ensuring consistent behavior across different inference phases.

Importantly, external KVs are not written into the model’s internal cache, preserving a clear separation between transient inference state and persistent external memory.

A.3 Comparison with Prior Paradigms
Paradigm	Knowledge Injection Level	Update Cost	Attention-Level Dependency
Prompt Engineering	Textual	Low	Weak
RAG	Textual	Medium	Weak–Implicit
Fine-tuning	Parameter	High	Strong but Static
External KVI (Ours)	Attention (KV)	Low	Strong & Explicit
A.4 Summary

External KVI reframes external knowledge integration as an attention-space augmentation problem rather than a context expansion problem. By injecting external key-value memories into frozen LLMs, the framework enables scalable, controllable, and interpretable knowledge-enhanced reasoning, making it particularly suitable for high-stakes and domain-specific applications.