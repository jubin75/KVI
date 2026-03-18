实验组合 + 展示形式

下面给你完整设计。

Experiment 1
Main QA Performance

目的：

KVI 是否提升回答准确率
展示形式

表格

因为 reviewer 要看精确数值。

示例
Method	Retrieval	Injection	HotpotQA EM	NQ EM
LLM	none	none	32.4	28.1
RAG	ANN	prompt	55.2	47.8
GraphRAG	graph	prompt	58.7	50.3
KV Prefix	ANN	KV	57.9	49.1
KVI	graph	KV + prompt	66.4	56.2

说明：

KVI outperform baselines
Experiment 2
Hallucination Reduction

目的：

KVI 是否减少 hallucination

数据：

TruthfulQA

FEVER

展示形式

柱状图

因为 reviewer 更容易直观看出差距。

图结构

横轴：

Method

纵轴：

Hallucination Rate (%)

示意：

Hallucination Rate

LLM        █████████████████
RAG        ███████████
KV Prefix  █████████
KVI        █████

说明：

structured retrieval suppresses hallucination
Experiment 3
Retrieval Quality

目的：

Graph retrieval 是否更好

数据：

HotpotQA

展示形式

表格

Retrieval	Recall@5	Recall@10	MRR
ANN	58.1	64.7	0.41
Graph	71.5	78.2	0.54

说明：

graph retrieval improves evidence recall
Experiment 4
KV Prefix vs KVI

这是 reviewer 最关注的实验。

展示形式

表格

Method	Evidence Type	Accuracy	Hallucination
KV Prefix	raw text	57.9	26.3
KVI	triples	66.4	14.2

说明：

structured triples outperform raw text prefix
Experiment 5
KV Length Analysis

目的：

验证 short triple 设计
展示形式

折线图

横轴：

KV token length

纵轴：

Hallucination rate

示意：

Hallucination
30 |        *
25 |     *
20 |   *
15 | *
10 |
     0  20  50  200
        KV tokens

解释：

long KV behaves like prompt
Experiment 6
Ablation Study
展示形式

表格

Variant	Accuracy	Hallucination
Full KVI	66.4	14.2
– graph retrieval	60.1	21.7
– DRM	62.8	19.4
– KV injection	58.5	24.9
– dual channel	57.1	26.2

说明：

each module contributes
Experiment 7
Model Generalization

模型：

Qwen2.5 7B

Mistral 7B

展示形式

表格

Model	Method	Accuracy
Qwen2.5	RAG	55.2
Qwen2.5	KVI	66.4
Mistral	RAG	51.0
Mistral	KVI	62.7


Experiment 7
Attention heatmap

Attention Distribution Visualization
图名建议
Figure X: Attention distribution under different knowledge injection strategies
图类型


对比方法
方法
RAG
KV Prefix
KVI
横轴

token sequence：

[Query tokens] [Knowledge tokens] [Generated answer]
纵轴
Attention heads
图的含义

预期结果：

RAG

attention mainly in prompt evidence
weak structural guidance

KV Prefix

attention scattered
long text prefix dominates

KVI

attention concentrated on triples
stable reasoning path

简化示意：

RAG
████░░░░░░░░░░░░░

KV Prefix
███░██░░██░░░░░░

KVI
████████░░░░░░░░