You are asked to implement a research-grade experiment to analyze head-level attention behavior in a transformer-based LLM under two conditions:

(A) Retrieval-Augmented Generation (RAG) baseline
(B) RAG + external KV cache injection (KVI-style)

The goal is to measure how injected KV pairs affect attention distribution across layers and heads, and whether they influence token generation probabilities.

---

1. DATASET

---

Use ONE of the following QA datasets:

- Natural Questions (NQ)
- HotpotQA (preferred for multi-hop)
- MedHopQA (preferred for biomedical reasoning)

Requirements:

- Sample 200–500 examples
- For each example, you must have:
  - question
  - gold answer
  - retrieved evidence passages (simulate RAG)

---

1. MODEL SETUP

---

- Use a HuggingFace causal LLM (e.g., LLaMA, Mistral, or similar)
- Enable:
  - output_attentions=True
  - use_cache=True
- Run inference in greedy or temperature=0 mode

---

1. TWO EXPERIMENT CONDITIONS

---

(A) RAG baseline:

- Input = [retrieved evidence + question]
- Standard forward pass

(B) RAG + KV injection:

- Construct a small set of "knowledge triples" manually or heuristically from evidence:
  (subject, relation, object)
- Convert each triple into a short canonical sentence (≤15 tokens)
- Run a forward pass on these sentences to extract past_key_values (KV cache)
- Inject these KV tensors as prefix (past_key_values) during generation

IMPORTANT:

- Do NOT duplicate the same text in prompt and KV
- KV content must be semantically related but not identical to prompt text

---

1. ATTENTION ANALYSIS

---

For each generated answer, extract attention tensors:

Shape:
  attentions[layer][batch][head][query_position][key_position]

Focus on the LAST generated token (or final answer token).

Define:

- KV tokens: positions corresponding to injected KV prefix
- Prompt tokens: positions corresponding to input text

---

1. METRICS TO IMPLEMENT

---

(1) KV Attention Mass

For each layer l and head h:

```
KV_mass(l,h) = sum(attention over KV positions)
```

Compute:

- mean KV_mass across samples
- per-layer average
- per-head distribution

---

(2) Attention Shift (RAG vs KVI)

For each (layer, head):

```
Δ_attn = KV_mass_KVI - KV_mass_RAG
```

Also compute:

- attention allocated to top-k evidence tokens
- whether attention shifts from irrelevant tokens → KV

---

(3) Head Specialization

Identify top heads with highest KV_mass.

Cluster heads by:

- layer index
- KV sensitivity

Output:

- top 10 heads most affected by KV injection

---

(4) Logit Shift (critical)

For each example:

- Identify the gold answer token (or its first token)
- Compute:
    logit_gain = log P_KVI(token) - log P_RAG(token)

Aggregate:

- mean logit gain
- correlation with KV_mass

---

1. OPTIONAL (STRONGLY RECOMMENDED)

Head Ablation Study:

- Mask KV attention (set attention weights to zero for KV positions)
- Compare:
  - normal KVI
  - KV masked
  - random KV

Measure:

- accuracy drop
- logit drop

---

---

1. VISUALIZATION OUTPUTS (EXTENDED)

---

Generate the following plots:

(1) KV Attention Mass vs Layer

- Line plot showing average KV_mass across layers
- Compare RAG vs KVI

---

(2) Head × Layer Attention Heatmap (CRITICAL)

- 2 heatmaps:
  (A) RAG baseline
  (B) KVI
- Axes:
  x-axis: layer
  y-axis: head
- Values:
  KV attention mass (or normalized attention)

---

(3) Attention Shift Heatmap (NEW, IMPORTANT)

- Plot Δattention = attention_KVI - attention_RAG
- Highlight where attention increases/decreases
- This visualizes redistribution of attention

---

(4) Token-level Attention Visualization (NEW)
For selected examples:

- Show attention from final answer token to:
  - KV tokens
  - prompt tokens
- Visualize as:
  - bar chart or connection diagram
- Compare:
  - RAG (diffuse attention)
  - KVI (focused attention)

---

(5) Logit Gain Distribution

- Histogram or bar plot of logit_gain
- Highlight tokens with highest gain

---

(6) Head Importance Ranking

- Rank heads by KV_mass or Δattention
- Plot top-k heads

---

(7) Attention Trajectory (ADVANCED, OPTIONAL BUT RECOMMENDED)

- Track KV attention mass across generation steps
- x-axis: generation step
- y-axis: KV_mass
- Show how KV influence evolves during decoding

---

1. OUTPUT FORMAT

The code should:

- Be modular and runnable
- Include:
  data loader
  KV builder
  attention extractor
  metrics computation
  plotting functions

Use PyTorch + HuggingFace Transformers.

---

1. GOAL

The experiment should demonstrate:

- Whether KV injection changes attention distribution
- Which heads/layers are responsible
- Whether attention shift leads to improved token probability

---

IMPORTANT CONSTRAINTS:

- Do NOT rely on any custom model modifications
- Use standard HuggingFace APIs only
- Keep KV length small (≤5 triples per example)
- Ensure reproducibility (set random seeds)

---

Now generate full Python code implementing this experiment.

---

## Implementation (this repo)

- **Code**: `experiments/exp04_attention_hotmap/code/`
  - `run_exp04_attention.py` — CLI (data load, RAG vs KVI+prefix, metrics, figures)
  - `exp04_lib.py` — data/triples, prefix `past_key_values`, greedy decode + attentions
  - `exp04_plots.py` — figures (PNG+PDF) and `exp04_metrics.json`
- **Detached run** (nohup):

```bash
export ROOT=/home/zd/dev/KVI
bash experiments/exp04_attention_hotmap/code/run_exp04_detached.sh
# Optional: USE_HF_HOTPOT=1 for real Hotpot paragraphs (needs HF + datasets)
# Optional: AB_RANDOM=1 for random-prefix KV ablation
```

- **Foreground / quick test** (small `limit`, CPU will be slow):

```bash
python experiments/exp04_attention_hotmap/code/run_exp04_attention.py \
  --model /path/to/Qwen2.5-7B-Instruct --limit 8 --device cuda --out_dir experiments/exp04_attention_hotmap/results/smoke
```

- **Data**: default `hotpot_eval.jsonl` (question+answer only) uses **simulated evidence** when `evidence` / `gold_supporting_sentences` are missing; for paper-grade runs use `--use_hf_hotpot` or provide a JSONL with `evidence` or `gold_supporting_sentences`.

- **Outputs**: under `--out_dir`: `exp04_metrics.json`, `fig01_*.pdf` … `fig07_*.pdf` (and `.png`).