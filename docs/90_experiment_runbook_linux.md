# Linux(/home/jb/KVI) Experiment Runbook: Dataset → Training → Testing (Reproducible)

> Convention: All commands are executed under `/home/jb/KVI`.
> Goal: Follow this runbook to go from scratch: PDF → raw context → blocks → KVBank → (optional) projector/gate training → single-step/multi-step injection testing.
> **Architecture spec**: slot-aware schema injection follows `docs/slot_enum.md`; schema code review follows `docs/73_schema_code_review.md`.

## 0) Install Dependencies (Python + System Packages)

### 0.1 Python Environment

```bash
cd /home/jb/KVI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

If you are using an older version of `transformers`, loading Qwen2/DeepSeek may produce:
`Tokenizer class Qwen2Tokenizer does not exist ...`. Recommended upgrade:

```bash
pip install -U "transformers>=4.41" accelerate safetensors tokenizers sentencepiece
```

### 0.2 System Dependencies (OCR)

If you need to process scanned PDFs (OCR), install `tesseract`:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
```

## 1) Data Construction (Production Recommended): PDF → raw_chunks(4096) → blocks(256) → KVBank

### 1.1 Set Experiment Parameters

```bash
# Absolute paths recommended; if you've cd /home/jb/KVI, relative ./pdfs also works
export PDF_DIR="/home/jb/KVI/pdfs"
export WORK_DIR="/home/jb/KVI/_exp_prod"
export BASE_LLM="Qwen/Qwen2.5-7B-Instruct"
export DOMAIN_ENCODER="sentence-transformers/all-MiniLM-L6-v2"

# DeepSeek (knowledge density filtering)
export DEEPSEEK_API_KEY="sk-bc1bf3f7edd344c69ca74b2279340434"
```

### 1.1.1 (Highly Recommended) Topic-based Model: Use Separate KBs for Datasets Moving Forward

> Conclusion: External KV injection is high-gain, low-tolerance; **irrelevant corpora / noisy block hits** get amplified by injection.
> Therefore we recommend splitting a "large medical corpus" into multiple "topic KBs" (SFTSV, SARS‑CoV‑2, etc.), and only load/build one topic KB at a time.

There are two equivalent pipelines:

- **Recommended mainline (more convenient)**: `rebuild_topic_kvbank_from_config.py`
  One command completes: doc-level DeepSeek (abstract) filtering → raw_chunks → blocks → KVBank (optional split_tables)
- **Debug mainline (decomposable)**: `build_topic_pdf_subset_deepseek.py` → `build_raw_context_from_pdfs.py` → `build_kvbank_from_pdf_dir_multistep.py`
  Suitable for checking which step (filtering/extraction/chunking/KVBank construction) has a problem.

Relationship between the two pipelines:
- `rebuild_topic_kvbank_from_config.py` essentially **chains** the "debug mainline" steps (internally calls doc-filter and directly invokes pipeline functions to produce raw_chunks/blocks/KVBank).
- Therefore: **results are equivalent**; `rebuild` is more one-click; the three-stage approach is easier for pinpointing issues.

### 1.1.2 (You Only Need to Edit config.json) Topic Config File Locations and Directory Conventions

> Convention: This runbook targets your remote directory (flat layout) `/home/jb/KVI`.
> If in another environment you have a monorepo (with `external_kv_injection/` subdirectory under the repo root), just prepend the relative paths below with that prefix.

Template locations (remote `/home/jb/KVI`):
- `config/topics/SFTSV/config.json`
- `config/topics/SARS2/config.json`

Recommended directory structure (already in use):
- **Topic source PDFs** (manually organized or symlinked):
  - SFTSV: `/home/jb/KVI/pdfs/sftsvpdf` (or `/home/jb/KVI/pdfs/SFTSV`)
  - SARS2: `/home/jb/KVI/pdfs/sarspdf` (or `/home/jb/KVI/pdfs/SARS2`)
- **Topic output directories**:
  - SFTSV: `/home/jb/KVI/topics/SFTSV/`
  - SARS2: `/home/jb/KVI/topics/SARS2/`

Important notes (duplicate PDFs / symlinks / results writing):
- Doc-level filtering defaults to `mode=symlink`: under `out_pdf_dir` you'll see **KEEP PDF symlinks** pointing to `source_pdf_dir` (saves disk, faster).
- If a directory has duplicate PDF names: `dedupe_by_basename=true` will skip duplicates and mark `DUPLICATE/SKIP` in results.jsonl.
- **results_jsonl defaults to append mode**: convenient for preserving history. If you want a "clean result set" every rerun, add this in the `doc_filter` section of config:
  - `"overwrite_results": true`

### 1.1.3 (UI Workflow Prerequisite) PDF → raw_chunks → blocks.evidence.jsonl (Literature Import Data Preparation)

> Goal: Prepare data for the KVI Console UI's **Literature Import → Build Graph** feature.
> Build Graph requires `blocks.evidence.jsonl` to compile a full triple KV Bank.
> If the work dir does not contain this file, run the following two steps first.

**Step 1: PDF → raw_chunks.jsonl**

> Note: `pdf_to_raw_context_chunks.py` is a library module (no CLI entrypoint); must be invoked via `python -c`.

```bash
cd /home/jb/KVI

python -c "
from pathlib import Path
from src.pipelines.pdf_to_raw_context_chunks import build_raw_context_chunks_from_pdf_dir, RawChunkConfig
cfg = RawChunkConfig(tokenizer_name_or_path='Qwen/Qwen2.5-7B-Instruct', chunk_tokens=4096)
n = build_raw_context_chunks_from_pdf_dir(
    pdf_dir=Path('/home/jb/topics/SFTSV/pdfs'),
    out_jsonl=Path('/home/jb/topics/SFTSV/work/raw_chunks.jsonl'),
    cfg=cfg,
)
print(f'Done: {n} chunks written')
"

# Verify
wc -l /home/jb/topics/SFTSV/work/raw_chunks.jsonl
```

**Step 2: raw_chunks → blocks.evidence.jsonl (DeepSeek extraction, ~5-15 minutes)**

> **`--topic_goal` parameter explanation**: This text is directly embedded into the prompt sent to DeepSeek (see `USER_TEMPLATE` in `src/llm_filter/extractive_evidence.py`),
> serving as extraction guidance: when DeepSeek reads each PDF paragraph, it decides whether to keep (`keep=true/false`) and which evidence sentences to extract based on topic_goal.
>
> The keywords after the colon **determine the coverage of evidence**:
> - `transmission routes` → keeps sentences about tick bites, blood contact, etc.
> - `clinical symptoms` → keeps descriptions like fever, thrombocytopenia
> - `pathogenesis` → keeps descriptions like immune response, cytokine storm
> - `treatment and prevention` → keeps mentions of favipiravir, symptomatic treatment
> - Paragraphs unrelated to all keywords → `keep=false`, skipped without extraction
>
> **The more comprehensive the keywords, the broader the evidence coverage from DeepSeek extraction.**
> If only `"SFTSV topic"` is specified, epidemiological or treatment-related evidence may be missed.

```bash
cd /home/jb/KVI

python scripts/build_evidence_blocks_from_raw_chunks_jsonl_deepseek.py \
  --raw_chunks_jsonl /home/jb/topics/SFTSV/work/raw_chunks.jsonl \
  --out_jsonl /home/jb/topics/SFTSV/work/blocks.evidence.jsonl \
  --out_docs_meta_jsonl /home/jb/topics/SFTSV/work/docs.meta.jsonl \
  --kb_id SFTSV \
  --topic_goal "Build a topic knowledge base for SFTSV (Severe Fever with Thrombocytopenia Syndrome): transmission routes, host vectors, epidemiology, clinical symptoms, pathogenesis, diagnostic points, treatment and prevention"

# Verify
wc -l /home/jb/topics/SFTSV/work/blocks.evidence.jsonl
wc -l /home/jb/topics/SFTSV/work/docs.meta.jsonl
```

Upon completion:
- `blocks.evidence.jsonl`: complete evidence extracted from PDFs (expected hundreds of entries)
- `docs.meta.jsonl`: document-level metadata (title, DOI, year, etc.)
- UI Literature Import page can display the document list
- **Build Graph will compile a triple KV Bank from evidence of ALL documents in this topic** (i.e., all blocks in the entire `blocks.evidence.jsonl` participate in extraction, not from a single document). Sentence count = total blocks across all PDFs in this topic. See `docs/091_Build_Graph_Sentences_And_Extraction.md` for details.

**Substituting parameters for other Topics**: Modify the paths and target descriptions in `--pdf_dir` / `--out_jsonl` / `--topic_goal`.

### 1.2 (Recommended First) Quick Validation: Run Only PDF → raw_chunks to Ensure Extraction/Parsing Works

If this step fails, it indicates a PDF extraction/OCR/dependency issue (not a block segmentation issue).

```bash
python scripts/build_raw_context_from_pdfs.py \
  --pdf_dir "$PDF_DIR" \
  --out "$WORK_DIR/raw_chunks.jsonl" \
  --tokenizer "$BASE_LLM" \
  --chunk_tokens 4096 \
  --chunk_overlap 256 \
  --ocr auto \
  --knowledge_filter \
  --deepseek_model deepseek-chat
```

### 1.2 One-Click Build: raw context + KVBank (Table-First + DeepSeek Filtering)

Note: If `--knowledge_filter` is enabled, a DeepSeek API call is made for **each paragraph** (serial), which may be slow under network/rate-limiting conditions and CPU/GPU load will be low; this is normal.
It is recommended to first verify the pipeline without `--knowledge_filter` to confirm PDF extraction and chunking work, then enable filtering later.

```bash
python scripts/build_kvbank_from_pdf_dir_multistep.py \
  --pdf_dir "$PDF_DIR" \
  --work_dir "$WORK_DIR" \
  --base_llm "$BASE_LLM" \
  --retrieval_encoder_model "$DOMAIN_ENCODER" \
  --layers 0,1,2,3 \
  --chunk_tokens 4096 \
  --chunk_overlap 256 \
  --block_tokens 256 \
  --keep_last_incomplete_block \
  --ocr auto \
  --knowledge_filter \
  --deepseek_model deepseek-chat
```

### 1.2.1 (Topic-based) One-Click Build: SFTSV / SARS‑CoV‑2 Two Topic KV Banks

> Recommended directory structure:
> - SFTSV: `$WORK_DIR/topics/sftsv/...`
> - SARS‑CoV‑2: `$WORK_DIR/topics/sarscov2/...`

```bash
export WORK_DIR_TOPIC="$WORK_DIR/topics"

python -u scripts/build_kvbank_from_pdf_dir_multistep.py \
  --pdf_dir "$PDF_DIR_SFTSV" \
  --work_dir "$WORK_DIR_TOPIC/sftsv" \
  --base_llm "$BASE_LLM" \
  --retrieval_encoder_model "$DOMAIN_ENCODER" \
  --layers 0,1,2,3 \
  --chunk_tokens 4096 --chunk_overlap 256 \
  --block_tokens 256 --block_overlap_tokens 64 --keep_last_incomplete_block \
  --ocr auto \
  --knowledge_filter --deepseek_model deepseek-chat \
  --split_tables \
  --shard_size 1024

python -u scripts/build_kvbank_from_pdf_dir_multistep.py \
  --pdf_dir "$PDF_DIR_SARSCOV2" \
  --work_dir "$WORK_DIR_TOPIC/sarscov2" \
  --base_llm "$BASE_LLM" \
  --retrieval_encoder_model "$DOMAIN_ENCODER" \
  --layers 0,1,2,3 \
  --chunk_tokens 4096 --chunk_overlap 256 \
  --block_tokens 256 --block_overlap_tokens 64 --keep_last_incomplete_block \
  --ocr auto \
  --knowledge_filter --deepseek_model deepseek-chat \
  --split_tables \
  --shard_size 1024
```

### 1.2.2 (Recommended Mainline) One Command to Rebuild Topic KB: doc-level DS (abstract) → pipeline → KVBank

> You only need to modify in config.json:
> - `goal`: the goal for building the topic KB (user input)
> - `extract_tables`: whether to extract tables
> - Paths: `source_pdf_dir / out_pdf_dir / build.work_dir`

```bash
# SFTSV topic KB (output dir: /home/jb/KVI/topics/SFTSV)
python -u scripts/rebuild_topic_kvbank_from_config.py \
  --config config/topics/SFTSV/config.json

# SARS2 topic KB (output dir: /home/jb/KVI/topics/SARS2)
python -u scripts/rebuild_topic_kvbank_from_config.py \
  --config config/topics/SARS2/config.json
```

Optional: Start from a completely clean state (delete only target artifacts)

If you want to ensure no old files interfere (not required), delete the two evidence-related items before rebuilding:

```bash
rm -rf /home/jb/KVI/topics/SFTSV/work/kvbank_evidence
rm -f /home/jb/KVI/topics/SFTSV/work/blocks.evidence.jsonl
```

Upon completion, how to judge success (full build from scratch + dual KB, 6 hard artifacts total):
- `topics/<TOPIC>/doc_filter_results.jsonl`: doc-level filtering records (KEEP/DROP/UNCERTAIN)
- `topics/<TOPIC>/pdfs/`: KEEP PDFs (default symlinked)
- `topics/<TOPIC>/work/raw_chunks.jsonl`, `topics/<TOPIC>/work/blocks.jsonl`
- `topics/<TOPIC>/work/kvbank_blocks/manifest.json` (and `kvbank_tables/manifest.json` if `split_tables=true`)
- `topics/<TOPIC>/work/blocks.evidence.jsonl` (DeepSeek extractive evidence sentences, recommended to extract from raw_chunks paragraphs)
- `topics/<TOPIC>/work/kvbank_evidence/manifest.json` (evidence KVBank)

#### 1.2.2.1 (New, Recommended) Evidence Version: blocks.evidence + kvbank_evidence

To address the problem of "raw block noise being too high and knowledge fragmentation causing injection degradation", we introduce the **evidence-first dual-bank strategy**:

- **Evidence bank**: DeepSeek **extractive** evidence sentences (extractive-only) → `blocks.evidence.jsonl` → `kvbank_evidence/`
- **Raw bank**: Preserve `blocks.jsonl` + `kvbank_blocks/` for traceback and context supplementation

Now `rebuild_topic_kvbank_from_config.py`, after **PDF→raw_chunks→blocks**, will by default continue to generate evidence (from scratch):
- `topics/<TOPIC>/work/blocks.evidence.jsonl`
- `topics/<TOPIC>/work/kvbank_evidence/manifest.json`

If you only want to **separately build evidence on top of existing artifacts** (without re-running PDF→raw_chunks→blocks), you may use (not recommended as the "cleanest" mainline):

```bash
export WORK_DIR="/home/jb/KVI/topics/SFTSV/work"

# 1) raw_chunks -> evidence blocks (DeepSeek extractive evidence sentences; cleaner, recommended)
python -u scripts/build_evidence_blocks_from_raw_chunks_jsonl_deepseek.py \
  --raw_chunks_jsonl "$WORK_DIR/raw_chunks.jsonl" \
  --out_jsonl "$WORK_DIR/blocks.evidence.jsonl" \
  --topic_goal "$(jq -r .goal config/topics/SFTSV/config.json)" \
  --max_sentences_per_paragraph 3

# Tip: If you find evidence coverage insufficient (extraction is too conservative), you can raise max_sentences_per_paragraph to 4 or 5,
# which increases evidence sentence density (may also bring slight noise; revalidate with keyword sampling from 1.3).

# 2) evidence blocks -> kvbank_evidence
python -u scripts/build_kvbank_from_blocks_jsonl.py \
  --blocks_jsonl "$WORK_DIR/blocks.evidence.jsonl" \
  --out_dir "$WORK_DIR/kvbank_evidence" \
  --base_llm "$BASE_LLM" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --layers 0,1,2,3 \
  --block_tokens 256 \
  --shard_size 1024
```

#### 1.2.2.1.a (Debug Essential) Mini evidence.txt One-Click Build: evidence.txt → blocks → pattern sidecar → pattern_contract → kvbank_blocks

> Purpose: You only have a few lines of `evidence.txt` and want to first run the minimum closed loop of "retrieval → Evidence Units → injection → answer" (for architecture debugging / patching gaps), without depending on PDFs.
>
> Convention directory (replaceable with any topic):
> - `TOPIC_DIR=/home/jb/KVI/config/topics/SFTSV`
> - Input: `$TOPIC_DIR/evidence.txt`
> - Artifacts: `$TOPIC_DIR/blocks.jsonl`, `$TOPIC_DIR/blocks.enriched.jsonl`, `$TOPIC_DIR/pattern_contract.json`, `$TOPIC_DIR/kvbank_blocks/`

```bash
cd /home/jb/KVI

export TOPIC_DIR="/home/jb/KVI/config/topics/SFTSV"
export BASE_LLM="Qwen/Qwen2.5-7B-Instruct"
export DOMAIN_ENCODER="sentence-transformers/all-MiniLM-L6-v2"

# 1) evidence.txt -> blocks.jsonl
python scripts/build_blocks_from_raw_text.py \
  --raw_text "$TOPIC_DIR/evidence.txt" \
  --out "$TOPIC_DIR/blocks.jsonl" \
  --tokenizer "$BASE_LLM" \
  --chunk_tokens 4096 --chunk_overlap 256 \
  --block_tokens 256 \
  --keep_last_incomplete_block

# 2) blocks.jsonl -> blocks.enriched.jsonl + pattern sidecar (place pattern_out_dir in topic_dir)
python scripts/build_pattern_index_from_blocks_v2.py \
  --blocks_jsonl_in "$TOPIC_DIR/blocks.jsonl" \
  --blocks_jsonl_out "$TOPIC_DIR/blocks.enriched.jsonl" \
  --pattern_out_dir "$TOPIC_DIR"

# 3) blocks.enriched.jsonl -> pattern_contract.json (for PatternContractLoader + matcher/scoring usage)
python scripts/pattern_contract_autogen.py \
  --blocks_jsonl_in "$TOPIC_DIR/blocks.enriched.jsonl" \
  --out "$TOPIC_DIR/pattern_contract.json" \
  --topic SFTSV \
  --min_abbr_count 1 \
  --min_slot_count 1 \
  --max_abbr 50 \
  --max_slots 50

# 4) blocks.enriched.jsonl -> kvbank_blocks (mini KVBank)
python scripts/build_kvbank_from_blocks_jsonl.py \
  --blocks_jsonl "$TOPIC_DIR/blocks.enriched.jsonl" \
  --out_dir "$TOPIC_DIR/kvbank_blocks" \
  --base_llm "$BASE_LLM" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --layers 0,1,2,3 \
  --block_tokens 256 \
  --shard_size 1024 \
  --device cuda \
  --dtype bfloat16
```

##### 1.2.2.1.a.1 Validation (simple pipeline): routing + Evidence Units + injected output

> Goal: Verify that "retrieval hits + Evidence Units count + injected output" all work correctly (output JSON will include step_debug).
> Note: The simple pipeline is an architecture debugging mode; the core chain is: prompt → similarity retrieval (kvbank_blocks) → Evidence Units (sentence-level) → multi-step injection → text answer.

```bash
cd /home/jb/KVI

export TOPIC_DIR="/home/jb/KVI/config/topics/SFTSV"
export BASE_LLM="Qwen/Qwen2.5-7B-Instruct"
export DOMAIN_ENCODER="sentence-transformers/all-MiniLM-L6-v2"

# A) Symptoms (should produce selected_unit_counts>0 and evidence_units_shown containing symptom enumeration sentences)
python scripts/run_kvi2_runtime_test.py \
  --pipeline simple \
  --model "$BASE_LLM" \
  --prompt "What are the main clinical symptoms of SFTSV?" \
  --kv_dir "$TOPIC_DIR/kvbank_blocks" \
  --blocks_jsonl "$TOPIC_DIR/blocks.enriched.jsonl" \
  --pattern_index_dir "$TOPIC_DIR" \
  --sidecar_dir "$TOPIC_DIR" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --use_chat_template \
  --top_k 8 \
  --simple_use_evidence_units \
  --simple_require_units \
  --simple_max_steps 1 \
  --simple_max_blocks_per_step 4 \
  --simple_max_unit_sentences 4 \
  --show_baseline

# B) Region + symptoms (multi-intent, evidence_units_shown should contain both region and symptom sentences)
python scripts/run_kvi2_runtime_test.py \
  --pipeline simple \
  --model "$BASE_LLM" \
  --prompt "Which regions in China had the highest incidence of SFTSV from 2009-2014? What are the main clinical symptoms?" \
  --kv_dir "$TOPIC_DIR/kvbank_blocks" \
  --blocks_jsonl "$TOPIC_DIR/blocks.enriched.jsonl" \
  --pattern_index_dir "$TOPIC_DIR" \
  --sidecar_dir "$TOPIC_DIR" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --use_chat_template \
  --top_k 8 \
  --simple_use_evidence_units \
  --simple_require_units \
  --simple_max_steps 1 \
  --simple_max_blocks_per_step 4 \
  --simple_max_unit_sentences 6 \
  --show_baseline
```

#### 1.2.2.2 (New, Recommended) Schema Version: blocks.schema + kvbank_schema (slot-aware injection)

To address the problem of "repeated injection of the same semantic dimension causing answer collapse", we introduce the **schema-first + slot-aware strategy** on top of evidence:

- **Schema bank**: High information-density slot constraints aggregated from evidence → `blocks.schema.jsonl` → `kvbank_schema/`
- **Evidence bank**: Preserved for grounding/citation (retrieval-only, **never injected**)
- **Raw bank**: Preserved for traceback/fallback context (retrieval-only, **never injected**)

**Core constraints (see `docs/slot_enum.md`)**:
- Schema KV is the **only cache allowed for injection**
- Evidence/raw may **only append to prompt** (grounding), never inject into KV
- At most 1 schema injected per step; stop when slot coverage is exhausted

##### 1) Generate schema blocks from evidence blocks

```bash
export WORK_DIR="/home/jb/KVI/topics/SFTSV/work"

python -u scripts/build_schema_blocks_from_evidence_jsonl.py \
  --blocks_jsonl_evidence "$WORK_DIR/blocks.evidence.jsonl" \
  --out_jsonl "$WORK_DIR/blocks.schema.jsonl"
```

Explanation:
- This script **heuristically infers** `answerable_slots` (for slot-aware selection) from evidence text, including but not limited to:
  - `transmission` / `pathogenesis` / `diagnosis` / `treatment`
  - `disease_full_name` (full name/abbreviation expansion, taxonomy.definition)
  - `geographic_distribution` (region distribution, epidemiology.geography)
- The `vector` field has been removed from schema text (to avoid mistranslation/overreach from species common names in Chinese).

⚠️ Important: If you upgrade the slot or schema compilation logic, you must **rebuild**:
- `blocks.schema.jsonl`
- `kvbank_schema`

##### 2) Build schema KVBank

```bash
python -u scripts/build_kvbank_from_blocks_jsonl.py \
  --blocks_jsonl "$WORK_DIR/blocks.schema.jsonl" \
  --out_dir "$WORK_DIR/kvbank_schema" \
  --base_llm "$BASE_LLM" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --layers 0,1,2,3 \
  --block_tokens 256 \
  --shard_size 1024
```

##### 3) Inspect schema block quality

```bash
python -u scripts/inspect_blocks_quality.py \
  --blocks_jsonl "$WORK_DIR/blocks.schema.jsonl" \
  --sample 10
```

Upon completion, how to judge success (schema-first adds 2 hard artifacts):
- `$WORK_DIR/blocks.schema.jsonl` (schema blocks, with `slots` field)
- `$WORK_DIR/kvbank_schema/manifest.json` (schema KVBank)

### 1.2.3 (Debug Mainline) Break the Pipeline into Three Stages for Stepwise Debugging

When you need to pinpoint "whether filtering is inaccurate, extraction is poor, or KB construction has errors", use this three-stage approach:

1) **Doc-level filtering (abstract)**: Source PDFs → `topics/<TOPIC>/pdfs/` + `doc_filter_results.jsonl`

```bash
python -u scripts/build_topic_pdf_subset_deepseek.py \
  --config config/topics/SFTSV/config.json \
  --max_pdfs 200
```

2) **Extract raw_chunks (no KB construction)**: KEEP PDFs → raw_chunks

```bash
python -u scripts/build_raw_context_from_pdfs.py \
  --pdf_dir "/home/jb/KVI/topics/SFTSV/pdfs" \
  --out "/home/jb/KVI/topics/SFTSV/work/raw_chunks.jsonl" \
  --tokenizer "$BASE_LLM" \
  --chunk_tokens 4096 \
  --chunk_overlap 256 \
  --ocr auto \
  --knowledge_filter \
  --deepseek_model deepseek-chat
```

3) **One-click blocks+KVBank**: KEEP PDFs → work_dir (raw_chunks/blocks/kvbank)

```bash
python -u scripts/build_kvbank_from_pdf_dir_multistep.py \
  --pdf_dir "/home/jb/KVI/topics/SFTSV/pdfs" \
  --work_dir "/home/jb/KVI/topics/SFTSV/work" \
  --base_llm "$BASE_LLM" \
  --retrieval_encoder_model "$DOMAIN_ENCODER" \
  --layers 0,1,2,3 \
  --chunk_tokens 4096 --chunk_overlap 256 \
  --block_tokens 256 --block_overlap_tokens 64 --keep_last_incomplete_block \
  --ocr auto \
  --knowledge_filter --deepseek_model deepseek-chat \
  --split_tables \
  --shard_size 1024
```

Why do both `rebuild_topic_kvbank_from_config.py` and `build_kvbank_from_pdf_dir_multistep.py` appear in the runbook?

- **`rebuild_topic_kvbank_from_config.py` (recommended mainline)**: Topic KB "one-command" pipeline. It follows `config/topics/<TOPIC>/config.json` to first perform doc-level DeepSeek (abstract) filtering, then runs PDF→raw_chunks→blocks→KVBank, and by default continues to generate `blocks.evidence.jsonl` + `kvbank_evidence/` (dual-bank strategy).
- **`build_kvbank_from_pdf_dir_multistep.py` (debug/decompose entrypoint)**: A generic script not dependent on topic config, used when you **already have a batch of PDFs** and want to directly do PDF→work_dir→KVBank, suitable for pinpointing issues (e.g., which step of extraction/OCR, chunking, KVBank construction has anomalies), also suitable for temporarily creating a non-topic experiment directory.

Conclusion: **Under normal circumstances, only run `rebuild_topic_kvbank_from_config.py`**; only run `build_kvbank_from_pdf_dir_multistep.py` when you need to "debug step-by-step / do a non-topic quick experiment".

### 1.3 Quality Check (evidence-first): How to Confirm Block Text "Extraction Quality is Good"

> Conclusion: Under the "dual-bank strategy", you **prioritize evidence verification** (determines retrieval relevance and injection noise); raw only matters when tables/context supplementation is needed.
> Simple principle: **Whichever bank you prioritize for final inference/injection, check that bank's corresponding blocks file first.**

First set the work directory for the current topic (using SFTSV as example):

```bash
export TOPIC_WORK_DIR="/home/jb/KVI/topics/SFTSV/work"
```

#### 1.3.1 (Priority) Inspect Evidence Blocks: `blocks.evidence.jsonl`

1) **Overall statistics + sampling** (empty block rate, duplicate rate, suspected garbage ratio, sampled original text)

```bash
python -u scripts/inspect_blocks_quality.py \
  --blocks_jsonl "$TOPIC_WORK_DIR/blocks.evidence.jsonl" \
  --sample 10
```

> Note: Evidence blocks target "shorter, more single-intent, more directly answerable", typically do not emphasize `--tables_only`.

2) **Keyword sampling validation: Does the KB contain certain types of evidence sentences (e.g., "pathogenesis/immune mechanism")?**

> Purpose: Quickly answer "does the evidence KB contain mechanism/pathogenesis/immune related evidence sentences", avoiding guesswork based on intuition alone.
> The script streams through `blocks.evidence.jsonl`, counts hits per keyword, and randomly samples and prints matching blocks (including `doc_id/source_uri/line_no` + snippet).

```bash
python -u scripts/sample_blocks_by_keywords.py \
  --blocks_jsonl "$TOPIC_WORK_DIR/blocks.evidence.jsonl" \
  --keywords "pathogenesis,mechanism,immune,cytokine,MODS,multi-organ,致病,发病机制,免疫,细胞因子,器官功能衰竭" \
  --sample 20 \
  --seed 0 \
  --max_chars 600
```

#### 1.3.2 (Optional) Review Raw Blocks: `blocks.jsonl` (tables/context/identify fragmentation issues)

1) **Overall statistics + sampling**

```bash
python -u scripts/inspect_blocks_quality.py \
  --blocks_jsonl "$TOPIC_WORK_DIR/blocks.jsonl" \
  --sample 10
```

2) **Sample only table-related blocks** (only needed when you enable `extract_tables/split_tables` and actually want tables in the raw bank/table routing)

```bash
python -u scripts/inspect_blocks_quality.py \
  --blocks_jsonl "$TOPIC_WORK_DIR/blocks.jsonl" \
  --tables_only \
  --sample 10
```

> If your repo uses a "monorepo layout" (i.e., has `external_kv_injection/` subdirectory under the repo root), change the script paths in the above commands to:
> `python -u external_kv_injection/scripts/inspect_blocks_quality.py ...`

Artifacts (under topic work_dir):
- `$TOPIC_WORK_DIR/raw_chunks.jsonl` (raw context after PDF extraction, does not enter attention)
- `$TOPIC_WORK_DIR/blocks.evidence.jsonl` (DeepSeek extractive evidence sentences, recommended for priority retrieval/injection)
- `$TOPIC_WORK_DIR/kvbank_evidence/` (evidence KVBank)
- (Optional traceback) `$TOPIC_WORK_DIR/blocks.jsonl` + `$TOPIC_WORK_DIR/kvbank_blocks/` (raw blocks/KVBank, for context/table supplementation)

### 1.4 Background KVBank Construction (evidence-first): blocks → kvbank, nohup + live log viewing

> The `blocks_to_kvbank` stage is computationally heavy and **very memory-hungry**. It is recommended to enable **Option A: Sharded KVBank** (`--shard_size`), letting it write to disk incrementally, avoiding memory blowup from a single `np.stack`.

Similarly, first set the current topic work directory (using SFTSV as example):

```bash
export TOPIC_WORK_DIR="/home/jb/KVI/topics/SFTSV/work"
```

#### 1.4.1 (Priority) Build Evidence KVBank: `blocks.evidence.jsonl` → `kvbank_evidence/`

1) Start background task (log written to file simultaneously, convenient for `tail -f` anytime)

```bash
mkdir -p "$TOPIC_WORK_DIR/logs"

nohup bash -lc "python -u scripts/build_kvbank_from_blocks_jsonl.py \
  --blocks_jsonl '$TOPIC_WORK_DIR/blocks.evidence.jsonl' \
  --out_dir '$TOPIC_WORK_DIR/kvbank_evidence' \
  --base_llm '$BASE_LLM' \
  --domain_encoder_model '$DOMAIN_ENCODER' \
  --layers 0,1,2,3 \
  --block_tokens 256 \
  --shard_size 1024 2>&1 | tee -a '$TOPIC_WORK_DIR/logs/evidence_blocks_to_kvbank.log'" \
  >/dev/null 2>&1 &
echo "started, log=$TOPIC_WORK_DIR/logs/evidence_blocks_to_kvbank.log"
```

2) Watch output live in the current terminal (does not affect background execution)

```bash
tail -f "$TOPIC_WORK_DIR/logs/evidence_blocks_to_kvbank.log"
```

3) Verify successful disk output (in sharded mode, you'll see `kvbank_evidence/manifest.json` + `kvbank_evidence/shards/00000/...`)

```bash
ls -alh "$TOPIC_WORK_DIR/kvbank_evidence"
ls -alh "$TOPIC_WORK_DIR/kvbank_evidence/shards" | head
```

#### 1.4.2 (Optional) Build Raw KVBank: `blocks.jsonl` → `kvbank_blocks/` (tables/context supplementation)

> Only run this section when you need raw fallback (or table routing); otherwise skip.

```bash
mkdir -p "$TOPIC_WORK_DIR/logs"

nohup bash -lc "python -u scripts/build_kvbank_from_blocks.py \
  --blocks '$TOPIC_WORK_DIR/blocks.jsonl' \
  --out_dir '$TOPIC_WORK_DIR/kvbank_blocks' \
  --base_llm '$BASE_LLM' \
  --retrieval_encoder_model '$DOMAIN_ENCODER' \
  --layers 0,1,2,3 \
  --block_tokens 256 \
  --shard_size 1024 2>&1 | tee -a '$TOPIC_WORK_DIR/logs/raw_blocks_to_kvbank.log'" \
  >/dev/null 2>&1 &
echo "started, log=$TOPIC_WORK_DIR/logs/raw_blocks_to_kvbank.log"
```

```bash
tail -f "$TOPIC_WORK_DIR/logs/raw_blocks_to_kvbank.log"
```

```bash
ls -alh "$TOPIC_WORK_DIR/kvbank_blocks"
ls -alh "$TOPIC_WORK_DIR/kvbank_blocks/shards" | head
```

## 2) Test: Multi-Step Injection

### 2.0 (Recommended, New) Schema-First Injection (slot-aware, strictest)

> Core rules (see `docs/slot_enum.md`):
> - **Only schema KV may be injected** (schema text forward → cache)
> - evidence/raw may **only append to prompt** (grounding), cannot inject into KV
> - At most 1 schema injected per step; stop when slot coverage is exhausted

All three banks must be provided simultaneously: `kvbank_schema` (injection) + `kvbank_evidence` (grounding) + `kvbank_blocks` (fallback)

```bash
export WORK_DIR="/home/jb/KVI/topics/SFTSV/work"

python -u scripts/run_multistep_inject_demo.py \
  --model "$BASE_LLM" \
  --kv_dir "$WORK_DIR/kvbank_blocks" \
  --kv_dir_evidence "$WORK_DIR/kvbank_evidence" \
  --kv_dir_schema "$WORK_DIR/kvbank_schema" \
  --blocks_jsonl_schema "$WORK_DIR/blocks.schema.jsonl" \
  --blocks_jsonl_evidence "$WORK_DIR/blocks.evidence.jsonl" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --prompt "What are the main transmission routes of SFTSV? Please answer in English and quote 1 evidence sentence verbatim." \
  --schema_required_slots "transmission,pathogenesis" \
  --use_struct_slots \
  --ground_with_selected_text \
  --no_repeat_ngram_size 12 \
  --enable_layer2 \
  --layer1_max_new_tokens 256 \
  --layer2_max_new_tokens 192 \
  --max_new_tokens 256 \
  --debug_print_candidates 10
```

Verification points:
- Logs should show `[retriever] routing=schema->evidence->raw ...`
- StepDebug.note should include `schema_selector=...` (slot coverage info)
- If selected schema introduces no new slots → stop (`stop_reason=no_new_slots`)
- Answer end will only append `【Evidence Sentence】/【Fallback Context (raw)】`, **schema text will not appear in prompt**

Three-layer knowledge output (mandatory format):
- The injected answer will be strictly divided into three layers (never skip/merge layers):
  - `### L0 | Evidence-Bound Conclusions` (Evidence-Bound: only based on evidence/documents, no extrapolation; insufficient evidence must state "No evidence supports this")
  - `### L1 | Domain Prior (LLM Internal Knowledge)` (Domain Prior: textbook-level consensus explanation; must not conflict with L0; prohibit "latest research/hypotheses")
  - `### L2 | Speculative or Interpretive Supplement` (Speculative: off by default; when enabled must explicitly state "speculative/not yet fully confirmed", must not override L0/L1)

Enabling L2 (optional):
- By default L2 is not generated, only a placeholder is retained; to enable the speculative layer, add to the demo:
  - `--enable_layer2`
  - Adjustable token budget: `--layer2_max_new_tokens 192`
  - L1 token budget: `--layer1_max_new_tokens 256`

### 2.1 (Legacy) Evidence-First Injection

> Conclusion: Use **evidence KVBank** by default for retrieval and injection (lower noise, stronger relevance), fall back to raw bank for context supplementation when necessary.

Below using SFTSV as example (flat layout: `/home/jb/KVI`):

```bash
python -u scripts/run_multistep_inject_demo.py \
  --model "$BASE_LLM" \
  --topic sftsv --topic_work_dir "/home/jb/KVI/topics" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --prompt "What are the main transmission routes of SFTSV? Please answer in English and quote 1 evidence sentence verbatim." \
  --blocks_jsonl "/home/jb/KVI/topics/SFTSV/work/blocks.jsonl" \
  --blocks_jsonl_evidence "/home/jb/KVI/topics/SFTSV/work/blocks.evidence.jsonl" \
  --allowed_langs "zh,en" \
  --layers 0,1,2,3 \
  --max_steps 1 \
  --max_blocks_per_step 1 \
  --top_k_blocks 16 \
  --ground_with_selected_text \
  --no_repeat_ngram_size 12 \
  --max_new_tokens 256
```

Explanation:
- `--topic ... --topic_work_dir ...`: The script will first probe `kvbank_evidence` (and fall back to raw `kvbank_blocks` when needed).
- `--blocks_jsonl(_evidence) + --allowed_langs`: Strongly recommended to enable, avoiding injection degradation from hitting non-target-language blocks in mixed-language corpora.
- `--max_steps=1 + --max_blocks_per_step=1`: First validate relevance and stability with "minimum injection"; increase `--max_steps` to 2/4 after stabilization.

### 2.1.1 (Equivalent Usage) Explicitly Specify Evidence + Raw (without topic mode)

If your topic KB work_dir contains:

- `kvbank_evidence/manifest.json`
- `blocks.evidence.jsonl`

If you prefer not to use `--topic` for auto-detection, you can explicitly specify paths for both banks (equivalent effect):

```bash
python -u scripts/run_multistep_inject_demo.py \
  --model "$BASE_LLM" \
  --kv_dir "/home/jb/KVI/topics/SFTSV/work/kvbank_blocks" \
  --kv_dir_evidence "/home/jb/KVI/topics/SFTSV/work/kvbank_evidence" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --prompt "What are the main transmission routes of SFTSV? Please answer in English and quote 1 evidence sentence verbatim." \
  --max_steps 1 \
  --max_blocks_per_step 1 \
  --top_k_blocks 16 \
  --blocks_jsonl "/home/jb/KVI/topics/SFTSV/work/blocks.jsonl" \
  --blocks_jsonl_evidence "/home/jb/KVI/topics/SFTSV/work/blocks.evidence.jsonl" \
  --allowed_langs "zh,en" \
  --ground_with_selected_text \
  --no_repeat_ngram_size 12 \
  --max_new_tokens 256
```

Verification points:
- `=== Step Debug ===`'s `selected_block_ids` should favor evidence blocks (shorter, more single-intent).
- Answers should show less degradation from raw noise like "unknown routes / mink bite".

### 2.2 (New) Evaluation Set A/B: Mandatory JSON Protocol Output + Faithfulness/Overclaim Metrics

Prepare evaluation set `prompts.jsonl` (each line must contain at least `prompt`):

```json
{"id":"sftsv_tx_001","prompt":"What are the main transmission routes of SFTSV?"}
{"id":"sftsv_tx_002","prompt":"Does human-to-human transmission of SFTSV exist? If so, through what type of contact?"}
```

Run A/B (baseline vs injection) with automatic coverage/overclaim metrics:

```bash
python -u scripts/run_ab_eval_protocol.py \
  --model "$BASE_LLM" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --kv_dir "/home/jb/KVI/topics/SFTSV/work/kvbank_blocks" \
  --kv_dir_evidence "/home/jb/KVI/topics/SFTSV/work/kvbank_evidence" \
  --blocks_jsonl "/home/jb/KVI/topics/SFTSV/work/blocks.jsonl" \
  --blocks_jsonl_evidence "/home/jb/KVI/topics/SFTSV/work/blocks.evidence.jsonl" \
  --prompts_jsonl "/home/jb/KVI/topics/SFTSV/eval/prompts.jsonl" \
  --out_jsonl "/home/jb/KVI/topics/SFTSV/eval/ab_results.jsonl" \
  --max_examples 0
```

Results and verification:
- `ab_results.jsonl` each entry contains baseline/injected raw output, parsed JSON, and `covered/overclaim` metrics.
- Terminal end prints summary: `baseline_valid/inj_valid`, `baseline_covered/inj_covered`, `baseline_over/inj_over`.

### 2.3 (New) Unit Test: Evidence Recall (Retrieval Hit Rate Smoke Test)

Purpose:
- Verify `blocks.evidence.jsonl` artifact is non-empty and structurally correct
- Verify `kvbank_evidence` has basic "hit capability" for a set of common queries (not requiring sentence-level evidence matching, just a recall sanity check)
- Catch issues like "retrieval vector degradation / always hitting the same batch of evidence" early

Run (using SFTSV as example):

```bash
export WORK_DIR="/home/jb/KVI/topics/SFTSV/work"

python -u scripts/test_evidence_recall.py \
  --kv_dir_evidence "$WORK_DIR/kvbank_evidence" \
  --blocks_jsonl_evidence "$WORK_DIR/blocks.evidence.jsonl" \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --top_k 16
```

Explanation:
- By default retrieves on five query categories ("transmission/pathogen/pathogenesis/epidemiology/diagnosis prevention") and asserts overall hit_rate meets threshold
- If you find evidence is mostly in English, Chinese queries can be retained: the test uses English anchors for hit determination (whether retrieval pulls back relevant English evidence sentences)

## 3) (Optional) Train Projector (Align to past_key_values space, max_kv_tokens=256)

> This training pipeline uses `ChunkStore` (see `external_kv_injection/scripts/build_chunkstore_from_pdfs.py`).
> If you want training data to also use "DeepSeek-filtered raw context", it is recommended to first convert the raw_chunks artifact from step 1 into a chunkstore (a script can be added later).

```bash
export WORK_TRAIN="external_kv_injection/_exp_train"
mkdir -p "$WORK_TRAIN"
```

### 3.1 PDF → chunkstore.jsonl (Quick Training Set)

```bash
python external_kv_injection/scripts/build_chunkstore_from_pdfs.py \
  --pdf_dir "$PDF_DIR" \
  --out "$WORK_TRAIN/chunkstore.jsonl" \
  --dataset_version v0
```

### 3.2 Generate Teacher KV Dataset

```bash
python external_kv_injection/scripts/build_teacher_kv_dataset.py \
  --chunkstore "$WORK_TRAIN/chunkstore.jsonl" \
  --out "$WORK_TRAIN/teacher_kv_dataset.pt" \
  --model "$BASE_LLM" \
  --layers 0,1,2,3 \
  --max_kv_tokens 256 \
  --max_samples 200
```

### 3.3 Train Projector

```bash
python external_kv_injection/scripts/train_projector_kv.py \
  --dataset "$WORK_TRAIN/teacher_kv_dataset.pt" \
  --model "$BASE_LLM" \
  --out_dir "$WORK_TRAIN/projector_ckpt" \
  --batch_size 1 \
  --lr 1e-4 \
  --epochs 1
```

### 3.4 Build KVBank with Projector (retrieval key uses DomainEncoder)

```bash
python external_kv_injection/scripts/build_kvbank_with_projector.py \
  --chunkstore "$WORK_TRAIN/chunkstore.jsonl" \
  --out_dir "$WORK_TRAIN/kvbank_projector" \
  --base_model "$BASE_LLM" \
  --projector_ckpt "$WORK_TRAIN/projector_ckpt/projector_kv.pt" \
  --max_kv_tokens 256 \
  --max_chunks 200 \
  --retrieval_encoder_model "$DOMAIN_ENCODER"
```

## 4) (Optional) Train/Use Gate (DomainEncoder(query) embedding)

```bash
python external_kv_injection/scripts/train_gate_query.py \
  --kv_dir "$WORK_TRAIN/kvbank_projector" \
  --out "$WORK_TRAIN/gate_query.pt"
```

Inference validation (single-step injection demo):

```bash
python external_kv_injection/scripts/run_qwen_inject_demo.py \
  --model "$BASE_LLM" \
  --kv_dir "$WORK_TRAIN/kvbank_projector" \
  --prompt "Based on the knowledge base content, please answer: What are the main transmission routes of SFTSV? Provide supporting evidence." \
  --layers 0,1,2,3 \
  --top_k 4 \
  --domain_encoder_model "$DOMAIN_ENCODER" \
  --gate_ckpt "$WORK_TRAIN/gate_query.pt" \
  --gate_mode scale_v \
  --max_new_tokens 128
```
