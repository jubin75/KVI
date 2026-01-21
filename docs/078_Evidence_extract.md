Evidence Unit Pipeline · Minimal Frozen Codegen Prompt

(Sentence-level Enumerative First)

GOAL

Implement an Evidence Unit Pipeline that extracts injectable Evidence Units from
blocks.enriched.jsonl for downstream SlotExtractor.

Hard rules:

Never inject low-quality evidence

Slot filling ONLY from Evidence Units

Sentence-level enumerative evidence is PRIMARY

List-like evidence is SECONDARY (fallback only)

INPUT

File: blocks.enriched.jsonl

Each block:

{
  "block_id": "string",
  "text": "string",
  "section_type": "paragraph | figure | table | supplementary | other",
  "sentences": [
    {
      "sentence_id": "string",
      "text": "string",
      "offset": int
    }
  ]
}

OUTPUT

File: evidence_units.jsonl

Each line:

{
  "unit_id": "string",
  "source": { "block_id": "string", "sentence_id": "string | null" },
  "unit_type": "sentence_enumerative | list_item | fragment",
  "semantic_role": "enumerative_fact | descriptive_fact | non_fact",
  "text": "string",
  "confidence": float,
  "injectability": {
    "allowed": boolean,
    "blocking_reasons": []
  }
}

PIPELINE SCOPE (FROZEN)

Included:

Sentence-level enumerative extraction (mandatory)

List-like / fragment fallback

Injectability judgment

Excluded:

Retrieval / ranking

Schema inference

Gate / decision

LLM generation

CORE PRINCIPLE

No Evidence Unit → No slot filling

SlotExtractor MUST consume ONLY:

injectability.allowed == true

STEP 1 — SENTENCE-LEVEL ENUMERATIVE (PRIMARY)

For each sentence in a block:

Promote to sentence_enumerative IFF ALL true:

Sentence expresses multiple parallel items

Enumeration signals exist: , ; and or

Enumerated items belong to same semantic dimension

Sentence is self-contained (not dependent on list context)

Examples (valid):
Patients present with fever, thrombocytopenia, leukopenia, and fatigue.

Output:
{
  "unit_type": "sentence_enumerative",
  "semantic_role": "enumerative_fact"
}

STEP 2 — LIST-LIKE / FRAGMENT (SECONDARY)

Used ONLY if no valid sentence-level enumerative unit exists in block.

Hard exclusions (discard immediately):

section_type in {figure, table, supplementary}

discourse lists: "in addition", "furthermore", "see also"

statistical / copyright / methods lists

Allowed output types:
unit_type = list_item | fragment
semantic_role = enumerative_fact (only if clearly factual)

STEP 3 — SEMANTIC ROLE ASSIGNMENT

Rules:

Multiple parallel factual items → enumerative_fact

Single factual statement → descriptive_fact

Anything else → non_fact

Only enumerative_fact can be injectable.

STEP 4 — INJECTABILITY (FROZEN)
injectable = (
  semantic_role == "enumerative_fact"
  and unit_type in {"sentence_enumerative", "list_item"}
  and no blocking_reasons
)

Common blocking_reasons:
non_enumerative
cross_semantic_mixed
discourse_list
insufficient_specificity

STEP 5 — OUTPUT CONTRACT

Output ALL evidence units (injectable or not)

SlotExtractor MUST ignore non-injectable units

Never fill slot directly from block or sentence

IMPLEMENTATION CONSTRAINTS

Language: Python

No heavy NLP libs (regex / heuristics allowed)

Deterministic, explainable logic

Each unit traceable to block_id (+ sentence_id if any)

SYSTEM GUARANTEES

This pipeline must ensure:

Missing Evidence Unit ⇒ slot must be missing

Rejection is explainable via blocking_reasons

Any injected fact is traceable to a sentence or list item

Sentence-level evidence dominates list-like evidence

FINAL FREEZE

This is the only implementation spec.
Sentence-level Enumerative Evidence is embedded, not optional.

Do NOT extend scope without explicit revision.