"""
Scheme C — Triple Extractor.

Extracts (subject, predicate, object) triples from evidence sentences
using a local LLM (same base model used in the pipeline).

Design choices:
* **Structured JSON output** — the extraction prompt forces JSON-array output
  with strict schema.  This avoids free-text parsing and improves reliability.
* **Batch processing** — sentences are grouped into small batches (≤ 5)
  to balance context utilisation and extraction quality.
* **Relation type vocabulary** — the prompt includes the allowed relation
  types so the LLM maps to canonical predicates, not free-form strings.
* **No external API** — runs on the same GPU-local LLM used elsewhere,
  keeping the system self-contained.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schema import (
    Triple,
    DEFAULT_RELATION_TYPES,
    DEFAULT_ENTITY_TYPES,
    load_relation_types,
    load_entity_types,
)


# ---------------------------------------------------------------------------
# Extraction prompt template
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = """\
你是一个医学知识三元组抽取专家。你的任务是从给定的医学证据句子中抽取结构化的知识三元组。

### 输出格式
返回一个 JSON 数组，每个元素是一个三元组对象：
```json
[
  {{
    "subject": "主语实体（如：SFTSV）",
    "subject_type": "实体类型",
    "predicate": "关系类型",
    "object": "宾语实体（如：发热）",
    "object_type": "实体类型",
    "confidence": 0.95,
    "sentence_index": 1
  }}
]
```

### 允许的关系类型（predicate）
{relation_list}

### 允许的实体类型（subject_type / object_type）
{entity_list}

### 规则
1. 每条句子至少抽取 1 个三元组，最多 5 个
2. subject 和 object 必须是句子中明确提到或可直接推断的实体
3. predicate 必须从上面的允许列表中选择
4. 如果一个句子包含多个事实，拆分为多个三元组
5. confidence 表示你对这个三元组的确信度（0.0-1.0）
6. **每个三元组必须包含 "sentence_index" 字段**，值为该三元组来源句子的编号（如 1, 2, 3）
7. 只输出 JSON 数组，不要输出任何其他文字
"""

_EXTRACTION_USER = """\
请从以下证据句子中抽取知识三元组：

{sentences_block}

只输出 JSON 数组："""


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    """Result of triple extraction for a batch of sentences."""
    triples: List[Triple]
    raw_output: str = ""
    parse_errors: List[str] = field(default_factory=list)


class TripleExtractor:
    """
    Extract knowledge triples from evidence sentences.

    Supports two backends:
    * **Local LLM** — pass ``model``, ``tokenizer``, ``device``.
    * **DeepSeek API** — pass ``deepseek_client`` (a :class:`DeepSeekClient`
      instance from ``src.llm_filter.deepseek_client``).  When provided,
      ``model``/``tokenizer``/``device`` are ignored and no GPU is needed.

    Usage (local)::

        extractor = TripleExtractor(model=model, tokenizer=tokenizer, device=device)

    Usage (DeepSeek)::

        from src.llm_filter.deepseek_client import DeepSeekClient, DeepSeekClientConfig
        client = DeepSeekClient(DeepSeekClientConfig())
        extractor = TripleExtractor(deepseek_client=client)
    """

    def __init__(
        self,
        *,
        model: Any = None,
        tokenizer: Any = None,
        device: Any = None,
        deepseek_client: Any = None,
        relation_types: Optional[Dict[str, Dict[str, Any]]] = None,
        entity_types: Optional[Dict[str, Dict[str, Any]]] = None,
        max_new_tokens: int = 1024,
        batch_size: int = 3,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.deepseek_client = deepseek_client
        self.relation_types = relation_types or dict(DEFAULT_RELATION_TYPES)
        self.entity_types = entity_types or dict(DEFAULT_ENTITY_TYPES)
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = max(1, int(batch_size))

    # -- Public API --

    def extract_from_sentences(
        self,
        sentences: Sequence[Dict[str, Any]],
    ) -> List[Triple]:
        """
        Extract triples from a list of sentence dicts.

        Each sentence dict must have at least ``"text"`` and ``"id"`` keys.
        Returns a flat list of :class:`Triple` objects.
        """
        all_triples: List[Triple] = []
        batches = _chunk_list(list(sentences), self.batch_size)
        for batch_idx, batch in enumerate(batches):
            result = self._extract_batch(batch, batch_idx=batch_idx)
            all_triples.extend(result.triples)
        return all_triples

    # -- Internal --

    def _extract_batch(
        self,
        sentences: List[Dict[str, Any]],
        batch_idx: int = 0,
    ) -> ExtractionResult:
        """Extract triples from a single batch of sentences."""
        # Build prompt
        relation_list = "\n".join(
            f"- `{k}`: {v.get('description', '')}"
            for k, v in self.relation_types.items()
        )
        entity_list = "\n".join(
            f"- `{k}`: {v.get('description', '')}"
            for k, v in self.entity_types.items()
        )

        sentences_block = ""
        sent_lookup: Dict[int, Dict[str, Any]] = {}
        for i, s in enumerate(sentences):
            sid = str(s.get("id") or s.get("sentence_id") or s.get("block_id") or f"sent_{i}")
            text = str(s.get("text") or "").strip()
            if not text:
                continue
            sentences_block += f"[{i+1}] (id={sid}) {text}\n"
            sent_lookup[i + 1] = s

        if not sentences_block.strip():
            return ExtractionResult(triples=[], raw_output="", parse_errors=["empty batch"])

        system_msg = _EXTRACTION_SYSTEM.format(
            relation_list=relation_list,
            entity_list=entity_list,
        )
        user_msg = _EXTRACTION_USER.format(sentences_block=sentences_block.strip())

        # Generate
        raw = self._generate(system_msg, user_msg)

        # Parse JSON output
        triples, errors = self._parse_output(raw, sentences, sent_lookup)
        return ExtractionResult(triples=triples, raw_output=raw, parse_errors=errors)

    def _generate(self, system_msg: str, user_msg: str) -> str:
        """Generate text using local LLM or DeepSeek API."""
        # --- DeepSeek API backend ---
        if self.deepseek_client is not None:
            return self.deepseek_client.chat(system=system_msg, user=user_msg)

        # --- Local LLM backend ---
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        # Try chat template first
        apply_fn = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_fn):
            try:
                prompt = apply_fn(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = system_msg + "\n\n" + user_msg
        else:
            prompt = system_msg + "\n\n" + user_msg

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        import torch
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.05,
            )
        # Decode only new tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def _parse_output(
        self,
        raw: str,
        sentences: List[Dict[str, Any]],
        sent_lookup: Dict[int, Dict[str, Any]],
    ) -> Tuple[List[Triple], List[str]]:
        """Parse LLM JSON output into Triple objects."""
        errors: List[str] = []
        # Extract JSON array from raw output (handle markdown fences)
        json_str = raw.strip()
        # Remove markdown code fences if present
        if "```" in json_str:
            match = re.search(r"```(?:json)?\s*\n?(.*?)```", json_str, re.DOTALL)
            if match:
                json_str = match.group(1).strip()

        # Try to parse as JSON array
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to find array within the text
            arr_match = re.search(r"\[.*\]", json_str, re.DOTALL)
            if arr_match:
                try:
                    parsed = json.loads(arr_match.group(0))
                except json.JSONDecodeError:
                    errors.append(f"JSON parse failed: {e}")
                    return [], errors
            else:
                errors.append(f"No JSON array found in output: {e}")
                return [], errors

        if not isinstance(parsed, list):
            errors.append(f"Expected list, got {type(parsed).__name__}")
            return [], errors

        # Determine provenance: if batch has 1 sentence, all triples share it
        default_prov: Dict[str, Any] = {}
        if len(sentences) == 1:
            s = sentences[0]
            default_prov = {
                "sentence_id": str(s.get("id") or s.get("sentence_id") or s.get("block_id") or ""),
                "sentence_text": str(s.get("text") or ""),
                "source_block_id": str(s.get("source_block_id") or s.get("block_id") or ""),
            }

        triples: List[Triple] = []
        allowed_relations = set(self.relation_types.keys())
        allowed_entities = set(self.entity_types.keys())

        for item in parsed:
            if not isinstance(item, dict):
                errors.append(f"Skipped non-dict item: {item}")
                continue
            subj = str(item.get("subject") or "").strip()
            pred = str(item.get("predicate") or "").strip()
            obj = str(item.get("object") or "").strip()
            if not subj or not pred or not obj:
                errors.append(f"Incomplete triple: {item}")
                continue
            # Validate relation type
            if pred not in allowed_relations:
                # Try to fuzzy-match
                pred_match = _fuzzy_match_key(pred, allowed_relations)
                if pred_match:
                    pred = pred_match
                else:
                    errors.append(f"Unknown predicate '{pred}', kept as-is")
            # Validate entity types
            subj_type = str(item.get("subject_type") or "").strip()
            obj_type = str(item.get("object_type") or "").strip()
            if subj_type and subj_type not in allowed_entities:
                subj_type_match = _fuzzy_match_key(subj_type, allowed_entities)
                if subj_type_match:
                    subj_type = subj_type_match
            if obj_type and obj_type not in allowed_entities:
                obj_type_match = _fuzzy_match_key(obj_type, allowed_entities)
                if obj_type_match:
                    obj_type = obj_type_match

            conf = float(item.get("confidence") or 0.9)
            # Build provenance
            prov = dict(default_prov) if default_prov else {}
            # If LLM provided a sentence index, use it
            sent_idx = item.get("sentence_index") or item.get("sent_idx")
            if sent_idx is not None:
                try:
                    idx_int = int(sent_idx)
                    if idx_int in sent_lookup:
                        s = sent_lookup[idx_int]
                        prov = {
                            "sentence_id": str(s.get("id") or s.get("sentence_id") or s.get("block_id") or ""),
                            "sentence_text": str(s.get("text") or ""),
                            "source_block_id": str(s.get("source_block_id") or s.get("block_id") or ""),
                        }
                except (ValueError, TypeError):
                    pass
            # Fallback: if provenance is still empty, try to match by text overlap
            if not prov.get("sentence_text") and len(sentences) > 1:
                best_match = _match_triple_to_sentence(subj, obj, sentences)
                if best_match:
                    prov = {
                        "sentence_id": str(best_match.get("id") or best_match.get("sentence_id") or best_match.get("block_id") or ""),
                        "sentence_text": str(best_match.get("text") or ""),
                        "source_block_id": str(best_match.get("source_block_id") or best_match.get("block_id") or ""),
                    }

            tid = _make_triple_id(subj, pred, obj, prov.get("sentence_id", ""))
            triples.append(Triple(
                triple_id=tid,
                subject=subj,
                subject_type=subj_type,
                predicate=pred,
                object=obj,
                object_type=obj_type,
                confidence=conf,
                provenance=prov,
            ))

        return triples, errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_list(lst: list, size: int) -> List[list]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def _make_triple_id(subj: str, pred: str, obj: str, sent_id: str = "") -> str:
    """Deterministic triple ID based on content hash."""
    raw = f"{subj}|{pred}|{obj}|{sent_id}"
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"t_{h}"


def _match_triple_to_sentence(
    subject: str, obj: str, sentences: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Match a triple's subject+object back to the most likely source sentence."""
    best: Optional[Dict[str, Any]] = None
    best_score = 0
    for s in sentences:
        text = str(s.get("text") or "").lower()
        if not text:
            continue
        score = 0
        if subject.lower() in text:
            score += len(subject)
        if obj.lower() in text:
            score += len(obj)
        if score > best_score:
            best_score = score
            best = s
    return best if best_score > 0 else None


def _fuzzy_match_key(value: str, allowed: set) -> Optional[str]:
    """Try to match a value to an allowed key (case-insensitive, underscore-tolerant)."""
    v = value.lower().replace("-", "_").replace(" ", "_")
    for k in allowed:
        if v == k.lower().replace("-", "_").replace(" ", "_"):
            return k
    # Substring match
    for k in allowed:
        if v in k.lower() or k.lower() in v:
            return k
    return None
