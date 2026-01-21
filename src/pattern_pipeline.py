"""
KVI 2.0 Pattern-first → Semantic-second → Introspection-gated pipeline interfaces.

This module defines minimal, explicit interfaces to separate concerns:
- PatternContractLoader: load topic-level pattern contracts (stable, prompt-agnostic).
- PatternMatcher: match contracts against a query using fast, non-semantic pattern retrievers.
- SlotSchema: represent which semantic fields are fillable vs empty.
- SemanticInstanceBuilder: build evidence-grounded semantic instances.
- IntrospectionGate: explainable gate that reasons about missing patterns/slots.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .pattern_contract import PatternContract
from .pattern_retriever import PatternRetriever, PatternRetrieveResult


@dataclass(frozen=True)
class SlotSpec:
    """
    Slot is an evidence constraint interface (not a text placeholder).
    """

    name: str
    required: bool
    evidence_type: List[str]
    min_evidence: int
    inference_level: str  # "hard" | "soft" | "schema"
    slot_type: str = "string"
    semantic_type: str = "generic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "required": bool(self.required),
            "evidence_type": list(self.evidence_type or []),
            "min_evidence": int(self.min_evidence),
            "inference_level": str(self.inference_level),
            "type": self.slot_type,
            "semantic_type": self.semantic_type,
        }


@dataclass(frozen=True)
class PatternSpec:
    """
    Topic-level pattern contract (prompt-agnostic).
    """

    pattern_id: str
    question_intent: str
    question_surface_forms: List[str]
    slots: Dict[str, SlotSpec]
    answer_style: str = "factual"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "question_skeleton": {
                "intent": self.question_intent,
                "surface_forms": list(self.question_surface_forms or []),
            },
            "slots": {k: v.to_dict() for k, v in (self.slots or {}).items()},
            "answer_style": self.answer_style,
        }


class PatternContractLoader:
    """
    Load topic-level Pattern Contracts from `topics/{topic}/pattern_contract.json`.

    Contracts are stable and prompt-agnostic. The prompt only affects matching/ranking,
    not contract generation (see docs/081_实验调试记录.md).
    """

    def load(
        self,
        *,
        topic_dir: Optional[str] = None,
        topic: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> List[PatternSpec]:
        path = self._resolve_path(topic_dir=topic_dir, topic=topic, base_dir=base_dir)
        if not path or not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return self._parse_contracts(payload)

    @staticmethod
    def infer_topic_dir_from_kv_dir(kv_dir: str) -> Optional[Path]:
        """
        Infer topic directory from a KVBank directory:
        {topic}/work/kvbank_blocks -> {topic}
        """
        p = Path(str(kv_dir))
        if p.name in {"kvbank_blocks", "kvbank_blocks_v2"}:
            # {topic}/work/kvbank_blocks
            if p.parent.name == "work":
                return p.parent.parent
            return p.parent
        return None

    def _resolve_path(
        self,
        *,
        topic_dir: Optional[str],
        topic: Optional[str],
        base_dir: Optional[str],
    ) -> Optional[Path]:
        if topic_dir:
            p = Path(str(topic_dir)) / "pattern_contract.json"
            return p
        if base_dir and topic:
            return Path(str(base_dir)) / str(topic) / "pattern_contract.json"
        return None

    def _parse_contracts(self, payload: Any) -> List[PatternSpec]:
        if not isinstance(payload, dict):
            return []
        patterns = payload.get("patterns")
        if isinstance(patterns, list):
            return self._parse_pattern_specs(patterns)
        if isinstance(patterns, dict):
            # Backward compatibility: patterns.hard / patterns.soft.
            out: List[PatternSpec] = []
            out.extend(self._parse_legacy_patterns(patterns.get("hard"), level="hard"))
            out.extend(self._parse_legacy_patterns(patterns.get("soft"), level="soft"))
            return out
        return []

    def _parse_pattern_specs(self, items: Any) -> List[PatternSpec]:
        if not isinstance(items, list):
            return []
        out: List[PatternSpec] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            pid = str(it.get("pattern_id") or "").strip()
            if not pid:
                continue
            intent, surface_forms = _parse_question_skeleton(it.get("question_skeleton"))
            slots_payload = it.get("slots") if isinstance(it.get("slots"), dict) else {}
            slots: Dict[str, SlotSpec] = {}
            for k, v in slots_payload.items():
                if not isinstance(v, dict):
                    continue
                name = str(k or "").strip()
                if not name:
                    continue
                sem_type = str(v.get("semantic_type") or "").strip() or _infer_semantic_type_from_slot(name)
                slots[name] = SlotSpec(
                    name=name,
                    required=bool(v.get("required", False)),
                    evidence_type=list(v.get("evidence_type") or []),
                    min_evidence=int(v.get("min_evidence", 1)),
                    inference_level=str(v.get("inference_level") or "schema"),
                    slot_type=str(v.get("type") or "string"),
                    semantic_type=sem_type,
                )
            out.append(
                PatternSpec(
                    pattern_id=pid,
                    question_intent=intent,
                    question_surface_forms=surface_forms,
                    slots=slots,
                    answer_style=str(it.get("answer_style") or "factual"),
                )
            )
        return out

    def _parse_legacy_patterns(self, items: Any, *, level: str) -> List[PatternSpec]:
        if not isinstance(items, list):
            return []
        out: List[PatternSpec] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            pid = str(it.get("id") or "").strip()
            if not pid:
                continue
            kind = _infer_kind(pid, str(it.get("type") or ""))
            slots: Dict[str, SlotSpec] = {}
            if kind == "abbr":
                slots["abbr"] = SlotSpec(
                    name="abbr",
                    required=True,
                    evidence_type=["abbreviation"],
                    min_evidence=1,
                    inference_level="hard" if level == "hard" else "soft",
                    slot_type="string",
                )
                slots["full_name"] = SlotSpec(
                    name="full_name",
                    required=False,
                    evidence_type=["definition", "abbreviation"],
                    min_evidence=1,
                    inference_level="soft",
                    slot_type="string",
                )
            if kind == "schema":
                slot_name = pid.split(":", 1)[1] if ":" in pid else "schema"
                sem_type = _infer_semantic_type_from_slot(slot_name)
                slots[slot_name] = SlotSpec(
                    name=slot_name,
                    required=False,
                    evidence_type=["schema"],
                    min_evidence=1,
                    inference_level="schema",
                    slot_type="string",
                    semantic_type=sem_type,
                )
            out.append(
                PatternSpec(
                    pattern_id=pid,
                    question_intent="legacy",
                    question_surface_forms=_default_question_skeleton(kind, slot_name=pid),
                    slots=slots,
                    answer_style="factual",
                )
            )
        return out


class PatternMatcher:
    """
    Match topic-level Pattern Contracts against a query using a PatternRetriever.

    This does not create new patterns from the prompt; it only selects which
    existing contracts are relevant for the current query.
    """

    def __init__(self, contracts: Sequence[PatternSpec], retriever: Optional[PatternRetriever] = None) -> None:
        self.contracts = list(contracts or [])
        self.retriever = retriever or PatternRetriever()

    def match(self, query: str) -> Tuple[PatternRetrieveResult, List[PatternSpec], Dict[str, str]]:
        q = str(query or "")
        result = self.retriever.retrieve(q)
        hit_ids = {str(h.block_id) for h in (result.pattern_hits or [])}
        q_low = q.lower()

        scored: List[Tuple[float, PatternSpec, str]] = []
        has_schema_slot_in_query = False
        for c in self.contracts:
            best_score = 0.0
            best_skel = ""
            for sk in c.question_surface_forms or []:
                score = _score_skeleton(q, sk)
                if score > best_score:
                    best_score = score
                    best_skel = sk
            if c.pattern_id in hit_ids:
                best_score = max(best_score, 1.0)
            # Prefer schema patterns when the query mentions a schema slot name.
            slot_names = [str(k).strip().lower() for k in (c.slots or {}).keys() if str(k).strip()]
            is_schema = c.pattern_id.startswith("schema:") or any(
                str(s.inference_level).lower() == "schema" for s in (c.slots or {}).values()
            )
            if slot_names and any(sn in q_low for sn in slot_names):
                if is_schema:
                    best_score += 0.2
                    has_schema_slot_in_query = True
            if best_score > 0.0:
                scored.append((best_score, c, best_skel))

        if has_schema_slot_in_query:
            adjusted: List[Tuple[float, PatternSpec, str]] = []
            for score, c, sk in scored:
                is_abbr = c.pattern_id.startswith("abbr:")
                if is_abbr:
                    score -= 0.2
                adjusted.append((score, c, sk))
            scored = adjusted

        scored.sort(key=lambda x: x[0], reverse=True)
        matched = [c for _, c, _ in scored]
        matched_skeletons = {c.pattern_id: sk for _, c, sk in scored if sk}
        return result, matched, matched_skeletons


@dataclass
class SlotSchema:
    """
    SlotSchema defines which semantic fields can be filled based on matched patterns.

    Slots are an interface for generation planning and introspection, not answers.
    """

    slots: Dict[str, SlotSpec]

    @classmethod
    def from_pattern(cls, pattern: Optional[PatternSpec]) -> "SlotSchema":
        slots = dict(pattern.slots) if pattern and pattern.slots else {}
        return cls(slots=slots)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v.to_dict() for k, v in (self.slots or {}).items()}

    def status(self, evidence_blocks: Sequence[Any]) -> Dict[str, str]:
        status: Dict[str, str] = {}
        for name, spec in (self.slots or {}).items():
            ev = _collect_evidence_for_slot(spec, evidence_blocks)
            if spec.required and len(ev) < int(spec.min_evidence):
                status[name] = "missing"
            elif len(ev) > 0:
                status[name] = "satisfied"
            else:
                status[name] = "missing"
        return status


class SemanticInstanceBuilder:
    """
    Build evidence-grounded semantic instances.

    Each instance must cite >=1 block_id. This builder does NOT invent content;
    it only packages evidence references and slot state.
    """

    def build(
        self,
        *,
        pattern: PatternSpec,
        evidence_blocks: Sequence[Any],
        slot_schema: SlotSchema,
        block_text_lookup: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        instances: List[Dict[str, Any]] = []
        slots_payload: Dict[str, List[Dict[str, Any]]] = {}
        filtered_out_non_symptom_lists: List[Dict[str, str]] = []
        for name, spec in (slot_schema.slots or {}).items():
            slots_payload[name] = _build_slot_evidence(
                spec,
                evidence_blocks,
                pattern_id=str(pattern.pattern_id),
                block_text_lookup=block_text_lookup,
                filtered_out=filtered_out_non_symptom_lists,
            )
        value_cleaning: Dict[str, Any] = {}
        value_cleaning_by_block: Dict[str, Dict[str, Any]] = {}
        try:
            from .semantic.schema_value_cleaner import SchemaValueCleaner  # type: ignore

            rule_dir = Path(__file__).resolve().parents[1] / "config" / "value_cleaning_rules"
            cleaner = SchemaValueCleaner(str(rule_dir))
            for name, spec in (slot_schema.slots or {}).items():
                ev_list = slots_payload.get(name, []) if isinstance(slots_payload.get(name), list) else []
                raw_values: List[str] = []
                evidence_ids: List[str] = []
                by_block: Dict[str, List[str]] = {}
                for ev in ev_list:
                    if not isinstance(ev, dict):
                        continue
                    bid = str(ev.get("evidence_id") or "")
                    evidence_ids.append(bid)
                    items = ev.get("list_items") if isinstance(ev.get("list_items"), list) else []
                    for it in items:
                        if str(it).strip():
                            raw_values.append(str(it).strip())
                            if bid:
                                by_block.setdefault(bid, []).append(str(it).strip())
                value_cleaning[name] = cleaner.clean(
                    values=raw_values,
                    semantic_type=str(spec.semantic_type or "generic"),
                    evidence_ids=[x for x in evidence_ids if x],
                )
                per_block: Dict[str, Any] = {}
                for bid, vals in by_block.items():
                    per_block[bid] = cleaner.clean(
                        values=vals,
                        semantic_type=str(spec.semantic_type or "generic"),
                        evidence_ids=[bid],
                    )
                value_cleaning_by_block[name] = per_block
        except Exception:
            value_cleaning = {}
            value_cleaning_by_block = {}

        instances.append(
            {
                "pattern_id": pattern.pattern_id,
                "slots": slots_payload,
                "value_cleaning": value_cleaning,
                "value_cleaning_by_block": value_cleaning_by_block,
                "filtered_out_non_symptom_lists": filtered_out_non_symptom_lists,
            }
        )
        return instances


class IntrospectionGate:
    """
    Explainable gate that reasons about pattern/slot coverage.

    The gate can reject only on hard pattern violations; soft/schema issues are
    advisory and used for explanation/planning.
    """

    def __init__(self, rim: Any) -> None:
        self.rim = rim

    def evaluate(
        self,
        *,
        q_prime_vec: Any,
        kv_relevance_delta: Optional[float],
        missing_hard: Sequence[str],
        missing_soft: Sequence[str],
        missing_schema: Sequence[str],
        pattern_id: str,
        matched_skeleton: str,
        slot_status: Dict[str, str],
        slot_schema: Optional[SlotSchema] = None,
        answer_style: str = "factual",
        question_intent: str = "legacy",
        semantic_instances: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        gate = self.rim.introspection_gate(
            q_prime_vec=q_prime_vec,
            kv_relevance_delta=kv_relevance_delta,
            pattern_mismatch=bool(missing_hard),
        )
        if missing_hard:
            gate["rationale"] = "pattern-missing-hard"
        elif missing_soft:
            gate["rationale"] = "pattern-missing-soft"
        elif missing_schema:
            gate["rationale"] = "schema-not-yet-instantiated"
        gate["pattern_id"] = pattern_id
        gate["matched_skeleton"] = matched_skeleton
        gate["matched_intent"] = question_intent
        gate["slot_status"] = dict(slot_status or {})
        gate["hard_rationale"] = list(missing_hard or [])
        gate["soft_rationale"] = list(missing_soft or [])
        schema_slot_missing: List[str] = []
        if slot_schema is not None:
            for name, spec in (slot_schema.slots or {}).items():
                if str(spec.inference_level).lower() != "schema":
                    continue
                if str((slot_status or {}).get(name) or "").lower() != "satisfied":
                    schema_slot_missing.append(str(name))
        if schema_slot_missing:
            missing_schema = list(missing_schema or []) + schema_slot_missing
            gate["rationale"] = "schema-not-yet-instantiated"
        gate["schema_rationale"] = list(missing_schema or [])

        allowed = _allowed_capabilities(slot_schema)
        final_style = _normalize_answer_style(answer_style)
        missing_all_required = _missing_all_required(slot_schema, semantic_instances)
        has_schema = bool(slot_schema) and any(
            str(spec.inference_level).lower() == "schema" for spec in (slot_schema.slots or {}).values()
        )
        schema_satisfied = has_schema and not bool(missing_schema)
        if schema_satisfied:
            allowed = allowed.intersection({"LIST_ONLY", "EXPLANATION"})
            if "LIST_ONLY" in allowed:
                final_style = "LIST_ONLY"
            elif "EXPLANATION" in allowed:
                final_style = "EXPLANATION"
        if missing_all_required:
            # R1/R2: forbid factual assertion when all required slots are empty.
            allowed.discard("FACTUAL_ASSERTION")
            if "EXPLANATION" in allowed:
                final_style = "EXPLANATION"
                gate["decision"] = "ALLOW"
                gate["decision_reason"] = "required slots missing; downgrade to explanation"
            else:
                gate["decision"] = "REFUSE"
                gate["decision_reason"] = "required slots missing; no safe answer capability"
        elif missing_hard:
            gate["decision"] = "REFUSE"
            gate["decision_reason"] = "hard-slot-missing"
        elif schema_satisfied and allowed:
            gate["decision"] = "ALLOW"
            gate["decision_reason"] = "schema-slot-satisfied; limit-style"
        elif not allowed:
            gate["decision"] = "REFUSE"
            gate["decision_reason"] = "no-allowed-capability"
        elif final_style not in allowed:
            gate["decision"] = "REFUSE"
            gate["decision_reason"] = "schema-level-slot-limits-answer"
        elif missing_schema:
            gate["decision"] = "REFUSE"
            gate["decision_reason"] = "schema-level-slot-limits-answer"
            if "EXPLANATION" in allowed:
                final_style = "EXPLANATION"
        else:
            gate["decision"] = "ALLOW"
            gate["decision_reason"] = "capability-allowed"
        gate["allowed_answer_capabilities"] = sorted(list(allowed))
        gate["final_answer_style"] = final_style

        if slot_schema is not None:
            gate["slot_schema"] = slot_schema.to_dict()
        if gate.get("decision") == "REFUSE":
            gate["final_answer_style"] = None
            gate["allowed_answer_capabilities"] = []
            if missing_schema or gate.get("decision_reason") == "schema-level-slot-limits-answer":
                gate["allow_rim_retry"] = True
        return gate


def _infer_kind(pattern_id: str, pattern_type: str) -> str:
    pid = str(pattern_id or "")
    if pid.startswith("abbr:") or pattern_type in {"lexical", "abbr", "abbreviation"}:
        return "abbr"
    if pid.startswith("schema:") or pattern_type in {"schema", "cooccurrence"}:
        return "schema"
    if pid.startswith("entity:") or pattern_type in {"entity"}:
        return "entity"
    return "other"


def _infer_semantic_type_from_slot(slot_name: str) -> str:
    s = str(slot_name or "").lower()
    if "clinical" in s or "symptom" in s:
        return "symptom"
    if "treatment" in s or "drug" in s:
        return "drug"
    return "generic"


def _default_question_skeleton(kind: str, slot_name: str = "") -> List[str]:
    if kind == "abbr":
        return ["X 是什么", "X 的全称是什么", "What is X", "What does X stand for"]
    if kind == "schema":
        sn = slot_name.split(":", 1)[1] if ":" in slot_name else slot_name
        sn = sn or "信息"
        return [f"X 的{sn}是什么", f"X 有哪些{sn}"]
    return ["X 是什么"]


def _score_skeleton(query: str, skeleton: str) -> float:
    q = str(query or "")
    sk = str(skeleton or "")
    if not q or not sk:
        return 0.0
    intent_words = ["什么", "哪些", "如何", "哪里", "why", "what", "how", "where"]
    score = 0.0
    for w in intent_words:
        if w in q.lower() and w in sk.lower():
            score += 0.5
    q_tokens = set(_tokenize(q))
    sk_tokens = set(t for t in _tokenize(sk) if t != "x")
    score += float(len(q_tokens.intersection(sk_tokens))) * 0.1
    return score


def _tokenize(text: str) -> List[str]:
    import re

    toks: List[str] = []
    toks.extend([t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9\-_/]+", text)])
    toks.extend(re.findall(r"[\u4e00-\u9fff]{2,4}", text))
    return toks


def _extract_evidence_types(block: Any) -> List[str]:
    meta = getattr(block, "meta", None) or {}
    types: List[str] = []
    if isinstance(meta.get("evidence_type"), str):
        types.append(meta.get("evidence_type"))
    if isinstance(meta.get("evidence_type"), list):
        types.extend(meta.get("evidence_type"))
    meta_payload = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    if isinstance(meta_payload.get("evidence_type"), str):
        types.append(meta_payload.get("evidence_type"))
    if isinstance(meta_payload.get("evidence_type"), list):
        types.extend(meta_payload.get("evidence_type"))
    if isinstance(meta_payload.get("block_type"), str):
        types.append(meta_payload.get("block_type"))
    if isinstance(meta.get("block_type"), str):
        types.append(meta.get("block_type"))
    # Treat blocks with valid abbreviation pairs as abbreviation evidence.
    pat = meta_payload.get("pattern") if isinstance(meta_payload.get("pattern"), dict) else {}
    abbr_pairs = pat.get("abbreviation_pairs") if isinstance(pat.get("abbreviation_pairs"), list) else []
    if _has_valid_abbr_pair(abbr_pairs):
        types.append("abbreviation")
    # Treat blocks with schema_slots as schema evidence.
    schema_slots = pat.get("schema_slots")
    if (isinstance(schema_slots, list) and schema_slots) or (
        isinstance(schema_slots, dict) and schema_slots
    ):
        types.append("schema")
    out = [str(t).strip().lower() for t in types if str(t).strip()]
    return list(dict.fromkeys(out))


def _has_valid_abbr_pair(abbr_pairs: List[Any]) -> bool:
    for ap in abbr_pairs or []:
        if not isinstance(ap, dict):
            continue
        abbr = str(ap.get("abbr") or "").strip()
        full = str(ap.get("full") or "").strip()
        if len(abbr) < 3:
            continue
        if len(full) < (len(abbr) + 4):
            continue
        full_low = full.lower()
        if full_low.startswith(("abstract", "keywords", "introduction")):
            continue
        first_word = full_low.split()[0] if full_low.split() else ""
        if len(first_word) < 4:
            continue
        if "confidence" in ap:
            try:
                conf = float(ap.get("confidence") or 0.0)
            except Exception:
                conf = 0.0
            if conf < 0.85:
                continue
        return True
    return False


class SymptomAwareListFilter:
    """
    Filter list-like candidates for schema:clinical_features.
    """

    _SYMPTOM_CUES = [
        "symptom",
        "symptoms",
        "clinical feature",
        "clinical features",
        "manifestation",
        "manifestations",
    ]
    _PATIENT_REGEX = [
        r"patients?\s+.*present\s+with",
        r"characterized\s+by",
        r"including\s+.*\b(fever|thrombocytopenia|bleeding|fatigue|vomiting|diarrhea)\b",
    ]
    _SYMPTOM_TERMS = [
        "fever",
        "thrombocytopenia",
        "bleeding",
        "fatigue",
        "vomiting",
        "diarrhea",
    ]
    _EXCLUDE_PATTERNS = [
        "supplementary",
        "table",
        "figure",
        "statistical analysis",
        "copyright",
        "methods",
        "p value",
        "confidence interval",
        "in addition,",
        "moreover,",
    ]

    @classmethod
    def allow_candidate(
        cls, *, list_features: Dict[str, Any], span_text: str, list_items: List[str]
    ) -> Tuple[bool, str, List[str]]:
        text = str(span_text or "")
        text_low = text.lower()
        for pat in cls._EXCLUDE_PATTERNS:
            if pat in text_low:
                return False, f"excluded:{pat}", []

        signals = list_features.get("signals") if isinstance(list_features.get("signals"), list) else []
        is_list_like = bool(list_features.get("is_list_like")) or bool(signals)
        if not is_list_like:
            return False, "missing_structure", []

        if not cls._has_symptom_cue(text):
            return False, "missing_symptom_cue", []

        filtered_items = cls._filter_list_items(list_items)
        if not filtered_items:
            return False, "list_items_not_symptom", []
        return True, "ok", filtered_items

    @classmethod
    def _has_symptom_cue(cls, text: str) -> bool:
        if not text:
            return False
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        windows: List[str] = []
        if not sentences:
            windows = [text]
        else:
            for i in range(len(sentences)):
                parts = []
                if i - 1 >= 0:
                    parts.append(sentences[i - 1])
                parts.append(sentences[i])
                if i + 1 < len(sentences):
                    parts.append(sentences[i + 1])
                windows.append(" ".join(parts))
        for w in windows:
            w_low = w.lower()
            if any(cue in w_low for cue in cls._SYMPTOM_CUES):
                return True
            for rx in cls._PATIENT_REGEX:
                try:
                    if re.search(rx, w_low, flags=re.IGNORECASE):
                        return True
                except Exception:
                    continue
        return False

    @classmethod
    def _filter_list_items(cls, list_items: List[str]) -> List[str]:
        if not list_items:
            return []
        kept: List[str] = []
        for it in list_items:
            item = str(it or "").strip()
            if not item:
                continue
            item_low = item.lower()
            if any(cue in item_low for cue in cls._SYMPTOM_CUES):
                kept.append(item)
                continue
            if any(term in item_low for term in cls._SYMPTOM_TERMS):
                kept.append(item)
                continue
            for rx in cls._PATIENT_REGEX:
                try:
                    if re.search(rx, item_low, flags=re.IGNORECASE):
                        kept.append(item)
                        break
                except Exception:
                    continue
        return list(dict.fromkeys(kept))


def _collect_evidence_for_slot(spec: SlotSpec, evidence_blocks: Sequence[Any]) -> List[Any]:
    out: List[Any] = []
    wanted = [str(x).strip().lower() for x in (spec.evidence_type or []) if str(x).strip()]
    for it in evidence_blocks or []:
        if not wanted:
            out.append(it)
            continue
        ev_types = _extract_evidence_types(it)
        if any(w in ev_types for w in wanted):
            out.append(it)
    return out


def _build_slot_evidence(
    spec: SlotSpec,
    evidence_blocks: Sequence[Any],
    *,
    pattern_id: str = "",
    block_text_lookup: Optional[Dict[str, str]] = None,
    filtered_out: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for it in _collect_evidence_for_slot(spec, evidence_blocks):
        meta = getattr(it, "meta", None) or {}
        bid = str(meta.get("block_id") or meta.get("chunk_id") or meta.get("id") or "")
        if not bid:
            continue
        source = str(meta.get("source") or meta.get("doc_id") or meta.get("doc_name") or "")
        span = ""
        if block_text_lookup and bid:
            span = str(block_text_lookup.get(bid) or "")
        elif isinstance(getattr(it, "text", None), str):
            span = str(getattr(it, "text") or "")
        meta_payload = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
        pat = meta_payload.get("pattern") if isinstance(meta_payload.get("pattern"), dict) else {}
        list_features = pat.get("list_features") if isinstance(pat.get("list_features"), dict) else {}
        list_items = []
        if isinstance(list_features.get("list_items"), list):
            list_items = list_features.get("list_items") or []
        elif isinstance(list_features.get("list_like_items"), list):
            list_items = list_features.get("list_like_items") or []
        if str(pattern_id) == "schema:clinical_features":
            allowed, reason, filtered_items = SymptomAwareListFilter.allow_candidate(
                list_features=list_features,
                span_text=span,
                list_items=list_items,
            )
            if not allowed:
                if isinstance(filtered_out, list) and bid:
                    filtered_out.append({"block_id": bid, "reason": reason})
                continue
            list_items = filtered_items
        items.append({"evidence_id": bid, "source": source, "span": span, "list_items": list_items})
    return items


def _parse_question_skeleton(payload: Any) -> Tuple[str, List[str]]:
    if isinstance(payload, dict):
        intent = str(payload.get("intent") or "legacy")
        forms = payload.get("surface_forms") or []
        if isinstance(forms, str):
            forms = [forms]
        return intent, [str(s) for s in forms if str(s).strip()]
    if isinstance(payload, list):
        return "legacy", [str(s) for s in payload if str(s).strip()]
    if isinstance(payload, str):
        return "legacy", [payload]
    return "legacy", []


def _normalize_answer_style(style: str) -> str:
    s = str(style or "").strip().lower()
    if s in {"factual", "factual_assertion", "fact"}:
        return "FACTUAL_ASSERTION"
    if s in {"explain", "explanation"}:
        return "EXPLANATION"
    if s in {"list", "list_only"}:
        return "LIST_ONLY"
    if s in {"refuse"}:
        return "REFUSE"
    return "FACTUAL_ASSERTION"


def _allowed_capabilities(slot_schema: Optional[SlotSchema]) -> set[str]:
    allowed = {"FACTUAL_ASSERTION", "EXPLANATION", "LIST_ONLY"}
    if slot_schema is None:
        return allowed
    has_schema = False
    for spec in (slot_schema.slots or {}).values():
        if spec.inference_level == "schema":
            has_schema = True
        if not spec.required:
            continue
        if spec.inference_level == "schema":
            allowed = allowed.intersection({"LIST_ONLY", "EXPLANATION"})
        elif spec.inference_level == "soft":
            allowed = allowed.intersection({"FACTUAL_ASSERTION", "EXPLANATION"})
        elif spec.inference_level == "hard":
            allowed = allowed.intersection({"FACTUAL_ASSERTION"})
    if has_schema:
        allowed.discard("FACTUAL_ASSERTION")
    return allowed


def find_unconsumed_evidence_blocks(
    evidence_blocks: Sequence[Any],
    slot_schema: Optional[SlotSchema],
) -> List[str]:
    """
    Return block_ids that contain schema metadata but are not consumed by any slot.
    """
    if not evidence_blocks or slot_schema is None:
        return []
    consumed: set[str] = set()
    for spec in (slot_schema.slots or {}).values():
        for it in _collect_evidence_for_slot(spec, evidence_blocks):
            meta = getattr(it, "meta", None) or {}
            bid = str(meta.get("block_id") or meta.get("chunk_id") or meta.get("id") or "")
            if bid:
                consumed.add(bid)
    out: List[str] = []
    for it in evidence_blocks or []:
        meta = getattr(it, "meta", None) or {}
        meta_payload = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
        pat = meta_payload.get("pattern") if isinstance(meta_payload.get("pattern"), dict) else {}
        schema_slots = pat.get("schema_slots")
        has_schema = (isinstance(schema_slots, list) and schema_slots) or (
            isinstance(schema_slots, dict) and schema_slots
        )
        if not has_schema:
            continue
        bid = str(meta.get("block_id") or meta.get("chunk_id") or meta.get("id") or "")
        if bid and bid not in consumed:
            out.append(bid)
    return out


def compute_slot_status_from_instances(
    slot_schema: Optional[SlotSchema],
    semantic_instances: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Compute slot_status from semantic_instances only (single source of truth).
    """
    status: Dict[str, str] = {}
    if slot_schema is None:
        return status
    slots_payload: Dict[str, List[Dict[str, Any]]] = {}
    for inst in semantic_instances or []:
        if isinstance(inst, dict) and isinstance(inst.get("slots"), dict):
            for k, v in inst["slots"].items():
                if isinstance(v, list):
                    slots_payload.setdefault(str(k), []).extend(v)
    for name, spec in (slot_schema.slots or {}).items():
        items = slots_payload.get(name, []) if isinstance(slots_payload.get(name), list) else []
        if len(items) >= int(spec.min_evidence):
            status[name] = "satisfied"
        else:
            status[name] = "missing"
    return status


def _missing_all_required(
    slot_schema: Optional[SlotSchema], semantic_instances: Optional[Sequence[Dict[str, Any]]]
) -> bool:
    if slot_schema is None:
        return False
    required_slots = [k for k, v in (slot_schema.slots or {}).items() if v.required]
    if not required_slots:
        return False
    if not semantic_instances:
        return True
    slot_state: Dict[str, Any] = {}
    for inst in semantic_instances:
        if isinstance(inst, dict) and isinstance(inst.get("slots"), dict):
            slot_state = inst["slots"]
            break
    if not slot_state:
        return True
    for s in required_slots:
        vals = slot_state.get(s) or []
        if isinstance(vals, list) and len(vals) > 0:
            return False
    return True

