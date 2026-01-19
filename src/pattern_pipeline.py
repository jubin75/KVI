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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .pattern_contract import PatternContract
from .pattern_retriever import PatternRetriever, PatternRetrieveResult


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
    ) -> List[PatternContract]:
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

    def _parse_contracts(self, payload: Any) -> List[PatternContract]:
        if not isinstance(payload, dict):
            return []
        patterns = payload.get("patterns")
        if not isinstance(patterns, dict):
            return []
        out: List[PatternContract] = []
        out.extend(self._parse_pattern_list(patterns.get("hard"), level="hard"))
        out.extend(self._parse_pattern_list(patterns.get("soft"), level="soft"))
        return out

    def _parse_pattern_list(self, items: Any, *, level: str) -> List[PatternContract]:
        if not isinstance(items, list):
            return []
        out: List[PatternContract] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            pid = str(it.get("id") or "").strip()
            if not pid:
                continue
            ptype = str(it.get("type") or "").strip().lower()
            rule = str(it.get("rule") or "").strip()
            kind = _infer_kind(pid, ptype)
            expected: Dict[str, Any] = {"pattern_type": ptype, "rule": rule}
            if kind == "abbr":
                expected["abbr"] = pid.split(":", 1)[1] if ":" in pid else pid
                expected.setdefault("expansions", [])
            elif kind == "schema":
                slots = it.get("slots")
                expected["schema_slots"] = slots if isinstance(slots, list) else []
            elif kind == "entity":
                ent = pid.split(":", 1)[1] if ":" in pid else pid
                expected["entities"] = [ent] if ent else []
            out.append(
                PatternContract(
                    pattern_id=pid,
                    expected_information=expected,
                    required_signals={"must_contain": [], "must_not_be_only": []},
                    min_information_density=0.2,
                    level=level,
                    kind=kind,
                )
            )
        return out


class PatternMatcher:
    """
    Match topic-level Pattern Contracts against a query using a PatternRetriever.

    This does not create new patterns from the prompt; it only selects which
    existing contracts are relevant for the current query.
    """

    def __init__(self, contracts: Sequence[PatternContract], retriever: Optional[PatternRetriever] = None) -> None:
        self.contracts = list(contracts or [])
        self.retriever = retriever or PatternRetriever()

    def match(self, query: str) -> Tuple[PatternRetrieveResult, List[PatternContract]]:
        result = self.retriever.retrieve(str(query or ""))
        hit_ids = {str(h.block_id) for h in (result.pattern_hits or [])}
        if hit_ids:
            matched = [c for c in self.contracts if str(c.pattern_id) in hit_ids]
        else:
            matched = list(self.contracts)
        return result, matched


@dataclass
class SlotSchema:
    """
    SlotSchema defines which semantic fields can be filled based on matched patterns.

    Slots are an interface for generation planning and introspection, not answers.
    """

    slots: Dict[str, str]

    @classmethod
    def from_contracts(cls, contracts: Sequence[PatternContract]) -> "SlotSchema":
        slots: Dict[str, str] = {}
        for c in contracts or []:
            if c.kind != "schema":
                continue
            schema_slots = c.expected_information.get("schema_slots") or []
            for s in schema_slots if isinstance(schema_slots, list) else []:
                s2 = str(s or "").strip()
                if not s2:
                    continue
                # "fillable" means pattern allows attempting this slot.
                slots[s2] = "fillable"
        return cls(slots=slots)

    def to_dict(self) -> Dict[str, str]:
        return dict(self.slots or {})


class SemanticInstanceBuilder:
    """
    Build evidence-grounded semantic instances.

    Each instance must cite >=1 block_id. This builder does NOT invent content;
    it only packages evidence references and slot state.
    """

    def build(self, *, evidence_blocks: Sequence[Any], slot_schema: SlotSchema) -> List[Dict[str, Any]]:
        instances: List[Dict[str, Any]] = []
        for it in evidence_blocks or []:
            meta = getattr(it, "meta", None) or {}
            bid = str(meta.get("block_id") or meta.get("chunk_id") or meta.get("id") or "")
            if not bid:
                continue
            instances.append({"block_id": bid, "evidence_ids": [bid], "slots": slot_schema.to_dict()})
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
        slot_schema: Optional[SlotSchema] = None,
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
        if slot_schema is not None:
            gate["slot_schema"] = slot_schema.to_dict()
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

