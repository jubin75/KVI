"""
Pattern Contract (Detoxification) for KVI 2.0.

Core rules (docs/08_Detoxification.md):
- Pattern Contract declares WHAT information is required.
- Validator checks Evidence metadata only (no LLM, no embeddings, no text RAG).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from .pattern_retriever import PatternRetrieveResult


@dataclass
class PatternContract:
    pattern_id: str
    # Information requirement (WHAT)
    expected_information: Dict[str, Any]
    # Structural acceptance criteria (HOW)
    required_signals: Dict[str, List[str]]
    # Minimum information density (0~1)
    min_information_density: float


@dataclass
class ContractValidationResult:
    pattern_id: str
    fulfilled: bool
    score: float
    violations: List[str]


class PatternContractValidator:
    """
    Validate PatternContract against Evidence metadata only.
    - No LLM
    - No embeddings
    - No text-level RAG
    """

    def validate(self, contract: PatternContract, evidence_blocks: Sequence[Any]) -> Dict[str, Any]:
        signal_set, info_density = _collect_signals_from_evidence(evidence_blocks)
        must_contain = list(contract.required_signals.get("must_contain") or [])
        must_not_be_only = list(contract.required_signals.get("must_not_be_only") or [])

        missing = [s for s in must_contain if s not in signal_set]
        violations: List[str] = []
        if missing:
            violations.append(f"missing_required={missing}")

        if must_not_be_only:
            # Violation if evidence has only generic/background signals.
            if signal_set and all(s in set(must_not_be_only) for s in signal_set):
                violations.append("evidence_only_generic")

        if info_density < float(contract.min_information_density):
            violations.append(f"low_information_density<{contract.min_information_density}")

        # Score: blend of required-signal coverage and information density
        coverage = 1.0
        if must_contain:
            coverage = 1.0 - (len(missing) / max(1, len(must_contain)))
        score = max(0.0, min(1.0, 0.7 * coverage + 0.3 * info_density))

        fulfilled = (len(violations) == 0)
        return {
            "pattern_id": contract.pattern_id,
            "fulfilled": bool(fulfilled),
            "score": float(score),
            "violations": violations,
            "info_density": float(info_density),
            "required_signals": dict(contract.required_signals),
        }

    def validate_all(self, contracts: Sequence[PatternContract], evidence_blocks: Sequence[Any]) -> Dict[str, Any]:
        results = [self.validate(c, evidence_blocks) for c in (contracts or [])]
        fulfilled = all(r.get("fulfilled") is True for r in results) if results else True
        score = sum(float(r.get("score") or 0.0) for r in results) / max(1, len(results)) if results else 1.0
        violations: List[str] = []
        for r in results:
            for v in r.get("violations") or []:
                violations.append(f"{r.get('pattern_id')}: {v}")
        return {
            "fulfilled": bool(fulfilled),
            "score": float(score),
            "violations": violations,
            "results": results,
        }


def run_pattern_first(query_text: str, pattern_result: PatternRetrieveResult) -> List[PatternContract]:
    """
    Pattern-first output must be PatternContract list (not keywords / ids).
    This function converts PatternRetriever hits -> PatternContracts.
    """
    _ = query_text
    contracts: List[PatternContract] = []
    for hit in pattern_result.pattern_hits or []:
        meta = hit.metadata or {}
        ptype = str(meta.get("pattern_type") or "")
        payload = meta.get("payload") or {}

        if ptype == "abbreviation_expansion":
            abbr = str(payload.get("abbr") or "")
            exps = payload.get("expansions") if isinstance(payload.get("expansions"), list) else []
            required = {"must_contain": [f"abbr:{abbr}"] if abbr else [], "must_not_be_only": []}
            contracts.append(
                PatternContract(
                    pattern_id=str(hit.block_id),
                    expected_information={"entity_types": ["abbreviation"], "abbr": abbr, "expansions": list(exps)},
                    required_signals=required,
                    min_information_density=0.3,
                )
            )
            continue

        if ptype == "schema_trigger":
            slots = payload.get("slots") if isinstance(payload.get("slots"), list) else []
            required = {
                "must_contain": [f"schema_slot:{s}" for s in slots if str(s).strip()],
                "must_not_be_only": ["block_type:general"],
            }
            contracts.append(
                PatternContract(
                    pattern_id=str(hit.block_id),
                    expected_information={"schema_slots": list(slots)},
                    required_signals=required,
                    min_information_density=0.45,
                )
            )
            continue

        if ptype == "fixed_entity":
            canonical = str(payload.get("canonical") or payload.get("entity") or "")
            required = {"must_contain": [f"entity:{canonical}"] if canonical else [], "must_not_be_only": []}
            contracts.append(
                PatternContract(
                    pattern_id=str(hit.block_id),
                    expected_information={"entities": [canonical] if canonical else []},
                    required_signals=required,
                    min_information_density=0.35,
                )
            )
            continue

        # Fallback: keep a minimal contract for unknown pattern types.
        contracts.append(
            PatternContract(
                pattern_id=str(hit.block_id),
                expected_information={"pattern_type": ptype, "hit_types": list(hit.hit_types or [])},
                required_signals={"must_contain": [], "must_not_be_only": []},
                min_information_density=0.2,
            )
        )
    return contracts


def _collect_signals_from_evidence(evidence_blocks: Sequence[Any]) -> Tuple[Set[str], float]:
    """
    Build a flat signal set from Evidence metadata only.
    Return (signals, info_density).
    """
    signals: Set[str] = set()
    info_signals: Set[str] = set()
    blocks = list(evidence_blocks or [])

    def _add(sig: str, *, info: bool = True) -> None:
        if not sig:
            return
        signals.add(sig)
        if info:
            info_signals.add(sig)

    for it in blocks:
        meta = getattr(it, "meta", None) or {}
        meta_payload = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
        pat = meta_payload.get("pattern") if isinstance(meta_payload.get("pattern"), dict) else {}

        # Abbreviation pairs
        abbr_pairs = pat.get("abbreviation_pairs") if isinstance(pat.get("abbreviation_pairs"), list) else []
        for ap in abbr_pairs:
            if not isinstance(ap, dict):
                continue
            abbr = str(ap.get("abbr") or "").upper()
            full = str(ap.get("full") or "")
            if abbr:
                _add(f"abbr:{abbr}")
            if abbr and full:
                _add(f"abbr_full:{abbr}::{full}")

        # Entities
        entities = pat.get("entities") if isinstance(pat.get("entities"), list) else []
        for e in entities:
            s = str(e or "").strip()
            if s:
                _add(f"entity:{s}")

        # Schema slots
        slots = pat.get("schema_slots") if isinstance(pat.get("schema_slots"), list) else []
        for s in slots:
            s2 = str(s or "").strip()
            if s2:
                _add(f"schema_slot:{s2}")

        # Block type
        block_type = str(meta_payload.get("block_type") or meta.get("block_type") or "").strip()
        if block_type:
            _add(f"block_type:{block_type}", info=False)
        if bool(meta.get("is_table")):
            _add("block_type:table", info=False)

        # Section / paragraph tags (if provided)
        for k in ("section", "section_title", "paragraph_type", "chunk_type"):
            v = meta_payload.get(k)
            if isinstance(v, str) and v.strip():
                _add(f"{k}:{v.strip()}")

    # Information density: ratio of info signals over a small scale of evidence size.
    denom = max(1.0, float(len(blocks)) * 2.0)
    info_density = min(1.0, float(len(info_signals)) / denom)
    return signals, info_density


# Example medical contract for testing/debug
EXAMPLE_SFTSV_CONTRACT = PatternContract(
    pattern_id="schema:sftsv_fda_drug_research",
    expected_information={
        "entity_types": ["drug"],
        "relation_types": ["approved_for", "studied_for", "repurposed_for"],
        "domain": "biomedical",
        "disease": "SFTSV",
    },
    required_signals={"must_contain": ["schema_slot:treatment"], "must_not_be_only": ["block_type:general"]},
    min_information_density=0.6,
)
