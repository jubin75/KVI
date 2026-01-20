"""
Pattern Contract (Detoxification) for KVI 2.0.

Core rules (docs/08_Detoxification.md):
- Pattern Contract declares WHAT information is required.
- Validator checks Evidence metadata only (no LLM, no embeddings, no text RAG).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

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
    # Contract level: hard must be satisfied; soft is advisory only.
    level: str = "soft"  # "hard" | "soft"
    # Contract kind: schema contracts are planning-only (not reject).
    kind: str = "other"  # "abbr" | "schema" | "entity" | "other"


@dataclass
class ContractValidationResult:
    pattern_id: str
    fulfilled: bool
    score: float
    violations: List[str]


class PatternContractValidator:
    """
    Validate PatternContract against Evidence metadata + optional block text lookup.
    - No LLM
    - No embeddings
    - No text-level RAG
    - Pattern→Evidence mapping is explicit (pattern != KV tag)
    """

    def validate(
        self,
        contract: PatternContract,
        evidence_blocks: Sequence[Any],
        *,
        block_text_lookup: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        signal_set, info_density = _collect_signals_from_evidence(evidence_blocks)
        must_contain = list(contract.required_signals.get("must_contain") or [])
        must_not_be_only = list(contract.required_signals.get("must_not_be_only") or [])

        satisfied = any(_satisfy_contract(contract, it, block_text_lookup) for it in (evidence_blocks or []))
        violations: List[str] = []
        if not satisfied and must_contain:
            violations.append(f"missing_required={must_contain}")

        if must_not_be_only:
            # Violation if evidence has only generic/background signals.
            if signal_set and all(s in set(must_not_be_only) for s in signal_set):
                violations.append("evidence_only_generic")

        if info_density < float(contract.min_information_density):
            violations.append(f"low_information_density<{contract.min_information_density}")

        # Score: blend of satisfy + density (soft contracts can be low without rejection).
        coverage = 1.0 if satisfied else 0.0
        score = max(0.0, min(1.0, 0.7 * coverage + 0.3 * info_density))

        fulfilled = (len(violations) == 0) and bool(satisfied or not must_contain)
        return {
            "pattern_id": contract.pattern_id,
            "fulfilled": bool(fulfilled),
            "score": float(score),
            "violations": violations,
            "info_density": float(info_density),
            "required_signals": dict(contract.required_signals),
            "level": contract.level,
            "kind": contract.kind,
            "satisfied": bool(satisfied),
        }

    def validate_all(
        self,
        contracts: Sequence[PatternContract],
        evidence_blocks: Sequence[Any],
        *,
        block_text_lookup: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        results = [self.validate(c, evidence_blocks, block_text_lookup=block_text_lookup) for c in (contracts or [])]

        hard_missing = [r["pattern_id"] for r in results if r.get("level") == "hard" and not r.get("satisfied")]
        soft_missing = [r["pattern_id"] for r in results if r.get("level") == "soft" and not r.get("satisfied") and r.get("kind") != "schema"]
        schema_missing = [r["pattern_id"] for r in results if r.get("kind") == "schema" and not r.get("satisfied")]

        fulfilled_hard = len(hard_missing) == 0
        score = sum(float(r.get("score") or 0.0) for r in results) / max(1, len(results)) if results else 1.0
        violations: List[str] = []
        for r in results:
            for v in r.get("violations") or []:
                violations.append(f"{r.get('pattern_id')}: {v}")
        return {
            "fulfilled": bool(fulfilled_hard),
            "score": float(score),
            "violations": violations,
            "results": results,
            "hard_missing": hard_missing,
            "soft_missing": soft_missing,
            "schema_missing": schema_missing,
            "hard_fulfilled": bool(fulfilled_hard),
            "soft_fulfilled": len(soft_missing) == 0,
        }


def filter_evidence_by_contracts(
    contracts: Sequence[PatternContract],
    evidence_blocks: Sequence[Any],
    *,
    block_text_lookup: Optional[Dict[str, str]] = None,
    pattern_id: Optional[str] = None,
) -> List[Any]:
    """
    Return evidence blocks that satisfy at least one PatternContract.
    If pattern_id is provided, only contracts with that id are considered.
    """
    if not contracts or not evidence_blocks:
        return list(evidence_blocks or [])
    if pattern_id:
        target = [c for c in contracts if str(c.pattern_id) == str(pattern_id)]
    else:
        target = list(contracts)
    if not target:
        return list(evidence_blocks or [])
    out: List[Any] = []
    for it in evidence_blocks or []:
        if any(_satisfy_contract(c, it, block_text_lookup=block_text_lookup) for c in target):
            out.append(it)
    return out


def run_pattern_first(
    query_text: str,
    pattern_result: PatternRetrieveResult,
    *,
    hard_pattern_ids: Optional[Sequence[str]] = None,
    soft_pattern_ids: Optional[Sequence[str]] = None,
) -> List[PatternContract]:
    """
    Pattern-first output must be PatternContract list (not keywords / ids).
    This function converts PatternRetriever hits -> PatternContracts.
    """
    _ = query_text
    contracts: List[PatternContract] = []
    hard_set = {str(x) for x in (hard_pattern_ids or []) if str(x).strip()}
    soft_set = {str(x) for x in (soft_pattern_ids or []) if str(x).strip()}
    for hit in pattern_result.pattern_hits or []:
        meta = hit.metadata or {}
        ptype = str(meta.get("pattern_type") or "")
        payload = meta.get("payload") or {}

        if ptype == "abbreviation_expansion":
            abbr = str(payload.get("abbr") or "")
            exps = payload.get("expansions") if isinstance(payload.get("expansions"), list) else []
            required = {"must_contain": [f"abbr:{abbr}"] if abbr else [], "must_not_be_only": []}
            level = "hard" if (len(abbr) >= 4 and float(getattr(hit, "confidence", 0.0)) >= 0.9) else "soft"
            if str(hit.block_id) in hard_set:
                level = "hard"
            if str(hit.block_id) in soft_set:
                level = "soft"
            contracts.append(
                PatternContract(
                    pattern_id=str(hit.block_id),
                    expected_information={"entity_types": ["abbreviation"], "abbr": abbr, "expansions": list(exps)},
                    required_signals=required,
                    min_information_density=0.3,
                    level=level,
                    kind="abbr",
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
                    level="soft" if str(hit.block_id) not in hard_set else "hard",
                    kind="schema",
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
                    level="soft" if str(hit.block_id) not in hard_set else "hard",
                    kind="entity",
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
                level="soft" if str(hit.block_id) not in hard_set else "hard",
                kind="other",
            )
        )
    return contracts


def _satisfy_contract(contract: PatternContract, block: Any, block_text_lookup: Optional[Dict[str, str]] = None) -> bool:
    meta = getattr(block, "meta", None) or {}
    meta_payload = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
    pat = meta_payload.get("pattern") if isinstance(meta_payload.get("pattern"), dict) else {}
    bid = str(meta.get("block_id") or meta.get("chunk_id") or meta.get("id") or "")
    text = ""
    if block_text_lookup and bid:
        text = str(block_text_lookup.get(bid) or "")
    text_low = text.lower()
    signals, _ = _collect_signals_from_evidence([block])

    if contract.kind == "abbr":
        abbr = str(contract.expected_information.get("abbr") or "")
        exps = contract.expected_information.get("expansions") or []
        abbr_pairs = pat.get("abbreviation_pairs") if isinstance(pat.get("abbreviation_pairs"), list) else []
        for ap in abbr_pairs:
            ok, ap_abbr, _ = _parse_valid_abbr_pair(ap)
            if ok and (not abbr or ap_abbr.upper() == abbr.upper()):
                return True
        for e in exps if isinstance(exps, list) else []:
            e2 = str(e or "").strip()
            if e2 and e2.lower() in text_low:
                return True
        return False

    if contract.kind == "schema":
        slots = contract.expected_information.get("schema_slots") or []
        for s in slots if isinstance(slots, list) else []:
            s2 = str(s or "").strip()
            if s2 and (f"schema_slot:{s2}" in signals):
                return True
        return False

    must_contain = list(contract.required_signals.get("must_contain") or [])
    return any(s in signals for s in must_contain) if must_contain else False


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
            ok, abbr, full = _parse_valid_abbr_pair(ap)
            if not ok:
                continue
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


def _parse_valid_abbr_pair(ap: Any) -> Tuple[bool, str, str]:
    if not isinstance(ap, dict):
        return False, "", ""
    abbr = str(ap.get("abbr") or "").strip()
    full = str(ap.get("full") or "").strip()
    if len(abbr) < 3:
        return False, "", ""
    if len(full) < (len(abbr) + 4):
        return False, "", ""
    full_low = full.lower()
    if full_low.startswith(("abstract", "keywords", "introduction")):
        return False, "", ""
    first_word = full_low.split()[0] if full_low.split() else ""
    if len(first_word) < 4:
        return False, "", ""
    if "confidence" in ap:
        try:
            conf = float(ap.get("confidence") or 0.0)
        except Exception:
            conf = 0.0
        if conf < 0.85:
            return False, "", ""
    return True, abbr, full


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
    level="soft",
    kind="schema",
)
