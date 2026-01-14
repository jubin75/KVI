"""
Pattern-first retrieval (KVI 2.0 / RIM v0.4).

Spec (docs/07_KVI_RMI.md):
- Pattern-first handles low-entropy, form-stable knowledge that MUST NOT be handled by ANN embeddings.
  Examples: abbreviation expansions, schema triggers, fixed entity aliases.
- Output is structured and used as priors/constraints for semantic-second retrieval and introspection gate.

This file provides a minimal, extensible implementation:
- Data types: PatternHit, PatternRetrieveResult
- Retriever: PatternRetriever with pluggable sub-retrievers

The default implementation is intentionally conservative (safe empty outputs) unless provided indices/maps.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class PatternHit:
    block_id: str
    hit_types: List[str]  # e.g. ["literal", "schema", "structure"]
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PatternRetrieveResult:
    pattern_hits: List[PatternHit]
    recall_size: int
    debug_info: Optional[Dict[str, Any]] = None


class PatternRetriever:
    """
    Stage-1 Retriever in KVI 2.0:
    Pattern-first (non-semantic, structure-aware, fast recall)
    """

    def __init__(
        self,
        *,
        alias_map: Optional[Dict[str, Sequence[str]]] = None,
        schema_triggers: Optional[Dict[str, Sequence[str]]] = None,
        fixed_entities: Optional[Dict[str, str]] = None,
    ) -> None:
        # All indices are optional to allow empty / incremental implementation.
        self.alias_map = alias_map or {}
        self.schema_triggers = schema_triggers or {}
        self.fixed_entities = fixed_entities or {}

    @classmethod
    def from_dir(cls, pattern_dir: str | Path) -> "PatternRetriever":
        """
        Load Pattern-first indices from a sidecar directory (built from blocks.v2.jsonl).

        Expected files (any subset is ok):
        - alias_map.json            # {ABBR: [full1, full2, ...]}
        - schema_triggers.json      # {keyword: [slot1, slot2, ...]}
        - fixed_entities.json       # {alias: canonical}
        """
        p = Path(pattern_dir)
        alias_map: Dict[str, Sequence[str]] = {}
        schema_triggers: Dict[str, Sequence[str]] = {}
        fixed_entities: Dict[str, str] = {}

        def _load_json(path: Path) -> Optional[dict]:
            try:
                if not path.exists():
                    return None
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None

        a = _load_json(p / "alias_map.json")
        if isinstance(a, dict):
            alias_map = {str(k): list(v) if isinstance(v, list) else [str(v)] for k, v in a.items()}
        s = _load_json(p / "schema_triggers.json")
        if isinstance(s, dict):
            schema_triggers = {str(k): list(v) if isinstance(v, list) else [str(v)] for k, v in s.items()}
        f = _load_json(p / "fixed_entities.json")
        if isinstance(f, dict):
            fixed_entities = {str(k): str(v) for k, v in f.items()}

        return cls(alias_map=alias_map, schema_triggers=schema_triggers, fixed_entities=fixed_entities)

    def retrieve(self, query: str) -> PatternRetrieveResult:
        literal_hits = self._retrieve_literal(query)
        structure_hits = self._retrieve_structure(query)
        block_hits = self._retrieve_block_type(query)

        merged_hits = self._merge_hits(literal_hits, structure_hits, block_hits)
        return PatternRetrieveResult(
            pattern_hits=merged_hits,
            recall_size=len(merged_hits),
            debug_info={
                "literal_hits": len(literal_hits),
                "structure_hits": len(structure_hits),
                "block_hits": len(block_hits),
            },
        )

    def _retrieve_literal(self, query: str) -> List[PatternHit]:
        """
        Literal / token / n-gram based pattern matching.
        Minimal implementation:
        - abbreviation expansion (via alias_map or uppercase token heuristic)
        - fixed entity aliasing (via fixed_entities)
        """
        q = str(query or "")
        hits: List[PatternHit] = []

        # fixed entity aliases (string contains -> canonical)
        for k, canonical in self.fixed_entities.items():
            if not k:
                continue
            if k in q:
                hits.append(
                    PatternHit(
                        block_id=f"fixed_entity:{canonical}",
                        hit_types=["fixed_entity", "literal"],
                        confidence=0.95,
                        metadata={"source": "pattern", "pattern_type": "fixed_entity", "payload": {"mention": k, "canonical": canonical}},
                    )
                )

        # explicit alias map expansions (token exact match, case-insensitive)
        toks = re.findall(r"[A-Za-z0-9\-_/]+", q)
        lower_toks = {t.lower() for t in toks if t}
        for abbr, exps in self.alias_map.items():
            if not abbr:
                continue
            if abbr.lower() in lower_toks:
                hits.append(
                    PatternHit(
                        block_id=f"abbr:{abbr}",
                        hit_types=["abbreviation_expansion", "literal"],
                        confidence=0.95,
                        metadata={"source": "pattern", "pattern_type": "abbreviation_expansion", "payload": {"abbr": abbr, "expansions": list(exps)}},
                    )
                )

        # heuristic abbreviations (uppercase 2-10) as low-confidence cues
        for t in toks:
            if 2 <= len(t) <= 10 and t.isupper():
                if t.lower() not in self.alias_map:
                    hits.append(
                        PatternHit(
                            block_id=f"abbr_heur:{t}",
                            hit_types=["abbreviation_candidate", "literal"],
                            confidence=0.35,
                            metadata={"source": "pattern", "pattern_type": "abbreviation_expansion", "payload": {"abbr": t, "expansions": []}},
                        )
                    )

        return hits

    def _retrieve_structure(self, query: str) -> List[PatternHit]:
        """
        Structural / schema / slot-level pattern matching.
        Minimal implementation:
        - keyword triggers to schema slots (schema_triggers)
        """
        q = str(query or "").lower()
        hits: List[PatternHit] = []
        for kw, slots in self.schema_triggers.items():
            if not kw:
                continue
            if kw.lower() in q:
                hits.append(
                    PatternHit(
                        block_id=f"schema:{kw}",
                        hit_types=["schema_trigger", "structure"],
                        confidence=0.7,
                        metadata={"source": "pattern", "pattern_type": "schema_trigger", "payload": {"keyword": kw, "slots": list(slots)}},
                    )
                )
        return hits

    def _retrieve_block_type(self, query: str) -> List[PatternHit]:
        """
        Block-type or document-template level matching.
        Kept empty by default (project-specific).
        """
        _ = query
        return []

    def _merge_hits(self, *hit_lists: List[PatternHit]) -> List[PatternHit]:
        """
        Merge hits from different pattern channels.
        Deduplicate by block_id and aggregate hit_types.
        """
        hit_map: Dict[str, PatternHit] = {}
        for hits in hit_lists:
            for hit in hits:
                if hit.block_id not in hit_map:
                    hit_map[hit.block_id] = hit
                else:
                    existing = hit_map[hit.block_id]
                    existing.hit_types = list(set(existing.hit_types + hit.hit_types))
                    existing.confidence = max(float(existing.confidence), float(hit.confidence))
        return list(hit_map.values())

