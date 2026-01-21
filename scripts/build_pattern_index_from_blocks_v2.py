"""
Build Pattern-first sidecar index + enrich blocks.jsonl metadata.

Inputs:
- blocks.jsonl (JSONL of evidence blocks)

Outputs:
- blocks.enriched.jsonl (same records, with enhanced `metadata` fields)
- pattern_out_dir/alias_map.json
- pattern_out_dir/schema_triggers.json
- pattern_out_dir/fixed_entities.json
- pattern_out_dir/block_patterns.jsonl (per-block extracted pattern signals; debug/traceability)

This does NOT require rebuilding the semantic KV bank. It produces a Pattern-first index that can
be used by `PatternRetriever.from_dir(...)`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple


def _ensure_repo_root_on_syspath() -> None:
    # Support both layouts:
    # 1) local: <repo>/external_kv_injection/scripts (package import external_kv_injection.*)
    # 2) remote: <repo>/scripts with flat src/ (import src.*)
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [repo_root, repo_root.parent]
    for p in candidates:
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


_ensure_repo_root_on_syspath()

try:
    from external_kv_injection.src.pattern_extraction import (  # type: ignore
        extract_abbreviation_pairs,
        extract_entities,
        infer_block_type,
        infer_schema_slots_from_text,
    )
    from external_kv_injection.src.evidence.list_feature_extractor import EvidenceListFeatureExtractor  # type: ignore
    from external_kv_injection.src.evidence.evidence_unit_extractor import EvidenceUnitExtractor  # type: ignore
except ModuleNotFoundError:
    from src.pattern_extraction import (  # type: ignore
        extract_abbreviation_pairs,
        extract_entities,
        infer_block_type,
        infer_schema_slots_from_text,
    )
    from src.evidence.list_feature_extractor import EvidenceListFeatureExtractor  # type: ignore
    from src.evidence.evidence_unit_extractor import EvidenceUnitExtractor  # type: ignore


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _default_schema_triggers() -> Dict[str, List[str]]:
    """
    Conservative trigger keywords to schema slots.
    This is meant as a starting point; users can edit schema_triggers.json.
    """
    return {
        "传播": ["transmission"],
        "途径": ["transmission"],
        "transmission": ["transmission"],
        "vector": ["transmission"],
        "蜱": ["transmission"],
        "tick": ["transmission"],
        "症状": ["clinical_features"],
        "表现": ["clinical_features"],
        "clinical": ["clinical_features"],
        "diagnosis": ["diagnosis"],
        "诊断": ["diagnosis"],
        "检测": ["diagnosis"],
        "treatment": ["treatment"],
        "治疗": ["treatment"],
        "drug": ["treatment"],
        "药物": ["treatment"],
        "prevention": ["prevention"],
        "预防": ["prevention"],
        "vaccine": ["prevention"],
        "疫苗": ["prevention"],
        "pathogenesis": ["pathogenesis"],
        "mechanism": ["mechanism"],
        "发病机制": ["pathogenesis"],
        "致病": ["pathogenesis"],
        "流行病学": ["epidemiology"],
        "epidemiology": ["epidemiology"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks_jsonl_in", required=True, help="Input blocks.jsonl")
    p.add_argument("--blocks_jsonl_out", required=True, help="Output enriched blocks jsonl (e.g., blocks.enriched.jsonl)")
    p.add_argument("--pattern_out_dir", required=True, help="Output directory for pattern index sidecar")
    p.add_argument("--max_blocks", type=int, default=0, help="If >0, only process first N blocks (debug).")
    p.add_argument("--max_pairs_per_block", type=int, default=32)
    p.add_argument("--max_entities_per_block", type=int, default=64)
    p.add_argument("--min_abbr_confidence", type=float, default=0.6, help="Only keep abbr pairs >= this confidence in alias_map.")
    args = p.parse_args()

    in_path = Path(str(args.blocks_jsonl_in))
    out_path = Path(str(args.blocks_jsonl_out))
    pat_dir = Path(str(args.pattern_out_dir))
    pat_dir.mkdir(parents=True, exist_ok=True)

    alias_counts: DefaultDict[str, Counter] = defaultdict(Counter)
    fixed_entities: Dict[str, str] = {}  # left empty by default; user can curate later
    schema_triggers = _default_schema_triggers()

    block_patterns: List[Dict[str, Any]] = []
    enriched_records: List[Dict[str, Any]] = []

    processed = 0
    rule_dir = Path(__file__).resolve().parents[1] / "config" / "list_feature_rules"
    list_extractor = EvidenceListFeatureExtractor(str(rule_dir))
    unit_extractor = EvidenceUnitExtractor()

    for rec in _read_jsonl(in_path):
        processed += 1
        if int(args.max_blocks) > 0 and processed > int(args.max_blocks):
            break

        text = str(rec.get("text") or "")
        meta = rec.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}

        abbr_pairs = extract_abbreviation_pairs(text, max_pairs=int(args.max_pairs_per_block))
        entities = extract_entities(text, max_entities=int(args.max_entities_per_block))
        slots = infer_schema_slots_from_text(text)
        block_id = str(rec.get("block_id") or rec.get("id") or rec.get("chunk_id") or "")
        list_features = list_extractor.extract(rec).get("list_features") or {}
        section_type = unit_extractor.infer_section_type(text=text, metadata=meta)
        sentences = unit_extractor.split_sentences(block_id=block_id, text=text)
        evidence_units = unit_extractor.extract_units(
            block_id=block_id,
            text=text,
            section_type=section_type,
            sentences=sentences,
            list_features=list_features if isinstance(list_features, dict) else {},
        )

        # aggregate alias map
        for ap in abbr_pairs:
            if float(ap.confidence) < float(args.min_abbr_confidence):
                continue
            if ap.abbr and ap.full:
                alias_counts[ap.abbr][ap.full] += 1

        # enrich metadata (do not destroy existing keys)
        pat_meta = meta.get("pattern") if isinstance(meta.get("pattern"), dict) else {}
        pat_meta = dict(pat_meta or {})
        if abbr_pairs:
            pat_meta["abbreviation_pairs"] = [
                {"abbr": ap.abbr, "full": ap.full, "confidence": ap.confidence, "source": ap.source} for ap in abbr_pairs
            ]
        if entities:
            pat_meta["entities"] = entities
        if slots:
            pat_meta["schema_slots"] = slots
        if list_features:
            pat_meta["list_features"] = list_features
        meta["pattern"] = pat_meta
        evidence_meta = meta.get("evidence") if isinstance(meta.get("evidence"), dict) else {}
        evidence_meta = dict(evidence_meta or {})
        evidence_meta["section_type"] = section_type
        evidence_meta["sentences"] = sentences
        evidence_meta["evidence_units"] = evidence_units
        meta["evidence"] = evidence_meta

        # add a coarse block_type (do not overwrite if already present)
        if "block_type" not in meta or not str(meta.get("block_type") or "").strip():
            meta["block_type"] = infer_block_type(text=text, metadata=meta)

        rec["metadata"] = meta

        bid = str(rec.get("block_id") or rec.get("id") or rec.get("chunk_id") or "")
        if bid:
            block_patterns.append(
                {
                    "block_id": bid,
                    "pattern": pat_meta,
                    "block_type": meta.get("block_type"),
                    "doc_id": rec.get("doc_id"),
                    "lang": rec.get("lang"),
                    "source_uri": rec.get("source_uri"),
                }
            )

        enriched_records.append(rec)

    # finalize alias_map.json (top expansions per abbr)
    alias_map_out: Dict[str, List[str]] = {}
    alias_debug: Dict[str, Any] = {}
    for abbr, ctr in alias_counts.items():
        tops = [full for full, _ in ctr.most_common(8)]
        if tops:
            alias_map_out[str(abbr)] = tops
            alias_debug[str(abbr)] = dict(ctr.most_common(8))

    (pat_dir / "alias_map.json").write_text(json.dumps(alias_map_out, ensure_ascii=False, indent=2), encoding="utf-8")
    (pat_dir / "schema_triggers.json").write_text(json.dumps(schema_triggers, ensure_ascii=False, indent=2), encoding="utf-8")
    (pat_dir / "fixed_entities.json").write_text(json.dumps(fixed_entities, ensure_ascii=False, indent=2), encoding="utf-8")
    (pat_dir / "alias_map.debug.json").write_text(json.dumps(alias_debug, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_jsonl(pat_dir / "block_patterns.jsonl", block_patterns)
    _write_jsonl(out_path, enriched_records)

    print(
        json.dumps(
            {
                "blocks_in": str(in_path),
                "blocks_out": str(out_path),
                "pattern_out_dir": str(pat_dir),
                "processed_blocks": int(len(enriched_records)),
                "alias_map_size": int(len(alias_map_out)),
                "block_patterns_written": int(len(block_patterns)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

