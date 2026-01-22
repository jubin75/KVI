"""
Semantic-type YAML linter (engineering gate).

This script enforces the constraints in docs/051_约束yaml.md:
- semantic_type must be in an allowlist
- only value-domain operations are allowed (normalize/split/filter, signals/confidence)
- forbid routing/inference/generation keys

It is intentionally conservative: it checks schema/shape + forbidden fields,
not "correctness" of the domain content.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


ALLOWED_SEMANTIC_TYPES: Set[str] = {"location", "symptom", "drug", "date", "other", "generic"}

# High-risk keywords that indicate routing/inference/protocol leakage.
FORBIDDEN_KEYWORDS = [
    "infer",
    "inference",
    "generate",
    "generation",
    "route",
    "routing",
    "promote",
    "policy",
    "gate",
    "decision",
    "pattern",
    "contract",
    "slot",
    "schema",
    "topic",
]


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required to lint semantic-type YAMLs.") from e
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _collect_keys(obj: Any, prefix: str = "") -> List[str]:
    keys: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            k2 = str(k)
            full = f"{prefix}.{k2}" if prefix else k2
            keys.append(full)
            keys.extend(_collect_keys(v, prefix=full))
    elif isinstance(obj, list):
        for it in obj:
            keys.extend(_collect_keys(it, prefix=prefix))
    return keys


def _has_forbidden_keys(payload: Dict[str, Any]) -> List[str]:
    keys = _collect_keys(payload)
    bad: List[str] = []
    for k in keys:
        kl = k.lower()
        for w in FORBIDDEN_KEYWORDS:
            if re.search(rf"(^|[^a-z]){re.escape(w)}([^a-z]|$)", kl):
                # Allow known safe occurrences: the word "schema" appears in documentation but must not appear in YAML.
                bad.append(k)
                break
    return sorted(list(dict.fromkeys(bad)))


def _lint_value_cleaning_yaml(payload: Dict[str, Any], *, path: Path) -> List[str]:
    errors: List[str] = []
    allowed_top = {"semantic_type", "normalize", "split", "filter"}
    for k in payload.keys():
        if str(k) not in allowed_top:
            errors.append(f"forbidden_top_level_key:{k}")

    # semantic_type
    st = str(payload.get("semantic_type") or "").strip().lower()
    if st and st not in ALLOWED_SEMANTIC_TYPES:
        errors.append(f"semantic_type_not_allowed:{st}")

    # normalize
    norm = payload.get("normalize")
    if norm is not None and not isinstance(norm, dict):
        errors.append("normalize_must_be_dict")
    if isinstance(norm, dict):
        # Allowed meta keys for regex-based normalization
        for meta_key in ["regex_subs", "strip_prefixes", "strip_suffixes"]:
            if meta_key in norm and not isinstance(norm.get(meta_key), list):
                errors.append(f"normalize.{meta_key}_must_be_list")
        if isinstance(norm.get("regex_subs"), list):
            for i, it in enumerate(norm.get("regex_subs") or []):
                if not isinstance(it, dict):
                    errors.append(f"normalize.regex_subs[{i}]_must_be_dict")
                    continue
                if "pattern" not in it:
                    errors.append(f"normalize.regex_subs[{i}]_missing_pattern")

    # split
    split = payload.get("split")
    if split is not None and not isinstance(split, dict):
        errors.append("split_must_be_dict")
    if isinstance(split, dict):
        if "connectors" in split and not isinstance(split.get("connectors"), list):
            errors.append("split.connectors_must_be_list")

    # filter
    flt = payload.get("filter")
    if flt is not None and not isinstance(flt, dict):
        errors.append("filter_must_be_dict")
    if isinstance(flt, dict):
        if "deny_terms" in flt and not isinstance(flt.get("deny_terms"), list):
            errors.append("filter.deny_terms_must_be_list")

    # forbidden keywords anywhere
    bad_keys = _has_forbidden_keys(payload)
    if bad_keys:
        errors.append("forbidden_keywords_in_yaml:" + ",".join(bad_keys[:12]) + ("..." if len(bad_keys) > 12 else ""))
    return errors


def _lint_list_feature_yaml(payload: Dict[str, Any], *, path: Path) -> List[str]:
    errors: List[str] = []
    allowed_top = {"semantic_type", "signals", "split", "confidence"}
    for k in payload.keys():
        if str(k) not in allowed_top:
            errors.append(f"forbidden_top_level_key:{k}")

    st = str(payload.get("semantic_type") or "").strip().lower()
    if st and st not in ALLOWED_SEMANTIC_TYPES:
        errors.append(f"semantic_type_not_allowed:{st}")

    sig = payload.get("signals")
    if sig is not None and not isinstance(sig, dict):
        errors.append("signals_must_be_dict")
    if isinstance(sig, dict):
        allowed_sig = {
            "bullets",
            "numbering_regex",
            "trigger_phrases",
            "paren_cases_regex",
            "paren_cases_capture_regex",
        }
        for k in sig.keys():
            if str(k) not in allowed_sig:
                errors.append(f"forbidden_signals_key:{k}")
        for lk in ["bullets", "numbering_regex", "trigger_phrases"]:
            if lk in sig and not isinstance(sig.get(lk), list):
                errors.append(f"signals.{lk}_must_be_list")
        for sk in ["paren_cases_regex", "paren_cases_capture_regex"]:
            if sk in sig and not isinstance(sig.get(sk), str):
                errors.append(f"signals.{sk}_must_be_string")

    split = payload.get("split")
    if split is not None and not isinstance(split, dict):
        errors.append("split_must_be_dict")
    if isinstance(split, dict):
        if "delimiters" in split and not isinstance(split.get("delimiters"), list):
            errors.append("split.delimiters_must_be_list")

    conf = payload.get("confidence")
    if conf is not None and not isinstance(conf, dict):
        errors.append("confidence_must_be_dict")

    bad_keys = _has_forbidden_keys(payload)
    if bad_keys:
        errors.append("forbidden_keywords_in_yaml:" + ",".join(bad_keys[:12]) + ("..." if len(bad_keys) > 12 else ""))
    return errors


def _lint_one(path: Path, *, mode: str) -> Dict[str, Any]:
    payload = _load_yaml(path)
    if mode == "value_cleaning":
        errors = _lint_value_cleaning_yaml(payload, path=path)
    elif mode == "list_feature":
        errors = _lint_list_feature_yaml(payload, path=path)
    else:
        errors = [f"unknown_mode:{mode}"]
    st = str(payload.get("semantic_type") or "").strip().lower() or path.stem
    return {"pass": len(errors) == 0, "semantic_type": st, "file": str(path), "errors": errors}


def main() -> None:
    p = argparse.ArgumentParser(description="Lint semantic-type YAML rules (value_cleaning_rules / list_feature_rules).")
    p.add_argument("--mode", choices=["value_cleaning", "list_feature"], required=True)
    p.add_argument("--path", required=True, help="YAML file path or a directory containing YAMLs")
    args = p.parse_args()

    target = Path(str(args.path))
    files: List[Path] = []
    if target.is_dir():
        files = sorted([p for p in target.glob("*.yaml") if p.is_file()])
    else:
        files = [target]

    results = [_lint_one(f, mode=str(args.mode)) for f in files]
    ok = all(bool(r.get("pass")) for r in results)
    out = {"pass": ok, "mode": str(args.mode), "results": results}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()

