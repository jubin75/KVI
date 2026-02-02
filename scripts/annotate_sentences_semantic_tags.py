"""
Annotate sentence records with semantic intent tags (offline, build-time).

Input: sentences.jsonl (each line is a dict with at least: {block_id, text, metadata:{...}})
Output: sentences.tagged.jsonl (same records + metadata.semantic_scores / metadata.semantic_tags / metadata.semantic_primary)

Design goals:
- No base LLM is used.
- Intent taxonomy is config-driven via semantic_type_specs.json (description + threshold).
- Tags are stored in metadata so they travel with KVBank metas for runtime rerank/filter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            if not isinstance(r, dict):
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _load_specs(path: Path) -> Dict[str, Dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            out: Dict[str, Dict[str, Any]] = {}
            for k, v in obj.items():
                if isinstance(v, dict):
                    out[str(k).strip().lower()] = v
                else:
                    out[str(k).strip().lower()] = {"description": str(v)}
            return {k: v for k, v in out.items() if k}
    except Exception:
        pass
    return {}


def _default_specs() -> Dict[str, Dict[str, Any]]:
    # Keep this aligned with runtime fallback.
    return {
        "symptom": {"description": "临床表现、症状体征、实验室异常、常见表现的枚举或陈述句。", "threshold": 0.28},
        "drug": {"description": "治疗、用药、药物、获批/批准、疗效、不良反应等相关陈述句。", "threshold": 0.28},
        "location": {"description": "地区分布、流行区域、病例报告地点、地理范围等相关陈述句。", "threshold": 0.28},
        "mechanism": {
            "description": "作用机制/发病机制：感染哪些细胞、免疫应答/免疫抑制、炎症反应、病理过程、通透性改变、多器官损伤等。",
            "threshold": 0.26,
        },
    }


def _spec_threshold(spec: Dict[str, Any], default_thr: float) -> float:
    try:
        return float(spec.get("threshold")) if spec.get("threshold") is not None else float(default_thr)
    except Exception:
        return float(default_thr)


def _anchor_text(st: str, desc: str) -> str:
    # Offline tagging should be query-agnostic.
    return f"[semantic_type]\n{st}\n\n[description]\n{desc}\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in_jsonl", required=True, help="sentences.jsonl path")
    p.add_argument("--out_jsonl", required=True, help="sentences.tagged.jsonl path")
    p.add_argument("--domain_encoder_model", required=True, help="HF encoder model (same as retrieval encoder)")
    p.add_argument("--semantic_type_specs", required=False, default="", help="semantic_type_specs.json path")
    p.add_argument("--device", default=None, help="cpu/cuda; default auto")
    args = p.parse_args()

    inp = Path(str(args.in_jsonl))
    outp = Path(str(args.out_jsonl))
    if not inp.exists():
        raise SystemExit(f"input not found: {inp}")

    specs_path = Path(str(args.semantic_type_specs)) if str(args.semantic_type_specs).strip() else None
    specs = _load_specs(specs_path) if (specs_path and specs_path.exists()) else {}
    if not specs:
        specs = _default_specs()

    # Lazy import to keep CLI snappy for non-encoder operations.
    try:
        from external_kv_injection.src.domain_encoder import DomainEncoder, DomainEncoderConfig  # type: ignore
    except ModuleNotFoundError:
        from src.domain_encoder import DomainEncoder, DomainEncoderConfig  # type: ignore
    import torch

    dev = torch.device(str(args.device) if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    enc = DomainEncoder(
        DomainEncoderConfig(
            model_name_or_path=str(args.domain_encoder_model),
            max_length=256,
            normalize=True,
            device=str(dev),
        )
    )

    keys = list(specs.keys())
    anchors: List[str] = []
    thresholds: Dict[str, float] = {}
    for k in keys:
        desc = str((specs.get(k) or {}).get("description") or "").strip() or k
        anchors.append(_anchor_text(k, desc))
        thresholds[k] = _spec_threshold(specs.get(k) or {}, 0.28)

    # Pre-embed anchors once.
    a = enc.encode(anchors, batch_size=min(16, max(1, len(anchors))))

    recs = _read_jsonl(inp)
    annotated = 0
    for r in recs:
        text = str(r.get("text") or "").strip()
        if not text:
            continue
        meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
        # Embed sentence and compute dot product sims
        v = enc.encode([text], batch_size=1)[0]
        # v shape [d], a shape [K,d]
        sims = (a @ v).astype(float)
        scores: Dict[str, float] = {k: float(sims[i]) for i, k in enumerate(keys)}
        # Tags: keep >= threshold; if empty, keep top1.
        order = sorted(keys, key=lambda k: scores.get(k, 0.0), reverse=True)
        kept = [k for k in order if scores.get(k, 0.0) >= float(thresholds.get(k, 0.28))]
        if not kept and order:
            kept = [order[0]]
        primary = kept[0] if kept else (order[0] if order else "unknown")
        meta["semantic_scores"] = scores
        meta["semantic_tags"] = kept
        meta["semantic_primary"] = primary
        r["metadata"] = meta
        annotated += 1

    _write_jsonl(outp, recs)
    print(
        json.dumps(
            {
                "ok": True,
                "in": str(inp),
                "out": str(outp),
                "annotated": int(annotated),
                "types": keys,
                "device": str(dev),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

