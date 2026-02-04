"""
Minimal KVI2Runtime.run_ab test harness.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore


def _load_semantic_type_specs(*, pattern_index_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Scheme B (productized): semantic relevance is configured by lightweight "type descriptions"
    and filtered by embedding similarity (no hard-coded semantic types).
    """
    pdir = Path(str(pattern_index_dir or "").strip())
    # Default location under work_dir/pattern_sidecar
    cand = pdir / "semantic_type_specs.json"
    if cand.exists():
        try:
            obj = json.loads(cand.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                return {str(k): (v if isinstance(v, dict) else {"description": str(v)}) for k, v in obj.items()}
        except Exception:
            pass
    # Built-in defaults (generic, works across medical topics).
    return {
        "symptom": {
            "description": "临床表现、症状体征、实验室异常、常见表现的枚举或陈述句。",
            "threshold": 0.28,
        },
        "drug": {
            "description": "治疗、用药、药物、获批/批准、疗效、不良反应等相关陈述句。",
            "threshold": 0.28,
        },
        "location": {
            "description": "地区分布、流行区域、病例报告地点、地理范围等相关陈述句。",
            "threshold": 0.28,
        },
        "mechanism": {
            "description": "作用机制/发病机制：感染哪些细胞、免疫应答/免疫抑制、炎症反应、病理过程、通透性改变、多器官损伤等。",
            "threshold": 0.26,
        },
    }


def _load_semantic_type_specs_any(*, pattern_index_dir: str, semantic_type_specs_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load semantic type specs from an explicit path first (work_dir/semantic_type_specs.json),
    then fall back to pattern_index_dir/semantic_type_specs.json, then built-in defaults.
    """
    sp = str(semantic_type_specs_path or "").strip()
    if sp:
        p = Path(sp)
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    return {str(k): (v if isinstance(v, dict) else {"description": str(v)}) for k, v in obj.items()}
            except Exception:
                pass
    return _load_semantic_type_specs(pattern_index_dir=str(pattern_index_dir or ""))


def _infer_intent_from_specs(*, enc: Any, query: str, specs: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """
    Config-driven intent inference (no hard-coded intent types).
    Use embedding similarity between query and each type anchor (description + query).
    """
    q = str(query or "").strip()
    if not q:
        return "unknown", {"method": "specs_embedding", "reason": "empty_query"}
    if not isinstance(specs, dict) or not specs:
        return "unknown", {"method": "specs_embedding", "reason": "empty_specs"}

    keys = [str(k).strip().lower() for k in specs.keys() if str(k).strip()]
    keys = [k for k in keys if k]
    if not keys:
        return "unknown", {"method": "specs_embedding", "reason": "no_keys"}

    anchors: List[str] = []
    for k in keys:
        spec = specs.get(k) if isinstance(specs.get(k), dict) else {}
        desc = str((spec or {}).get("description") or (spec or {}).get("desc") or "").strip() or k
        anchors.append(f"[semantic_type]\n{k}\n\n[description]\n{desc}\n\n[query]\n{q}\n")

    try:
        import numpy as np  # type: ignore

        a = enc.encode(anchors, batch_size=min(16, max(1, len(anchors))))
        qv = enc.encode([q], batch_size=1)[0]
        sims = (a @ qv).astype(float)
        order = list(np.argsort(-sims))
        best = keys[order[0]] if order else "unknown"
        dbg = {
            "method": "specs_embedding",
            "best": best,
            "top": [{"type": keys[i], "sim": float(sims[i])} for i in order[: min(5, len(order))]],
        }
        return best, dbg
    except Exception as e:
        return "unknown", {"method": "specs_embedding", "error": f"{type(e).__name__}"}

def _semantic_filter_units_by_embedding(
    *,
    enc: Any,
    units: List[str],
    query: str,
    semantic_type: str,
    specs: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Embedding-based semantic relevance filter (Scheme B).
    - anchor_text = (semantic_type description + query)
    - keep units above threshold; if none, keep a small top slice as a safe fallback (avoid empty).
    Returns (filtered_units, debug_info).
    """
    st = str(semantic_type or "").strip().lower()
    q = str(query or "").strip()
    us = [str(u or "").strip() for u in (units or []) if str(u or "").strip()]
    if not st or not us:
        return us, {"method": "embedding", "semantic_type": st, "kept": len(us), "reason": "empty_input"}

    spec = specs.get(st) if isinstance(specs, dict) else None
    desc = ""
    thr = 0.28
    if isinstance(spec, dict):
        desc = str(spec.get("description") or spec.get("desc") or "").strip()
        try:
            thr = float(spec.get("threshold")) if spec.get("threshold") is not None else thr
        except Exception:
            thr = thr
    if not desc:
        # Unknown semantic types: use the label itself as a weak description (no code change needed).
        desc = st
        thr = 0.24

    anchor_text = f"[semantic_type]\n{st}\n\n[description]\n{desc}\n\n[query]\n{q}\n"
    try:
        import numpy as np  # type: ignore

        a = enc.encode([anchor_text], batch_size=1)[0]
        m = enc.encode(us, batch_size=min(16, max(1, len(us))))
        sims = (m @ a).astype(float)  # cosine sim if encoder normalized
        order = list(np.argsort(-sims))
        kept = [us[i] for i in order if float(sims[i]) >= float(thr)]
        # Avoid fail-closed: keep a tiny top slice if everything is below threshold.
        if not kept:
            topk = min(2, len(us))
            kept = [us[i] for i in order[:topk]]
        dbg = {
            "method": "embedding",
            "semantic_type": st,
            "threshold": float(thr),
            "units_in": len(us),
            "units_kept": len(kept),
            "top_sims": [float(sims[i]) for i in order[: min(5, len(order))]],
        }
        return kept, dbg
    except Exception as e:
        # If encoder fails, fall back to keeping units (do not drop to empty).
        return us, {"method": "embedding", "semantic_type": st, "kept": len(us), "fallback": f"{type(e).__name__}"}


def _rank_units_by_anchor_similarity(
    *,
    enc: Any,
    units: List[str],
    query: str,
    semantic_type: str,
    specs: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Rank units by similarity to the semantic anchor (description + query).
    Returns (sorted_units, debug).
    """
    st = str(semantic_type or "").strip().lower()
    us = [str(u or "").strip() for u in (units or []) if str(u or "").strip()]
    if not us:
        return [], {"method": "embedding_rank", "semantic_type": st, "reason": "no_units"}
    spec = specs.get(st) if isinstance(specs, dict) else None
    desc = str((spec or {}).get("description") or "").strip() if isinstance(spec, dict) else ""
    if not desc:
        desc = st or "generic"
    anchor_text = f"[semantic_type]\n{st}\n\n[description]\n{desc}\n\n[query]\n{str(query or '').strip()}\n"
    try:
        import numpy as np  # type: ignore

        a = enc.encode([anchor_text], batch_size=1)[0]
        m = enc.encode(us, batch_size=min(16, max(1, len(us))))
        sims = (m @ a).astype(float)
        order = list(np.argsort(-sims))
        ranked = [us[i] for i in order]
        return ranked, {"method": "embedding_rank", "semantic_type": st, "top_sims": [float(sims[i]) for i in order[: min(5, len(order))]]}
    except Exception as e:
        return us, {"method": "embedding_rank", "semantic_type": st, "fallback": f"{type(e).__name__}"}
def _load_block_text_lookup(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            bid = rec.get("block_id") or (rec.get("metadata") or {}).get("block_id")
            if bid:
                out[str(bid)] = str(rec.get("text") or "")
    return out


def _load_sentence_text_lookup(path: str) -> Dict[str, str]:
    """
    Load sentence texts (block_id -> text) from sentences.jsonl emitted by the UI compile step.
    """
    out: Dict[str, str] = {}
    sp = str(path or "").strip()
    if not sp:
        return out
    p = Path(sp)
    if not p.exists():
        return out
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                bid = rec.get("block_id") or (rec.get("metadata") or {}).get("block_id")
                if bid:
                    out[str(bid)] = str(rec.get("text") or "")
    except Exception:
        return out
    return out


_ABBR_PARENS_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_-]{1,})[（(]([^）)]{1,64})[）)]")
_ALIAS_PARENS_ABBR_RE = re.compile(r"([^\s]{2,64})[（(]([A-Za-z][A-Za-z0-9_-]{1,})[）)]")


def _detect_violations(
    *,
    answer: str,
    user_prompt: str,
    evidence_texts: Sequence[str],
    intent: str,
    specs: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Lightweight validator for "regen-on-violation".
    Keep it conservative: only flag clear violations.
    """
    a = str(answer or "").strip()
    q = str(user_prompt or "").strip()
    ev = "\n".join([str(x or "") for x in (evidence_texts or [])])
    if not a:
        return [{"type": "empty_answer"}]

    violations: List[Dict[str, Any]] = []

    # 1) Abbreviation expansion must appear verbatim in evidence.
    for m in _ABBR_PARENS_RE.finditer(a):
        abbr = m.group(1)
        inner = m.group(2)
        full_zh = f"{abbr}（{inner}）"
        full_en = f"{abbr}({inner})"
        if full_zh not in ev and full_en not in ev:
            violations.append({"type": "abbr_expansion_not_in_evidence", "span": m.group(0), "abbr": abbr})

    # 1b) Alias-before-(ABBR) must also appear verbatim in evidence (e.g., "某某病毒（SFTSV）").
    for m in _ALIAS_PARENS_ABBR_RE.finditer(a):
        alias = m.group(1)
        abbr = m.group(2)
        # only treat as "alias expansion" if alias contains any CJK character (avoid punishing normal English parentheses too much)
        if not re.search(r"[\u4e00-\u9fff]", alias):
            continue
        full_zh = f"{alias}（{abbr}）"
        full_en = f"{alias}({abbr})"
        if full_zh not in ev and full_en not in ev:
            violations.append({"type": "alias_with_abbr_not_in_evidence", "span": m.group(0), "abbr": abbr, "alias": alias})

    # 2) Config-driven deny terms: if answer contains deny terms for the inferred intent, flag.
    it = str(intent or "").strip().lower()
    spec = specs.get(it) if isinstance(specs, dict) and it else None
    deny_terms = (spec or {}).get("deny_terms") if isinstance(spec, dict) else None
    if isinstance(deny_terms, list) and deny_terms:
        leaked: List[str] = []
        for t in deny_terms:
            tt = str(t or "").strip()
            if not tt:
                continue
            # allow if user explicitly asked about it
            if tt in q:
                continue
            if tt in a:
                leaked.append(tt)
        if leaked:
            violations.append({"type": "intent_drift_deny_terms", "intent": it, "terms": leaked})

    # 3) Symptom strictness (validator-only): if answer enumerates symptom items not present in injected evidence,
    # flag so regen can rewrite. This does NOT feed evidence text into the prompt; it only checks coverage.
    def _extract_enum_items(s: str) -> List[str]:
        out: List[str] = []
        if not s:
            return out
        # capture common enumeration contexts
        patterns = [
            r"(?:包括|常见(?:症状)?包括|表现为|可出现|可能出现|可见|伴有|起病(?:为)?)[：:]*\s*([^。！？\n]{2,160})",
        ]
        for pat in patterns:
            for m in re.finditer(pat, s):
                seg = m.group(1)
                # cut off trailing clauses after "此外/另外/并" to keep list-like part
                seg = re.split(r"(?:此外|另外|并且|且|其中|同时|在严重情况下|严重情况下)", seg)[0]
                # normalize separators
                seg = seg.replace("以及", "、").replace("及", "、").replace("和", "、")
                parts = [p.strip() for p in re.split(r"[、,，;；]", seg) if p.strip()]
                for p0 in parts:
                    # remove common modifiers
                    p1 = re.sub(r"^(?:多为|常为|主要为|通常为|多见|常见)", "", p0).strip()
                    p1 = re.sub(r"(?:等|等症状|等表现)$", "", p1).strip()
                    # keep short-to-medium noun phrases
                    if 1 < len(p1) <= 24:
                        out.append(p1)
        # de-dup
        seen: set[str] = set()
        dedup: List[str] = []
        for x in out:
            if x in seen:
                continue
            seen.add(x)
            dedup.append(x)
        return dedup

    if it == "symptom":
        ev_join = "\n".join([str(x or "") for x in (evidence_texts or [])])
        items = _extract_enum_items(a)
        if items and ev_join:
            missing = []
            for x in items:
                if x not in ev_join:
                    missing.append(x)
            # Only flag when we are clearly enumerating and at least one item is missing.
            if missing:
                violations.append(
                    {
                        "type": "answer_items_not_in_injected_evidence",
                        "intent": "symptom",
                        "items_missing": missing[:20],
                        "items_total": len(items),
                    }
                )

    return violations


def _strip_unapproved_abbr_expansions(*, text: str, evidence_texts: Sequence[str]) -> str:
    """
    Enforce "缩写规则" post-hoc: only allow acronym expansions if the exact form appears in
    injected evidence texts. Otherwise strip the parenthetical part.
    """
    t = str(text or "")
    ev = [str(x or "") for x in (evidence_texts or [])]
    if not t or not ev:
        # If we don't have evidence texts, still be conservative: strip expansions to reduce hallucinated glosses.
        # This matches the product rule: do not add bracket expansions unless evidence shows it.
        ev = []

    def repl(m: re.Match) -> str:
        abbr = m.group(1)
        inner = m.group(2)
        full_zh = f"{abbr}（{inner}）"
        full_en = f"{abbr}({inner})"
        if ev:
            for e in ev:
                if full_zh in e or full_en in e:
                    return m.group(0)  # keep
        # strip expansion
        return abbr

    # First, strip alias-before-(ABBR) patterns that are not present in evidence, e.g. "宋热...病毒（SFTSV）" -> "SFTSV"
    def repl_alias(m: re.Match) -> str:
        alias = m.group(1)
        abbr = m.group(2)
        if not re.search(r"[\u4e00-\u9fff]", alias):
            return m.group(0)
        full_zh = f"{alias}（{abbr}）"
        full_en = f"{alias}({abbr})"
        if ev:
            for e in ev:
                if full_zh in e or full_en in e:
                    return m.group(0)
        return abbr

    t2 = _ALIAS_PARENS_ABBR_RE.sub(repl_alias, t)
    return _ABBR_PARENS_RE.sub(repl, t2)


def _run_evidence_routing(
    *,
    prompt: str,
    kv_dir: str,
    sentences_lookup: Dict[str, str],
    top_k: int,
    domain_encoder_model: str,
    semantic_type_specs_path: str,
    pattern_index_dir: str,
    w_ann: float,
    w_intent: float,
    w_quality: float,
) -> Dict[str, Any]:
    """
    Evidence routing (no generation). Returns evidence_projection + ids/texts.
    """
    try:
        from external_kv_injection.src.domain_encoder import DomainEncoder, DomainEncoderConfig  # type: ignore
        from external_kv_injection.src.kv_bank import FaissKVBank  # type: ignore
        from external_kv_injection.src.retriever import Retriever  # type: ignore
    except ModuleNotFoundError:
        from src.domain_encoder import DomainEncoder, DomainEncoderConfig  # type: ignore
        from src.kv_bank import FaissKVBank  # type: ignore
        from src.retriever import Retriever  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = DomainEncoder(
        DomainEncoderConfig(
            model_name_or_path=str(domain_encoder_model),
            max_length=256,
            normalize=True,
            device=str(device),
        )
    )
    bank = FaissKVBank.load(Path(str(kv_dir)))
    retriever = Retriever(bank)

    q = str(prompt or "").strip()
    qv = enc.encode(q)[0]
    # Stage 1: broader ANN pool
    top_pool = max(int(top_k) * 10, int(top_k))
    rr = retriever.search(qv, top_k=int(top_pool), filters=None, query_text=q)
    items = []
    # Intent inference from specs (config-driven)
    specs = _load_semantic_type_specs_any(
        pattern_index_dir=str(pattern_index_dir or ""),
        semantic_type_specs_path=str(semantic_type_specs_path or ""),
    )
    intent, intent_dbg = _infer_intent_from_specs(enc=enc, query=q, specs=specs)
    intent_key = str(intent or "").strip().lower()
    intent_spec = specs.get(intent_key) if isinstance(specs, dict) else None
    intent_desc = str((intent_spec or {}).get("description") or "").strip() if isinstance(intent_spec, dict) else intent_key
    anchor_text = f"[semantic_type]\n{intent_key}\n\n[description]\n{intent_desc}\n\n[query]\n{q}\n"
    anchor_vec = enc.encode([anchor_text], batch_size=1)[0]
    # Batch encode candidate texts for fallback intent sim
    cand_texts = [str(sentences_lookup.get(str(_kv_id(it)), "") or "") for it in (rr.items or [])]
    if cand_texts:
        cand_vecs = enc.encode(cand_texts, batch_size=min(16, max(1, len(cand_texts))))
    else:
        cand_vecs = []

    def _quality_score(text: str) -> float:
        t = str(text or "")
        if not t:
            return 0.0
        if "http://" in t or "https://" in t or "www." in t:
            return -0.6
        # penalize extremely short or very long sentences
        if len(t) < 8:
            return -0.4
        if len(t) > 240:
            return -0.2
        return 0.2

    # weights are provided by caller (UI-configurable)
    w_ann = float(w_ann)
    w_intent = float(w_intent)
    w_quality = float(w_quality)
    for i, it in enumerate(rr.items or []):
        bid = _kv_id(it)
        if not bid:
            continue
        meta = getattr(it, "meta", None) or {}
        payload = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
        text = str(sentences_lookup.get(str(bid), "") or "")
        ann_score = float(getattr(it, "score", 0.0) or 0.0)
        # intent sim: prefer precomputed semantic_scores if present
        sem_scores = payload.get("semantic_scores") if isinstance(payload.get("semantic_scores"), dict) else {}
        intent_sim = None
        if intent_key and isinstance(sem_scores.get(intent_key), (int, float)):
            intent_sim = float(sem_scores.get(intent_key))
        else:
            try:
                if i < len(cand_vecs):
                    intent_sim = float((cand_vecs[i] @ anchor_vec).astype(float))
            except Exception:
                intent_sim = 0.0
        if intent_sim is None:
            intent_sim = 0.0
        quality = _quality_score(text)
        final_score = (w_ann * ann_score) + (w_intent * intent_sim) + (w_quality * quality)
        items.append(
            {
                "id": str(bid),
                "score": float(ann_score),
                "intent_sim": float(intent_sim),
                "quality": float(quality),
                "final_score": float(final_score),
                "text": text,
                "semantic_tags": payload.get("semantic_tags") if isinstance(payload.get("semantic_tags"), list) else [],
                "semantic_primary": payload.get("semantic_primary"),
            }
        )
    # Stage 3: rerank and truncate
    # Strategy A (semi-hard):
    # 1) prefer items with semantic_primary == intent
    # 2) fallback to items whose semantic_tags contains intent
    # 3) fallback to all items
    primary_filtered = []
    if intent_key:
        for it in items:
            sp = str(it.get("semantic_primary") or "").strip().lower()
            if sp and sp == intent_key:
                primary_filtered.append(it)
    tag_filtered = []
    if not primary_filtered and intent_key:
        for it in items:
            tags = it.get("semantic_tags") if isinstance(it.get("semantic_tags"), list) else []
            if any(str(t).strip().lower() == intent_key for t in tags):
                tag_filtered.append(it)
    if primary_filtered:
        use_items = primary_filtered
        soft_filter_used = "primary"
    elif tag_filtered:
        use_items = tag_filtered
        soft_filter_used = "tags"
    else:
        use_items = items
        soft_filter_used = "none"
    use_items.sort(key=lambda x: (x.get("final_score", 0.0), x.get("score", 0.0)), reverse=True)
    use_items = use_items[: int(top_k)]
    evidence_ids = [x.get("id") for x in use_items]
    evidence_texts = [x.get("text") for x in use_items]
    status = "OK" if use_items else "NO_EVIDENCE_FOUND"
    return {
        "status": status,
        "routing_debug": {
            "intent": intent_key or "unknown",
            "intent_debug": intent_dbg,
            "weights": {"ann": w_ann, "intent": w_intent, "quality": w_quality},
            "pool_size": int(top_pool),
            "soft_filter_intent": intent_key or "",
            "soft_filter_used": soft_filter_used,
        },
        "evidence_projection": use_items,
        "evidence_ids": evidence_ids,
        "evidence_texts": evidence_texts,
    }
def _kv_id(it: Any) -> str:
    meta = getattr(it, "meta", None) or {}
    return str(meta.get("block_id") or meta.get("chunk_id") or meta.get("id") or "")


def _load_block_metadata_lookup(path: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            bid = rec.get("block_id") or (rec.get("metadata") or {}).get("block_id")
            if not bid:
                continue
            meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
            out[str(bid)] = meta
    return out


def _extract_sentence_units_only(
    *, extractor: Any, block_id: str, text: str, metadata: Dict[str, Any]
) -> List[str]:
    """
    Evidence Unit Pipeline (sentence-level only; no list-item fallback).
    In simple pipeline, keep:
    - sentence_enumerative units (injectable)
    - OR "sentence fragments" that look like self-contained factual sentences (soft evidence units),
      so mechanism/pathogenesis claims can still be grounded.
    """
    bid = str(block_id or "").strip()
    t = str(text or "")
    meta = metadata if isinstance(metadata, dict) else {}
    try:
        section_type = extractor.infer_section_type(text=t, metadata=meta)
        sentences = extractor.split_sentences(block_id=bid, text=t)
        units = extractor.extract_units(
            block_id=bid,
            text=t,
            section_type=section_type,
            sentences=sentences,
            list_features={},  # do NOT enable list-item fallback in simple mode
        )
    except Exception:
        return []
    out: List[str] = []
    for u in units or []:
        if not isinstance(u, dict):
            continue
        inj = u.get("injectability") if isinstance(u.get("injectability"), dict) else {}
        s = str(u.get("text") or "").strip()
        if not s:
            continue
        ut = str(u.get("unit_type") or "").strip()
        # Primary: enumerative facts (best quality)
        if ut == "sentence_enumerative" and bool(inj.get("allowed")):
            out.append(s)
            continue
        # Secondary: allow well-formed factual sentence fragments (mechanism/pathogenesis, definitions, etc.)
        # Skip if explicitly blocked by section type / metadata.
        br = inj.get("blocking_reasons") if isinstance(inj.get("blocking_reasons"), list) else []
        br = [str(x) for x in br if str(x)]
        if any(b.startswith("section_excluded:") for b in br):
            continue
        if any(b in {"procedural_or_metadata"} for b in br):
            continue
        # Heuristic: keep only sentence-like strings (end punctuation) and medium length.
        if len(s) < 12 or len(s) > 360:
            continue
        if not re.search(r"[\.\!\?\。\！\？]\s*$", s):
            continue
        out.append(s)
    # dedupe keep order
    seen: set[str] = set()
    dedup: List[str] = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        dedup.append(s)
    return dedup


def _infer_target_semantic_type_for_query(
    *,
    query: str,
    kv_dir: str,
    pattern_index_dir: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Scheme #1: infer semantic_type WITHOUT using base LLM.
    Prefer: pattern contracts -> schema scoring -> slot.semantic_type.
    Fallback: conservative keyword router (fail-closed to "unknown").
    """
    q = str(query or "").strip()
    ql = q.lower()
    dbg: Dict[str, Any] = {"method": "", "pattern_id": "", "semantic_type": "", "rationale": []}

    # Try contract-driven path first.
    try:
        try:
            from external_kv_injection.src.pattern_pipeline import (  # type: ignore
                PatternContractLoader,
                PatternMatcher,
                SlotSchema,
                score_candidate_schemas,
            )
            from external_kv_injection.src.pattern_retriever import PatternRetriever  # type: ignore
        except ModuleNotFoundError:
            from src.pattern_pipeline import PatternContractLoader, PatternMatcher, SlotSchema, score_candidate_schemas  # type: ignore
            from src.pattern_retriever import PatternRetriever  # type: ignore

        loader = PatternContractLoader()
        topic_dir = loader.infer_topic_dir_from_kv_dir(str(kv_dir))
        contracts = loader.load(topic_dir=str(topic_dir) if topic_dir else None)
        if contracts:
            retr = (
                PatternRetriever.from_dir(str(pattern_index_dir))
                if str(pattern_index_dir or "").strip()
                else PatternRetriever()
            )
            matcher = PatternMatcher(contracts, retriever=retr)
            _res, matched_patterns, _matched_skeletons = matcher.match(q)
            if matched_patterns:
                scored = score_candidate_schemas(q, matched_patterns)
                best_id = str(scored[0].get("schema_id") or "").strip() if scored else str(matched_patterns[0].pattern_id)
                best = None
                for p in matched_patterns:
                    if str(p.pattern_id) == best_id:
                        best = p
                        break
                best = best or matched_patterns[0]
                slot_schema = SlotSchema.from_pattern(best)
                stypes = {
                    str(spec.semantic_type or "").strip().lower()
                    for spec in (slot_schema.slots or {}).values()
                    if str(spec.semantic_type or "").strip()
                }
                stypes = {s for s in stypes if s not in {"generic", "other"}}
                if len(stypes) == 1:
                    st = list(stypes)[0]
                    dbg.update(
                        {
                            "method": "contracts+scoring",
                            "pattern_id": str(best.pattern_id),
                            "semantic_type": st,
                            "rationale": list(scored[0].get("rationale") or []) if scored else [],
                        }
                    )
                    return st, dbg
                # IMPORTANT: do NOT fail-closed here. Topic-level contracts can be incomplete
                # (e.g., mini evidence topics), or schema slots may remain "generic".
                # Fall back to the conservative heuristic router below so Evidence Units can
                # still be filtered by a stable semantic_type without using base LLM.
                dbg.update(
                    {
                        "method": "contracts+scoring",
                        "pattern_id": str(best.pattern_id),
                        "semantic_type": "unknown",
                        "rationale": ["ambiguous_semantic_type"],
                    }
                )
    except Exception:
        pass

    # Fallback heuristic router (language-agnostic minimal cues).
    symptom = bool(
        ("临床表现" in q) or ("症状" in q) or ("表现" in q) or ("manifestation" in ql) or ("symptom" in ql)
    )
    drug = bool(
        ("治疗" in q) or ("药物" in q) or ("用药" in q) or ("批准" in q) or ("获批" in q) or ("fda" in ql) or ("drug" in ql) or ("treat" in ql)
    )
    location = bool(
        ("地区" in q)
        or ("哪里" in q)
        or ("哪些地方" in q)
        or ("分布" in q)
        or ("省" in q)
        or ("市" in q)
        or ("where" in ql)
        or ("distribution" in ql)
        or ("reported in" in ql)
    )
    intents = {
        "symptom": bool(symptom),
        "drug": bool(drug),
        "location": bool(location),
    }
    # Mechanism/pathogenesis is common for scientific topics; treat it as its own intent.
    mechanism = bool(
        ("机制" in q)
        or ("作用机制" in q)
        or ("发病机制" in q)
        or ("致病机制" in q)
        or ("pathogenesis" in ql)
        or ("mechanism" in ql)
    )
    intents["mechanism"] = bool(mechanism)
    active = [k for k, v in intents.items() if v]
    if len(active) != 1:
        # Multi-intent is common in Chinese: "哪些地区？主要症状有哪些？"
        # In simple pipeline, we prefer a safe UNION filter (deletion-only) instead of fail-closed to empty.
        if active:
            if dbg.get("method") == "contracts+scoring":
                dbg["method"] = "contracts+scoring+heuristic"
                dbg["semantic_type"] = "multi"
                dbg["semantic_types"] = active
                dbg["rationale"] = list(dbg.get("rationale") or []) + ["heuristic:multi_intent"]
                return "multi", dbg
            dbg.update({"method": "heuristic", "semantic_type": "multi", "semantic_types": active, "rationale": ["heuristic:multi_intent"]})
            return "multi", dbg
        # No intent cues
        if dbg.get("method") == "contracts+scoring":
            dbg["method"] = "contracts+scoring+heuristic"
            dbg["rationale"] = list(dbg.get("rationale") or []) + ["heuristic:ambiguous_or_no_intent"]
            dbg["semantic_type"] = "unknown"
            return "unknown", dbg
        dbg.update({"method": "heuristic", "semantic_type": "unknown", "rationale": ["ambiguous_or_no_intent"]})
        return "unknown", dbg
    if symptom:
        if dbg.get("method") == "contracts+scoring":
            dbg["method"] = "contracts+scoring+heuristic"
            dbg["semantic_type"] = "symptom"
            dbg["rationale"] = list(dbg.get("rationale") or []) + ["cue:symptom"]
            return "symptom", dbg
        dbg.update({"method": "heuristic", "semantic_type": "symptom", "rationale": ["cue:symptom"]})
        return "symptom", dbg
    if drug:
        if dbg.get("method") == "contracts+scoring":
            dbg["method"] = "contracts+scoring+heuristic"
            dbg["semantic_type"] = "drug"
            dbg["rationale"] = list(dbg.get("rationale") or []) + ["cue:drug"]
            return "drug", dbg
        dbg.update({"method": "heuristic", "semantic_type": "drug", "rationale": ["cue:drug"]})
        return "drug", dbg
    if mechanism:
        if dbg.get("method") == "contracts+scoring":
            dbg["method"] = "contracts+scoring+heuristic"
            dbg["semantic_type"] = "mechanism"
            dbg["rationale"] = list(dbg.get("rationale") or []) + ["cue:mechanism"]
            return "mechanism", dbg
        dbg.update({"method": "heuristic", "semantic_type": "mechanism", "rationale": ["cue:mechanism"]})
        return "mechanism", dbg
    if dbg.get("method") == "contracts+scoring":
        dbg["method"] = "contracts+scoring+heuristic"
        dbg["semantic_type"] = "location"
        dbg["rationale"] = list(dbg.get("rationale") or []) + ["cue:location"]
        return "location", dbg
    dbg.update({"method": "heuristic", "semantic_type": "location", "rationale": ["cue:location"]})
    return "location", dbg


def _unit_relevant_to_any_semantic_type(*, unit_text: str, semantic_types: List[str], query: str) -> bool:
    sts = [str(s or "").strip().lower() for s in (semantic_types or [])]
    sts = [s for s in sts if s and s not in {"unknown", "generic", "other"}]
    if not sts:
        return False
    return any(_unit_relevant_to_semantic_type(unit_text=unit_text, semantic_type=st, query=query) for st in sts)


def _format_evidence_units_with_ids(evidence_units: List[str]) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Returns (formatted_text, pairs) where pairs = [(E1, text), ...]
    """
    pairs: List[Tuple[str, str]] = []
    for i, s in enumerate(evidence_units or [], start=1):
        t = str(s or "").strip()
        if not t:
            continue
        pairs.append((f"E{i}", t))
    formatted = "\n".join([f"[{eid}] {txt}" for eid, txt in pairs])
    return formatted, pairs


_SIMPLE_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?\。\！\？])\s*|\n+")
_SIMPLE_CITE_RE = re.compile(r"\[E\d+\]")


def _postprocess_answer_citation_guardrails(
    *,
    answer: str,
    evidence_pairs: List[Tuple[str, str]],
) -> Tuple[str, Dict[str, Any]]:
    """
    Deletion-only guardrails (simple pipeline):
    - If a sentence looks like it asserts concrete facts but has no [E#] citation, drop it.
    - If it contains an SFTSV parenthetical expansion not present verbatim in evidence, drop the whole sentence.
    """
    a = str(answer or "").strip()
    evidence_blob = "\n".join([t for _eid, t in (evidence_pairs or [])])
    dropped: List[Dict[str, Any]] = []
    auto_cited: List[Dict[str, Any]] = []

    def _looks_like_fact(s: str) -> bool:
        sl = s.lower()
        # "Facty" cues: lists, numbers, places, symptoms, explicit definition/alias, institutions.
        if re.search(r"\d", s):
            return True
        if any(k in s for k in ["河南", "湖北", "安徽", "山东", "浙江", "江苏", "辽宁", "吉林", "省", "市", "地区", "分布", "交界处"]):
            return True
        if any(k in s for k in ["临床表现", "症状", "包括", "表现为", "常见", "发热", "血小板", "白细胞", "胃肠道", "神经系统", "出血"]):
            return True
        if any(k in s for k in ["SFTSV（", "SFTSV(", "简称", "全称", "也称", "又称"]):
            return True
        if any(k in sl for k in ["cdc", "fda", "who"]) or any(k in s for k in ["疾控", "疾病预防控制中心"]):
            return True
        return False

    def _has_cite(s: str) -> bool:
        return bool(_SIMPLE_CITE_RE.search(s))

    def _strip_cites(s: str) -> str:
        return _SIMPLE_CITE_RE.sub("", s).strip()

    def _tokenize_shingles(s: str) -> List[str]:
        """
        Very lightweight tokenization for overlap scoring:
        - English words / numbers as tokens
        - Chinese text as 2-char shingles (to avoid single-char noise)
        """
        t = str(s or "")
        # keep alnum tokens
        en = re.findall(r"[A-Za-z0-9]+", t)
        # Chinese characters only
        zh = "".join(re.findall(r"[\u4e00-\u9fff]+", t))
        z2 = [zh[i : i + 2] for i in range(0, max(0, len(zh) - 1))]
        out = [x.lower() for x in en] + z2
        # drop very short shingles
        return [x for x in out if len(x) >= 2]

    def _best_evidence_cites_for_sentence(s: str) -> List[str]:
        """
        Return a small list like ["[E2]", "[E3]"] if the sentence matches evidence well.
        """
        stoks = set(_tokenize_shingles(_strip_cites(s)))
        if not stoks:
            return []
        scored: List[Tuple[float, str]] = []
        for eid, etxt in (evidence_pairs or []):
            etoks = set(_tokenize_shingles(etxt))
            if not etoks:
                continue
            inter = len(stoks & etoks)
            # fraction of sentence tokens covered by evidence tokens
            frac = float(inter) / float(max(1, len(stoks)))
            if inter >= 3 and frac >= 0.22:
                scored.append((frac, f"[{eid}]"))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _f, c in scored[:2]]

    def _has_unsupported_sftsv_expansion(s: str) -> bool:
        # If model expands SFTSV in parentheses, require verbatim presence in evidence units.
        m = re.search(r"SFTSV[（(]([^）)]+)[）)]", s)
        if not m:
            return False
        expanded = str(m.group(0) or "").strip()
        if not expanded:
            return False
        return expanded not in evidence_blob

    # Split and filter sentence-by-sentence.
    parts = [p.strip() for p in _SIMPLE_SENT_SPLIT_RE.split(a) if p and p.strip()]
    kept: List[str] = []
    for s in parts:
        # Drop citation-only fragments
        if not _strip_cites(s):
            dropped.append({"sentence": s, "reason": "citation_only_fragment"})
            continue
        if _has_unsupported_sftsv_expansion(s):
            dropped.append({"sentence": s, "reason": "unsupported_sftsv_parenthetical"})
            continue
        if _looks_like_fact(s) and not _has_cite(s):
            cites = _best_evidence_cites_for_sentence(s)
            if cites:
                s2 = (s.rstrip() + " " + "".join(cites)).strip()
                auto_cited.append({"before": s, "after": s2, "cites": cites})
                kept.append(s2)
                continue
            dropped.append({"sentence": s, "reason": "fact_without_citation"})
            continue
        kept.append(s)
    out = "\n".join(kept).strip()
    dbg = {
        "dropped_count": len(dropped),
        "dropped_examples": dropped[:3],
        "auto_cited_count": len(auto_cited),
        "auto_cited_examples": auto_cited[:3],
    }
    return out, dbg


def _unit_relevant_to_semantic_type(*, unit_text: str, semantic_type: str, query: str) -> bool:
    """
    Minimal relevance filter (deletion-only) to avoid cross-intent contamination.
    """
    st = str(semantic_type or "").strip().lower()
    if st in {"", "unknown"}:
        return False  # fail-closed: if we don't know semantic_type, do not inject units
    t = str(unit_text or "").strip()
    tl = t.lower()
    q = str(query or "")
    ql = q.lower()

    symptom_cues = ["临床表现", "症状", "表现", "manifestation", "symptom", "present with", "including", "includes", "表现包括", "包括"]
    drug_cues = ["治疗", "药物", "用药", "获批", "批准", "fda", "approved", "approval", "drug", "therapy", "treatment"]
    loc_cues = ["地区", "分布", "省", "市", "县", "区域", "reported in", "distributed in", "occurred in", "found in"]
    mech_cues = ["机制", "作用机制", "发病机制", "致病机制", "pathogenesis", "mechanism", "免疫", "免疫应答", "免疫抑制", "感染", "内皮细胞", "单核", "巨噬", "树突", "细胞因子", "炎症", "通透性", "多器官"]

    has_symptom = any(c in t for c in symptom_cues) or any(c in tl for c in symptom_cues)
    has_drug = any(c in t for c in drug_cues) or any(c in tl for c in drug_cues)
    has_loc = any(c in t for c in loc_cues) or any(c in tl for c in loc_cues)
    has_mech = any(c in t for c in mech_cues) or any(c in tl for c in mech_cues)

    # Allow explicit query overlap as a weak signal.
    query_overlap = sum(
        1
        for c in ["临床表现", "症状", "治疗", "药物", "分布", "地区", "fda", "approved", "机制", "作用机制", "发病机制", "致病机制", "pathogenesis", "mechanism"]
        if (c in ql or c in q) and (c in tl or c in t)
    )

    if st == "symptom":
        if has_drug and not any(c in ql or c in q for c in drug_cues):
            return False
        if has_loc and not any(c in ql or c in q for c in loc_cues):
            return False
        return has_symptom or query_overlap >= 1
    if st == "drug":
        if has_symptom and not any(c in ql or c in q for c in symptom_cues):
            return False
        if has_loc and not any(c in ql or c in q for c in loc_cues):
            return False
        return has_drug or query_overlap >= 1
    if st == "location":
        if has_drug and not any(c in ql or c in q for c in drug_cues):
            return False
        if has_symptom and not any(c in ql or c in q for c in symptom_cues):
            return False
        return has_loc or query_overlap >= 1
    if st == "mechanism":
        # Mechanism queries are often phrased without explicit keywords besides "机制".
        # Keep deletion-only filtering: prefer mech-ish sentences, but don't fail-closed if query is clearly mechanism.
        if any(k in q for k in ["机制", "作用机制", "发病机制", "致病机制"]) or any(k in ql for k in ["mechanism", "pathogenesis"]):
            return has_mech or query_overlap >= 1 or (len(t) >= 12)
        # If query isn't explicitly mechanism, be conservative.
        return has_mech or query_overlap >= 2
    return False

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    p = argparse.ArgumentParser(description="Run KVI2Runtime.run_ab test")
    p.add_argument("--model", required=True, help="HF model name or local path")
    p.add_argument("--prompt", required=True, help="User prompt")
    p.add_argument("--kv_dir", required=True, help="KVBank directory")
    p.add_argument(
        "--blocks_jsonl",
        default="",
        help="blocks.enriched.jsonl path (required for pipeline=kvi2; unused for pipeline=simple)",
    )
    p.add_argument(
        "--sentences_jsonl",
        default="",
        help="sentences.jsonl path (optional for pipeline=simple; used for stricter postprocess rules)",
    )
    p.add_argument(
        "--semantic_type_specs",
        default="",
        help="semantic_type_specs.json path (work_dir-level config for intent taxonomy; optional)",
    )
    p.add_argument("--pattern_index_dir", required=True, help="pattern sidecar directory")
    p.add_argument("--sidecar_dir", required=True, help="sidecar directory")
    p.add_argument("--domain_encoder_model", required=True, help="HF encoder model")
    p.add_argument(
        "--pipeline",
        choices=["kvi2", "simple", "route", "modeA", "modeB"],
        default="kvi2",
        help=(
            "kvi2: Pattern+Gate+RIM pipeline; "
            "simple: prompt->similarity retrieval (KVBank)->multi-step KV injection->text answer; "
            "route: evidence routing only; "
            "modeA: evidence routing + free reasoning (LLM); "
            "modeB: evidence routing + evidence projection (no generation)."
        ),
    )
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--kv_refresh_rounds", type=int, default=2)
    p.add_argument("--kv_irrelevant_logit_delta_threshold", type=float, default=0.05)
    p.add_argument("--debug_retrieved_ids", action="store_true")
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument(
        "--answer_mode",
        choices=["list_only", "narrative", "llm", "llm_prose"],
        default="llm_prose",
        help="Answer rendering: list_only / narrative / llm (bypass LIST_ONLY) / llm_prose (bypass LIST_ONLY + force prose, no bullets).",
    )
    # Simple pipeline knobs (architecture debugging)
    p.add_argument("--simple_max_steps", type=int, default=3)
    p.add_argument("--simple_step_new_tokens", type=int, default=96)
    p.add_argument("--simple_max_blocks_per_step", type=int, default=8)
    p.add_argument("--simple_max_sentence_tokens", type=int, default=128)
    p.add_argument("--simple_max_total_injected_tokens", type=int, default=512)
    p.add_argument("--simple_regen_on_violation", action="store_true", help="If validator flags violations, re-inject & re-generate once (max rounds controlled by --simple_max_regen_rounds).")
    p.add_argument("--simple_max_regen_rounds", type=int, default=1, help="Max extra regen rounds after the first answer (0 disables retries).")
    # Routing weights (UI adjustable)
    p.add_argument("--route_w_ann", type=float, default=1.0)
    p.add_argument("--route_w_intent", type=float, default=0.6)
    p.add_argument("--route_w_quality", type=float, default=0.2)
    # NOTE (iron law): Evidence Units text MUST NOT be appended to prompt at runtime.
    # Simple pipeline only supports KV cache injection.
    # Output controls: baseline is frequently hallucinated; keep it opt-in.
    p.add_argument("--show_baseline", action="store_true", help="Include baseline_answer in JSON output")
    p.add_argument("--final_only", action="store_true", help="Print only final_answer (rim_answer) and exit")
    args = p.parse_args()

    try:
        from external_kv_injection.src.runtime.kvi2_runtime import KVI2Runtime, KVI2Config  # type: ignore
        from external_kv_injection.src.domain_encoder import DomainEncoder, DomainEncoderConfig  # type: ignore
        from external_kv_injection.src.kv_bank import FaissKVBank  # type: ignore
        from external_kv_injection.src.retriever import Retriever  # type: ignore
        from external_kv_injection.src.runtime.hf_cache_prefix_injection import (  # type: ignore
            build_past_key_values_prefix,
            stack_ext_kv_items_by_layer,
        )
        from external_kv_injection.src.runtime.multistep_injector import MultiStepInjector  # type: ignore
    except ModuleNotFoundError:
        from src.runtime.kvi2_runtime import KVI2Runtime, KVI2Config  # type: ignore
        from src.domain_encoder import DomainEncoder, DomainEncoderConfig  # type: ignore
        from src.kv_bank import FaissKVBank  # type: ignore
        from src.retriever import Retriever  # type: ignore
        from src.runtime.hf_cache_prefix_injection import build_past_key_values_prefix, stack_ext_kv_items_by_layer  # type: ignore
        from src.runtime.multistep_injector import MultiStepInjector  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = str(args.pipeline)
    tok = None
    model = None
    if pipeline in {"simple", "kvi2", "modeA"}:
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
        torch_dtype = torch.bfloat16 if device.type == "cuda" else None
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, trust_remote_code=True)
        model.to(device).eval()

    # blocks_jsonl is only needed for pipeline=kvi2 (LIST_ONLY, debug/citation helpers).
    block_text_lookup = {}
    if str(args.pipeline) == "kvi2":
        if not str(args.blocks_jsonl or "").strip():
            raise SystemExit("--blocks_jsonl is required for pipeline=kvi2")
        block_text_lookup = _load_block_text_lookup(args.blocks_jsonl)

    sentence_text_lookup: Dict[str, str] = {}
    sentences_jsonl_path = ""
    sj = str(args.sentences_jsonl or "").strip()
    if sj:
        sentences_jsonl_path = sj
        sentence_text_lookup = _load_sentence_text_lookup(sj)

    # ----------------------------------------
    # MODE A/B and ROUTING
    # ----------------------------------------
    if pipeline in {"route", "modeA", "modeB"}:
        if not sentences_jsonl_path:
            raise SystemExit("--sentences_jsonl is required for pipeline=route/modeA/modeB")
        routing = _run_evidence_routing(
            prompt=str(args.prompt),
            kv_dir=str(args.kv_dir),
            sentences_lookup=sentence_text_lookup,
            top_k=int(args.top_k),
            domain_encoder_model=str(args.domain_encoder_model),
            semantic_type_specs_path=str(args.semantic_type_specs),
            pattern_index_dir=str(args.pattern_index_dir),
            w_ann=float(args.route_w_ann),
            w_intent=float(args.route_w_intent),
            w_quality=float(args.route_w_quality),
        )
        if pipeline == "route":
            print(json.dumps(routing, ensure_ascii=False, indent=2))
            return
        if pipeline == "modeB":
            # Evidence Projection only (no generation)
            out = {
                "mode": "B",
                "status": routing.get("status"),
                "evidence_projection": routing.get("evidence_projection"),
                "evidence_ids": routing.get("evidence_ids"),
                "evidence_texts": routing.get("evidence_texts"),
            }
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return
        if pipeline == "modeA":
            # Evidence Routing + Free Reasoning (LLM)
            ev_texts = routing.get("evidence_texts") or []
            # Build a lightweight RAG prompt (allowed in Mode A).
            ev_block = "\n".join([f"[E{i+1}] {str(t)}" for i, t in enumerate(ev_texts) if str(t).strip()])
            modeA_prompt = (
                str(args.prompt).strip()
                + "\n\n请基于以下证据进行自由推理并给出诊断性结论：\n"
                + ev_block
                + "\n\n要求：可归纳、可综合，但不得捏造证据中不存在的事实。"
            )
            if tok is None or model is None:
                raise SystemExit("Mode A requires model/tokenizer")
            # Base LLM output (no evidence prompt) for comparison
            base_prompt = KVI2Runtime._format_prompt(tok, str(args.prompt).strip(), use_chat_template=bool(args.use_chat_template))
            base_answer = MultiStepInjector._greedy_generate_with_past_prefix(
                model=model,
                tokenizer=tok,
                prompt=base_prompt,
                device=device,
                past_key_values=None,
                max_new_tokens=192,
                no_repeat_ngram_size=12,
                repetition_penalty=1.08,
            ).strip()
            modeA_prompt = KVI2Runtime._format_prompt(tok, modeA_prompt, use_chat_template=bool(args.use_chat_template))
            answer = MultiStepInjector._greedy_generate_with_past_prefix(
                model=model,
                tokenizer=tok,
                prompt=modeA_prompt,
                device=device,
                past_key_values=None,
                max_new_tokens=192,
                no_repeat_ngram_size=12,
                repetition_penalty=1.08,
            ).strip()
            out = {
                "mode": "A",
                "diagnosis_result": str(answer or ""),
                "base_llm_result": str(base_answer or ""),
            }
            print(json.dumps(out, ensure_ascii=False, indent=2))
            return

    # ----------------------------------------
    # SIMPLE PIPELINE (architecture debugging)
    # ----------------------------------------
    if pipeline == "simple":
        if tok is None or model is None:
            raise SystemExit("Simple pipeline requires model/tokenizer")
        user_prompt = str(args.prompt)
        base_prompt = KVI2Runtime._format_prompt(tok, user_prompt, use_chat_template=bool(args.use_chat_template))
        base_answer = MultiStepInjector._greedy_generate_with_past_prefix(
            model=model,
            tokenizer=tok,
            prompt=base_prompt,
            device=device,
            past_key_values=None,
            max_new_tokens=192,
            no_repeat_ngram_size=12,
            repetition_penalty=1.08,
        ).strip()

        bank = FaissKVBank.load(Path(str(args.kv_dir)))
        retriever = Retriever(bank)
        enc = DomainEncoder(
            DomainEncoderConfig(
                model_name_or_path=str(args.domain_encoder_model),
                max_length=256,
                normalize=True,
                device=str(device),
            )
        )

        # We want a prose answer (no bullets) in this debug mode.
        # Iron law: do NOT append any evidence text into prompt; rely on KV injection only.
        prose_guard = (
            "\n\n请用自然语言中文段落回答（1-2段），不要使用项目符号或编号列表。"
            "【范围约束】只回答用户问题涉及的维度；不要扩展到症状/治疗/地区分布等其他维度，除非用户明确问到。\n"
            "回答必须严格与证据一致，不要编造。\n"
            "【缩写规则】除非检索/注入的知识内容原文出现该括号形式，否则禁止输出类似 “SFTSV（……）” 的括号扩展。"
        )

        specs = _load_semantic_type_specs_any(pattern_index_dir=str(args.pattern_index_dir), semantic_type_specs_path=str(args.semantic_type_specs))

        max_steps = int(args.simple_max_steps)
        step_new_tokens = int(args.simple_step_new_tokens)
        # Hard budgets (sentence-KVBank)
        max_sentence_tokens = max(16, min(256, int(args.simple_max_sentence_tokens)))
        max_total_injected_tokens = max(64, min(2048, int(args.simple_max_total_injected_tokens)))
        max_blocks = int(args.simple_max_blocks_per_step)
        # UI hard-cap: at most 4 sentences per step.
        max_blocks = max(1, min(4, int(max_blocks)))
        top_k = int(args.top_k)
        def _run_once(*, prompt_for_user: str, extra_guard: str) -> Tuple[str, List[Dict[str, Any]]]:
            used: set[str] = set()
            generated = ""
            step_debug: List[Dict[str, Any]] = []
            for step in range(max_steps):
                qtxt = prompt_for_user + ("\n" + generated if generated.strip() else "")
                target_st, st_dbg = _infer_intent_from_specs(enc=enc, query=qtxt, specs=specs)
                qv = enc.encode(qtxt)[0]
                rr = retriever.search(qv, top_k=int(top_k * 3), filters=None, query_text=qtxt)
                candidates: List[Tuple[float, Any, str]] = []
                cand_ids: List[str] = []
                for it in (rr.items or []):
                    bid = _kv_id(it)
                    if not bid:
                        continue
                    cand_ids.append(bid)
                    score = float(getattr(it, "score", 0.0) or 0.0)
                    candidates.append((float(score), it, bid))
                candidates.sort(key=lambda x: x[0], reverse=True)

                semantic_rerank_dbg: Dict[str, Any] = {"enabled": False, "semantic_type": str(target_st), "method": "none"}
                # Semantic rerank driven by build-time tags if present; fallback to embedding rerank.
                if str(target_st) != "unknown":
                    try:
                        import numpy as np  # type: ignore

                        enriched2: List[Tuple[float, float, Any, str]] = []
                        used_tags = 0
                        for ann_score, it, bid in candidates:
                            meta = getattr(it, "meta", None) or {}
                            payload = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
                            scores = payload.get("semantic_scores") if isinstance(payload.get("semantic_scores"), dict) else {}
                            sem = scores.get(str(target_st).lower())
                            if isinstance(sem, (int, float)):
                                used_tags += 1
                                sem_score = float(sem)
                            else:
                                # fallback: compute sim against the sentence text
                                if sentence_text_lookup:
                                    spec = specs.get(str(target_st).lower()) if isinstance(specs, dict) else None
                                    desc = str((spec or {}).get("description") or "").strip() if isinstance(spec, dict) else str(target_st)
                                    anchor_text = f"[semantic_type]\n{str(target_st).lower()}\n\n[description]\n{desc}\n\n[query]\n{qtxt}\n"
                                    a1 = enc.encode([anchor_text], batch_size=1)[0]
                                    txt = str(sentence_text_lookup.get(str(bid), "") or "")
                                    v1 = enc.encode([txt], batch_size=1)[0] if txt else a1 * 0
                                    sem_score = float((v1 @ a1).astype(float))
                                else:
                                    sem_score = 0.0
                            enriched2.append((float(sem_score), float(ann_score), it, bid))
                        enriched2.sort(key=lambda x: (x[0], x[1]), reverse=True)
                        candidates = [(ann, it, bid) for sem, ann, it, bid in enriched2]
                        cand_ids = [bid for _ann, _it, bid in candidates]
                        semantic_rerank_dbg = {
                            "enabled": True,
                            "semantic_type": str(target_st),
                            "method": "tag_score_then_fallback_embedding",
                            "used_tag_scores": int(used_tags),
                            "top_semantic_sims": [float(x[0]) for x in enriched2[: min(5, len(enriched2))]],
                            "top_ids": [str(x[3]) for x in enriched2[: min(5, len(enriched2))]],
                        }
                    except Exception as e:
                        semantic_rerank_dbg = {
                            "enabled": False,
                            "semantic_type": str(target_st),
                            "method": "failed",
                            "error": f"{type(e).__name__}",
                        }

                selected: List[Any] = []
                selected_ids: List[str] = []
                selected_kv_lens: List[int] = []
                selected_texts: List[Dict[str, Any]] = []
                total_kv_tokens = 0
                for _score, it, bid in candidates:
                    if bid in used:
                        continue
                    kv_len = None
                    try:
                        if hasattr(it, "meta") and isinstance(getattr(it, "meta"), dict):
                            kv_len = it.meta.get("kv_len")
                    except Exception:
                        kv_len = None
                    kv_len_i = int(kv_len) if isinstance(kv_len, int) else int(max_sentence_tokens)
                    kv_len_i = max(0, min(int(max_sentence_tokens), int(kv_len_i)))
                    if (total_kv_tokens + kv_len_i) > int(max_total_injected_tokens):
                        continue
                    selected.append(it)
                    selected_ids.append(bid)
                    selected_kv_lens.append(int(kv_len_i))
                    total_kv_tokens += int(kv_len_i)
                    used.add(bid)
                    if len(selected) >= max_blocks:
                        break

                # Attach injected sentence texts for debug (if sentences_jsonl provided).
                if sentence_text_lookup and selected_ids:
                    for i, sid in enumerate(selected_ids):
                        try:
                            selected_texts.append(
                                {
                                    "id": str(sid),
                                    "kv_len": int(selected_kv_lens[i]) if i < len(selected_kv_lens) else None,
                                    "text": str(sentence_text_lookup.get(str(sid), "") or ""),
                                }
                            )
                        except Exception:
                            continue

                dtype2 = next(model.parameters()).dtype
                ext_by_layer: Dict[int, Any] = {}
                for li in (0, 1, 2, 3):
                    try:
                        ext_by_layer[int(li)] = stack_ext_kv_items_by_layer(
                            items=selected,
                            layer_id=int(li),
                            batch_size=1,
                            device=device,
                            dtype=dtype2,
                        )
                    except Exception:
                        continue
                pkv = build_past_key_values_prefix(model=model, ext_kv_by_layer=ext_by_layer) if ext_by_layer else None

                # Intent guard from config-driven type (no evidence text in prompt).
                intent_guard = ""
                if str(target_st) != "unknown":
                    st_key = str(target_st).strip().lower()
                    spec = specs.get(st_key) if isinstance(specs, dict) else None
                    focus_terms = (spec or {}).get("focus_terms") if isinstance(spec, dict) else None
                    deny_terms = (spec or {}).get("deny_terms") if isinstance(spec, dict) else None
                    focus_line = ""
                    if isinstance(focus_terms, list) and focus_terms:
                        focus = [str(x).strip() for x in focus_terms if str(x).strip()]
                        if focus:
                            focus_line = f"\n【关注关键词】{', '.join(focus)}。"
                    deny_line = ""
                    if isinstance(deny_terms, list) and deny_terms:
                        deny = [str(x).strip() for x in deny_terms if str(x).strip()]
                        if deny:
                            deny_line = f"\n【避免关键词】{', '.join(deny)}。"
                    intent_guard = (
                        f"\n【意图约束】本轮目标语义维度：{str(target_st)}。只回答该维度相关内容；不要扩展到其他维度。"
                        + focus_line
                        + deny_line
                        + "\n【证据一致性】不得补充注入知识未支持的具体事实；如无法从注入知识确定，请明确说明“无法从注入知识确定”。"
                    )
                step_prompt = (
                    prompt_for_user + prose_guard + intent_guard + (extra_guard or "") + ("\n\n" + generated if generated.strip() else "")
                ).strip()
                step_prompt = KVI2Runtime._format_prompt(tok, step_prompt, use_chat_template=bool(args.use_chat_template))
                chunk = MultiStepInjector._greedy_generate_with_past_prefix(
                    model=model,
                    tokenizer=tok,
                    prompt=step_prompt,
                    device=device,
                    past_key_values=pkv,
                    max_new_tokens=int(step_new_tokens),
                    no_repeat_ngram_size=12,
                    repetition_penalty=1.08,
                )
                chunk = str(chunk or "").strip()
                generated = (generated + ("\n" if (generated and chunk) else "") + chunk).strip()
                step_debug.append(
                    {
                        "step": int(step),
                        "retrieved_ids_top": cand_ids[: min(12, len(cand_ids))],
                        "selected_ids": selected_ids,
                        "selected_kv_lens": selected_kv_lens,
                        "selected_texts": selected_texts,
                        "selected_total_kv_tokens": int(total_kv_tokens),
                        "max_injected_sentences_per_step": int(max_blocks),
                        "max_sentence_tokens": int(max_sentence_tokens),
                        "max_total_injected_tokens": int(max_total_injected_tokens),
                        "semantic_intent": st_dbg,
                        "semantic_rerank": semantic_rerank_dbg,
                    }
                )
                if not chunk:
                    break
            return generated.strip(), step_debug

        def _gather_ev_texts(all_steps: List[Dict[str, Any]]) -> List[str]:
            if not sentence_text_lookup:
                return []
            evs: List[str] = []
            for st in all_steps or []:
                for sid in (st.get("selected_ids") or []) if isinstance(st.get("selected_ids"), list) else []:
                    evs.append(sentence_text_lookup.get(str(sid), ""))
            return [x for x in evs if str(x or "").strip()]

        regen_enabled = bool(args.simple_regen_on_violation)
        max_regen_rounds = max(0, int(args.simple_max_regen_rounds))

        injected_round0, steps0 = _run_once(prompt_for_user=user_prompt, extra_guard="")
        ev_texts0 = _gather_ev_texts(steps0)
        intent0, intent_dbg0 = _infer_intent_from_specs(enc=enc, query=user_prompt, specs=specs)
        violations0 = _detect_violations(
            answer=injected_round0,
            user_prompt=user_prompt,
            evidence_texts=ev_texts0,
            intent=str(intent0),
            specs=specs,
        )
        # Some violations are "format-only" and can be safely handled by postprocess (e.g. acronym expansions).
        # Do NOT trigger regen for those, otherwise the model may overreact and refuse to answer.
        regen_trigger_types = {
            "empty_answer",
            "intent_drift_deny_terms",
            "answer_items_not_in_injected_evidence",
        }
        violations0_for_regen = [v for v in (violations0 or []) if str(v.get("type") or "") in regen_trigger_types]

        injected_final = injected_round0
        steps_final = steps0
        violations_final = violations0
        regen_used = 0

        if regen_enabled and violations0_for_regen and max_regen_rounds > 0:
            # Build a minimal "regen guard" using violation info (no evidence text appended).
            # Keep it short and deterministic.
            deny_terms: List[str] = []
            for v in violations0_for_regen:
                if v.get("type") in {"intent_drift_deny_terms"}:
                    deny_terms.extend([str(x) for x in (v.get("terms") or []) if str(x).strip()])
            deny_terms = list(dict.fromkeys([t for t in deny_terms if t]))
            deny_line = f"\n【禁止扩展】禁止提及：{', '.join(deny_terms)}。" if deny_terms else ""
            regen_guard = (
                "\n\n【检测到上轮回答存在违规】请重写答案：\n"
                "1) 严格禁止输出未经注入 sentence 原文出现的括号扩写（例如 SFTSV（...））。\n"
                "2) 只回答用户问题涉及维度；不要扩展到症状/治疗/地区分布等其他维度，除非用户明确问到。\n"
                "【症状严格规则】若问题是“症状/临床表现”，只允许列出注入知识原文中出现过的症状/体征/实验室异常词组，不得新增其他症状名。\n"
                "3) 先列出“注入知识明确支持的要点”（用短句/逗号分隔）。\n"
                "4) 若注入知识未覆盖某些细节，请仅说明“注入知识未覆盖该细节/证据不足”，不要直接放弃回答。"
                + deny_line
            )
            injected_round1, steps1 = _run_once(prompt_for_user=user_prompt, extra_guard=regen_guard)
            ev_texts1 = _gather_ev_texts(steps1)
            violations1 = _detect_violations(
                answer=injected_round1,
                user_prompt=user_prompt,
                evidence_texts=ev_texts1,
                intent=str(intent0),
                specs=specs,
            )
            injected_final = injected_round1
            steps_final = steps1
            violations_final = violations1
            regen_used = 1

        # Final safety: enforce acronym expansion rule post-hoc (should be rare after regen).
        ev_texts_final = _gather_ev_texts(steps_final)
        stripped = _strip_unapproved_abbr_expansions(text=injected_final.strip(), evidence_texts=ev_texts_final).strip()
        postprocess_changed = stripped != injected_final.strip()
        injected_final = stripped
        violations_after_postprocess = _detect_violations(
            answer=injected_final.strip(),
            user_prompt=user_prompt,
            evidence_texts=ev_texts_final,
            intent=str(intent0),
            specs=specs,
        )

        out_simple = {
            "pipeline": "simple",
            "prompt": user_prompt,
            "base_answer": base_answer,
            "injected_answer": injected_final.strip(),
            "injected_answer_round0": injected_round0.strip(),
            "violations_round0": violations0,
            "violations_round0_for_regen": violations0_for_regen,
            "violations_final": violations_after_postprocess,
            "violations_before_postprocess": violations_final,
            "intent": str(intent0),
            "intent_debug": intent_dbg0,
            "regen_on_violation": bool(regen_enabled),
            "regen_rounds_used": int(regen_used),
            "postprocess_stripped_abbr": bool(postprocess_changed),
            "steps": steps_final,
        }
        if bool(args.final_only):
            # Per your requirement: include base LLM output as well.
            print("=== Base LLM (no injection) ===\n")
            print(base_answer)
            print("\n\n=== Injected (multi-step) ===\n")
            print(out_simple["injected_answer"])
            return
        print(json.dumps(out_simple, ensure_ascii=False, indent=2))
        return

    # ----------------------------------------
    # KVI2 PIPELINE (default)
    # ----------------------------------------
    cfg = KVI2Config(
        top_k=int(args.top_k),
        kv_refresh_rounds=int(args.kv_refresh_rounds),
        kv_irrelevant_logit_delta_threshold=float(args.kv_irrelevant_logit_delta_threshold),
        pattern_index_dir=str(args.pattern_index_dir),
        debug_retrieved_ids=bool(args.debug_retrieved_ids),
        answer_mode=str(args.answer_mode),
    )
    runtime = KVI2Runtime(cfg=cfg, domain_encoder_model=str(args.domain_encoder_model))

    out = runtime.run_ab(
        model=model,
        tokenizer=tok,
        prompt=str(args.prompt),
        kv_dir=str(args.kv_dir),
        device=device,
        use_chat_template=bool(args.use_chat_template),
        block_text_lookup=block_text_lookup,
        sidecar_dir=str(args.sidecar_dir),
    )
    final_answer = str(out.get("rim_answer", "") or "").strip()
    out["final_answer"] = final_answer
    if not bool(args.show_baseline):
        # Avoid visual pollution: baseline is debug-only unless explicitly requested.
        if "baseline_answer" in out:
            del out["baseline_answer"]
    if bool(args.final_only):
        print(final_answer)
        return
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
