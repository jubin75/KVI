"""
Minimal KVI2Runtime.run_ab test harness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore


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
    Keep only injectable sentence_enumerative units.
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
        if str(u.get("unit_type") or "") != "sentence_enumerative":
            continue
        inj = u.get("injectability") if isinstance(u.get("injectability"), dict) else {}
        if not bool(inj.get("allowed")):
            continue
        s = str(u.get("text") or "").strip()
        if s:
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
                dbg.update(
                    {
                        "method": "contracts+scoring",
                        "pattern_id": str(best.pattern_id),
                        "semantic_type": "unknown",
                        "rationale": ["ambiguous_semantic_type"],
                    }
                )
                return "unknown", dbg
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
    if sum([int(symptom), int(drug), int(location)]) != 1:
        dbg.update({"method": "heuristic", "semantic_type": "unknown", "rationale": ["ambiguous_or_no_intent"]})
        return "unknown", dbg
    if symptom:
        dbg.update({"method": "heuristic", "semantic_type": "symptom", "rationale": ["cue:symptom"]})
        return "symptom", dbg
    if drug:
        dbg.update({"method": "heuristic", "semantic_type": "drug", "rationale": ["cue:drug"]})
        return "drug", dbg
    dbg.update({"method": "heuristic", "semantic_type": "location", "rationale": ["cue:location"]})
    return "location", dbg


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

    has_symptom = any(c in t for c in symptom_cues) or any(c in tl for c in symptom_cues)
    has_drug = any(c in t for c in drug_cues) or any(c in tl for c in drug_cues)
    has_loc = any(c in t for c in loc_cues) or any(c in tl for c in loc_cues)

    # Allow explicit query overlap as a weak signal.
    query_overlap = sum(1 for c in ["临床表现", "症状", "治疗", "药物", "分布", "地区", "fda", "approved"] if (c in ql or c in q) and (c in tl or c in t))

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
    p.add_argument("--blocks_jsonl", required=True, help="blocks.enriched.jsonl path")
    p.add_argument("--pattern_index_dir", required=True, help="pattern sidecar directory")
    p.add_argument("--sidecar_dir", required=True, help="sidecar directory")
    p.add_argument("--domain_encoder_model", required=True, help="HF encoder model")
    p.add_argument(
        "--pipeline",
        choices=["kvi2", "simple"],
        default="kvi2",
        help="kvi2: Pattern+Gate+RIM pipeline; simple: prompt->similarity retrieval (KVBank)->multi-step KV injection->text answer (also prints base LLM).",
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
    # Evidence Unit Pipeline knobs (sentence-level only; no LIST_ONLY)
    p.add_argument("--simple_use_evidence_units", action="store_true", help="Use sentence-level evidence units to select/ground injected blocks")
    p.add_argument("--simple_max_unit_sentences", type=int, default=6)
    p.add_argument("--simple_require_units", action="store_true", help="If set, only inject blocks that have injectable sentence-level units (fallback to ANN if none)")
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
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    torch_dtype = torch.bfloat16 if device.type == "cuda" else None
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype, trust_remote_code=True)
    model.to(device).eval()

    block_text_lookup = _load_block_text_lookup(args.blocks_jsonl)

    # ----------------------------------------
    # SIMPLE PIPELINE (architecture debugging)
    # ----------------------------------------
    if str(args.pipeline) == "simple":
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

        # Evidence Unit Pipeline (sentence-level only; no list-item / no LIST_ONLY)
        unit_lookup: Dict[str, List[str]] = {}
        router_dbg: Dict[str, Any] = {}
        target_semantic_type = "unknown"
        if bool(args.simple_use_evidence_units):
            target_semantic_type, router_dbg = _infer_target_semantic_type_for_query(
                query=user_prompt,
                kv_dir=str(args.kv_dir),
                pattern_index_dir=str(args.pattern_index_dir),
            )
            try:
                from external_kv_injection.src.evidence.evidence_unit_extractor import EvidenceUnitExtractor  # type: ignore
            except ModuleNotFoundError:
                from src.evidence.evidence_unit_extractor import EvidenceUnitExtractor  # type: ignore
            eu = EvidenceUnitExtractor()
            meta_lookup = _load_block_metadata_lookup(str(args.blocks_jsonl))
            for bid, txt in (block_text_lookup or {}).items():
                meta = meta_lookup.get(str(bid), {})
                units = _extract_sentence_units_only(
                    extractor=eu,
                    block_id=str(bid),
                    text=str(txt or ""),
                    metadata=meta,
                )
                # Scheme #1: semantic_type-aware relevance filter (deletion-only; fail-closed to empty if unknown)
                if target_semantic_type != "unknown":
                    units = [
                        u
                        for u in units
                        if _unit_relevant_to_semantic_type(unit_text=u, semantic_type=target_semantic_type, query=user_prompt)
                    ]
                else:
                    units = []
                unit_lookup[str(bid)] = units

        # We want a prose answer (no bullets) in this debug mode.
        prose_guard = "\n\n请用自然语言中文段落回答（1-2段），不要使用项目符号或编号列表。回答必须与证据一致，不要编造。"

        max_steps = int(args.simple_max_steps)
        step_new_tokens = int(args.simple_step_new_tokens)
        max_blocks = int(args.simple_max_blocks_per_step)
        max_unit_sents = int(args.simple_max_unit_sentences)
        top_k = int(args.top_k)
        used: set[str] = set()
        generated = ""
        step_debug: List[Dict[str, Any]] = []

        for step in range(max_steps):
            qtxt = user_prompt + ("\n" + generated if generated.strip() else "")
            qv = enc.encode(qtxt)[0]
            rr = retriever.search(qv, top_k=int(top_k * 3), filters=None, query_text=qtxt)
            # Rank candidates by evidence-unit richness first (if enabled), then ANN score.
            candidates: List[Tuple[int, float, Any, str]] = []
            cand_ids: List[str] = []
            for it in (rr.items or []):
                bid = _kv_id(it)
                if not bid:
                    continue
                cand_ids.append(bid)
                unit_cnt = len(unit_lookup.get(bid) or []) if unit_lookup else 0
                score = float(getattr(it, "score", 0.0) or 0.0)
                candidates.append((int(unit_cnt), float(score), it, bid))
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            selected: List[Any] = []
            selected_ids: List[str] = []
            require_units = bool(args.simple_require_units) and bool(unit_lookup)
            # First pass: pick blocks with sentence-level injectable units.
            if unit_lookup:
                for unit_cnt, _score, it, bid in candidates:
                    if bid in used:
                        continue
                    if require_units and int(unit_cnt) <= 0:
                        continue
                    if int(unit_cnt) <= 0:
                        continue
                    selected.append(it)
                    selected_ids.append(bid)
                    used.add(bid)
                    if len(selected) >= max_blocks:
                        break
            # Fallback: pick top ANN hits if no unit-rich blocks found.
            if not selected:
                for _unit_cnt, _score, it, bid in candidates:
                    if bid in used:
                        continue
                    selected.append(it)
                    selected_ids.append(bid)
                    used.add(bid)
                    if len(selected) >= max_blocks:
                        break

            evidence_sents: List[str] = []
            if unit_lookup:
                for bid in selected_ids:
                    evidence_sents.extend(unit_lookup.get(bid) or [])
                    if len(evidence_sents) >= max_unit_sents:
                        break
            evidence_sents = evidence_sents[: max(0, max_unit_sents)]
            evidence_appendix = ""
            if evidence_sents:
                evidence_appendix = "\n\n[Evidence Units]\n" + "\n".join([s for s in evidence_sents]) + "\n"
            # Build injected prefix from selected blocks (layers 0..3).
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

            step_prompt = (user_prompt + prose_guard + evidence_appendix + ("\n\n" + generated if generated.strip() else "")).strip()
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
                    "selected_unit_counts": {bid: int(len(unit_lookup.get(bid) or [])) for bid in selected_ids} if unit_lookup else {},
                    "evidence_units_shown": evidence_sents[: min(3, len(evidence_sents))],
                }
            )
            if not chunk:
                break

        out_simple = {
            "pipeline": "simple",
            "prompt": user_prompt,
            "base_answer": base_answer,
            "injected_answer": generated.strip(),
            "steps": step_debug,
            "semantic_type_router": {"target_semantic_type": target_semantic_type, **(router_dbg or {})}
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
