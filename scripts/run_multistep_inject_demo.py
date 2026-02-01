"""
CLI：Multi-step Injection Demo（Qwen + blocks KVBank）

严格遵循 PRD/多步注入的工程实现.md 的核心约束：
- 每步注入 ≤1024 tokens（由 256-token blocks 组成）
- 注入层默认 0..3
- retrieval 是推理的一部分：每步基于当前状态重新检索
- stopping policy（边际收益 + 冗余 + 安全上限）
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json
import re
import unicodedata
from typing import Optional
from enum import Enum

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.kv_bank import FaissKVBank  # type: ignore
    from external_kv_injection.src.retriever import (  # type: ignore
        EvidenceFirstRetriever,
        EvidenceFirstRetrieverConfig,
        Retriever,
        RoutedRetriever,
        RoutedRetrieverConfig,
    )
    from external_kv_injection.src.runtime.multistep_injector import MultiStepConfig, MultiStepInjector  # type: ignore
    from external_kv_injection.src.runtime.hf_cache_prefix_injection import (  # type: ignore
        build_past_key_values_prefix,
        stack_ext_kv_items_by_layer,
    )
    from external_kv_injection.src.encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig  # type: ignore
    from external_kv_injection.src.runtime.postprocess import postprocess_answer  # type: ignore
except ModuleNotFoundError:
    from src.kv_bank import FaissKVBank  # type: ignore
    from src.retriever import EvidenceFirstRetriever, EvidenceFirstRetrieverConfig, Retriever, RoutedRetriever, RoutedRetrieverConfig  # type: ignore
    from src.runtime.multistep_injector import MultiStepConfig, MultiStepInjector  # type: ignore
    from src.runtime.hf_cache_prefix_injection import build_past_key_values_prefix, stack_ext_kv_items_by_layer  # type: ignore
    from src.encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig  # type: ignore
    from src.runtime.postprocess import postprocess_answer  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--kv_dir", default=None, help="Path to KVBank. If omitted, use --topic + --topic_work_dir.")
    p.add_argument(
        "--kv_dir_schema",
        default=None,
        help="Path to Schema KVBank (kvbank_schema). This is the ONLY bank allowed to be injected. "
        "If omitted, use --topic + --topic_work_dir auto-detection.",
    )
    p.add_argument(
        "--kv_dir_evidence",
        default=None,
        help="Optional evidence KVBank dir (e.g. $WORK_DIR/kvbank_evidence). "
        "If set (or auto-detected in --topic mode), retriever will search evidence first and fall back to raw.",
    )
    p.add_argument("--topic", choices=["sftsv", "sarscov2"], default=None, help="Optional topic name for 专题库 mode.")
    p.add_argument(
        "--topic_work_dir",
        default=None,
        help="If set with --topic, resolve paths like: <topic_work_dir>/<topic>/{blocks.jsonl,kvbank_blocks,kvbank_tables}.",
    )
    p.add_argument("--kv_dir_tables", default=None, help="Optional tables KVBank dir (built by --split_tables).")
    p.add_argument("--enable_table_routing", action="store_true", help="If set, route table-like queries to kv_dir_tables.")
    p.add_argument("--table_top_k", type=int, default=4, help="When routing hits, retrieve up to N table blocks.")
    p.add_argument(
        "--blocks_jsonl",
        default=None,
        help="Optional blocks.jsonl path (e.g. $WORK_DIR/blocks.jsonl). If provided, print text snippets for selected blocks.",
    )
    p.add_argument(
        "--blocks_jsonl_evidence",
        default=None,
        help="Optional evidence blocks jsonl path (e.g. $WORK_DIR/blocks.evidence.jsonl). "
        "Used for debug text lookups / lang filtering when evidence KVBank is enabled.",
    )
    p.add_argument(
        "--blocks_jsonl_schema",
        default=None,
        help="Optional schema blocks jsonl path (e.g. $WORK_DIR/blocks.schema.jsonl). "
        "Required for schema injection (schema texts must be forwarded to build KV cache).",
    )
    p.add_argument(
        "--allowed_langs",
        default="zh,en",
        help="Comma-separated allowlist of langs for block selection when --blocks_jsonl is provided (e.g. 'zh,en'). "
        "Uses a lightweight heuristic (kana->ja, han->zh, else en).",
    )
    p.add_argument("--prompt", required=True)
    p.add_argument(
        "--retrieval_only",
        action="store_true",
        help="If set, run ONLY retrieval (no KV injection, no LLM forward/generation). "
        "Recommended to debug whether top-k candidates contain evidence sentences (tick/vector/transmission/蜱/媒介).",
    )
    p.add_argument(
        "--use_chat_template",
        action="store_true",
        help="If tokenizer supports it, format the model prompt using tokenizer.apply_chat_template. "
        "This often reduces chat transcript artifacts like 'Human:' in outputs.",
    )
    p.add_argument(
        "--append_evidence_to_prompt",
        action="store_true",
        help="FORBIDDEN (iron law): do not append any evidence text into prompt.",
    )
    p.add_argument(
        "--rewrite_query_for_retrieval",
        choices=["off", "acronym_lexicon_from_blocks", "zh_en_intent", "auto"],
        default="auto",
        help="Rewrite ONLY the retrieval query (not the model prompt). "
        "This can improve cross-lingual alignment and acronym disambiguation without requiring users to type expansions. "
        "Modes: "
        "acronym_lexicon_from_blocks=append expansions inferred from blocks.jsonl; "
        "zh_en_intent=append an English gloss for common ZH intent keywords (传播/途径/媒介/证据句); "
        "auto=best-effort combo (safe default).",
    )
    p.add_argument(
        "--quality_filter_from_blocks",
        action="store_true",
        help="If set, build a quality allowlist from blocks_jsonl and only allow injection of 'good' blocks "
        "(filters out control-char garbage / low-content fragments). Recommended.",
    )
    p.add_argument("--max_nonprintable_ratio", type=float, default=0.02)
    p.add_argument("--min_quality_score", type=float, default=0.6)
    p.add_argument("--domain_encoder_model", required=True, help="DomainEncoder for query embedding (must match KVBank keys)")
    p.add_argument("--layers", default="0,1,2,3")
    p.add_argument("--max_steps", type=int, default=8)
    p.add_argument("--max_step_tokens", type=int, default=1024)
    p.add_argument("--max_total_tokens", type=int, default=2048)
    p.add_argument("--top_k_blocks", type=int, default=8)
    p.add_argument("--max_blocks_per_step", type=int, default=8, help="Cap selected blocks per step. For RoPE models, try 1 first.")
    p.add_argument(
        "--schema_required_slots",
        default="",
        help="Optional comma-separated required schema slots (derived upstream). "
        "If empty, defaults to ALL slots and will only prevent repeated slot injection.",
    )
    p.add_argument(
        "--min_kv_len_to_inject",
        type=int,
        default=None,
        help="Minimum kv_len to inject. Default: 128 for raw blocks; auto-lowered to 16 when evidence KVBank is enabled "
        "(evidence sentences are shorter by design).",
    )
    p.add_argument("--max_new_tokens", type=int, default=128)
    # Baseline is extremely useful for A/B debug; default to printing it.
    p.add_argument("--skip_baseline", action="store_true", help="If set, do NOT print baseline answer (no injection).")
    p.add_argument(
        "--print_baseline",
        action="store_true",
        help="(legacy) Kept for backward compatibility; baseline is printed by default unless --skip_baseline.",
    )
    p.add_argument("--use_attention_entropy", action="store_true", help="Enable external KV attention entropy stopping signal")
    p.add_argument("--entropy_threshold", type=float, default=0.35, help="Normalized entropy threshold in [0,1]")
    p.add_argument(
        "--debug_single_block",
        action="store_true",
        help="Debug preset: force max_blocks_per_step=1, max_steps=1, disable table routing. "
        "NOTE: it does NOT override top_k_blocks, so you can still inspect top-N candidates via --debug_print_candidates.",
    )
    p.add_argument(
        "--debug_print_candidates",
        type=int,
        default=0,
        help="If >0, print top-N retrieved candidate ids + scores for each step.",
    )
    p.add_argument(
        "--kvi1_mode",
        action="store_true",
        help="Force KVI1.0 single-pass semantic KV injection (no gate/slots). "
        "This bypasses AnswerMode and multistep injector.",
    )
    p.add_argument(
        "--kvi1_use_generate",
        action="store_true",
        help="In --kvi1_mode, use model.generate (same path as base LLM) instead of manual greedy decode.",
    )
    p.add_argument("--do_sample", action="store_true", help="Enable sampling for KVI1 decode (applies to kvi1_mode path).")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for KVI1 decode.")
    p.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling for KVI1 decode.")
    p.add_argument("--top_k", type=int, default=0, help="Top-k sampling for KVI1 decode (0 disables).")
    p.add_argument("--min_new_tokens", type=int, default=0, help="Minimum new tokens for KVI1 decode (prevents early EOS).")
    p.add_argument(
        "--print_retrieval_query",
        action="store_true",
        help="If set, print the exact retrieval query text (after optional rewrite).",
    )
    p.add_argument(
        "--ground_with_selected_text",
        action="store_true",
        help="FORBIDDEN (iron law): do not append any evidence text into prompt.",
    )
    p.add_argument(
        "--use_struct_slots",
        action="store_true",
        help="If set, enable Evidence->Schema(struct slots)->Injection/Generation: "
        "build a compact schema summary from selected evidence blocks and inject that schema prefix "
        "(instead of injecting evidence sentences as answer text). Requires --blocks_jsonl or --blocks_jsonl_evidence.",
    )
    p.add_argument(
        "--grounding_instructions",
        type=str,
        default="",
        help="Optional extra instructions appended after the evidence sentences when --ground_with_selected_text is set. "
        "Use this to control answer format without changing code. Example: "
        "'请用中文输出：主要传播途径：...\\n其他途径：...\\n证据原文：\"...\"'",
    )
    p.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=12,
        help="No-repeat ngram constraint during final decode to reduce repetitive filler (0 disables).",
    )
    # Three-layer knowledge output (L0/L1/L2)
    p.add_argument(
        "--enable_layer2",
        action="store_true",
        help="Enable L2 (Speculative / Reasoned) section in the strict three-layer output. "
        "Applies to GROUNDED mode only (schema injection path).",
    )
    p.add_argument(
        "--layer1_max_new_tokens",
        type=int,
        default=256,
        help="Token budget for L1 (Domain Prior) generation (GROUNDED mode only).",
    )
    p.add_argument(
        "--layer2_max_new_tokens",
        type=int,
        default=192,
        help="Token budget for L2 generation when --enable_layer2 is set (GROUNDED mode only).",
    )
    p.add_argument(
        "--print_selected_block_text",
        action="store_true",
        help="If set, print the FULL original block text (from blocks_jsonl) for selected blocks. "
        "This is useful to manually judge retrieval correctness.",
    )
    p.add_argument(
        "--selected_block_max_chars",
        type=int,
        default=0,
        help="When printing full block text, optionally truncate to first N chars (0 = no truncation).",
    )
    p.add_argument(
        "--print_selected_block_token_ids",
        action="store_true",
        help="If set, print token ids (from the current tokenizer) for selected blocks. Useful for low-level debug.",
    )
    p.add_argument(
        "--selected_block_max_token_ids",
        type=int,
        default=0,
        help="When printing token ids, optionally truncate to first N ids (0 = no truncation).",
    )
    p.add_argument(
        "--print_top_candidates_text",
        type=int,
        default=0,
        help="If >0, print FULL text for top-N retrieved candidates (requires --blocks_jsonl). "
        "Tip: combine with --retrieval_only to avoid loading the base LLM.",
    )
    p.add_argument(
        "--top_candidates_max_chars",
        type=int,
        default=2000,
        help="When printing top candidates full text, truncate each candidate to first N chars (0 = no truncation).",
    )
    args = p.parse_args()
    if bool(getattr(args, "append_evidence_to_prompt", False)) or bool(getattr(args, "ground_with_selected_text", False)):
        raise SystemExit("Forbidden by iron law: do not append any evidence text into prompt. Use KV injection only.")

    class AnswerMode(str, Enum):
        GROUNDED = "GROUNDED"  # schema injection + evidence appended grounding
        UNGROUNDED = "UNGROUNDED"  # base LLM (no injection)
        REFUSE = "REFUSE"  # deterministic refusal/fallback

    # 专题库 mode: resolve paths from topic_work_dir/topic
    if args.kv_dir is None:
        if not args.topic or not args.topic_work_dir:
            raise SystemExit("Missing --kv_dir. Either pass --kv_dir or use --topic + --topic_work_dir.")
        twd = Path(str(args.topic_work_dir))
        topic = str(args.topic)

        # Support multiple common layouts:
        # 1) <topic_work_dir>/<topic>/{kvbank_blocks,kvbank_tables,blocks.jsonl}
        # 2) <topic_work_dir>/<topic>/work/{kvbank_blocks,kvbank_tables,blocks.jsonl}
        # 3) Same as above but with topic folder uppercased (e.g. SFTSV, SARS2)
        candidates = [
            twd / topic,
            twd / topic / "work",
            twd / topic.upper(),
            twd / topic.upper() / "work",
        ]

        def _pick_kv_dir(base_dir: Path) -> Path:
            # Prefer kvbank_blocks (or kvbank_blocks_v2) that actually contains a manifest.
            opts = [base_dir / "kvbank_blocks", base_dir / "kvbank_blocks_v2"]
            for p in opts:
                if (p / "manifest.json").exists():
                    return p
            # fall back to the non-existing default to preserve older behavior (FaissKVBank.load will hint)
            return base_dir / "kvbank_blocks"

        def _pick_blocks_jsonl(base_dir: Path) -> Path:
            opts = [base_dir / "blocks.jsonl"]
            for p in opts:
                if p.exists():
                    return p
            return base_dir / "blocks.jsonl"

        def _pick_evidence_kv_dir(base_dir: Path) -> Optional[Path]:
            opts = [base_dir / "kvbank_evidence", base_dir / "kvbank_evidence_v2"]
            for p in opts:
                if (p / "manifest.json").exists():
                    return p
            return None

        def _pick_evidence_blocks_jsonl(base_dir: Path) -> Optional[Path]:
            opts = [base_dir / "blocks.evidence.jsonl", base_dir / "blocks.evidence.v2.jsonl"]
            for p in opts:
                if p.exists():
                    return p
            return None

        def _pick_schema_kv_dir(base_dir: Path) -> Optional[Path]:
            opts = [base_dir / "kvbank_schema", base_dir / "kvbank_schema_v2"]
            for p in opts:
                if (p / "manifest.json").exists():
                    return p
            return None

        def _pick_schema_blocks_jsonl(base_dir: Path) -> Optional[Path]:
            opts = [base_dir / "blocks.schema.jsonl", base_dir / "blocks.schema.v2.jsonl"]
            for p in opts:
                if p.exists():
                    return p
            return None

        chosen_base = None
        chosen_kv = None
        for b in candidates:
            kvp = _pick_kv_dir(b)
            if (kvp / "manifest.json").exists():
                chosen_base = b
                chosen_kv = kvp
                break
        if chosen_base is None:
            # none of the candidates has a manifest; pick the first candidate and keep hints
            chosen_base = candidates[0]
            chosen_kv = _pick_kv_dir(chosen_base)

        args.kv_dir = str(chosen_kv)
        if args.kv_dir_tables is None:
            # tables are optional; only set default if exists
            tbl = Path(str(chosen_base)) / "kvbank_tables"
            tbl_v2 = Path(str(chosen_base)) / "kvbank_tables_v2"
            if (tbl / "manifest.json").exists():
                args.kv_dir_tables = str(tbl)
            elif (tbl_v2 / "manifest.json").exists():
                args.kv_dir_tables = str(tbl_v2)
            else:
                args.kv_dir_tables = str(tbl)  # default path (may not exist)
        if args.blocks_jsonl is None:
            args.blocks_jsonl = str(_pick_blocks_jsonl(Path(str(chosen_base))))

        # Optional: evidence library auto-detect
        if args.kv_dir_evidence is None:
            ev = _pick_evidence_kv_dir(Path(str(chosen_base)))
            if ev is not None:
                args.kv_dir_evidence = str(ev)
        if args.blocks_jsonl_evidence is None:
            bev = _pick_evidence_blocks_jsonl(Path(str(chosen_base)))
            if bev is not None:
                args.blocks_jsonl_evidence = str(bev)

        # Optional: schema library auto-detect (required for schema injection).
        if args.kv_dir_schema is None:
            sk = _pick_schema_kv_dir(Path(str(chosen_base)))
            if sk is not None:
                args.kv_dir_schema = str(sk)
        if args.blocks_jsonl_schema is None:
            bs = _pick_schema_blocks_jsonl(Path(str(chosen_base)))
            if bs is not None:
                args.blocks_jsonl_schema = str(bs)

        print(
            f"[topic_mode] topic={args.topic} topic_work_dir={args.topic_work_dir} chosen_base={chosen_base} "
            f"kv_dir={args.kv_dir} kv_dir_schema={args.kv_dir_schema} kv_dir_evidence={args.kv_dir_evidence} "
            f"kv_dir_tables={args.kv_dir_tables} blocks_jsonl={args.blocks_jsonl} "
            f"blocks_jsonl_schema={args.blocks_jsonl_schema} blocks_jsonl_evidence={args.blocks_jsonl_evidence}",
            flush=True,
        )
    else:
        print(
            f"[config] kv_dir={args.kv_dir} kv_dir_schema={args.kv_dir_schema} kv_dir_evidence={args.kv_dir_evidence} kv_dir_tables={args.kv_dir_tables} "
            f"blocks_jsonl={args.blocks_jsonl} blocks_jsonl_schema={args.blocks_jsonl_schema} blocks_jsonl_evidence={args.blocks_jsonl_evidence}",
            flush=True,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = None
    model = None

    _ACRONYM_PAREN_RE = re.compile(
        r"\b([A-Z][A-Z0-9]{2,11})\b\s*[（(]\s*([^）)\n\r]{3,120}?)\s*[）)]"
    )
    _ACRONYM_IN_QUERY_RE = re.compile(r"\b([A-Z][A-Z0-9]{2,11})\b")
    _HAS_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
    _ZH_TRANSMISSION_INTENT_RE = re.compile(r"(怎么传播|传播途径|传播方式|传播路径|主要传播|媒介|叮咬|蜱|人传人|体液|接触|证据句)")

    def _format_model_prompt(user_prompt: str) -> str:
        if not bool(args.use_chat_template):
            return user_prompt
        if tok is not None and hasattr(tok, "apply_chat_template"):
            try:
                msgs = [{"role": "user", "content": user_prompt}]
                return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)  # type: ignore[attr-defined]
            except Exception:
                return user_prompt
        return user_prompt

    def _build_acronym_lexicon_from_blocks(blocks_jsonl: Path, max_lines: int = 200_000) -> dict[str, str]:
        """
        Build a lightweight acronym->expansion mapping by scanning blocks text for patterns like:
        'SFTSV（Severe Fever with Thrombocytopenia Syndrome Virus）'
        'ABC (some expansion)'
        """
        counts: dict[str, dict[str, int]] = {}
        n = 0
        with blocks_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if n >= max_lines:
                    break
                n += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                txt = str(rec.get("text") or "")
                if not txt:
                    continue
                for m in _ACRONYM_PAREN_RE.finditer(txt):
                    ac = m.group(1)
                    exp = m.group(2).strip()
                    if not exp or len(exp) < 3:
                        continue
                    counts.setdefault(ac, {})
                    counts[ac][exp] = counts[ac].get(exp, 0) + 1

        out: dict[str, str] = {}
        for ac, exp_counts in counts.items():
            # pick the most frequent expansion
            best = max(exp_counts.items(), key=lambda kv: kv[1])[0]
            out[ac] = best
        return out

    def _rewrite_query_for_retrieval(user_prompt: str) -> str:
        mode = str(args.rewrite_query_for_retrieval)
        if mode == "off":
            return user_prompt
        rewritten = user_prompt
        notes: list[str] = []

        def _apply_acronym_lexicon(text: str) -> str:
            if not args.blocks_jsonl:
                return text
            blocks_path = Path(str(args.blocks_jsonl))
            try:
                lex = _build_acronym_lexicon_from_blocks(blocks_path)
            except Exception as e:
                print(f"[query_rewrite] acronym_lexicon disabled: {type(e).__name__}: {e}", flush=True)
                return text
            acronyms = sorted(set(_ACRONYM_IN_QUERY_RE.findall(text)))
            if not acronyms:
                return text
            aug_parts = []
            for ac in acronyms:
                exp = lex.get(ac)
                if exp:
                    aug_parts.append(f"{ac}（{exp}）")
            if not aug_parts:
                return text
            notes.append(f"acronym_lexicon+{len(aug_parts)}")
            return text + "\n\n" + "缩写解释：" + "；".join(aug_parts)

        def _apply_zh_en_intent(text: str) -> str:
            # Only trigger for Chinese queries with transmission-like intent.
            if not _HAS_CJK_RE.search(text or ""):
                return text
            if not _ZH_TRANSMISSION_INTENT_RE.search(text or ""):
                return text
            # Add an English gloss that aligns well with many sentence encoders trained predominantly on EN.
            gloss = (
                "English gloss for retrieval: route of transmission; mode of transmission; transmission route; "
                "tick bite; tick-borne; vector; arthropod vector; Haemaphysalis longicornis; "
                "person-to-person transmission; close contact; blood/body fluids; evidence sentence."
            )
            notes.append("zh_en_intent")
            return text + "\n\n" + gloss

        if mode in {"acronym_lexicon_from_blocks", "auto"}:
            rewritten = _apply_acronym_lexicon(rewritten)
        if mode in {"zh_en_intent", "auto"}:
            rewritten = _apply_zh_en_intent(rewritten)

        if notes and rewritten != user_prompt:
            print(f"[query_rewrite] mode={mode} applied={','.join(notes)}", flush=True)
        return rewritten

    raw_user_prompt = str(args.prompt)
    # In retrieval-only mode, we do not need the model prompt at all.
    model_prompt = raw_user_prompt
    retrieval_query_text = _rewrite_query_for_retrieval(raw_user_prompt)
    if bool(args.print_retrieval_query):
        print(f"[retrieval_query] {retrieval_query_text}", flush=True)

    # Light guardrail: warn if prompt topic keywords and selected library path look mismatched.
    p_low = raw_user_prompt.lower()
    kv_low = str(args.kv_dir).lower() if args.kv_dir else ""
    blocks_low = str(args.blocks_jsonl).lower() if args.blocks_jsonl else ""
    if ("sftsv" in p_low) and (("sars2" in kv_low) or ("sars" in kv_low) or ("sars2" in blocks_low) or ("sars" in blocks_low)):
        print(
            "[warn] prompt mentions 'SFTSV' but kv_dir/blocks_jsonl path looks like SARS2/SARS-CoV-2. "
            "You may be querying the wrong topic library.",
            flush=True,
        )
    if (("sars" in p_low) or ("cov" in p_low)) and (("sftsv" in kv_low) or ("sftsv" in blocks_low)):
        print(
            "[warn] prompt looks SARS-CoV-2 related but kv_dir/blocks_jsonl path looks like SFTSV. "
            "You may be querying the wrong topic library.",
            flush=True,
        )

    # Optional: build an allowlist of block_ids by language from blocks_jsonl (raw + evidence if provided).
    allowed_block_ids = None
    block_text_by_id = None
    allowed_langs = {s.strip() for s in str(args.allowed_langs).split(",") if s.strip()}
    blocks_jsonls = []
    if args.blocks_jsonl:
        blocks_jsonls.append(str(args.blocks_jsonl))
    if args.blocks_jsonl_schema:
        blocks_jsonls.append(str(args.blocks_jsonl_schema))
    if args.blocks_jsonl_evidence:
        blocks_jsonls.append(str(args.blocks_jsonl_evidence))

    if blocks_jsonls and allowed_langs:
        try:
            # Prefer local imports (works for both monorepo and flat layouts)
            try:
                from external_kv_injection.src.cleaning_and_dedupe import detect_lang, quality_score  # type: ignore
            except ModuleNotFoundError:
                from src.cleaning_and_dedupe import detect_lang, quality_score  # type: ignore

            def _nonprintable_ratio(s: str) -> float:
                if not s:
                    return 1.0
                bad = 0
                tot = 0
                for ch in s:
                    if ch in ("\n", "\t", "\r"):
                        continue
                    tot += 1
                    cat = unicodedata.category(ch)
                    # C*: control/surrogate/unassigned/private-use; also treat replacement char as bad
                    if cat and cat[0] == "C":
                        bad += 1
                    elif ch == "\ufffd":
                        bad += 1
                return float(bad / max(tot, 1))

            allowed_block_ids = set()
            # Schema injection requires block_text lookup too (schema texts must be forwarded).
            if bool(args.ground_with_selected_text) or bool(args.use_struct_slots):
                block_text_by_id = {}
            dropped_lang = 0
            dropped_nonprintable = 0
            dropped_quality = 0
            for bp in blocks_jsonls:
                blocks_path = Path(str(bp))
                with blocks_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            continue
                        bid = rec.get("block_id")
                        if not bid:
                            continue
                        txt = str(rec.get("text") or "")
                        lang = detect_lang(txt)
                        if lang not in allowed_langs:
                            dropped_lang += 1
                            continue

                        if bool(args.quality_filter_from_blocks):
                            npr = _nonprintable_ratio(txt)
                            if npr > float(args.max_nonprintable_ratio):
                                dropped_nonprintable += 1
                                continue
                            qs = float(quality_score(txt))
                            if qs < float(args.min_quality_score):
                                dropped_quality += 1
                                continue

                        allowed_block_ids.add(str(bid))
                        if block_text_by_id is not None and str(bid) not in block_text_by_id:
                            block_text_by_id[str(bid)] = txt
            print(
                f"[lang_filter] enabled via blocks_jsonl allow_langs={sorted(allowed_langs)} "
                f"allowed_blocks={len(allowed_block_ids)} files={blocks_jsonls}",
                flush=True,
            )
            if bool(args.quality_filter_from_blocks):
                print(
                    f"[quality_filter] enabled max_nonprintable_ratio={float(args.max_nonprintable_ratio)} "
                    f"min_quality_score={float(args.min_quality_score)} "
                    f"dropped_lang={dropped_lang} dropped_nonprintable={dropped_nonprintable} dropped_quality={dropped_quality}",
                    flush=True,
                )
        except Exception as e:
            allowed_block_ids = None
            block_text_by_id = None
            print(f"[lang_filter] disabled (failed to build allowlist): {type(e).__name__}: {e}", flush=True)

    # Debug preset: make it hard to accidentally introduce noise.
    if bool(args.debug_single_block):
        args.max_blocks_per_step = 1
        args.max_steps = 1
        args.enable_table_routing = False
        args.table_top_k = 0
        print(
            "[debug_single_block] applied: max_blocks_per_step=1 max_steps=1 enable_table_routing=false "
            f"(top_k_blocks={int(args.top_k_blocks)})",
            flush=True,
        )

    # ---------------------------
    # Retrieval routing (MUST FOLLOW):
    # schema (answerability/injection) -> evidence (grounding/citation) -> raw (fallback expansion)
    # ---------------------------
    if not args.kv_dir_schema:
        raise SystemExit("Missing --kv_dir_schema (schema KVBank). Schema KV is the ONLY KV allowed to be injected.")
    if not args.blocks_jsonl_schema:
        raise SystemExit("Missing --blocks_jsonl_schema (required to forward schema texts for KV cache injection).")
    if not args.kv_dir_evidence:
        raise SystemExit("Missing --kv_dir_evidence (evidence bank is required for grounding/citation).")
    if not args.kv_dir:
        raise SystemExit("Missing --kv_dir (raw bank required for evidence->raw fallback expansion).")

    bank_raw = FaissKVBank.load(Path(args.kv_dir))
    bank_schema = FaissKVBank.load(Path(str(args.kv_dir_schema)))
    bank_evidence = FaissKVBank.load(Path(str(args.kv_dir_evidence)))

    print(
        f"[retriever] routing=schema->evidence->raw kv_dir_schema={args.kv_dir_schema} kv_dir_evidence={args.kv_dir_evidence} kv_dir_raw={args.kv_dir}",
        flush=True,
    )

    # Injection retriever: schema-first (answerability). Only schema is eligible for injection.
    retriever = Retriever(bank_schema)
    # Grounding retriever: evidence-first then raw fallback. Retrieval-only; never injected.
    retriever_grounding = EvidenceFirstRetriever(
        evidence_kv_bank=bank_evidence, raw_kv_bank=bank_raw, cfg=EvidenceFirstRetrieverConfig()
    )

    if bool(args.enable_table_routing):
        print("[warn] table routing is ignored under schema-only injection; evidence/raw are retrieval-only.", flush=True)

    enc = HFSentenceEncoder(
        HFSentenceEncoderConfig(model_name_or_path=args.domain_encoder_model, max_length=256, normalize=True)
    )

    def query_embed_fn(text: str):
        return enc.encode(text)[0]

    # Retrieval-only mode: do one search and optionally print top-N candidate full texts.
    if bool(args.retrieval_only):
        qv = query_embed_fn(retrieval_query_text)
        res = retriever.search(qv, top_k=int(args.top_k_blocks), filters=None, query_text=retrieval_query_text)
        print("=== Retrieval Only ===")
        print(f"kv_dir={args.kv_dir}", flush=True)
        print(f"kv_dir_evidence={args.kv_dir_evidence}", flush=True)
        print(f"top_k_blocks={int(args.top_k_blocks)} retrieved_candidates={len(res.items)}", flush=True)
        for i, it in enumerate(res.items, start=1):
            bid = it.meta.get("block_id") or it.meta.get("chunk_id") or it.meta.get("id")
            did = it.meta.get("doc_id")
            src = it.meta.get("source_uri")
            print(
                f"[cand {i:02d}] block_id={bid} score={float(getattr(it, 'score', 0.0)):.4f} doc_id={did} source_uri={src}",
                flush=True,
            )

        if int(args.print_top_candidates_text) > 0:
            if (not args.blocks_jsonl) and (not args.blocks_jsonl_evidence):
                raise SystemExit("--print_top_candidates_text requires --blocks_jsonl or --blocks_jsonl_evidence")
            n = int(args.print_top_candidates_text)
            wanted = [
                str((it.meta.get("block_id") or it.meta.get("chunk_id") or it.meta.get("id"))) for it in res.items[:n]
            ]
            wanted_set = set(wanted)
            # Prefer evidence blocks for printing when available, otherwise fall back to raw blocks.
            blocks_path = Path(str(args.blocks_jsonl_evidence or args.blocks_jsonl))
            found: dict[str, dict] = {}
            with blocks_path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    bid = rec.get("block_id")
                    if bid in wanted_set and str(bid) not in found:
                        found[str(bid)] = {"line_no": int(line_no), **rec}
                    if len(found) >= len(wanted_set):
                        break
            print("=== Top Candidate Texts (from blocks_jsonl) ===")
            print(f"blocks_jsonl={blocks_path}", flush=True)
            for bid in wanted:
                rec = found.get(bid)
                if not rec:
                    print(f"\n--- block_id={bid} (NOT FOUND in blocks_jsonl) ---", flush=True)
                    continue
                txt = str(rec.get("text") or "")
                if int(args.top_candidates_max_chars) > 0:
                    txt = txt[: int(args.top_candidates_max_chars)]
                print(f"\n--- block_id={bid} line_no={rec.get('line_no')} doc_id={rec.get('doc_id')} ---", flush=True)
                print(txt, flush=True)
        return

    # Injection mode: load the base model + tokenizer.
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model_prompt = _format_model_prompt(raw_user_prompt)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device.type == "cuda" else None, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    def _postprocess_answer(text: str, user_prompt: str) -> str:
        """
        Backward-compatible signature.
        IMPORTANT (new design): postprocess MUST NOT decide answer mode (grounded vs ungrounded vs refuse).
        It only performs generic hygiene/dedup/anti-degeneration.
        """
        return postprocess_answer(text or "", user_prompt=user_prompt)

    def _base_llm_answer() -> str:
        inputs0 = tok(model_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out0 = model.generate(
                **inputs0,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                use_cache=True,
                no_repeat_ngram_size=int(args.no_repeat_ngram_size),
            )
        in_len = int(inputs0["input_ids"].shape[1])
        return tok.decode(out0[0][in_len:], skip_special_tokens=True)

    def _base_llm_answer_unguarded(extra_guard: str) -> str:
        """
        UNGROUNDED mode very-light guard:
        - NOT schema, NOT slots; a single sentence to discourage fabricated sources.
        """
        guarded_prompt = (model_prompt or "").rstrip() + "\n\n" + (extra_guard or "").strip()
        inputs0 = tok(guarded_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out0 = model.generate(
                **inputs0,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                use_cache=True,
                no_repeat_ngram_size=int(args.no_repeat_ngram_size),
            )
        in_len = int(inputs0["input_ids"].shape[1])
        return tok.decode(out0[0][in_len:], skip_special_tokens=True)

    _EVIDENCE_REQUEST_RE = re.compile(r"(基于证据|给出证据|证据句|引用|文献|出处|doi|citation|evidence)", re.I)
    _HIGH_RISK_RE = re.compile(
        r"(诊断|确诊|用药剂量|剂量|处方|用药方案|指南|规范|法律|法规|合规|regulation|legal)",
        re.I,
    )

    _INTENT_MECHANISM_RE = re.compile(r"(mechanism|pathogenesis|发病机制|致病机制|机制|免疫机制)", re.I)
    _INTENT_TRANSMISSION_RE = re.compile(r"(transmit|transmission|传播|途径|媒介|vector|tick|蜱)", re.I)
    _INTENT_TREATMENT_RE = re.compile(r"(treat|treatment|治疗|用药|药物|方案|antivir)", re.I)
    _INTENT_EPI_RE = re.compile(r"(epidemiology|流行|发病率|发生率|分布|prevalence|incidence)", re.I)
    _INTENT_OVERVIEW_RE = re.compile(r"(overview|概述|介绍|是什么|定义|what is|简介)", re.I)

    def _infer_intent(*, user_prompt: str) -> str:
        """
        Determine a coarse question intent (stable, minimal).
        This is used ONLY for AnswerMode decision, not for retrieval or slot/schema changes.
        """
        up = str(user_prompt or "")
        # Prefer slot-derived intent if available.
        if _INTENT_TRANSMISSION_RE.search(up):
            return "transmission"
        if _INTENT_MECHANISM_RE.search(up):
            return "mechanism"
        if _INTENT_TREATMENT_RE.search(up):
            return "general_treatment"
        if _INTENT_EPI_RE.search(up):
            return "epidemiology"
        if _INTENT_OVERVIEW_RE.search(up):
            return "overview"
        # Unknown intent (treat as not-answerable without evidence to be safe).
        return "unknown"

    def _decide_answer_mode(*, has_grounding_evidence: bool, user_prompt: str) -> tuple[AnswerMode, dict]:
        """
        Decision rule (must be pre-injection):
        - REFUSE can only be triggered by (question type + user instruction):
            evidence_required_by_policy && !has_grounding_evidence
        - If evidence missing but question is answerable, fall back to UNGROUNDED base LLM
        """
        intent = _infer_intent(user_prompt=user_prompt)
        question_answerable = intent in {"mechanism", "transmission", "general_treatment", "epidemiology", "overview"}

        # Patch 2: REFUSE only by "question type + user instruction"
        user_requests_evidence = bool(_EVIDENCE_REQUEST_RE.search(user_prompt or ""))
        high_risk = bool(_HIGH_RISK_RE.search(user_prompt or ""))
        evidence_required_by_policy = bool(user_requests_evidence or high_risk or (not question_answerable))

        if evidence_required_by_policy and not bool(has_grounding_evidence):
            return AnswerMode.REFUSE, {
                "intent": intent,
                "question_answerable": bool(question_answerable),
                "evidence_required_by_policy": bool(evidence_required_by_policy),
                "user_requests_evidence": bool(user_requests_evidence),
                "high_risk": bool(high_risk),
            }

        # GROUNDED requires evidence hit.
        if bool(has_grounding_evidence):
            return AnswerMode.GROUNDED, {
                "intent": intent,
                "question_answerable": bool(question_answerable),
                "evidence_required_by_policy": bool(evidence_required_by_policy),
                "user_requests_evidence": bool(user_requests_evidence),
                "high_risk": bool(high_risk),
            }

        # Otherwise use base LLM (no injection).
        return AnswerMode.UNGROUNDED, {
            "intent": intent,
            "question_answerable": bool(question_answerable),
            "evidence_required_by_policy": bool(evidence_required_by_policy),
            "user_requests_evidence": bool(user_requests_evidence),
            "high_risk": bool(high_risk),
        }

    base_answer_raw = None
    if (not bool(args.skip_baseline)) or bool(args.print_baseline):
        base_answer_raw = _base_llm_answer()
        print("=== Baseline (no injection) ===")
        print(_postprocess_answer(base_answer_raw, raw_user_prompt))

    # Schema-only injection forwards schema texts and injects schema cache; min_kv_len_to_inject is not used
    # for schema selection. Keep the arg for backward compatibility, but set a safe default if omitted.
    if args.min_kv_len_to_inject is None:
        args.min_kv_len_to_inject = 0
        print(f"[inject] schema-only mode; min_kv_len_to_inject={int(args.min_kv_len_to_inject)} (unused)", flush=True)

    cfg = MultiStepConfig(
        inject_layers=[int(x.strip()) for x in args.layers.split(",") if x.strip() != ""],
        block_tokens=256,
        max_step_tokens=args.max_step_tokens,
        max_total_tokens=args.max_total_tokens,
        max_steps=args.max_steps,
        top_k_blocks=args.top_k_blocks,
        max_blocks_per_step=int(args.max_blocks_per_step),
        min_kv_len_to_inject=int(args.min_kv_len_to_inject),
        use_attention_entropy=bool(args.use_attention_entropy),
        entropy_threshold=float(args.entropy_threshold),
        debug_print_candidates_top_n=int(args.debug_print_candidates),
        schema_required_slots=[s.strip() for s in str(args.schema_required_slots or "").split(",") if s.strip()] or None,
        enable_three_knowledge_layers=True,
        enable_speculative_layer=bool(args.enable_layer2),
        layer1_max_new_tokens=int(args.layer1_max_new_tokens),
        layer2_max_new_tokens=int(args.layer2_max_new_tokens),
        append_evidence_to_prompt=bool(args.append_evidence_to_prompt),
    )
    lookup = None
    if bool(args.ground_with_selected_text) or bool(args.use_struct_slots):
        if not args.blocks_jsonl_schema:
            raise SystemExit("--use_struct_slots requires --blocks_jsonl_schema (schema text lookup).")
        if not isinstance(block_text_by_id, dict):
            raise SystemExit(
                "--ground_with_selected_text/--use_struct_slots requires building block text lookup; "
                "please ensure blocks_jsonl is provided and --allowed_langs is non-empty."
            )
        lookup = lambda bid: block_text_by_id.get(str(bid))  # type: ignore[assignment]

    injector = MultiStepInjector(
        retriever=retriever,
        cfg=cfg,
        allowed_block_ids=allowed_block_ids,
        block_text_lookup=lookup,
        grounding_retriever=retriever_grounding,
    )

    # ---------------------------
    # KVI1.0 single-pass semantic KV injection (no gate / no AnswerMode)
    # ---------------------------
    if bool(args.kvi1_mode):
        qv = query_embed_fn(retrieval_query_text)
        res = retriever.search(qv, top_k=int(args.top_k_blocks), filters=None, query_text=retrieval_query_text)
        if int(args.debug_print_candidates) > 0:
            topn = int(args.debug_print_candidates)
            cands = []
            for it in (res.items or [])[:topn]:
                bid = it.meta.get("block_id") or it.meta.get("chunk_id") or it.meta.get("id")
                cands.append(f"{bid}@{float(getattr(it, 'score', 0.0)):.4f}")
            shown = min(int(topn), len(res.items or []))
            print(f"[retrieval] top{shown}=" + " | ".join(cands), flush=True)

        layer_ids = [int(x.strip()) for x in args.layers.split(",") if x.strip() != ""]
        dtype = next(model.parameters()).dtype
        ext_by_layer = {}
        for li in layer_ids:
            try:
                ext_by_layer[li] = stack_ext_kv_items_by_layer(
                    items=res.items or [],
                    layer_id=int(li),
                    batch_size=1,
                    device=device,
                    dtype=dtype,
                )
            except Exception:
                continue
        if not ext_by_layer:
            print("[warn] KVI1.0 injection skipped: ext_by_layer empty (no compatible KV)", flush=True)
            answer = _base_llm_answer()
            print("=== Answer ===")
            print(_postprocess_answer(answer, raw_user_prompt))
            return

        past_key_values = build_past_key_values_prefix(model=model, ext_kv_by_layer=ext_by_layer)
        if bool(args.kvi1_use_generate):
            print(
                "[warn] kvi1_use_generate: HF generate is not safe with external prefix KV; "
                "using prefix-aware decode with sampling/min_length instead.",
                flush=True,
            )
        answer = MultiStepInjector._greedy_generate_with_past_prefix(
            model=model,
            tokenizer=tok,
            prompt=model_prompt,
            device=device,
            past_key_values=past_key_values,
            max_new_tokens=int(args.max_new_tokens),
            no_repeat_ngram_size=int(args.no_repeat_ngram_size),
            repetition_penalty=float(cfg.repetition_penalty),
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            min_new_tokens=int(args.min_new_tokens),
        )
        print("=== Answer ===")
        print(_postprocess_answer(answer, raw_user_prompt))
        return

    # ---------------------------
    # AnswerMode decision (MUST be BEFORE injection)
    # ---------------------------
    # Evidence hit check: must be evidence-layer text we can actually bind (via block_text_by_id lookup).
    has_grounding_evidence = False
    try:
        qv = query_embed_fn(retrieval_query_text)
        ev_items, _ev_dbg = bank_evidence.search(qv, top_k=8, filters=None)
        if isinstance(block_text_by_id, dict):
            for it in ev_items:
                bid = (it.meta or {}).get("block_id") or (it.meta or {}).get("chunk_id") or (it.meta or {}).get("id")
                if not bid:
                    continue
                t = block_text_by_id.get(str(bid))
                if isinstance(t, str) and t.strip() and re.search(r"[A-Za-z\u4e00-\u9fff]", t) and len(t.strip()) >= 16:
                    has_grounding_evidence = True
                    break
    except Exception:
        has_grounding_evidence = False

    mode, mode_dbg = _decide_answer_mode(has_grounding_evidence=bool(has_grounding_evidence), user_prompt=raw_user_prompt)
    print(
        f"[mode] AnswerMode={mode} has_grounding_evidence={bool(has_grounding_evidence)} mode_dbg={mode_dbg}",
        flush=True,
    )

    answer = ""
    dbg = []
    if mode == AnswerMode.REFUSE:
        # Deterministic refusal/fallback when evidence is explicitly required or task is high-risk.
        answer = "未检索到可用证据，无法按要求回答。"
        dbg = []
    elif mode == AnswerMode.UNGROUNDED:
        # Base LLM mode (no injection).
        print(
            f"[event] mode_fallback=UNGROUNDED reason=policy_decision mode_dbg={mode_dbg}",
            flush=True,
        )
        # Ensure UNGROUNDED is exactly the same as baseline generation (no prompt rewrite / no extra guard injected).
        if base_answer_raw is None:
            base_answer_raw = _base_llm_answer()
        answer = base_answer_raw
        dbg = []
    else:
        # GROUNDED: schema injection + evidence appended grounding
        answer, dbg = injector.run(
            model=model,
            tokenizer=tok,
            prompt=model_prompt,
            query_text=retrieval_query_text,
            device=device,
            max_new_tokens=args.max_new_tokens,
            query_embed_fn=query_embed_fn,
            no_repeat_ngram_size=int(args.no_repeat_ngram_size),
            ground_with_selected_text=bool(args.ground_with_selected_text),
            grounding_instructions=str(args.grounding_instructions or ""),
            use_struct_slots=bool(args.use_struct_slots),
            use_chat_template=bool(args.use_chat_template),
        )

    print("=== Step Debug ===")
    if not dbg:
        print("[debug] no steps executed (no blocks injected).", flush=True)
    else:
        for d in dbg:
            print(d)
    print("=== Answer ===")
    final_out = _postprocess_answer(answer, raw_user_prompt)
    # Ensure UNGROUNDED output is identical to baseline answer text. Any notice goes to logs only.
    if mode == AnswerMode.UNGROUNDED:
        print(
            "[notice] AnswerMode=UNGROUNDED: response is from base LLM (no injection), not guaranteed to be fully evidence-grounded.",
            flush=True,
        )
    # Guard against "empty answer" degeneration (common when the model emits EOS immediately).
    if not (final_out or "").strip():
        print(
            f"[warn] empty answer after postprocess (mode={mode}). raw_len={len(str(answer or ''))} "
            f"post_len={len(str(final_out or ''))}. Falling back to UNGROUNDED base LLM.",
            flush=True,
        )
        print(
            f"[event] mode_fallback=UNGROUNDED reason=empty_after_postprocess prev_mode={mode} mode_dbg={mode_dbg}",
            flush=True,
        )
        if base_answer_raw is None:
            base_answer_raw = _base_llm_answer()
        fallback_raw = base_answer_raw
        # For debugging, print a short excerpt of the raw GROUNDED output that got wiped.
        wiped_excerpt = re.sub(r"\s+", " ", str(answer or "")).strip()[:400]
        if wiped_excerpt:
            print(f"[debug] grounded_raw_excerpt={wiped_excerpt}", flush=True)
        final_out = _postprocess_answer(fallback_raw, raw_user_prompt)
        print(
            "[notice] Fallback to base LLM due to empty grounded output after postprocess (answer text unchanged vs baseline).",
            flush=True,
        )
    print(final_out)

    # Optional: print selected block snippets for debugging retrieval quality.
    if args.blocks_jsonl or args.blocks_jsonl_evidence:
        wanted = set()
        for d in dbg:
            for bid in getattr(d, "selected_block_ids", []) or []:
                wanted.add(str(bid))
        if not wanted:
            print("=== Selected Block Lookup (blocks_jsonl path + line_no) ===")
            print(
                f"blocks_jsonl={args.blocks_jsonl} blocks_jsonl_schema={args.blocks_jsonl_schema} blocks_jsonl_evidence={args.blocks_jsonl_evidence}",
                flush=True,
            )
            print("[debug] no selected_block_ids (no blocks injected or no steps executed).", flush=True)
            print("=== Selected Block Snippets ===")
            # Still useful: show top evidence hits so we can verify retrieval isn't degenerate.
            try:
                if bank_evidence is not None and isinstance(block_text_by_id, dict):
                    qv = query_embed_fn(retrieval_query_text)
                    ev_items, _ev_dbg = bank_evidence.search(qv, top_k=8, filters=None)
                    shown = 0
                    for it in ev_items:
                        bid = (it.meta or {}).get("block_id") or (it.meta or {}).get("chunk_id") or (it.meta or {}).get("id")
                        if not bid:
                            continue
                        txt = block_text_by_id.get(str(bid), "")
                        if not (isinstance(txt, str) and txt.strip()):
                            continue
                        snippet = re.sub(r"\s+", " ", txt).strip()[:320]
                        print(f"\n--- evidence_hit block_id={bid} score={float(getattr(it, 'score', 0.0)):.4f} ---", flush=True)
                        print(f"text_snippet={snippet}", flush=True)
                        shown += 1
                        if shown >= 5:
                            break
                    if shown == 0:
                        print("[debug] (empty; no readable evidence hits found)", flush=True)
                else:
                    print("[debug] (empty; evidence bank/text lookup not available)", flush=True)
            except Exception as e:
                print(f"[debug] failed to print evidence hits: {e}", flush=True)
            return
        if wanted:
            found = {}
            try:
                scan_paths = []
                if args.blocks_jsonl:
                    scan_paths.append(Path(str(args.blocks_jsonl)))
                # Also scan schema blocks for selected schema ids (otherwise they will show as "not found").
                if args.blocks_jsonl_schema:
                    scan_paths.append(Path(str(args.blocks_jsonl_schema)))
                if args.blocks_jsonl_evidence:
                    scan_paths.append(Path(str(args.blocks_jsonl_evidence)))
                for blocks_path in scan_paths:
                    if len(found) >= len(wanted):
                        break
                    with blocks_path.open("r", encoding="utf-8") as f:
                        for line_no, line in enumerate(f, start=1):
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                            except Exception:
                                continue
                            bid = rec.get("block_id")
                            if bid in wanted and bid not in found:
                                txt = str(rec.get("text") or "")
                                snippet = re.sub(r"\s+", " ", txt).strip()[:700]
                                found[bid] = {
                                    "blocks_jsonl": str(blocks_path),
                                    "line_no": int(line_no),
                                    "doc_id": rec.get("doc_id"),
                                    "token_count": rec.get("token_count"),
                                    "source_uri": rec.get("source_uri"),
                                    "snippet": snippet,
                                    "text_full": txt,
                                    "metadata": rec.get("metadata"),
                                }
                            if len(found) >= len(wanted):
                                break
            except Exception as e:
                print(f"[debug] failed to read blocks_jsonl: {e}", flush=True)
                return

            # Make it explicit whether selected block_ids were found in this blocks.jsonl file.
            print("=== Selected Block Lookup (blocks_jsonl path + line_no) ===")
            print(
                f"blocks_jsonl={args.blocks_jsonl} blocks_jsonl_evidence={args.blocks_jsonl_evidence}",
                flush=True,
            )
            for bid in sorted(wanted):
                info = found.get(bid)
                if info is None:
                    print(f"- block_id={bid} found=false", flush=True)
                    continue
                print(
                    f"- block_id={bid} found=true line_no={info.get('line_no')} "
                    f"doc_id={info.get('doc_id')} source_uri={info.get('source_uri')}",
                    flush=True,
                )

            print("=== Selected Block Snippets ===")
            for bid in sorted(wanted):
                info = found.get(bid)
                if info is None:
                    print(f"[debug] block_id not found in blocks_jsonl: {bid}", flush=True)
                    continue
                print(f"\n--- block_id={bid} ---", flush=True)
                print(
                    f"doc_id={info.get('doc_id')} token_count={info.get('token_count')} "
                    f"blocks_jsonl={info.get('blocks_jsonl')} line_no={info.get('line_no')}",
                    flush=True,
                )
                md = info.get("metadata") or {}
                if isinstance(md, dict):
                    # Common, high-signal provenance fields
                    # (source_uri is added by our pipeline; may be missing for older blocks.jsonl)
                    src = md.get("source_uri") or md.get("source_path") or md.get("pdf_path")
                    if src:
                        print(f"source_uri={src}", flush=True)
                # Also show top-level provenance when present
                if isinstance(info.get("source_uri"), str):
                    print(f"source_uri={info.get('source_uri')}", flush=True)
                print(f"text_snippet={info.get('snippet')}", flush=True)
                if bool(args.print_selected_block_text):
                    full = str(info.get("text_full") or "")
                    if isinstance(args.selected_block_max_chars, int) and int(args.selected_block_max_chars) > 0:
                        full = full[: int(args.selected_block_max_chars)]
                    print("----- block_text_begin -----", flush=True)
                    print(full, flush=True)
                    print("----- block_text_end -----", flush=True)
                if bool(args.print_selected_block_token_ids):
                    ids = tok(str(info.get("text_full") or ""), add_special_tokens=False)["input_ids"]
                    if isinstance(args.selected_block_max_token_ids, int) and int(args.selected_block_max_token_ids) > 0:
                        ids = ids[: int(args.selected_block_max_token_ids)]
                    print("----- block_token_ids_begin -----", flush=True)
                    print(ids, flush=True)
                    print("----- block_token_ids_end -----", flush=True)


if __name__ == "__main__":
    main()


