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

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.kv_bank import FaissKVBank  # type: ignore
    from external_kv_injection.src.retriever import Retriever, RoutedRetriever, RoutedRetrieverConfig  # type: ignore
    from external_kv_injection.src.runtime.multistep_injector import MultiStepConfig, MultiStepInjector  # type: ignore
    from external_kv_injection.src.encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig  # type: ignore
except ModuleNotFoundError:
    from src.kv_bank import FaissKVBank  # type: ignore
    from src.retriever import Retriever, RoutedRetriever, RoutedRetrieverConfig  # type: ignore
    from src.runtime.multistep_injector import MultiStepConfig, MultiStepInjector  # type: ignore
    from src.encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig  # type: ignore


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--kv_dir", default=None, help="Path to KVBank. If omitted, use --topic + --topic_work_dir.")
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
        help="Optional blocks.jsonl path (e.g. $WORK_DIR/blocks.v2.jsonl). If provided, print text snippets for selected blocks.",
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
        "--print_retrieval_query",
        action="store_true",
        help="If set, print the exact retrieval query text (after optional rewrite).",
    )
    p.add_argument(
        "--ground_with_selected_text",
        action="store_true",
        help="If set, append a few evidence sentences extracted from the selected blocks to the prompt during final decode. "
        "This greatly reduces hallucinations like 'mosquito transmission' when selected blocks clearly say 'tick bites'. "
        "Requires --blocks_jsonl.",
    )
    p.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=6,
        help="No-repeat ngram constraint during final decode to reduce repetitive filler (0 disables).",
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
            opts = [base_dir / "blocks.jsonl", base_dir / "blocks.v2.jsonl"]
            for p in opts:
                if p.exists():
                    return p
            return base_dir / "blocks.jsonl"

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

        print(
            f"[topic_mode] topic={args.topic} topic_work_dir={args.topic_work_dir} chosen_base={chosen_base} "
            f"kv_dir={args.kv_dir} kv_dir_tables={args.kv_dir_tables} blocks_jsonl={args.blocks_jsonl}",
            flush=True,
        )
    else:
        print(f"[config] kv_dir={args.kv_dir} kv_dir_tables={args.kv_dir_tables} blocks_jsonl={args.blocks_jsonl}", flush=True)

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

    def _strip_chat_artifacts(text: str) -> str:
        # Truncate if the model starts a new turn (common in chatty base models).
        # Keep it conservative: only cut on newline + Human/User markers.
        m = re.search(r"\n\s*(Human|User)\s*:\s*", text)
        if m:
            text = text[: m.start()]
        return text.strip()

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

    # Optional: build an allowlist of block_ids by language from blocks_jsonl.
    allowed_block_ids = None
    block_text_by_id = None
    allowed_langs = {s.strip() for s in str(args.allowed_langs).split(",") if s.strip()}
    if args.blocks_jsonl and allowed_langs:
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
            if bool(args.ground_with_selected_text):
                block_text_by_id = {}
            dropped_lang = 0
            dropped_nonprintable = 0
            dropped_quality = 0
            blocks_path = Path(str(args.blocks_jsonl))
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
                    if block_text_by_id is not None:
                        block_text_by_id[str(bid)] = txt
            print(
                f"[lang_filter] enabled via blocks_jsonl allow_langs={sorted(allowed_langs)} "
                f"allowed_blocks={len(allowed_block_ids)} file={blocks_path}",
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

    bank = FaissKVBank.load(Path(args.kv_dir))
    if bool(args.enable_table_routing) and args.kv_dir_tables and int(args.table_top_k) > 0:
        table_bank = FaissKVBank.load(Path(args.kv_dir_tables))
        retriever = RoutedRetriever(
            kv_bank=bank,
            table_kv_bank=table_bank,
            cfg=RoutedRetrieverConfig(enable_table_routing=True, table_top_k=int(args.table_top_k)),
        )
    else:
        retriever = Retriever(bank)

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
            if not args.blocks_jsonl:
                raise SystemExit("--print_top_candidates_text requires --blocks_jsonl")
            n = int(args.print_top_candidates_text)
            wanted = [
                str((it.meta.get("block_id") or it.meta.get("chunk_id") or it.meta.get("id"))) for it in res.items[:n]
            ]
            wanted_set = set(wanted)
            blocks_path = Path(str(args.blocks_jsonl))
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

    if (not bool(args.skip_baseline)) or bool(args.print_baseline):
        inputs0 = tok(model_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out0 = model.generate(**inputs0, max_new_tokens=int(args.max_new_tokens), do_sample=False, use_cache=True)
        print("=== Baseline (no injection) ===")
        # Print only newly generated tokens (exclude prompt) for easier A/B compare.
        in_len = int(inputs0["input_ids"].shape[1])
        print(_strip_chat_artifacts(tok.decode(out0[0][in_len:], skip_special_tokens=True)))

    cfg = MultiStepConfig(
        inject_layers=[int(x.strip()) for x in args.layers.split(",") if x.strip() != ""],
        block_tokens=256,
        max_step_tokens=args.max_step_tokens,
        max_total_tokens=args.max_total_tokens,
        max_steps=args.max_steps,
        top_k_blocks=args.top_k_blocks,
        max_blocks_per_step=int(args.max_blocks_per_step),
        use_attention_entropy=bool(args.use_attention_entropy),
        entropy_threshold=float(args.entropy_threshold),
        debug_print_candidates_top_n=int(args.debug_print_candidates),
    )
    lookup = None
    if bool(args.ground_with_selected_text):
        if not args.blocks_jsonl:
            raise SystemExit("--ground_with_selected_text requires --blocks_jsonl")
        if not isinstance(block_text_by_id, dict):
            raise SystemExit(
                "--ground_with_selected_text requires building block text lookup; "
                "please ensure --blocks_jsonl is provided and --allowed_langs is non-empty."
            )
        lookup = lambda bid: block_text_by_id.get(str(bid))  # type: ignore[assignment]

    injector = MultiStepInjector(
        retriever=retriever, cfg=cfg, allowed_block_ids=allowed_block_ids, block_text_lookup=lookup
    )
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
    )

    print("=== Step Debug ===")
    if not dbg:
        print("[debug] no steps executed (no blocks injected).", flush=True)
    else:
        for d in dbg:
            print(d)
    print("=== Answer ===")
    print(_strip_chat_artifacts(answer))

    # Optional: print selected block snippets for debugging retrieval quality.
    if args.blocks_jsonl:
        wanted = set()
        for d in dbg:
            for bid in getattr(d, "selected_block_ids", []) or []:
                wanted.add(str(bid))
        if wanted:
            blocks_path = Path(str(args.blocks_jsonl))
            found = {}
            try:
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
                print(f"[debug] failed to read blocks_jsonl={blocks_path}: {e}", flush=True)
                return

            # Make it explicit whether selected block_ids were found in this blocks.jsonl file.
            print("=== Selected Block Lookup (blocks_jsonl path + line_no) ===")
            print(f"blocks_jsonl={blocks_path}", flush=True)
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


