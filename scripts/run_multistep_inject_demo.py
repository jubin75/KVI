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
        "--use_chat_template",
        action="store_true",
        help="If tokenizer supports it, format the model prompt using tokenizer.apply_chat_template. "
        "This often reduces chat transcript artifacts like 'Human:' in outputs.",
    )
    p.add_argument(
        "--rewrite_query_for_retrieval",
        choices=["off", "acronym_lexicon_from_blocks"],
        default="acronym_lexicon_from_blocks",
        help="Rewrite ONLY the retrieval query (not the model prompt). "
        "This improves acronym disambiguation without requiring users to type expansions.",
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
    p.add_argument("--print_baseline", action="store_true", help="Print baseline answer without KV injection for A/B compare.")
    p.add_argument("--use_attention_entropy", action="store_true", help="Enable external KV attention entropy stopping signal")
    p.add_argument("--entropy_threshold", type=float, default=0.35, help="Normalized entropy threshold in [0,1]")
    args = p.parse_args()

    # 专题库 mode: resolve paths from topic_work_dir/topic
    if args.kv_dir is None:
        if not args.topic or not args.topic_work_dir:
            raise SystemExit("Missing --kv_dir. Either pass --kv_dir or use --topic + --topic_work_dir.")
        base = Path(str(args.topic_work_dir)) / str(args.topic)
        args.kv_dir = str(base / "kvbank_blocks")
        if args.kv_dir_tables is None:
            args.kv_dir_tables = str(base / "kvbank_tables")
        if args.blocks_jsonl is None:
            args.blocks_jsonl = str(base / "blocks.jsonl")
        print(
            f"[topic_mode] topic={args.topic} topic_work_dir={args.topic_work_dir} "
            f"kv_dir={args.kv_dir} kv_dir_tables={args.kv_dir_tables} blocks_jsonl={args.blocks_jsonl}",
            flush=True,
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if device.type == "cuda" else None, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    _ACRONYM_PAREN_RE = re.compile(
        r"\b([A-Z][A-Z0-9]{2,11})\b\s*[（(]\s*([^）)\n\r]{3,120}?)\s*[）)]"
    )
    _ACRONYM_IN_QUERY_RE = re.compile(r"\b([A-Z][A-Z0-9]{2,11})\b")

    def _format_model_prompt(user_prompt: str) -> str:
        if not bool(args.use_chat_template):
            return user_prompt
        if hasattr(tok, "apply_chat_template"):
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
        if str(args.rewrite_query_for_retrieval) == "off":
            return user_prompt
        if str(args.rewrite_query_for_retrieval) == "acronym_lexicon_from_blocks":
            if not args.blocks_jsonl:
                return user_prompt
            blocks_path = Path(str(args.blocks_jsonl))
            try:
                lex = _build_acronym_lexicon_from_blocks(blocks_path)
            except Exception as e:
                print(f"[query_rewrite] disabled (failed to build lexicon): {type(e).__name__}: {e}", flush=True)
                return user_prompt
            acronyms = sorted(set(_ACRONYM_IN_QUERY_RE.findall(user_prompt)))
            if not acronyms:
                return user_prompt
            aug_parts = []
            for ac in acronyms:
                exp = lex.get(ac)
                if exp:
                    aug_parts.append(f"{ac}（{exp}）")
            if not aug_parts:
                return user_prompt
            rewritten = user_prompt + "\n\n" + "缩写解释：" + "；".join(aug_parts)
            print(
                f"[query_rewrite] mode=acronym_lexicon_from_blocks acronyms={acronyms} "
                f"added={len(aug_parts)} blocks_jsonl={blocks_path}",
                flush=True,
            )
            return rewritten
        return user_prompt

    raw_user_prompt = str(args.prompt)
    model_prompt = _format_model_prompt(raw_user_prompt)
    retrieval_query_text = _rewrite_query_for_retrieval(raw_user_prompt)

    # Optional: build an allowlist of block_ids by language from blocks_jsonl.
    allowed_block_ids = None
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
            print(f"[lang_filter] disabled (failed to build allowlist): {type(e).__name__}: {e}", flush=True)

    if bool(args.print_baseline):
        inputs0 = tok(model_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out0 = model.generate(**inputs0, max_new_tokens=int(args.max_new_tokens), do_sample=False, use_cache=True)
        print("=== Baseline (no injection) ===")
        # Print only newly generated tokens (exclude prompt) for easier A/B compare.
        in_len = int(inputs0["input_ids"].shape[1])
        print(_strip_chat_artifacts(tok.decode(out0[0][in_len:], skip_special_tokens=True)))

    bank = FaissKVBank.load(Path(args.kv_dir))
    if bool(args.enable_table_routing) and args.kv_dir_tables:
        table_bank = FaissKVBank.load(Path(args.kv_dir_tables))
        retriever = RoutedRetriever(
            kv_bank=bank,
            table_kv_bank=table_bank,
            cfg=RoutedRetrieverConfig(enable_table_routing=True, table_top_k=int(args.table_top_k)),
        )
    else:
        retriever = Retriever(bank)

    enc = HFSentenceEncoder(HFSentenceEncoderConfig(model_name_or_path=args.domain_encoder_model, max_length=256, normalize=True))

    def query_embed_fn(text: str):
        return enc.encode(text)[0]

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
    )
    injector = MultiStepInjector(retriever=retriever, cfg=cfg, allowed_block_ids=allowed_block_ids)
    answer, dbg = injector.run(
        model=model,
        tokenizer=tok,
        prompt=model_prompt,
        query_text=retrieval_query_text,
        device=device,
        max_new_tokens=args.max_new_tokens,
        query_embed_fn=query_embed_fn,
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
                    for line in f:
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
                                "doc_id": rec.get("doc_id"),
                                "token_count": rec.get("token_count"),
                                "snippet": snippet,
                                "metadata": rec.get("metadata"),
                            }
                        if len(found) >= len(wanted):
                            break
            except Exception as e:
                print(f"[debug] failed to read blocks_jsonl={blocks_path}: {e}", flush=True)
                return

            print("=== Selected Block Snippets ===")
            for bid in sorted(wanted):
                info = found.get(bid)
                if info is None:
                    print(f"[debug] block_id not found in blocks_jsonl: {bid}", flush=True)
                    continue
                print(f"\n--- block_id={bid} ---", flush=True)
                print(f"doc_id={info.get('doc_id')} token_count={info.get('token_count')}", flush=True)
                print(f"text_snippet={info.get('snippet')}", flush=True)


if __name__ == "__main__":
    main()


