"""
Doc-level topic filtering (DeepSeek): build a topic-specific PDF subset dir.

Why:
- External KV injection is high-gain/low-tolerance. If the corpus is off-topic, injection amplifies errors.
- For "专题库" product mode, we curate PDFs per topic (e.g., SFTSV vs SARS‑CoV‑2).

Output:
- results.jsonl: per-PDF KEEP/DROP/UNCERTAIN decisions
- out_pdf_dir/: symlinks (or copies) of kept PDFs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.cleaning_and_dedupe import normalize_text  # type: ignore
    from external_kv_injection.src.llm_filter.deepseek_client import DeepSeekClient, DeepSeekClientConfig  # type: ignore
    from external_kv_injection.src.pdf_ingestion import ingest_pdf  # type: ignore
except ModuleNotFoundError:
    from src.cleaning_and_dedupe import normalize_text  # type: ignore
    from src.llm_filter.deepseek_client import DeepSeekClient, DeepSeekClientConfig  # type: ignore
    from src.pdf_ingestion import ingest_pdf  # type: ignore


def _iter_pdfs(pdf_dir: Path) -> List[Path]:
    return sorted([p for p in pdf_dir.rglob("*.pdf") if p.is_file()])


def _safe_parse_json(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    # extract first JSON object
    import re

    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _link_or_copy(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    if mode == "copy":
        import shutil

        shutil.copy2(src, dst)
        return
    raise ValueError(f"Unknown mode: {mode}")


def _load_config(path: Path) -> Dict[str, Any]:
    # Be tolerant to repo layout differences:
    # - monorepo layout: <repo>/external_kv_injection/config/...
    # - flat layout (KVI root): <repo>/config/...
    #
    # Users often copy-paste commands between environments; if the provided path
    # doesn't exist, try to resolve it against repo root and/or strip the
    # "external_kv_injection/" prefix.
    p = path
    if not p.is_absolute():
        # first try relative to cwd, then to repo root
        p1 = (Path.cwd() / p).resolve()
        if p1.exists():
            p = p1
        else:
            p2 = (_REPO_ROOT / p).resolve()
            if p2.exists():
                p = p2
    if not p.exists():
        parts = list(p.parts)
        if "external_kv_injection" in parts:
            i = parts.index("external_kv_injection")
            alt_rel = Path(*parts[i + 1 :])  # strip prefix
            alt = (_REPO_ROOT / alt_rel).resolve()
            if alt.exists():
                p = alt
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("config must be a JSON object")
    return obj


def _extract_abstract(full_text: str) -> Optional[str]:
    """
    Heuristic abstract extraction.
    We try to locate 'Abstract' / '摘要' and take text until the next major section heading.
    """
    import re

    t = full_text.replace("\r", "\n")
    if not t.strip():
        return None

    m_abs = re.search(r"(?im)^\s*abstract\s*[:\n]\s*", t)
    m_zh = re.search(r"(?im)^\s*摘要\s*[:\n]\s*", t)
    starts = [m for m in [m_abs, m_zh] if m is not None]
    if not starts:
        return None
    start = min(starts, key=lambda m: m.start()).end()

    end_pat = re.compile(
        r"(?im)^\s*(introduction|background|methods?|materials\s+and\s+methods?|results?|conclusion|"
        r"keywords|keyword|关键词|关键字|引言|背景|方法|材料与方法|结果|结论)\s*[:\n]\s*"
    )
    m_end = end_pat.search(t, pos=start)
    end = m_end.start() if m_end else len(t)
    abs_text = t[start:end].strip()
    if len(abs_text) < 200:
        return None
    return abs_text


def _doc_filter_prompts(goal: str, *, file_name: str, abstract_text: str) -> Tuple[str, str]:
    system = "你是医学文献的专题筛选器。你的任务是判断一篇论文是否适合进入指定专题知识库。只输出严格 JSON。"
    user = f"""判断下面文献是否应进入专题库。只输出 JSON。

专题库目标（用户输入，必须严格对齐）：
{goal}

规则：
1) 只基于“摘要”判断主题相关性；只有当摘要明确与目标强相关时才 KEEP。
2) 如果只是顺带提到/对照提到/引用相关术语，但主题并非该目标，则 DROP。
3) 如果摘要信息不足以判断，输出 UNCERTAIN（并在 reason 里说明缺少什么）。

输出 JSON schema：
{{"label":"KEEP|DROP|UNCERTAIN","reason":"一句话原因"}}

file_name: {file_name}

摘要（完整）：
<<<
{abstract_text}
>>>
"""
    return system, user


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Topic config JSON (goal + paths + DeepSeek settings).")
    p.add_argument("--goal", default=None, help="Topic goal. If --config is provided, this overrides config.goal.")
    p.add_argument("--pdf_dir", default=None, help="Source PDF dir. If --config provided, defaults to config.source_pdf_dir.")
    p.add_argument("--out_pdf_dir", default=None, help="Output dir for kept PDFs. If --config provided, defaults to config.out_pdf_dir.")
    p.add_argument("--results_jsonl", default=None, help="Decision log path (jsonl). If --config provided, defaults to config.results_jsonl.")
    p.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    p.add_argument("--overwrite_results", action="store_true", help="If set, overwrite results_jsonl instead of appending.")
    p.add_argument("--ocr", default="off", choices=["off", "auto", "on"])
    p.add_argument("--max_pages", type=int, default=2, help="Scan first N pages to find the full abstract.")
    p.add_argument("--max_abstract_chars", type=int, default=12000, help="Max abstract chars sent to DeepSeek.")
    p.add_argument("--max_pdfs", type=int, default=0, help="0=no limit; for quick tests set e.g. 200.")
    p.add_argument("--deepseek_base_url", default="https://api.deepseek.com")
    p.add_argument("--deepseek_model", default="deepseek-chat")
    p.add_argument("--deepseek_api_key_env", default="DEEPSEEK_API_KEY")
    p.add_argument("--strict_drop_uncertain", action="store_true", help="If set, treat UNCERTAIN as DROP.")
    p.add_argument("--dedupe_by_basename", action="store_true", help="If set, skip duplicate PDFs with the same file name.")
    args = p.parse_args()

    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = _load_config(Path(str(args.config)))

    # override tunables from config
    if cfg.get("ocr") is not None:
        args.ocr = str(cfg.get("ocr"))
    args.deepseek_base_url = str(cfg.get("deepseek_base_url", args.deepseek_base_url))
    args.deepseek_model = str(cfg.get("deepseek_model", args.deepseek_model))
    args.deepseek_api_key_env = str(cfg.get("deepseek_api_key_env", args.deepseek_api_key_env))
    if bool(cfg.get("strict_drop_uncertain", False)):
        args.strict_drop_uncertain = True
    df_cfg = cfg.get("doc_filter") if isinstance(cfg.get("doc_filter"), dict) else {}
    if isinstance(df_cfg, dict):
        args.mode = str(df_cfg.get("mode", args.mode))
        args.max_pages = int(df_cfg.get("max_pages_for_abstract", args.max_pages))
        args.max_abstract_chars = int(df_cfg.get("max_abstract_chars", args.max_abstract_chars))
        args.dedupe_by_basename = bool(df_cfg.get("dedupe_by_basename", bool(args.dedupe_by_basename)))
        if bool(df_cfg.get("overwrite_results", False)):
            args.overwrite_results = True

    goal = str(args.goal or cfg.get("goal") or "").strip()
    if not goal:
        raise SystemExit("Missing topic goal. Provide --goal or config.goal")

    src_dir = Path(str(args.pdf_dir or cfg.get("source_pdf_dir") or ""))
    out_dir = Path(str(args.out_pdf_dir or cfg.get("out_pdf_dir") or ""))
    if not str(src_dir):
        raise SystemExit("Missing pdf_dir. Provide --pdf_dir or config.source_pdf_dir")
    if not str(out_dir):
        raise SystemExit("Missing out_pdf_dir. Provide --out_pdf_dir or config.out_pdf_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(str(args.results_jsonl or cfg.get("results_jsonl") or (out_dir / "results.jsonl")))

    client = DeepSeekClient(
        DeepSeekClientConfig(
            base_url=str(args.deepseek_base_url),
            model=str(args.deepseek_model),
            api_key_env=str(args.deepseek_api_key_env),
        )
    )

    pdfs = _iter_pdfs(src_dir)
    if int(args.max_pdfs) > 0:
        pdfs = pdfs[: int(args.max_pdfs)]

    print(
        f"[doc_filter] pdfs={len(pdfs)} mode={args.mode} ocr={args.ocr} max_pages={int(args.max_pages)} "
        f"max_abstract_chars={int(args.max_abstract_chars)} strict_drop_uncertain={bool(args.strict_drop_uncertain)} "
        f"dedupe_by_basename={bool(args.dedupe_by_basename)} overwrite_results={bool(args.overwrite_results)} "
        f"out_pdf_dir={out_dir} results={results_path}",
        flush=True,
    )

    kept = 0
    dropped = 0
    uncertain = 0
    errors = 0
    dup_skipped = 0
    seen_names: set[str] = set()

    write_mode = "w" if bool(args.overwrite_results) else "a"
    with results_path.open(write_mode, encoding="utf-8") as fout:
        for i, pdf in enumerate(pdfs, start=1):
            try:
                if bool(args.dedupe_by_basename):
                    name = pdf.name
                    if name in seen_names:
                        dup_skipped += 1
                        rec = {"pdf": str(pdf), "file_name": name, "label": "DUPLICATE", "decision": "SKIP"}
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fout.flush()
                        continue
                    seen_names.add(name)

                doc = ingest_pdf(pdf, ocr=str(args.ocr), extract_tables=False)
                parts: List[str] = []
                for page in doc.pages[: int(args.max_pages)]:
                    parts.append(page.text or "")
                full = normalize_text("\n".join(parts))
                abs_text = _extract_abstract(full)
                if not abs_text:
                    label = "UNCERTAIN"
                    reason = "abstract_not_found"
                else:
                    abs_text = abs_text[: int(args.max_abstract_chars)]
                    system, user = _doc_filter_prompts(goal, file_name=pdf.name, abstract_text=abs_text)
                    raw = client.chat(system=system, user=user, temperature=0.0)
                    obj = _safe_parse_json(raw) or {"label": "UNCERTAIN", "reason": "parse_failed"}
                    label = str(obj.get("label", "UNCERTAIN")).upper()
                    if label not in {"KEEP", "DROP", "UNCERTAIN"}:
                        label = "UNCERTAIN"
                    reason = str(obj.get("reason", ""))[:300]

                decision = label
                if label == "UNCERTAIN" and not bool(args.strict_drop_uncertain):
                    decision = "KEEP"

                rec = {
                    "goal": goal[:500],
                    "pdf": str(pdf),
                    "file_name": pdf.name,
                    "label": label,
                    "decision": decision,
                    "reason": reason,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

                if decision == "KEEP":
                    kept += 1
                    _link_or_copy(pdf, out_dir / pdf.name, mode=str(args.mode))
                elif label == "DROP":
                    dropped += 1
                else:
                    uncertain += 1

                if i == 1 or i % 10 == 0 or i == len(pdfs):
                    print(
                        f"[doc_filter] {i}/{len(pdfs)} kept={kept} dropped={dropped} uncertain={uncertain} "
                        f"errors={errors} dup_skipped={dup_skipped} last={pdf.name}",
                        flush=True,
                    )
            except Exception as e:
                errors += 1
                rec = {
                    "pdf": str(pdf),
                    "file_name": pdf.name,
                    "error": f"{type(e).__name__}: {e}",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                if i == 1 or i % 10 == 0 or i == len(pdfs):
                    print(
                        f"[doc_filter] {i}/{len(pdfs)} kept={kept} dropped={dropped} uncertain={uncertain} "
                        f"errors={errors} dup_skipped={dup_skipped} last_error={type(e).__name__}",
                        flush=True,
                    )

    print(
        f"[doc_filter] done kept={kept} dropped={dropped} uncertain={uncertain} errors={errors} "
        f"dup_skipped={dup_skipped} out_pdf_dir={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()


