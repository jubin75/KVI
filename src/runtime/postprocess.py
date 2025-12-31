from __future__ import annotations

"""
Post-processing utilities for model outputs.

Design goals:
- Split responsibilities into composable stages:
    raw_model_output -> generic_text_hygiene() -> schema_aware_formatter() -> final_text
- Keep logic task-agnostic: no disease/slot/question hard-coding.
- Be conservative: only remove obvious prompt/instruction echoes, boilerplate, and exact/near-exact repeats.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
import re
from typing import Dict, List, Optional


def _normalize_newlines(s: str) -> str:
    return (s or "").replace("\r\n", "\n").replace("\r", "\n")


def _collapse_spaces(s: str) -> str:
    return re.sub(r"[ \t\f\v]+", " ", s or "").strip()


def _collapse_blank_lines(s: str, *, max_consecutive: int = 1) -> str:
    """
    Collapse excessive blank lines while preserving paragraph breaks.
    max_consecutive=1 means at most one blank line between paragraphs.
    """
    s = _normalize_newlines(s)
    lines = s.split("\n")
    out: List[str] = []
    blank = 0
    for ln in lines:
        if ln.strip() == "":
            blank += 1
            if blank <= max_consecutive:
                out.append("")
            continue
        blank = 0
        out.append(ln.rstrip())
    return "\n".join(out).strip()


def _strip_chat_artifacts(s: str) -> str:
    """
    Truncate if the model starts a new turn marker (generic chat transcripts).
    This is intentionally generic and not model/disease/task-specific.
    """
    s = s or ""
    # Common markers: "Human:", "User:", "Assistant:", "System:".
    m = re.search(r"(\n\s*)?(Human|User|Assistant|System)\s*:\s*", s, flags=re.IGNORECASE)
    if m:
        s = s[: m.start()]
    return s.strip()


def _remove_fenced_code_blocks(s: str) -> str:
    # Fenced blocks often contain prompt echoes or tool output; remove conservatively.
    return re.sub(r"```[\s\S]*?```", "", s or "")


def _compress_repeated_symbols(s: str) -> str:
    """
    Reduce degenerate runs like '】】】】】】' or '-----' without changing semantics.
    Only compresses single-character repeats beyond a threshold.
    """
    if not s:
        return s
    # Common bracket/punct chars that show up in degeneration.
    # Keep threshold relatively high to avoid harming legitimate ASCII art.
    return re.sub(r"([】\]）\)\}\>\-_=~\*#])\1{5,}", r"\1\1", s)


def _is_pure_punct_line(ln: str) -> bool:
    """
    True if the line has no letters/numbers/CJK and is mostly punctuation/brackets.
    Used to drop degenerate placeholder lines like '】】】】】】' safely.
    """
    t = (ln or "").strip()
    if not t:
        return True
    # If it has any alnum or CJK, keep it.
    if re.search(r"[A-Za-z0-9\u4e00-\u9fff]", t):
        return False
    # If it's long and only punctuation/symbols, drop.
    if len(t) >= 6 and re.fullmatch(r"[\W_]+", t) is not None:
        return True
    return False


_INSTRUCTION_LINE_RE = re.compile(
    r"("
    # CN prompt/instruction echoes
    r"请(用|以)?(中文|英文)?(回答|输出)|"
    r"回答要求|输出要求|格式要求|"
    r"不要(输出|复述|编造|杜撰)|"
    r"根据(以上|下列|如下)|"
    r"证据句|证据原文|逐字引用|"
    # EN prompt/instruction echoes
    r"^(\s*)?(answer\s+the\s+question|requirements|output\s+format)\b|"
    r"\bevidence\s*:|\bbased\s+on\s+the\s+provided\b|"
    r"\bdo\s+not\s+(repeat|output|include|hallucinate)\b"
    r")",
    flags=re.IGNORECASE,
)


def _should_drop_instruction_line(ln: str) -> bool:
    """
    Drop only if the line is very likely an instruction echo / template header.
    Conservative: we don't drop long content lines just because they contain a marker.
    """
    t = (ln or "").strip()
    if not t:
        return False
    if _is_pure_punct_line(t):
        return True
    # Standalone bracket headers like "【回答要求】"
    if re.fullmatch(r"[【\[][^】\]]{1,24}[】\]]", t):
        return True
    # If it matches instruction markers and is short-ish, it's likely an echo line.
    if _INSTRUCTION_LINE_RE.search(t) and len(t) <= 200:
        return True
    return False


_BOILERPLATE_PARA_RE = re.compile(
    r"("
    # CN boilerplate tails
    r"如有(任何|其他)问题|请(随时)?告知|欢迎继续提问|"
    r"(以上|上述|本)回答(仅供参考)?|"
    r"建议(咨询|就医|联系)(医生|专业人士)|"
    r"祝您(健康|顺利)|"
    # EN boilerplate tails
    r"if\s+you\s+have\s+any\s+other\s+questions|"
    r"please\s+let\s+me\s+know|"
    r"this\s+is\s+for\s+informational\s+purposes\s+only"
    r")",
    flags=re.IGNORECASE,
)


def _split_paragraphs(s: str) -> List[str]:
    s = _collapse_blank_lines(s)
    if not s:
        return []
    return [p.strip() for p in re.split(r"\n\s*\n+", s) if p.strip()]


def _join_paragraphs(paras: List[str]) -> str:
    return "\n\n".join([p.strip() for p in paras if p and p.strip()]).strip()


def _norm_for_dedupe(s: str) -> str:
    s = _collapse_spaces(_normalize_newlines(s))
    s = s.lower()
    # Normalize common quote variants; keep conservative.
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return s


def _near_identical(a: str, b: str, *, threshold: float = 0.97) -> bool:
    """
    Conservative near-duplicate check: only triggers for reasonably long texts.
    """
    na = _norm_for_dedupe(a)
    nb = _norm_for_dedupe(b)
    if na == nb:
        return True
    # Only apply fuzzy matching for non-trivial lengths.
    if min(len(na), len(nb)) < 48:
        return False
    return SequenceMatcher(a=na, b=nb).ratio() >= float(threshold)


def _dedupe_paragraphs(paras: List[str]) -> List[str]:
    kept: List[str] = []
    seen_norm: List[str] = []
    for p in paras:
        pn = _norm_for_dedupe(p)
        # Exact duplicates
        if pn in seen_norm:
            continue
        # Near-duplicates against recent kept paragraphs only (conservative).
        if any(_near_identical(p, q) for q in kept[-5:]):
            continue
        kept.append(p)
        seen_norm.append(pn)
    return kept


def _dedupe_sentences_within_paragraph(p: str) -> str:
    """
    Conservative sentence-level dedupe within a paragraph.
    Only removes exact/near-identical sentences.
    """
    s = p.strip()
    # Split while preserving delimiters
    parts = re.split(r"([。！？!?\.])", s)
    if len(parts) <= 1:
        return s
    out: List[str] = []
    seen: List[str] = []
    for i in range(0, len(parts), 2):
        sent = (parts[i] or "").strip()
        delim = parts[i + 1] if i + 1 < len(parts) else ""
        if not sent:
            continue
        # Avoid over-aggressive dedupe on very short fragments.
        # Still dedupe repeated short fallback-like sentences (exact duplicates only).
        contentful = bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", sent))
        if contentful and len(sent) >= 6:
            if len(s) < 240 or len(sent) < 12:
                # Short paragraph or short sentence: exact duplicates only.
                ns = _norm_for_dedupe(sent)
                if any(_norm_for_dedupe(prev) == ns for prev in seen[-32:]):
                    continue
            else:
                # Longer sentences in longer paragraphs: allow near-identical dedupe.
                if any(_near_identical(sent, prev) for prev in seen[-12:]):
                    continue
            seen.append(sent)
        out.append(sent + delim)
    return "".join(out).strip()


def generic_text_hygiene(text: str, user_prompt: str | None = None) -> str:
    """
    Stage A: Generic text hygiene.
    - Remove obvious prompt/instruction echoes (conservatively)
    - Remove assistant boilerplate tails
    - De-dupe exact / near-identical paragraphs/sentences conservatively
    - Normalize whitespace/line breaks
    Hard constraints:
    - No disease/slot/question specific assumptions
    - Must not fabricate content
    """
    s = _strip_chat_artifacts(text or "")
    if not s:
        return ""
    s = _normalize_newlines(s)
    s = _remove_fenced_code_blocks(s)
    s = _compress_repeated_symbols(s)

    # Remove exact prompt echo at the very beginning if present.
    up = (user_prompt or "").strip()
    if up:
        s_strip = s.lstrip()
        if s_strip.startswith(up):
            s_strip = s_strip[len(up) :].lstrip()
            s = s_strip

    # Line-level filtering (very conservative).
    lines_in = [ln.rstrip() for ln in s.split("\n")]
    lines_out: List[str] = []
    for ln in lines_in:
        t = ln.strip()
        if t == "":
            lines_out.append("")
            continue
        if _should_drop_instruction_line(t):
            continue
        # Drop pure punctuation placeholder lines.
        if _is_pure_punct_line(t):
            continue
        lines_out.append(ln.strip())
    s = "\n".join(lines_out)
    s = _collapse_blank_lines(s)

    # Paragraph-level boilerplate removal + dedupe
    paras = _split_paragraphs(s)
    kept: List[str] = []
    for p in paras:
        pn = _collapse_spaces(p)
        # Drop short boilerplate-only paragraphs.
        if _BOILERPLATE_PARA_RE.search(pn) and len(pn) <= 240:
            continue
        kept.append(p.strip())
    kept = _dedupe_paragraphs(kept)

    # Sentence-level dedupe within each paragraph (conservative).
    kept = [_dedupe_sentences_within_paragraph(p) for p in kept]
    out = _join_paragraphs(kept)
    out = _compress_repeated_symbols(out)
    out = _collapse_blank_lines(out)
    return out


@dataclass(frozen=True)
class _Section:
    slot: str
    title: str
    body: str


def schema_aware_formatter(
    cleaned_text: str,
    answered_slots: List[str] | None = None,
    slot_to_section_title: Dict[str, str] | None = None,
) -> str:
    """
    Stage B: Schema-aware formatter.

    - If answered_slots is provided:
        - Only keep sections whose slot is in answered_slots
        - Only use titles provided by slot_to_section_title (no hard-coded labels)
    - If answered_slots is None/empty OR mapping missing:
        - Return cleaned_text unchanged

    This formatter is *structure-aware* but does not assume a specific language or number of sections.
    It will only activate when it can parse headings that match provided titles.
    """
    s = (cleaned_text or "").strip()
    if not s:
        return ""
    if not answered_slots:
        return s
    if not slot_to_section_title:
        return s

    answered = [str(x) for x in answered_slots if str(x).strip()]
    if not answered:
        return s
    allow = set(answered)

    # Build title -> slot mapping for ALL slots so we can detect section boundaries,
    # but only emit sections for slots in `allow`.
    title_to_slot: Dict[str, str] = {}
    for slot, title in slot_to_section_title.items():
        if isinstance(title, str) and title.strip():
            title_to_slot[title.strip()] = slot
    if not title_to_slot:
        return s

    # Parse sections by detecting headings that match any provided title.
    # Heading forms supported:
    #   "<Title>:" or "<Title>：" or "<Title>\n"
    lines = _normalize_newlines(s).split("\n")
    # Precompile heading regexes.
    heading_res: List[tuple[str, re.Pattern]] = []
    for title, slot in title_to_slot.items():
        pat = re.compile(rf"^\s*{re.escape(title)}\s*(?:[:：]\s*(.*))?\s*$")
        heading_res.append((title, pat))

    sections: List[_Section] = []
    cur_slot: Optional[str] = None
    cur_title: Optional[str] = None
    cur_body: List[str] = []

    def _flush():
        nonlocal cur_slot, cur_title, cur_body, sections
        if cur_slot and cur_title and cur_slot in allow:
            body = _collapse_blank_lines("\n".join(cur_body)).strip()
            if body:
                sections.append(_Section(slot=cur_slot, title=cur_title, body=body))
        cur_slot = None
        cur_title = None
        cur_body = []

    for ln in lines:
        matched = False
        for title, pat in heading_res:
            m = pat.match(ln)
            if not m:
                continue
            _flush()
            cur_title = title
            cur_slot = title_to_slot[title]
            inline = (m.group(1) or "").strip()
            if inline:
                cur_body.append(inline)
            matched = True
            break
        if matched:
            continue
        # If no section started yet, we can't safely assign content; keep original unchanged.
        if cur_slot is None:
            return s
        cur_body.append(ln)
    _flush()

    if not sections:
        return s

    # Re-emit sections in the order they appeared; omit any slot not in answered_slots (flush() gates).
    out_lines: List[str] = []
    for sec in sections:
        out_lines.append(f"{sec.title}：")
        out_lines.append(sec.body.strip())
        out_lines.append("")
    return _collapse_blank_lines("\n".join(out_lines)).strip()


def postprocess_answer(
    raw_text: str,
    user_prompt: str | None = None,
    answered_slots: List[str] | None = None,
    slot_to_section_title: Dict[str, str] | None = None,
) -> str:
    """
    Stage C: Final assembly wrapper (orchestration only).
    """
    cleaned = generic_text_hygiene(raw_text, user_prompt=user_prompt)
    return schema_aware_formatter(
        cleaned,
        answered_slots=answered_slots,
        slot_to_section_title=slot_to_section_title,
    )


