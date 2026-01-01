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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


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
    Remove common chat transcript artifacts (generic).

    Goals:
    - Preserve assistant answer content even if it is prefixed with "Assistant:"
    - Drop user-turn echoes like "Human:" / "User:"
    - If the model starts a new user turn mid-string, truncate at that point
    """
    s = _normalize_newlines(s or "").strip()
    if not s:
        return s

    # 1) If a new *user* turn begins mid-string, truncate there.
    # This catches degenerate outputs like "... answer ... Human: <next question>" even without newlines.
    m_user = re.search(r"(Human|User)\s*:\s*", s, flags=re.IGNORECASE)
    if m_user and m_user.start() > 0:
        s = s[: m_user.start()].rstrip()

    # 2) If content is line-based transcript, strip role prefixes instead of truncating.
    out_lines: List[str] = []
    role_re = re.compile(r"^\s*(Human|User|Assistant|System)\s*:\s*(.*)$", flags=re.IGNORECASE)
    for ln in s.split("\n"):
        m = role_re.match(ln)
        if not m:
            out_lines.append(ln.rstrip())
            continue
        role = (m.group(1) or "").lower()
        rest = (m.group(2) or "").strip()
        # Drop user turns; keep assistant/system content.
        if role in {"human", "user"}:
            continue
        if rest:
            out_lines.append(rest)
    s2 = "\n".join(out_lines).strip()
    return s2


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
    r"请基于|"
    r"回答要求|输出要求|格式要求|"
    r"不要(输出|复述|编造|杜撰)|"
    r"(根据|依据)(以上|上述|下列|如下)|"
    r"证据句|证据原文|逐字引用|"
    # EN prompt/instruction echoes
    r"^(\s*)?(answer\s+the\s+question|requirements|output\s+format)\b|"
    r"\bevidence\s*:|\bbased\s+on\s+the\s+provided\b|"
    r"\bdo\s+not\s+(repeat|output|include|hallucinate)\b"
    r")",
    flags=re.IGNORECASE,
)


_RULE_LIKE_LINE_RE = re.compile(
    r"^\s*(?:\d+\s*[)\]）】\.、]|[-*]\s*)\s*.+$"
)

_RULE_KEYWORDS_RE = re.compile(
    r"(规则|遵循|要求|禁止|必须|请勿|不要|仅输出|只输出|不得|do\s+not|must|only\s+output|rule)",
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
    # Rule-like numbered/bulleted lines that contain imperative keywords are almost always template echoes.
    # Keep conservative: only drop if BOTH looks like a rule item AND contains rule keywords.
    if len(t) <= 220 and _RULE_LIKE_LINE_RE.match(t) and _RULE_KEYWORDS_RE.search(t):
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

    def _threshold_for_sentence(sent: str, *, paragraph_len: int) -> float:
        """
        Adaptive near-duplicate threshold.
        Rationale:
        - Short factual medical sentences can be legitimately similar; use stricter threshold to avoid false removals.
        - Long loop-y sentences often differ by minor typos/spaces; allow looser threshold there.
        """
        ls = len(sent)
        if ls <= 24:
            return 0.985
        if ls <= 64:
            return 0.97
        if ls <= 140:
            return 0.96
        # Very long sentences in long paragraphs: allow looser to break degeneration loops.
        if paragraph_len >= 240:
            return 0.94
        return 0.96
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
                thr = _threshold_for_sentence(sent, paragraph_len=len(s))
                if any(_near_identical(sent, prev, threshold=thr) for prev in seen[-12:]):
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
    raw = raw_text or ""
    cleaned = generic_text_hygiene(raw, user_prompt=user_prompt)
    out = schema_aware_formatter(
        cleaned,
        answered_slots=answered_slots,
        slot_to_section_title=slot_to_section_title,
    )
    # Safety floor: avoid returning empty text when the raw output clearly contains readable language.
    # This prevents over-filtering from wiping the entire answer (generic; no task assumptions).
    if not (out or "").strip():
        if re.search(r"[A-Za-z\u4e00-\u9fff]", raw):
            s = _strip_chat_artifacts(raw)
            s = _normalize_newlines(s)
            s = _compress_repeated_symbols(s)
            # Remove exact prompt echo prefix if present (conservative).
            up = (user_prompt or "").strip()
            if up:
                s_strip = s.lstrip()
                if s_strip.startswith(up):
                    s = s_strip[len(up) :].lstrip()
            s = _collapse_blank_lines(s)
            if s.strip():
                return s.strip()
    return (out or "").strip()


# -----------------------------------------------------------------------------
# AnswerUnit-based post-processing (minimal, dependency-free)
#
# This pipeline is designed to address:
# - repeated answers (including repeated "insufficient evidence" fallbacks)
# - degenerate / template / garbage outputs (e.g. '】】】】】')
# - common medical intents where we should retry evidence lookup before failing
#
# IMPORTANT constraints:
# - Does NOT change slot/schema/evidence data structures
# - Does NOT introduce new deps
# - Can be used behind an existing _postprocess_answer(text, user_prompt) wrapper
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AnswerUnit:
    """
    A small, local representation of an answer fragment.
    This does NOT modify upstream structures; it is only for post-processing.
    """

    text: str
    slot: Optional[str] = None
    evidence_ids: Tuple[str, ...] = ()
    is_fallback: bool = False
    failure_reason: Optional[str] = None  # e.g. "slot_miss" | "evidence_missing" | "pdf_miss"


@dataclass(frozen=True)
class FinalAnswer:
    text: str
    failure_reason: Optional[str] = None
    used_evidence: bool = False


_FALLBACK_RE = re.compile(
    r"("
    r"现有证据不足|证据不足|缺乏证据|无法.*确定|无法.*提供|未找到.*证据|"
    r"insufficient\s+evidence|no\s+evidence|not\s+enough\s+evidence|cannot\s+determine"
    r")",
    flags=re.IGNORECASE,
)

_TEMPLATE_GARBAGE_RE = re.compile(
    r"("
    r"【[^】]{0,32}】|"
    r"\{\{[^}]{0,64}\}\}|"
    r"\[INST\]|\[/INST\]|"
    r"<\|[^|]{0,48}\|>"
    r")"
)


# Slots that are expected to be evidence-grounded (cost-aware).
# For other slots (e.g. definition/background/research_gap), forcing an evidence retry on every fallback
# can waste retrieval time and is not necessary for a good UX.
EVIDENCE_EXPECTED_SLOTS: set[str] = {
    "transmission",
    "pathogenesis",
    "clinical_features",
    "diagnosis",
    "treatment",
    "epidemiology",
    "risk_factors",
    "complications",
    "prognosis",
    "prevention",
}


def _unit_norm_key(u: AnswerUnit) -> Tuple[str, str]:
    # normalized_text + slot is the dedupe key as required
    return (_norm_for_dedupe(u.text), str(u.slot or ""))


def _looks_like_garbage_text(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return True
    # Allow bracketed section headings used by the runtime (three knowledge layers).
    # These are intentional structure markers, not "template residue".
    if t in {
        "【证据支持的结论】",
        "【领域共识解释】",
        "【推测与研究进展（可选）】",
        "[Evidence-backed conclusions]",
        "[Domain-prior explanation]",
        "[Speculative / open (optional)]",
    }:
        return False
    # Template residue
    if _TEMPLATE_GARBAGE_RE.search(t):
        # But don't drop if it's a long, contentful sentence with bracket usage; be conservative.
        if len(t) <= 220:
            return True
    # Degenerate symbol-only lines
    if _is_pure_punct_line(t):
        return True
    # Excessive repeated single symbols (e.g. 】】】】)
    if re.search(r"([】\]）\)\}\>\-_=~\*#])\1{3,}", t):
        return True
    return False


def _parse_answer_units_from_text(cleaned_text: str) -> List[AnswerUnit]:
    """
    Minimal parser:
    - Split into paragraphs as units.
    - Attempt to extract a lightweight "slot label" from a leading 'Label:' line.
      This is NOT slot-specific; it's generic "prefix label" parsing.
    - evidence_ids are left empty unless explicit ids are detected.
    """
    s = (cleaned_text or "").strip()
    if not s:
        return []
    paras = _split_paragraphs(s)
    units: List[AnswerUnit] = []
    # Very lightweight: detect leading "Label:" if it is short.
    label_re = re.compile(r"^\s*([^:：]{1,40})\s*[:：]\s*(.+)\s*$")
    for p in paras:
        p0 = p.strip()
        slot = None
        txt = p0
        m = label_re.match(p0)
        if m:
            # treat the label as slot identifier (generic, not medical-specific)
            slot = m.group(1).strip()
            txt = m.group(2).strip()
        is_fb = bool(_FALLBACK_RE.search(txt))
        units.append(AnswerUnit(text=txt, slot=slot, evidence_ids=(), is_fallback=is_fb, failure_reason=None))
    return units


def normalize_answer_units(answer_units: Sequence[AnswerUnit]) -> List[AnswerUnit]:
    """
    职责：
    - 去重语义等价回答（字符串 + slot + evidence_id）
    - 合并多条“证据不足”类回答为 1 条
    - 过滤空内容 / 异常 token（如 '】】】'）
    """
    units_in = list(answer_units or [])
    out: List[AnswerUnit] = []
    seen_keys: set[Tuple[str, str, Tuple[str, ...]]] = set()
    fallback_seen_for_slot: set[str] = set()

    for u in units_in:
        txt = (u.text or "").strip()
        if not txt:
            continue
        if _looks_like_garbage_text(txt):
            continue

        evid = tuple(str(x) for x in (u.evidence_ids or ()) if str(x).strip())
        slot = str(u.slot) if u.slot is not None else ""
        key = (_norm_for_dedupe(txt), slot, evid)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        is_fb = bool(u.is_fallback or _FALLBACK_RE.search(txt))
        if is_fb:
            # merge multiple fallback variants per slot into ONE
            if slot in fallback_seen_for_slot:
                continue
            fallback_seen_for_slot.add(slot)
            # normalize fallback surface form (keep minimal + stable)
            fb_text = "现有证据不足以回答该问题。"
            out.append(
                AnswerUnit(
                    text=fb_text,
                    slot=u.slot,
                    evidence_ids=evid,
                    is_fallback=True,
                    failure_reason=u.failure_reason or "evidence_missing",
                )
            )
            continue

        out.append(AnswerUnit(text=txt, slot=u.slot, evidence_ids=evid, is_fallback=False, failure_reason=None))
    return out


def validate_answer_structure(answer_units: Sequence[AnswerUnit]) -> List[AnswerUnit]:
    """
    职责：
    - 校验回答是否包含可读自然语言
    - 若检测到 prompt 崩坏 / 模板残留 / 乱码，丢弃该 unit
    - 防止输出退化为【证据句】】】】类内容
    """
    units = list(answer_units or [])
    out: List[AnswerUnit] = []
    for u in units:
        txt = (u.text or "").strip()
        if not txt:
            continue
        if _looks_like_garbage_text(txt):
            continue
        # Must contain some language content (CJK or letters)
        if not re.search(r"[A-Za-z\u4e00-\u9fff]", txt):
            continue
        # Drop if it's mostly template markers / instruction-y fragments
        if _INSTRUCTION_LINE_RE.search(txt) and len(txt) <= 200:
            continue
        out.append(u)
    return out


def _infer_question_intent_slots(user_prompt: str) -> List[str]:
    """
    Minimal intent inference for common medical slots.
    This does NOT change slot structures; it's only used for evidence policy decisions.
    """
    q = (user_prompt or "").strip().lower()
    if not q:
        return []
    slots: List[str] = []
    # Keep mapping small & stable; no disease-specific strings.
    patterns: List[Tuple[str, re.Pattern]] = [
        ("risk_factors", re.compile(r"(risk\s*factor|危险因素|风险因素|高危人群|易感人群)", re.I)),
        ("complications", re.compile(r"(complicat\w*|并发症|后遗症)", re.I)),
        ("prognosis", re.compile(r"(prognos\w*|预后|转归|死亡率|mortality|fatality)", re.I)),
        ("treatment", re.compile(r"(treat\w*|治疗|用药|药物|antivir)", re.I)),
        ("diagnosis", re.compile(r"(diagnos\w*|诊断|检测|pcr|elisa)", re.I)),
        ("clinical_features", re.compile(r"(symptom|症状|表现|临床)", re.I)),
        ("transmission", re.compile(r"(transmit|传播|途径|媒介|vector)", re.I)),
        ("pathogenesis", re.compile(r"(pathogen|机制|发病机制|cytokine|炎症|immun)", re.I)),
    ]
    for slot, pat in patterns:
        if pat.search(q):
            slots.append(slot)
    # de-dupe keep order
    seen = set()
    out = []
    for s in slots:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def enforce_evidence_policy(answer_units: Sequence[AnswerUnit], question_intent: Dict[str, Any]) -> FinalAnswer:
    """
    职责：
    - 若问题属于常见医学 slot（如 risk_factors / complications / prognosis）
      且 evidence 为空 → 回退到 evidence 层重新匹配
    - 若仍无 evidence，只允许输出一次标准 fallback
    - 明确区分：
        - evidence 缺失
        - slot 未命中
        - PDF 中确实无相关信息
    """
    units = list(answer_units or [])
    user_prompt = str((question_intent or {}).get("user_prompt") or "")
    inferred_slots = (question_intent or {}).get("intent_slots")
    if not isinstance(inferred_slots, list) or not inferred_slots:
        inferred_slots = _infer_question_intent_slots(user_prompt)
    # Evidence policy (cost-aware):
    # Only force an evidence-layer retry for slots that are expected to be evidence-grounded.
    inferred_set = {str(s) for s in inferred_slots if str(s).strip()}
    needs_evidence = bool(inferred_set & set(EVIDENCE_EXPECTED_SLOTS))

    # Detect whether any unit has evidence ids (if caller provides them).
    any_evidence = any(bool(u.evidence_ids) for u in units)
    any_non_fallback = any(not u.is_fallback for u in units)

    # If answer contains useful content, do not override.
    if any_non_fallback:
        txt = _join_paragraphs([u.text for u in units if u.text.strip()])
        return FinalAnswer(text=txt, failure_reason=None, used_evidence=any_evidence)

    # At this point, we only have fallbacks or nothing.
    evidence_lookup_fn = (question_intent or {}).get("evidence_lookup_fn")
    query_text = str((question_intent or {}).get("query_text") or user_prompt)

    # If we need evidence for this intent and we have a lookup fn, retry evidence.
    if needs_evidence and callable(evidence_lookup_fn):
        try:
            ev_texts = evidence_lookup_fn(query_text)
        except Exception:
            ev_texts = []
        ev_texts = [str(x).strip() for x in (ev_texts or []) if str(x).strip()]
        if ev_texts:
            # We cannot re-generate with LLM here; we can at least return evidence-backed minimal answer.
            ev_block = "\n".join([f"- {t}" for t in ev_texts[:3]])
            out = "基于检索到的证据：\n" + ev_block
            return FinalAnswer(text=out.strip(), failure_reason=None, used_evidence=True)
        # evidence retry still empty -> pdf miss (library has no relevant evidence for this intent)
        return FinalAnswer(text="现有证据不足以回答该问题。", failure_reason="pdf_miss", used_evidence=False)

    # If we cannot infer any intent slot, mark slot miss.
    if not inferred_slots:
        return FinalAnswer(text="现有证据不足以回答该问题。", failure_reason="slot_miss", used_evidence=False)
    # Otherwise, we have an intent but no evidence + no retry path -> evidence missing.
    return FinalAnswer(text="现有证据不足以回答该问题。", failure_reason="evidence_missing", used_evidence=False)


def postprocess_answer_units(
    raw_text: str,
    user_prompt: str | None = None,
    *,
    question_intent: Optional[Dict[str, Any]] = None,
) -> FinalAnswer:
    """
    Convenience wrapper:
      raw_text -> generic_text_hygiene -> parse units -> normalize -> validate -> enforce policy
    """
    cleaned = generic_text_hygiene(raw_text, user_prompt=user_prompt)
    units = _parse_answer_units_from_text(cleaned)
    units = normalize_answer_units(units)
    units = validate_answer_structure(units)
    qi = dict(question_intent or {})
    if user_prompt is not None and "user_prompt" not in qi:
        qi["user_prompt"] = user_prompt
    if "query_text" not in qi and user_prompt is not None:
        qi["query_text"] = user_prompt
    return enforce_evidence_policy(units, qi)


