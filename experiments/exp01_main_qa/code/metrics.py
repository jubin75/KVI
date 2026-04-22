from __future__ import annotations

import re
import string
from typing import Any, Dict, Iterable, Optional, Tuple


_WS_RE = re.compile(r"\s+")

# FEVER / KILT-style veracity labels (longest phrase first for stable substring scan).
_FEVER_LABEL_SCAN_ORDER = ("NOT ENOUGH INFO", "REFUTES", "SUPPORTS")


def parse_fever_label(pred: str) -> Optional[str]:
    """
    Heuristic: first occurrence of a standard label in model output (case-insensitive).
    Use when gold is SUPPORTS / REFUTES / NOT ENOUGH INFO.
    """
    if not str(pred or "").strip():
        return None
    u = pred.upper()
    best_pos: Optional[int] = None
    best_lab: Optional[str] = None
    for lab in _FEVER_LABEL_SCAN_ORDER:
        p = u.find(lab)
        if p < 0:
            continue
        if best_pos is None or p < best_pos:
            best_pos = p
            best_lab = lab
    return best_lab


def best_fever_label_em(pred: str, golds: Iterable[str]) -> int:
    """1 iff parsed label matches any gold string after upper/strip."""
    parsed = parse_fever_label(pred)
    if parsed is None:
        return 0
    gold_set = {str(g).strip().upper() for g in golds if str(g).strip()}
    return int(parsed in gold_set)


def _normalize_text(s: str) -> str:
    s = str(s or "")
    s = s.strip().lower()
    # remove punctuation-like characters (keep CJK/letters/digits)
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s, flags=re.UNICODE)
    s = _WS_RE.sub(" ", s).strip()
    return s


def normalize_answer_squad(s: str) -> str:
    """
    SQuAD / DrQA style normalization (common for HotpotQA & NQ EM/F1).
    """
    s = str(s or "").lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = _WS_RE.sub(" ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> int:
    """Strict: full prediction string equals gold after light normalization."""
    return int(_normalize_text(pred) == _normalize_text(gold))


def best_exact_match(pred: str, golds: Iterable[str]) -> int:
    g = list(golds)
    if not g:
        return 0
    return int(any(exact_match(pred, x) for x in g))


def _gold_in_prediction(pred_norm: str, gold_norm: str) -> bool:
    """Whether gold answer appears as a supported match inside prediction (for long generations)."""
    if not gold_norm:
        return False
    if pred_norm == gold_norm:
        return True
    # Short yes/no: word boundary to avoid 'yes' in unrelated words
    if gold_norm in ("yes", "no"):
        if bool(re.search(rf"\b{re.escape(gold_norm)}\b", pred_norm)):
            return True
        return _binary_yes_no_paraphrase(gold_norm, pred_norm)
    # Longer answers: substring after squad-style normalize (phrase in paragraph)
    if gold_norm in pred_norm:
        return True
    return False


def _binary_yes_no_paraphrase(gold_norm: str, pred_norm: str) -> bool:
    """
    Hotpot/NQ often use gold 'yes'/'no' while models answer with paraphrases (no literal 'yes').
    Conservative heuristics to avoid counting unrelated 'yes' tokens.
    """
    if gold_norm == "yes":
        if re.search(r"\b(yes|yeah)\b", pred_norm):
            return True
        # Chinese / paraphrase signals
        if any(
            frag in pred_norm
            for frag in (
                "同国籍",
                "国籍一致",
                "相同的国籍",
                "都是美国",
                "都是美国人",
                "答案是肯定的",
                "因此答案是肯定",
                "same nationality",
                "both american",
                "both americans",
            )
        ):
            return True
    if gold_norm == "no":
        if re.search(r"\bno\b", pred_norm):
            return True
        if any(
            frag in pred_norm
            for frag in (
                "not in the same",
                "not the same",
                "are not in the same",
                "is not in the same",
                "not located in the same",
                "不位于同一",
                "不在同一个",
                "不是同一",
                "distinct",
                "different districts",
            )
        ):
            return True
    return False


def relaxed_em(pred: str, gold: str) -> int:
    """
    Open QA / Hotpot-style EM when models return paragraphs:
    1) strict match on squad-normalized full strings, or
    2) squad-normalized gold appears in squad-normalized prediction (substring / yes-no word).
    """
    pn = normalize_answer_squad(pred)
    gn = normalize_answer_squad(gold)
    if not gn:
        return 0
    if pn == gn:
        return 1
    return int(_gold_in_prediction(pn, gn))


def best_relaxed_em(pred: str, golds: Iterable[str]) -> int:
    g = list(golds)
    if not g:
        return 0
    return int(any(relaxed_em(pred, x) for x in g))


def f1_score_tokens(pred: str, gold: str) -> float:
    """Token-level F1 (SQuAD) between prediction and a single gold string."""
    pred_t = normalize_answer_squad(pred).split()
    gold_t = normalize_answer_squad(gold).split()
    if not gold_t:
        return 0.0
    if not pred_t:
        return 0.0
    common = {}
    for t in pred_t:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in gold_t:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_t)
    recall = overlap / len(gold_t)
    return 2 * precision * recall / (precision + recall)


def best_f1(pred: str, golds: Iterable[str]) -> float:
    gs = list(golds)
    if not gs:
        return 0.0
    return max(f1_score_tokens(pred, g) for g in gs)


def parse_truthfulqa_targets(obj: Any) -> Optional[Tuple[list[str], list[int]]]:
    """
    Parse TruthfulQA multiple-choice target object:
    {"choices": [...], "labels": [0/1,...]}.
    """
    if not isinstance(obj, dict):
        return None
    ch = obj.get("choices")
    lb = obj.get("labels")
    if not isinstance(ch, list) or not isinstance(lb, list) or len(ch) != len(lb) or not ch:
        return None
    choices = [str(x).strip() for x in ch]
    if any(not c for c in choices):
        return None
    labels: list[int] = []
    for x in lb:
        try:
            labels.append(1 if int(x) > 0 else 0)
        except Exception:
            return None
    if sum(labels) <= 0:
        return None
    return choices, labels


def truthfulqa_mc_proxy_scores(pred: str, targets: Any) -> Dict[str, float]:
    """
    TruthfulQA MC proxy from free-form prediction.
    - mc_acc_proxy: pick best-matching option by token-F1; 1 iff it is labeled true.
    - mc_mass_proxy: normalized overlap mass on true options:
      sum(sim_true) / (sum(sim_all)+eps), in [0,1].
    NOTE: This is NOT official MC1/MC2 (which depends on option likelihoods).
    """
    parsed = parse_truthfulqa_targets(targets)
    if parsed is None:
        return {"mc_acc_proxy": 0.0, "mc_mass_proxy": 0.0, "valid": 0.0}
    choices, labels = parsed
    sims = [max(relaxed_em(pred, c), f1_score_tokens(pred, c)) for c in choices]
    best_i = max(range(len(choices)), key=lambda i: sims[i])
    acc = float(labels[best_i] > 0)
    s_all = float(sum(sims))
    s_true = float(sum(s for s, y in zip(sims, labels) if y > 0))
    mass = (s_true / (s_all + 1e-8)) if s_all > 0 else 0.0
    return {"mc_acc_proxy": acc, "mc_mass_proxy": mass, "valid": 1.0}

