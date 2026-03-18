from __future__ import annotations

import re
from typing import Iterable


_WS_RE = re.compile(r"\s+")


def _normalize_text(s: str) -> str:
    s = str(s or "")
    s = s.strip().lower()
    # remove punctuation-like characters (keep CJK/letters/digits)
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s, flags=re.UNICODE)
    s = _WS_RE.sub(" ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> int:
    return int(_normalize_text(pred) == _normalize_text(gold))


def best_exact_match(pred: str, golds: Iterable[str]) -> int:
    g = list(golds)
    if not g:
        return 0
    return int(any(exact_match(pred, x) for x in g))

