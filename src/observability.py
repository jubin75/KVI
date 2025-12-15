"""
observability: 轻量观测实现（JSONL logger + timers）

目标
- 让 demo/工程调试可复现：输出检索命中、注入 token 数、gamma、延迟等。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def ms(self) -> float:
        return float((time.perf_counter() - self.t0) * 1000.0)


@dataclass
class ObservabilityConfig:
    export_debug: bool = True
    export_citations: bool = True
    export_gamma_stats: bool = True
    export_latency_breakdown: bool = True


class JsonlLogger:
    def __init__(self, path: Path, cfg: Optional[ObservabilityConfig] = None) -> None:
        self.path = path
        self.cfg = cfg or ObservabilityConfig()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, payload: Dict[str, Any]) -> None:
        rec = {"ts": time.time(), "event": event, **payload}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")



