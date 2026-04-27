"""
projector: runtime entry point of Projector (runnable implementation)

The current "strong consistency" approach of this repo:
- Projector aligns directly to the target LLM's past_key_values space (see training/projector_kv.py)

Therefore this module mainly provides:
- Checkpoint loading
- Generate per-layer K/V from last_hidden (for building KB or injection)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .training.projector_kv import KVProjector, ProjectorConfig


@dataclass(frozen=True)
class ProjectorCheckpoint:
    projector: KVProjector
    layer_ids: List[int]
    base_model: str


def load_kv_projector(ckpt_path: Path, *, device: Optional[str] = None) -> ProjectorCheckpoint:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg_d = ckpt["projector_cfg"]
    layer_ids = list(cfg_d["layer_ids"])
    proj = KVProjector(
        ProjectorConfig(
            hidden_size=int(cfg_d["hidden_size"]),
            kv_heads=int(cfg_d["kv_heads"]),
            head_dim=int(cfg_d["head_dim"]),
            layer_ids=layer_ids,
        )
    )
    proj.load_state_dict(ckpt["projector_state_dict"])
    if device:
        proj.to(torch.device(device))
    proj.eval()
    return ProjectorCheckpoint(projector=proj, layer_ids=layer_ids, base_model=str(ckpt.get("base_model", "")))



