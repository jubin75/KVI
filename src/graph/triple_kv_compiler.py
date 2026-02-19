"""
Triple KVI Compiler — Compile knowledge graph triples into short KV caches.

This module is the core of the v3 "三元 KVI" architecture:

* **Subject Anchoring**: For each entity matched in the query, inject a short
  KV cache (~15–20 tokens) of its canonical description into shallow
  Transformer layers (layers 0–3).
* **Triple KV**: For each relevant triple (s, r, o), compile a short Chinese
  sentence (≤15 tokens) into KV cache and inject into Transformer layers
  selected by the relation type via ``RELATION_LAYER_MAP``.
* **Per-Layer Masking**: KV is only active in the designated layer range;
  other layers see no prefix.  This avoids the signal collision that caused
  token corruption in v1/v2.

Design constraints (防 token 腐蚀):
- Each KV text ≤ 20 tokens, pure Chinese
- Each triple sentence ≤ 15 tokens
- No English terms, no numbers (unless unavoidable)
- KV content is *complementary* to the prompt evidence (never duplicate)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Relation → Layer mapping  (32-layer Qwen2.5-7B as reference)
# ---------------------------------------------------------------------------

# Tuple = (layer_start_inclusive, layer_end_inclusive)
RELATION_LAYER_MAP: Dict[str, Tuple[int, int]] = {
    # 定义/分类 → 浅层 (token alignment)
    "is_a":               (0, 7),
    "has_subtype":        (0, 7),
    "also_known_as":      (0, 7),
    "has_molecular_weight": (0, 7),
    "utilizes":           (0, 7),
    "detects":            (0, 7),

    # 因果/机制 → 中层 (semantic reasoning)
    "causes":             (8, 15),
    "caused_by":          (8, 15),
    "manifests_as":       (8, 15),
    "manifestation_of":   (8, 15),
    "leads_to":           (8, 15),
    "associated_with":    (8, 15),

    # 治疗/诊断 → 中高层 (behavioral reasoning)
    "treats":             (12, 19),
    "treated_by":         (12, 19),
    "prevents":           (12, 19),
    "prevented_by":       (12, 19),
    "diagnosed_by":       (12, 19),

    # 结构/位置 → 早中层 (spatial reasoning)
    "located_in":         (4, 11),
    "location_of":        (4, 11),
    "part_of":            (4, 11),
    "has_part":           (4, 11),
    "transmits_via":      (4, 11),
    "transmission_route_for": (4, 11),
    "distributed_in":     (4, 11),

    # 抑制 → 中层
    "inhibits":           (8, 15),
    "inhibited_by":       (8, 15),
}

SUBJECT_ANCHOR_LAYERS: Tuple[int, int] = (0, 3)
DEFAULT_LAYERS: Tuple[int, int] = (0, 7)

# Predicate → short Chinese verb for triple sentence
_PRED_VERB: Dict[str, str] = {
    "causes":           "导致",
    "caused_by":        "由…引起",
    "treats":           "治疗",
    "treated_by":       "被…治疗",
    "manifests_as":     "表现为",
    "manifestation_of": "是…的表现",
    "is_a":             "属于",
    "has_subtype":      "包括",
    "part_of":          "是…的一部分",
    "has_part":         "包含",
    "located_in":       "分布于",
    "location_of":      "是…的分布区",
    "inhibits":         "抑制",
    "inhibited_by":     "被…抑制",
    "transmits_via":    "经…传播",
    "transmission_route_for": "是…的传播途径",
    "prevents":         "预防",
    "prevented_by":     "被…预防",
    "associated_with":  "与…相关",
    "also_known_as":    "又称",
    "distributed_in":   "分布于",
    "has_molecular_weight": "分子量约为",
    "utilizes":         "利用",
    "detects":          "检测",
}


# ---------------------------------------------------------------------------
# Data structures for compiled triple KV bank
# ---------------------------------------------------------------------------

@dataclass
class TripleKVItem:
    """Metadata for one compiled KV item (subject anchor or triple sentence)."""
    item_id: str
    entity_name: str
    item_type: str          # "subject_anchor" or "triple"
    text: str               # short Chinese text used for KV compilation
    relation: str           # "" for subject_anchor, relation type for triple
    layer_start: int
    layer_end: int
    token_count: int = 0    # filled after tokenization
    graph_triple_id: str = ""  # original triple_id from graph (for walk-based filtering)
    object_name: str = ""      # object entity name (for debug display)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TripleKVItem":
        return cls(
            item_id=str(d.get("item_id") or ""),
            entity_name=str(d.get("entity_name") or ""),
            item_type=str(d.get("item_type") or ""),
            text=str(d.get("text") or ""),
            relation=str(d.get("relation") or ""),
            layer_start=int(d.get("layer_start") or 0),
            layer_end=int(d.get("layer_end") or 0),
            token_count=int(d.get("token_count") or 0),
            graph_triple_id=str(d.get("graph_triple_id") or ""),
            object_name=str(d.get("object_name") or ""),
        )


@dataclass
class TripleKVManifest:
    """Manifest for a compiled triple KV bank."""
    # entity_name → list of item_ids
    entity_items: Dict[str, List[str]] = field(default_factory=dict)
    # item_id → TripleKVItem metadata
    items: Dict[str, TripleKVItem] = field(default_factory=dict)
    # graph_triple_id → item_id (for walk-based filtering)
    triple_id_index: Dict[str, str] = field(default_factory=dict)
    # model info
    model_name: str = ""
    num_layers: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "num_layers": self.num_layers,
            "entity_items": self.entity_items,
            "items": {k: v.to_dict() for k, v in self.items.items()},
            "triple_id_index": self.triple_id_index,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TripleKVManifest":
        items_raw = d.get("items") or {}
        items = {k: TripleKVItem.from_dict(v) for k, v in items_raw.items()}
        # Rebuild triple_id_index from items if not in data
        triple_id_index = dict(d.get("triple_id_index") or {})
        if not triple_id_index:
            for item_id, item in items.items():
                if item.graph_triple_id:
                    triple_id_index[item.graph_triple_id] = item_id
        return cls(
            entity_items=dict(d.get("entity_items") or {}),
            items=items,
            triple_id_index=triple_id_index,
            model_name=str(d.get("model_name") or ""),
            num_layers=int(d.get("num_layers") or 0),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "TripleKVManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Text generation helpers
# ---------------------------------------------------------------------------

def _build_subject_anchor_text(entity_name: str, description: str, aliases: List[str]) -> str:
    """
    Build a short Chinese text for subject anchoring KV.
    Target: ≤ 20 tokens, pure Chinese.
    """
    # Prefer description if available
    if description:
        # Truncate long descriptions
        desc = description.strip()
        if len(desc) > 40:
            desc = desc[:40]
        return f"{entity_name}（{desc}）"

    # Fallback to aliases
    if aliases:
        alias_str = "、".join(aliases[:2])
        return f"{entity_name}（{alias_str}）"

    return entity_name


def _build_triple_sentence(subject: str, predicate: str, obj: str) -> str:
    """
    Build a short Chinese sentence from a triple.
    Target: ≤ 15 tokens. Insert separators so subject/verb/object don't run together.
    """
    verb = _PRED_VERB.get(predicate, predicate)
    # Avoid "subjectverbobject"粘连: add thin space between parts when verb is long or ASCII
    parts = [subject.strip(), verb.strip(), obj.strip()]
    sent = " ".join(p for p in parts if p)
    # Truncate by char to stay within ~15 tokens (roughly 45 chars for Chinese/English mix)
    if len(sent) > 45:
        sent = sent[:45].rstrip()
    return sent


def get_layer_range(relation: str) -> Tuple[int, int]:
    """Get the injection layer range for a given relation type."""
    return RELATION_LAYER_MAP.get(relation, DEFAULT_LAYERS)


# ---------------------------------------------------------------------------
# Compile function (model-dependent, uses torch + HF)
# ---------------------------------------------------------------------------

def compile_triple_kvbank(
    *,
    graph_index_path: Path,
    model_name_or_path: str,
    out_dir: Path,
    device: str = "auto",
    dtype: str = "float16",
    max_anchor_tokens: int = 20,
    max_triple_tokens: int = 15,
) -> TripleKVManifest:
    """
    Compile a triple KV bank from a knowledge graph index.

    For each entity in the graph:
    1. Generate a subject anchor KV (short description/alias text)
    2. For each triple where entity is the subject, generate a triple KV

    Each KV item is stored as a safetensors file containing the
    ``past_key_values`` extracted from a forward pass of the short text.

    Returns:
        TripleKVManifest with metadata about all compiled items.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load graph
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.graph.schema import KnowledgeGraphIndex

    graph = KnowledgeGraphIndex.load(graph_index_path)
    print(f"[triple_kv] Loaded graph: nodes={len(graph.nodes)} triples={len(graph.triples)}", file=sys.stderr)

    # Load model
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    print(f"[triple_kv] Loading model: {model_name_or_path} → {dev}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, use_fast=True, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True,
    )
    model.to(dev).eval()

    num_layers = model.config.num_hidden_layers
    print(f"[triple_kv] Model loaded, num_layers={num_layers}", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = TripleKVManifest(model_name=model_name_or_path, num_layers=num_layers)

    def _extract_kv(text: str, max_tokens: int) -> Optional[Tuple[Any, int]]:
        """Tokenize text, forward through model, return (past_key_values, token_count)."""
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_tokens:
            ids = ids[:max_tokens]
        if not ids:
            return None
        input_ids = torch.tensor([ids], device=dev)
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
        # out.past_key_values: tuple of (key, value) per layer
        # Each key/value shape: [batch=1, num_heads, seq_len, head_dim]
        pkv = out.past_key_values
        return pkv, len(ids)

    def _save_kv(item_id: str, pkv: Any) -> Path:
        """Save past_key_values to a .pt file (safetensors adds complexity)."""
        # Convert to CPU float16 for storage efficiency
        kv_data = []
        for layer_kv in pkv:
            k, v = layer_kv[0], layer_kv[1]
            kv_data.append((k.cpu().half(), v.cpu().half()))
        fpath = out_dir / f"{item_id}.pt"
        torch.save(kv_data, fpath)
        return fpath

    # Process each entity
    for node_id, node in sorted(graph.nodes.items()):
        entity = node.entity
        ename = entity.name
        safe_name = ename.replace("/", "_").replace(" ", "_")[:30]
        item_ids_for_entity: List[str] = []

        # 1. Subject anchor
        anchor_text = _build_subject_anchor_text(
            ename, entity.description, entity.aliases,
        )
        anchor_id = f"anchor_{safe_name}"
        result = _extract_kv(anchor_text, max_anchor_tokens)
        if result:
            pkv, tok_count = result
            _save_kv(anchor_id, pkv)
            item = TripleKVItem(
                item_id=anchor_id,
                entity_name=ename,
                item_type="subject_anchor",
                text=anchor_text,
                relation="",
                layer_start=SUBJECT_ANCHOR_LAYERS[0],
                layer_end=SUBJECT_ANCHOR_LAYERS[1],
                token_count=tok_count,
            )
            manifest.items[anchor_id] = item
            item_ids_for_entity.append(anchor_id)
            print(f"[triple_kv]   anchor: {anchor_text} ({tok_count} tokens)", file=sys.stderr)

        # 2. Triple KVs (where this entity is subject)
        triple_idx = 0
        seen_triple_ids: set = set()
        for rel_type, edges in node.outgoing.items():
            for edge in edges:
                tid = edge.get("triple_id", "")
                if not tid or tid in seen_triple_ids:
                    continue
                seen_triple_ids.add(tid)
                triple = graph.triples.get(tid)
                if not triple:
                    continue
                triple_text = _build_triple_sentence(
                    triple.subject, triple.predicate, triple.object,
                )
                layer_range = get_layer_range(triple.predicate)
                triple_item_id = f"triple_{safe_name}_{triple_idx}"
                result = _extract_kv(triple_text, max_triple_tokens)
                if result:
                    pkv, tok_count = result
                    _save_kv(triple_item_id, pkv)
                    item = TripleKVItem(
                        item_id=triple_item_id,
                        entity_name=ename,
                        item_type="triple",
                        text=triple_text,
                        relation=triple.predicate,
                        layer_start=layer_range[0],
                        layer_end=layer_range[1],
                        token_count=tok_count,
                        graph_triple_id=tid,
                        object_name=triple.object,
                    )
                    manifest.items[triple_item_id] = item
                    manifest.triple_id_index[tid] = triple_item_id
                    item_ids_for_entity.append(triple_item_id)
                    print(
                        f"[triple_kv]   triple: {triple_text} ({tok_count} tokens, "
                        f"layers {layer_range[0]}-{layer_range[1]}, tid={tid})",
                        file=sys.stderr,
                    )
                triple_idx += 1

        if item_ids_for_entity:
            manifest.entity_items[ename] = item_ids_for_entity

    # Save manifest
    manifest.save(out_dir / "manifest.json")
    print(
        f"[triple_kv] Compiled: {len(manifest.items)} items for "
        f"{len(manifest.entity_items)} entities → {out_dir}",
        file=sys.stderr,
    )
    return manifest


# ---------------------------------------------------------------------------
# Runtime: load & assemble KV for generation
# ---------------------------------------------------------------------------

def load_triple_kvbank(kvbank_dir: Path) -> Tuple[TripleKVManifest, Dict[str, Any]]:
    """
    Load a compiled triple KV bank from disk.

    Returns:
        (manifest, kv_cache_dict)
        kv_cache_dict: item_id → list of (key_tensor, value_tensor) per layer
    """
    import torch

    manifest_path = kvbank_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Triple KV manifest not found: {manifest_path}")
    manifest = TripleKVManifest.load(manifest_path)

    kv_cache_dict: Dict[str, Any] = {}
    for item_id in manifest.items:
        pt_path = kvbank_dir / f"{item_id}.pt"
        if pt_path.exists():
            kv_cache_dict[item_id] = torch.load(pt_path, map_location="cpu", weights_only=True)

    return manifest, kv_cache_dict


def assemble_kv_for_entities(
    *,
    matched_entity_names: List[str],
    walk_triple_ids: Optional[List[str]] = None,
    manifest: TripleKVManifest,
    kv_cache_dict: Dict[str, Any],
    device: Any = None,
    dtype: Any = None,
) -> Optional[Tuple[Any, List[str]]]:
    """
    Assemble per-layer KV prefix from matched entities' KV items.

    v4 contract (DRM pre-filtering):
    - The caller (``run_graph_inference.py``) MUST apply DRM scoring and
      relation gating **before** calling this function.
    - ``walk_triple_ids`` should be the DRM-filtered, budget-limited list
      of triple_ids.  Only those triples' KVs will be included.
    - Subject anchors are **always** included for matched entities.
    - If ``walk_triple_ids`` is an empty list ``[]``, no triple KVs are
      injected (only subject anchors).
    - ``walk_triple_ids=None`` is treated as "include no triples" (safe
      default).  The old fallback behaviour (include ALL triples when
      None) has been removed to prevent injection of irrelevant KVs.

    Returns:
        (past_key_values, selected_item_ids):
        - past_key_values: tuple of (key, value) per layer, HF-compatible
        - selected_item_ids: list of item_ids that were actually assembled
        Returns (None, []) if no KV to inject.
    """
    import torch

    num_layers = manifest.num_layers
    if num_layers <= 0:
        return None, []

    # Build set of allowed item_ids from walk triple_ids
    # v4: None is treated as empty (no triples), NOT as "include all"
    allowed_triple_item_ids: set = set()
    if walk_triple_ids:
        for tid in walk_triple_ids:
            iid = manifest.triple_id_index.get(tid)
            if iid:
                allowed_triple_item_ids.add(iid)

    # Collect relevant item_ids
    relevant_items: List[str] = []
    for ename in matched_entity_names:
        item_ids = manifest.entity_items.get(ename, [])
        for iid in item_ids:
            meta = manifest.items.get(iid)
            if not meta:
                continue
            if meta.item_type == "subject_anchor":
                # Subject anchors are always included
                relevant_items.append(iid)
            elif meta.item_type == "triple":
                # Only include triples explicitly allowed by DRM filtering
                if iid in allowed_triple_item_ids:
                    relevant_items.append(iid)

    if not relevant_items:
        return None, []

    # Build per-layer KV lists
    per_layer_keys: List[List[Any]] = [[] for _ in range(num_layers)]
    per_layer_vals: List[List[Any]] = [[] for _ in range(num_layers)]

    for iid in relevant_items:
        meta = manifest.items.get(iid)
        kv_data = kv_cache_dict.get(iid)
        if meta is None or kv_data is None:
            continue

        for layer_idx in range(min(len(kv_data), num_layers)):
            if meta.layer_start <= layer_idx <= meta.layer_end:
                k, v = kv_data[layer_idx]
                if device is not None:
                    k = k.to(device)
                    v = v.to(device)
                if dtype is not None:
                    k = k.to(dtype)
                    v = v.to(dtype)
                per_layer_keys[layer_idx].append(k)
                per_layer_vals[layer_idx].append(v)

    # Merge: concat along sequence dimension for each layer
    result = []
    has_any = False
    for layer_idx in range(num_layers):
        if per_layer_keys[layer_idx]:
            merged_k = torch.cat(per_layer_keys[layer_idx], dim=2)  # [batch, heads, seq, dim]
            merged_v = torch.cat(per_layer_vals[layer_idx], dim=2)
            result.append((merged_k, merged_v))
            has_any = True
        else:
            # No KV for this layer — create empty prefix
            # We need a reference shape; get it from any existing layer
            result.append(None)

    if not has_any:
        return None, []

    # For layers with None (no KV), we need to either:
    # (a) create zero-length KV, or (b) pad to match max seq_len.
    # HuggingFace `generate` expects all layers to have the same seq_len
    # in past_key_values. So we need to pad with zeros.
    max_seq = 0
    ref_shape = None
    for entry in result:
        if entry is not None:
            k_shape = entry[0].shape  # [batch, heads, seq, dim]
            if k_shape[2] > max_seq:
                max_seq = k_shape[2]
            ref_shape = k_shape

    if ref_shape is None or max_seq == 0:
        return None, []

    batch, heads, _, head_dim = ref_shape
    padded_result = []
    for entry in result:
        if entry is not None:
            k, v = entry
            cur_seq = k.shape[2]
            if cur_seq < max_seq:
                # Pad with zeros
                pad_k = torch.zeros(batch, heads, max_seq - cur_seq, head_dim,
                                    device=k.device, dtype=k.dtype)
                pad_v = torch.zeros(batch, heads, max_seq - cur_seq, head_dim,
                                    device=v.device, dtype=v.dtype)
                k = torch.cat([pad_k, k], dim=2)
                v = torch.cat([pad_v, v], dim=2)
            padded_result.append((k, v))
        else:
            # All zeros for inactive layers
            target_device = device if device is not None else (
                result[0][0].device if result[0] is not None else "cpu"
            )
            target_dtype = dtype if dtype is not None else (
                result[0][0].dtype if result[0] is not None else torch.float16
            )
            zero_k = torch.zeros(batch, heads, max_seq, head_dim,
                                 device=target_device, dtype=target_dtype)
            zero_v = torch.zeros(batch, heads, max_seq, head_dim,
                                 device=target_device, dtype=target_dtype)
            padded_result.append((zero_k, zero_v))

    return tuple(padded_result), relevant_items


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Compile triple KV bank from knowledge graph")
    p.add_argument("--graph_index", required=True, help="Path to graph_index.json")
    p.add_argument("--model", required=True, help="Base LLM model path")
    p.add_argument("--out_dir", required=True, help="Output directory for triple KV bank")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max_anchor_tokens", type=int, default=20)
    p.add_argument("--max_triple_tokens", type=int, default=15)
    p.add_argument("--local_files_only", action="store_true")
    args = p.parse_args()

    manifest = compile_triple_kvbank(
        graph_index_path=Path(args.graph_index),
        model_name_or_path=args.model,
        out_dir=Path(args.out_dir),
        device=args.device,
        dtype=args.dtype,
        max_anchor_tokens=args.max_anchor_tokens,
        max_triple_tokens=args.max_triple_tokens,
    )

    # Output summary
    summary = {
        "ok": True,
        "num_items": len(manifest.items),
        "num_entities": len(manifest.entity_items),
        "num_layers": manifest.num_layers,
        "out_dir": str(args.out_dir),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
