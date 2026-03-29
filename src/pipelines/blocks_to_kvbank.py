"""
Pipeline：memory blocks（256-token）→ KVBank（FAISS）

严格对齐 PRD/多步注入的工程实现.md：
- KVBank 存的是 256-token memory blocks 的 K/V（不是 raw text）
- 建库时提取 layers 0..3 的 past_key_values（teacher cache，或后续用 projector）
- 单条 block 的 kv_len 必须 <= 256，且作为注入 token 粒度

检索向量（retrieval_keys）
- 推荐使用 DomainEncoder（独立 encoder）对 block 文本编码并归一化
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig
from ..vector_store.faiss_kv_bank import FaissKVBank


def _layer_kv_from_past_key_values(pkv: Any, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract (K, V) for one layer from `past_key_values`.

    Newer transformers return a `DynamicCache` (or `Cache`) that is not subscriptable like a tuple;
    older code used `tuple[tuple[K,V], ...]` indexed by layer.
    """
    if pkv is None:
        raise RuntimeError("past_key_values is None")
    # Legacy: tuple of (K, V) per layer
    try:
        pair = pkv[layer_idx]
        if isinstance(pair, (tuple, list)) and len(pair) >= 2:
            return pair[0], pair[1]
    except (TypeError, IndexError, KeyError):
        pass
    # Cache / DynamicCache: per-layer tensors on .layers[layer_idx]
    layers = getattr(pkv, "layers", None)
    if isinstance(layers, list) and layer_idx < len(layers):
        layer = layers[layer_idx]
        k = getattr(layer, "keys", None)
        v = getattr(layer, "values", None)
        if k is not None and v is not None:
            return k, v
    raise RuntimeError(f"Cannot read K/V for layer {layer_idx} from past_key_values type {type(pkv)}")


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


@dataclass(frozen=True)
class BuildBlocksKVBankStats:
    total_read: int
    total_written: int
    skipped_bad_len: int
    skipped_errors: int
    text_written: int = 0
    tables_written: int = 0
    shards_written: int = 0
    text_shards_written: int = 0
    tables_shards_written: int = 0


def build_kvbank_from_blocks_jsonl(
    *,
    blocks_jsonl: Path,
    out_dir: Path,
    split_tables: bool = False,
    out_dir_tables: Optional[Path] = None,
    base_llm_name_or_path: str,
    retrieval_encoder_model: str,
    layers: Sequence[int] = tuple(range(16)),
    block_tokens: int = 256,
    max_blocks: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    trust_remote_code: bool = True,
    shard_size: Optional[int] = None,
    table_pipe_threshold: int = 8,
) -> BuildBlocksKVBankStats:
    """
    对每条 block：
    - 用 base LLM tokenizer 截断到 block_tokens，并 forward 抽取 past_key_values(layers)
    - 用 DomainEncoder 编码 block 文本得到 retrieval_key
    - 写入 FaissKVBank（multi-layer KV）
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    if split_tables:
        if out_dir_tables is None:
            raise ValueError("split_tables=True requires out_dir_tables")
        out_dir_tables.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_dtype = None
    if dtype:
        torch_dtype = getattr(torch, dtype)
    elif dev.type == "cuda":
        torch_dtype = torch.bfloat16

    print(f"[blocks_to_kvbank] Loading base tokenizer/model: {base_llm_name_or_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(base_llm_name_or_path, use_fast=True, trust_remote_code=bool(trust_remote_code))
    model = AutoModelForCausalLM.from_pretrained(
        base_llm_name_or_path, torch_dtype=torch_dtype, trust_remote_code=bool(trust_remote_code)
    )
    model.to(dev)
    model.eval()
    print(f"[blocks_to_kvbank] model_loaded device={dev.type} dtype={torch_dtype}", flush=True)

    num_layers = int(model.config.num_hidden_layers)
    layer_ids = list(layers)
    for li in layer_ids:
        if li < 0 or li >= num_layers:
            raise ValueError(f"layer {li} out of range [0,{num_layers-1}]")

    print(f"[blocks_to_kvbank] Loading retrieval encoder: {retrieval_encoder_model}", flush=True)
    encoder = HFSentenceEncoder(
        HFSentenceEncoderConfig(model_name_or_path=retrieval_encoder_model, max_length=block_tokens, normalize=True)
    )

    retrieval_keys_text: List[np.ndarray] = []
    metas_text: List[Dict[str, Any]] = []
    k_list_text: List[np.ndarray] = []
    v_list_text: List[np.ndarray] = []

    retrieval_keys_tbl: List[np.ndarray] = []
    metas_tbl: List[Dict[str, Any]] = []
    k_list_tbl: List[np.ndarray] = []
    v_list_tbl: List[np.ndarray] = []

    total_read = 0
    total_written = 0
    text_written = 0
    tables_written = 0
    skipped_bad_len = 0
    skipped_errors = 0
    shards_written = 0
    text_shards_written = 0
    tables_shards_written = 0
    error_types: Dict[str, int] = {}
    first_error: Optional[str] = None

    import re
    _TABLE_MARKER_RE = re.compile(r"<\s*!\s*-\s*-\s*table\s*:\s*(\d+)\s*-\s*-\s*>", flags=re.IGNORECASE)

    def _is_table_block(text: str, meta: Dict[str, Any]) -> bool:
        tables = meta.get("tables") if isinstance(meta, dict) else None
        if isinstance(tables, dict):
            ids = tables.get("table_ids") or []
            if isinstance(ids, (list, tuple)) and len(ids) > 0:
                return True
        if _TABLE_MARKER_RE.search(text):
            return True
        if int(text.count("|")) >= int(table_pipe_threshold):
            return True
        return False

    def _list_feature_meta(text: str, meta_payload: Dict[str, Any]) -> Dict[str, Any]:
        pat = meta_payload.get("pattern") if isinstance(meta_payload.get("pattern"), dict) else {}
        lf = pat.get("list_features") if isinstance(pat.get("list_features"), dict) else {}
        list_items = lf.get("list_items") if isinstance(lf.get("list_items"), list) else []
        if not list_items:
            list_items = lf.get("list_like_items") if isinstance(lf.get("list_like_items"), list) else []
        list_like = bool(lf.get("is_list_like") or list_items or lf.get("has_bullets") or lf.get("has_enumeration"))
        list_feature_count = int(len(list_items or []))
        signals = lf.get("signals") if isinstance(lf.get("signals"), list) else []
        has_symptom_cue = any("trigger_phrase" in str(s) for s in signals) or any(
            "clinical" in str(s).lower() or "symptom" in str(s).lower() for s in (pat.get("schema_slots") or [])
        )
        has_enum = bool(lf.get("has_bullets") or lf.get("has_enumeration")) or any(
            "bullet" in str(s) or "numbering" in str(s) for s in signals
        )
        has_location_cue = any("paren_cases" in str(s) for s in signals) or any(
            str(s).lower().startswith("trigger_phrase:") for s in signals
        )
        list_confidence = min(
            1.0,
            0.3 * float(list_feature_count)
            + (0.1 if has_symptom_cue else 0.0)
            + (0.1 if has_enum else 0.0)
            + (0.1 if has_location_cue else 0.0),
        )
        # Prefer semantic_type inferred by the list_feature extractor (stored inside list_features),
        # because schema_slots can be absent for many evidence blocks.
        # This is still "semantic_type-level" and does not encode topic/question specific rules.
        inferred_list_type = str(lf.get("list_type") or "").strip().lower()
        if inferred_list_type in {"location", "symptom", "drug", "clinical_feature", "other"}:
            list_type = inferred_list_type
        else:
            slots = pat.get("schema_slots") if isinstance(pat.get("schema_slots"), list) else []
            slot_low = [str(s).lower() for s in slots if str(s).strip()]
            if any("clinical" in s or "symptom" in s for s in slot_low):
                list_type = "symptom"
            elif any("clinical_feature" in s for s in slot_low):
                list_type = "clinical_feature"
            elif any(
                ("geographic" in s)
                or ("distribution" in s)
                or ("region" in s)
                or ("location" in s)
                or ("epidemiolog" in s)
                for s in slot_low
            ):
                list_type = "location"
            else:
                list_type = "other"
        list_features = []
        if list_like:
            list_features.append(
                {
                    "type": list_type,
                    "items": list_items,
                    "source_span": (text[:200] + "...") if len(text) > 200 else text,
                }
            )
        return {
            "list_like": bool(list_like),
            "list_feature_count": int(list_feature_count),
            "list_features": list_features,
            "list_confidence": float(list_confidence),
            "list_signals": [str(s) for s in signals if str(s).strip()],
            "list_type": str(list_type),
        }

    def _flush_bank(
        *,
        rk: List[np.ndarray],
        kl: List[np.ndarray],
        vl: List[np.ndarray],
        ms: List[Dict[str, Any]],
        root: Path,
        shard_idx: int,
    ) -> None:
        shard_root = root / "shards"
        shard_root.mkdir(parents=True, exist_ok=True)
        shard_dir = shard_root / f"{shard_idx:05d}"
        print(f"[blocks_to_kvbank] flushing_shard={shard_idx} items={len(rk)} out_dir={shard_dir}", flush=True)
        bank = FaissKVBank.build(
            retrieval_keys=np.stack(rk, axis=0),
            k_ext=np.stack(kl, axis=0),
            v_ext=np.stack(vl, axis=0),
            metas=ms,
            normalize=True,
            metric="ip",
        )
        bank.save(shard_dir)
        rk.clear()
        kl.clear()
        vl.clear()
        ms.clear()

    for idx, rec in enumerate(_read_jsonl(blocks_jsonl), start=1):
        total_read += 1
        if max_blocks is not None and total_written >= max_blocks:
            break

        text = (rec.get("text") or "").strip()
        if not text:
            skipped_bad_len += 1
            continue

        try:
            # enforce 256-token block at tokenizer level
            inputs = tok(text, return_tensors="pt", truncation=True, max_length=block_tokens)
            input_ids = inputs["input_ids"].to(dev)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(dev)
            kv_len = int(input_ids.shape[1])
            if kv_len > block_tokens:
                kv_len = block_tokens

            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
            pkv = out.past_key_values
            if pkv is None:
                raise RuntimeError("past_key_values missing")

            # build per-layer padded K/V to block_tokens
            K0, _ = _layer_kv_from_past_key_values(pkv, layer_ids[0])
            kv_heads = int(K0.shape[1])
            head_dim = int(K0.shape[-1])

            per_layer_k: List[np.ndarray] = []
            per_layer_v: List[np.ndarray] = []
            for li in layer_ids:
                K, V = _layer_kv_from_past_key_values(pkv, li)  # [1, kv_heads, T, head_dim]
                K = K[0, :, :kv_len, :]
                V = V[0, :, :kv_len, :]
                K_pad = torch.zeros((kv_heads, block_tokens, head_dim), device="cpu", dtype=torch.float32)
                V_pad = torch.zeros((kv_heads, block_tokens, head_dim), device="cpu", dtype=torch.float32)
                K_pad[:, :kv_len, :] = K.detach().to(device="cpu", dtype=torch.float32)
                V_pad[:, :kv_len, :] = V.detach().to(device="cpu", dtype=torch.float32)
                per_layer_k.append(K_pad.numpy())
                per_layer_v.append(V_pad.numpy())

            k_item = np.stack(per_layer_k, axis=0)  # [L, kv_heads, block_tokens, head_dim]
            v_item = np.stack(per_layer_v, axis=0)

            key = encoder.encode(text)[0].astype(np.float32)

            meta_payload = rec.get("metadata") or {}
            is_table = bool(_is_table_block(text, meta_payload))
            # Schema blocks may carry a top-level `slots` field (recommended) or `metadata.slots` (fallback).
            slots = rec.get("slots", None)
            if not isinstance(slots, list):
                slots = meta_payload.get("slots") if isinstance(meta_payload, dict) else None
            if not isinstance(slots, list):
                slots = []
            slots = [str(s) for s in slots if isinstance(s, (str, int, float)) and str(s).strip()]

            # Schema blocks may also carry `answerable_slots` (preferred) which is the subset of slots
            # substantively covered by evidence. Runtime selection prefers this field.
            answerable_slots = rec.get("answerable_slots", None)
            if not isinstance(answerable_slots, list):
                answerable_slots = meta_payload.get("answerable_slots") if isinstance(meta_payload, dict) else None
            # Back-compat: schema builder always stores an unfiltered trace under metadata.inferred_answerable_slots.
            # If upstream accidentally filtered declared slots too narrowly, use the inferred list as a conservative
            # retrieval-side signal (still only affects injection eligibility/prioritization).
            if not isinstance(answerable_slots, list) or not answerable_slots:
                answerable_slots = (
                    meta_payload.get("inferred_answerable_slots") if isinstance(meta_payload, dict) else None
                )
            if not isinstance(answerable_slots, list):
                answerable_slots = []
            answerable_slots = [
                str(s) for s in answerable_slots if isinstance(s, (str, int, float)) and str(s).strip()
            ]

            list_meta = _list_feature_meta(text, meta_payload)
            meta = {
                "block_id": rec.get("block_id"),
                "parent_chunk_id": rec.get("parent_chunk_id"),
                "source_id": rec.get("source_id"),
                "doc_id": rec.get("doc_id"),
                "lang": rec.get("lang"),
                "source_uri": rec.get("source_uri"),
                "ocr_used": rec.get("ocr_used"),
                "layer_ids": layer_ids,
                "kv_len": kv_len,
                "max_kv_tokens": block_tokens,
                "citation": rec.get("block_id"),
                # slot availability (schema-first selector uses this; non-schema blocks can be empty)
                "slots": slots,
                # Preferred gating slots for schema-first selection.
                "answerable_slots": answerable_slots,
                # carry structured metadata (tables/disease/date/paragraph_type, etc.)
                "metadata": meta_payload,
                "is_table": bool(is_table),
                "list_like": bool(list_meta.get("list_like")),
                "list_feature_count": int(list_meta.get("list_feature_count") or 0),
                "list_features": list_meta.get("list_features") or [],
                "list_confidence": float(list_meta.get("list_confidence") or 0.0),
                # Flattened list feature signals/types for downstream ranking/debug.
                # (Runtime uses these to boost high-precision signals like paren_cases_capture.)
                "list_signals": list_meta.get("list_signals") if isinstance(list_meta.get("list_signals"), list) else [],
                "list_type": str(list_meta.get("list_type") or ""),
            }

            total_written += 1
            if split_tables and is_table:
                retrieval_keys_tbl.append(key)
                k_list_tbl.append(k_item)
                v_list_tbl.append(v_item)
                metas_tbl.append(meta)
                tables_written += 1
            else:
                retrieval_keys_text.append(key)
                k_list_text.append(k_item)
                v_list_text.append(v_item)
                metas_text.append(meta)
                text_written += 1
            if idx % 50 == 0:
                print(
                    f"[blocks_to_kvbank] processed_blocks={idx} written={total_written} skipped_empty_text={skipped_bad_len} skipped_errors={skipped_errors}",
                    flush=True,
                )

        except Exception as e:
            skipped_errors += 1
            name = type(e).__name__
            error_types[name] = int(error_types.get(name, 0)) + 1
            if first_error is None:
                first_error = f"{name}: {e}"
            # Always log the skipped sentence so data loss is visible
            bid = rec.get("block_id") or rec.get("id") or f"idx_{idx}"
            print(
                f"[blocks_to_kvbank] SKIPPED block={bid} "
                f"text={text[:60]!r} error={name}: {e}",
                flush=True,
            )
            continue

        # Sharded flushing (方案A): write every `shard_size` items to disk to control RAM.
        if shard_size is not None and shard_size > 0:
            if len(retrieval_keys_text) >= int(shard_size):
                _flush_bank(
                    rk=retrieval_keys_text,
                    kl=k_list_text,
                    vl=v_list_text,
                    ms=metas_text,
                    root=out_dir,
                    shard_idx=text_shards_written,
                )
                text_shards_written += 1
            if split_tables and len(retrieval_keys_tbl) >= int(shard_size):
                _flush_bank(
                    rk=retrieval_keys_tbl,
                    kl=k_list_tbl,
                    vl=v_list_tbl,
                    ms=metas_tbl,
                    root=out_dir_tables or out_dir,
                    shard_idx=tables_shards_written,
                )
                tables_shards_written += 1

    # ---- Compile summary (always print, highlight data loss) ----
    print(
        f"[blocks_to_kvbank] compile_summary: "
        f"read={total_read} written={total_written} "
        f"skipped_empty={skipped_bad_len} skipped_errors={skipped_errors} "
        f"layers={list(layers)} block_tokens={block_tokens}",
        flush=True,
    )
    if skipped_errors > 0:
        top = sorted(error_types.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print(
            f"[blocks_to_kvbank] WARNING: {skipped_errors} sentence(s) LOST during compilation! "
            f"error_types={dict(top)} first_error={first_error}",
            flush=True,
        )

    if total_written == 0:
        hint = (
            f"No blocks written from {blocks_jsonl}. "
            f"read={total_read}, skipped_empty_text={skipped_bad_len}, skipped_errors={skipped_errors}. "
        )
        if skipped_errors > 0:
            top = sorted(error_types.items(), key=lambda kv: kv[1], reverse=True)[:5]
            hint += f"error_types={dict(top)}. first_error={first_error}. "
            hint += "Common fixes: try a smaller base model, set device=cpu, ensure enough GPU RAM, and confirm model supports past_key_values."
        else:
            hint += "Common fixes: ensure blocks.jsonl has non-empty 'text' fields; rerun blocks step with --keep_last_incomplete_block."
        raise RuntimeError(hint)

    def _finalize(root: Path, rk: List[np.ndarray], kl: List[np.ndarray], vl: List[np.ndarray], ms: List[Dict[str, Any]], shard_count: int) -> int:
        if shard_size is not None and shard_size > 0:
            if rk:
                _flush_bank(rk=rk, kl=kl, vl=vl, ms=ms, root=root, shard_idx=shard_count)
                shard_count += 1
            shard_rel = [f"shards/{i:05d}" for i in range(shard_count)]
            manifest = {
                "format": "sharded",
                "shard_size": int(shard_size),
                "shards": shard_rel,
                "block_tokens": int(block_tokens),
                "layers": list(layers),
                "retrieval_encoder_model": str(retrieval_encoder_model),
                "base_llm_name_or_path": str(base_llm_name_or_path),
            }
            root.mkdir(parents=True, exist_ok=True)
            (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[blocks_to_kvbank] Sharded KVBank saved. shards={shard_count} out_dir={root}", flush=True)
            return shard_count

        if not rk:
            # allow empty tables bank when split_tables=True
            print(f"[blocks_to_kvbank] KVBank empty (no items) out_dir={root}", flush=True)
            return 0

        print(f"[blocks_to_kvbank] Building FAISS KVBank: written={len(rk)} out_dir={root}", flush=True)
        bank = FaissKVBank.build(
            retrieval_keys=np.stack(rk, axis=0),
            k_ext=np.stack(kl, axis=0),
            v_ext=np.stack(vl, axis=0),
            metas=ms,
            normalize=True,
            metric="ip",
        )
        bank.save(root)
        print("[blocks_to_kvbank] KVBank saved.", flush=True)
        return 0

    # finalize text bank
    shards_written = _finalize(out_dir, retrieval_keys_text, k_list_text, v_list_text, metas_text, text_shards_written)

    # finalize tables bank (optional)
    if split_tables:
        _finalize(out_dir_tables or out_dir, retrieval_keys_tbl, k_list_tbl, v_list_tbl, metas_tbl, tables_shards_written)

    return BuildBlocksKVBankStats(
        total_read=total_read,
        total_written=total_written,
        text_written=int(text_written),
        tables_written=int(tables_written),
        skipped_bad_len=skipped_bad_len,
        skipped_errors=skipped_errors,
        shards_written=int(shards_written),
        text_shards_written=int(text_shards_written),
        tables_shards_written=int(tables_shards_written),
    )


