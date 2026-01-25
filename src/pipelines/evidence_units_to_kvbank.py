"""
Pipeline: Authoring EvidenceUnits (approved-only) -> FAISS KVBank.

This implements the frozen requirements in:
- docs/11_Knowledge_Authoring_Layer.md
- docs/012_Authoring _FAISS_KVBank.md

Key constraints:
- Only approved evidence enters runtime KVBank.
- FAISS is used for semantic similarity search (embeddings of evidence semantic text).
- Structured fields live in metadata/KV (here stored in KVBank `metas` for demo simplicity).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from ..authoring.models import RuntimeEvidenceRecord
from ..encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig
from ..vector_store.faiss_kv_bank import FaissKVBank


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


@dataclass(frozen=True)
class BuildEvidenceKVBankStats:
    total_read: int
    approved_read: int
    total_written: int
    skipped_empty_text: int
    skipped_not_approved: int
    skipped_errors: int


def build_kvbank_from_authoring_evidence_jsonl(
    *,
    evidence_jsonl: Path,
    out_dir: Path,
    base_llm_name_or_path: str,
    retrieval_encoder_model: str,
    layers: Sequence[int] = (0, 1, 2, 3),
    max_tokens: int = 256,
    max_items: Optional[int] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    trust_remote_code: bool = True,
) -> BuildEvidenceKVBankStats:
    """
    Build an evidence KVBank from authoring runtime records jsonl.

    Input jsonl line format: RuntimeEvidenceRecord.to_dict()
      {
        "evidence_id": "...",
        "semantic_text": "...",
        "semantic_type": "...",
        "schema_id": "...",
        "slot_projection": {...},
        "contract": {...},
        ...
      }

    Output: a standard `FaissKVBank` directory containing:
    - index.faiss
    - retrieval_keys.npy
    - k_ext.npy / v_ext.npy   (generated from base LLM past_key_values)
    - metas.jsonl             (contains runtime-safe evidence metadata)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_dtype = None
    if dtype:
        torch_dtype = getattr(torch, dtype)
    elif dev.type == "cuda":
        torch_dtype = torch.bfloat16

    tok = AutoTokenizer.from_pretrained(base_llm_name_or_path, use_fast=True, trust_remote_code=bool(trust_remote_code))
    model = AutoModelForCausalLM.from_pretrained(
        base_llm_name_or_path, torch_dtype=torch_dtype, trust_remote_code=bool(trust_remote_code)
    )
    model.to(dev)
    model.eval()

    num_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    layer_ids = list(layers)
    for li in layer_ids:
        if li < 0 or (num_layers and li >= num_layers):
            raise ValueError(f"layer {li} out of range [0,{max(0, num_layers-1)}]")

    encoder = HFSentenceEncoder(
        HFSentenceEncoderConfig(model_name_or_path=retrieval_encoder_model, max_length=int(max_tokens), normalize=True)
    )

    retrieval_keys: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []
    k_list: List[np.ndarray] = []
    v_list: List[np.ndarray] = []

    total_read = 0
    approved_read = 0
    total_written = 0
    skipped_empty_text = 0
    skipped_not_approved = 0
    skipped_errors = 0

    for rec in _read_jsonl(evidence_jsonl):
        total_read += 1
        if max_items is not None and total_written >= int(max_items):
            break

        try:
            # Runtime evidence record is already approved-only, but keep a defensive check.
            status = str(rec.get("status") or "").strip().lower()
            if status and status != "approved":
                skipped_not_approved += 1
                continue

            r = RuntimeEvidenceRecord(
                evidence_id=str(rec.get("evidence_id") or ""),
                semantic_text=str(rec.get("semantic_text") or rec.get("claim") or ""),
                evidence_type=str(rec.get("evidence_type") or "clinical_guideline"),
                semantic_type=str(rec.get("semantic_type") or "generic"),
                schema_id=str(rec.get("schema_id") or ""),
                polarity=str(rec.get("polarity") or "neutral"),
                slot_projection=rec.get("slot_projection") if isinstance(rec.get("slot_projection"), dict) else {},
                external_refs=rec.get("external_refs") if isinstance(rec.get("external_refs"), dict) else {},
                provenance=rec.get("provenance") if isinstance(rec.get("provenance"), dict) else {},
                contract=rec.get("contract") if isinstance(rec.get("contract"), dict) else {},
            )
            txt = str(r.semantic_text or "").strip()
            if not txt:
                skipped_empty_text += 1
                continue
            if not str(r.evidence_id or "").strip():
                skipped_errors += 1
                continue

            approved_read += 1

            inputs = tok(txt, return_tensors="pt", truncation=True, max_length=int(max_tokens))
            input_ids = inputs["input_ids"].to(dev)
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(dev)
            kv_len = int(input_ids.shape[1])
            kv_len = min(kv_len, int(max_tokens))

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
            pkv = out.past_key_values
            if pkv is None:
                raise RuntimeError("past_key_values missing")

            K0, _ = pkv[layer_ids[0]]
            kv_heads = int(K0.shape[1])
            head_dim = int(K0.shape[-1])

            per_layer_k: List[np.ndarray] = []
            per_layer_v: List[np.ndarray] = []
            for li in layer_ids:
                K, V = pkv[li]  # [1, kv_heads, T, head_dim]
                K = K[0, :, :kv_len, :]
                V = V[0, :, :kv_len, :]
                K_pad = torch.zeros((kv_heads, int(max_tokens), head_dim), device="cpu", dtype=torch.float32)
                V_pad = torch.zeros((kv_heads, int(max_tokens), head_dim), device="cpu", dtype=torch.float32)
                K_pad[:, :kv_len, :] = K.detach().to(device="cpu", dtype=torch.float32)
                V_pad[:, :kv_len, :] = V.detach().to(device="cpu", dtype=torch.float32)
                per_layer_k.append(K_pad.numpy())
                per_layer_v.append(V_pad.numpy())

            k_item = np.stack(per_layer_k, axis=0)  # [L, kv_heads, max_tokens, head_dim]
            v_item = np.stack(per_layer_v, axis=0)

            key = encoder.encode(txt)[0].astype(np.float32)

            # FAISS vector_id is insertion order in this demo implementation.
            vector_id = int(len(retrieval_keys))

            meta: Dict[str, Any] = {
                # Compatibility keys: many downstream components expect block_id/chunk_id/id.
                # For Authoring evidence, we treat evidence_id as the stable identifier.
                "block_id": str(r.evidence_id),
                "chunk_id": str(r.evidence_id),
                "id": str(r.evidence_id),
                # Required by docs/012: explicit mapping between vector_id and evidence_id.
                "vector_id": int(vector_id),
                "evidence_id": str(r.evidence_id),
                "citation": str(r.evidence_id),
                "layer_ids": list(layer_ids),
                "kv_len": int(kv_len),
                "max_kv_tokens": int(max_tokens),
                # runtime-safe evidence fields (structure in KV/metadata, not in vector)
                "semantic_text": str(r.semantic_text),
                "evidence_type": str(r.evidence_type),
                "semantic_type": str(r.semantic_type),
                "schema_id": str(r.schema_id),
                "polarity": str(r.polarity),
                "slot_projection": r.slot_projection,
                "external_refs": r.external_refs,
                "provenance": r.provenance,
                "contract": r.contract,
            }
            # Helpful grounding fields commonly surfaced in logs/UI (optional).
            if isinstance(r.external_refs, dict):
                doc_id = r.external_refs.get("document_id") or r.external_refs.get("source_id")
                if doc_id is not None:
                    meta["doc_id"] = str(doc_id)
            if isinstance(r.external_refs, dict):
                src = r.external_refs.get("source_uri") or r.external_refs.get("url")
                if src is not None:
                    meta["source_uri"] = src

            retrieval_keys.append(key)
            k_list.append(k_item)
            v_list.append(v_item)
            metas.append(meta)
            total_written += 1
        except Exception:
            skipped_errors += 1
            continue

    if total_written <= 0:
        raise RuntimeError(
            f"No approved evidence written from {evidence_jsonl}. "
            f"read={total_read} approved_read={approved_read} skipped_empty_text={skipped_empty_text} "
            f"skipped_not_approved={skipped_not_approved} skipped_errors={skipped_errors}"
        )

    bank = FaissKVBank.build(
        retrieval_keys=np.stack(retrieval_keys, axis=0),
        k_ext=np.stack(k_list, axis=0),
        v_ext=np.stack(v_list, axis=0),
        metas=metas,
        normalize=True,
        metric="ip",
    )
    bank.save(out_dir)

    return BuildEvidenceKVBankStats(
        total_read=int(total_read),
        approved_read=int(approved_read),
        total_written=int(total_written),
        skipped_empty_text=int(skipped_empty_text),
        skipped_not_approved=int(skipped_not_approved),
        skipped_errors=int(skipped_errors),
    )

