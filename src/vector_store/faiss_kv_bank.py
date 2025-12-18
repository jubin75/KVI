"""
FAISS KV Bank（Demo 实现）

你当前选择：
- 1) 向量索引：本地 FAISS（文件落盘）
- 2) KV Bank 存储：直接存 `K_ext/V_ext`（检索命中后可直接注入 attention）

重要说明（工程现实）
- FAISS 只能对“检索向量（retrieval_key）”建立 ANN 索引；因此每条知识条目至少需要：
  - retrieval_key: shape [d] 的 float32（用于 ANN 检索）
  - K_ext/V_ext: 供注入使用的张量（通常更大）
- 本文件的实现是 demo 级：适合小规模（知识库不大）验证链路；生产级会加入分片、增量构建、压缩/量化、过滤策略等。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


def _require_faiss() -> Any:
    try:
        import faiss  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "faiss is required for the demo FAISS KV bank. "
            "Install one of: faiss-cpu / faiss-gpu."
        ) from e
    return faiss


def _as_float32(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class KVItem:
    """
    一条可注入知识条目（chunk 级）。

    - score：ANN 相似度分数（内积/余弦）
    - meta：必须至少包含 chunk_id 与 citation（用于可解释性）
    - K_ext/V_ext：支持两种形态
      - 4D: [kv_heads, ext_len, head_dim]（单层 KV）
      - 5D: [L, kv_heads, ext_len, head_dim]（多层 KV；L 对应 layer_ids）
    """

    score: float
    meta: Dict[str, Any]
    K_ext: np.ndarray
    V_ext: np.ndarray

    def get_kv_for_layer(self, layer_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回指定 layer_id 的 (K,V)：shape [kv_heads, ext_len, head_dim]。
        要求 meta["layer_ids"] 存在且与 5D 的第一维对应。
        """

        if self.K_ext.ndim == 4 and self.V_ext.ndim == 4:
            return self.K_ext, self.V_ext

        layer_ids = self.meta.get("layer_ids")
        if not isinstance(layer_ids, list):
            raise ValueError("KVItem.meta['layer_ids'] must exist for multi-layer KV")
        try:
            li = layer_ids.index(layer_id)
        except ValueError as e:
            raise KeyError(f"layer_id {layer_id} not present in layer_ids={layer_ids}") from e

        if self.K_ext.ndim != 5 or self.V_ext.ndim != 5:
            raise ValueError("Unexpected KV tensor rank for multi-layer KV")
        return self.K_ext[li], self.V_ext[li]


class FaissKVBank:
    """
    demo 级 KV Bank：
    - 用 FAISS 索引 `retrieval_keys`（shape [N, d]）
    - 同步存储 K_ext/V_ext（shape [N, ...]）
    - 保存/加载到本地目录
    """

    def __init__(
        self,
        index: Any,
        retrieval_keys: np.ndarray,
        k_ext: np.ndarray,
        v_ext: np.ndarray,
        metas: List[Dict[str, Any]],
        *,
        normalize: bool,
        metric: str,
    ) -> None:
        self._faiss = _require_faiss()
        self.index = index
        self.retrieval_keys = retrieval_keys
        self.k_ext = k_ext
        self.v_ext = v_ext
        self.metas = metas
        self.normalize = normalize
        self.metric = metric

        if len(self.metas) != self.retrieval_keys.shape[0]:
            raise ValueError("metas length must match number of vectors")

        if self.k_ext.shape[0] != self.retrieval_keys.shape[0]:
            raise ValueError("k_ext first dim must match number of vectors")
        if self.v_ext.shape[0] != self.retrieval_keys.shape[0]:
            raise ValueError("v_ext first dim must match number of vectors")

        # multi-layer KV 时，要求每条 meta 都包含 layer_ids（用于映射 K/V 的第一维）
        if self.k_ext.ndim == 5 or self.v_ext.ndim == 5:
            for m in self.metas:
                if "layer_ids" not in m:
                    raise ValueError("meta['layer_ids'] is required when storing multi-layer K/V")

    @property
    def size(self) -> int:
        return int(self.retrieval_keys.shape[0])

    @property
    def dim(self) -> int:
        return int(self.retrieval_keys.shape[1])

    @classmethod
    def build(
        cls,
        *,
        retrieval_keys: np.ndarray,
        k_ext: np.ndarray,
        v_ext: np.ndarray,
        metas: List[Dict[str, Any]],
        normalize: bool = True,
        metric: str = "ip",
        index_factory: Optional[str] = None,
    ) -> "FaissKVBank":
        """
        构建 FAISS 索引并返回 KVBank。

        - metric="ip"：内积（若 normalize=True 则等价于 cosine）
        - index_factory：FAISS index_factory 字符串（demo 默认 Flat）
        """

        faiss = _require_faiss()

        retrieval_keys = _as_float32(np.asarray(retrieval_keys))
        if retrieval_keys.ndim != 2:
            raise ValueError("retrieval_keys must be 2D: [N, d]")

        if normalize:
            retrieval_keys = _l2_normalize(retrieval_keys)

        d = retrieval_keys.shape[1]

        if metric not in {"ip"}:
            raise ValueError("demo only supports metric='ip' (inner product)")

        if index_factory is None:
            # demo：FlatIP，适合小规模；生产可换 IVF/HNSW/PQ 等
            index = faiss.IndexFlatIP(d)
        else:
            index = faiss.index_factory(d, index_factory, faiss.METRIC_INNER_PRODUCT)

        index.add(retrieval_keys)
        return cls(
            index=index,
            retrieval_keys=retrieval_keys,
            k_ext=np.asarray(k_ext),
            v_ext=np.asarray(v_ext),
            metas=metas,
            normalize=normalize,
            metric=metric,
        )

    def save(self, dir_path: Union[str, Path]) -> None:
        """
        保存 KV Bank 到目录：
        - index.faiss
        - retrieval_keys.npy
        - k_ext.npy / v_ext.npy
        - metas.jsonl
        - manifest.json
        """

        faiss = _require_faiss()
        out = _ensure_dir(dir_path)

        faiss.write_index(self.index, str(out / "index.faiss"))
        np.save(out / "retrieval_keys.npy", self.retrieval_keys)
        np.save(out / "k_ext.npy", self.k_ext)
        np.save(out / "v_ext.npy", self.v_ext)

        with (out / "metas.jsonl").open("w", encoding="utf-8") as f:
            for m in self.metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        manifest = {
            "size": self.size,
            "dim": self.dim,
            "normalize": self.normalize,
            "metric": self.metric,
            "k_ext_shape": list(self.k_ext.shape),
            "v_ext_shape": list(self.v_ext.shape),
            "files": {
                "index": "index.faiss",
                "retrieval_keys": "retrieval_keys.npy",
                "k_ext": "k_ext.npy",
                "v_ext": "v_ext.npy",
                "metas": "metas.jsonl",
            },
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, dir_path: Union[str, Path]) -> "FaissKVBank":
        faiss = _require_faiss()
        p = Path(dir_path)
        manifest_path = p / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in: {p}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Sharded KVBank: return a bank-like object that implements the same `search()` API.
        if isinstance(manifest, dict) and manifest.get("format") == "sharded":
            return ShardedFaissKVBank.load(p)  # type: ignore[return-value]

        index = faiss.read_index(str(p / manifest["files"]["index"]))
        # IMPORTANT:
        # Use mmap to avoid loading the full K/V tensors into RAM. They can be multi-GB per shard.
        # Downstream we only slice a small number of blocks for injection, so mmap is ideal.
        retrieval_keys = np.load(p / manifest["files"]["retrieval_keys"], mmap_mode="r")
        k_ext = np.load(p / manifest["files"]["k_ext"], mmap_mode="r")
        v_ext = np.load(p / manifest["files"]["v_ext"], mmap_mode="r")

        metas: List[Dict[str, Any]] = []
        with (p / manifest["files"]["metas"]).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                metas.append(json.loads(line))

        return cls(
            index=index,
            retrieval_keys=retrieval_keys,
            k_ext=k_ext,
            v_ext=v_ext,
            metas=metas,
            normalize=bool(manifest.get("normalize", True)),
            metric=str(manifest.get("metric", "ip")),
        )

    def search(
        self,
        query: np.ndarray,
        *,
        top_k: int = 32,
        filters: Optional[Dict[str, Any]] = None,
        oversample: int = 5,
    ) -> Tuple[List[KVItem], Dict[str, Any]]:
        """
        搜索 top-k（单 query demo）。

        filters（demo 版）
        - 仅做后过滤：要求 meta 中字段值相等或属于列表（如 lang in ["zh","en"]）
        """

        if top_k <= 0:
            return [], {"top_k": top_k, "filtered": 0}

        q = _as_float32(np.asarray(query))
        if q.ndim == 1:
            q = q[None, :]
        if q.shape[1] != self.dim:
            raise ValueError(f"query dim {q.shape[1]} != bank dim {self.dim}")
        if q.shape[0] != 1:
            raise ValueError("demo search only supports a single query (shape [d] or [1,d])")

        if self.normalize:
            q = _l2_normalize(q)

        # oversample 以便后过滤仍能得到 top_k
        k_search = min(self.size, max(top_k, top_k * oversample))
        scores, idxs = self.index.search(q, k_search)
        scores = scores[0]
        idxs = idxs[0]

        def _pass_filters(meta: Dict[str, Any]) -> bool:
            if not filters:
                return True
            for k, v in filters.items():
                mv = meta.get(k)
                if isinstance(v, (list, tuple, set)):
                    if mv not in v:
                        return False
                else:
                    if mv != v:
                        return False
            return True

        items: List[KVItem] = []
        filtered_out = 0
        for s, i in zip(scores, idxs):
            if i < 0:
                continue
            meta = self.metas[int(i)]
            if not _pass_filters(meta):
                filtered_out += 1
                continue
            items.append(
                KVItem(
                    score=float(s),
                    meta=meta,
                    K_ext=self.k_ext[int(i)],
                    V_ext=self.v_ext[int(i)],
                )
            )
            if len(items) >= top_k:
                break

        debug = {
            "top_k": int(top_k),
            "k_searched": int(k_search),
            "filtered_out": int(filtered_out),
            "size": int(self.size),
            "dim": int(self.dim),
            "metric": str(self.metric),
            "normalize": bool(self.normalize),
        }
        return items, debug


class ShardedFaissKVBank:
    """
    A sharded wrapper around multiple `FaissKVBank` shards.

    Disk layout:
      out_dir/
        manifest.json            # format="sharded", shards=["shards/00000", ...]
        shards/00000/manifest.json  # normal FaissKVBank manifest
        shards/00001/...

    Search strategy:
      - Run per-shard search with `per_shard_top_k`
      - Merge by score (global top_k)
      - Optionally apply the same metadata filters
    """

    def __init__(self, *, shards: List[FaissKVBank], shard_dirs: List[Path]) -> None:
        if not shards:
            raise ValueError("ShardedFaissKVBank requires at least 1 shard")
        self.shards = shards
        self.shard_dirs = shard_dirs

        # sanity: all dims must match
        d0 = shards[0].dim
        for s in shards[1:]:
            if s.dim != d0:
                raise ValueError(f"Shard dim mismatch: {s.dim} != {d0}")

    @property
    def size(self) -> int:
        return int(sum(s.size for s in self.shards))

    @property
    def dim(self) -> int:
        return int(self.shards[0].dim)

    @classmethod
    def load(cls, out_dir: Union[str, Path]) -> "ShardedFaissKVBank":
        p = Path(out_dir)
        manifest_path = p / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found in: {p}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("format") != "sharded":
            raise ValueError(f"Not a sharded KVBank manifest: {manifest_path}")

        shard_rel = manifest.get("shards") or []
        if not isinstance(shard_rel, list) or not shard_rel:
            raise ValueError("sharded manifest missing non-empty 'shards' list")

        shard_dirs: List[Path] = []
        shards: List[FaissKVBank] = []
        for rel in shard_rel:
            sd = (p / str(rel)).resolve()
            shard_dirs.append(sd)
            shards.append(FaissKVBank.load(sd))
        return cls(shards=shards, shard_dirs=shard_dirs)

    def search(
        self,
        query: np.ndarray,
        *,
        top_k: int = 32,
        filters: Optional[Dict[str, Any]] = None,
        oversample: int = 5,
        per_shard_top_k: Optional[int] = None,
    ) -> Tuple[List[KVItem], Dict[str, Any]]:
        if top_k <= 0:
            return [], {"top_k": top_k, "shards": len(self.shards), "filtered_out": 0}

        # Ensure enough candidates from each shard; this prevents "missed" global top_k when many shards exist.
        if per_shard_top_k is None:
            per_shard_top_k = max(1, int(top_k))

        all_items: List[KVItem] = []
        shard_debug: List[Dict[str, Any]] = []
        for i, s in enumerate(self.shards):
            items, dbg = s.search(query, top_k=per_shard_top_k, filters=filters, oversample=oversample)
            all_items.extend(items)
            shard_debug.append({"shard": i, "dir": str(self.shard_dirs[i]), **dbg})

        # Merge by score (descending)
        all_items.sort(key=lambda it: float(it.score), reverse=True)
        merged = all_items[: min(int(top_k), len(all_items))]
        debug = {
            "top_k": int(top_k),
            "shards": int(len(self.shards)),
            "per_shard_top_k": int(per_shard_top_k),
            "candidates": int(len(all_items)),
            "size": int(self.size),
            "dim": int(self.dim),
            "shard_debug": shard_debug[:5],  # keep small
        }
        return merged, debug


