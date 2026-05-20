"""Visualization for Exp04 (matplotlib)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=160)
    fig.savefig(path.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_kv_mass_vs_layer(
    mean_rag: np.ndarray,
    mean_kvi: np.ndarray,
    out: Path,
    *,
    label_rag: str = "RAG (evidence mass)",
    label_kvi: str = "KVI (KV-prefix mass)",
) -> None:
    """mean_rag, mean_kvi: shape [n_layers]"""
    x = np.arange(len(mean_rag))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, mean_rag, marker="o", label=label_rag)
    ax.plot(x, mean_kvi, marker="s", label=label_kvi)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean attention mass (last token)")
    ax.set_title("Attention mass vs layer (last generated token)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out)


def plot_kv_mass_vs_layer_multi(
    curves: List[Tuple[np.ndarray, str]],
    out: Path,
    *,
    title: str = "Attention mass vs layer (last generated token)",
    ylabel: str = "Mean attention mass (last token)",
) -> None:
    """curves: list of (per-layer mean vector, legend label)."""
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    markers = ["o", "s", "^", "D", "v"]
    x = np.arange(len(curves[0][0]))
    for i, (vec, lab) in enumerate(curves):
        ax.plot(x, vec, marker=markers[i % len(markers)], label=lab)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    _save(fig, out)


def plot_heatmap(mat: np.ndarray, title: str, out: Path, *, cbar_label: str = "mass") -> None:
    """mat: [n_layers, n_heads] — layers on y, heads on x (transpose for imshow convention)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(mat.T, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    _save(fig, out)


def plot_delta_heatmap(delta: np.ndarray, out: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    vmax = float(np.nanmax(np.abs(delta))) + 1e-9
    im = ax.imshow(delta.T, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\Delta$ mass")
    _save(fig, out)


def plot_logit_gain_hist(gains: List[float], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(gains, bins=30, color="#1976d2", edgecolor="white")
    ax.axvline(float(np.mean(gains)), color="red", linestyle="--", label=f"mean={np.mean(gains):.4f}")
    ax.set_xlabel(r"$\log P_{\mathrm{KVI}}(t^*) - \log P_{\mathrm{RAG}}(t^*)$ (first answer token)")
    ax.set_ylabel("Count")
    ax.set_title("Logit gain distribution")
    ax.legend()
    _save(fig, out)


def plot_top_heads(
    ranks: List[Tuple[int, int, float]],
    out: Path,
    *,
    top_k: int = 10,
    title: str = f"Top heads by |Δ| (paper)",
    xlabel: str = r"Mean $\Delta$ mass",
) -> None:
    ranks = sorted(ranks, key=lambda x: -abs(x[2]))[:top_k]
    labels = [f"L{l}H{h}" for l, h, _ in ranks]
    vals = [v for _, _, v in ranks]
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.barh(labels[::-1], vals[::-1], color="#1565c0")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
    _save(fig, out)


def plot_token_bars(
    labels: List[str],
    rag_vals: List[float],
    kvi_vals: List[float],
    title: str,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, rag_vals, w, label="RAG")
    ax.bar(x + w / 2, kvi_vals, w, label="KVI")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Attention mass (last token)")
    ax.set_title(title)
    ax.legend()
    _save(fig, out)


def plot_trajectory(
    mean_steps_a: np.ndarray,
    mean_steps_b: np.ndarray,
    out: Path,
    *,
    label_a: str = "RAG (evidence keys)",
    label_b: str = "KVI (evidence keys)",
    title: str = "Paired evidence attention over decode steps",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(len(mean_steps_a)), mean_steps_a, label=label_a)
    ax.plot(np.arange(len(mean_steps_b)), mean_steps_b, label=label_b)
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Mean mass (layers × heads)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, out)


def plot_paper_demo_mass_bars(labels: List[str], values: List[float], title: str, out: Path) -> None:
    """Single-example qualitative bar chart (e.g. RAG vs KVI on same evidence vs KV routing)."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    colors = ["#1565c0", "#2e7d32", "#6a1b9a", "#ef6c00"]
    xs = np.arange(len(labels))
    ax.bar(xs, values, color=[colors[i % len(colors)] for i in range(len(labels))], edgecolor="white")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean attention mass (last token, all L×H)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out)


def write_metrics_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
