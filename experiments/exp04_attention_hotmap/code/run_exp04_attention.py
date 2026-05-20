#!/usr/bin/env python3
"""
Exp04: RAG vs RAG+KV-prefix attention analysis (research harness).

Example:
  python experiments/exp04_attention_hotmap/code/run_exp04_attention.py \\
    --model /path/to/Qwen2.5-7B-Instruct \\
    --out_dir experiments/exp04_attention_hotmap/results/run01 \\
    --limit 200

See experiments/exp04_attention_hotmap/exp04.md for the full protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parent
for _p in (_REPO, _CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import exp04_plots  # noqa: E402
from exp04_lib import (  # noqa: E402
    Exp04Example,
    build_prompt_ids,
    first_answer_token_id,
    greedy_generate_attentions,
    load_examples_hf_hotpot,
    load_examples_jsonl,
    load_model,
    logit_for_token_first_step,
    per_layer_head_mass,
    forward_prefix_past,
    set_seed,
    triples_to_canonical_text,
)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Avoid RuntimeWarning / nan from corrcoef when variance is zero or n too small."""
    if x.size < 3 or y.size < 3 or x.shape != y.shape:
        return float("nan")
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return float("nan")
    m = np.corrcoef(x, y)
    v = m[0, 1]
    return float(v) if np.isfinite(v) else float("nan")


def _mean_pad_traj(rows: List[List[float]]) -> np.ndarray:
    if not rows:
        return np.array([])
    m = max(len(r) for r in rows)
    mat = np.full((len(rows), m), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        mat[i, : len(r)] = r
    return np.nanmean(mat, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(_REPO / "models/Qwen2.5-7B-Instruct"))
    p.add_argument("--out_dir", default=str(_REPO / "experiments/exp04_attention_hotmap/results/latest"))
    p.add_argument("--data_jsonl", default=str(_REPO / "experiments/exp01_main_qa/data/benchmarks/hotpot_eval.jsonl"))
    p.add_argument("--use_hf_hotpot", action="store_true", help="Load HotpotQA distractor from HF (needs network + datasets).")
    p.add_argument("--hotpot_config", default="distractor")
    p.add_argument("--hotpot_split", default="validation")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=24)
    p.add_argument("--max_evidence_chars", type=int, default=2000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--ablation_random_kv", action="store_true")
    args = p.parse_args()

    set_seed(int(args.seed))
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        dtype = torch.float32

    if bool(args.use_hf_hotpot):
        examples = load_examples_hf_hotpot(
            config=str(args.hotpot_config),
            split=str(args.hotpot_split),
            max_examples=int(args.limit),
            seed=int(args.seed),
        )
    else:
        examples = load_examples_jsonl(Path(args.data_jsonl), limit=int(args.limit), seed=int(args.seed))

    if not examples:
        raise SystemExit("No examples loaded; check --data_jsonl or --use_hf_hotpot.")

    model, tok = load_model(str(args.model), device=device, dtype=dtype)

    mats_rag: List[np.ndarray] = []
    mats_kvi_kv: List[np.ndarray] = []
    mats_kvi_ev: List[np.ndarray] = []
    mats_rand: List[np.ndarray] = []
    gains: List[float] = []
    gains_rand: List[float] = []
    mean_kv_last: List[float] = []
    traj_rag: List[List[float]] = []
    traj_kvi_ev: List[List[float]] = []

    # Qualitative demos (first 2 examples): scalar masses for bar chart
    demo_mass_rows: List[Tuple[str, List[str], List[float]]] = []

    for i, ex in enumerate(examples):
        prompt_ids, ev_s, ev_e = build_prompt_ids(
            tok,
            ex.evidence,
            ex.question,
            max_evidence_chars=int(args.max_evidence_chars),
        )
        triple_txt = triples_to_canonical_text(ex.triples)
        kv_ids = torch.tensor([tok.encode(triple_txt, add_special_tokens=False)], dtype=torch.long)

        past, kv_len = forward_prefix_past(model, kv_ids, None, device=device)

        if bool(args.ablation_random_kv):
            vs = min(getattr(tok, "vocab_size", 32000) - 1, 32000)
            rand_ids = torch.randint(50, max(51, vs), kv_ids.shape, device=device, dtype=torch.long)
            past_rand, kv_len_r = forward_prefix_past(model, rand_ids, None, device=device)
            if kv_len_r != kv_len:
                raise RuntimeError(f"random KV length mismatch {kv_len_r} vs {kv_len}")
        else:
            past_rand = None

        gen_rag = greedy_generate_attentions(
            model,
            tok,
            prompt_ids,
            past_key_values=None,
            kv_prefix_len=0,
            evidence_start=ev_s,
            evidence_end=ev_e,
            device=device,
            max_new_tokens=int(args.max_new_tokens),
        )
        gen_kvi = greedy_generate_attentions(
            model,
            tok,
            prompt_ids,
            past_key_values=past,
            kv_prefix_len=kv_len,
            evidence_start=ev_s,
            evidence_end=ev_e,
            device=device,
            max_new_tokens=int(args.max_new_tokens),
        )

        mat_rag = per_layer_head_mass(
            gen_rag["last_attentions"],
            key_start=ev_s,
            key_end=ev_e,
        )
        mat_kvi_kv = per_layer_head_mass(
            gen_kvi["last_attentions"],
            key_start=0,
            key_end=gen_kvi["attn_kv_end"],
        )
        mat_kvi_ev = per_layer_head_mass(
            gen_kvi["last_attentions"],
            key_start=int(gen_kvi["attn_ev_start"]),
            key_end=int(gen_kvi["attn_ev_end"]),
        )
        mats_rag.append(mat_rag)
        mats_kvi_kv.append(mat_kvi_kv)
        mats_kvi_ev.append(mat_kvi_ev)

        gold_t = first_answer_token_id(tok, ex.answer)
        lr = logit_for_token_first_step(
            model, prompt_ids, past_key_values=None, kv_prefix_len=0, token_id=gold_t, device=device
        )
        lk = logit_for_token_first_step(
            model, prompt_ids, past_key_values=past, kv_prefix_len=kv_len, token_id=gold_t, device=device
        )
        gains.append(lk - lr)
        mean_kv_last.append(float(np.mean(mat_kvi_kv)))

        traj_rag.append(gen_rag["step_ev_mass"])
        traj_kvi_ev.append(gen_kvi["step_ev_mass"])

        if past_rand is not None:
            gen_rand = greedy_generate_attentions(
                model,
                tok,
                prompt_ids,
                past_key_values=past_rand,
                kv_prefix_len=kv_len,
                evidence_start=ev_s,
                evidence_end=ev_e,
                device=device,
                max_new_tokens=int(args.max_new_tokens),
            )
            mats_rand.append(
                per_layer_head_mass(gen_rand["last_attentions"], key_start=0, key_end=kv_len)
            )
            lrand = logit_for_token_first_step(
                model,
                prompt_ids,
                past_key_values=past_rand,
                kv_prefix_len=kv_len,
                token_id=gold_t,
                device=device,
            )
            gains_rand.append(lrand - lr)

        if i < 2:
            demo_mass_rows.append(
                (
                    ex.ex_id,
                    [
                        "RAG → evidence",
                        "KVI → evidence (same keys)",
                        "KVI → KV-prefix",
                    ],
                    [
                        float(np.mean(mat_rag)),
                        float(np.mean(mat_kvi_ev)),
                        float(np.mean(mat_kvi_kv)),
                    ],
                )
            )

    rag_mean = np.mean(np.stack(mats_rag, axis=0), axis=0)
    kvi_kv_mean = np.mean(np.stack(mats_kvi_kv, axis=0), axis=0)
    kvi_ev_mean = np.mean(np.stack(mats_kvi_ev, axis=0), axis=0)
    delta_cross = kvi_kv_mean - rag_mean
    delta_paired = kvi_ev_mean - rag_mean

    n_layers, n_heads = rag_mean.shape
    flat_rank_cross: List[Tuple[int, int, float]] = []
    flat_rank_paired: List[Tuple[int, int, float]] = []
    for li in range(n_layers):
        for hi in range(n_heads):
            flat_rank_cross.append((li, hi, float(delta_cross[li, hi])))
            flat_rank_paired.append((li, hi, float(delta_paired[li, hi])))
    flat_rank_cross.sort(key=lambda x: -abs(x[2]))
    flat_rank_paired.sort(key=lambda x: -abs(x[2]))

    corr = _safe_pearson(np.asarray(gains, dtype=np.float64), np.asarray(mean_kv_last, dtype=np.float64))

    metrics: Dict[str, Any] = {
        "n_examples": len(examples),
        "model": str(args.model),
        "mean_logit_gain": float(np.mean(gains)),
        "std_logit_gain": float(np.std(gains)),
        "corr_logit_gain_vs_mean_kv_mass": corr,
        "mean_ev_mass_rag_last_token_global": float(np.mean(rag_mean)),
        "mean_ev_mass_kvi_same_keys_last_token_global": float(np.mean(kvi_ev_mean)),
        "mean_kv_prefix_mass_kvi_last_token_global": float(np.mean(kvi_kv_mean)),
        "mean_delta_paired_evidence_keys_kvi_minus_rag": float(np.mean(delta_paired)),
        "mean_delta_cross_kv_prefix_minus_rag_evidence": float(np.mean(delta_cross)),
        "top10_heads_delta_paired_same_evidence": [
            {"layer": l, "head": h, "delta": v} for l, h, v in flat_rank_paired[:10]
        ],
        "top10_heads_delta_cross_region": [{"layer": l, "head": h, "delta": v} for l, h, v in flat_rank_cross[:10]],
    }
    if gains_rand:
        metrics["mean_logit_gain_random_kv_vs_rag"] = float(np.mean(gains_rand))
        metrics["mean_logit_gain_kvi_minus_random"] = float(np.mean(np.array(gains) - np.array(gains_rand)))

    exp04_plots.write_metrics_json(out / "exp04_metrics.json", metrics)

    exp04_plots.plot_kv_mass_vs_layer_multi(
        [
            (rag_mean.mean(axis=1), "RAG → evidence keys"),
            (kvi_ev_mean.mean(axis=1), "KVI → evidence keys (paired)"),
            (kvi_kv_mean.mean(axis=1), "KVI → KV-prefix keys"),
        ],
        out / "fig01_attention_mass_vs_layer",
        title="Last-token attention mass vs layer (mean over heads)",
    )
    exp04_plots.plot_heatmap(
        rag_mean,
        "RAG: mass on evidence key positions (layer × head)",
        out / "fig02a_heatmap_rag_evidence",
    )
    exp04_plots.plot_heatmap(
        kvi_ev_mean,
        "KVI: mass on the same evidence key positions (layer × head)",
        out / "fig02b_heatmap_kvi_same_evidence",
    )
    exp04_plots.plot_heatmap(
        kvi_kv_mean,
        "KVI: mass on injected KV-prefix keys (layer × head)",
        out / "fig02c_heatmap_kvi_kv_prefix",
    )
    exp04_plots.plot_delta_heatmap(
        delta_paired,
        out / "fig03_paired_evidence_delta_kvi_minus_rag",
        title=r"$\Delta$ mass: KVI $-$ RAG on the same evidence keys (paired, paper)",
    )
    exp04_plots.plot_delta_heatmap(
        delta_cross,
        out / "fig03_cross_region_kv_prefix_minus_rag_evidence",
        title=r"$\Delta$ mass: KVI (KV-prefix) $-$ RAG (evidence) — different key regions (supplement)",
    )
    exp04_plots.plot_logit_gain_hist(gains, out / "fig05_logit_gain")
    exp04_plots.plot_top_heads(
        flat_rank_paired,
        out / "fig06_paired_evidence_top_heads",
        top_k=10,
        title="Top-10 heads by |Δ| on paired evidence keys (KVI − RAG)",
        xlabel=r"Paired $\Delta$ mass (same evidence span)",
    )
    exp04_plots.plot_top_heads(
        flat_rank_cross,
        out / "fig06_supplement_cross_region_top_heads",
        top_k=10,
        title="Top-10 heads by |Δ| (KV-prefix vs RAG evidence)",
        xlabel=r"Cross-region $\Delta$ mass",
    )

    tr_r = _mean_pad_traj(traj_rag)
    tr_ke = _mean_pad_traj(traj_kvi_ev)
    if tr_r.size and tr_ke.size:
        m = min(len(tr_r), len(tr_ke))
        exp04_plots.plot_trajectory(
            tr_r[:m],
            tr_ke[:m],
            out / "fig07_trajectory_paired_evidence",
            label_a="RAG (evidence keys)",
            label_b="KVI (evidence keys)",
            title="Mean attention to evidence keys over greedy decode steps",
        )

    for j, (eid, labels, vals) in enumerate(demo_mass_rows):
        exp04_plots.plot_paper_demo_mass_bars(
            labels,
            vals,
            f"Example {eid}: last-token mass (all layers × heads)",
            out / f"fig04_demo_mass_example{j}",
        )

    fig_pdfs = sorted(out.glob("fig*.pdf"))
    manifest = out / "figures_manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                f"Exp04 paper figures (PDF + PNG alongside each stem)",
                f"out_dir={out}",
                "",
                *[str(p) for p in fig_pdfs],
                "",
                f"Total: {len(fig_pdfs)} figure stems (each has .pdf and .png).",
            ]
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "ok": True,
                "out_dir": str(out),
                "figures_dir": str(out),
                "figures_manifest": str(manifest),
                "figure_pdfs": [str(p) for p in fig_pdfs],
                "n": len(examples),
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        flush=True,
    )
    print("\n=== Exp04 result figures (PDF) ===\n" + "\n".join(str(p) for p in fig_pdfs) + "\n", flush=True)


if __name__ == "__main__":
    main()
