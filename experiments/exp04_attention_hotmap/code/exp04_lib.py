"""
Exp04: head-level attention under RAG vs RAG+KV-prefix (standard HF only).

Conventions:
- RAG: single forward pass over full prompt (evidence + question + answer suffix).
- KVI: forward on canonical triple sentences to obtain past_key_values, then same prompt
  (no duplicate triple text in prompt — triples are compact S|R|O style).
- For RAG, "retrieval mass" = attention sum over *evidence* key positions (analogous region).
- For KVI, "KV mass" = attention sum over *injected prefix* key positions.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Exp04Example:
    ex_id: str
    question: str
    answer: str
    evidence: str
    triples: List[Tuple[str, str, str]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_examples_jsonl(
    path: Path,
    *,
    limit: int,
    seed: int,
) -> List[Exp04Example]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    rng = random.Random(seed)
    if limit and len(rows) > limit:
        rows = rng.sample(rows, limit)
    out: List[Exp04Example] = []
    for r in rows:
        qid = str(r.get("id") or f"ex_{len(out)}")
        q = str(r.get("question") or "").strip()
        a = str(r.get("answer") or "").strip()
        if not q or not a:
            continue
        ev = str(r.get("evidence") or r.get("retrieved_evidence") or "").strip()
        gss = r.get("gold_supporting_sentences")
        if not ev and isinstance(gss, list) and gss:
            ev = " ".join(str(s).strip() for s in gss if str(s).strip())
        if not ev:
            # Simulated RAG when corpus not shipped: short synthetic context (documented limitation).
            ev = (
                f"Supporting context (simulated): The question concerns factual details related to: "
                f"{a[:120]}. Key phrases from the question: {q[:200]}."
            )
        triples = triples_from_evidence(ev, max_triples=5)
        out.append(Exp04Example(qid, q, a, ev, triples))
    return out


def load_examples_hf_hotpot(
    *,
    config: str,
    split: str,
    max_examples: int,
    seed: int,
) -> List[Exp04Example]:
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", config, split=split, streaming=False)
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    out: List[Exp04Example] = []
    for i in indices:
        if max_examples and len(out) >= max_examples:
            break
        ex = ds[int(i)]
        q = str(ex.get("question") or "").strip()
        a = str(ex.get("answer") or "").strip()
        qid = str(ex.get("id") or f"hp_{i}")
        if not q or not a:
            continue
        ctx = ex.get("context") or {}
        titles = ctx.get("title") or []
        sents = ctx.get("sentences") or []
        parts: List[str] = []
        if isinstance(titles, list) and isinstance(sents, list):
            for ti, title in enumerate(titles):
                if ti >= len(sents):
                    break
                para = sents[ti]
                if not isinstance(para, list):
                    continue
                for sid, sent in enumerate(para):
                    st = str(sent or "").strip()
                    if st:
                        parts.append(f"[{title}] {st}")
        ev = " ".join(parts[:12]) if parts else ""
        if not ev:
            continue
        triples = triples_from_evidence(ev, max_triples=5)
        out.append(Exp04Example(qid, q, a, ev, triples))
    return out


def triples_from_evidence(evidence: str, *, max_triples: int) -> List[Tuple[str, str, str]]:
    """Heuristic (subject, relation, object) from sentences; canonical, not copy-paste of full evidence."""
    sents = re.split(r"(?<=[.!?])\s+", evidence.strip())
    sents = [s.strip() for s in sents if len(s.strip()) > 10]
    triples: List[Tuple[str, str, str]] = []
    for s in sents[: max_triples * 2]:
        if len(triples) >= max_triples:
            break
        toks = s.split()
        if len(toks) < 4:
            continue
        subj = " ".join(toks[:3])[:40]
        obj = " ".join(toks[-4:])[:60]
        rel = "relates_to"
        if " is " in s.lower():
            rel = "is_a"
        elif " was " in s.lower():
            rel = "was"
        triples.append((subj, rel, obj))
    while len(triples) < min(2, max_triples):
        triples.append(("entity", "mentions", sents[0][:50] if sents else "context"))
    return triples[:max_triples]


def triples_to_canonical_text(triples: Sequence[Tuple[str, str, str]]) -> str:
    parts = []
    for s, r, o in triples:
        parts.append(f"{s} | {r} | {o}.")
    return " ".join(parts)


def build_prompt_ids(
    tokenizer: AutoTokenizer,
    evidence: str,
    question: str,
    *,
    max_evidence_chars: int,
) -> Tuple[torch.Tensor, int, int]:
    """
    Returns (input_ids[1,L], evidence_start, evidence_end) key positions for RAG (no KV prefix).
    evidence span: [evidence_start, evidence_end) in token indices of the full prompt.
    """
    ev = evidence[:max_evidence_chars]
    ev_header = "Evidence:\n"
    q_header = "\n\nQuestion:\n"
    tail = "\n\nAnswer briefly:\n"
    ev_text = ev_header + ev
    rest = q_header + question + tail
    ev_ids = tokenizer.encode(ev_text, add_special_tokens=False, return_tensors=None)
    rest_ids = tokenizer.encode(rest, add_special_tokens=False, return_tensors=None)
    full = ev_ids + rest_ids
    ids = torch.tensor([full], dtype=torch.long)
    ev_start = 0
    ev_end = len(ev_ids)
    return ids, ev_start, ev_end


def random_prefix_text(tokenizer: AutoTokenizer, target_len: int, *, vocab_sample: int) -> torch.Tensor:
    """Random token ids of length ~target_len (for random-KV ablation)."""
    vs = min(int(tokenizer.vocab_size) - 1, max(1000, vocab_sample))
    lo, hi = 100, vs
    toks = [random.randint(lo, hi) for _ in range(max(1, target_len))]
    return torch.tensor([toks], dtype=torch.long)


def forward_prefix_past(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    device: torch.device,
) -> Tuple[Any, int]:
    input_ids = input_ids.to(device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )
    return out.past_key_values, int(input_ids.shape[1])


def greedy_generate_attentions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    *,
    past_key_values: Optional[Any],
    kv_prefix_len: int,
    evidence_start: int,
    evidence_end: int,
    device: torch.device,
    max_new_tokens: int,
) -> Dict[str, Any]:
    """
    Greedy decode; returns last-step attentions (all layers), full generated ids, per-step optional.
    Positions are indexed in KEY space: [0..kv_prefix_len-1] = KV; then prompt evidence/prompt.
    For RAG, kv_prefix_len=0 and evidence_start/end are prompt-relative (0 = start of prompt keys).
    """
    prompt_ids = prompt_ids.to(device)
    prompt_len = int(prompt_ids.shape[1])
    if past_key_values is None:
        total_prefix = prompt_len
        attn_ev_start = evidence_start
        attn_ev_end = evidence_end
        attn_kv_end = 0
        cur_mask = torch.ones((1, prompt_len), device=device, dtype=torch.long)
    else:
        total_prefix = kv_prefix_len + prompt_len
        attn_ev_start = kv_prefix_len + evidence_start
        attn_ev_end = kv_prefix_len + evidence_end
        attn_kv_end = kv_prefix_len
        cur_mask = torch.ones((1, total_prefix), device=device, dtype=torch.long)

    cur_ids = prompt_ids
    cur_past = past_key_values
    generated: List[int] = []
    step_attns_last: Optional[Tuple[torch.Tensor, ...]] = None
    step_kv_mass: List[float] = []
    step_ev_mass: List[float] = []

    def kv_mass_from_step(attn_tuple: Tuple[torch.Tensor, ...], kv_end: int, ev_s: int, ev_e: int) -> Tuple[float, float]:
        """Mean over layers of mean over heads: mass on KV keys and evidence keys (last query)."""
        kv_m = []
        ev_m = []
        for al in attn_tuple:
            if al.dim() == 3:
                al = al.unsqueeze(0)
            # [B, H, Q, K]
            a = al[0, :, -1, :].float()
            a = a / (a.sum(dim=-1, keepdim=True) + 1e-9)
            if kv_end > 0:
                kv_m.append(float(a[:, :kv_end].sum(dim=-1).mean().item()))
            else:
                kv_m.append(0.0)
            if ev_e > ev_s:
                ev_m.append(float(a[:, ev_s:ev_e].sum(dim=-1).mean().item()))
            else:
                ev_m.append(0.0)
        return float(np.mean(kv_m)), float(np.mean(ev_m))

    for step in range(max_new_tokens):
        with torch.no_grad():
            out = model(
                input_ids=cur_ids,
                past_key_values=cur_past,
                attention_mask=cur_mask,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
        logits = out.logits[:, -1, :]
        next_id = int(logits.argmax(dim=-1).item())
        attns = out.attentions
        if not attns:
            raise RuntimeError(
                "output_attentions produced no weights (empty attentions). "
                "Reload the model with attn_implementation='eager' (SDPA/Flash omit attention probs)."
            )
        step_attns_last = tuple(t.float().cpu() for t in attns)
        km, em = kv_mass_from_step(attns, attn_kv_end, attn_ev_start, attn_ev_end)
        step_kv_mass.append(km)
        step_ev_mass.append(em)

        generated.append(next_id)
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break

        next_t = torch.tensor([[next_id]], device=device, dtype=torch.long)
        cur_ids = next_t
        cur_past = out.past_key_values
        cur_mask = torch.cat([cur_mask, torch.ones((1, 1), device=device, dtype=cur_mask.dtype)], dim=1)

    assert step_attns_last is not None
    return {
        "generated_ids": generated,
        "last_attentions": step_attns_last,
        "kv_prefix_len": kv_prefix_len,
        "attn_kv_end": attn_kv_end,
        "attn_ev_start": int(attn_ev_start),
        "attn_ev_end": int(attn_ev_end),
        "step_kv_mass": step_kv_mass,
        "step_ev_mass": step_ev_mass,
    }


def per_layer_head_mass(
    attn_layers: Tuple[torch.Tensor, ...],
    *,
    key_start: int,
    key_end: int,
) -> np.ndarray:
    """Shape [n_layers, n_heads]: sum attention mass over key range for last query."""
    if not attn_layers:
        raise ValueError(
            "per_layer_head_mass: empty attentions (need eager attention weights; see load_model attn_implementation)."
        )
    n_layers = len(attn_layers)
    al0 = attn_layers[0]
    if al0.dim() == 3:
        al0 = al0.unsqueeze(0)
    if al0.dim() != 4:
        raise ValueError(f"Expected attention tensor [B,H,Q,K], got shape {tuple(attn_layers[0].shape)}")
    n_heads = int(al0.shape[1])
    mat = np.zeros((n_layers, n_heads), dtype=np.float64)
    for li, al in enumerate(attn_layers):
        if al.dim() == 3:
            al = al.unsqueeze(0)
        a = al[0, :, -1, :].float()
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-9)
        if key_end > key_start:
            mat[li] = a[:, key_start:key_end].sum(dim=-1).numpy()
        else:
            mat[li] = 0.0
    return mat


def first_answer_token_id(tokenizer: AutoTokenizer, answer: str) -> int:
    ids = tokenizer.encode(answer.strip(), add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(" " + answer.strip(), add_special_tokens=False)
    return int(ids[0])


def logit_for_token_first_step(
    model: AutoModelForCausalLM,
    prompt_ids: torch.Tensor,
    *,
    past_key_values: Optional[Any],
    kv_prefix_len: int,
    token_id: int,
    device: torch.device,
) -> float:
    prompt_ids = prompt_ids.to(device)
    if past_key_values is None:
        m = torch.ones_like(prompt_ids, device=device)
    else:
        m = torch.ones((1, kv_prefix_len + prompt_ids.shape[1]), device=device, dtype=torch.long)
    with torch.no_grad():
        out = model(
            input_ids=prompt_ids,
            past_key_values=past_key_values,
            attention_mask=m,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
    logp = F.log_softmax(out.logits[:, -1, :], dim=-1)[0, token_id].item()
    return float(logp)


def load_model(
    model_path: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    # SDPA / Flash paths return None attention weights; HF hooks then collect nothing → empty attentions tuple.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=None,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    model.to(device)
    model.eval()
    return model, tok
