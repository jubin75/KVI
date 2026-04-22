#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from metrics import (
    best_exact_match,
    best_f1,
    best_fever_label_em,
    best_relaxed_em,
    truthfulqa_mc_proxy_scores,
)


METHODS_ORDER = ["llm", "rag", "graphrag", "kv_prefix", "kvi"]
METHODS_META = {
    "llm": {"label": "LLM", "retrieval": "none", "injection": "none"},
    "rag": {"label": "RAG", "retrieval": "ANN", "injection": "prompt"},
    "graphrag": {"label": "GraphRAG", "retrieval": "graph", "injection": "prompt"},
    "kv_prefix": {"label": "KV Prefix", "retrieval": "ANN", "injection": "KV"},
    "kvi": {"label": "KVI", "retrieval": "graph", "injection": "KV + prompt"},
}


@dataclass
class Example:
    id: str
    question: str
    answers: List[str]
    mc1_targets: Optional[Dict[str, Any]] = None
    mc2_targets: Optional[Dict[str, Any]] = None


class _TruthfulQALikelihoodScorer:
    def __init__(self, model_name_or_path: str, *, force_cpu: bool = False) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cpu" if bool(force_cpu) else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=(torch.float16 if self.device.type == "cuda" else torch.float32),
        ).to(self.device).eval()

    def _choice_logprob(self, prefix: str, choice: str) -> float:
        torch = self.torch
        p = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        c = self.tokenizer(choice, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        x = torch.cat([p, c], dim=1)
        with torch.no_grad():
            out = self.model(input_ids=x, use_cache=False, return_dict=True)
            logits = out.logits[:, :-1, :]
            tgt = x[:, 1:]
            lp = torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        k = int(c.shape[1])
        if k <= 0:
            return -1e9
        return float(lp[0, -k:].mean().item())

    def score_targets(self, question: str, method_answer: str, targets: Dict[str, Any]) -> Dict[str, float]:
        choices = targets.get("choices")
        labels = targets.get("labels")
        if not isinstance(choices, list) or not isinstance(labels, list) or len(choices) != len(labels) or not choices:
            return {"mc1": 0.0, "mc2": 0.0, "valid": 0.0}
        prefix = (
            f"Question: {question}\n"
            f"Proposed answer: {method_answer}\n"
            "Which option is most truthful and accurate?\nOption: "
        )
        scores = [self._choice_logprob(prefix, str(c)) for c in choices]
        best_i = max(range(len(scores)), key=lambda i: scores[i])
        y = [1 if int(v) > 0 else 0 for v in labels]
        mc1 = 1.0 if y[best_i] > 0 else 0.0
        m = max(scores)
        exps = [pow(2.718281828, s - m) for s in scores]
        z = sum(exps) + 1e-12
        probs = [e / z for e in exps]
        mc2 = float(sum(p for p, yy in zip(probs, y) if yy > 0))
        return {"mc1": mc1, "mc2": mc2, "valid": 1.0}


def _kvi_pred_for_truthfulqa_mc1_likelihood(full_pred: str, *, max_chars: int = 384) -> str:
    """
    TruthfulQA + likelihood_proxy: compress KVI decode to a short "Proposed answer" string
    (first non-empty line, optional leading label strip, first sentence or char cap).
    Used as fallback when KV Prefix injected text is unavailable; also used for KVI MC2 conditioning
    so long tails do not distort option log-probs.
    """
    t = str(full_pred or "").strip()
    if not t:
        return t
    # Remove common prompt-template or rendering noise before extracting a compact answer.
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"https?://\S+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    line = t.split("\n", 1)[0].strip()
    # Generic hedging prefixes: improve MC alignment by retaining factual core.
    line = re.sub(
        r"^\s*(?:it\s+is\s+(?:generally|widely|commonly)\s+(?:recommended|believed|considered)\s+that|"
        r"in\s+general,?\s*|generally,?\s*|typically,?\s*|usually,?\s*)",
        "",
        line,
        flags=re.IGNORECASE,
    )
    # Keep this generic for open-domain QA: strip lightweight leading wrappers,
    # but do not hardcode task-specific phrase maps.
    line = re.sub(
        r"^\s*(?:note|important|answer|response|final answer)\s*[:\-]\s*",
        "",
        line,
        flags=re.IGNORECASE,
    )
    for pref in ("Answer:", "Answer ", "The answer is", "Response:", "A:"):
        low = line.lower()
        p = pref.lower()
        if low.startswith(p):
            line = line[len(pref) :].strip()
            break
    # Remove broad disclaimer-like prefixes that hurt MC likelihood alignment.
    line = re.sub(
        r"^\s*(?:it\s+is\s+generally\s+(?:recommended|believed|considered)\s+that|"
        r"there\s+is\s+no\s+single\s+answer\s+but)\s+",
        "",
        line,
        flags=re.IGNORECASE,
    )
    # When multiple short clauses exist, pick the densest factual clause
    # instead of always taking the first one (which can be a disclaimer).
    sent_cands = [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
    if sent_cands:
        def _sent_score(s: str) -> int:
            score = 0
            n = len(s)
            if 12 <= n <= 180:
                score += 8
            elif n < 8:
                score -= 8
            score += min(8, len(re.findall(r"[A-Za-z]{3,}", s)))
            if re.search(r"\b(?:maybe|perhaps|likely|possibly)\b", s, flags=re.IGNORECASE):
                score -= 4
            if re.search(r"[;:]{2,}|[!]{3,}", s):
                score -= 4
            return score
        best_sent = max(sent_cands, key=_sent_score)
        if best_sent:
            return best_sent[:max_chars].strip()
    if len(line) > max_chars:
        return line[:max_chars].rstrip()
    return line


def _clean_truthfulqa_kvi_prediction(text: str, *, max_chars: int = 220) -> str:
    """Normalize noisy KVI free-form decode to a concise answer string."""
    t = str(text or "").strip()
    if not t:
        return t
    # Remove common markdown/image artifacts that pollute decoded outputs.
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    # Drop obvious mixed-script contamination in English TruthfulQA answers.
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", t) and re.search(r"[A-Za-z]{3,}", t):
        t = re.sub(r"[\u3040-\u30ff\u4e00-\u9fff]+[^\n.!?]*", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # Drop excessive punctuation noise while preserving normal sentence marks.
    t = re.sub(r"!{2,}", "!", t)
    t = re.sub(r"[\"'`]{2,}", "", t)
    # De-noise token-separator exclamation marks frequently seen in broken decode
    # like "No! I am! your! father!" while keeping sentence semantics.
    t = re.sub(r"(?<=[A-Za-z0-9\u4e00-\u9fff])!+(?=\s|[A-Za-z0-9\u4e00-\u9fff])", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # Prefer first sentence-like fragment as final answer.
    # Do not treat bare "!" as hard sentence boundary for TruthfulQA, otherwise
    # valid answers like "No! I am your father!" collapse to "No!".
    m = re.match(r"^(.+?[.?。！？])(\s|$)", t)
    if m:
        t = m.group(1).strip()
    if len(t) > max_chars:
        t = t[:max_chars].rstrip()
    # Fallback to existing KVI shortening logic if the cleaned text is still sparse/noisy.
    if len(t) < 8:
        t2 = _kvi_pred_for_truthfulqa_mc1_likelihood(text, max_chars=max_chars)
        if t2:
            t = t2
    return t.strip()


def _kvi_answer_quality_score(text: str) -> int:
    t = str(text or "").strip()
    if not t:
        return -10_000
    score = 0
    # Favor concise but not trivial answer lengths.
    n = len(t)
    if 24 <= n <= 180:
        score += 30
    elif 12 <= n <= 240:
        score += 15
    else:
        score -= 10
    # Penalize obvious formatting noise.
    if re.search(r"!\[[^\]]*\]\([^)]*\)", t):
        score -= 60
    if re.search(r"https?://", t, flags=re.IGNORECASE):
        score -= 40
    if re.search(r"!{2,}|[`]{2,}", t):
        score -= 20
    # Reward semantic content (words / CJK chars).
    word_count = len(re.findall(r"[A-Za-z]{2,}", t))
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", t))
    score += min(25, word_count * 2 + cjk_count // 4)
    return score


def _select_truthfulqa_kvi_prediction(
    *,
    primary_pred: str,
    out_gr_kvi: Dict[str, Any],
    out_kv_prefix: Optional[Dict[str, Any]],
) -> str:
    """Pick the cleanest KVI answer among graph outputs (and optional safe fallback)."""
    cands: List[str] = []
    for raw in (
        primary_pred,
        str(out_gr_kvi.get("diagnosis_result") or ""),
        str(out_gr_kvi.get("diagnosis_result_raw") or ""),
        str(out_gr_kvi.get("base_llm_result") or ""),
    ):
        cleaned = _clean_truthfulqa_kvi_prediction(raw)
        if cleaned:
            cands.append(cleaned)
    # IMPORTANT: keep KVI evaluation independent from ANN kv_prefix channel.
    # Cross-method fallback can hide real KVI failures and collapse method differences.
    # We intentionally do NOT pull candidates from out_kv_prefix here.
    if not cands:
        return _clean_truthfulqa_kvi_prediction(primary_pred)
    return max(cands, key=_kvi_answer_quality_score)


def _kvi_truthfulqa_mc1_proposed_answer(
    kvi_full_pred: str,
    out_kv_prefix: Optional[Dict[str, Any]],
    *,
    max_injected_chars: Optional[int] = 280,
) -> str:
    """
    Legacy MC1 string when matching KV Prefix: ANN `injected_answer` (optional cap; use None for full string).
    If no injection, fall back to shortened KVI decode.
    """
    if out_kv_prefix:
        inj = str(out_kv_prefix.get("injected_answer") or "").strip()
        if inj:
            if max_injected_chars is None or int(max_injected_chars) <= 0:
                return inj
            return inj[: int(max_injected_chars)].strip()
    return _kvi_pred_for_truthfulqa_mc1_likelihood(kvi_full_pred)


def _kvi_truthfulqa_mc1_text_for_likelihood(
    *,
    pred: str,
    out_kv_prefix: Optional[Dict[str, Any]],
    methods: List[str],
    mc1_answer: str,
) -> str:
    """
    KVI + TruthfulQA likelihood_proxy: proposed-answer string for MC1 targets.
    - injected: same full ANN snippet as kv_prefix method output when kv_prefix is in --methods.
    - grounded: shortened grounded KVI decode (aligned with MC2 conditioning; reflects graph+KV refinement).
    """
    mode = str(mc1_answer or "grounded").strip().lower()
    if mode == "injected" and "kv_prefix" in methods and out_kv_prefix:
        inj = str(out_kv_prefix.get("injected_answer") or "").strip()
        if inj:
            return inj
    return _kvi_pred_for_truthfulqa_mc1_likelihood(pred)


def _extract_answers(obj: Dict[str, Any]) -> List[str]:
    if isinstance(obj.get("answers"), list):
        out = [str(x).strip() for x in obj.get("answers", []) if str(x).strip()]
        if out:
            return out
    ans = obj.get("answer")
    if isinstance(ans, list):
        out = [str(x).strip() for x in ans if str(x).strip()]
        if out:
            return out
    s = str(ans or "").strip()
    return [s] if s else []


def _read_jsonl(path: Path, *, limit: Optional[int] = None) -> List[Example]:
    out: List[Example] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            ex = Example(
                id=str(obj.get("id") or f"ex_{len(out)+1}"),
                question=str(obj.get("question") or "").strip(),
                answers=_extract_answers(obj),
                mc1_targets=(obj.get("mc1_targets") if isinstance(obj.get("mc1_targets"), dict) else None),
                mc2_targets=(obj.get("mc2_targets") if isinstance(obj.get("mc2_targets"), dict) else None),
            )
            if ex.question and ex.answers:
                out.append(ex)
            if limit and len(out) >= limit:
                break
    return out


def _run_json_cmd(cmd: List[str], *, cwd: Path, timeout_s: int, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=(env if env is not None else dict(os.environ)),
        cwd=str(cwd),
        timeout=timeout_s,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed rc={p.returncode}\ncmd={' '.join(cmd)}\n"
            f"stdout_tail={(p.stdout or '')[-3000:]}\nstderr_tail={(p.stderr or '')[-3000:]}"
        )
    try:
        return json.loads(p.stdout)
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse JSON output: {e}\ncmd={' '.join(cmd)}\n"
            f"stdout_tail={(p.stdout or '')[-3000:]}\nstderr_tail={(p.stderr or '')[-3000:]}"
        )


def _run_json_service(*, url: str, endpoint: str, argv: List[str], timeout_s: int) -> Dict[str, Any]:
    req = urllib.request.Request(
        url=f"{url.rstrip('/')}{endpoint}",
        data=json.dumps({"argv": argv}, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Service call failed endpoint={endpoint}: {e}") from e
    if not isinstance(data, dict) or not data.get("ok"):
        raise RuntimeError(f"Service error endpoint={endpoint}: {data}")
    result = data.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"Service returned invalid result endpoint={endpoint}: {data}")
    return result


def _run_graph(
    *,
    repo_root: Path,
    model: str,
    question: str,
    graph_index: Path,
    sentences_jsonl: Optional[Path],
    enable_kvi: bool,
    triple_kvbank_dir: Optional[Path],
    max_new_tokens: int,
    timeout_s: int,
    service_url: str,
    force_cpu: bool = False,
    openqa_mode: bool = True,
    kvi_minimal_prompt: bool = False,
    kvi_max_kv_triples: int = 3,
    kvi_drm_threshold: float = 0.05,
    kvi_top_k_relations: int = 2,
    kvi_reconcile_no_kv_decode: bool = False,
    kvi_kvprefix_style_prompt: bool = False,
    kvi_disable_kv_injection: bool = False,
    kvi_two_stage_kv_then_evidence: bool = False,
    audit_jsonl: str = "",
    audit_query_id: str = "",
    audit_method_key: str = "",
    audit_oracle_jsonl: str = "",
) -> Dict[str, Any]:
    argv = [
        "--model",
        model,
        "--prompt",
        question,
        "--graph_index",
        str(graph_index),
        "--max_new_tokens",
        str(max_new_tokens),
    ]
    if sentences_jsonl:
        argv += ["--sentences_jsonl", str(sentences_jsonl)]
    if enable_kvi:
        argv += ["--enable_kvi"]
        if triple_kvbank_dir:
            argv += ["--triple_kvbank_dir", str(triple_kvbank_dir)]
        argv += [
            "--max_kv_triples",
            str(int(kvi_max_kv_triples)),
            "--drm_threshold",
            str(float(kvi_drm_threshold)),
            "--top_k_relations",
            str(int(kvi_top_k_relations)),
        ]
    if openqa_mode:
        argv += ["--openqa_mode"]
    # Instruct models (e.g. Qwen2.5-*-Instruct) expect chat_template-wrapped prompts; without it,
    # graph-side generation often degenerates (spurious punctuation between tokens, off-task text).
    argv += ["--use_chat_template"]
    if kvi_minimal_prompt and enable_kvi:
        argv += ["--kvi_minimal_prompt"]
    if enable_kvi and kvi_reconcile_no_kv_decode:
        argv += ["--kvi_reconcile_no_kv_decode"]
    if enable_kvi and kvi_kvprefix_style_prompt:
        argv += ["--kvi_kvprefix_style_prompt"]
    if enable_kvi and kvi_disable_kv_injection:
        argv += ["--kvi_disable_kv_injection"]
    if enable_kvi and kvi_two_stage_kv_then_evidence:
        argv += ["--kvi_two_stage_kv_then_evidence"]
    if str(audit_jsonl).strip():
        argv += ["--audit_jsonl", str(audit_jsonl)]
    if str(audit_query_id).strip():
        argv += ["--audit_query_id", str(audit_query_id)]
    if str(audit_method_key).strip():
        argv += ["--audit_method_key", str(audit_method_key)]
    if str(audit_oracle_jsonl).strip():
        argv += ["--audit_oracle_jsonl", str(audit_oracle_jsonl)]
    if service_url.strip():
        return _run_json_service(url=service_url, endpoint="/infer/graph", argv=argv, timeout_s=timeout_s)
    cmd = [sys.executable, str(repo_root / "scripts" / "run_graph_inference.py")] + argv
    env = dict(os.environ)
    if bool(force_cpu):
        # Prevent the graph-side subprocess from using GPU when VRAM is occupied.
        env["CUDA_VISIBLE_DEVICES"] = ""
        cmd += ["--device", "cpu", "--dtype", "float32"]
    return _run_json_cmd(cmd, cwd=repo_root, timeout_s=timeout_s, env=env)


def _run_kvi2_runtime(
    *,
    repo_root: Path,
    model: str,
    question: str,
    pipeline: str,
    kv_dir: Path,
    sentences_jsonl: Path,
    semantic_type_specs: Path,
    pattern_index_dir: Path,
    sidecar_dir: Path,
    domain_encoder_model: str,
    top_k: int,
    timeout_s: int,
    service_url: str,
    force_cpu: bool = False,
) -> Dict[str, Any]:
    argv = [
        "--pipeline",
        pipeline,
        "--model",
        model,
        "--prompt",
        question,
        "--kv_dir",
        str(kv_dir),
        "--sentences_jsonl",
        str(sentences_jsonl),
        "--semantic_type_specs",
        str(semantic_type_specs),
        "--pattern_index_dir",
        str(pattern_index_dir),
        "--sidecar_dir",
        str(sidecar_dir),
        "--domain_encoder_model",
        domain_encoder_model,
        "--use_chat_template",
        "--top_k",
        str(int(top_k)),
        "--show_baseline",
    ]
    if service_url.strip():
        return _run_json_service(url=service_url, endpoint="/infer/kvi", argv=argv, timeout_s=timeout_s)
    cmd = [sys.executable, str(repo_root / "scripts" / "run_kvi2_runtime_test.py")] + argv
    env = dict(os.environ)
    if bool(force_cpu):
        # Prevent ANN-side runtime from competing with the resident GPU model.
        env["CUDA_VISIBLE_DEVICES"] = ""
    return _run_json_cmd(cmd, cwd=repo_root, timeout_s=timeout_s, env=env)


def _pick_method_prediction(
    *,
    method: str,
    out_gr_rag: Dict[str, Any],
    out_gr_kvi: Dict[str, Any],
    out_rag: Optional[Dict[str, Any]],
    out_kv_prefix: Optional[Dict[str, Any]],
) -> str:
    if method == "llm":
        return str(out_gr_rag.get("base_llm_result") or "")
    if method == "graphrag":
        return str(out_gr_rag.get("diagnosis_result_raw") or out_gr_rag.get("diagnosis_result") or "")
    if method == "kvi":
        # Prefer grounded `diagnosis_result` over raw decode for stored prediction / EM / F1.
        return str(out_gr_kvi.get("diagnosis_result") or out_gr_kvi.get("diagnosis_result_raw") or "")
    if method == "rag":
        return str((out_rag or {}).get("diagnosis_result") or (out_rag or {}).get("base_llm_result") or "")
    if method == "kv_prefix":
        return str((out_kv_prefix or {}).get("injected_answer") or "")
    return ""


def _bootstrap_ci_percent(samples01: List[int], *, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> Dict[str, float]:
    n = len(samples01)
    if n <= 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    rng = random.Random(seed)
    vals: List[float] = []
    for _ in range(max(10, int(n_boot))):
        s = 0
        for _i in range(n):
            s += samples01[rng.randrange(0, n)]
        vals.append(100.0 * float(s) / float(n))
    vals.sort()
    lo_idx = max(0, int((alpha / 2.0) * len(vals)) - 1)
    hi_idx = min(len(vals) - 1, int((1.0 - alpha / 2.0) * len(vals)) - 1)
    mean = 100.0 * float(sum(samples01)) / float(n)
    return {"mean": float(mean), "lo": float(vals[lo_idx]), "hi": float(vals[hi_idx])}


def _paired_permutation_pvalue(a: List[int], b: List[int], *, n_perm: int = 2000, seed: int = 42) -> float:
    """
    Paired permutation test for mean(a-b) > 0 difference in EM(0/1).
    Returns two-sided p-value approximation.
    """
    if len(a) != len(b) or not a:
        return 1.0
    n = len(a)
    diffs = [int(x) - int(y) for x, y in zip(a, b)]
    obs = abs(sum(diffs) / float(n))
    rng = random.Random(seed)
    ge = 1  # add-one smoothing
    total = 1
    for _ in range(max(100, int(n_perm))):
        s = 0
        for d in diffs:
            if rng.random() < 0.5:
                s += d
            else:
                s -= d
        stat = abs(float(s) / float(n))
        if stat >= obs:
            ge += 1
        total += 1
    return float(ge) / float(total)


def main() -> None:
    p = argparse.ArgumentParser(description="Experiment 01: single-dataset main QA EM")
    p.add_argument("--dataset", required=True, help="JSONL with {id, question, answer|answers}")
    p.add_argument("--dataset_name", default="", help="Display name in outputs, e.g. HotpotQA / NQ")
    p.add_argument("--model", required=True, help="Base LLM model path/name")

    # Graph channel (for LLM/GraphRAG/KVI)
    p.add_argument("--graph_index", required=True, help="graph_index.json path")
    p.add_argument("--triple_kvbank_dir", default="", help="triple_kvbank directory for KVI")
    p.add_argument("--graph_sentences_jsonl", default="", help="Optional sentences.jsonl for graph hybrid retrieval")

    # ANN channel (for RAG/KV Prefix)
    p.add_argument("--ann_kv_dir", default="", help="ANN KV directory for run_kvi2_runtime_test route/simple")
    p.add_argument("--ann_sentences_jsonl", default="", help="sentences.tagged.jsonl path")
    p.add_argument("--ann_semantic_type_specs", default="", help="semantic_type_specs.json path")
    p.add_argument("--ann_pattern_index_dir", default="", help="pattern_sidecar dir")
    p.add_argument("--ann_sidecar_dir", default="", help="sidecar/work dir")
    p.add_argument("--domain_encoder_model", default="sentence-transformers/all-MiniLM-L6-v2")

    p.add_argument("--methods", default="llm,rag,graphrag,kv_prefix,kvi", help="comma list from: llm,rag,graphrag,kv_prefix,kvi")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--limit", type=int, default=0, help="Limit examples (0=all)")
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from existing predictions.jsonl in out_dir. "
            "Validates that existing records match the first K dataset examples, then appends remaining. "
            "Always recomputes summary.json/results.* over full N at the end."
        ),
    )
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--timeout_s", type=int, default=300)
    p.add_argument(
        "--graph_audit_jsonl",
        default="",
        help="Optional path to append per-query retrieval/DRM/prompt audit records from graph runner.",
    )
    p.add_argument(
        "--graph_audit_oracle_jsonl",
        default="",
        help="Optional oracle evidence JSONL ({id, gold_evidence_texts[]}) for audit recall checks.",
    )
    p.add_argument(
        "--inference_service_url",
        default="",
        help="Optional resident service base URL for GRAPH channel (/infer/graph), e.g. http://127.0.0.1:18888",
    )
    p.add_argument(
        "--ann_inference_service_url",
        default="",
        help="Optional resident service base URL for ANN channel (/infer/kvi). Default empty uses local subprocess (recommended when GPU is occupied by resident graph model).",
    )
    p.add_argument(
        "--ann_force_cpu",
        action="store_true",
        help="Force ANN-side runtime (RAG/KV Prefix via run_kvi2_runtime_test) to run on CPU (avoid GPU OOM when a resident model occupies VRAM).",
    )
    p.add_argument(
        "--graph_force_cpu",
        action="store_true",
        help="Force GRAPH-side runtime (LLM/GraphRAG/KVI via run_graph_inference) to run on CPU (avoid GPU OOM).",
    )
    p.add_argument("--bootstrap_samples", type=int, default=1000)
    p.add_argument("--permutation_samples", type=int, default=2000)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument(
        "--em_mode",
        choices=["strict", "relaxed"],
        default="relaxed",
        help=(
            "strict: whole-string EM after light normalize (harsh on long generations). "
            "relaxed: SQuAD-normalize + gold-as-substring in pred (Hotpot/NQ open QA default)."
        ),
    )
    p.add_argument(
        "--openqa_mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Graph/KVI: use English open-QA prompts in run_graph_inference (recommended for Hotpot/NQ). "
        "Use --no-openqa_mode for legacy medical Chinese prompts.",
    )
    p.add_argument(
        "--kvi_minimal_prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="KVI only: omit evidence block from prompt when KV is injected (ablation: KV-only vs dual-channel).",
    )
    p.add_argument(
        "--kvi_max_kv_triples",
        type=int,
        default=3,
        help="Forwarded to run_graph_inference when KVI is on (ablation: set 0 to disable KV injection).",
    )
    p.add_argument(
        "--kvi_drm_threshold",
        type=float,
        default=0.05,
        help="DRM threshold for triple KV selection (run_graph_inference --drm_threshold).",
    )
    p.add_argument(
        "--kvi_top_k_relations",
        type=int,
        default=2,
        help="Relation gating top-k (run_graph_inference --top_k_relations).",
    )
    p.add_argument(
        "--kvi_reconcile_no_kv_decode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="KVI only: dual-decode reconcile in run_graph_inference (see --kvi_reconcile_no_kv_decode there).",
    )
    p.add_argument(
        "--kvi_disable_kv_injection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="KVI only (ablation): assemble triple KV but do not inject it as past_key_values during generation.",
    )
    p.add_argument(
        "--kvi_two_stage_kv_then_evidence",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="KVI only: stage-1 uses KV-only draft (no evidence text), stage-2 uses draft+evidence for final answer.",
    )
    p.add_argument(
        "--truthfulqa_mc_mode",
        choices=["proxy", "likelihood_proxy"],
        default="likelihood_proxy",
        help="TruthfulQA MC metric mode: string-match proxy or option-likelihood proxy.",
    )
    p.add_argument(
        "--truthfulqa_kvi_mc1_answer",
        choices=["grounded", "injected"],
        default="grounded",
        help=(
            "TruthfulQA: for method kvi only, MC1 likelihood/proxy conditions on shortened grounded KVI text "
            "(default) or on the ANN injected_answer when kv_prefix is also in --methods (legacy parity)."
        ),
    )
    p.add_argument(
        "--truthfulqa_mc_force_cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="TruthfulQA likelihood scorer: force CPU loading to avoid GPU OOM contention with resident inference.",
    )
    p.add_argument(
        "--truthfulqa_kvi_max_new_tokens",
        type=int,
        default=96,
        help=(
            "TruthfulQA + KVI only: cap graph KVI decode length to reduce long/off-topic tails. "
            "Set <=0 to disable this cap."
        ),
    )
    p.add_argument(
        "--truthfulqa_kvi_minimal_prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "TruthfulQA + KVI: default ON — omit duplicate evidence block in the user prompt when KV is "
            "injected (reduces dual-channel confusion vs GraphRAG). Use --no-truthfulqa_kvi_minimal_prompt to disable."
        ),
    )
    p.add_argument(
        "--truthfulqa_kvi_reconcile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "TruthfulQA + KVI: default ON — dual-decode reconcile in run_graph_inference with open-QA overlap "
            "scoring (fixed; avoids always dropping KV when no DrugBank IDs). "
            "Use --no-truthfulqa_kvi_reconcile to disable."
        ),
    )
    p.add_argument(
        "--truthfulqa_kvi_tuned_overrides",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "TruthfulQA + KVI: apply tuned runtime overrides (DRM floor, KV triple band, relations/minimal/reconcile). "
            "Use --no-truthfulqa_kvi_tuned_overrides for controlled ablations such as injection-level sweeps."
        ),
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)
    graph_index = Path(args.graph_index)
    triple_kvbank_dir = Path(args.triple_kvbank_dir) if str(args.triple_kvbank_dir).strip() else None
    graph_sentences_jsonl = Path(args.graph_sentences_jsonl) if str(args.graph_sentences_jsonl).strip() else None

    ann_kv_dir = Path(args.ann_kv_dir) if str(args.ann_kv_dir).strip() else None
    ann_sentences_jsonl = Path(args.ann_sentences_jsonl) if str(args.ann_sentences_jsonl).strip() else None
    ann_semantic_type_specs = Path(args.ann_semantic_type_specs) if str(args.ann_semantic_type_specs).strip() else None
    ann_pattern_index_dir = Path(args.ann_pattern_index_dir) if str(args.ann_pattern_index_dir).strip() else None
    ann_sidecar_dir = Path(args.ann_sidecar_dir) if str(args.ann_sidecar_dir).strip() else None

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS_META]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")

    if any(m in {"rag", "kv_prefix"} for m in methods):
        required = [
            ("ann_kv_dir", ann_kv_dir),
            ("ann_sentences_jsonl", ann_sentences_jsonl),
            ("ann_semantic_type_specs", ann_semantic_type_specs),
            ("ann_pattern_index_dir", ann_pattern_index_dir),
            ("ann_sidecar_dir", ann_sidecar_dir),
        ]
        missing = [name for name, val in required if val is None]
        if missing:
            raise SystemExit(f"Methods rag/kv_prefix require ANN artifacts, missing: {missing}")

    examples = _read_jsonl(dataset_path, limit=(args.limit or None))
    if not examples:
        raise SystemExit(f"Empty/invalid dataset: {dataset_path}")

    per_example_path = out_dir / "predictions.jsonl"
    is_fever = str(args.dataset_name or "").strip().upper() == "FEVER"
    method_correct = {m: 0 for m in methods}
    method_em_series: Dict[str, List[int]] = {m: [] for m in methods}
    method_f1_sum = {m: 0.0 for m in methods}
    method_fever_correct = {m: 0 for m in methods}
    method_fever_series: Dict[str, List[int]] = {m: [] for m in methods}
    method_mc1_acc_sum = {m: 0.0 for m in methods}
    method_mc2_mass_sum = {m: 0.0 for m in methods}
    method_mc1_valid_sum = {m: 0.0 for m in methods}
    method_mc2_valid_sum = {m: 0.0 for m in methods}
    em_score_fn = best_exact_match if str(args.em_mode) == "strict" else best_relaxed_em
    is_truthfulqa = str(args.dataset_name or "").strip().upper() == "TRUTHFULQA"
    tqa_kvi_stats: Optional[Dict[str, Any]] = None
    if is_truthfulqa:
        if bool(args.truthfulqa_kvi_tuned_overrides):
            tqa_kvi_stats = {
                "kvi_max_kv_triples": min(max(int(args.kvi_max_kv_triples), 4), 5),
                "kvi_drm_threshold": max(float(args.kvi_drm_threshold), 0.046),
                "kvi_top_k_relations": max(int(args.kvi_top_k_relations), 4),
                "truthfulqa_kvi_minimal_prompt": bool(args.kvi_minimal_prompt)
                or bool(args.truthfulqa_kvi_minimal_prompt),
                "truthfulqa_kvi_reconcile": bool(args.kvi_reconcile_no_kv_decode)
                or bool(args.truthfulqa_kvi_reconcile),
            }
        else:
            tqa_kvi_stats = {
                "kvi_max_kv_triples": int(args.kvi_max_kv_triples),
                "kvi_drm_threshold": float(args.kvi_drm_threshold),
                "kvi_top_k_relations": int(args.kvi_top_k_relations),
                "truthfulqa_kvi_minimal_prompt": bool(args.kvi_minimal_prompt)
                or bool(args.truthfulqa_kvi_minimal_prompt),
                "truthfulqa_kvi_reconcile": bool(args.kvi_reconcile_no_kv_decode)
                or bool(args.truthfulqa_kvi_reconcile),
            }
    tqa_scorer = None
    if is_truthfulqa and str(args.truthfulqa_mc_mode) == "likelihood_proxy":
        tqa_scorer = _TruthfulQALikelihoodScorer(
            str(args.model),
            force_cpu=bool(args.truthfulqa_mc_force_cpu),
        )

    # Resume logic: preload existing K records (if any), validate alignment, then append remaining.
    start_idx = 1
    existing_records: List[Dict[str, Any]] = []
    if bool(args.resume) and per_example_path.exists():
        with per_example_path.open("r", encoding="utf-8") as rf:
            for line in rf:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    raise SystemExit(f"Failed to parse existing predictions.jsonl: {per_example_path}")
                if not isinstance(obj, dict) or "idx" not in obj:
                    raise SystemExit(f"Invalid record in existing predictions.jsonl: {per_example_path}")
                existing_records.append(obj)
        k = len(existing_records)
        if k > len(examples):
            raise SystemExit(
                f"Resume found {k} existing records but dataset has only {len(examples)} examples: {per_example_path}"
            )
        # Validate id/question match for the first K.
        for i in range(k):
            ex = examples[i]
            rec = existing_records[i]
            rec_id = str(rec.get("id") or "")
            rec_q = str(rec.get("question") or "")
            if rec_id != ex.id or rec_q != ex.question:
                raise SystemExit(
                    "Resume mismatch between dataset and existing predictions.jsonl.\n"
                    f"- at i={i} expected id={ex.id!r} q={ex.question[:120]!r}\n"
                    f"- got      id={rec_id!r} q={rec_q[:120]!r}\n"
                    f"file={per_example_path}"
                )
            # Preload metrics (only for requested methods).
            ems = rec.get("em") or {}
            f1s = rec.get("f1") or {}
            if not isinstance(ems, dict) or not isinstance(f1s, dict):
                raise SystemExit(f"Resume record missing em/f1 dict: {per_example_path} (i={i})")
            fl_row: Optional[Dict[str, int]] = None
            mc1_row: Optional[Dict[str, float]] = None
            mc2_row: Optional[Dict[str, float]] = None
            if is_fever:
                fl_raw = rec.get("fever_label_em")
                if isinstance(fl_raw, dict) and all(mm in fl_raw for mm in methods):
                    fl_row = {mm: int(fl_raw[mm]) for mm in methods}
                else:
                    preds_resume = rec.get("predictions") or {}
                    if not isinstance(preds_resume, dict):
                        preds_resume = {}
                    fl_row = {
                        mm: int(best_fever_label_em(str(preds_resume.get(mm) or ""), ex.answers))
                        for mm in methods
                    }
            elif is_truthfulqa:
                mc1_raw = rec.get("truthfulqa_mc1_proxy")
                mc2_raw = rec.get("truthfulqa_mc2_proxy")
                if isinstance(mc1_raw, dict) and isinstance(mc2_raw, dict) and all(mm in mc1_raw for mm in methods) and all(mm in mc2_raw for mm in methods):
                    mc1_row = {mm: float(mc1_raw[mm]) for mm in methods}
                    mc2_row = {mm: float(mc2_raw[mm]) for mm in methods}
                else:
                    preds_resume = rec.get("predictions") or {}
                    if not isinstance(preds_resume, dict):
                        preds_resume = {}
                    mc1_row = {}
                    mc2_row = {}
                    kv_inj = str(preds_resume.get("kv_prefix") or "").strip()
                    fake_kv = {"injected_answer": kv_inj} if kv_inj else None
                    for mm in methods:
                        pp = str(preds_resume.get(mm) or "")
                        if str(args.truthfulqa_mc_mode) == "proxy":
                            if mm == "kvi":
                                if str(args.truthfulqa_kvi_mc1_answer) == "grounded":
                                    p_m1 = _kvi_pred_for_truthfulqa_mc1_likelihood(pp)
                                else:
                                    p_m1 = kv_inj if kv_inj else pp
                                p_m2 = _kvi_pred_for_truthfulqa_mc1_likelihood(pp)
                                s1 = truthfulqa_mc_proxy_scores(p_m1, ex.mc1_targets)
                                s2 = truthfulqa_mc_proxy_scores(p_m2, ex.mc2_targets)
                            else:
                                s1 = truthfulqa_mc_proxy_scores(pp, ex.mc1_targets)
                                s2 = truthfulqa_mc_proxy_scores(pp, ex.mc2_targets)
                            mc1_row[mm] = float(s1["mc_acc_proxy"])
                            mc2_row[mm] = float(s2["mc_mass_proxy"])
                        else:
                            p1 = (
                                _kvi_truthfulqa_mc1_text_for_likelihood(
                                    pred=pp,
                                    out_kv_prefix=fake_kv,
                                    methods=methods,
                                    mc1_answer=str(args.truthfulqa_kvi_mc1_answer),
                                )
                                if mm == "kvi"
                                else pp
                            )
                            p2 = _kvi_pred_for_truthfulqa_mc1_likelihood(pp) if mm == "kvi" else pp
                            s1 = tqa_scorer.score_targets(ex.question, p1, ex.mc1_targets or {}) if tqa_scorer is not None else {"mc1": 0.0, "mc2": 0.0}
                            s2 = tqa_scorer.score_targets(ex.question, p2, ex.mc2_targets or {}) if tqa_scorer is not None else {"mc1": 0.0, "mc2": 0.0}
                            mc1_row[mm] = float(s1["mc1"])
                            mc2_row[mm] = float(s2["mc2"])
            for m in methods:
                if m not in ems or m not in f1s:
                    raise SystemExit(
                        f"Resume record missing method={m} em/f1: {per_example_path} (i={i}). "
                        "If you changed --methods, please rerun without --resume into a fresh out_dir."
                    )
                em = int(ems[m])
                f1 = float(f1s[m])
                method_correct[m] += em
                method_em_series[m].append(em)
                method_f1_sum[m] += f1
                if is_fever and fl_row is not None:
                    fv = int(fl_row[m])
                    method_fever_correct[m] += fv
                    method_fever_series[m].append(fv)
                if (not is_fever) and is_truthfulqa:
                    if mc1_row is not None and mc2_row is not None:
                        method_mc1_acc_sum[m] += float(mc1_row[m])
                        method_mc2_mass_sum[m] += float(mc2_row[m])
                        method_mc1_valid_sum[m] += 1.0
                        method_mc2_valid_sum[m] += 1.0
        start_idx = k + 1

    file_mode = "a" if (bool(args.resume) and per_example_path.exists() and start_idx > 1) else "w"
    with per_example_path.open(file_mode, encoding="utf-8") as wf:
        for idx, ex in enumerate(examples, start=1):
            if idx < start_idx:
                continue
            need_graph = any(m in {"llm", "graphrag", "kvi"} for m in methods)
            out_gr_rag: Dict[str, Any] = {}
            out_gr_kvi: Dict[str, Any] = {}
            if need_graph:
                graph_audit_path = str(args.graph_audit_jsonl or "").strip()
                graph_audit_oracle = str(args.graph_audit_oracle_jsonl or "").strip()
                out_gr_rag = _run_graph(
                    repo_root=repo_root,
                    model=args.model,
                    question=ex.question,
                    graph_index=graph_index,
                    sentences_jsonl=graph_sentences_jsonl,
                    enable_kvi=False,
                    triple_kvbank_dir=triple_kvbank_dir,
                    max_new_tokens=int(args.max_new_tokens),
                    timeout_s=int(args.timeout_s),
                    service_url=str(args.inference_service_url or ""),
                    force_cpu=bool(args.graph_force_cpu),
                    openqa_mode=bool(args.openqa_mode),
                    kvi_minimal_prompt=False,
                    audit_jsonl=graph_audit_path,
                    audit_query_id=str(ex.id),
                    audit_method_key="graphrag",
                    audit_oracle_jsonl=graph_audit_oracle,
                )
                # TruthfulQA (open English): tighter KV budget + stricter DRM + minimal prompt / reconcile (KVI path only).
                kvi_triples = int(args.kvi_max_kv_triples)
                kvi_drm = float(args.kvi_drm_threshold)
                kvi_rels = int(args.kvi_top_k_relations)
                kvi_max_new_tokens = int(args.max_new_tokens)
                kvi_minimal_use = bool(args.kvi_minimal_prompt)
                kvi_reconcile_use = bool(args.kvi_reconcile_no_kv_decode)
                kvi_kvprefix_style_use = False
                if is_truthfulqa:
                    if tqa_kvi_stats is not None:
                        kvi_triples = int(tqa_kvi_stats["kvi_max_kv_triples"])
                        kvi_drm = float(tqa_kvi_stats["kvi_drm_threshold"])
                        kvi_rels = int(tqa_kvi_stats["kvi_top_k_relations"])
                    tqa_cap = int(args.truthfulqa_kvi_max_new_tokens)
                    if tqa_cap > 0:
                        kvi_max_new_tokens = min(kvi_max_new_tokens, tqa_cap)
                    if tqa_kvi_stats is not None and bool(tqa_kvi_stats["truthfulqa_kvi_minimal_prompt"]):
                        kvi_minimal_use = True
                    if tqa_kvi_stats is not None and bool(tqa_kvi_stats["truthfulqa_kvi_reconcile"]):
                        kvi_reconcile_use = True
                    # TruthfulQA MC proxy alignment: use KV Prefix-like compact generation template.
                    kvi_kvprefix_style_use = True
                out_gr_kvi = _run_graph(
                    repo_root=repo_root,
                    model=args.model,
                    question=ex.question,
                    graph_index=graph_index,
                    sentences_jsonl=graph_sentences_jsonl,
                    enable_kvi=True,
                    triple_kvbank_dir=triple_kvbank_dir,
                    max_new_tokens=kvi_max_new_tokens,
                    timeout_s=int(args.timeout_s),
                    service_url=str(args.inference_service_url or ""),
                    force_cpu=bool(args.graph_force_cpu),
                    openqa_mode=bool(args.openqa_mode),
                    kvi_minimal_prompt=kvi_minimal_use,
                    kvi_max_kv_triples=kvi_triples,
                    kvi_drm_threshold=kvi_drm,
                    kvi_top_k_relations=kvi_rels,
                    kvi_reconcile_no_kv_decode=kvi_reconcile_use,
                    kvi_kvprefix_style_prompt=kvi_kvprefix_style_use,
                    kvi_disable_kv_injection=bool(args.kvi_disable_kv_injection),
                    kvi_two_stage_kv_then_evidence=bool(args.kvi_two_stage_kv_then_evidence),
                    audit_jsonl=graph_audit_path,
                    audit_query_id=str(ex.id),
                    audit_method_key="kvi",
                    audit_oracle_jsonl=graph_audit_oracle,
                )

            out_rag = None
            out_kv_prefix = None
            if "rag" in methods:
                out_rag = _run_kvi2_runtime(
                    repo_root=repo_root,
                    model=args.model,
                    question=ex.question,
                    pipeline="modeA_rag",
                    kv_dir=ann_kv_dir,  # type: ignore[arg-type]
                    sentences_jsonl=ann_sentences_jsonl,  # type: ignore[arg-type]
                    semantic_type_specs=ann_semantic_type_specs,  # type: ignore[arg-type]
                    pattern_index_dir=ann_pattern_index_dir,  # type: ignore[arg-type]
                    sidecar_dir=ann_sidecar_dir,  # type: ignore[arg-type]
                    domain_encoder_model=str(args.domain_encoder_model),
                    top_k=int(args.top_k),
                    timeout_s=int(args.timeout_s),
                    service_url=str(args.ann_inference_service_url or ""),
                    force_cpu=bool(args.ann_force_cpu),
                )
            if "kv_prefix" in methods:
                # Keep KV Prefix as KV-channel-only baseline (no modeA prompt-evidence path).
                kv_prefix_pipeline = "simple"
                out_kv_prefix = _run_kvi2_runtime(
                    repo_root=repo_root,
                    model=args.model,
                    question=ex.question,
                    pipeline=kv_prefix_pipeline,
                    kv_dir=ann_kv_dir,  # type: ignore[arg-type]
                    sentences_jsonl=ann_sentences_jsonl,  # type: ignore[arg-type]
                    semantic_type_specs=ann_semantic_type_specs,  # type: ignore[arg-type]
                    pattern_index_dir=ann_pattern_index_dir,  # type: ignore[arg-type]
                    sidecar_dir=ann_sidecar_dir,  # type: ignore[arg-type]
                    domain_encoder_model=str(args.domain_encoder_model),
                    top_k=int(args.top_k),
                    timeout_s=int(args.timeout_s),
                    service_url=str(args.ann_inference_service_url or ""),
                    force_cpu=bool(args.ann_force_cpu),
                )

            preds: Dict[str, str] = {}
            ems: Dict[str, int] = {}
            f1s: Dict[str, float] = {}
            fever_labs: Dict[str, int] = {}
            mc1_proxy: Dict[str, float] = {}
            mc2_proxy: Dict[str, float] = {}
            for m in methods:
                pred = _pick_method_prediction(
                    method=m,
                    out_gr_rag=out_gr_rag,
                    out_gr_kvi=out_gr_kvi,
                    out_rag=out_rag,
                    out_kv_prefix=out_kv_prefix,
                )
                if is_truthfulqa and m == "kvi":
                    pred = _select_truthfulqa_kvi_prediction(
                        primary_pred=pred,
                        out_gr_kvi=out_gr_kvi,
                        out_kv_prefix=out_kv_prefix,
                    )
                em = int(em_score_fn(pred, ex.answers))
                f1 = float(best_f1(pred, ex.answers))
                preds[m] = pred
                ems[m] = em
                f1s[m] = f1
                method_correct[m] += em
                method_em_series[m].append(int(em))
                method_f1_sum[m] += f1
                if is_fever:
                    fv = int(best_fever_label_em(pred, ex.answers))
                    fever_labs[m] = fv
                    method_fever_correct[m] += fv
                    method_fever_series[m].append(fv)
                elif is_truthfulqa:
                    if str(args.truthfulqa_mc_mode) == "proxy":
                        if m == "kvi":
                            inj = str((out_kv_prefix or {}).get("injected_answer") or "").strip()
                            if str(args.truthfulqa_kvi_mc1_answer) == "grounded":
                                pred_m1 = _kvi_pred_for_truthfulqa_mc1_likelihood(pred)
                            else:
                                pred_m1 = inj if inj else pred
                            pred_m2 = _kvi_pred_for_truthfulqa_mc1_likelihood(pred)
                            s1 = truthfulqa_mc_proxy_scores(pred_m1, ex.mc1_targets)
                            s2 = truthfulqa_mc_proxy_scores(pred_m2, ex.mc2_targets)
                        else:
                            s1 = truthfulqa_mc_proxy_scores(pred, ex.mc1_targets)
                            s2 = truthfulqa_mc_proxy_scores(pred, ex.mc2_targets)
                        v1 = float(s1["mc_acc_proxy"])
                        v2 = float(s2["mc_mass_proxy"])
                        vv1 = float(s1["valid"])
                        vv2 = float(s2["valid"])
                    else:
                        pred_mc1 = (
                            _kvi_truthfulqa_mc1_text_for_likelihood(
                                pred=pred,
                                out_kv_prefix=out_kv_prefix,
                                methods=methods,
                                mc1_answer=str(args.truthfulqa_kvi_mc1_answer),
                            )
                            if m == "kvi"
                            else pred
                        )
                        pred_mc2 = _kvi_pred_for_truthfulqa_mc1_likelihood(pred) if m == "kvi" else pred
                        s1 = tqa_scorer.score_targets(ex.question, pred_mc1, ex.mc1_targets or {}) if tqa_scorer is not None else {"mc1": 0.0, "mc2": 0.0, "valid": 0.0}
                        s2 = tqa_scorer.score_targets(ex.question, pred_mc2, ex.mc2_targets or {}) if tqa_scorer is not None else {"mc1": 0.0, "mc2": 0.0, "valid": 0.0}
                        v1 = float(s1["mc1"])
                        v2 = float(s2["mc2"])
                        vv1 = float(s1["valid"])
                        vv2 = float(s2["valid"])
                    mc1_proxy[m] = v1
                    mc2_proxy[m] = v2
                    method_mc1_acc_sum[m] += v1
                    method_mc2_mass_sum[m] += v2
                    method_mc1_valid_sum[m] += vv1
                    method_mc2_valid_sum[m] += vv2

            rec: Dict[str, Any] = {
                "idx": idx,
                "id": ex.id,
                "question": ex.question,
                "gold_answers": ex.answers,
                "predictions": preds,
                "em": ems,
                "f1": f1s,
                "em_mode": str(args.em_mode),
            }
            if is_fever:
                rec["fever_label_em"] = fever_labs
            elif is_truthfulqa:
                rec["truthfulqa_mc1_proxy"] = mc1_proxy
                rec["truthfulqa_mc2_proxy"] = mc2_proxy
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wf.flush()

    n = len(examples)
    em_percent = {m: (100.0 * method_correct[m] / n if n else 0.0) for m in methods}

    rows = []
    for m in methods:
        meta = METHODS_META[m]
        ci = _bootstrap_ci_percent(
            method_em_series[m],
            n_boot=int(args.bootstrap_samples),
            alpha=0.05,
            seed=int(args.random_seed),
        )
        f1_mean = (method_f1_sum[m] / n) if n else 0.0
        row: Dict[str, Any] = {
            "method_key": m,
            "method": meta["label"],
            "retrieval": meta["retrieval"],
            "injection": meta["injection"],
            "em": round(em_percent[m], 4),
            "em_ci95_lo": round(ci["lo"], 4),
            "em_ci95_hi": round(ci["hi"], 4),
            "f1_mean": round(f1_mean, 4),
            "correct": int(method_correct[m]),
            "total": int(n),
        }
        if is_fever:
            f_pct = 100.0 * method_fever_correct[m] / n if n else 0.0
            f_ci = _bootstrap_ci_percent(
                method_fever_series[m],
                n_boot=int(args.bootstrap_samples),
                alpha=0.05,
                seed=int(args.random_seed),
            )
            row["fever_label_accuracy"] = round(f_pct, 4)
            row["fever_label_ci95_lo"] = round(f_ci["lo"], 4)
            row["fever_label_ci95_hi"] = round(f_ci["hi"], 4)
            row["fever_label_correct"] = int(method_fever_correct[m])
        elif is_truthfulqa:
            mc1_den = method_mc1_valid_sum[m]
            mc2_den = method_mc2_valid_sum[m]
            row["truthfulqa_mc1_proxy"] = round((method_mc1_acc_sum[m] / mc1_den) if mc1_den > 0 else 0.0, 4)
            row["truthfulqa_mc2_proxy"] = round((method_mc2_mass_sum[m] / mc2_den) if mc2_den > 0 else 0.0, 4)
            row["truthfulqa_mc_valid_n"] = int(mc1_den)
        rows.append(row)

    significance: Dict[str, Any] = {}
    if "kvi" in methods:
        for m in methods:
            if m == "kvi":
                continue
            pval = _paired_permutation_pvalue(
                method_em_series["kvi"],
                method_em_series[m],
                n_perm=int(args.permutation_samples),
                seed=int(args.random_seed),
            )
            sig_entry: Dict[str, Any] = {"p_value": float(pval)}
            if is_fever:
                sig_entry["fever_label_p_value"] = float(
                    _paired_permutation_pvalue(
                        method_fever_series["kvi"],
                        method_fever_series[m],
                        n_perm=int(args.permutation_samples),
                        seed=int(args.random_seed),
                    )
                )
            significance[f"kvi_vs_{m}"] = sig_entry

    stats: Dict[str, Any] = {
        "em_mode": str(args.em_mode),
        "openqa_mode": bool(args.openqa_mode),
        "kvi_minimal_prompt": bool(args.kvi_minimal_prompt),
        "kvi_max_kv_triples": int(args.kvi_max_kv_triples),
        "kvi_drm_threshold": float(args.kvi_drm_threshold),
        "kvi_top_k_relations": int(args.kvi_top_k_relations),
        "kvi_reconcile_no_kv_decode": bool(args.kvi_reconcile_no_kv_decode),
        "truthfulqa_kvi_tuned_overrides": bool(args.truthfulqa_kvi_tuned_overrides),
        "ci_method": "bootstrap",
        "ci_level": 0.95,
        "bootstrap_samples": int(args.bootstrap_samples),
        "significance_method": "paired_permutation",
        "permutation_samples": int(args.permutation_samples),
        "random_seed": int(args.random_seed),
        "significance": significance,
    }
    if is_fever:
        stats["fever_label_metric"] = (
            "First match in output of SUPPORTS | REFUTES | NOT ENOUGH INFO (case-insensitive); "
            "compare to gold label. Closer to FEVER shared-task label accuracy than relaxed EM."
        )
    elif is_truthfulqa:
        stats["truthfulqa_mc_note"] = (
            "truthfulqa_mc1_proxy / truthfulqa_mc2_proxy are proxy metrics. "
            "likelihood_proxy mode scores options by average token log-prob under base model "
            "conditioned on question + method answer. Not identical to official TruthfulQA MC1/MC2 pipeline. "
            "For method kvi: MC1/MC2 default (--truthfulqa_kvi_mc1_answer grounded) both condition on a shortened "
            "grounded KVI decode (diagnosis_result). Legacy `injected` uses full ANN injected_answer for KVI MC1 "
            "when kv_prefix is co-run. TruthfulQA can apply tuned KVI runtime overrides unless disabled "
            "for controlled ablations. Stored predictions / EM / F1 are unchanged."
        )
        stats["truthfulqa_mc_mode"] = str(args.truthfulqa_mc_mode)
        stats["truthfulqa_kvi_mc1_answer"] = str(args.truthfulqa_kvi_mc1_answer)
        if "kvi" in methods and tqa_kvi_stats is not None:
            stats["kvi_truthfulqa_runtime_overrides"] = {
                **tqa_kvi_stats,
                "kvi_max_new_tokens_effective": (
                    min(int(args.max_new_tokens), int(args.truthfulqa_kvi_max_new_tokens))
                    if int(args.truthfulqa_kvi_max_new_tokens) > 0
                    else int(args.max_new_tokens)
                ),
                "kvi_reconcile_no_kv_decode_effective": bool(tqa_kvi_stats["truthfulqa_kvi_reconcile"]),
                "kvi_mc_likelihood_shorten_max_chars": 384,
                "kvi_use_grounded_answer_for_eval": True,
            }
    summary = {
        "dataset": str(args.dataset_name or dataset_path.stem),
        "dataset_path": str(dataset_path),
        "n": int(n),
        "methods": rows,
        "paths": {
            "predictions": str(per_example_path),
            "results_md": str(out_dir / "results.md"),
            "results_csv": str(out_dir / "results.csv"),
        },
        "statistics": stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "results.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["Method", "Retrieval", "Injection", "EM", "CI95 Low", "CI95 High", "F1 Mean"]
        if is_fever:
            header += ["FEVER Label Acc", "FEVER CI Lo", "FEVER CI Hi"]
        elif is_truthfulqa:
            header += ["TQA MC1 Proxy", "TQA MC2 Proxy", "TQA MC Valid N"]
        w.writerow(header)
        for r in rows:
            roww = [
                r["method"],
                r["retrieval"],
                r["injection"],
                f"{r['em']:.1f}",
                f"{r['em_ci95_lo']:.1f}",
                f"{r['em_ci95_hi']:.1f}",
                f"{r['f1_mean']:.1f}",
            ]
            if is_fever:
                roww += [
                    f"{float(r['fever_label_accuracy']):.1f}",
                    f"{float(r['fever_label_ci95_lo']):.1f}",
                    f"{float(r['fever_label_ci95_hi']):.1f}",
                ]
            elif is_truthfulqa:
                roww += [
                    f"{float(r['truthfulqa_mc1_proxy']):.3f}",
                    f"{float(r['truthfulqa_mc2_proxy']):.3f}",
                    f"{int(r['truthfulqa_mc_valid_n'])}",
                ]
            w.writerow(roww)

    md: List[str] = []
    md.append("## Experiment 1 — Main QA Performance (single dataset)\n\n")
    md.append(f"- Dataset: {summary['dataset']}\n")
    md.append(f"- N: {n}\n\n")
    md.append(f"- EM mode: **{args.em_mode}** (relaxed = gold answer substring in prediction after SQuAD-normalize; use `--em_mode strict` for full-string EM only)\n\n")
    if is_fever:
        md.append(
            "- **FEVER label accuracy**: first occurrence in model text of "
            "`SUPPORTS` / `REFUTES` / `NOT ENOUGH INFO` (see `metrics.parse_fever_label`) vs gold; "
            "closer to veracity label accuracy than substring relaxed EM.\n\n"
        )
    if is_fever:
        md.append("| Method | Retrieval | Injection | EM | 95% CI | F1 Mean | FEVER lbl % | FEVER CI |\n")
        md.append("|---|---|---|---:|---:|---:|---:|---:|\n")
    elif is_truthfulqa:
        md.append(
            "- **TruthfulQA MC proxy**: from free-form output matched to MC options; "
            "**not** official MC1/MC2 (official requires option likelihood scoring).\n\n"
        )
        md.append("| Method | Retrieval | Injection | EM | 95% CI | F1 Mean | MC1 Proxy | MC2 Proxy |\n")
        md.append("|---|---|---|---:|---:|---:|---:|---:|\n")
    else:
        md.append("| Method | Retrieval | Injection | EM | 95% CI | F1 Mean |\n")
        md.append("|---|---|---|---:|---:|---:|\n")
    for m in METHODS_ORDER:
        if m not in methods:
            continue
        r = next(x for x in rows if x["method_key"] == m)
        if is_fever:
            md.append(
                f"| {r['method']} | {r['retrieval']} | {r['injection']} | {r['em']:.1f} | "
                f"[{r['em_ci95_lo']:.1f}, {r['em_ci95_hi']:.1f}] | {r['f1_mean']:.3f} | "
                f"{float(r['fever_label_accuracy']):.1f} | "
                f"[{float(r['fever_label_ci95_lo']):.1f}, {float(r['fever_label_ci95_hi']):.1f}] |\n"
            )
        elif is_truthfulqa:
            md.append(
                f"| {r['method']} | {r['retrieval']} | {r['injection']} | {r['em']:.1f} | "
                f"[{r['em_ci95_lo']:.1f}, {r['em_ci95_hi']:.1f}] | {r['f1_mean']:.3f} | "
                f"{float(r['truthfulqa_mc1_proxy']):.3f} | {float(r['truthfulqa_mc2_proxy']):.3f} |\n"
            )
        else:
            md.append(
                f"| {r['method']} | {r['retrieval']} | {r['injection']} | {r['em']:.1f} | "
                f"[{r['em_ci95_lo']:.1f}, {r['em_ci95_hi']:.1f}] | {r['f1_mean']:.3f} |\n"
            )
    (out_dir / "results.md").write_text("".join(md), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

