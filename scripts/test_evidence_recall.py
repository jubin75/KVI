"""
Evidence Recall Unit Test (keyword queries -> retrieval hits)

Goal:
- Sanity-check that extracted evidence blocks exist and that evidence retrieval can "hit" them
  for a set of representative user queries (Chinese).

This is a lightweight, script-style unittest (similar to test_postprocess.py).

Run (flat layout, remote):
  python -u scripts/test_evidence_recall.py \
    --kv_dir_evidence /home/jb/KVI/topics/SFTSV/work/kvbank_evidence \
    --blocks_jsonl_evidence /home/jb/KVI/topics/SFTSV/work/blocks.evidence.jsonl \
    --domain_encoder_model sentence-transformers/all-MiniLM-L6-v2

Run (monorepo layout, local):
  python -u external_kv_injection/scripts/test_evidence_recall.py \
    --kv_dir_evidence .../kvbank_evidence \
    --blocks_jsonl_evidence .../blocks.evidence.jsonl \
    --domain_encoder_model sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

try:
    from external_kv_injection.src.kv_bank import FaissKVBank  # type: ignore
    from external_kv_injection.src.encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig  # type: ignore
except ModuleNotFoundError:
    from src.kv_bank import FaissKVBank  # type: ignore
    from src.encoders.hf_sentence_encoder import HFSentenceEncoder, HFSentenceEncoderConfig  # type: ignore


@dataclass(frozen=True)
class TestCfg:
    kv_dir_evidence: str
    blocks_jsonl_evidence: str
    domain_encoder_model: str
    top_k: int = 16
    min_blocks_loaded: int = 200
    min_hit_rate_overall: float = 0.35
    min_categories_with_hits: int = 3
    queries_by_category: Dict[str, List[str]] = None  # type: ignore[assignment]


_CFG: TestCfg | None = None


def _has_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s or ""))


def _load_blocks_text(blocks_jsonl: str) -> Dict[str, str]:
    p = Path(str(blocks_jsonl))
    if not p.exists():
        raise FileNotFoundError(f"blocks_jsonl_evidence not found: {p}")
    out: Dict[str, str] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            bid = rec.get("block_id")
            txt = rec.get("text")
            if isinstance(bid, str) and isinstance(txt, str) and bid and txt:
                out[bid] = txt
    return out


def _norm_for_anchor(s: str) -> str:
    t = (s or "").lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _category_anchors() -> Dict[str, re.Pattern]:
    # Anchors are intentionally English-heavy because evidence blocks are often English.
    # This test is about recall/hits, not exact keyword string matching in Chinese.
    return {
        "transmission": re.compile(
            r"(tick|tick\s*bite|hard\s*tick|vector|mosquito|aerosol|droplet|airborne|blood|body\s*fluid|fecal|oral|contact)",
            re.I,
        ),
        "pathogen": re.compile(r"(virus|viral|bacteria|bacterial|rna\s*virus|dna\s*virus|bunyavirus|zoonotic|host|reservoir)", re.I),
        "pathogenesis": re.compile(r"(pathogenesis|mechanism|immune|immun|inflamm|cytokine|chemokine|organ\s*damage|MODS)", re.I),
        "epidemiology": re.compile(r"(incubation|season|fatality|mortality|incidence|prevalence|geographic|province|china|henan|anhui|hubei|hunan|jiangxi)", re.I),
        "diagnosis_prevention": re.compile(r"(diagnos|pcr|elisa|serolog|assay|vaccine|prevent|control|prophylaxis)", re.I),
    }


def _queries_by_category() -> Dict[str, List[str]]:
    return {
        "transmission": [
            "主要传播途径",
            "传播方式",
            "是否人传人",
            "是否经空气传播",
            "是否经飞沫传播",
            "是否经血液传播",
            "是否经体液传播",
            "是否经蚊虫传播",
            "是否经蜱虫传播",
            "是否经粪口途径传播",
        ],
        "pathogen": [
            "病原体是什么",
            "病毒还是细菌",
            "病毒分类",
            "RNA病毒",
            "DNA病毒",
            "是否为人畜共患病",
            "病毒宿主",
        ],
        "pathogenesis": [
            "发病机制",
            "致病机制",
            "病理机制",
            "免疫机制",
            "炎症反应",
            "免疫风暴",
            "多器官损伤",
            "靶器官",
        ],
        "epidemiology": [
            "潜伏期",
            "高发季节",
            "易感人群",
            "病死率",
            "发病率",
            "地区分布",
        ],
        "diagnosis_prevention": [
            "诊断标准",
            "核酸检测",
            "血清学检测",
            "防控措施",
            "预防方法",
            "是否有疫苗",
        ],
    }


def _load_topic_config(topic_config: str) -> Dict[str, Any]:
    p = Path(str(topic_config))
    if not p.exists():
        raise FileNotFoundError(f"topic_config not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _queries_from_config(cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    q = cfg.get("evidence_recall_queries")
    if not isinstance(q, dict) or not q:
        raise ValueError("Missing config key: evidence_recall_queries (required for this test)")
    out: Dict[str, List[str]] = {}
    for k, v in q.items():
        if not isinstance(k, str) or not isinstance(v, list):
            continue
        out[k] = [str(x) for x in v if str(x).strip()]
    if not out:
        raise ValueError("evidence_recall_queries is empty after parsing")
    return out


class TestEvidenceRecall(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _CFG is None:
            raise RuntimeError("TestCfg not initialized")
        cls.cfg = _CFG
        cls.blocks = _load_blocks_text(cls.cfg.blocks_jsonl_evidence)
        cls.bank = FaissKVBank.load(cls.cfg.kv_dir_evidence)
        cls.encoder = HFSentenceEncoder(HFSentenceEncoderConfig(model_name_or_path=cls.cfg.domain_encoder_model))
        cls.anchors = _category_anchors()

    def test_blocks_loaded(self):
        self.assertGreaterEqual(
            len(self.blocks),
            int(self.cfg.min_blocks_loaded),
            msg=f"Too few evidence blocks loaded from {self.cfg.blocks_jsonl_evidence}",
        )

    def test_retrieval_hits_overall(self):
        queries = dict(self.cfg.queries_by_category or {})
        total = 0
        hit = 0
        categories_with_hits = set()
        top1_ids: List[str] = []
        per_query: List[Tuple[str, str, int, int, str]] = []  # (cat, q, hit_n, top_k, top1_id)

        for cat, qs in queries.items():
            pat = self.anchors.get(cat)
            if pat is None:
                # Unknown category keys in config are ignored (forward-compatible).
                continue
            cat_hit = 0
            for q in qs:
                total += 1
                qv = self.encoder.encode(q)[0]
                items, _dbg = self.bank.search(qv, top_k=int(self.cfg.top_k), filters=None)
                texts: List[str] = []
                ids: List[str] = []
                for it in items:
                    bid = (it.meta or {}).get("block_id") or (it.meta or {}).get("chunk_id") or (it.meta or {}).get("id")
                    if not bid:
                        continue
                    ids.append(str(bid))
                    t = self.blocks.get(str(bid), "")
                    if t:
                        texts.append(t)
                if ids:
                    top1_ids.append(ids[0])

                hit_n = sum(1 for t in texts if pat.search(t or ""))
                ok = hit_n > 0
                per_query.append((cat, q, int(hit_n), int(len(texts)), str(ids[0]) if ids else ""))
                if ok:
                    hit += 1
                    cat_hit += 1
            if cat_hit > 0:
                categories_with_hits.add(cat)

        hit_rate = (hit / max(1, total))
        # ---- Print a detailed recall report (not just "OK") ----
        print("\n=== Evidence Recall Report ===", flush=True)
        print(f"top_k={int(self.cfg.top_k)} blocks_loaded={len(self.blocks)} bank_dim={getattr(self.bank, 'dim', None)}", flush=True)
        for cat, q, hit_n, denom, top1 in per_query:
            # Example: 地区分布 hits=3/16 top1=...
            print(f"[recall] category={cat} query={q} hits={hit_n}/{denom} top1={top1}", flush=True)
        # Category summary
        by_cat: Dict[str, Tuple[int, int]] = {}
        for cat, _q, hit_n, _denom, _top1 in per_query:
            h, n = by_cat.get(cat, (0, 0))
            by_cat[cat] = (h + (1 if hit_n > 0 else 0), n + 1)
        for cat in sorted(by_cat.keys()):
            h, n = by_cat[cat]
            print(f"[summary] category={cat} hit_queries={h}/{n} rate={(h/max(1,n)):.3f}", flush=True)
        print(f"[summary] overall hit_queries={hit}/{total} rate={hit_rate:.3f}", flush=True)

        self.assertGreaterEqual(
            hit_rate,
            float(self.cfg.min_hit_rate_overall),
            msg=f"Low overall evidence retrieval hit rate: hit={hit} total={total} rate={hit_rate:.3f}",
        )
        self.assertGreaterEqual(
            len(categories_with_hits),
            int(self.cfg.min_categories_with_hits),
            msg=f"Too few categories with any hits: {sorted(list(categories_with_hits))}",
        )

        # Degeneracy check: top1 should not be identical for most queries.
        if top1_ids:
            uniq = len(set(top1_ids))
            self.assertGreaterEqual(uniq, 3, msg=f"Retrieval appears degenerate (top1 ids too few unique): uniq={uniq}")

    def test_alignment_of_bank_and_blocks_jsonl(self):
        # Sample a few queries and ensure returned ids exist in blocks jsonl.
        qs = ["传播途径", "发病机制", "诊断标准", "地区分布", "是否有疫苗"]
        missing = 0
        total = 0
        for q in qs:
            qv = self.encoder.encode(q)[0]
            items, _dbg = self.bank.search(qv, top_k=8, filters=None)
            for it in items:
                bid = (it.meta or {}).get("block_id") or (it.meta or {}).get("chunk_id") or (it.meta or {}).get("id")
                if not bid:
                    continue
                total += 1
                if str(bid) not in self.blocks:
                    missing += 1
        # Allow a small mismatch (older banks / ids), but not too much.
        if total > 0:
            self.assertLessEqual(missing / total, 0.25, msg=f"Too many retrieved ids missing in blocks jsonl: {missing}/{total}")


def _parse_args() -> TestCfg:
    p = argparse.ArgumentParser()
    p.add_argument("--kv_dir_evidence", default=None, help="If omitted, infer from --topic_config build.work_dir.")
    p.add_argument("--blocks_jsonl_evidence", default=None, help="If omitted, infer from --topic_config build.work_dir.")
    p.add_argument("--domain_encoder_model", default=None, help="If omitted, infer from --topic_config build.retrieval_encoder_model.")
    p.add_argument(
        "--topic_config",
        default=str(Path(__file__).resolve().parents[1] / "config" / "topics" / "SFTSV" / "config.json"),
        help="Topic config.json path. Queries are loaded from its evidence_recall_queries field.",
    )
    p.add_argument("--top_k", type=int, default=16)
    p.add_argument("--min_blocks_loaded", type=int, default=200)
    p.add_argument("--min_hit_rate_overall", type=float, default=0.35)
    p.add_argument("--min_categories_with_hits", type=int, default=3)
    args, _rest = p.parse_known_args()

    cfg = _load_topic_config(str(args.topic_config))
    queries_by_cat = _queries_from_config(cfg)
    work_dir = ((cfg.get("build") or {}) if isinstance(cfg.get("build"), dict) else {}).get("work_dir")
    retrieval_encoder_model = ((cfg.get("build") or {}) if isinstance(cfg.get("build"), dict) else {}).get("retrieval_encoder_model")

    kv_dir_evidence = str(args.kv_dir_evidence or (str(work_dir) + "/kvbank_evidence"))
    blocks_jsonl_evidence = str(args.blocks_jsonl_evidence or (str(work_dir) + "/blocks.evidence.jsonl"))
    domain_encoder_model = str(args.domain_encoder_model or retrieval_encoder_model or "")
    if not domain_encoder_model:
        raise SystemExit("Missing --domain_encoder_model and build.retrieval_encoder_model in topic_config.")

    return TestCfg(
        kv_dir_evidence=kv_dir_evidence,
        blocks_jsonl_evidence=blocks_jsonl_evidence,
        domain_encoder_model=domain_encoder_model,
        top_k=int(args.top_k),
        min_blocks_loaded=int(args.min_blocks_loaded),
        min_hit_rate_overall=float(args.min_hit_rate_overall),
        min_categories_with_hits=int(args.min_categories_with_hits),
        queries_by_category=queries_by_cat,
    )


if __name__ == "__main__":
    _CFG = _parse_args()
    unittest.main(argv=[sys.argv[0]])


