"""
知识含量过滤（DeepSeek）

输出目标
- 对段落打标签：KEEP / DROP
- 保留摘要、结果、关键结论、指南/共识、诊断/治疗要点、数据表格解读等
- 丢弃：前言泛泛背景、病例患者叙事细节、未来展望、方法学局限/讨论噪声、致谢等

重要策略
- 表格（markdown table）强制 KEEP（医疗场景表格价值高）
- 对不确定输出：默认 KEEP（防止误删），可配置 strict 模式改为 DROP
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .deepseek_client import DeepSeekClient, DeepSeekClientConfig


SYSTEM_PROMPT = """你是医学论文知识抽取过滤器。你的任务是判断给定段落是否应进入“raw context（用于知识库）”。只输出严格 JSON。"""

USER_TEMPLATE = """判断下面段落是否“知识含量高且可复用”，并输出 JSON：

规则（必须遵守）：
1) 如果是表格（markdown 表格或包含表格 marker），一律 KEEP。
2) KEEP：摘要、研究结果、核心结论/指南推荐、诊断治疗要点、药物剂量/禁忌、统计结果解读、表格数据说明。
3) DROP：前言泛背景、病例患者叙事/基本信息、未来展望、作者讨论的“方法学问题/局限”类、致谢/基金/伦理声明、参考文献。
4) 如果不确定，输出 UNCERTAIN，并给出理由。

输出 JSON schema：
{{"label":"KEEP|DROP|UNCERTAIN","category":"ABSTRACT|RESULT|GUIDELINE|CONCLUSION|TABLE|BACKGROUND|CASE|FUTURE|LIMITATION|METHOD|OTHER","reason":"一句话原因"}}

段落：
<<<
{text}
>>>
"""


@dataclass(frozen=True)
class KnowledgeFilterConfig:
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"
    strict_drop_uncertain: bool = False


def _looks_like_table(text: str) -> bool:
    if "<!-- table:" in text:
        return True
    # markdown table heuristic
    if re.search(r"^\s*\|.+\|\s*$", text, flags=re.MULTILINE) and re.search(r"^\s*\|\s*---", text, flags=re.MULTILINE):
        return True
    return False


def _safe_parse_json(s: str) -> Optional[Dict[str, Any]]:
    s = s.strip()
    # try to extract first json object
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class DeepSeekKnowledgeFilter:
    def __init__(self, cfg: KnowledgeFilterConfig) -> None:
        self.cfg = cfg
        self.client = DeepSeekClient(
            DeepSeekClientConfig(
                base_url=cfg.deepseek_base_url,
                api_key_env=cfg.api_key_env,
                model=cfg.deepseek_model,
            )
        )

    def classify(self, text: str) -> Dict[str, Any]:
        if _looks_like_table(text):
            return {"label": "KEEP", "category": "TABLE", "reason": "table content"}

        user = USER_TEMPLATE.format(text=text[:6000])
        out = self.client.chat(system=SYSTEM_PROMPT, user=user, temperature=0.0)
        obj = _safe_parse_json(out) or {"label": "UNCERTAIN", "category": "OTHER", "reason": "parse_failed"}

        label = str(obj.get("label", "UNCERTAIN")).upper()
        if label not in {"KEEP", "DROP", "UNCERTAIN"}:
            label = "UNCERTAIN"
        obj["label"] = label
        if "category" not in obj:
            obj["category"] = "OTHER"
        if "reason" not in obj:
            obj["reason"] = ""
        return obj

    def filter_paragraphs(self, paragraphs: List[str]) -> Tuple[List[str], Dict[str, Any]]:
        kept: List[str] = []
        stats = {"kept": 0, "dropped": 0, "uncertain": 0, "details": []}
        for p in paragraphs:
            res = self.classify(p)
            label = res["label"]
            if label == "KEEP":
                kept.append(p)
                stats["kept"] += 1
            elif label == "DROP":
                stats["dropped"] += 1
            else:
                stats["uncertain"] += 1
                if self.cfg.strict_drop_uncertain:
                    stats["dropped"] += 1
                else:
                    kept.append(p)
                    stats["kept"] += 1
            stats["details"].append(res)
        return kept, stats


