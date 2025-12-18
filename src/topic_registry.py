"""
Topic registry for "专题库" workflows.

We keep this intentionally small and explicit (SFTSV / SARS-CoV-2) to make evaluation predictable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class TopicSpec:
    slug: str
    display_name: str
    # quick keywords for cheap heuristics / logging (not authoritative)
    keywords: Tuple[str, ...]
    # DeepSeek doc-level filtering prompt
    system_prompt: str
    user_template: str


_SYSTEM = "你是医学文献的主题筛选器。你的任务是判断一篇 PDF 文献是否与给定专题高度相关。只输出严格 JSON。"

_USER_TEMPLATE = """判断下面这篇文献是否应进入“{topic}专题知识库”。只输出 JSON。

要求：
1) 只有当文献的主要主题与专题强相关时才 KEEP（例如研究对象、病原体/疾病主体就是该专题）。
2) 如果只是顺带提到/对照提到/引用相关术语，但主题并非该专题，则 DROP。
3) 如果不确定，输出 UNCERTAIN，并在 reason 里说明你缺少什么信息。

输出 JSON schema：
{{"label":"KEEP|DROP|UNCERTAIN","reason":"一句话原因"}}

文献信息：
- file_name: {file_name}

文献正文片段（可能包含 OCR/PDF 断行噪声）：
<<<
{text}
>>>
"""


def get_topic_specs() -> Dict[str, TopicSpec]:
    return {
        "sftsv": TopicSpec(
            slug="sftsv",
            display_name="SFTSV（发热伴血小板减少综合征）",
            keywords=("sftsv", "severe fever with thrombocytopenia", "发热伴血小板减少", "蜱"),
            system_prompt=_SYSTEM,
            user_template=_USER_TEMPLATE,
        ),
        "sarscov2": TopicSpec(
            slug="sarscov2",
            display_name="SARS-CoV-2（新冠）",
            keywords=("sars-cov-2", "covid", "covid-19", "新冠", "coronavirus"),
            system_prompt=_SYSTEM,
            user_template=_USER_TEMPLATE,
        ),
    }


def get_topic_spec(slug: str) -> TopicSpec:
    specs = get_topic_specs()
    if slug not in specs:
        raise KeyError(f"Unknown topic slug: {slug}. Available: {sorted(specs.keys())}")
    return specs[slug]


