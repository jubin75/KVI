"""
Scheme C — Graph Retriever (runtime, v3: 三元 KVI).

Given a user query, this module:
1. **Entity Recognition** — identifies entities in the query that match
   graph nodes (via the entity index).
2. **Intent → Relation mapping** — maps the query intent to relevant
   relation types for graph traversal.
3. **Graph Walk** — traverses the knowledge graph from matched entities
   along relevant relation edges.
4. **Evidence Collection** — collects provenance sentences from the
   traversed triples for inclusion in the RAG prompt.
5. **Entity Context** — builds a prompt-level entity context sentence
   (complementary to KV injection subject anchoring).

v3: Entity context in the prompt works **alongside** triple KV injection
(short KV for attention structure) rather than *replacing* it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .schema import (
    KnowledgeGraphIndex,
    Triple,
    _normalise,
    load_relation_types,
    DEFAULT_RELATION_TYPES,
)


# ---------------------------------------------------------------------------
# Intent → Relation mapping
# ---------------------------------------------------------------------------

def _build_intent_relation_map(
    relation_types: Dict[str, Dict[str, Any]]
) -> Dict[str, List[str]]:
    """
    Build a mapping from query intent → list of relevant relation types.

    Each relation type declares which intents it serves via
    ``retrieval_intents`` in its config.
    """
    intent_map: Dict[str, List[str]] = {}
    for rel_name, rel_cfg in relation_types.items():
        for intent in (rel_cfg.get("retrieval_intents") or []):
            intent_map.setdefault(str(intent).lower(), []).append(rel_name)
    return intent_map


# ---------------------------------------------------------------------------
# Entity recognition (lightweight, index-based)
# ---------------------------------------------------------------------------

def recognise_entities(
    query: str,
    graph: KnowledgeGraphIndex,
) -> List[Dict[str, Any]]:
    """
    Recognise entity mentions in *query* by matching against the graph's
    entity index.

    Returns a list of matches::

        [{"mention": "SFTSV", "node_id": "node_0001", "entity_name": "SFTSV"}, ...]

    Strategy:
    * Longest-match-first — prefer "发热伴血小板减少综合征病毒" over "发热"
    * Case-insensitive + normalised matching
    """
    query_lower = query.lower()
    # Sort index entries by length (longest first) to prefer longer matches
    candidates = sorted(
        graph.entity_index.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    )
    matches: List[Dict[str, Any]] = []
    consumed_spans: List[Tuple[int, int]] = []

    for norm_name, node_id in candidates:
        if not norm_name:
            continue
        # Find all occurrences in query
        start = 0
        while True:
            idx = query_lower.find(norm_name, start)
            if idx < 0:
                break
            end = idx + len(norm_name)
            # Check for span overlap with already-matched entities
            overlaps = any(
                not (end <= cs or idx >= ce)
                for cs, ce in consumed_spans
            )
            if not overlaps:
                node = graph.nodes.get(node_id)
                entity_name = node.entity.name if node else norm_name
                matches.append({
                    "mention": query[idx:end],
                    "node_id": node_id,
                    "entity_name": entity_name,
                    "span": (idx, end),
                })
                consumed_spans.append((idx, end))
            start = end

    return matches


# ---------------------------------------------------------------------------
# Retrieval result
# ---------------------------------------------------------------------------

@dataclass
class GraphRetrievalResult:
    """Result of graph-based retrieval for a single query."""
    # Matched entities in the query
    matched_entities: List[Dict[str, Any]] = field(default_factory=list)
    # Collected evidence sentences (provenance from triples)
    evidence_sentences: List[Dict[str, Any]] = field(default_factory=list)
    # Entity context sentence (for prompt-level subject anchoring)
    entity_context: str = ""
    # All triple_ids from graph walk (for KV injection filtering)
    walk_triple_ids: List[str] = field(default_factory=list)
    # Debug information
    debug: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Graph Retriever
# ---------------------------------------------------------------------------

class GraphRetriever:
    """
    Runtime graph-based retriever.

    Replaces the flat tag routing pipeline with entity-anchored
    graph traversal.

    Usage::

        retriever = GraphRetriever(graph=graph_index)
        result = retriever.retrieve(query="SFTSV的临床症状有哪些？", intent="symptom")
    """

    def __init__(
        self,
        *,
        graph: KnowledgeGraphIndex,
        relation_types: Optional[Dict[str, Dict[str, Any]]] = None,
        max_hops: int = 2,
        max_evidence: int = 10,
    ) -> None:
        self.graph = graph
        self.relation_types = relation_types or load_relation_types()
        self.intent_relation_map = _build_intent_relation_map(self.relation_types)
        self.max_hops = max(1, int(max_hops))
        self.max_evidence = max(1, int(max_evidence))

    def retrieve(
        self,
        query: str,
        *,
        intent: str = "",
        max_evidence: Optional[int] = None,
    ) -> GraphRetrievalResult:
        """
        Retrieve evidence via entity-anchored graph walk.

        Steps:
        1. Recognise entities in query
        2. Map intent to relation types
        3. Walk graph from each matched entity
        4. Collect provenance sentences
        5. Build entity context for prompt
        """
        max_ev = int(max_evidence or self.max_evidence)
        debug: Dict[str, Any] = {"query": query, "intent": intent}

        # 1. Entity recognition
        matches = recognise_entities(query, self.graph)
        debug["entity_matches"] = [
            {"mention": m["mention"], "node_id": m["node_id"], "entity_name": m["entity_name"]}
            for m in matches
        ]

        if not matches:
            debug["status"] = "no_entity_match"
            return GraphRetrievalResult(
                matched_entities=[],
                evidence_sentences=[],
                entity_context="",
                walk_triple_ids=[],
                debug=debug,
            )

        # 2. Map intent → relation types
        intent_lower = str(intent or "").strip().lower()
        target_relations: List[str] = []
        if intent_lower and intent_lower in self.intent_relation_map:
            target_relations = self.intent_relation_map[intent_lower]
        debug["target_relations"] = target_relations

        # 3. Graph walk from each matched entity
        walk_results: List[Dict[str, Any]] = []
        for m in matches:
            node_id = m["node_id"]
            walked = self.graph.walk(
                node_id,
                relation_types=target_relations,
                direction="outgoing",
                max_hops=self.max_hops,
            )
            walk_results.extend(walked)
            # Also walk incoming if no outgoing results
            if not walked:
                walked_in = self.graph.walk(
                    node_id,
                    relation_types=target_relations,
                    direction="incoming",
                    max_hops=self.max_hops,
                )
                walk_results.extend(walked_in)

        # If still empty and intent was specific, try without relation filter
        if not walk_results and target_relations:
            debug["fallback"] = "walk_without_relation_filter"
            for m in matches:
                walked_all = self.graph.walk(
                    m["node_id"],
                    relation_types=[],  # no filter
                    direction="outgoing",
                    max_hops=self.max_hops,
                )
                walk_results.extend(walked_all)

        debug["walk_results_count"] = len(walk_results)

        # 3b. Collect all triple_ids from walk (for KV injection filtering)
        walk_triple_ids: List[str] = []
        seen_tids: set = set()
        for wr in walk_results:
            tid = wr.get("triple_id", "")
            if tid and tid not in seen_tids:
                seen_tids.add(tid)
                walk_triple_ids.append(tid)

        # 4. Collect provenance sentences from triples
        seen_sentences: set = set()
        evidence: List[Dict[str, Any]] = []
        for wr in walk_results:
            tid = wr.get("triple_id", "")
            triple = self.graph.triples.get(tid)
            if not triple:
                continue
            prov = triple.provenance or {}
            sent_id = str(prov.get("sentence_id") or "")
            sent_text = str(prov.get("sentence_text") or "")
            if not sent_text or sent_id in seen_sentences:
                continue
            seen_sentences.add(sent_id)
            evidence.append({
                "sentence_id": sent_id,
                "text": sent_text,
                "source_block_id": str(prov.get("source_block_id") or ""),
                "relation": wr.get("relation", ""),
                "triple_id": tid,
                "hop": wr.get("hop", 1),
                "subject": triple.subject,
                "object": triple.object,
            })
            if len(evidence) >= max_ev:
                break

        debug["evidence_count"] = len(evidence)

        # 5. Build entity context sentence (prompt-level subject anchoring)
        entity_context = self._build_entity_context(matches)
        debug["entity_context"] = entity_context

        return GraphRetrievalResult(
            matched_entities=matches,
            evidence_sentences=evidence,
            entity_context=entity_context,
            walk_triple_ids=walk_triple_ids,
            debug=debug,
        )

    def _build_entity_context(self, matches: List[Dict[str, Any]]) -> str:
        """
        Build a prompt-level entity context sentence.

        v3: This is the *prompt channel* of subject anchoring.  It works
        alongside the KV channel (triple_kv_compiler's subject anchor KV)
        to provide complementary information — detailed description in
        prompt, short anchor signal in KV.
        """
        parts: List[str] = []
        for m in matches:
            node = self.graph.nodes.get(m["node_id"])
            if not node:
                continue
            ent = node.entity
            desc = ent.description.strip() if ent.description else ""
            aliases = [a for a in ent.aliases if a != ent.name]
            etype = ent.entity_type.strip() if ent.entity_type else ""
            alias_str = f"（{', '.join(aliases[:3])}）" if aliases else ""
            if desc:
                parts.append(f"{ent.name}{alias_str}：{desc}")
            elif etype:
                parts.append(f"{ent.name}{alias_str}，类型：{etype}")
            elif aliases:
                parts.append(f"{ent.name}（又称 {', '.join(aliases[:3])}）")
        if not parts:
            return ""
        return "### 实体背景\n" + "\n".join(f"- {p}" for p in parts)
