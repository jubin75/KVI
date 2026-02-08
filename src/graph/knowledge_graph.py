"""
Scheme C — Knowledge Graph Builder.

Takes extracted triples and builds a traversable knowledge graph index.
The graph is:
* **Compile-time artefact** — built once from triples.jsonl, persisted as JSON.
* **Entity-indexed** — fast O(1) lookup from entity name/alias to node.
* **Relation-typed edges** — each edge carries its relation type, enabling
  intent-directed graph walk at query time.
* **Provenance-linked** — every edge traces back to the original evidence
  sentence, so the retriever can collect provenance for RAG prompt.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .schema import (
    Entity,
    GraphNode,
    KnowledgeGraphIndex,
    Triple,
    _normalise,
    load_entity_types,
    load_relation_types,
)


class KnowledgeGraphBuilder:
    """
    Build a :class:`KnowledgeGraphIndex` from a collection of triples.

    Usage::

        builder = KnowledgeGraphBuilder()
        for triple in triples:
            builder.add_triple(triple)
        graph = builder.build()
        graph.save(Path("graph_index.json"))
    """

    def __init__(
        self,
        *,
        relation_types: Optional[Dict[str, Dict[str, Any]]] = None,
        entity_types: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.relation_types = relation_types or load_relation_types()
        self.entity_types = entity_types or load_entity_types()
        self._triples: Dict[str, Triple] = {}
        # entity canonical name → Entity
        self._entities: Dict[str, Entity] = {}
        # Normalised name/alias → canonical name
        self._name_index: Dict[str, str] = {}

    def add_triple(self, triple: Triple) -> None:
        """Add a triple to the builder."""
        self._triples[triple.triple_id] = triple
        # Register entities
        self._register_entity(triple.subject, triple.subject_type)
        self._register_entity(triple.object, triple.object_type)

    def add_entity_alias(self, canonical_name: str, alias: str) -> None:
        """Register an alias for an entity."""
        canon = canonical_name.strip()
        alias = alias.strip()
        if not canon or not alias:
            return
        ent = self._entities.get(canon)
        if ent and alias not in ent.aliases:
            ent.aliases.append(alias)
        self._name_index[_normalise(alias)] = canon

    def build(self) -> KnowledgeGraphIndex:
        """Build the knowledge graph index from accumulated triples."""
        # 1. Create nodes from entities
        nodes: Dict[str, GraphNode] = {}
        canon_to_node_id: Dict[str, str] = {}
        for idx, (canon, ent) in enumerate(sorted(self._entities.items())):
            nid = f"node_{idx:04d}"
            nodes[nid] = GraphNode(node_id=nid, entity=ent)
            canon_to_node_id[canon] = nid

        # 2. Wire edges from triples
        for tid, triple in self._triples.items():
            src_nid = canon_to_node_id.get(triple.subject)
            tgt_nid = canon_to_node_id.get(triple.object)
            if not src_nid or not tgt_nid:
                continue
            rel = triple.predicate
            edge_payload = {"target_node_id": tgt_nid, "triple_id": tid}
            edge_incoming = {"target_node_id": src_nid, "triple_id": tid}

            # Outgoing from subject
            nodes[src_nid].outgoing.setdefault(rel, []).append(edge_payload)
            # Incoming to object
            nodes[tgt_nid].incoming.setdefault(rel, []).append(edge_incoming)

            # Also add inverse relation if defined
            inv_rel = (self.relation_types.get(rel) or {}).get("inverse")
            if inv_rel and inv_rel != rel:
                inv_edge_out = {"target_node_id": src_nid, "triple_id": tid}
                inv_edge_in = {"target_node_id": tgt_nid, "triple_id": tid}
                nodes[tgt_nid].outgoing.setdefault(inv_rel, []).append(inv_edge_out)
                nodes[src_nid].incoming.setdefault(inv_rel, []).append(inv_edge_in)

        # 3. Build entity index (normalised name/alias → node_id)
        entity_index: Dict[str, str] = {}
        for canon, nid in canon_to_node_id.items():
            entity_index[_normalise(canon)] = nid
            ent = self._entities.get(canon)
            if ent:
                for alias in ent.aliases:
                    entity_index[_normalise(alias)] = nid

        # 4. Deduplicate edges (same target + triple_id)
        for node in nodes.values():
            node.outgoing = _dedup_edges(node.outgoing)
            node.incoming = _dedup_edges(node.incoming)

        # 5. Build metadata
        meta = {
            "num_nodes": len(nodes),
            "num_triples": len(self._triples),
            "num_entity_index_entries": len(entity_index),
            "relation_types": sorted(self.relation_types.keys()),
            "entity_types": sorted(self.entity_types.keys()),
        }

        return KnowledgeGraphIndex(
            nodes=nodes,
            triples=self._triples,
            entity_index=entity_index,
            meta=meta,
        )

    # -- Internal --

    def _register_entity(self, name: str, entity_type: str = "") -> None:
        name = name.strip()
        if not name:
            return
        if name not in self._entities:
            self._entities[name] = Entity(
                name=name,
                entity_type=entity_type.strip(),
            )
        elif entity_type.strip() and not self._entities[name].entity_type:
            self._entities[name].entity_type = entity_type.strip()
        self._name_index[_normalise(name)] = name


# ---------------------------------------------------------------------------
# Build from files (CLI-friendly)
# ---------------------------------------------------------------------------

def build_graph_from_triples_jsonl(
    triples_path: Path,
    *,
    aliases_path: Optional[Path] = None,
    relation_types_path: Optional[Path] = None,
    entity_types_path: Optional[Path] = None,
) -> KnowledgeGraphIndex:
    """
    Build a knowledge graph from a ``triples.jsonl`` file.

    Optionally load:
    * ``aliases.jsonl`` — entity alias mappings
    * ``relation_types.json`` / ``entity_types.json`` — custom taxonomies
    """
    rel_types = load_relation_types(relation_types_path)
    ent_types = load_entity_types(entity_types_path)
    builder = KnowledgeGraphBuilder(relation_types=rel_types, entity_types=ent_types)

    # Load triples
    with triples_path.open("r", encoding="utf-8") as f:
        for line in f:
            triple = Triple.from_jsonl_line(line)
            if triple:
                builder.add_triple(triple)

    # Load aliases
    if aliases_path and aliases_path.exists():
        with aliases_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                    canon = str(rec.get("canonical") or rec.get("name") or "").strip()
                    for alias in (rec.get("aliases") or []):
                        builder.add_entity_alias(canon, str(alias))
                except Exception:
                    continue

    return builder.build()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedup_edges(
    edges_by_rel: Dict[str, List[Dict[str, str]]]
) -> Dict[str, List[Dict[str, str]]]:
    """Remove duplicate edges within each relation type."""
    out: Dict[str, List[Dict[str, str]]] = {}
    for rel, edges in edges_by_rel.items():
        seen: set = set()
        deduped: List[Dict[str, str]] = []
        for e in edges:
            key = (e.get("target_node_id", ""), e.get("triple_id", ""))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(e)
        out[rel] = deduped
    return out
