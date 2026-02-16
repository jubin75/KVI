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

    def enrich_entity(self, name: str, *, description: str = "", entity_type: str = "") -> None:
        """Enrich an entity with description and/or entity_type from aliases config."""
        name = name.strip()
        if not name:
            return
        ent = self._entities.get(name)
        if ent:
            if description and not ent.description:
                ent.description = description
            if entity_type and not ent.entity_type:
                ent.entity_type = entity_type
        else:
            # Entity not yet registered — register it
            self._register_entity(name, entity_type)
            ent = self._entities.get(name)
            if ent and description:
                ent.description = description

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
        # 0. Merge aliased entities: if an entity name maps to a different
        #    canonical name via _name_index, merge it.
        self._merge_aliased_entities()

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
        sentence_index: Dict[str, Dict[str, Any]] = {}
        triple_sentence_index: Dict[str, List[str]] = {}
        for tid, triple in self._triples.items():
            prov = triple.provenance or {}
            sid = str(prov.get("sentence_id") or "").strip()
            if not sid:
                continue
            triple_sentence_index[tid] = [sid]
            if sid not in sentence_index:
                sentence_index[sid] = {
                    "text": str(prov.get("sentence_text") or ""),
                    "source_block_id": str(prov.get("source_block_id") or ""),
                    "source_doc_id": str(prov.get("source_doc_id") or ""),
                    "triple_ids": [],
                }
            sentence_index[sid]["triple_ids"].append(tid)

        for sid, rec in sentence_index.items():
            rec["triple_ids"] = sorted(set(rec.get("triple_ids") or []))

        meta = {
            "num_nodes": len(nodes),
            "num_triples": len(self._triples),
            "num_entity_index_entries": len(entity_index),
            "num_sentences_indexed": len(sentence_index),
            "relation_types": sorted(self.relation_types.keys()),
            "entity_types": sorted(self.entity_types.keys()),
        }

        return KnowledgeGraphIndex(
            nodes=nodes,
            triples=self._triples,
            entity_index=entity_index,
            sentence_index=sentence_index,
            triple_sentence_index=triple_sentence_index,
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

    def _merge_aliased_entities(self) -> None:
        """
        Merge entities that are aliases of a canonical entity.

        After aliases are registered, ``_name_index`` may map a normalised
        entity name to a *different* canonical name.  This method:
        1. Identifies entities whose normalised name resolves to a different
           canonical entity via ``_name_index``.
        2. Merges their aliases and entity_type into the canonical entity.
        3. Rewrites all triples to use the canonical name.
        4. Removes the alias entity from ``_entities``.
        """
        # Build merge map: alias_name → canonical_name
        merge_map: Dict[str, str] = {}
        for ent_name in list(self._entities.keys()):
            norm = _normalise(ent_name)
            canonical = self._name_index.get(norm, ent_name)
            if canonical != ent_name:
                merge_map[ent_name] = canonical

        if not merge_map:
            return

        # Merge entity metadata
        for alias_name, canon_name in merge_map.items():
            alias_ent = self._entities.get(alias_name)
            canon_ent = self._entities.get(canon_name)
            if not alias_ent or not canon_ent:
                continue
            # Merge aliases
            if alias_name not in canon_ent.aliases:
                canon_ent.aliases.append(alias_name)
            for a in alias_ent.aliases:
                if a not in canon_ent.aliases and a != canon_name:
                    canon_ent.aliases.append(a)
            # Merge entity_type (prefer canonical's type)
            if not canon_ent.entity_type and alias_ent.entity_type:
                canon_ent.entity_type = alias_ent.entity_type
            # Merge description
            if not canon_ent.description and alias_ent.description:
                canon_ent.description = alias_ent.description
            # Remove alias entity
            del self._entities[alias_name]

        # Rewrite triples to use canonical names
        for triple in self._triples.values():
            if triple.subject in merge_map:
                triple.subject = merge_map[triple.subject]
            if triple.object in merge_map:
                triple.object = merge_map[triple.object]


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

    # Load aliases (with optional description and entity_type)
    if aliases_path and aliases_path.exists():
        with aliases_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                    canon = str(rec.get("canonical") or rec.get("name") or "").strip()
                    if not canon:
                        continue
                    for alias in (rec.get("aliases") or []):
                        builder.add_entity_alias(canon, str(alias))
                    # Load description and entity_type if provided
                    desc = str(rec.get("description") or "").strip()
                    etype = str(rec.get("entity_type") or "").strip()
                    if desc or etype:
                        builder.enrich_entity(canon, description=desc, entity_type=etype)
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
