"""
Scheme C — Triple & Graph data schema.

Core data model for the knowledge graph.  Every piece of domain knowledge
is decomposed into triples ``(subject, predicate, object)`` with typed
entities and relations.  The graph is built from these triples at compile
time and traversed at query time for entity-anchored retrieval.

Design decisions (grounded in empirical findings):
* Relations guide **retrieval routing**, NOT attention-head routing.
  Our experiments proved that KV prefix injection degrades RAG quality
  when evidence is already in the prompt.  Relations select *which*
  evidence enters the prompt, not *how* the model attends.
* Subject anchoring is prompt-level (entity context sentence prepended),
  NOT KV-injection-level.  This avoids the token corruption observed
  with entity priming KV injection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Entity types  (configurable via entity_types.json)
# ---------------------------------------------------------------------------

# Default entity type taxonomy for medical domain.
# Can be overridden per-topic by placing entity_types.json in sidecar.
DEFAULT_ENTITY_TYPES: Dict[str, Dict[str, Any]] = {
    "pathogen":    {"aliases": ["virus", "bacterium", "pathogen", "病原体", "病毒", "细菌"],
                    "description": "Disease-causing agent (virus, bacterium, etc.)"},
    "disease":     {"aliases": ["disease", "syndrome", "condition", "疾病", "综合征"],
                    "description": "A disease, syndrome, or clinical condition"},
    "symptom":     {"aliases": ["symptom", "sign", "manifestation", "症状", "体征", "表现"],
                    "description": "Clinical symptom or sign"},
    "drug":        {"aliases": ["drug", "treatment", "therapy", "药物", "治疗", "疗法"],
                    "description": "Pharmacological agent or treatment approach"},
    "mechanism":   {"aliases": ["mechanism", "pathway", "process", "机制", "通路", "过程"],
                    "description": "Biological mechanism or pathway"},
    "anatomy":     {"aliases": ["organ", "tissue", "cell", "location", "器官", "组织", "细胞"],
                    "description": "Anatomical structure, organ, tissue, or cell type"},
    "lab_value":   {"aliases": ["lab", "measurement", "biomarker", "指标", "检测", "生物标志物"],
                    "description": "Laboratory measurement or biomarker"},
    "epidemiology": {"aliases": ["epidemiology", "prevalence", "incidence", "流行病学"],
                     "description": "Epidemiological data (prevalence, incidence, distribution)"},
}


# ---------------------------------------------------------------------------
# Relation types  (configurable via relation_types.json)
# ---------------------------------------------------------------------------

# Each relation type has:
#   - description: natural-language definition (used as embedding anchor)
#   - inverse: the inverse relation name (for bidirectional graph walk)
#   - retrieval_intents: which query intents this relation serves
#     (maps to the existing intent routing vocabulary)

DEFAULT_RELATION_TYPES: Dict[str, Dict[str, Any]] = {
    "causes": {
        "description": "Subject causes or leads to object (causal relationship)",
        "inverse": "caused_by",
        "retrieval_intents": ["symptom", "mechanism", "pathogenesis"],
    },
    "caused_by": {
        "description": "Subject is caused by object",
        "inverse": "causes",
        "retrieval_intents": ["etiology", "mechanism"],
    },
    "treats": {
        "description": "Subject treats or alleviates object",
        "inverse": "treated_by",
        "retrieval_intents": ["drug", "treatment", "therapy"],
    },
    "treated_by": {
        "description": "Subject is treated by object",
        "inverse": "treats",
        "retrieval_intents": ["drug", "treatment"],
    },
    "manifests_as": {
        "description": "Subject manifests as or presents with object (clinical presentation)",
        "inverse": "manifestation_of",
        "retrieval_intents": ["symptom", "clinical"],
    },
    "manifestation_of": {
        "description": "Subject is a manifestation of object",
        "inverse": "manifests_as",
        "retrieval_intents": ["symptom", "clinical"],
    },
    "is_a": {
        "description": "Subject is a type/subclass of object (taxonomy)",
        "inverse": "has_subtype",
        "retrieval_intents": ["definition", "identity", "classification"],
    },
    "has_subtype": {
        "description": "Subject has object as a subtype",
        "inverse": "is_a",
        "retrieval_intents": ["definition", "classification"],
    },
    "part_of": {
        "description": "Subject is a component or part of object",
        "inverse": "has_part",
        "retrieval_intents": ["anatomy", "structure"],
    },
    "has_part": {
        "description": "Subject contains or has object as a component",
        "inverse": "part_of",
        "retrieval_intents": ["anatomy", "structure"],
    },
    "located_in": {
        "description": "Subject is located in or found in object (spatial/anatomical)",
        "inverse": "location_of",
        "retrieval_intents": ["location", "distribution", "epidemiology"],
    },
    "location_of": {
        "description": "Subject is the location/site of object",
        "inverse": "located_in",
        "retrieval_intents": ["location"],
    },
    "inhibits": {
        "description": "Subject inhibits or suppresses object",
        "inverse": "inhibited_by",
        "retrieval_intents": ["mechanism", "drug"],
    },
    "inhibited_by": {
        "description": "Subject is inhibited by object",
        "inverse": "inhibits",
        "retrieval_intents": ["mechanism", "drug"],
    },
    "associated_with": {
        "description": "Subject is associated with object (general association)",
        "inverse": "associated_with",
        "retrieval_intents": [],  # generic — no specific intent preference
    },
    "transmits_via": {
        "description": "Subject is transmitted via object (transmission route)",
        "inverse": "transmission_route_for",
        "retrieval_intents": ["transmission", "epidemiology", "prevention"],
    },
    "transmission_route_for": {
        "description": "Subject is the transmission route for object",
        "inverse": "transmits_via",
        "retrieval_intents": ["transmission", "epidemiology"],
    },
    "prevents": {
        "description": "Subject prevents object",
        "inverse": "prevented_by",
        "retrieval_intents": ["prevention", "treatment"],
    },
    "prevented_by": {
        "description": "Subject is prevented by object",
        "inverse": "prevents",
        "retrieval_intents": ["prevention"],
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A node in the knowledge graph."""
    name: str                           # canonical name (e.g., "SFTSV")
    entity_type: str = ""               # key in entity_types (e.g., "pathogen")
    aliases: List[str] = field(default_factory=list)  # alternative names / abbreviations
    description: str = ""               # one-line definition (optional)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Entity":
        return cls(
            name=str(d.get("name") or ""),
            entity_type=str(d.get("entity_type") or ""),
            aliases=list(d.get("aliases") or []),
            description=str(d.get("description") or ""),
        )


@dataclass
class Triple:
    """
    A single knowledge triple  (subject, predicate, object)
    with provenance back to the original evidence sentence.
    """
    triple_id: str
    subject: str                        # entity name (canonical)
    subject_type: str = ""              # entity type key
    predicate: str = ""                 # relation type key
    object: str = ""                    # entity name (canonical)
    object_type: str = ""               # entity type key
    confidence: float = 1.0             # extraction confidence [0, 1]
    provenance: Dict[str, Any] = field(default_factory=dict)
    # provenance = {"sentence_id": "...", "sentence_text": "...", "source_block_id": "..."}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Triple":
        return cls(
            triple_id=str(d.get("triple_id") or ""),
            subject=str(d.get("subject") or ""),
            subject_type=str(d.get("subject_type") or ""),
            predicate=str(d.get("predicate") or ""),
            object=str(d.get("object") or ""),
            object_type=str(d.get("object_type") or ""),
            confidence=float(d.get("confidence") or 1.0),
            provenance=dict(d.get("provenance") or {}),
        )

    @classmethod
    def from_jsonl_line(cls, line: str) -> Optional["Triple"]:
        line = line.strip()
        if not line:
            return None
        try:
            return cls.from_dict(json.loads(line))
        except Exception:
            return None


@dataclass
class GraphNode:
    """A node in the built knowledge graph (enriched entity)."""
    node_id: str                        # unique, e.g., "node_sftsv"
    entity: Entity
    # Edges grouped by relation type → list of (target_node_id, triple_id)
    outgoing: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    incoming: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "entity": self.entity.to_dict(),
            "outgoing": self.outgoing,
            "incoming": self.incoming,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphNode":
        return cls(
            node_id=str(d.get("node_id") or ""),
            entity=Entity.from_dict(d.get("entity") or {}),
            outgoing=dict(d.get("outgoing") or {}),
            incoming=dict(d.get("incoming") or {}),
        )


@dataclass
class KnowledgeGraphIndex:
    """
    Serializable knowledge graph index.

    Stored as a single JSON file at compile time, loaded at query time
    for entity recognition and graph walk.
    """
    nodes: Dict[str, GraphNode] = field(default_factory=dict)  # node_id → GraphNode
    triples: Dict[str, Triple] = field(default_factory=dict)   # triple_id → Triple
    # Fast lookup: normalised entity name/alias → node_id
    entity_index: Dict[str, str] = field(default_factory=dict)
    # Metadata
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- Persistence --

    def save(self, path: Path) -> None:
        """Save graph index to JSON file."""
        data = {
            "meta": self.meta,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "triples": {tid: t.to_dict() for tid, t in self.triples.items()},
            "entity_index": self.entity_index,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraphIndex":
        """Load graph index from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        nodes = {nid: GraphNode.from_dict(nd) for nid, nd in (data.get("nodes") or {}).items()}
        triples = {tid: Triple.from_dict(td) for tid, td in (data.get("triples") or {}).items()}
        return cls(
            nodes=nodes,
            triples=triples,
            entity_index=dict(data.get("entity_index") or {}),
            meta=dict(data.get("meta") or {}),
        )

    # -- Query helpers --

    def resolve_entity(self, text: str) -> Optional[str]:
        """Resolve a text mention to a node_id via the entity index."""
        key = _normalise(text)
        return self.entity_index.get(key)

    def walk(
        self,
        node_id: str,
        relation_types: Sequence[str] = (),
        direction: str = "outgoing",
        max_hops: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Walk the graph from *node_id* along edges of the given relation types.

        Returns a list of dicts:
            {"node_id": ..., "relation": ..., "triple_id": ..., "hop": ...}
        """
        if node_id not in self.nodes:
            return []
        results: List[Dict[str, Any]] = []
        visited: set = {node_id}
        frontier = [(node_id, 0)]
        while frontier:
            cur_id, hop = frontier.pop(0)
            if hop >= max_hops:
                continue
            node = self.nodes.get(cur_id)
            if not node:
                continue
            edges = node.outgoing if direction == "outgoing" else node.incoming
            for rel, targets in edges.items():
                if relation_types and rel not in relation_types:
                    continue
                for tgt in targets:
                    tgt_id = str(tgt.get("target_node_id") or "")
                    tid = str(tgt.get("triple_id") or "")
                    if not tgt_id or tgt_id in visited:
                        continue
                    visited.add(tgt_id)
                    results.append({
                        "node_id": tgt_id,
                        "relation": rel,
                        "triple_id": tid,
                        "hop": hop + 1,
                    })
                    frontier.append((tgt_id, hop + 1))
        return results


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def load_entity_types(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Load entity type taxonomy.  Falls back to DEFAULT_ENTITY_TYPES."""
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return dict(DEFAULT_ENTITY_TYPES)


def load_relation_types(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Load relation type taxonomy.  Falls back to DEFAULT_RELATION_TYPES."""
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return dict(DEFAULT_RELATION_TYPES)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Normalise entity name for index lookup."""
    import re
    t = str(text or "").strip()
    # lowercase, collapse whitespace, strip punctuation
    t = re.sub(r"\s+", " ", t).lower().strip()
    return t
