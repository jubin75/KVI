#!/usr/bin/env python3
"""
Scheme C — CLI: Build knowledge graph index from extracted triples.

Usage::

    python scripts/build_knowledge_graph.py \\
        --triples_jsonl /path/to/triples.jsonl \\
        --out_graph /path/to/graph_index.json \\
        [--aliases_jsonl /path/to/aliases.jsonl]

Input:  triples.jsonl       (output of extract_triples.py)
Output: graph_index.json    (KnowledgeGraphIndex, loadable at runtime)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    p = argparse.ArgumentParser(description="Build knowledge graph from triples")
    p.add_argument("--triples_jsonl", required=True, help="Input triples JSONL file")
    p.add_argument("--out_graph", required=True, help="Output graph index JSON file")
    p.add_argument("--aliases_jsonl", default="", help="Entity aliases JSONL (canonical + aliases)")
    p.add_argument("--relation_types_json", default="", help="Custom relation types JSON")
    p.add_argument("--entity_types_json", default="", help="Custom entity types JSON")
    args = p.parse_args()

    from src.graph.knowledge_graph import build_graph_from_triples_jsonl

    triples_path = Path(args.triples_jsonl)
    if not triples_path.exists():
        print(f"ERROR: triples file not found: {triples_path}", file=sys.stderr)
        sys.exit(1)

    aliases_path = Path(args.aliases_jsonl) if args.aliases_jsonl else None
    rel_path = Path(args.relation_types_json) if args.relation_types_json else None
    ent_path = Path(args.entity_types_json) if args.entity_types_json else None

    print(f"Building graph from {triples_path}...", file=sys.stderr)
    graph = build_graph_from_triples_jsonl(
        triples_path,
        aliases_path=aliases_path,
        relation_types_path=rel_path,
        entity_types_path=ent_path,
    )

    out_path = Path(args.out_graph)
    graph.save(out_path)

    # Summary
    print(f"\n=== Knowledge Graph Summary ===", file=sys.stderr)
    print(f"  Nodes:              {graph.meta.get('num_nodes', 0)}", file=sys.stderr)
    print(f"  Triples:            {graph.meta.get('num_triples', 0)}", file=sys.stderr)
    print(f"  Entity index size:  {graph.meta.get('num_entity_index_entries', 0)}", file=sys.stderr)
    print(f"  Saved to:           {out_path}", file=sys.stderr)

    # List top entities by edge count
    print(f"\n  Top entities (by edge count):", file=sys.stderr)
    node_edge_counts = []
    for nid, node in graph.nodes.items():
        out_count = sum(len(v) for v in node.outgoing.values())
        in_count = sum(len(v) for v in node.incoming.values())
        node_edge_counts.append((node.entity.name, node.entity.entity_type, out_count + in_count))
    node_edge_counts.sort(key=lambda x: -x[2])
    for name, etype, count in node_edge_counts[:15]:
        print(f"    {name} ({etype}): {count} edges", file=sys.stderr)


if __name__ == "__main__":
    main()
