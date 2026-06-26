"""
src/nlp/cooccurrence.py
────────────────────────────────────────────────────────────────
Entity co-occurrence graph builder.

Two entities "co-occur" when they appear in the same clinical note.
Frequent co-occurrence suggests a clinical relationship — for
example, "hypertension" and "diabetes mellitus" appear together
often because they are common comorbidities.

This module builds a weighted undirected graph where:
  - Nodes  = unique entity texts
  - Edges  = pairs that co-occur in at least `min_count` notes
  - Weight = number of notes in which the pair co-occurs

The graph is returned as a NetworkX object so callers can
apply any NetworkX algorithm (centrality, community detection,
shortest path, etc.) without this module needing to know about
the downstream use.

The Streamlit dashboard uses pyvis to render an interactive
HTML visualisation of the graph.
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_cooccurrence_graph(
    note_entities: dict[int, list],
    entity_label:  str = "DISEASE",
    min_count:     int = 3,
    max_nodes:     int = 50,
):
    """Build a co-occurrence graph from per-note entity lists.

    Args:
        note_entities : Dict mapping note_id → list of Entity objects.
                        Typically from the NER pipeline output.
        entity_label  : Only include entities with this label.
                        Defaults to ``"DISEASE"`` — disease-disease
                        co-occurrences are the most clinically meaningful.
        min_count     : Minimum number of notes a pair must co-occur
                        in to be included as an edge.  Lower values
                        create denser, noisier graphs.
        max_nodes     : Limit the graph to this many nodes (highest
                        degree nodes are kept) to keep the
                        visualisation readable.

    Returns:
        A :class:`networkx.Graph` with ``weight`` edge attributes
        and ``count`` node attributes (number of notes the entity
        appears in).  Returns an empty graph if networkx is not
        installed or no pairs are found.

    Example::

        graph = build_cooccurrence_graph(note_entities)
        print(graph.number_of_nodes(), graph.number_of_edges())
        # → 38  91
    """
    try:
        import networkx as nx
    except ImportError:
        logger.error(
            "networkx not installed. Run: pip install networkx"
        )
        import networkx as nx   # re-raise the real error
        return nx.Graph()

    # ── Step 1: count pairwise co-occurrences ─────────────────────
    # For each note, collect the unique entity texts with the
    # target label, then count every pair.
    pair_counts:  defaultdict[tuple[str, str], int] = defaultdict(int)
    entity_counts: defaultdict[str, int]             = defaultdict(int)

    for note_id, entities in note_entities.items():
        # Deduplicate entities within this note — we care about
        # co-occurrence, not how many times each entity appears.
        relevant = list({
            ent.text.lower()
            for ent in entities
            if ent.label.upper() == entity_label.upper()
            and len(ent.text.strip()) >= 3
        })

        for text in relevant:
            entity_counts[text] += 1

        # Generate all unique pairs (order-independent)
        for i in range(len(relevant)):
            for j in range(i + 1, len(relevant)):
                pair = tuple(sorted([relevant[i], relevant[j]]))
                pair_counts[pair] += 1

    if not pair_counts:
        logger.warning(
            "No co-occurrence pairs found for label='%s'. "
            "Check that your NER pipeline has run and produced entities.",
            entity_label,
        )
        return nx.Graph()

    # ── Step 2: build the NetworkX graph ──────────────────────────
    graph = nx.Graph()

    # Add nodes with their document frequency as an attribute
    for entity, count in entity_counts.items():
        graph.add_node(entity, count=count)

    # Add edges that meet the minimum co-occurrence threshold
    edges_added = 0
    for (entity_a, entity_b), count in pair_counts.items():
        if count >= min_count:
            graph.add_edge(entity_a, entity_b, weight=count)
            edges_added += 1

    logger.info(
        "Co-occurrence graph: %d nodes, %d edges (min_count=%d)",
        graph.number_of_nodes(), edges_added, min_count,
    )

    # ── Step 3: prune to max_nodes if needed ──────────────────────
    # Keep the nodes with the highest degree (most connections).
    # This produces a more coherent visualisation than random pruning.
    if graph.number_of_nodes() > max_nodes:
        # Sort by degree descending and keep the top max_nodes
        top_nodes = sorted(
            graph.nodes(), key=lambda n: graph.degree(n), reverse=True
        )[:max_nodes]
        graph = graph.subgraph(top_nodes).copy()
        logger.info(
            "Graph pruned to %d nodes (max_nodes=%d)",
            graph.number_of_nodes(), max_nodes,
        )

    return graph


def graph_to_pyvis(
    graph,
    height:    str = "600px",
    bgcolor:   str = "#ffffff",
    font_color: str = "#000000",
) -> Optional[str]:
    """Convert a NetworkX graph to an interactive pyvis HTML string.

    Node size scales with document frequency (how often the entity
    appears across the dataset).  Edge thickness scales with
    co-occurrence count.

    Args:
        graph      : NetworkX Graph from :func:`build_cooccurrence_graph`.
        height     : Height of the HTML iframe (CSS string).
        bgcolor    : Background colour hex code.
        font_color : Node label font colour.

    Returns:
        HTML string suitable for embedding in Streamlit via
        ``st.components.v1.html()``, or None if pyvis is not installed.

    Example::

        html = graph_to_pyvis(graph)
        if html:
            st.components.v1.html(html, height=620)
    """
    try:
        from pyvis.network import Network
    except ImportError:
        logger.warning(
            "pyvis not installed — falling back to static chart. "
            "Run: pip install pyvis"
        )
        return None

    net = Network(
        height     = height,
        bgcolor    = bgcolor,
        font_color = font_color,
        notebook   = False,
    )

    # Add nodes — size proportional to document frequency
    for node, attrs in graph.nodes(data=True):
        count     = attrs.get("count", 1)
        node_size = max(10, min(50, count * 2))   # clamp 10–50
        net.add_node(
            node,
            label = node,
            size  = node_size,
            title = f"{node}<br>appears in {count} notes",
        )

    # Add edges — width proportional to co-occurrence weight
    for source, target, attrs in graph.edges(data=True):
        weight     = attrs.get("weight", 1)
        edge_width = max(1, min(10, weight // 2))   # clamp 1–10
        net.add_edge(
            source, target,
            width = edge_width,
            title = f"co-occurs in {weight} notes",
        )

    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100
        },
        "solver": "forceAtlas2Based",
        "minVelocity": 0.75
      }
    }
    """)

    return net.generate_html()


def graph_summary(graph) -> dict:
    """Return summary statistics for a co-occurrence graph.

    Useful for the dashboard metrics panel.

    Args:
        graph: NetworkX Graph.

    Returns:
        Dict with ``nodes``, ``edges``, ``density``,
        ``top_entities`` (list of (name, degree) tuples).

    Example::

        summary = graph_summary(graph)
        print(summary["top_entities"][:3])
        # → [("hypertension", 28), ("diabetes", 22), ("pain", 19)]
    """
    try:
        import networkx as nx
    except ImportError:
        return {}

    if graph.number_of_nodes() == 0:
        return {
            "nodes": 0, "edges": 0,
            "density": 0.0, "top_entities": [],
        }

    top_entities = sorted(
        graph.degree(), key=lambda x: x[1], reverse=True
    )[:10]

    return {
        "nodes":        graph.number_of_nodes(),
        "edges":        graph.number_of_edges(),
        "density":      round(nx.density(graph), 4),
        "top_entities": [(name, deg) for name, deg in top_entities],
    }
