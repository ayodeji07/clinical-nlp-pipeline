"""
dashboard/views/explorer.py
────────────────────────────────────────────────────────────────
Entity Explorer page — frequency charts and co-occurrence network.

Shows:
  - Top-N entity frequency bar chart (filterable by type)
  - Interactive co-occurrence network graph (pyvis in an iframe)
  - Specialty distribution donut chart
  - Severity distribution breakdown
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import streamlit as st
import streamlit.components.v1 as components

from dashboard.api_client import (
    get_cooccurrence_pairs,
    get_stats,
    get_top_entities,
)

_LABEL_COLOURS: dict[str, str] = {
    "DISEASE":    "#e74c3c",
    "MEDICATION": "#3498db",
    "PROCEDURE":  "#2ecc71",
    "SYMPTOM":    "#f39c12",
    "ANATOMY":    "#9b59b6",
}

_SEVERITY_COLOURS: dict[str, str] = {
    "routine":  "#2ecc71",
    "urgent":   "#f39c12",
    "critical": "#e74c3c",
}


def render() -> None:
    """Render the Entity Explorer page."""

    st.title("📊 Entity Explorer")
    st.markdown(
        "Explore the most common clinical entities and how they "
        "relate to each other across the dataset."
    )

    # ── Sidebar controls ──────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Filter")
        entity_label = st.selectbox(
            "Entity type",
            options = ["All", "DISEASE", "MEDICATION", "PROCEDURE",
                       "SYMPTOM", "ANATOMY"],
        )
        top_n = st.slider("Top N entities", min_value=5, max_value=50, value=20)
        st.markdown("---")
        st.markdown("### Co-occurrence graph")
        min_cooccur = st.slider(
            "Min co-occurrence", min_value=2, max_value=20, value=5
        )
        graph_limit = st.slider(
            "Max edges", min_value=20, max_value=200, value=100
        )

    # ── Top entities chart ────────────────────────────────────────
    st.markdown("### Most Frequent Entities")

    label_filter = None if entity_label == "All" else entity_label
    entities     = get_top_entities(label=label_filter, limit=top_n)

    if entities:
        import plotly.graph_objects as go

        texts  = [e["text"] for e in entities]
        counts = [e["count"] for e in entities]
        colour = _LABEL_COLOURS.get(label_filter, "#3498db")

        fig = go.Figure(go.Bar(
            x            = counts,
            y            = texts,
            orientation  = "h",
            marker_color = colour,
            text         = counts,
            textposition = "outside",
        ))
        fig.update_layout(
            yaxis   = dict(autorange="reversed"),
            height  = max(300, top_n * 22),
            margin  = dict(l=10, r=40, t=10, b=10),
            plot_bgcolor = "white",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No entity data available. Run the ETL pipeline first.")

    # ── Co-occurrence network ─────────────────────────────────────
    st.markdown("### Co-occurrence Network")
    st.caption(
        "Nodes = entities · Edges = co-appear in the same note · "
        "Edge thickness = frequency"
    )

    with st.spinner("Building network graph..."):
        pairs = get_cooccurrence_pairs(
            label     = label_filter or "DISEASE",
            min_count = min_cooccur,
            limit     = graph_limit,
        )

    if pairs:
        html = _build_pyvis_html(pairs)
        if html:
            components.html(html, height=620, scrolling=False)
        else:
            _render_fallback_graph(pairs)
    else:
        st.info(
            "No co-occurrence pairs found. "
            "Try lowering the minimum co-occurrence threshold."
        )

    # ── Stats overview ────────────────────────────────────────────
    st.markdown("---")
    stats = get_stats()
    if not stats:
        st.warning("Could not load dataset statistics.")
        return

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Specialty Distribution")
        top_specs = stats.get("top_specialties", [])
        if top_specs:
            import plotly.express as px
            df_spec = {
                "specialty": [s["specialty"] for s in top_specs],
                "count":     [s["count"]     for s in top_specs],
            }
            fig_spec = px.pie(
                df_spec,
                names   = "specialty",
                values  = "count",
                hole    = 0.4,
                height  = 340,
            )
            fig_spec.update_traces(textposition="inside", textinfo="percent+label")
            fig_spec.update_layout(showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig_spec, use_container_width=True)

    with col_b:
        st.markdown("### Severity Distribution")
        sev = stats.get("severity_distribution", {})
        if sev:
            import plotly.graph_objects as go
            labels = ["routine", "urgent", "critical"]
            values = [sev.get(l, 0) for l in labels]
            colours_list = [_SEVERITY_COLOURS[l] for l in labels]
            fig_sev = go.Figure(go.Pie(
                labels    = labels,
                values    = values,
                hole      = 0.4,
                marker    = dict(colors=colours_list),
            ))
            fig_sev.update_traces(textposition="inside", textinfo="percent+label")
            fig_sev.update_layout(
                showlegend = False,
                height     = 340,
                margin     = dict(t=10, b=10),
            )
            st.plotly_chart(fig_sev, use_container_width=True)


def _build_pyvis_html(pairs: list[dict]) -> str | None:
    """Build an interactive pyvis network from co-occurrence pairs.

    Args:
        pairs: List of ``{"source", "target", "weight"}`` dicts.

    Returns:
        HTML string for the network, or None if pyvis is unavailable.
    """
    try:
        import networkx as nx
        from pyvis.network import Network
    except ImportError:
        return None

    # Build NetworkX graph from the pairs
    graph = nx.Graph()
    for pair in pairs:
        graph.add_edge(
            pair["source"],
            pair["target"],
            weight = pair["weight"],
        )

    # Compute degree centrality for node sizing
    centrality = nx.degree_centrality(graph)

    net = Network(height="600px", bgcolor="#ffffff", font_color="#333333")

    for node in graph.nodes():
        size = max(10, min(40, int(centrality[node] * 200)))
        net.add_node(
            node,
            label = node,
            size  = size,
            title = f"{node}<br>connections: {graph.degree(node)}",
        )

    for src, tgt, attrs in graph.edges(data=True):
        w = attrs.get("weight", 1)
        net.add_edge(
            src, tgt,
            width = max(1, min(8, w // 2)),
            title = f"co-occurs in {w} notes",
        )

    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.005,
          "springLength": 120,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "minVelocity": 0.75
      }
    }
    """)

    return net.generate_html()


def _render_fallback_graph(pairs: list[dict]) -> None:
    """Render a simple Plotly scatter graph when pyvis is unavailable.

    A basic spring-layout visualisation using networkx + plotly.
    Not as interactive as pyvis but works without pyvis installed.
    """
    try:
        import networkx as nx
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Install networkx and plotly for graph visualisation.")
        return

    graph = nx.Graph()
    for pair in pairs:
        graph.add_edge(pair["source"], pair["target"],
                       weight=pair["weight"])

    pos = nx.spring_layout(graph, seed=42)

    edge_x, edge_y = [], []
    for src, tgt in graph.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = [pos[n][0] for n in graph.nodes()]
    node_y = [pos[n][1] for n in graph.nodes()]
    node_labels = list(graph.nodes())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="#aaa"),
        hoverinfo="none",
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_labels,
        textposition="top center",
        marker=dict(size=10, color="#3498db"),
        hoverinfo="text",
    ))
    fig.update_layout(
        showlegend=False, height=500,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(t=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
