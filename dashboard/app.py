"""
dashboard/app.py
────────────────────────────────────────────────────────────────
Streamlit application entry point.

Run locally:
    streamlit run dashboard/app.py

Deploy to Streamlit Cloud:
    1. Push this repo to GitHub
    2. Go to share.streamlit.io → New app
    3. Set main file path: dashboard/app.py
    4. Add secrets (see VSCODE_GUIDE.md for full instructions):
         API_BASE_URL = "https://your-api-host.com"
         DATABASE_URL = "postgresql://..."

Navigation
──────────
  🔬 Demo       — paste clinical text, get real-time analysis
  📊 Explorer   — entity frequency charts and co-occurrence graph
  🧠 Metrics    — classifier performance and training history
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this script directly (`streamlit run dashboard/app.py`),
# which puts this file's own directory (dashboard/) on sys.path -- not the
# project root. That makes `dashboard` itself unimportable as a package
# from within its own folder. Insert the project root (this file's parent)
# so absolute imports like `from dashboard.api_client import ...` resolve,
# same pattern the notebooks use (`sys.path.insert(0, '..')`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from dashboard.api_client import check_health
from dashboard.views import demo, explorer, model_metrics

# ── Page config ───────────────────────────────────────────────────
# Must be the first Streamlit call in the script.
st.set_page_config(
    page_title     = "Clinical NLP Pipeline",
    page_icon      = "🔬",
    layout         = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
# Minimal styling — keeps the app looking clean without overriding
# Streamlit's built-in theme.
st.markdown("""
<style>
  /* Tighten the sidebar width slightly */
  [data-testid="stSidebar"] { min-width: 220px; max-width: 280px; }

  /* Subtle card background for metric containers */
  [data-testid="metric-container"] {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    padding: 12px 16px;
  }

  /* Reduce top padding on the main content area */
  .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar navigation ────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 Clinical NLP")
    st.markdown("---")

    page = st.radio(
        label     = "Navigation",
        options   = ["Demo", "Explorer", "Metrics"],
        format_func = lambda p: {
            "Demo":     "🔬  Live Demo",
            "Explorer": "📊  Explorer",
            "Metrics":  "🧠  Model Metrics",
        }[p],
        label_visibility = "collapsed",
    )

    st.markdown("---")

    # API health indicator
    health = check_health()
    status = health.get("status", "unknown")
    db_status = health.get("database", "unknown")

    colour = "#2ecc71" if status == "ok" else "#e74c3c"
    st.markdown(
        f"<div style='font-size:0.8rem; color:#666'>"
        f"API &nbsp;"
        f"<span style='color:{colour}; font-weight:600'>"
        f"{'● online' if status == 'ok' else '● offline'}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if db_status not in ("unknown", "unreachable"):
        db_colour = "#2ecc71" if db_status == "connected" else "#e74c3c"
        st.markdown(
            f"<div style='font-size:0.8rem; color:#666'>"
            f"DB &nbsp;"
            f"<span style='color:{db_colour}; font-weight:600'>"
            f"{'● connected' if db_status == 'connected' else '● unreachable'}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#aaa'>"
        "Clinical NLP Pipeline · v1.0<br>"
        "Built with spaCy · Bio_ClinicalBERT<br>"
        "FastAPI · Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Page routing ──────────────────────────────────────────────────

if page == "Demo":
    demo.render()
elif page == "Explorer":
    explorer.render()
elif page == "Metrics":
    model_metrics.render()
