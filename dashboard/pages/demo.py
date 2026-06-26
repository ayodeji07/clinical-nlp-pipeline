"""
dashboard/pages/demo.py
────────────────────────────────────────────────────────────────
Live demo page — paste clinical text and get entities,
ICD-10 mappings, and severity classification in real time.

This is the flagship page of the dashboard and the one most
relevant for LinkedIn / portfolio demos.  It shows:
  - Entity extraction with colour-coded labels
  - ICD-10 code matches with confidence scores
  - Severity classification with probability bar
  - Entity breakdown by type
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import streamlit as st

from dashboard.api_client import analyse_note

# ── Label colours (matching the co-occurrence graph palette) ──────
_LABEL_COLOURS: dict[str, str] = {
    "DISEASE":   "#e74c3c",   # red
    "MEDICATION": "#3498db",  # blue
    "PROCEDURE": "#2ecc71",   # green
    "SYMPTOM":   "#f39c12",   # orange
    "ANATOMY":   "#9b59b6",   # purple
}

_SEVERITY_COLOURS: dict[str, str] = {
    "routine":  "#2ecc71",
    "urgent":   "#f39c12",
    "critical": "#e74c3c",
}

# Sample notes for quick demo — good for LinkedIn recordings
_SAMPLE_NOTES: dict[str, str] = {
    "Cardiac (urgent)": (
        "62-year-old male presenting to the emergency department with "
        "acute onset chest pain radiating to the left arm, diaphoresis, "
        "and shortness of breath. History of hypertension, type 2 diabetes "
        "mellitus, and hyperlipidaemia. Current medications: metformin 500mg "
        "twice daily, atorvastatin 40mg once daily, lisinopril 10mg. ECG shows "
        "ST elevation in leads V1-V4. Troponin elevated at 2.4 ng/mL. "
        "Impression: STEMI. Plan: urgent PCI, aspirin 300mg loading dose, "
        "heparin infusion, cardiology consult."
    ),
    "Orthopaedics (routine)": (
        "45-year-old female referred for right knee pain. Patient reports "
        "gradual onset pain over 6 months, worse with stairs and prolonged "
        "standing. No locking or giving way. BMI 32. X-ray shows mild "
        "medial compartment osteoarthritis with joint space narrowing. "
        "Impression: right knee osteoarthritis. Plan: physiotherapy, "
        "weight management, naproxen 500mg twice daily as needed, "
        "follow-up in 3 months."
    ),
    "Neurology (critical)": (
        "Patient found unresponsive at home by family. GCS 8 on arrival. "
        "History of atrial fibrillation, not on anticoagulation. CT head "
        "shows large left MCA territory infarction with midline shift. "
        "Blood pressure 198/110 mmHg. INR 1.1. Neurosurgery consulted for "
        "possible decompressive craniectomy. Patient transferred to the "
        "neurocritical care unit. Family informed of critical prognosis."
    ),
}


def render() -> None:
    """Render the live demo page."""

    st.title("🔬 Clinical NLP — Live Demo")
    st.markdown(
        "Paste any clinical note below to extract named entities, "
        "map to ICD-10 codes, and predict severity in real time."
    )

    # ── Input ─────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 1])

    with col_right:
        st.markdown("**Quick examples**")
        for label, text in _SAMPLE_NOTES.items():
            if st.button(label, use_container_width=True):
                st.session_state["demo_text"] = text

    with col_left:
        note_text = st.text_area(
            label        = "Clinical note",
            value        = st.session_state.get("demo_text", ""),
            height       = 220,
            placeholder  = "Paste a discharge summary, clinic letter, or operative note...",
            label_visibility = "collapsed",
        )

    # Options
    with st.expander("Options", expanded=False):
        include_icd10    = st.checkbox("ICD-10 mapping",          value=True)
        include_severity = st.checkbox("Severity classification",  value=True)

    analyse_btn = st.button(
        "Analyse",
        type             = "primary",
        use_container_width = True,
        disabled         = not note_text.strip(),
    )

    if not analyse_btn:
        st.info("Enter a clinical note above and press **Analyse** to begin.")
        return

    # ── Call the API ──────────────────────────────────────────────
    with st.spinner("Extracting entities..."):
        result = analyse_note(
            text             = note_text,
            include_icd10    = include_icd10,
            include_severity = include_severity,
        )

    if result is None:
        st.error(
            "The API did not respond.  "
            "Check that the backend is running and try again."
        )
        return

    entities = result.get("entities", [])

    # ── Summary cards ─────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Words",    result.get("word_count", 0))
    c2.metric("Entities", len(entities))
    c3.metric(
        "Unique types",
        len(result.get("entity_counts", {}))
    )
    if result.get("processing_ms"):
        c4.metric("Time", f"{result['processing_ms']:.0f} ms")

    # ── Severity ──────────────────────────────────────────────────
    if include_severity and result.get("severity"):
        sev       = result["severity"]
        sev_label = sev["label"]
        sev_conf  = sev["confidence"]
        colour    = _SEVERITY_COLOURS.get(sev_label, "#95a5a6")

        st.markdown("### Severity")
        st.markdown(
            f"<div style='padding:12px; border-left: 5px solid {colour}; "
            f"border-radius:4px; background:#fafafa'>"
            f"<span style='font-size:1.4rem; font-weight:600; color:{colour}'>"
            f"{sev_label.upper()}</span>"
            f"<span style='color:#666; margin-left:12px'>"
            f"confidence {sev_conf:.0%}</span></div>",
            unsafe_allow_html=True,
        )

        # Probability bars
        probs = sev.get("probabilities", {})
        if probs:
            import plotly.graph_objects as go
            labels = list(probs.keys())
            values = [probs[l] for l in labels]
            colours_list = [_SEVERITY_COLOURS.get(l, "#95a5a6") for l in labels]
            fig = go.Figure(go.Bar(
                x           = labels,
                y           = values,
                marker_color = colours_list,
                text        = [f"{v:.1%}" for v in values],
                textposition = "outside",
            ))
            fig.update_layout(
                yaxis       = dict(range=[0, 1], tickformat=".0%"),
                height      = 220,
                margin      = dict(t=10, b=10),
                showlegend  = False,
                plot_bgcolor = "white",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Entity breakdown ──────────────────────────────────────────
    if entities:
        st.markdown("### Extracted Entities")

        # Counts by type
        counts = result.get("entity_counts", {})
        if counts:
            import plotly.graph_objects as go
            labels = list(counts.keys())
            values = list(counts.values())
            fig_c = go.Figure(go.Bar(
                x            = labels,
                y            = values,
                marker_color = [
                    _LABEL_COLOURS.get(l, "#95a5a6") for l in labels
                ],
                text         = values,
                textposition = "outside",
            ))
            fig_c.update_layout(
                height      = 200,
                margin      = dict(t=10, b=10),
                showlegend  = False,
                plot_bgcolor = "white",
            )
            st.plotly_chart(fig_c, use_container_width=True)

        # Entity table grouped by type
        for label_type in ["DISEASE", "MEDICATION", "PROCEDURE", "SYMPTOM", "ANATOMY"]:
            type_entities = [e for e in entities if e["label"] == label_type]
            if not type_entities:
                continue

            colour = _LABEL_COLOURS.get(label_type, "#95a5a6")
            st.markdown(
                f"<span style='background:{colour}; color:white; "
                f"padding:3px 10px; border-radius:12px; font-size:0.85rem'>"
                f"{label_type}</span>",
                unsafe_allow_html=True,
            )

            for ent in type_entities:
                icd_matches = ent.get("icd10_matches", [])
                if icd_matches and include_icd10:
                    best = icd_matches[0]
                    st.markdown(
                        f"**{ent['text']}** &nbsp;→&nbsp; "
                        f"`{best['icd10_code']}` {best['description']} "
                        f"<span style='color:#aaa; font-size:0.8rem'>"
                        f"({best['match_method']}, {best['confidence']:.0%})</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"- {ent['text']}")

            st.markdown("")   # spacing between label groups

    else:
        st.info("No entities were extracted from this note.")
