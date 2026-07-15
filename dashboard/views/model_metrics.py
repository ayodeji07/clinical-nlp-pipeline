"""
dashboard/views/model_metrics.py
────────────────────────────────────────────────────────────────
Model Metrics page — classifier performance visualisation.

Shows:
  - Current model accuracy and F1 score
  - Confusion matrix
  - Per-class precision / recall / F1
  - Training history (if multiple runs exist)
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the Model Metrics page."""

    st.title("🧠 Model Metrics")
    st.markdown(
        "Performance of the severity classification model "
        "fine-tuned on MTSamples."
    )

    # ── Model info ────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Base model",  "Bio_ClinicalBERT")
    col_b.metric("Task",        "Severity (3-class)")
    col_c.metric("Dataset",     "MTSamples")

    st.markdown("---")

    # ── Load metrics from saved model directory ────────────────────
    import json
    from src.utils.config import ModelConfig

    metrics_path = ModelConfig.fine_tuned_dir / "training_metrics.json"

    if not metrics_path.exists():
        st.info(
            "No trained model found yet.  "
            "Run the training notebook to fine-tune Bio_ClinicalBERT:\n\n"
            "```\nnotebooks/03_classification.ipynb\n```"
        )
        _render_placeholder_metrics()
        return

    with open(metrics_path) as fh:
        metrics = json.load(fh)

    # ── Summary metrics ───────────────────────────────────────────
    st.markdown("### Final Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Val Accuracy",  f"{metrics.get('val_accuracy',  0):.1%}")
    m2.metric("Val F1",        f"{metrics.get('val_f1',        0):.3f}")
    m3.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.1%}")
    m4.metric("Test F1",       f"{metrics.get('test_f1',       0):.3f}")

    # ── Per-class breakdown ───────────────────────────────────────
    per_class = metrics.get("per_class", {})
    if per_class:
        st.markdown("### Per-class Performance")
        import plotly.graph_objects as go

        classes   = list(per_class.keys())
        precision = [per_class[c].get("precision", 0) for c in classes]
        recall    = [per_class[c].get("recall",    0) for c in classes]
        f1_scores = [per_class[c].get("f1",        0) for c in classes]

        fig = go.Figure()
        for metric_vals, name, colour in [
            (precision, "Precision", "#3498db"),
            (recall,    "Recall",    "#2ecc71"),
            (f1_scores, "F1",        "#e74c3c"),
        ]:
            fig.add_trace(go.Bar(
                name         = name,
                x            = classes,
                y            = metric_vals,
                marker_color = colour,
                text         = [f"{v:.2f}" for v in metric_vals],
                textposition = "outside",
            ))

        fig.update_layout(
            barmode      = "group",
            yaxis        = dict(range=[0, 1.1]),
            height       = 320,
            margin       = dict(t=10, b=10),
            plot_bgcolor = "white",
            legend       = dict(orientation="h", yanchor="bottom", y=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Confusion matrix ──────────────────────────────────────────
    conf_matrix = metrics.get("confusion_matrix")
    if conf_matrix:
        st.markdown("### Confusion Matrix")
        import plotly.figure_factory as ff
        import numpy as np

        labels = ["routine", "urgent", "critical"]
        fig_cm = ff.create_annotated_heatmap(
            z          = np.array(conf_matrix),
            x          = labels,
            y          = labels,
            colorscale = "Blues",
            showscale  = True,
        )
        fig_cm.update_layout(
            height         = 350,
            xaxis_title    = "Predicted",
            yaxis_title    = "Actual",
            yaxis_autorange = "reversed",
            margin         = dict(t=10, b=60),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Training history ──────────────────────────────────────────
    history = metrics.get("history", [])
    if history:
        st.markdown("### Training History")
        import plotly.graph_objects as go

        epochs    = [h["epoch"] for h in history]
        val_acc   = [h.get("accuracy", 0) for h in history]
        val_f1    = [h.get("f1",       0) for h in history]
        loss      = [h.get("loss",     0) for h in history]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=epochs, y=val_acc, name="Val Accuracy",
            line=dict(color="#3498db", width=2)
        ))
        fig_hist.add_trace(go.Scatter(
            x=epochs, y=val_f1, name="Val F1",
            line=dict(color="#2ecc71", width=2)
        ))
        fig_hist.add_trace(go.Scatter(
            x=epochs, y=loss, name="Train Loss",
            line=dict(color="#e74c3c", width=2, dash="dash"),
            yaxis="y2",
        ))
        fig_hist.update_layout(
            height       = 320,
            xaxis_title  = "Epoch",
            yaxis        = dict(title="Score", range=[0, 1]),
            yaxis2       = dict(title="Loss", overlaying="y", side="right"),
            margin       = dict(t=10, b=40),
            plot_bgcolor = "white",
            legend       = dict(orientation="h", yanchor="bottom", y=1),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ── Model notes ───────────────────────────────────────────────
    notes = metrics.get("notes")
    if notes:
        st.markdown("### Training Notes")
        st.markdown(notes)


def _render_placeholder_metrics() -> None:
    """Show placeholder charts when no trained model exists yet."""
    import plotly.graph_objects as go

    st.markdown("### Expected performance (reference)")
    st.caption(
        "These are indicative values for Bio_ClinicalBERT fine-tuned "
        "on MTSamples with weak supervision labels.  "
        "Actual results will vary."
    )

    classes    = ["routine", "urgent", "critical"]
    ref_f1     = [0.87,      0.74,     0.71]
    ref_prec   = [0.89,      0.72,     0.73]
    ref_recall = [0.85,      0.76,     0.69]

    fig = go.Figure()
    for vals, name, colour in [
        (ref_prec,   "Precision", "#3498db"),
        (ref_recall, "Recall",    "#2ecc71"),
        (ref_f1,     "F1",        "#e74c3c"),
    ]:
        fig.add_trace(go.Bar(
            name=name, x=classes, y=vals,
            marker_color=colour,
            text=[f"{v:.2f}" for v in vals],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1.1]),
        height=300,
        margin=dict(t=10, b=10),
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Train the model using `notebooks/03_classification.ipynb` "
        "to see your actual results here."
    )
