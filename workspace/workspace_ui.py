"""
workspace/workspace_ui.py
Student Workspace UI – sidebar section + full workspace page.
Renders experiments, datasets, saved models, and comparison charts.
All colours match the existing dark theme.
"""

import json
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from storage.experiments import get_experiments, delete_experiment
from storage.datasets     import get_datasets, delete_dataset_meta
from storage.models       import get_models, delete_model_meta, load_model

# ── colours (match app.py palette) ────────────────────────────────────────────
_BG     = "#0f172a"
_CARD   = "#111827"
_BORDER = "#1e293b"
_ACCENT = "#6C63FF"
_CYAN   = "#22d3ee"
_GREEN  = "#34d399"
_AMBER  = "#f59e0b"
_PINK   = "#f87171"
_MUTED  = "#64748b"
_SUB    = "#475569"
COLORS  = [_ACCENT, _CYAN, _GREEN, _AMBER, _PINK, "#a78bfa", "#fb923c"]


def _section_header(icon: str, title: str, color: str = _ACCENT):
    st.markdown(
        f"""
        <div style='display:flex;align-items:center;gap:10px;margin:18px 0 10px 0;'>
            <span style='font-size:1.4rem;'>{icon}</span>
            <span style='font-size:1.1rem;font-weight:700;color:{color};'>{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _mini_card(label: str, value, color: str = _ACCENT):
    return (
        f"<div style='background:{_CARD};border:1px solid {color}33;border-radius:10px;"
        f"padding:10px 14px;text-align:center;'>"
        f"<div style='font-size:.72rem;color:{_MUTED};'>{label}</div>"
        f"<div style='font-size:1.1rem;font-weight:700;color:{color};'>{value}</div>"
        f"</div>"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar section
# ══════════════════════════════════════════════════════════════════════════════

def render_workspace_sidebar(is_guest: bool = False) -> bool:
    """
    Renders the 'Student Workspace' section in the sidebar.
    Returns True if the user clicked to open the workspace page.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size:.7rem;color:#475569;text-transform:uppercase;"
        "letter-spacing:2px;margin-bottom:8px;'>Student Workspace</div>",
        unsafe_allow_html=True,
    )

    if is_guest:
        st.sidebar.markdown(
            "<div style='background:#111827;border:1px solid #1e293b;border-radius:10px;"
            "padding:10px 14px;font-size:.78rem;color:#64748b;'>"
            "🔒 <b style='color:#a5b4fc;'>Login</b> to save experiments,<br>"
            "datasets, and models.</div>",
            unsafe_allow_html=True,
        )
        return False

    return st.sidebar.button(
        "🗂️  Open My Workspace",
        key="open_workspace_btn",
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main workspace page
# ══════════════════════════════════════════════════════════════════════════════

def render_workspace_page(user: dict):
    user_id  = user["id"]
    username = user["username"]

    st.markdown(
        f"""
        <div style='padding:10px 0 24px 0;'>
            <div style='font-size:2rem;font-weight:800;
                        background:linear-gradient(135deg,#6C63FF,#22d3ee);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text;'>🗂️ My Workspace</div>
            <div style='font-size:.9rem;color:#64748b;margin-top:4px;'>
                Personal workspace for <b style='color:#a5b4fc;'>{username}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧪 Experiments",
        "📦 Datasets",
        "🤖 Models",
        "📊 Compare",
    ])

    with tab1:
        _experiments_tab(user_id)
    with tab2:
        _datasets_tab(user_id)
    with tab3:
        _models_tab(user_id)
    with tab4:
        _compare_tab(user_id)


# ══════════════════════════════════════════════════════════════════════════════
# Experiments tab
# ══════════════════════════════════════════════════════════════════════════════

def _experiments_tab(user_id: int):
    _section_header("🧪", "Saved Experiments", _ACCENT)

    experiments = get_experiments(user_id)

    if not experiments:
        st.info("No experiments saved yet. Run an algorithm and click **Save Experiment**!")
        return

    # Summary metrics
    algo_counts: dict = {}
    for e in experiments:
        algo_counts[e["algorithm"]] = algo_counts.get(e["algorithm"], 0) + 1

    cols = st.columns(min(len(algo_counts), 4))
    for i, (algo, cnt) in enumerate(list(algo_counts.items())[:4]):
        cols[i].markdown(_mini_card(algo, cnt, COLORS[i % len(COLORS)]),
                         unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Filter
    algorithms = sorted({e["algorithm"] for e in experiments})
    sel_algo = st.selectbox(
        "Filter by algorithm", ["All"] + algorithms, key="ws_exp_filter"
    )

    filtered = (
        experiments if sel_algo == "All"
        else [e for e in experiments if e["algorithm"] == sel_algo]
    )

    for exp in filtered:
        with st.expander(
            f"{'🔍' if exp['algorithm'] in ('BFS','DFS','DLS','IDS','A*') else '🤖'} "
            f"**{exp['algorithm']}** — {exp['created_at'][:16]}"
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",    f"{exp['accuracy']:.2%}"  if exp.get("accuracy")    is not None else "—")
            c2.metric("Cost",        f"{exp['cost']:.2f}"      if exp.get("cost")        is not None else "—")
            c3.metric("Path Length", exp["path_length"]         if exp.get("path_length") is not None else "—")
            c4.metric("Saved",       exp["created_at"][:10])

            if exp["parameters"]:
                st.json(exp["parameters"])
            if exp.get("notes"):
                st.caption(f"📝 {exp['notes']}")

            if st.button("🗑️ Delete", key=f"del_exp_{exp['id']}"):
                delete_experiment(exp["id"], user_id)
                st.success("Deleted.")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Datasets tab
# ══════════════════════════════════════════════════════════════════════════════

def _datasets_tab(user_id: int):
    _section_header("📦", "Saved Datasets", _CYAN)

    datasets = get_datasets(user_id)

    if not datasets:
        st.info("No datasets saved yet. Upload a dataset in any ML module and click **Save Dataset**!")
        return

    for ds in datasets:
        with st.expander(f"📄 **{ds['name']}** — {ds['upload_date'][:10]}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows",    ds["rows"])
            c2.metric("Columns", ds["cols"])
            c3.metric("Uploaded", ds["upload_date"][:10])

            if ds["features"]:
                st.caption("Features: " + ", ".join(ds["features"][:10])
                           + (" …" if len(ds["features"]) > 10 else ""))

            col_a, col_b = st.columns(2)
            with col_a:
                if ds.get("file_path") and os.path.exists(ds["file_path"]):
                    if st.button("🔄 Reload into Session", key=f"reload_ds_{ds['id']}"):
                        try:
                            df = pd.read_csv(ds["file_path"])
                            st.session_state["active_dataset"] = {
                                "df":       df,
                                "features": ds["features"],
                                "target":   None,
                                "name":     ds["name"],
                            }
                            st.success(f"Dataset **{ds['name']}** loaded into session!")
                        except Exception as e:
                            st.error(f"Could not reload: {e}")
                else:
                    st.caption("(File not on disk – metadata only)")
            with col_b:
                if st.button("🗑️ Delete", key=f"del_ds_{ds['id']}"):
                    delete_dataset_meta(ds["id"], user_id)
                    st.success("Deleted.")
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Models tab
# ══════════════════════════════════════════════════════════════════════════════

def _models_tab(user_id: int):
    _section_header("🤖", "Saved Models", _GREEN)

    models = get_models(user_id)

    if not models:
        st.info("No models saved yet. Train a model in ML Playground and click **Save Model**!")
        return

    for mdl in models:
        with st.expander(f"🤖 **{mdl['name']}** ({mdl['algorithm']}) — {mdl['created_at'][:10]}"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Algorithm", mdl["algorithm"])
            c2.metric("Accuracy", f"{mdl['accuracy']:.2%}" if mdl.get("accuracy") is not None else "—")
            c3.metric("Saved",    mdl["created_at"][:10])

            col_a, col_b = st.columns(2)
            with col_a:
                if mdl.get("file_path") and os.path.exists(mdl["file_path"]):
                    if st.button("🔄 Load Model", key=f"load_mdl_{mdl['id']}"):
                        try:
                            loaded = load_model(mdl["file_path"])
                            st.session_state["loaded_model"] = {
                                "model": loaded,
                                "name":  mdl["name"],
                                "algorithm": mdl["algorithm"],
                            }
                            st.success(f"Model **{mdl['name']}** loaded into session!")
                        except Exception as e:
                            st.error(f"Could not load: {e}")
                else:
                    st.caption("(Model file not found on disk)")
            with col_b:
                if st.button("🗑️ Delete", key=f"del_mdl_{mdl['id']}"):
                    delete_model_meta(mdl["id"], user_id)
                    st.success("Deleted.")
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Comparison tab
# ══════════════════════════════════════════════════════════════════════════════

def _compare_tab(user_id: int):
    _section_header("📊", "Compare Experiments", _AMBER)

    experiments = get_experiments(user_id)

    if len(experiments) < 2:
        st.info("Save at least **2 experiments** to compare them here.")
        return

    algorithms = sorted({e["algorithm"] for e in experiments})
    sel_algos  = st.multiselect(
        "Select algorithms to compare",
        algorithms,
        default=algorithms[:min(3, len(algorithms))],
        key="cmp_algos",
    )

    filtered = [e for e in experiments if e["algorithm"] in sel_algos]

    if not filtered:
        st.warning("No experiments match the selection.")
        return

    # Build comparison dataframe
    rows = []
    for e in filtered:
        rows.append({
            "Algorithm":  e["algorithm"],
            "Accuracy":   e.get("accuracy"),
            "Cost":       e.get("cost"),
            "Path Len":   e.get("path_length"),
            "Saved":      e["created_at"][:16],
            "ID":         e["id"],
        })
    df_cmp = pd.DataFrame(rows)

    st.dataframe(df_cmp.drop(columns=["ID"]), use_container_width=True)

    # Accuracy bar chart
    acc_df = df_cmp.dropna(subset=["Accuracy"])
    if not acc_df.empty:
        st.markdown("##### 🎯 Accuracy Comparison")
        fig = go.Figure()
        for i, row in acc_df.iterrows():
            fig.add_trace(go.Bar(
                x=[f"{row['Algorithm']} #{row['ID']}"],
                y=[row["Accuracy"]],
                marker_color=COLORS[i % len(COLORS)],
                name=row["Algorithm"],
            ))
        fig.update_layout(
            paper_bgcolor=_BG, plot_bgcolor=_BG,
            xaxis=dict(color="#475569", showgrid=False),
            yaxis=dict(color="#475569", gridcolor=_BORDER, title="Accuracy"),
            legend=dict(bgcolor=_CARD, font=dict(color="#94a3b8")),
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Cost line chart
    cost_df = df_cmp.dropna(subset=["Cost"])
    if not cost_df.empty:
        st.markdown("##### 💸 Cost Comparison")
        fig2 = go.Figure()
        for algo in sel_algos:
            subset = cost_df[cost_df["Algorithm"] == algo]
            if not subset.empty:
                fig2.add_trace(go.Scatter(
                    x=subset["Saved"], y=subset["Cost"],
                    mode="lines+markers",
                    name=algo,
                    line=dict(width=2),
                    marker=dict(size=8),
                ))
        fig2.update_layout(
            paper_bgcolor=_BG, plot_bgcolor=_BG,
            xaxis=dict(color="#475569", showgrid=False),
            yaxis=dict(color="#475569", gridcolor=_BORDER, title="Cost"),
            legend=dict(bgcolor=_CARD, font=dict(color="#94a3b8")),
            height=280, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Path length comparison
    plen_df = df_cmp.dropna(subset=["Path Len"])
    if not plen_df.empty:
        st.markdown("##### 🛤️ Path Length Comparison")
        fig3 = px.bar(
            plen_df,
            x="Algorithm",
            y="Path Len",
            color="Algorithm",
            color_discrete_sequence=COLORS,
            labels={"Path Len": "Path Length"},
        )
        fig3.update_layout(
            paper_bgcolor=_BG, plot_bgcolor=_BG,
            xaxis=dict(color="#475569", showgrid=False),
            yaxis=dict(color="#475569", gridcolor=_BORDER),
            legend=dict(bgcolor=_CARD, font=dict(color="#94a3b8")),
            height=260, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)
