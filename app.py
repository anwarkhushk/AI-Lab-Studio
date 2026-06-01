import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Lab Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global CSS ─────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- HIDE STREAMLIT DEFAULTS ---------- */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    /* Keep sidebar collapse/expand button accessible */
    [data-testid="collapsedControl"] { visibility: visible !important; }
    header [data-testid="stSidebarCollapsedControl"] { visibility: visible !important; }
    section[data-testid="stSidebarCollapsedControl"] { visibility: visible !important; }


    /* ---------- GLOBAL BACKGROUND ---------- */
    .stApp {
        background: #080c18;
    }

    /* ---------- SIDEBAR ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #0a0e1a 100%);
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }

    /* ---------- SIDEBAR RADIO (nav items) ---------- */
    div[data-testid="stSidebar"] .stRadio > label {
        display: none;
    }
    div[data-testid="stSidebar"] .stRadio > div {
        gap: 6px;
    }
    div[data-testid="stSidebar"] .stRadio [data-testid="stWidgetLabel"] {
        display: none;
    }

    /* ---------- METRIC CARDS ---------- */
    [data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 14px 18px;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    [data-testid="stMetricValue"] { color: #a5b4fc !important; font-weight: 700; }

    /* ---------- BUTTONS ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #4f46e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 15px rgba(108,99,255,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(108,99,255,0.5) !important;
    }

    /* ---------- SELECTBOX / SLIDER ---------- */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stSlider"] > div {
        border-radius: 8px;
    }

    /* ---------- TABS ---------- */
    .stTabs [data-baseweb="tab-list"] {
        background: #111827;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: #94a3b8 !important;
        font-weight: 500;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6C63FF, #4f46e5) !important;
        color: white !important;
        font-weight: 600;
    }

    /* ---------- SUCCESS / ERROR / INFO ---------- */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* ---------- CODE BLOCKS ---------- */
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
        background: #111827 !important;
        border-radius: 8px;
    }

    /* ---------- DIVIDERS ---------- */
    hr {
        border-color: #1e293b;
    }

    /* ---------- EXPANDER ---------- */
    .streamlit-expanderHeader {
        background: #111827;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }

    /* ---------- DATAFRAMES ---------- */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Import modules ─────────────────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from modules import (
    intelligent_agents,
    search_algorithms,
    optimization,
    genetic_algorithm,
    csp,
    ml_playground,
)
from auth import render_auth_page, render_logout_button, init_session
from workspace import render_workspace_sidebar, render_workspace_page


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo / Title
        st.markdown(
            """
            <div style='text-align:center;padding:20px 0 10px 0;'>
                <div style='font-size:2.5rem;'>🧠</div>
                <div style='font-size:1.3rem;font-weight:800;
                            background:linear-gradient(135deg,#6C63FF,#22d3ee);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                            background-clip:text;'>AI Lab Studio</div>
                <div style='font-size:.72rem;color:#475569;margin-top:2px;'>
                    Interactive AI Learning Platform
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:.7rem;color:#475569;text-transform:uppercase;"
            "letter-spacing:2px;margin-bottom:8px;'>Navigation</div>",
            unsafe_allow_html=True,
        )

        pages = {
            "🏠  Home": "Home",
            "🤖  Intelligent Agents": "Agents",
            "🔍  Search Algorithms": "Search",
            "⚙️  Optimization": "Optimization",
            "🧬  Genetic Algorithm": "Genetic",
            "🧩  CSP": "CSP",
            "🤖  ML Playground": "ML",
        }

        page = st.radio("Navigate", list(pages.keys()), key="nav_radio", label_visibility="collapsed")

        st.markdown("---")

        # Info card
        st.markdown(
            """
            <div style='background:#111827;border:1px solid #1e293b;border-radius:10px;
                        padding:14px;font-size:.78rem;color:#64748b;'>
                <b style='color:#a5b4fc;'>About</b><br><br>
                An interactive platform to explore and visualize core AI &amp; ML algorithms.
                Built with Python + Streamlit.
                <br><br>
                <b style='color:#a5b4fc;'>Stack</b><br>
                Python · Streamlit · NumPy · Pandas · Plotly · Scikit-learn · NetworkX
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Auth / workspace in sidebar ───────────────────────────────
        is_guest = st.session_state.get("guest_mode", False)
        open_ws  = render_workspace_sidebar(is_guest=is_guest)
        if open_ws:
            st.session_state["show_workspace"] = True
        render_logout_button()

    return pages[page], open_ws


# ── HOME PAGE ──────────────────────────────────────────────────────────────────
def render_home():
    # Hero
    st.markdown(
        """
        <div style='text-align:center;padding:50px 0 30px 0;'>
            <div style='font-size:4rem;line-height:1;'>🧠</div>
            <h1 style='font-size:3rem;font-weight:800;margin:16px 0 8px 0;
                       background:linear-gradient(135deg,#6C63FF 0%,#22d3ee 50%,#34d399 100%);
                       -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                       background-clip:text;'>
                AI Lab Studio
            </h1>
            <p style='font-size:1.15rem;color:#64748b;max-width:600px;margin:0 auto 30px auto;
                      line-height:1.7;'>
                An interactive educational platform to <b style="color:#a5b4fc;">visualize</b>,
                <b style="color:#22d3ee;">simulate</b>, and
                <b style="color:#34d399;">experiment</b> with core Artificial Intelligence
                and Machine Learning algorithms.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Stats row
    col1, col2, col3, col4, col5 = st.columns(5)
    stats = [
        ("🤖", "3", "Agent Types"),
        ("🔍", "5", "Search Algos"),
        ("⚙️", "3", "Optimizers"),
        ("🧩", "3", "CSP Problems"),
        ("🤖", "4", "ML Models"),
    ]
    for col, (icon, val, label) in zip([col1, col2, col3, col4, col5], stats):
        col.markdown(
            f"""
            <div style='background:#111827;border:1px solid #1e293b;border-radius:14px;
                        padding:20px;text-align:center;'>
                <div style='font-size:1.8rem;'>{icon}</div>
                <div style='font-size:2rem;font-weight:800;color:#6C63FF;'>{val}</div>
                <div style='font-size:.78rem;color:#64748b;'>{label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    features = [
        ("🤖", "Intelligent Agents", "#6C63FF",
         "Simulate Vacuum Cleaner, Reflex, and Goal-Based agents with PEAS visualization."),
        ("🔍", "Search Algorithms", "#22d3ee",
         "BFS, DFS, DLS, IDS, and A* with animated graph traversal and performance stats."),
        ("⚙️", "Optimization", "#34d399",
         "Hill Climbing, Simulated Annealing, and 8-Queens problem solver."),
        ("🧬", "Genetic Algorithm", "#f59e0b",
         "Watch chromosomes evolve by selection, crossover, and mutation over generations."),
        ("🧩", "CSP Problems", "#f87171",
         "Map Coloring, N-Queens, and Sudoku solved by backtracking constraint satisfaction."),
        ("🤖", "ML Playground", "#a78bfa",
         "KNN, Naïve Bayes, K-Means, and Logistic Regression with decision boundary plots."),
    ]

    cols = st.columns(3)
    for i, (icon, title, color, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style='background:#111827;border:1px solid {color}33;border-radius:16px;
                            padding:24px;margin-bottom:16px;transition:all .2s;
                            box-shadow:0 4px 20px {color}11;'>
                    <div style='font-size:2rem;margin-bottom:10px;'>{icon}</div>
                    <div style='font-size:1rem;font-weight:700;color:{color};margin-bottom:8px;'>
                        {title}
                    </div>
                    <div style='font-size:.82rem;color:#64748b;line-height:1.6;'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📊 Platform Architecture")

    # Architecture diagram using Plotly
    nodes = [
        ("AI Lab Studio", 0, 3, "#6C63FF"),
        ("Intelligent Agents", -3.5, 1.5, "#22d3ee"),
        ("Search Algorithms", -1.2, 1.5, "#34d399"),
        ("Optimization", 1.2, 1.5, "#f59e0b"),
        ("CSP Problems", 3.5, 1.5, "#f87171"),
        ("ML Playground", 0, 1.5, "#a78bfa"),
        ("Genetic Algo", 2.3, 0, "#fb923c"),
    ]

    fig = go.Figure()
    connections = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]
    for s, e in connections:
        x0, y0 = nodes[s][1], nodes[s][2]
        x1, y1 = nodes[e][1], nodes[e][2]
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                 line=dict(width=2, color="#334155"), hoverinfo="none"))

    for name, x, y, color in nodes:
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=18, color=color, line=dict(width=2, color="white")),
            text=[name], textposition="bottom center",
            textfont=dict(color="#94a3b8", size=11),
        ))

    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=280, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div style='text-align:center;padding:20px 0;color:#334155;font-size:.8rem;'>
            AI Lab Studio · Built with Python + Streamlit · 2026
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    # 1. Initialise auth session state
    init_session()

    # 2. Require login / guest mode before showing any content
    authenticated = st.session_state.get("authenticated", False)
    guest_mode    = st.session_state.get("guest_mode", False)

    if not authenticated and not guest_mode:
        render_auth_page()
        st.stop()

    # 3. Normal app flow
    page, open_ws = render_sidebar()

    # 4. Workspace page (intercepts normal routing)
    user = st.session_state.get("user")
    if st.session_state.get("show_workspace") and user:
        render_workspace_page(user)
        if st.button("← Back to App", key="ws_back_btn"):
            st.session_state["show_workspace"] = False
            st.rerun()
        return

    # Clear workspace flag on normal nav
    if not open_ws:
        st.session_state["show_workspace"] = False

    if page == "Home":
        render_home()
    elif page == "Agents":
        intelligent_agents.show()
    elif page == "Search":
        search_algorithms.show()
    elif page == "Optimization":
        optimization.show()
    elif page == "Genetic":
        genetic_algorithm.show()
    elif page == "CSP":
        csp.show()
    elif page == "ML":
        ml_playground.show()


if __name__ == "__main__":
    main()

