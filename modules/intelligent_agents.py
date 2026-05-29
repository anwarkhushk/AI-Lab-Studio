import streamlit as st
import plotly.graph_objects as go
import numpy as np
import random
import time


# ──────────────────────────────────────────────────────────────────────────────
# Helper: styled section header
# ──────────────────────────────────────────────────────────────────────────────
def section_header(title: str, icon: str = ""):
    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,#1e1b4b,#312e81);
                    border-left:4px solid #6C63FF;border-radius:8px;
                    padding:12px 18px;margin:10px 0 18px 0;'>
            <span style='font-size:1.1rem;font-weight:700;color:#a5b4fc;'>
                {icon} {title}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# PEAS visualizer
# ──────────────────────────────────────────────────────────────────────────────
PEAS_DATA = {
    "Vacuum Cleaner Agent": {
        "Performance": "Clean floors, minimize steps, minimize energy",
        "Environment": "2-room grid (A & B), dirt present/absent",
        "Actuators": "Move Left, Move Right, Suck (vacuum)",
        "Sensors": "Location sensor, Dirt sensor",
        "color": "#6C63FF",
    },
    "Reflex Agent": {
        "Performance": "Respond instantly to percepts, minimize latency",
        "Environment": "Fully observable, static, discrete states",
        "Actuators": "Execute mapped action from condition-action rules",
        "Sensors": "Current state sensor",
        "color": "#22d3ee",
    },
    "Goal-Based Agent": {
        "Performance": "Reach goal efficiently, minimize cost",
        "Environment": "Partially observable, dynamic environment",
        "Actuators": "Plan execution, state transitions",
        "Sensors": "Goal sensor, state sensor, heuristic evaluator",
        "color": "#34d399",
    },
}


def show_peas(agent_name: str):
    data = PEAS_DATA[agent_name]
    categories = ["Performance", "Environment", "Actuators", "Sensors"]
    cols = st.columns(4)
    icons = ["🏆", "🌍", "⚙️", "👁️"]
    for col, cat, icon in zip(cols, categories, icons):
        with col:
            st.markdown(
                f"""
                <div style='background:#1e293b;border:1px solid {data["color"]}44;
                            border-radius:12px;padding:16px;text-align:center;min-height:140px'>
                    <div style='font-size:2rem;'>{icon}</div>
                    <div style='color:{data["color"]};font-weight:700;font-size:.85rem;
                                margin:6px 0 8px 0;text-transform:uppercase;letter-spacing:1px;'>{cat}</div>
                    <div style='color:#94a3b8;font-size:.78rem;line-height:1.5;'>{data[cat]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Vacuum Cleaner Agent
# ──────────────────────────────────────────────────────────────────────────────
def vacuum_agent_step(position: str, room_a: bool, room_b: bool):
    """Simple reflex vacuum agent."""
    if position == "A" and room_a:
        return "Suck", position, False, room_b
    elif position == "B" and room_b:
        return "Suck", position, room_a, False
    elif position == "A":
        return "Move Right", "B", room_a, room_b
    else:
        return "Move Left", "A", room_a, room_b


def draw_vacuum_world(pos: str, room_a: bool, room_b: bool, step: int, action: str):
    fig = go.Figure()

    colors_a = "#ef4444" if room_a else "#22c55e"
    colors_b = "#ef4444" if room_b else "#22c55e"

    fig.add_shape(type="rect", x0=0, y0=0, x1=4, y1=3, line=dict(color="#6C63FF", width=3))
    fig.add_shape(type="rect", x0=4, y0=0, x1=8, y1=3, line=dict(color="#6C63FF", width=3))
    fig.add_shape(type="line", x0=4, y0=0, x1=4, y1=3, line=dict(color="#6C63FF", width=2))

    for rx, color, label, dirty in [(2, colors_a, "Room A", room_a), (6, colors_b, "Room B", room_b)]:
        fig.add_annotation(x=rx, y=2.5, text=f"<b>{label}</b>", showarrow=False,
                           font=dict(size=14, color="white"))
        status = "🔴 Dirty" if dirty else "🟢 Clean"
        fig.add_annotation(x=rx, y=2.0, text=status, showarrow=False,
                           font=dict(size=11, color=color))

    # Agent
    ax = 2 if pos == "A" else 6
    fig.add_trace(go.Scatter(x=[ax], y=[1.2], mode="markers+text",
                             marker=dict(size=60, color="#6C63FF", symbol="circle"),
                             text=["🤖"], textfont=dict(size=24), name="Agent"))

    fig.update_layout(
        title=dict(text=f"Step {step}: <b>{action}</b>", font=dict(color="#a5b4fc", size=14)),
        xaxis=dict(range=[-0.5, 8.5], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[-0.5, 3.5], showgrid=False, showticklabels=False, zeroline=False),
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        height=280, margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    return fig


def vacuum_cleaner_demo():
    section_header("Vacuum Cleaner Agent Simulation", "🧹")

    col1, col2, col3 = st.columns(3)
    with col1:
        room_a_init = st.selectbox("Room A initial state", ["Dirty", "Clean"], key="va_ra") == "Dirty"
    with col2:
        room_b_init = st.selectbox("Room B initial state", ["Dirty", "Clean"], key="va_rb") == "Dirty"
    with col3:
        start_pos = st.selectbox("Agent starts at", ["A", "B"], key="va_sp")

    if st.button("▶  Run Simulation", key="va_run", use_container_width=True):
        pos, ra, rb = start_pos, room_a_init, room_b_init
        log = []
        chart_placeholder = st.empty()
        log_placeholder = st.empty()

        for step in range(1, 9):
            action, pos, ra, rb = vacuum_agent_step(pos, ra, rb)
            log.append(f"**Step {step}** → {action} | Pos: {pos} | A: {'Dirty' if ra else 'Clean'} | B: {'Dirty' if rb else 'Clean'}")
            chart_placeholder.plotly_chart(draw_vacuum_world(pos, ra, rb, step, action),
                                           use_container_width=True)
            with log_placeholder.container():
                for entry in log[-4:]:
                    st.markdown(entry)
            if not ra and not rb:
                st.success("✅ Both rooms are clean! Agent task completed.")
                break
            time.sleep(0.8)


# ──────────────────────────────────────────────────────────────────────────────
# Reflex Agent
# ──────────────────────────────────────────────────────────────────────────────
def reflex_agent_demo():
    section_header("Simple Reflex Agent", "⚡")
    st.markdown("""
    A **Simple Reflex Agent** acts based on the *current percept* only,
    using condition–action rules. It has no memory of past states.
    """)

    rules = {
        "obstacle_ahead": "Turn Right",
        "goal_reached": "Stop",
        "path_clear": "Move Forward",
        "low_battery": "Return to Base",
        "unknown_object": "Scan & Classify",
    }

    st.markdown("#### 📋 Condition → Action Rule Table")
    for condition, action in rules.items():
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"`IF {condition.replace('_',' ').title()}`")
        with col2:
            st.markdown(f"→ **{action}**")

    st.markdown("---")
    st.markdown("#### 🎮 Interactive Percept Tester")
    percept = st.selectbox("Choose current percept:", list(rules.keys()), key="reflex_percept")
    if st.button("Get Agent Action", key="reflex_btn"):
        action = rules[percept]
        st.markdown(
            f"""<div style='background:linear-gradient(135deg,#1e1b4b,#0f172a);
                           border:1px solid #6C63FF;border-radius:12px;padding:20px;text-align:center'>
                <div style='font-size:.85rem;color:#94a3b8;'>Percept: <b style="color:#a5b4fc">{percept.replace("_"," ").title()}</b></div>
                <div style='font-size:1.8rem;margin:10px 0;'>⚡</div>
                <div style='font-size:1.2rem;color:#22d3ee;font-weight:700;'>Action: {action}</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Goal-Based Agent
# ──────────────────────────────────────────────────────────────────────────────
def goal_agent_demo():
    section_header("Goal-Based Agent", "🎯")
    st.markdown("""
    A **Goal-Based Agent** considers *future states* and selects actions
    that bring it closer to its goal using a simple BFS on a grid.
    """)

    grid_size = st.slider("Grid Size", 4, 8, 5, key="goal_grid")
    col1, col2 = st.columns(2)
    with col1:
        start = st.text_input("Start (row,col)", "0,0", key="goal_start")
    with col2:
        goal_inp = st.text_input("Goal (row,col)", f"{grid_size-1},{grid_size-1}", key="goal_goal")

    if st.button("▶  Find Path", key="goal_run", use_container_width=True):
        try:
            sr, sc = map(int, start.split(","))
            gr, gc = map(int, goal_inp.split(","))
        except Exception:
            st.error("Invalid coordinates."); return

        from collections import deque

        def bfs(n, s, g):
            visited, parent = {s}, {s: None}
            queue = deque([s])
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            while queue:
                curr = queue.popleft()
                if curr == g:
                    path = []
                    while curr:
                        path.append(curr); curr = parent[curr]
                    return path[::-1]
                for dr, dc in dirs:
                    nxt = (curr[0] + dr, curr[1] + dc)
                    if 0 <= nxt[0] < n and 0 <= nxt[1] < n and nxt not in visited:
                        visited.add(nxt); parent[nxt] = curr; queue.append(nxt)
            return []

        path = bfs(grid_size, (sr, sc), (gr, gc))

        if not path:
            st.error("No path found."); return

        grid = np.zeros((grid_size, grid_size))
        for r, c in path:
            grid[r][c] = 0.5
        grid[sr][sc] = 1.0; grid[gr][gc] = 0.9

        annotations = []
        for r in range(grid_size):
            for c in range(grid_size):
                if (r, c) == (sr, sc):
                    txt = "🤖"
                elif (r, c) == (gr, gc):
                    txt = "🏁"
                elif (r, c) in path:
                    txt = "·"
                else:
                    txt = ""
                annotations.append(dict(x=c, y=r, text=txt, showarrow=False,
                                        font=dict(size=16 if txt in ["🤖", "🏁"] else 20)))

        fig = go.Figure(data=go.Heatmap(z=grid, colorscale=[[0, "#0f172a"], [0.5, "#312e81"], [1.0, "#6C63FF"]],
                                         showscale=False))
        fig.update_layout(annotations=annotations, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                          height=350, xaxis=dict(showgrid=False, showticklabels=False),
                          yaxis=dict(showgrid=False, showticklabels=False, autorange="reversed"),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.success(f"✅ Path found! Length = {len(path)} steps")
        st.code(" → ".join(str(p) for p in path))


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────
def show():
    st.markdown("## 🤖 Intelligent Agent Simulation")
    st.markdown("Explore different types of AI agents and their PEAS representations.")

    agent_type = st.selectbox(
        "Select Agent Type",
        ["Vacuum Cleaner Agent", "Reflex Agent", "Goal-Based Agent"],
        key="agent_select",
    )

    section_header("PEAS Representation", "📊")
    show_peas(agent_type)
    st.markdown("---")

    if agent_type == "Vacuum Cleaner Agent":
        vacuum_cleaner_demo()
    elif agent_type == "Reflex Agent":
        reflex_agent_demo()
    else:
        goal_agent_demo()
