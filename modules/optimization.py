import streamlit as st
import plotly.graph_objects as go
import numpy as np
import random
import math
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from workspace.save_helpers import render_save_experiment_btn


# ──────────────────────────────────────────────────────────────────────────────
# Hill Climbing – finds maximum of f(x) = sin(x)*cos(x/2) + some bumps
# ──────────────────────────────────────────────────────────────────────────────
def objective(x):
    return np.sin(x) * np.cos(x / 2) + 0.3 * np.sin(3 * x)


def hill_climbing(start, step_size=0.1, max_iter=200):
    x = start
    history = [x]
    for _ in range(max_iter):
        neighbors = [x - step_size, x + step_size]
        best = max(neighbors, key=objective)
        if objective(best) <= objective(x):
            break
        x = best
        history.append(x)
    return x, objective(x), history


def hill_climbing_demo():
    st.markdown("### 🏔️ Hill Climbing")
    st.markdown("""
    Hill Climbing iteratively moves to a **better neighboring state** until no
    improvement is possible (local maximum / minimum). It can get stuck in local optima.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        start = st.slider("Starting x", -6.0, 6.0, -4.0, 0.1, key="hc_start")
    with col2:
        step = st.slider("Step size", 0.05, 0.5, 0.1, 0.01, key="hc_step")
    with col3:
        runs = st.slider("Random restarts", 1, 10, 3, key="hc_runs")

    if st.button("▶  Run Hill Climbing", key="hc_run", use_container_width=True):
        xs = np.linspace(-7, 7, 400)
        ys = objective(xs)

        all_results = []
        for i in range(runs):
            s = start if i == 0 else random.uniform(-6, 6)
            xf, yf, hist = hill_climbing(s, step)
            all_results.append((s, xf, yf, hist))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f(x)",
                                 line=dict(color="#6C63FF", width=2)))

        colors = ["#f59e0b", "#22d3ee", "#34d399", "#f87171", "#a78bfa",
                  "#fb923c", "#8b5cf6", "#e879f9", "#4ade80", "#38bdf8"]
        best_y = -float("inf")
        best_x = None
        for i, (s, xf, yf, hist) in enumerate(all_results):
            fig.add_trace(go.Scatter(
                x=hist, y=[objective(h) for h in hist], mode="lines+markers",
                name=f"Run {i+1}", line=dict(color=colors[i % len(colors)], width=1.5),
                marker=dict(size=5),
            ))
            fig.add_trace(go.Scatter(x=[s], y=[objective(s)], mode="markers",
                                     marker=dict(size=10, color=colors[i % len(colors)], symbol="star"),
                                     showlegend=False))
            if yf > best_y:
                best_y = yf; best_x = xf

        fig.add_trace(go.Scatter(x=[best_x], y=[best_y], mode="markers",
                                 marker=dict(size=18, color="#f59e0b", symbol="star"),
                                 name="Best found"))

        fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                          height=380, legend=dict(bgcolor="#111827", font=dict(color="#94a3b8")),
                          xaxis=dict(gridcolor="#1e293b", color="#94a3b8"),
                          yaxis=dict(gridcolor="#1e293b", color="#94a3b8"),
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.info(f"🏆 Best found: x = **{best_x:.4f}**, f(x) = **{best_y:.4f}**")
        render_save_experiment_btn(
            algorithm="Hill Climbing",
            parameters={"start": start, "step_size": step, "restarts": runs},
            results={"best_x": best_x, "best_y": best_y},
            cost=float(-best_y),
            key_suffix="hc",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Simulated Annealing
# ──────────────────────────────────────────────────────────────────────────────
def simulated_annealing(start, T=100.0, cooling=0.95, max_iter=1000):
    x, T_curr = start, T
    best_x, best_y = x, objective(x)
    history_x, history_T, history_y = [x], [T_curr], [objective(x)]

    for _ in range(max_iter):
        neighbor = x + random.uniform(-1, 1)
        delta = objective(neighbor) - objective(x)
        if delta > 0 or random.random() < math.exp(delta / T_curr):
            x = neighbor
        if objective(x) > best_y:
            best_y = objective(x); best_x = x
        history_x.append(x); history_T.append(T_curr); history_y.append(objective(x))
        T_curr *= cooling
        if T_curr < 0.001:
            break

    return best_x, best_y, history_x, history_T, history_y


def simulated_annealing_demo():
    st.markdown("### 🌡️ Simulated Annealing")
    st.markdown("""
    Simulated Annealing escapes local optima by occasionally accepting **worse solutions**
    with a probability that decreases as the *temperature* cools — inspired by metallurgical annealing.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        T_init = st.slider("Initial Temperature", 10.0, 500.0, 100.0, 10.0, key="sa_T")
    with col2:
        cooling = st.slider("Cooling Rate", 0.80, 0.99, 0.95, 0.01, key="sa_cool")
    with col3:
        start = st.slider("Start position", -6.0, 6.0, -3.0, 0.1, key="sa_start")

    if st.button("▶  Run Simulated Annealing", key="sa_run2", use_container_width=True):
        bx, by, hx, hT, hy = simulated_annealing(start, T_init, cooling)

        xs = np.linspace(-7, 7, 400)
        ys_func = objective(xs)

        tab1, tab2 = st.tabs(["🗺️ Search Path", "📉 Temperature Curve"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ys_func, name="f(x)",
                                     line=dict(color="#6C63FF", width=2)))
            fig.add_trace(go.Scatter(x=hx, y=hy, mode="lines+markers",
                                     name="SA path", line=dict(color="#f59e0b", width=1.5),
                                     marker=dict(size=3, color="#f59e0b")))
            fig.add_trace(go.Scatter(x=[bx], y=[by], mode="markers",
                                     marker=dict(size=18, color="#22d3ee", symbol="star"),
                                     name="Best"))
            fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", height=350,
                              xaxis=dict(gridcolor="#1e293b", color="#94a3b8"),
                              yaxis=dict(gridcolor="#1e293b", color="#94a3b8"),
                              legend=dict(bgcolor="#111827", font=dict(color="#94a3b8")),
                              margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=list(range(len(hT))), y=hT, mode="lines",
                                      name="Temperature", line=dict(color="#ef4444", width=2)))
            fig2.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", height=300,
                               xaxis=dict(title="Iteration", gridcolor="#1e293b", color="#94a3b8"),
                               yaxis=dict(title="Temperature", gridcolor="#1e293b", color="#94a3b8"),
                               legend=dict(bgcolor="#111827", font=dict(color="#94a3b8")),
                               margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.success(f"✅ Best found: x = **{bx:.4f}**, f(x) = **{by:.4f}** after {len(hx)} iterations")
        render_save_experiment_btn(
            algorithm="Simulated Annealing",
            parameters={"T_init": T_init, "cooling": cooling, "start": start},
            results={"best_x": bx, "best_y": by, "iterations": len(hx)},
            cost=float(-by),
            key_suffix="sa",
        )


# ──────────────────────────────────────────────────────────────────────────────
# 8-Queens
# ──────────────────────────────────────────────────────────────────────────────
def count_conflicts(queens):
    n, conflicts = len(queens), 0
    for i in range(n):
        for j in range(i + 1, n):
            if queens[i] == queens[j] or abs(queens[i] - queens[j]) == abs(i - j):
                conflicts += 1
    return conflicts


def solve_8queens_sa(n=8, T=10.0, cooling=0.999, max_iter=10000):
    queens = list(range(n))
    random.shuffle(queens)
    history = [queens[:]]

    for _ in range(max_iter):
        i, j = random.sample(range(n), 2)
        queens[i], queens[j] = queens[j], queens[i]
        new_c = count_conflicts(queens)
        if new_c == 0:
            return queens, history, True
        old_c = count_conflicts(queens)
        delta = old_c - new_c
        if delta < 0 and random.random() > math.exp(delta / max(T, 0.001)):
            queens[i], queens[j] = queens[j], queens[i]
        T *= cooling
        if len(history) < 100:
            history.append(queens[:])

    return queens, history, count_conflicts(queens) == 0


def draw_queens(queens, n=8):
    z = [[0] * n for _ in range(n)]
    for col, row in enumerate(queens):
        z[row][col] = 1

    # Checkerboard
    board = np.array([[(i + j) % 2 for j in range(n)] for i in range(n)], dtype=float)
    for col, row in enumerate(queens):
        board[row][col] = 2

    fig = go.Figure(data=go.Heatmap(
        z=board,
        colorscale=[[0, "#1e293b"], [0.5, "#334155"], [1.0, "#6C63FF"]],
        showscale=False,
    ))

    for col, row in enumerate(queens):
        fig.add_annotation(x=col, y=row, text="♛", showarrow=False, font=dict(size=22, color="white"))

    conflicts = count_conflicts(queens)
    fig.update_layout(
        title=dict(text=f"Conflicts: {conflicts}", font=dict(color="#a5b4fc", size=13)),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False, autorange="reversed"),
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        height=380, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def queens_demo():
    st.markdown("### ♛ 8-Queens Solver")
    st.markdown("Solve the N-Queens problem using **Simulated Annealing**. No two queens should share row, col, or diagonal.")

    col1, col2 = st.columns(2)
    with col1:
        n = st.slider("N (board size)", 4, 12, 8, key="q_n")
    with col2:
        T_init = st.slider("Initial Temperature", 1.0, 50.0, 10.0, 1.0, key="q_T")

    if st.button("▶  Solve N-Queens", key="q_run", use_container_width=True):
        queens, history, solved = solve_8queens_sa(n, T_init)
        fig = draw_queens(queens, n)
        st.plotly_chart(fig, use_container_width=True)
        if solved:
            st.success(f"✅ Solution found! No conflicts with {n} queens.")
        else:
            st.warning(f"⚠️ Best found: {count_conflicts(queens)} conflicts remain. Try again or adjust parameters.")
        st.metric("Queens placed", n)
        st.metric("Final conflicts", count_conflicts(queens))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def show():
    st.markdown("## ⚙️ Optimization Algorithms")
    st.markdown("Explore hill climbing, simulated annealing, and combinatorial optimization.")

    tab1, tab2, tab3 = st.tabs(["🏔️ Hill Climbing", "🌡️ Simulated Annealing", "♛ 8-Queens"])
    with tab1:
        hill_climbing_demo()
    with tab2:
        simulated_annealing_demo()
    with tab3:
        queens_demo()
