import streamlit as st
import plotly.graph_objects as go
import numpy as np
import random
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from workspace.save_helpers import render_save_experiment_btn


# ──────────────────────────────────────────────────────────────────────────────
# Map Coloring with Backtracking
# ──────────────────────────────────────────────────────────────────────────────
AUSTRALIA_MAP = {
    "WA":  ["NT", "SA"],
    "NT":  ["WA", "SA", "Q"],
    "SA":  ["WA", "NT", "Q", "NSW", "V"],
    "Q":   ["NT", "SA", "NSW"],
    "NSW": ["Q", "SA", "V"],
    "V":   ["SA", "NSW"],
    "T":   [],
}

AUSTRALIA_POS = {
    "WA": (1.5, 3.5), "NT": (4, 5.5), "SA": (5, 3.5),
    "Q": (7.5, 5.5), "NSW": (8, 3.5), "V": (7.5, 2),
    "T": (8, 0.5),
}

COLORS_PALETTE = ["#ef4444", "#22d3ee", "#22c55e", "#f59e0b", "#a855f7"]


def is_valid(assignment, var, color, graph):
    for neighbor in graph[var]:
        if neighbor in assignment and assignment[neighbor] == color:
            return False
    return True


def backtrack_color(graph, variables, colors, assignment, steps):
    if len(assignment) == len(variables):
        return assignment

    var = next(v for v in variables if v not in assignment)
    for color in colors:
        if is_valid(assignment, var, color, graph):
            assignment[var] = color
            steps.append(dict(assignment))
            result = backtrack_color(graph, variables, colors, assignment, steps)
            if result:
                return result
            del assignment[var]
    return None


def draw_map(graph, pos, assignment, title="Map Coloring"):
    node_colors = {v: assignment.get(v, "#1e293b") for v in graph}
    fig = go.Figure()

    drawn = set()
    for u, neighbors in graph.items():
        for v in neighbors:
            if (v, u) not in drawn:
                drawn.add((u, v))
                x0, y0 = pos[u]; x1, y1 = pos[v]
                fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines",
                                         line=dict(width=2, color="#334155"), hoverinfo="none"))

    for node in graph:
        x, y = pos[node]
        color = node_colors[node]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=48, color=color, line=dict(width=2, color="#94a3b8")),
            text=[node], textfont=dict(color="white", size=12, family="monospace"),
            hoverinfo="text", hovertext=f"{node}: {color}",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#a5b4fc", size=13)),
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        height=380, showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def map_coloring_demo():
    st.markdown("### 🗺️ Map Coloring (CSP)")
    st.markdown("""
    The **Map Coloring** problem assigns colors to regions such that no two adjacent regions share the same color.
    Solved using **Backtracking** constraint satisfaction.
    """)

    num_colors = st.slider("Max colors", 3, 5, 3, key="mc_nc")
    animate = st.checkbox("Animate backtracking", value=True, key="mc_anim")

    if st.button("▶  Solve Map Coloring", key="mc_run", use_container_width=True):
        variables = list(AUSTRALIA_MAP.keys())
        colors = COLORS_PALETTE[:num_colors]
        steps = []
        result = backtrack_color(AUSTRALIA_MAP, variables, colors, {}, steps)

        chart_ph = st.empty()
        if animate:
            for i, step in enumerate(steps):
                chart_ph.plotly_chart(
                    draw_map(AUSTRALIA_MAP, AUSTRALIA_POS, step,
                             f"Backtracking — Step {i+1}/{len(steps)}"),
                    use_container_width=True
                )
                time.sleep(0.25)

        if result:
            chart_ph.plotly_chart(
                draw_map(AUSTRALIA_MAP, AUSTRALIA_POS, result, "✅ Solution Found"),
                use_container_width=True
            )
            st.success(f"✅ Solved with {num_colors} colors in {len(steps)} steps!")
            st.json(result)
            render_save_experiment_btn(
                algorithm="CSP Map Coloring",
                parameters={"num_colors": num_colors},
                results={"steps": len(steps), "solution": result},
                key_suffix="mc",
            )
        else:
            st.error(f"❌ Cannot color the map with only {num_colors} colors.")


# ──────────────────────────────────────────────────────────────────────────────
# N-Queens with Backtracking
# ──────────────────────────────────────────────────────────────────────────────
def solve_nqueens_bt(n):
    solutions = []
    board = [-1] * n

    def is_safe(row, col):
        for r in range(row):
            if board[r] == col or abs(board[r] - col) == abs(r - row):
                return False
        return True

    def backtrack(row):
        if row == n:
            solutions.append(board[:])
            return
        for col in range(n):
            if is_safe(row, col):
                board[row] = col
                backtrack(row + 1)
                board[row] = -1

    backtrack(0)
    return solutions


def draw_nqueens(solution, n):
    board = np.array([[(i + j) % 2 * 0.3 for j in range(n)] for i in range(n)])
    for row, col in enumerate(solution):
        board[row][col] = 1.0

    fig = go.Figure(data=go.Heatmap(
        z=board,
        colorscale=[[0, "#1e293b"], [0.3, "#334155"], [1.0, "#6C63FF"]],
        showscale=False,
    ))
    for row, col in enumerate(solution):
        fig.add_annotation(x=col, y=row, text="♛", showarrow=False,
                           font=dict(size=20, color="#f59e0b"))

    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        height=350, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False, autorange="reversed"),
    )
    return fig


def nqueens_demo():
    st.markdown("### ♛ N-Queens (Backtracking CSP)")
    n = st.slider("Board size N", 4, 10, 6, key="nq_n")

    if st.button("▶  Find All Solutions", key="nq_run", use_container_width=True):
        with st.spinner("Solving..."):
            solutions = solve_nqueens_bt(n)
        st.success(f"Found **{len(solutions)}** solutions for {n}-Queens!")

        if solutions:
            idx = st.slider("View solution #", 1, len(solutions), 1, key="nq_idx")
            st.plotly_chart(draw_nqueens(solutions[idx - 1], n), use_container_width=True)
            st.code(f"Solution {idx}: {solutions[idx-1]}")
            render_save_experiment_btn(
                algorithm="CSP N-Queens",
                parameters={"n": n},
                results={"total_solutions": len(solutions)},
                key_suffix="nq",
            )


# ──────────────────────────────────────────────────────────────────────────────
# Sudoku Solver
# ──────────────────────────────────────────────────────────────────────────────
SAMPLE_SUDOKUS = {
    "Easy": [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9],
    ],
    "Hard": [
        [0, 0, 0, 2, 6, 0, 7, 0, 1],
        [6, 8, 0, 0, 7, 0, 0, 9, 0],
        [1, 9, 0, 0, 0, 4, 5, 0, 0],
        [8, 2, 0, 1, 0, 0, 0, 4, 0],
        [0, 0, 4, 6, 0, 2, 9, 0, 0],
        [0, 5, 0, 0, 0, 3, 0, 2, 8],
        [0, 0, 9, 3, 0, 0, 0, 7, 4],
        [0, 4, 0, 0, 5, 0, 0, 3, 6],
        [7, 0, 3, 0, 1, 8, 0, 0, 0],
    ],
}


def solve_sudoku(board):
    def is_valid(b, r, c, num):
        if num in b[r]: return False
        if num in [b[i][c] for i in range(9)]: return False
        br, bc = 3 * (r // 3), 3 * (c // 3)
        if num in [b[br+i][bc+j] for i in range(3) for j in range(3)]: return False
        return True

    def backtrack(b):
        for r in range(9):
            for c in range(9):
                if b[r][c] == 0:
                    for num in range(1, 10):
                        if is_valid(b, r, c, num):
                            b[r][c] = num
                            if backtrack(b): return True
                            b[r][c] = 0
                    return False
        return True

    import copy
    b = copy.deepcopy(board)
    backtrack(b)
    return b


def draw_sudoku(original, solved):
    fig = go.Figure()

    cell_size = 1
    for r in range(9):
        for c in range(9):
            val = solved[r][c]
            orig = original[r][c]
            color = "#0f172a" if (r // 3 + c // 3) % 2 == 0 else "#1e293b"
            text_color = "#a5b4fc" if orig == 0 else "#e2e8f0"
            fig.add_shape(type="rect", x0=c, y0=8-r, x1=c+1, y1=9-r,
                          fillcolor=color, line=dict(color="#334155", width=0.5))
            if val > 0:
                fig.add_annotation(x=c + 0.5, y=8.5 - r, text=str(val), showarrow=False,
                                   font=dict(size=14, color=text_color, family="monospace"))

    # Bold lines for 3x3 boxes
    for i in [0, 3, 6, 9]:
        fig.add_shape(type="line", x0=i, y0=0, x1=i, y1=9, line=dict(color="#6C63FF", width=2))
        fig.add_shape(type="line", x0=0, y0=i, x1=9, y1=i, line=dict(color="#6C63FF", width=2))

    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        height=400, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[0, 9], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[0, 9], showgrid=False, showticklabels=False, zeroline=False),
    )
    return fig


def sudoku_demo():
    st.markdown("### 🔢 Sudoku Solver (Backtracking)")
    difficulty = st.selectbox("Puzzle difficulty", ["Easy", "Hard"], key="su_diff")

    if st.button("▶  Solve Sudoku", key="su_run", use_container_width=True):
        puzzle = SAMPLE_SUDOKUS[difficulty]
        with st.spinner("Solving with backtracking..."):
            solution = solve_sudoku(puzzle)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Puzzle**")
            st.plotly_chart(draw_sudoku(puzzle, puzzle), use_container_width=True)
        with col2:
            st.markdown("**Solution**")
            st.plotly_chart(draw_sudoku(puzzle, solution), use_container_width=True)
        st.success("✅ Sudoku solved using Backtracking CSP!")
        render_save_experiment_btn(
            algorithm="CSP Sudoku",
            parameters={"difficulty": difficulty},
            results={"solved": True},
            key_suffix="su",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def show():
    st.markdown("## 🧩 Constraint Satisfaction Problems")
    st.markdown("Solve classic CSP problems with backtracking and constraint propagation.")

    tab1, tab2, tab3 = st.tabs(["🗺️ Map Coloring", "♛ N-Queens", "🔢 Sudoku"])
    with tab1:
        map_coloring_demo()
    with tab2:
        nqueens_demo()
    with tab3:
        sudoku_demo()
