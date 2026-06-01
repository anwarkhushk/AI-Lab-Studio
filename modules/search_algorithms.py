import streamlit as st
import plotly.graph_objects as go
import networkx as nx
from collections import deque
import heapq
import time
import random
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from workspace.save_helpers import render_save_experiment_btn


# ──────────────────────────────────────────────────────────────────────────────
# Graph builder
# ──────────────────────────────────────────────────────────────────────────────
def build_demo_graph():
    G = nx.Graph()
    edges = [
        (0, 1, 1), (0, 2, 4), (1, 3, 2), (1, 4, 5),
        (2, 4, 1), (2, 5, 3), (3, 6, 3), (4, 6, 2),
        (4, 7, 4), (5, 7, 2), (6, 8, 1), (7, 8, 3), (7, 9, 5), (8, 9, 2),
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G, edges


def get_layout(G):
    pos = {
        0: (0, 2), 1: (1, 3), 2: (1, 1), 3: (2, 4),
        4: (2, 2), 5: (2, 0), 6: (3, 3), 7: (3, 1),
        8: (4, 2), 9: (5, 2),
    }
    return pos


def draw_graph(G, pos, visited=None, path=None, current=None, title="Graph"):
    visited = visited or set()
    path = path or []

    edge_x, edge_y, edge_labels = [], [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]

    node_colors = []
    for n in G.nodes():
        if n == current:
            node_colors.append("#f59e0b")
        elif n in path:
            node_colors.append("#6C63FF")
        elif n in visited:
            node_colors.append("#22d3ee")
        else:
            node_colors.append("#1e293b")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=2, color="#334155"), hoverinfo="none",
    ))

    for u, v, d in G.edges(data=True):
        mx, my = (pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2
        fig.add_annotation(x=mx, y=my, text=str(d["weight"]), showarrow=False,
                           font=dict(size=10, color="#94a3b8"), bgcolor="#0f172a")

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(size=36, color=node_colors, line=dict(width=2, color="#6C63FF")),
        text=[str(n) for n in G.nodes()],
        textfont=dict(color="white", size=13),
        hoverinfo="text",
        hovertext=[f"Node {n}" for n in G.nodes()],
    ))

    legend_items = [
        ("Unvisited", "#1e293b"), ("Visited", "#22d3ee"),
        ("Path", "#6C63FF"), ("Current", "#f59e0b"),
    ]
    for lbl, color in legend_items:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=10, color=color),
                                 name=lbl))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#a5b4fc", size=13)),
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        height=380, showlegend=True,
        legend=dict(bgcolor="#111827", font=dict(color="#94a3b8", size=11)),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# BFS
# ──────────────────────────────────────────────────────────────────────────────
def bfs(G, start, goal):
    queue, visited, parent = deque([start]), {start}, {start: None}
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        if node == goal:
            path = []
            while node is not None:
                path.append(node); node = parent[node]
            return order, path[::-1]
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor); parent[neighbor] = node; queue.append(neighbor)
    return order, []


# ──────────────────────────────────────────────────────────────────────────────
# DFS
# ──────────────────────────────────────────────────────────────────────────────
def dfs(G, start, goal):
    stack, visited, parent = [start], set(), {start: None}
    order = []
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node); order.append(node)
        if node == goal:
            path = []
            while node is not None:
                path.append(node); node = parent[node]
            return order, path[::-1]
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                if neighbor not in parent:
                    parent[neighbor] = node
                stack.append(neighbor)
    return order, []


# ──────────────────────────────────────────────────────────────────────────────
# DLS
# ──────────────────────────────────────────────────────────────────────────────
def dls(G, start, goal, limit):
    def _dls(node, depth, visited, parent):
        visited.add(node)
        order.append(node)
        if node == goal:
            return True
        if depth == 0:
            return None
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                parent[neighbor] = node
                result = _dls(neighbor, depth - 1, visited, parent)
                if result is True:
                    return True
        return None

    order, parent = [], {start: None}
    found = _dls(start, limit, set(), parent)
    if found:
        path, node = [], goal
        while node is not None:
            path.append(node); node = parent[node]
        return order, path[::-1]
    return order, []


# ──────────────────────────────────────────────────────────────────────────────
# IDS
# ──────────────────────────────────────────────────────────────────────────────
def ids(G, start, goal, max_depth=10):
    all_order = []
    for depth in range(max_depth + 1):
        order, path = dls(G, start, goal, depth)
        all_order.extend(order)
        if path:
            return all_order, path, depth
    return all_order, [], max_depth


# ──────────────────────────────────────────────────────────────────────────────
# A*
# ──────────────────────────────────────────────────────────────────────────────
def astar(G, start, goal, pos):
    def h(n):
        x1, y1 = pos[n]; x2, y2 = pos[goal]
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    heap, visited, g_cost, parent = [(h(start), 0, start)], set(), {start: 0}, {start: None}
    order = []
    while heap:
        f, g, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node); order.append(node)
        if node == goal:
            path, curr = [], goal
            while curr is not None:
                path.append(curr); curr = parent[curr]
            return order, path[::-1]
        for neighbor in G.neighbors(node):
            ng = g + G[node][neighbor]["weight"]
            if neighbor not in visited and ng < g_cost.get(neighbor, float("inf")):
                g_cost[neighbor] = ng; parent[neighbor] = node
                heapq.heappush(heap, (ng + h(neighbor), ng, neighbor))
    return order, []


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def show():
    st.markdown("## 🔍 Search Algorithm Visualizer")
    st.markdown("Visualize how different search algorithms traverse a weighted graph.")

    G, _ = build_demo_graph()
    pos = get_layout(G)

    col1, col2, col3 = st.columns(3)
    with col1:
        algo = st.selectbox("Algorithm", ["BFS", "DFS", "DLS", "IDS", "A*"], key="sa_algo")
    with col2:
        start_node = st.selectbox("Start Node", list(G.nodes()), key="sa_start")
    with col3:
        goal_node = st.selectbox("Goal Node", list(G.nodes()), index=9, key="sa_goal")

    extra = {}
    if algo == "DLS":
        extra["limit"] = st.slider("Depth Limit", 1, 10, 3, key="sa_limit")
    if algo == "IDS":
        extra["max_depth"] = st.slider("Max Depth", 1, 15, 10, key="sa_maxd")

    animate = st.checkbox("🎬 Animate traversal", value=True, key="sa_anim")
    speed = st.slider("Animation speed (steps/sec)", 1, 5, 2, key="sa_speed") if animate else 2

    if st.button("▶  Run Algorithm", key="sa_run", use_container_width=True):
        # Run algorithm
        if algo == "BFS":
            order, path = bfs(G, start_node, goal_node)
            ids_depth = None
        elif algo == "DFS":
            order, path = dfs(G, start_node, goal_node)
            ids_depth = None
        elif algo == "DLS":
            order, path = dls(G, start_node, goal_node, extra["limit"])
            ids_depth = None
        elif algo == "IDS":
            order, path, ids_depth = ids(G, start_node, goal_node, extra.get("max_depth", 10))
        else:
            order, path = astar(G, start_node, goal_node, pos)
            ids_depth = None

        chart_ph = st.empty()
        stats_ph = st.empty()

        visited_so_far = set()
        if animate:
            for i, node in enumerate(order):
                visited_so_far.add(node)
                fig = draw_graph(G, pos, visited_so_far, path if node == goal_node else [],
                                 current=node, title=f"{algo} — Step {i+1}/{len(order)}")
                chart_ph.plotly_chart(fig, use_container_width=True)
                time.sleep(1 / speed)
        else:
            visited_so_far = set(order)

        # Final state
        fig = draw_graph(G, pos, set(order), path, title=f"{algo} — Final")
        chart_ph.plotly_chart(fig, use_container_width=True)

        # Stats
        with stats_ph.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Nodes Explored", len(order))
            c2.metric("Path Length", len(path))
            cost = sum(G[path[i]][path[i+1]]["weight"] for i in range(len(path)-1)) if len(path) > 1 else 0
            c3.metric("Path Cost", cost)
            if ids_depth is not None:
                c4.metric("Iterations (IDS)", ids_depth + 1)
            else:
                c4.metric("Algorithm", algo)

            if path:
                st.success(f"✅ Path found: {' → '.join(str(n) for n in path)}")
            else:
                st.error("❌ No path found.")

        # ── Save experiment ──────────────────────────────────────────────
        _sa_cost = sum(G[path[i]][path[i+1]]["weight"] for i in range(len(path)-1)) if len(path) > 1 else 0
        render_save_experiment_btn(
            algorithm=algo,
            parameters={"start": start_node, "goal": goal_node, **extra},
            results={"order": order, "path": path},
            cost=float(_sa_cost),
            path_length=len(path),
            key_suffix="search",
        )

    # Always show graph
    st.markdown("#### Graph Structure")
    st.plotly_chart(draw_graph(G, pos, title="Demo Graph (weighted)"), use_container_width=True)
