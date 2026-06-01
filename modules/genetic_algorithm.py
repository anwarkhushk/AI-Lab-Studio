import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import random
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from workspace.save_helpers import render_save_experiment_btn


# ──────────────────────────────────────────────────────────────────────────────
# Generic GA for maximising a bit-string fitness: f = count of 1s
# ──────────────────────────────────────────────────────────────────────────────
def fitness(chromosome):
    return sum(chromosome)


def selection(population, fitnesses, k=3):
    """Tournament selection."""
    tournament = random.sample(list(zip(population, fitnesses)), k)
    return max(tournament, key=lambda x: x[1])[0]


def crossover(p1, p2, rate=0.8):
    if random.random() < rate:
        pt = random.randint(1, len(p1) - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
    return p1[:], p2[:]


def mutate(chromosome, rate=0.02):
    return [1 - g if random.random() < rate else g for g in chromosome]


def run_ga(chrom_len=20, pop_size=50, generations=100,
           crossover_rate=0.8, mutation_rate=0.02):
    population = [[random.randint(0, 1) for _ in range(chrom_len)]
                  for _ in range(pop_size)]

    best_per_gen, avg_per_gen, best_chrom = [], [], None

    for gen in range(generations):
        fitnesses = [fitness(c) for c in population]
        best_idx = np.argmax(fitnesses)
        best_per_gen.append(fitnesses[best_idx])
        avg_per_gen.append(np.mean(fitnesses))
        best_chrom = population[best_idx][:]

        if fitnesses[best_idx] == chrom_len:
            break  # Perfect solution

        new_pop = []
        while len(new_pop) < pop_size:
            p1 = selection(population, fitnesses)
            p2 = selection(population, fitnesses)
            c1, c2 = crossover(p1, p2, crossover_rate)
            new_pop.append(mutate(c1, mutation_rate))
            if len(new_pop) < pop_size:
                new_pop.append(mutate(c2, mutation_rate))
        population = new_pop

    return best_per_gen, avg_per_gen, best_chrom


# ──────────────────────────────────────────────────────────────────────────────
# Draw chromosome
# ──────────────────────────────────────────────────────────────────────────────
def draw_chromosome(chrom, title="Best Chromosome"):
    n = len(chrom)
    colors = ["#6C63FF" if g == 1 else "#1e293b" for g in chrom]
    fig = go.Figure(data=go.Bar(
        x=list(range(n)), y=[1] * n,
        marker_color=colors,
        text=[str(g) for g in chrom],
        textposition="inside",
        textfont=dict(color="white", size=12),
        hovertemplate="Gene %{x}: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#a5b4fc", size=13)),
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        xaxis=dict(showgrid=False, title="Gene Index", color="#94a3b8"),
        yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1.5]),
        height=180, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Fitness curve
# ──────────────────────────────────────────────────────────────────────────────
def draw_fitness(best_per_gen, avg_per_gen):
    gens = list(range(1, len(best_per_gen) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=best_per_gen, mode="lines",
                             name="Best Fitness", line=dict(color="#6C63FF", width=2)))
    fig.add_trace(go.Scatter(x=gens, y=avg_per_gen, mode="lines",
                             name="Avg Fitness", line=dict(color="#22d3ee", width=1.5, dash="dot")))
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        xaxis=dict(title="Generation", gridcolor="#1e293b", color="#94a3b8"),
        yaxis=dict(title="Fitness (# of 1s)", gridcolor="#1e293b", color="#94a3b8"),
        legend=dict(bgcolor="#111827", font=dict(color="#94a3b8")),
        height=320, margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Crossover visualizer
# ──────────────────────────────────────────────────────────────────────────────
def crossover_visualizer():
    st.markdown("#### ✂️ Crossover Visualizer")
    n = 12
    col1, col2, col3 = st.columns(3)
    with col1:
        p1_str = st.text_input("Parent 1 (0/1, length 12)", "110110001101", key="ga_p1")
    with col2:
        p2_str = st.text_input("Parent 2 (0/1, length 12)", "001001110010", key="ga_p2")
    with col3:
        pt = st.slider("Crossover point", 1, 11, 6, key="ga_pt")

    if st.button("🔀 Crossover", key="ga_cross", use_container_width=True):
        try:
            p1 = [int(c) for c in p1_str if c in "01"][:n]
            p2 = [int(c) for c in p2_str if c in "01"][:n]
            if len(p1) < n or len(p2) < n:
                st.error("Parents must have at least 12 bits."); return
        except Exception:
            st.error("Invalid input."); return

        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]

        for label, chrom in [("Parent 1", p1), ("Parent 2", p2),
                              ("Child 1", c1), ("Child 2", c2)]:
            colors = []
            for i, g in enumerate(chrom):
                if label.startswith("Child"):
                    if (label == "Child 1" and i < pt) or (label == "Child 2" and i >= pt):
                        colors.append("#6C63FF" if g == 1 else "#312e81")
                    else:
                        colors.append("#22d3ee" if g == 1 else "#164e63")
                else:
                    colors.append("#6C63FF" if g == 1 else "#1e293b")

            fig = go.Figure(data=go.Bar(
                x=list(range(n)), y=[1] * n,
                marker_color=colors,
                text=[str(g) for g in chrom],
                textposition="inside",
                textfont=dict(color="white", size=13),
            ))
            fig.update_layout(
                title=dict(text=f"{label} (fitness={sum(chrom)})", font=dict(color="#a5b4fc", size=12)),
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                xaxis=dict(showgrid=False, color="#94a3b8"),
                yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1.5]),
                height=130, margin=dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def show():
    st.markdown("## 🧬 Genetic Algorithm Simulator")
    st.markdown("Watch evolution in action: populations of binary chromosomes evolve over generations to maximise fitness.")

    tab1, tab2 = st.tabs(["🚀 GA Runner", "✂️ Crossover & Mutation Demo"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            chrom_len = st.slider("Chromosome Length", 8, 40, 20, key="ga_cl")
            pop_size = st.slider("Population Size", 10, 200, 50, key="ga_ps")
        with col2:
            generations = st.slider("Max Generations", 20, 500, 100, key="ga_gen")
            crossover_rate = st.slider("Crossover Rate", 0.5, 1.0, 0.8, 0.05, key="ga_cr")
        with col3:
            mutation_rate = st.slider("Mutation Rate", 0.001, 0.1, 0.02, 0.001,
                                      format="%.3f", key="ga_mr")
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("▶  Evolve!", key="ga_run", use_container_width=True)

        if run_btn:
            with st.spinner("Evolving population..."):
                best_per_gen, avg_per_gen, best_chrom = run_ga(
                    chrom_len, pop_size, generations, crossover_rate, mutation_rate
                )

            st.plotly_chart(draw_fitness(best_per_gen, avg_per_gen), use_container_width=True)
            st.plotly_chart(draw_chromosome(best_chrom,
                            f"Best Chromosome (fitness = {sum(best_chrom)}/{chrom_len})"),
                            use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Generations run", len(best_per_gen))
            c2.metric("Best fitness", best_per_gen[-1])
            c3.metric("Avg fitness (final)", f"{avg_per_gen[-1]:.2f}")
            c4.metric("Fitness %", f"{100*best_per_gen[-1]/chrom_len:.1f}%")

            if sum(best_chrom) == chrom_len:
                st.success("🏆 Perfect solution found — all genes are 1!")
            render_save_experiment_btn(
                algorithm="Genetic Algorithm",
                parameters={"chrom_len": chrom_len, "pop_size": pop_size,
                            "generations": generations, "crossover_rate": crossover_rate,
                            "mutation_rate": mutation_rate},
                results={"best_fitness": best_per_gen[-1], "generations_run": len(best_per_gen)},
                accuracy=best_per_gen[-1] / chrom_len,
                key_suffix="ga",
            )

    with tab2:
        crossover_visualizer()
        st.markdown("---")
        st.markdown("#### 🎲 Mutation Demo")
        mr_demo = st.slider("Mutation rate for demo", 0.0, 0.5, 0.1, 0.01, key="ga_mr2")
        if st.button("Mutate random chromosome", key="ga_mut"):
            chrom = [random.randint(0, 1) for _ in range(20)]
            mutated = mutate(chrom, mr_demo)
            st.plotly_chart(draw_chromosome(chrom, f"Original (fitness={sum(chrom)})"),
                            use_container_width=True)
            st.plotly_chart(draw_chromosome(mutated, f"After Mutation (fitness={sum(mutated)}, rate={mr_demo})"),
                            use_container_width=True)
