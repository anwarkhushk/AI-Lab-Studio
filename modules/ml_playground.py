"""
ML Playground — AI Lab Studio
Fixes: invalid colorscale 'Vivid', key-collision in elbow dict-comprehension.
New:   Dataset Upload tab (CSV → any ML model).
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_blobs, make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Safe discrete color palette (no Vivid — use explicit hex list instead)
PALETTE = [
    "#6C63FF", "#22d3ee", "#34d399", "#f59e0b",
    "#f87171", "#a78bfa", "#fb923c", "#38bdf8",
    "#4ade80", "#e879f9", "#facc15", "#60a5fa",
]

DARK_LAYOUT = dict(
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    legend=dict(bgcolor="#111827", font=dict(color="#94a3b8")),
    xaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
    yaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
    margin=dict(l=10, r=10, t=40, b=10),
)

# Safe continuous colorscales (validated against Plotly)
CONT_COLORSCALE = "Plasma"   # fallback: Viridis, Turbo, Plasma, Inferno, Cividis


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def safe_colorscale(name: str) -> str:
    """Return `name` if valid, else fall back to 'Viridis'."""
    valid = {"viridis", "turbo", "plasma", "inferno", "cividis", "magma",
             "blues", "reds", "greens", "purples", "oranges",
             "rdbu", "picnic", "rainbow", "jet", "hot", "blackbody"}
    return name if name.lower() in valid else "Viridis"


def dark_layout(**overrides) -> dict:
    """Return DARK_LAYOUT merged with any axis overrides."""
    layout = {
        "paper_bgcolor": "#0f172a",
        "plot_bgcolor": "#0f172a",
        "legend": dict(bgcolor="#111827", font=dict(color="#94a3b8")),
        "margin": dict(l=10, r=10, t=40, b=10),
    }
    layout.update(overrides)
    return layout


def draw_decision_boundary(model, X_scaled, y, scaler=None,
                           title="Decision Boundary", h=0.08):
    """
    Plot a filled decision-boundary contour plus scatter data points.
    Uses a safe discrete PALETTE for class colours — no 'Vivid' colorscale.
    """
    try:
        # Determine display coordinates
        X_disp = scaler.inverse_transform(X_scaled) if scaler else X_scaled

        x_min, x_max = X_disp[:, 0].min() - 1, X_disp[:, 0].max() + 1
        y_min, y_max = X_disp[:, 1].min() - 1, X_disp[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        grid_disp = np.c_[xx.ravel(), yy.ravel()]
        grid_scaled = scaler.transform(grid_disp) if scaler else grid_disp
        Z = model.predict(grid_scaled).reshape(xx.shape)

        classes = sorted(np.unique(y))
        n_classes = len(classes)

        # Build a piecewise colorscale from PALETTE for the contour
        contour_cs = []
        for i, cls in enumerate(classes):
            frac = i / max(n_classes - 1, 1)
            contour_cs.append([frac, PALETTE[i % len(PALETTE)] + "80"])  # 50% opacity hex

        fig = go.Figure()

        # Background decision zones
        fig.add_trace(go.Contour(
            x=np.arange(x_min, x_max, h),
            y=np.arange(y_min, y_max, h),
            z=Z,
            showscale=False,
            colorscale=[[0, "#1e1b4b"], [0.5, "#164e63"], [1.0, "#14532d"]],
            opacity=0.45,
            contours=dict(showlines=False),
        ))

        # Scatter points — one trace per class using discrete colours
        for i, cls in enumerate(classes):
            mask = y == cls
            fig.add_trace(go.Scatter(
                x=X_disp[mask, 0], y=X_disp[mask, 1],
                mode="markers",
                name=f"Class {cls}",
                marker=dict(
                    size=8,
                    color=PALETTE[i % len(PALETTE)],
                    line=dict(width=1, color="white"),
                ),
                hovertemplate=f"x=%{{x:.2f}}, y=%{{y:.2f}}<extra>Class {cls}</extra>",
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(color="#a5b4fc", size=13)),
            height=420,
            **dark_layout(
                xaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
                yaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
            ),
        )
        return fig

    except Exception as exc:
        # Fallback: plain scatter
        fig = go.Figure()
        classes = sorted(np.unique(y))
        X_disp = scaler.inverse_transform(X_scaled) if scaler else X_scaled
        for i, cls in enumerate(classes):
            mask = y == cls
            fig.add_trace(go.Scatter(
                x=X_disp[mask, 0], y=X_disp[mask, 1],
                mode="markers", name=f"Class {cls}",
                marker=dict(size=7, color=PALETTE[i % len(PALETTE)]),
            ))
        fig.update_layout(title=dict(text=f"{title} (boundary unavailable: {exc})",
                                     font=dict(color="#f87171", size=12)),
                          height=400, **dark_layout(
                              xaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
                              yaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
                          ))
        return fig


def draw_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale=safe_colorscale("Purples"),
        x=[f"Pred {l}" for l in labels],
        y=[f"True {l}" for l in labels],
        title=title,
    )
    fig.update_layout(height=330, **dark_layout(
        xaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
        yaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loader (built-in synthetic)
# ──────────────────────────────────────────────────────────────────────────────

def load_builtin(dataset_name: str, n_samples: int, seed: int = 42):
    if dataset_name == "moons":
        return make_moons(n_samples=n_samples, noise=0.28, random_state=seed)
    elif dataset_name == "blobs":
        return make_blobs(n_samples=n_samples, centers=3, random_state=seed)
    else:
        return make_classification(
            n_samples=n_samples, n_features=2,
            n_redundant=0, n_clusters_per_class=1, random_state=seed,
        )


# ──────────────────────────────────────────────────────────────────────────────
# KNN
# ──────────────────────────────────────────────────────────────────────────────

def knn_demo(X=None, y=None, from_upload=False):
    st.markdown("### 🔵 K-Nearest Neighbors (KNN)")
    st.markdown("KNN classifies a new point by a **majority vote** of its K nearest neighbors.")

    col1, col2, col3 = st.columns(3)
    with col1:
        k = st.slider("K (neighbors)", 1, 15, 5, key="knn_k")
        n_samples = st.slider("# Samples (synthetic)", 100, 500, 200, 50,
                              key="knn_ns", disabled=from_upload)
    with col2:
        dataset = st.selectbox("Dataset (synthetic)", ["moons", "blobs", "classification"],
                               key="knn_ds", disabled=from_upload)
        test_size = st.slider("Test %", 10, 40, 20, key="knn_ts")
    with col3:
        metric = st.selectbox("Distance metric",
                              ["euclidean", "manhattan", "minkowski"], key="knn_met")
        weights = st.selectbox("Weights", ["uniform", "distance"], key="knn_w")

    if st.button("▶  Train KNN", key="knn_run", use_container_width=True):
        if from_upload and X is not None:
            X_data, y_data = X, y
        else:
            X_data, y_data = load_builtin(dataset, n_samples)

        # Restrict to 2D for the boundary plot
        X_2d = X_data[:, :2] if X_data.shape[1] > 2 else X_data

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_2d)
        Xtr, Xte, ytr, yte = train_test_split(
            X_scaled, y_data, test_size=test_size / 100, random_state=42, stratify=y_data
        )

        model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights)
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        acc = accuracy_score(yte, ypred)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc * 100:.2f}%")
        c2.metric("Train size", len(Xtr))
        c3.metric("Test size", len(Xte))
        c4.metric("K", k)

        st.plotly_chart(
            draw_decision_boundary(model, X_scaled, y_data, scaler,
                                   f"KNN (k={k}) Decision Boundary"),
            use_container_width=True,
        )
        st.plotly_chart(draw_confusion_matrix(yte, ypred), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Naïve Bayes
# ──────────────────────────────────────────────────────────────────────────────

def naive_bayes_demo(X=None, y=None, from_upload=False):
    st.markdown("### 📊 Naïve Bayes Classifier")
    st.markdown(
        "Naïve Bayes applies **Bayes' theorem** with the \"naïve\" assumption of "
        "feature independence. Despite simplicity, it works surprisingly well."
    )

    col1, col2 = st.columns(2)
    with col1:
        n_samples = st.slider("# Samples (synthetic)", 100, 600, 300, 50,
                              key="nb_ns", disabled=from_upload)
        dataset = st.selectbox("Dataset (synthetic)", ["moons", "blobs", "classification"],
                               key="nb_ds", disabled=from_upload)
    with col2:
        test_size = st.slider("Test %", 10, 40, 20, key="nb_ts")

    if st.button("▶  Train Naïve Bayes", key="nb_run", use_container_width=True):
        if from_upload and X is not None:
            X_data, y_data = X, y
        else:
            X_data, y_data = load_builtin(dataset, n_samples, seed=7)

        X_2d = X_data[:, :2] if X_data.shape[1] > 2 else X_data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_2d)
        Xtr, Xte, ytr, yte = train_test_split(
            X_scaled, y_data, test_size=test_size / 100, random_state=7, stratify=y_data
        )

        model = GaussianNB()
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        acc = accuracy_score(yte, ypred)

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{acc * 100:.2f}%")
        c2.metric("Classes", len(np.unique(y_data)))
        c3.metric("Features used", X_2d.shape[1])

        st.plotly_chart(
            draw_decision_boundary(model, X_scaled, y_data, scaler,
                                   "Naïve Bayes Decision Boundary"),
            use_container_width=True,
        )

        # Class priors bar
        priors_fig = go.Figure(data=go.Bar(
            x=[f"Class {i}" for i in range(len(model.class_prior_))],
            y=model.class_prior_,
            marker_color=[PALETTE[i % len(PALETTE)] for i in range(len(model.class_prior_))],
            text=[f"{p:.3f}" for p in model.class_prior_],
            textposition="outside",
        ))
        priors_fig.update_layout(
            title=dict(text="Class Prior Probabilities P(C)",
                       font=dict(color="#a5b4fc", size=13)),
            height=280,
            **dark_layout(
                xaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
                yaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
            ),
        )
        st.plotly_chart(priors_fig, use_container_width=True)
        st.plotly_chart(draw_confusion_matrix(yte, ypred), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# K-Means
# ──────────────────────────────────────────────────────────────────────────────

def kmeans_demo(X=None, from_upload=False):
    st.markdown("### 🔵 K-Means Clustering")
    st.markdown("K-Means partitions data into **K clusters** by minimising within-cluster variance.")

    col1, col2, col3 = st.columns(3)
    with col1:
        k = st.slider("K (clusters)", 2, 8, 3, key="km_k")
        n_samples = st.slider("# Samples (synthetic)", 100, 600, 300,
                              key="km_ns", disabled=from_upload)
    with col2:
        n_centers = st.slider("True # centres (synthetic)", 2, 8, 3,
                              key="km_nc", disabled=from_upload)
        max_iter = st.slider("Max iterations", 10, 300, 100, key="km_mi")
    with col3:
        init = st.selectbox("Init method", ["k-means++", "random"], key="km_init")
        show_elbow = st.checkbox("Show Elbow Curve", True, key="km_elbow")

    if st.button("▶  Cluster Data", key="km_run", use_container_width=True):
        if from_upload and X is not None:
            X_data = X
        else:
            X_data, _ = make_blobs(n_samples=n_samples, centers=n_centers,
                                   cluster_std=1.2, random_state=42)

        X_2d = X_data[:, :2] if X_data.shape[1] > 2 else X_data

        model = KMeans(n_clusters=k, init=init, max_iter=max_iter,
                       random_state=42, n_init="auto")
        labels = model.fit_predict(X_2d)
        centers = model.cluster_centers_
        sil = silhouette_score(X_2d, labels) if k > 1 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Silhouette Score", f"{sil:.4f}")
        c2.metric("Inertia", f"{model.inertia_:.2f}")
        c3.metric("Iterations", model.n_iter_)
        c4.metric("K", k)

        # Cluster scatter — discrete colours, no broken colorscale
        fig = go.Figure()
        for ki in range(k):
            mask = labels == ki
            fig.add_trace(go.Scatter(
                x=X_2d[mask, 0], y=X_2d[mask, 1],
                mode="markers", name=f"Cluster {ki}",
                marker=dict(size=7, color=PALETTE[ki % len(PALETTE)]),
            ))
        fig.add_trace(go.Scatter(
            x=centers[:, 0], y=centers[:, 1],
            mode="markers", name="Centroids",
            marker=dict(size=18, symbol="x", color="white",
                        line=dict(width=2, color="white")),
        ))
        fig.update_layout(
            title=dict(text=f"K-Means (k={k})", font=dict(color="#a5b4fc", size=13)),
            height=420,
            **dark_layout(
                xaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
                yaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        if show_elbow:
            inertias, ks_list = [], list(range(2, min(10, len(X_2d) // 10 + 2)))
            for ki in ks_list:
                km = KMeans(n_clusters=ki, random_state=42, n_init="auto")
                km.fit(X_2d)
                inertias.append(km.inertia_)

            # FIX: use explicit xaxis_title/yaxis_title, avoid dict-key collision
            elbow_fig = go.Figure(go.Scatter(
                x=ks_list, y=inertias, mode="lines+markers",
                line=dict(color="#6C63FF", width=2),
                marker=dict(size=8, color="#22d3ee"),
            ))
            elbow_fig.update_layout(
                title=dict(text="Elbow Curve (Inertia vs K)",
                           font=dict(color="#a5b4fc", size=13)),
                xaxis_title="K",
                yaxis_title="Inertia",
                height=300,
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                xaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
                yaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
                legend=dict(bgcolor="#111827", font=dict(color="#94a3b8")),
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(elbow_fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Logistic Regression
# ──────────────────────────────────────────────────────────────────────────────

def logistic_regression_demo(X=None, y=None, from_upload=False):
    st.markdown("### 📈 Logistic Regression")
    st.markdown(
        "Logistic Regression models class membership probability using the **sigmoid function**, "
        "ideal for binary and multi-class classification."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_samples = st.slider("# Samples (synthetic)", 100, 600, 300,
                              key="lr_ns", disabled=from_upload)
        dataset = st.selectbox("Dataset (synthetic)", ["moons", "blobs", "classification"],
                               key="lr_ds", disabled=from_upload)
    with col2:
        C = st.slider("Regularization C", 0.01, 10.0, 1.0, 0.01, key="lr_C")
        test_size = st.slider("Test %", 10, 40, 20, key="lr_ts")
    with col3:
        solver = st.selectbox("Solver", ["lbfgs", "saga", "liblinear"], key="lr_solver")

    if st.button("▶  Train Logistic Regression", key="lr_run", use_container_width=True):
        if from_upload and X is not None:
            X_data, y_data = X, y
        else:
            X_data, y_data = load_builtin(dataset, n_samples, seed=0)

        X_2d = X_data[:, :2] if X_data.shape[1] > 2 else X_data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_2d)
        Xtr, Xte, ytr, yte = train_test_split(
            X_scaled, y_data, test_size=test_size / 100, random_state=0, stratify=y_data
        )

        model = LogisticRegression(C=C, solver=solver, max_iter=1000)
        model.fit(Xtr, ytr)
        ypred = model.predict(Xte)
        acc = accuracy_score(yte, ypred)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc * 100:.2f}%")
        c2.metric("Train size", len(Xtr))
        c3.metric("Test size", len(Xte))
        c4.metric("C", C)

        st.plotly_chart(
            draw_decision_boundary(model, X_scaled, y_data, scaler,
                                   f"Logistic Regression Boundary (C={C})"),
            use_container_width=True,
        )

        # Sigmoid demo
        z_vals = np.linspace(-6, 6, 200)
        sig_fig = go.Figure(go.Scatter(
            x=z_vals, y=1 / (1 + np.exp(-z_vals)),
            mode="lines", line=dict(color="#6C63FF", width=2), name="σ(z)",
        ))
        sig_fig.add_hline(y=0.5, line=dict(color="#ef4444", dash="dash"),
                          annotation_text="Decision threshold 0.5",
                          annotation_font_color="#ef4444")
        sig_fig.update_layout(
            title=dict(text="Sigmoid Function  σ(z) = 1 / (1 + e⁻ᶻ)",
                       font=dict(color="#a5b4fc", size=13)),
            height=280,
            **dark_layout(
                xaxis=dict(gridcolor="#1e293b", color="#94a3b8",
                           zeroline=False, title="z"),
                yaxis=dict(gridcolor="#1e293b", color="#94a3b8",
                           zeroline=False, title="σ(z)"),
            ),
        )
        st.plotly_chart(sig_fig, use_container_width=True)

        # Coefficient bar (2D features only)
        if hasattr(model, "coef_"):
            coefs = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
            if len(coefs) <= 20:
                coef_fig = go.Figure(go.Bar(
                    x=[f"Feature {i}" for i in range(len(coefs))],
                    y=coefs,
                    marker_color=["#22d3ee" if c > 0 else "#ef4444" for c in coefs],
                    text=[f"{c:.3f}" for c in coefs],
                    textposition="outside",
                ))
                coef_fig.update_layout(
                    title=dict(text="Feature Weights (Coefficients)",
                               font=dict(color="#a5b4fc", size=13)),
                    height=280,
                    **dark_layout(
                        xaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
                        yaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
                    ),
                )
                st.plotly_chart(coef_fig, use_container_width=True)

        st.plotly_chart(draw_confusion_matrix(yte, ypred), use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset Upload Module
# ──────────────────────────────────────────────────────────────────────────────

def dataset_upload_tab():
    """
    Full CSV upload workflow:
      1. Upload → preview
      2. Detect numeric / categorical columns
      3. Choose features + target
      4. Missing-value handling
      5. Feature scaling toggle
      6. Run any ML model
    """
    st.markdown("### 📂 Custom Dataset Upload")
    st.markdown(
        "Upload your own **CSV file** and apply any ML model from this playground. "
        "The system auto-detects column types and handles preprocessing for you."
    )

    uploaded = st.file_uploader(
        "Drop a CSV file here",
        type=["csv"],
        key="upload_csv",
        help="The file should have a header row. Numeric and text columns are both supported.",
    )

    if uploaded is None:
        st.info("📋 No file uploaded yet. Using a demo — upload a CSV to get started.")
        # Demo: generate a quick synthetic dataset so the UI is never empty
        X_demo, y_demo = make_classification(
            n_samples=200, n_features=4, n_redundant=0,
            n_informative=3, random_state=99,
        )
        demo_df = pd.DataFrame(X_demo, columns=[f"feature_{i}" for i in range(4)])
        demo_df["target"] = y_demo
        df = demo_df
        is_demo = True
    else:
        try:
            df = pd.read_csv(uploaded)
            is_demo = False
            st.success(f"✅ Loaded **{uploaded.name}** — {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as exc:
            st.error(f"❌ Could not parse file: {exc}")
            return

    # ── Preview ────────────────────────────────────────────────────────────────
    with st.expander("📋 Dataset Preview", expanded=True):
        st.dataframe(
            df.head(20).style.background_gradient(cmap="Blues", axis=0),
            use_container_width=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Missing values", int(df.isna().sum().sum()))

    # ── Column type detection ──────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    st.markdown("#### ⚙️ Preprocessing")
    col1, col2 = st.columns(2)

    with col1:
        # Missing values
        mv_strategy = st.selectbox(
            "Missing value strategy",
            ["Drop rows", "Fill with mean/mode", "Fill with 0"],
            key="up_mv",
        )

        # Scale
        do_scale = st.checkbox("Apply Standard Scaling", value=True, key="up_scale")

    with col2:
        # Target column
        all_cols = df.columns.tolist()
        default_target_idx = len(all_cols) - 1 if all_cols else 0
        target_col = st.selectbox(
            "Target / label column",
            all_cols,
            index=default_target_idx,
            key="up_target",
        )

        task_type = st.selectbox(
            "Task type",
            ["Classification", "Clustering (no target)"],
            key="up_task",
        )

    # Feature columns
    available_features = [c for c in numeric_cols if c != target_col]
    if not available_features:
        st.warning("⚠️ No numeric feature columns detected after excluding the target.")
        return

    feat_cols = st.multiselect(
        "Select feature columns (numeric only)",
        available_features,
        default=available_features[:min(len(available_features), 6)],
        key="up_feats",
    )

    if not feat_cols:
        st.warning("Select at least one feature column."); return

    # ── Apply preprocessing ────────────────────────────────────────────────────
    df_work = df.copy()

    # Missing values
    if mv_strategy == "Drop rows":
        df_work.dropna(subset=feat_cols + ([target_col] if task_type == "Classification" else []),
                       inplace=True)
    elif mv_strategy == "Fill with mean/mode":
        for c in feat_cols:
            df_work[c].fillna(df_work[c].mean(), inplace=True)
        if task_type == "Classification":
            df_work[target_col].fillna(df_work[target_col].mode()[0], inplace=True)
    else:
        df_work[feat_cols] = df_work[feat_cols].fillna(0)

    if len(df_work) < 10:
        st.error("Not enough rows after preprocessing (need ≥ 10)."); return

    X_raw = df_work[feat_cols].values.astype(float)

    # Encode target for classification
    if task_type == "Classification":
        le = LabelEncoder()
        y_enc = le.fit_transform(df_work[target_col].astype(str))
        n_classes = len(np.unique(y_enc))
        if n_classes < 2:
            st.error("Target column must have at least 2 distinct classes."); return
    else:
        y_enc = None

    # Scale
    scaler_up = None
    if do_scale:
        scaler_up = StandardScaler()
        X_proc = scaler_up.fit_transform(X_raw)
    else:
        X_proc = X_raw

    # ── Split ──────────────────────────────────────────────────────────────────
    test_pct = st.slider("Test split %", 10, 40, 20, key="up_ts")
    st.markdown("---")

    # ── Stats ─────────────────────────────────────────────────────────────────
    with st.expander("📊 Dataset Statistics"):
        st.dataframe(df_work[feat_cols].describe().round(4), use_container_width=True)

        # Correlation heatmap
        if len(feat_cols) >= 2:
            corr = df_work[feat_cols].corr()
            corr_fig = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale=safe_colorscale("Plasma"),
                title="Feature Correlation Matrix",
            )
            corr_fig.update_layout(
                height=max(300, len(feat_cols) * 50),
                **dark_layout(
                    xaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
                    yaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
                ),
            )
            st.plotly_chart(corr_fig, use_container_width=True)

    # ── Model selection ────────────────────────────────────────────────────────
    st.markdown("#### 🤖 Choose a Model")

    if task_type == "Classification":
        model_choice = st.selectbox(
            "Model",
            ["KNN", "Naïve Bayes", "Logistic Regression"],
            key="up_model",
        )
    else:
        model_choice = "K-Means"
        st.info("Clustering task → K-Means will be applied automatically.")

    st.markdown("---")

    if st.button("▶  Run Model on Uploaded Data", key="up_run", use_container_width=True):
        st.markdown(f"#### Results — {model_choice}")

        # ── Classification models ──────────────────────────────────────────────
        if task_type == "Classification":
            try:
                Xtr, Xte, ytr, yte = train_test_split(
                    X_proc, y_enc, test_size=test_pct / 100,
                    random_state=42, stratify=y_enc,
                )
            except ValueError:
                Xtr, Xte, ytr, yte = train_test_split(
                    X_proc, y_enc, test_size=test_pct / 100, random_state=42,
                )

            if model_choice == "KNN":
                ku = st.session_state.get("knn_k", 5)
                met = st.session_state.get("knn_met", "euclidean")
                mdl = KNeighborsClassifier(n_neighbors=ku, metric=met)
            elif model_choice == "Naïve Bayes":
                mdl = GaussianNB()
            else:
                Cu = st.session_state.get("lr_C", 1.0)
                sol = st.session_state.get("lr_solver", "lbfgs")
                mdl = LogisticRegression(C=Cu, solver=sol, max_iter=1000)

            mdl.fit(Xtr, ytr)
            ypred = mdl.predict(Xte)
            acc = accuracy_score(yte, ypred)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Accuracy", f"{acc * 100:.2f}%")
            mc2.metric("Train rows", len(Xtr))
            mc3.metric("Test rows", len(Xte))
            mc4.metric("Classes", n_classes)

            # Decision boundary (2D projection)
            if X_proc.shape[1] >= 2:
                X_2d = X_proc[:, :2]
                sc2 = StandardScaler()
                X_2d_s = sc2.fit_transform(X_2d) if not do_scale else X_2d[:, :2]
                mdl2d = type(mdl)(**mdl.get_params())
                Xtr2, Xte2, ytr2, yte2 = train_test_split(
                    X_2d_s, y_enc, test_size=test_pct / 100, random_state=42
                )
                mdl2d.fit(Xtr2, ytr2)
                st.plotly_chart(
                    draw_decision_boundary(
                        mdl2d, X_2d_s, y_enc, None,
                        f"{model_choice} — 2D projection ({feat_cols[0]} vs {feat_cols[1]})",
                    ),
                    use_container_width=True,
                )

            st.plotly_chart(draw_confusion_matrix(yte, ypred), use_container_width=True)

        # ── Clustering ────────────────────────────────────────────────────────
        else:
            ku = st.session_state.get("km_k", 3)
            mdl = KMeans(n_clusters=ku, random_state=42, n_init="auto")
            labels = mdl.fit_predict(X_proc)
            sil = silhouette_score(X_proc, labels) if ku > 1 else 0.0

            mc1, mc2 = st.columns(2)
            mc1.metric("Silhouette Score", f"{sil:.4f}")
            mc2.metric("Inertia", f"{mdl.inertia_:.2f}")

            X_2d = X_proc[:, :2]
            fig = go.Figure()
            for ki in range(ku):
                mask = labels == ki
                fig.add_trace(go.Scatter(
                    x=X_2d[mask, 0], y=X_2d[mask, 1],
                    mode="markers", name=f"Cluster {ki}",
                    marker=dict(size=7, color=PALETTE[ki % len(PALETTE)]),
                ))
            ctr = mdl.cluster_centers_[:, :2]
            fig.add_trace(go.Scatter(
                x=ctr[:, 0], y=ctr[:, 1],
                mode="markers", name="Centroids",
                marker=dict(size=16, symbol="x", color="white",
                            line=dict(width=2)),
            ))
            fig.update_layout(
                title=dict(text=f"K-Means k={ku} (2D projection)",
                           font=dict(color="#a5b4fc", size=13)),
                height=420,
                **dark_layout(
                    xaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
                    yaxis=dict(gridcolor="#1e293b", color="#94a3b8", zeroline=False),
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Surface info footer
    st.markdown(
        f"""
        <div style='background:#111827;border:1px solid #1e293b;border-radius:10px;
                    padding:14px;font-size:.78rem;color:#64748b;margin-top:12px;'>
            <b style='color:#a5b4fc;'>Loaded:</b>
            {'<i>Demo synthetic dataset</i>' if is_demo else f'<b style="color:#22d3ee;">{uploaded.name}</b>'}
            &nbsp;·&nbsp;
            <b style='color:#a5b4fc;'>Features:</b> {', '.join(feat_cols[:5])}{'…' if len(feat_cols)>5 else ''}
            &nbsp;·&nbsp;
            <b style='color:#a5b4fc;'>Target:</b>
            {'—' if task_type == 'Clustering (no target)' else target_col}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def show():
    st.markdown("## 🤖 Machine Learning Playground")
    st.markdown("Train, visualise, and compare classic ML models interactively.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔵 KNN",
        "📊 Naïve Bayes",
        "🔵 K-Means",
        "📈 Logistic Regression",
        "📂 Dataset Upload",
    ])

    with tab1:
        knn_demo()
    with tab2:
        naive_bayes_demo()
    with tab3:
        kmeans_demo()
    with tab4:
        logistic_regression_demo()
    with tab5:
        dataset_upload_tab()
