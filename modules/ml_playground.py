"""
ML Playground – AI Lab Studio module
Provides interactive demos for:
  • KNN Classification
  • Naïve Bayes Classification
  • K-Means Clustering
  • Logistic Regression

Each tab supports:
  • Built-in synthetic datasets  (original behaviour, unchanged)
  • Uploaded CSV / TXT / XLSX datasets (new)
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_classification, make_blobs, make_moons
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from workspace.save_helpers import render_save_experiment_btn, render_save_model_btn
from workspace.ml_dataset import get_ml_dataset

# ── Colour palette ──────────────────────────────────────────────────────────────
COLORS = ["#6C63FF", "#22d3ee", "#34d399", "#f59e0b", "#f87171", "#a78bfa"]
BG     = "#0f172a"
CARD   = "#111827"
BORDER = "#1e293b"


# ── Internal helpers (unchanged) ────────────────────────────────────────────────

def _make_mesh(X, pad=0.5, steps=120):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, steps),
        np.linspace(y_min, y_max, steps),
    )
    return xx, yy


def _boundary_trace(xx, yy, Z, colorscale="Viridis", opacity=0.35):
    return go.Contour(
        x=xx[0], y=yy[:, 0], z=Z,
        colorscale=colorscale, showscale=False,
        opacity=opacity, contours=dict(showlines=False),
        hoverinfo="none",
    )


def _scatter_trace(X, y, labels=None):
    traces = []
    for i, cls in enumerate(np.unique(y)):
        mask = y == cls
        lbl  = labels[cls] if labels else f"Class {cls}"
        traces.append(go.Scatter(
            x=X[mask, 0], y=X[mask, 1], mode="markers", name=lbl,
            marker=dict(size=8, color=COLORS[i % len(COLORS)],
                        line=dict(width=1, color="white")),
        ))
    return traces


def _base_layout(title=""):
    return dict(
        paper_bgcolor=BG, plot_bgcolor=BG,
        title=dict(text=title, font=dict(color="#94a3b8", size=14)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        xaxis=dict(showgrid=False, zeroline=False, color="#475569"),
        yaxis=dict(showgrid=False, zeroline=False, color="#475569"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=400,
        font=dict(family="Inter, sans-serif"),
    )


def _get_dataset(name: str, n_samples: int, noise: float):
    if name == "Linear":
        X, y = make_classification(
            n_samples=n_samples, n_features=2, n_redundant=0,
            n_informative=2, random_state=42, class_sep=1.5 + noise,
        )
    elif name == "Moons":
        X, y = make_moons(n_samples=n_samples, noise=noise * 0.6, random_state=42)
    elif name == "Blobs":
        X, y = make_blobs(n_samples=n_samples, centers=3,
                          cluster_std=0.8 + noise, random_state=42)
        y = y % 2
    else:
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=n_samples, noise=noise * 0.3,
                            factor=0.5, random_state=42)
    return X, y


def _confusion_chart(cm, scale, height=300):
    fig = px.imshow(cm, text_auto=True, color_continuous_scale=scale,
                    labels=dict(x="Predicted", y="Actual"))
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                      font=dict(color="#94a3b8"), height=height)
    return fig


def _uploaded_scatter(X, labels, title):
    """Scatter for the first two features of an uploaded dataset."""
    fig = go.Figure()
    for i, lbl in enumerate(np.unique(labels)):
        mask = labels == lbl
        fig.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1], mode="markers",
            name=f"Cluster {lbl}" if isinstance(lbl, (int, np.integer)) else str(lbl),
            marker=dict(size=7, color=COLORS[i % len(COLORS)],
                        line=dict(width=0.5, color="white")),
        ))
    fig.update_layout(**_base_layout(title))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# KNN Tab
# ══════════════════════════════════════════════════════════════════════════════

def _knn_tab():
    st.markdown("### 🔵 K-Nearest Neighbours Classifier")

    # ── Dataset source ────────────────────────────────────────────────────────
    result = get_ml_dataset("knn", classifier=True)

    if result is None:
        # Built-in synthetic mode (original behaviour)
        c1, c2, c3, c4 = st.columns(4)
        k     = c1.slider("K (neighbours)", 1, 25, 5, key="knn_k")
        n     = c2.slider("Samples", 100, 600, 300, step=50, key="knn_n")
        noise = c3.slider("Noise", 0.0, 1.0, 0.3, step=0.05, key="knn_noise")
        ds    = c4.selectbox("Dataset", ["Moons", "Linear", "Blobs", "Circles"], key="knn_ds")

        X, y = _get_dataset(ds, n, noise)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        k_val = k
        clf = KNeighborsClassifier(n_neighbors=k_val)
        clf.fit(X_train_s, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_s))

        X_s = scaler.transform(X)
        xx, yy = _make_mesh(X_s)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig = go.Figure()
        fig.add_trace(_boundary_trace(xx, yy, Z.astype(float), "Blues", 0.30))
        for tr in _scatter_trace(X_s, y):
            fig.add_trace(tr)
        fig.update_layout(**_base_layout(f"KNN (k={k_val}) – Accuracy {acc:.2%}"))
        st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc:.2%}")
        m2.metric("Train samples", len(X_train))
        m3.metric("Test samples", len(X_test))
        with st.expander("📊 Confusion matrix"):
            st.plotly_chart(_confusion_chart(
                confusion_matrix(y_test, clf.predict(X_test_s)), "Viridis"), use_container_width=True)

        render_save_experiment_btn(
            algorithm="KNN",
            parameters={"k": k_val, "dataset": ds, "samples": n, "noise": noise},
            results={"accuracy": acc}, accuracy=acc, key_suffix="knn",
        )
        render_save_model_btn(clf, algorithm="KNN", accuracy=acc, key_suffix="knn")

    else:
        # ── Uploaded dataset mode ────────────────────────────────────────────
        X_train, X_test, y_train, y_test, df, features, target, ds_name = result

        k_val = st.slider("K (neighbours)", 1, 25, 5, key="knn_k_up")
        clf = KNeighborsClassifier(n_neighbors=k_val)
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))

        st.plotly_chart(
            _uploaded_scatter(X_train, y_train,
                              f"KNN (k={k_val}) Training Data – Accuracy {acc:.2%}"),
            use_container_width=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc:.2%}")
        m2.metric("Train samples", len(X_train))
        m3.metric("Test samples", len(X_test))
        with st.expander("📊 Confusion matrix"):
            st.plotly_chart(_confusion_chart(
                confusion_matrix(y_test, clf.predict(X_test)), "Viridis"), use_container_width=True)

        render_save_experiment_btn(
            algorithm="KNN",
            parameters={"k": k_val, "dataset": ds_name, "features": features, "target": target},
            results={"accuracy": acc}, accuracy=acc, key_suffix="knn_up",
        )
        render_save_model_btn(clf, algorithm="KNN", accuracy=acc, key_suffix="knn_up")


# ══════════════════════════════════════════════════════════════════════════════
# Naïve Bayes Tab
# ══════════════════════════════════════════════════════════════════════════════

def _nb_tab():
    st.markdown("### 🟢 Gaussian Naïve Bayes Classifier")

    result = get_ml_dataset("nb", classifier=True)

    if result is None:
        c1, c2, c3 = st.columns(3)
        n     = c1.slider("Samples", 100, 600, 300, step=50, key="nb_n")
        noise = c2.slider("Noise", 0.0, 1.0, 0.3, step=0.05, key="nb_noise")
        ds    = c3.selectbox("Dataset", ["Linear", "Moons", "Blobs", "Circles"], key="nb_ds")

        X, y = _get_dataset(ds, n, noise)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))

        xx, yy = _make_mesh(X)
        Z_proba = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        fig = go.Figure()
        fig.add_trace(_boundary_trace(xx, yy, Z_proba, "Teal", 0.35))
        for tr in _scatter_trace(X, y):
            fig.add_trace(tr)
        fig.update_layout(**_base_layout(f"Naïve Bayes – Accuracy {acc:.2%}"))
        st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc:.2%}")
        m2.metric("Classes", len(np.unique(y)))
        m3.metric("Features", X.shape[1])
        with st.expander("📋 Classification report"):
            report = classification_report(y_test, clf.predict(X_test), output_dict=True)
            st.dataframe(pd.DataFrame(report).T.style.background_gradient(cmap="Blues", axis=None))

        render_save_experiment_btn(
            algorithm="Naive Bayes",
            parameters={"dataset": ds, "samples": n, "noise": noise},
            results={"accuracy": acc}, accuracy=acc, key_suffix="nb",
        )
        render_save_model_btn(clf, algorithm="Naive Bayes", accuracy=acc, key_suffix="nb")

    else:
        X_train, X_test, y_train, y_test, df, features, target, ds_name = result
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))

        st.plotly_chart(
            _uploaded_scatter(X_train, y_train, f"Naïve Bayes Training Data – Accuracy {acc:.2%}"),
            use_container_width=True,
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc:.2%}")
        m2.metric("Classes", len(np.unique(y_train)))
        m3.metric("Features", len(features))
        with st.expander("📋 Classification report"):
            report = classification_report(y_test, clf.predict(X_test), output_dict=True)
            st.dataframe(pd.DataFrame(report).T.style.background_gradient(cmap="Blues", axis=None))

        render_save_experiment_btn(
            algorithm="Naive Bayes",
            parameters={"dataset": ds_name, "features": features, "target": target},
            results={"accuracy": acc}, accuracy=acc, key_suffix="nb_up",
        )
        render_save_model_btn(clf, algorithm="Naive Bayes", accuracy=acc, key_suffix="nb_up")


# ══════════════════════════════════════════════════════════════════════════════
# K-Means Tab
# ══════════════════════════════════════════════════════════════════════════════

def _kmeans_tab():
    st.markdown("### 🟡 K-Means Clustering")

    result = get_ml_dataset("km", classifier=False)

    if result is None:
        # Synthetic mode (unchanged)
        c1, c2, c3, c4 = st.columns(4)
        k       = c1.slider("K (clusters)", 2, 8, 3, key="km_k")
        n       = c2.slider("Samples", 100, 600, 300, step=50, key="km_n")
        centers = c3.slider("True centres", 2, 8, 4, key="km_centers")
        std     = c4.slider("Cluster std", 0.3, 2.5, 1.0, step=0.1, key="km_std")

        X, _ = make_blobs(n_samples=n, centers=centers, cluster_std=std, random_state=42)
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        centroids = km.cluster_centers_

        fig = go.Figure()
        for i in range(k):
            mask = labels == i
            fig.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, 1], mode="markers", name=f"Cluster {i}",
                marker=dict(size=7, color=COLORS[i % len(COLORS)],
                            line=dict(width=0.5, color="white")),
            ))
        fig.add_trace(go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1], mode="markers", name="Centroids",
            marker=dict(size=16, symbol="star", color="white",
                        line=dict(width=2, color="#6C63FF")),
        ))
        fig.update_layout(**_base_layout(f"K-Means (k={k}) – {n} points"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📉 Elbow Curve")
        inertias = [KMeans(n_clusters=ki, random_state=42, n_init="auto").fit(X).inertia_
                    for ki in range(1, 11)]
        fig2 = go.Figure(go.Scatter(x=list(range(1, 11)), y=inertias,
                                    mode="lines+markers",
                                    line=dict(color="#6C63FF", width=2),
                                    marker=dict(size=8, color="#22d3ee")))
        fig2.add_vline(x=k, line=dict(color="#f59e0b", dash="dash", width=2))
        fig2.update_layout(**{**_base_layout("Inertia vs K"), "height": 260,
                               "xaxis": dict(title="K", color="#475569", showgrid=False, zeroline=False),
                               "yaxis": dict(title="Inertia", color="#475569", showgrid=False, zeroline=False)})
        st.plotly_chart(fig2, use_container_width=True)

        render_save_experiment_btn(
            algorithm="K-Means",
            parameters={"k": k, "samples": n, "centers": centers, "std": std},
            results={"inertia": km.inertia_}, key_suffix="km",
        )
        render_save_model_btn(km, algorithm="K-Means", key_suffix="km")

    else:
        X_train, _, _, _, df, features, _, ds_name = result
        X = X_train  # all data for clustering

        k = st.slider("K (clusters)", 2, 10, 3, key="km_k_up")
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X)
        centroids = km.cluster_centers_

        fig = go.Figure()
        for i in range(k):
            mask = labels == i
            fig.add_trace(go.Scatter(
                x=X[mask, 0], y=X[mask, 1], mode="markers", name=f"Cluster {i}",
                marker=dict(size=7, color=COLORS[i % len(COLORS)],
                            line=dict(width=0.5, color="white")),
            ))
        fig.add_trace(go.Scatter(
            x=centroids[:, 0], y=centroids[:, 1], mode="markers", name="Centroids",
            marker=dict(size=16, symbol="star", color="white",
                        line=dict(width=2, color="#6C63FF")),
        ))
        fig.update_layout(**_base_layout(f"K-Means (k={k}) – {ds_name}"))
        st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Clusters (k)", k)
        m2.metric("Inertia", f"{km.inertia_:.2f}")
        m3.metric("Samples", len(X))

        st.markdown("#### 📉 Elbow Curve")
        max_k = min(10, len(X) - 1)
        inertias = [KMeans(n_clusters=ki, random_state=42, n_init="auto").fit(X).inertia_
                    for ki in range(1, max_k + 1)]
        fig2 = go.Figure(go.Scatter(x=list(range(1, max_k + 1)), y=inertias,
                                    mode="lines+markers",
                                    line=dict(color="#6C63FF", width=2),
                                    marker=dict(size=8, color="#22d3ee")))
        fig2.add_vline(x=k, line=dict(color="#f59e0b", dash="dash", width=2))
        fig2.update_layout(**{**_base_layout("Inertia vs K"), "height": 260,
                               "xaxis": dict(title="K", color="#475569", showgrid=False, zeroline=False),
                               "yaxis": dict(title="Inertia", color="#475569", showgrid=False, zeroline=False)})
        st.plotly_chart(fig2, use_container_width=True)

        render_save_experiment_btn(
            algorithm="K-Means",
            parameters={"k": k, "dataset": ds_name, "features": features},
            results={"inertia": km.inertia_}, key_suffix="km_up",
        )
        render_save_model_btn(km, algorithm="K-Means", key_suffix="km_up")


# ══════════════════════════════════════════════════════════════════════════════
# Logistic Regression Tab
# ══════════════════════════════════════════════════════════════════════════════

def _logreg_tab():
    st.markdown("### 🔴 Logistic Regression Classifier")

    result = get_ml_dataset("lr", classifier=True)

    if result is None:
        c1, c2, c3, c4 = st.columns(4)
        C      = c1.select_slider("Regularisation C", options=[0.01, 0.1, 0.5, 1, 5, 10, 50], value=1, key="lr_C")
        solver = c2.selectbox("Solver", ["lbfgs", "liblinear", "saga"], key="lr_solver")
        n      = c3.slider("Samples", 100, 600, 300, step=50, key="lr_n")
        ds     = c4.selectbox("Dataset", ["Linear", "Moons", "Blobs", "Circles"], key="lr_ds")
        noise  = st.slider("Noise", 0.0, 1.0, 0.25, step=0.05, key="lr_noise")

        X, y = _get_dataset(ds, n, noise)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        clf = LogisticRegression(C=C, solver=solver, max_iter=500, random_state=42)
        clf.fit(X_train_s, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_s))

        X_s = scaler.transform(X)
        xx, yy = _make_mesh(X_s)
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)
        fig = go.Figure()
        fig.add_trace(_boundary_trace(xx, yy, Z, "RdBu", 0.40))
        for tr in _scatter_trace(X_s, y):
            fig.add_trace(tr)
        fig.update_layout(**_base_layout(f"Logistic Regression – Accuracy {acc:.2%}"))
        st.plotly_chart(fig, use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc:.2%}")
        m2.metric("C (regularisation)", C)
        m3.metric("Solver", solver)
        m4.metric("Iterations", clf.n_iter_[0])
        with st.expander("📊 Confusion matrix"):
            st.plotly_chart(_confusion_chart(
                confusion_matrix(y_test, clf.predict(X_test_s)), "Plasma"), use_container_width=True)

        render_save_experiment_btn(
            algorithm="Logistic Regression",
            parameters={"C": C, "solver": solver, "dataset": ds, "samples": n},
            results={"accuracy": acc, "iterations": int(clf.n_iter_[0])},
            accuracy=acc, key_suffix="lr",
        )
        render_save_model_btn(clf, algorithm="Logistic Regression", accuracy=acc, key_suffix="lr")

    else:
        X_train, X_test, y_train, y_test, df, features, target, ds_name = result

        c1, c2 = st.columns(2)
        C      = c1.select_slider("Regularisation C", options=[0.01, 0.1, 0.5, 1, 5, 10, 50], value=1, key="lr_C_up")
        solver = c2.selectbox("Solver", ["lbfgs", "liblinear", "saga"], key="lr_solver_up")

        clf = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            st.error(f"Training failed: {e}")
            return
        acc = accuracy_score(y_test, clf.predict(X_test))

        st.plotly_chart(
            _uploaded_scatter(X_train, y_train,
                              f"Logistic Regression Training Data – Accuracy {acc:.2%}"),
            use_container_width=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc:.2%}")
        m2.metric("C", C)
        m3.metric("Solver", solver)
        m4.metric("Features", len(features))
        with st.expander("📊 Confusion matrix"):
            st.plotly_chart(_confusion_chart(
                confusion_matrix(y_test, clf.predict(X_test)), "Plasma"), use_container_width=True)

        render_save_experiment_btn(
            algorithm="Logistic Regression",
            parameters={"C": C, "solver": solver, "dataset": ds_name,
                        "features": features, "target": target},
            results={"accuracy": acc},
            accuracy=acc, key_suffix="lr_up",
        )
        render_save_model_btn(clf, algorithm="Logistic Regression", accuracy=acc, key_suffix="lr_up")


# ── Public entry point ──────────────────────────────────────────────────────────

def show():
    st.markdown(
        """
        <div style='padding:10px 0 24px 0;'>
            <div style='font-size:2rem;font-weight:800;
                        background:linear-gradient(135deg,#a78bfa,#22d3ee);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        background-clip:text;'>🤖 ML Playground</div>
            <div style='font-size:.9rem;color:#64748b;margin-top:4px;'>
                Interactive demos of KNN, Naïve Bayes, K-Means &amp; Logistic Regression
                with live decision-boundary visualisations.
                Use built-in datasets or <b style='color:#a78bfa;'>upload your own CSV/XLSX/TXT</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔵 KNN", "🟢 Naïve Bayes", "🟡 K-Means", "🔴 Logistic Regression"
    ])

    with tab1:
        _knn_tab()
    with tab2:
        _nb_tab()
    with tab3:
        _kmeans_tab()
    with tab4:
        _logreg_tab()
