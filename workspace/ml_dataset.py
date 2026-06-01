"""
workspace/ml_dataset.py
────────────────────────────────────────────────────────────────────────────
Per-tab dataset loader for ML Playground.
• Built-in (Iris, Wine, Breast Cancer, Digits) via sklearn
• Upload (CSV / TXT / XLSX)
• Preview, cleaning, scaling, feature/target selection
• Workspace save integration
All widget keys are prefixed with `tab_key` to avoid Streamlit conflicts.
"""

import os
import io
import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ── colour tokens (match existing palette) ─────────────────────────────────────
_CARD   = "#111827"
_BORDER = "#1e293b"
_ACCENT = "#6C63FF"
_CYAN   = "#22d3ee"
_MUTED  = "#64748b"
_MAX_MB = 20  # file size guard

BUILTIN = {
    "Iris":          load_iris,
    "Wine":          load_wine,
    "Breast Cancer": load_breast_cancer,
    "Digits":        load_digits,
}


# ── helpers ─────────────────────────────────────────────────────────────────────

def _info_banner(df: pd.DataFrame, name: str):
    st.markdown(
        f"""
        <div style='background:{_CARD};border:1px solid {_BORDER};border-radius:12px;
                    padding:14px 18px;margin:10px 0;display:flex;gap:24px;flex-wrap:wrap;'>
            <span style='color:{_ACCENT};font-weight:700;'>📄 {name}</span>
            <span style='color:{_MUTED};font-size:.82rem;'>
                {df.shape[0]:,} rows &nbsp;·&nbsp; {df.shape[1]} columns
                &nbsp;·&nbsp; {df.isnull().sum().sum()} missing values
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _preview(df: pd.DataFrame, tab_key: str):
    st.dataframe(df.head(10), use_container_width=True)
    with st.expander("📊 Column statistics", expanded=False):
        info = pd.DataFrame({
            "dtype":    df.dtypes.astype(str),
            "non-null": df.notnull().sum(),
            "missing":  df.isnull().sum(),
            "missing%": (df.isnull().mean() * 100).round(1),
        })
        st.dataframe(info, use_container_width=True)


def _clean(df: pd.DataFrame, tab_key: str) -> pd.DataFrame:
    strategy = st.selectbox(
        "Missing values",
        ["Keep As Is", "Remove Rows With Missing Values",
         "Fill Numeric With Mean", "Fill Numeric With Median"],
        key=f"{tab_key}_clean",
    )
    num_cols = df.select_dtypes(include=[np.number]).columns
    if strategy == "Remove Rows With Missing Values":
        before = len(df)
        df = df.dropna()
        st.info(f"Dropped {before - len(df)} rows → {len(df):,} remain.")
    elif strategy == "Fill Numeric With Mean":
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        st.info("Filled numeric NaN with column means.")
    elif strategy == "Fill Numeric With Median":
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        st.info("Filled numeric NaN with column medians.")
    return df


def _select_columns(df: pd.DataFrame, tab_key: str):
    cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    c1, c2 = st.columns(2)
    with c1:
        target = st.selectbox(
            "🎯 Target column", ["(none)"] + cols, key=f"{tab_key}_target"
        )
        target = None if target == "(none)" else target
    with c2:
        default_feat = [c for c in num_cols if c != target]
        features = st.multiselect(
            "📐 Feature columns (numeric only)",
            num_cols,
            default=default_feat,
            key=f"{tab_key}_features",
        )
    return features, target


def _scale(X_train, X_test, tab_key: str):
    method = st.selectbox(
        "Feature scaling",
        ["None", "StandardScaler", "MinMaxScaler"],
        key=f"{tab_key}_scale",
    )
    if method == "StandardScaler":
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test  = sc.transform(X_test)
    elif method == "MinMaxScaler":
        sc = MinMaxScaler()
        X_train = sc.fit_transform(X_train)
        X_test  = sc.transform(X_test)
    return X_train, X_test


def _save_to_workspace(df, name, features, tab_key):
    user = st.session_state.get("user")
    if user is None:
        st.caption("🔒 *Login to save datasets to your workspace.*")
        return
    try:
        from storage.datasets import save_dataset_meta, dataset_storage_path
        col_n, col_b = st.columns([3, 2])
        with col_n:
            save_name = st.text_input("Save as", value=name,
                                      key=f"{tab_key}_save_name")
        with col_b:
            if st.button("💾 Save Dataset", key=f"{tab_key}_save_btn"):
                fname = f"{save_name.replace(' ', '_')}_{int(time.time())}.csv"
                path  = dataset_storage_path(user["id"], fname)
                df.to_csv(path, index=False)
                save_dataset_meta(
                    user_id=user["id"], name=save_name,
                    rows=df.shape[0], cols=df.shape[1],
                    features=features, file_path=path,
                )
                st.success(f"✅ **{save_name}** saved to your workspace!")
    except Exception as e:
        st.error(f"Save failed: {e}")


# ── public API ──────────────────────────────────────────────────────────────────

def get_ml_dataset(tab_key: str, classifier: bool = True):
    """
    Render the Dataset Source selector for one ML tab.

    Parameters
    ----------
    tab_key     : unique prefix for all Streamlit widget keys (e.g. "knn")
    classifier  : if True show target column selector; False = clustering mode

    Returns
    -------
    (X_train, X_test, y_train, y_test, df, features, target, dataset_name)
    or  None  when no valid data is ready yet.
    """
    from sklearn.model_selection import train_test_split

    st.markdown(
        f"""
        <div style='background:{_CARD};border:1px solid {_BORDER};
                    border-radius:12px;padding:14px 18px;margin-bottom:14px;'>
            <span style='color:{_ACCENT};font-weight:700;font-size:.95rem;'>
                📦 Dataset Source
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source = st.radio(
        "Source",
        ["Built-in Dataset", "Upload Your Own Dataset"],
        horizontal=True,
        key=f"{tab_key}_source",
        label_visibility="collapsed",
    )

    df: pd.DataFrame | None = None
    dataset_name = ""

    # ── Built-in ──────────────────────────────────────────────────────────────
    if source == "Built-in Dataset":
        chosen = st.selectbox(
            "Select dataset",
            list(BUILTIN.keys()),
            key=f"{tab_key}_builtin",
        )
        bunch = BUILTIN[chosen]()
        df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        df["target"] = bunch.target
        dataset_name = chosen
        return None   # signal: use legacy synthetic-data path for built-in

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload CSV / TXT / XLSX",
        type=["csv", "txt", "xlsx"],
        key=f"{tab_key}_upload",
    )
    if uploaded is None:
        st.info("⬆️ Upload a file to get started, or switch to Built-in Dataset.")
        return None

    # Size guard
    size_mb = uploaded.size / 1_048_576
    if size_mb > _MAX_MB:
        st.warning(f"⚠️ File is {size_mb:.1f} MB. Consider using a smaller sample (< {_MAX_MB} MB).")

    # Parse
    try:
        ext = os.path.splitext(uploaded.name)[1].lower()
        if ext in (".csv", ".txt"):
            df = pd.read_csv(uploaded)
        elif ext == ".xlsx":
            df = pd.read_excel(uploaded)
        else:
            st.error("Unsupported format. Please upload CSV, TXT, or XLSX.")
            return None
        dataset_name = os.path.splitext(uploaded.name)[0]
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        return None

    if df.empty:
        st.warning("⚠️ The uploaded file appears to be empty.")
        return None

    # Info + preview
    _info_banner(df, uploaded.name)
    _preview(df, tab_key)

    st.markdown("---")
    col_clean, col_scale = st.columns(2)
    with col_clean:
        st.markdown("**🧹 Missing Values**")
        df = _clean(df, tab_key)
    with col_scale:
        st.markdown("**⚖️ Feature Scaling**")
        # scaling applied after split below

    features, target = _select_columns(df, tab_key)

    if not features:
        st.warning("⚠️ Please select at least one feature column.")
        return None

    if classifier and (target is None or target == "(none)"):
        st.warning("⚠️ Please select a target column for classification.")
        return None

    # Drop rows where target or features are NaN
    cols_needed = features + ([target] if target else [])
    df_clean = df[cols_needed].dropna()
    if len(df_clean) < 10:
        st.error("❌ Too few valid rows after cleaning. Please review your data.")
        return None

    # Non-numeric target check for classifiers
    if classifier:
        y_vals = df_clean[target]
        if not pd.api.types.is_numeric_dtype(y_vals):
            # Label-encode automatically
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_clean = df_clean.copy()
            df_clean[target] = le.fit_transform(y_vals)
            st.info(f"ℹ️ Target **{target}** label-encoded: {list(le.classes_)} → {list(range(len(le.classes_)))}")

    X = df_clean[features].values
    y = df_clean[target].values if classifier else None

    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, test_size=0.25, random_state=42)
        if classifier
        else (X, X, None, None)
    )

    # Apply scaling (from col_scale widget already rendered above)
    with col_scale:
        X_train, X_test = _scale(X_train, X_test, tab_key)

    st.markdown("---")
    with st.expander("💾 Save Dataset to Workspace", expanded=False):
        _save_to_workspace(df_clean, dataset_name, features, tab_key)

    return X_train, X_test, y_train, y_test, df_clean, features, target, dataset_name
