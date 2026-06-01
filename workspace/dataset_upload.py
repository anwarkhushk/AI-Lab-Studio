"""
workspace/dataset_upload.py
Reusable dataset upload + preview component.
Supports CSV, TXT (comma-sep), XLSX, and built-in sklearn datasets.
"""

import io
import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import (
    load_iris, load_breast_cancer, load_wine, load_digits
)
from storage.datasets import save_dataset_meta, dataset_storage_path

# ── colour tokens ──────────────────────────────────────────────────────────────
_CARD   = "#111827"
_BORDER = "#1e293b"
_ACCENT = "#6C63FF"
_MUTED  = "#64748b"

BUILTIN_DATASETS = {
    "Iris":          load_iris,
    "Breast Cancer": load_breast_cancer,
    "Wine":          load_wine,
    "Digits":        load_digits,
}


def _load_builtin(name: str) -> pd.DataFrame:
    loader = BUILTIN_DATASETS[name]
    bunch  = loader()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    return df


def _show_preview(df: pd.DataFrame):
    st.markdown(
        f"""
        <div style='background:{_CARD};border:1px solid {_BORDER};border-radius:12px;
                    padding:16px;margin:12px 0;'>
            <span style='color:{_ACCENT};font-weight:700;'>Dataset Preview</span>
            <span style='color:{_MUTED};font-size:.8rem;margin-left:12px;'>
                {df.shape[0]} rows × {df.shape[1]} columns
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(df.head(8), use_container_width=True)

    with st.expander("📋 Column types & missing values"):
        info_df = pd.DataFrame({
            "dtype":   df.dtypes.astype(str),
            "missing": df.isnull().sum(),
            "missing%": (df.isnull().mean() * 100).round(1),
        })
        st.dataframe(info_df, use_container_width=True)


def _cleaning_options(df: pd.DataFrame) -> pd.DataFrame:
    """Render cleaning controls and return cleaned dataframe."""
    st.markdown("#### 🧹 Data Cleaning")
    col1, col2 = st.columns(2)
    with col1:
        drop_na = st.checkbox("Drop rows with missing values", key="ds_drop_na")
    with col2:
        fill_na = st.checkbox("Fill missing values with column mean", key="ds_fill_na")

    if drop_na:
        df = df.dropna()
        st.info(f"Dropped missing rows → {df.shape[0]} rows remain.")
    elif fill_na:
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        st.info("Filled numeric NaN with column means.")

    return df


def _column_selection(df: pd.DataFrame) -> tuple[list[str], str | None]:
    """Render feature / target selectors; return (features, target)."""
    st.markdown("#### 🎯 Column Selection")
    cols = df.columns.tolist()

    target_col = st.selectbox(
        "Target column (label)", ["(none)"] + cols, key="ds_target"
    )
    target_col = None if target_col == "(none)" else target_col

    default_features = [c for c in cols if c != target_col]
    feature_cols = st.multiselect(
        "Feature columns", cols, default=default_features, key="ds_features"
    )
    return feature_cols, target_col


def render_dataset_upload(user_id: int | None = None, allow_save: bool = True):
    """
    Full dataset upload + preview widget.
    Returns (df, feature_cols, target_col) or (None, None, None).
    """
    st.markdown("#### 📦 Dataset Source")

    source = st.radio(
        "Choose source",
        ["Built-in Dataset", "Upload File"],
        horizontal=True,
        key="ds_source",
    )

    df: pd.DataFrame | None = None
    dataset_name = ""

    if source == "Built-in Dataset":
        ds_name = st.selectbox(
            "Select built-in dataset",
            list(BUILTIN_DATASETS.keys()),
            key="ds_builtin_choice",
        )
        df = _load_builtin(ds_name)
        dataset_name = ds_name

    else:  # Upload File
        uploaded = st.file_uploader(
            "Upload CSV / TXT / Excel",
            type=["csv", "txt", "xlsx"],
            key="ds_upload",
        )
        if uploaded is not None:
            try:
                ext = os.path.splitext(uploaded.name)[1].lower()
                if ext in (".csv", ".txt"):
                    df = pd.read_csv(uploaded)
                elif ext == ".xlsx":
                    df = pd.read_excel(uploaded)
                else:
                    st.error("Unsupported file type.")
                    return None, None, None
                dataset_name = os.path.splitext(uploaded.name)[0]
            except Exception as e:
                st.error(f"Could not read file: {e}")
                return None, None, None

    if df is None:
        return None, None, None

    _show_preview(df)
    df = _cleaning_options(df)
    feature_cols, target_col = _column_selection(df)

    # Save to workspace
    if allow_save and user_id is not None:
        st.markdown("---")
        col_s, col_n = st.columns([2, 3])
        with col_n:
            save_name = st.text_input(
                "Save as (name)", value=dataset_name, key="ds_save_name"
            )
        with col_s:
            if st.button("💾 Save Dataset to Workspace", key="ds_save_btn"):
                _save_dataset(df, user_id, save_name, feature_cols)

    # Cache in session for ML modules
    st.session_state["active_dataset"] = {
        "df":           df,
        "features":     feature_cols,
        "target":       target_col,
        "name":         dataset_name,
    }

    return df, feature_cols, target_col


def _save_dataset(df: pd.DataFrame, user_id: int, name: str, features: list):
    import time
    filename = f"{name.replace(' ', '_')}_{int(time.time())}.csv"
    path = dataset_storage_path(user_id, filename)
    df.to_csv(path, index=False)
    save_dataset_meta(
        user_id=user_id,
        name=name,
        rows=df.shape[0],
        cols=df.shape[1],
        features=features,
        file_path=path,
    )
    st.success(f"✅ Dataset **{name}** saved to your workspace!")


def get_active_dataset() -> dict | None:
    """Return the dataset cached in session from last upload widget render."""
    return st.session_state.get("active_dataset")
