"""
workspace/save_helpers.py
Reusable "Save Experiment" and "Save Model" buttons/widgets
that can be embedded inline in any module page.
"""

import streamlit as st
from storage.experiments import save_experiment
from storage.models import persist_model, save_model_meta

_CARD   = "#111827"
_BORDER = "#1e293b"
_ACCENT = "#6C63FF"


def render_save_experiment_btn(
    algorithm: str,
    parameters: dict = None,
    results: dict = None,
    accuracy: float = None,
    cost: float = None,
    path_length: int = None,
    key_suffix: str = "",
):
    """
    Render a collapsible 'Save Experiment' card.
    Only shown when the user is authenticated (not guest).
    """
    user = st.session_state.get("user")
    is_guest = st.session_state.get("guest_mode", False)

    if is_guest or user is None:
        st.caption("🔒 *Login to save this experiment to your workspace.*")
        return

    with st.expander(f"💾 Save this experiment ({algorithm})", expanded=False):
        notes = st.text_area(
            "Notes (optional)",
            placeholder="e.g. 'Best run with k=5 on Moons dataset'",
            key=f"save_exp_notes_{algorithm}_{key_suffix}",
            height=80,
        )
        if st.button(
            "💾 Save to Workspace",
            key=f"save_exp_btn_{algorithm}_{key_suffix}",
            use_container_width=True,
        ):
            eid = save_experiment(
                user_id=user["id"],
                algorithm=algorithm,
                parameters=parameters or {},
                results=results or {},
                accuracy=accuracy,
                cost=cost,
                path_length=path_length,
                notes=notes,
            )
            st.success(f"✅ Experiment saved! (ID #{eid})")


def render_save_model_btn(
    model,
    algorithm: str,
    accuracy: float = None,
    key_suffix: str = "",
):
    """
    Render a 'Save Trained Model' widget.
    Only shown for authenticated users.
    """
    user = st.session_state.get("user")
    is_guest = st.session_state.get("guest_mode", False)

    if is_guest or user is None:
        st.caption("🔒 *Login to save trained models.*")
        return

    with st.expander(f"🤖 Save trained {algorithm} model", expanded=False):
        model_name = st.text_input(
            "Model name",
            value=f"My_{algorithm}",
            key=f"save_mdl_name_{algorithm}_{key_suffix}",
        )
        if st.button(
            "💾 Save Model",
            key=f"save_mdl_btn_{algorithm}_{key_suffix}",
            use_container_width=True,
        ):
            import time
            filename = f"{model_name.replace(' ', '_')}_{int(time.time())}.pkl"
            try:
                path = persist_model(model, user["id"], filename)
                save_model_meta(
                    user_id=user["id"],
                    name=model_name,
                    algorithm=algorithm,
                    accuracy=accuracy,
                    file_path=path,
                )
                st.success(f"✅ Model **{model_name}** saved!")
            except Exception as e:
                st.error(f"Failed to save model: {e}")
