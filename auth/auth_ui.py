"""
auth/auth_ui.py – Login, Signup, Logout, and Guest-Mode UI for AI Lab Studio.
Matches the existing dark theme exactly; no new design language introduced.
"""

import re
import streamlit as st
from storage.users import (
    create_user, get_user_by_username, get_user_by_email,
    verify_password, username_exists, email_exists,
)
from storage.db import init_db

# ── Shared colours (match app.py) ─────────────────────────────────────────────
_BG     = "#080c18"
_CARD   = "#111827"
_BORDER = "#1e293b"
_ACCENT = "#6C63FF"
_MUTED  = "#64748b"
_TEXT   = "#cbd5e1"

# ── Custom CSS injected once ───────────────────────────────────────────────────
_AUTH_CSS = """
<style>
.auth-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 18px;
    padding: 36px 40px;
    max-width: 480px;
    margin: 0 auto;
    box-shadow: 0 8px 40px rgba(108,99,255,0.12);
}
.auth-title {
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF, #22d3ee);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
}
.auth-sub {
    font-size: .85rem;
    color: #64748b;
    margin-bottom: 24px;
}
.auth-divider {
    border: none;
    border-top: 1px solid #1e293b;
    margin: 20px 0;
}
.guest-badge {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: .75rem;
    color: #94a3b8;
}
</style>
"""


def init_session():
    """Ensure all required session_state keys exist."""
    init_db()
    defaults = {
        "authenticated": False,
        "user": None,           # dict with id / username / email
        "guest_mode": False,
        "auth_page": "login",   # "login" | "signup"
        "show_workspace": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _is_valid_email(email: str) -> bool:
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()))


# ── Login form ─────────────────────────────────────────────────────────────────

def _login_form():
    st.markdown(
        """
        <div class="auth-title">Welcome back 👋</div>
        <div class="auth-sub">Sign in to your AI Lab Studio account</div>
        """,
        unsafe_allow_html=True,
    )

    identifier = st.text_input(
        "Username or Email", key="login_id",
        placeholder="your_username or email@example.com",
    )
    password = st.text_input(
        "Password", type="password", key="login_pw", placeholder="••••••••"
    )

    if st.button("Sign In", key="login_btn", use_container_width=True):
        if not identifier or not password:
            st.error("Please fill in both fields.")
            return

        # Look up by username or email
        user = get_user_by_username(identifier) or get_user_by_email(identifier)

        if user and verify_password(password, user["password"]):
            st.session_state.authenticated = True
            st.session_state.guest_mode    = False
            st.session_state.user = {
                "id":       user["id"],
                "username": user["username"],
                "email":    user["email"],
            }
            st.success(f"✅ Welcome back, **{user['username']}**!")
            st.rerun()
        else:
            st.error("❌ Invalid credentials. Please try again.")

    st.markdown('<hr class="auth-divider">', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        if st.button("Create Account", key="go_signup", use_container_width=True):
            st.session_state.auth_page = "signup"
            st.rerun()
    with col_r:
        if st.button("Continue as Guest", key="go_guest", use_container_width=True):
            st.session_state.guest_mode    = True
            st.session_state.authenticated = False
            st.session_state.user          = None
            st.rerun()


# ── Signup form ────────────────────────────────────────────────────────────────

def _signup_form():
    st.markdown(
        """
        <div class="auth-title">Create Account ✨</div>
        <div class="auth-sub">Join AI Lab Studio and save your experiments</div>
        """,
        unsafe_allow_html=True,
    )

    username  = st.text_input("Username", key="su_user", placeholder="e.g. alice_ml")
    email     = st.text_input("Email",    key="su_email", placeholder="alice@example.com")
    password  = st.text_input("Password", type="password", key="su_pw",
                               placeholder="At least 6 characters")
    password2 = st.text_input("Confirm Password", type="password", key="su_pw2",
                               placeholder="Repeat password")

    if st.button("Create Account", key="signup_btn", use_container_width=True):
        errors = []

        if not username or len(username.strip()) < 3:
            errors.append("Username must be at least 3 characters.")
        elif not re.match(r"^[A-Za-z0-9_\-]+$", username.strip()):
            errors.append("Username may only contain letters, numbers, _ and -.")
        elif username_exists(username):
            errors.append("Username is already taken.")

        if not email or not _is_valid_email(email):
            errors.append("Please enter a valid email address.")
        elif email_exists(email):
            errors.append("Email is already registered.")

        if not password or len(password) < 6:
            errors.append("Password must be at least 6 characters.")
        elif password != password2:
            errors.append("Passwords do not match.")

        if errors:
            for e in errors:
                st.error(e)
            return

        user = create_user(username.strip(), email.strip(), password)
        if user:
            st.session_state.authenticated = True
            st.session_state.guest_mode    = False
            st.session_state.user = user
            st.success(f"🎉 Account created! Welcome, **{username}**!")
            st.rerun()
        else:
            st.error("❌ Account creation failed. Please try again.")

    st.markdown('<hr class="auth-divider">', unsafe_allow_html=True)

    if st.button("← Back to Login", key="go_login", use_container_width=True):
        st.session_state.auth_page = "login"
        st.rerun()


# ── Public: full-page auth gate ────────────────────────────────────────────────

def render_auth_page():
    """Render centred login / signup card. Returns True when authenticated."""
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)

    # Centre the card vertically with some spacing
    st.markdown("<br><br>", unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        # App branding
        st.markdown(
            """
            <div style='text-align:center;margin-bottom:28px;'>
                <div style='font-size:3rem;'>🧠</div>
                <div style='font-size:1.5rem;font-weight:800;
                            background:linear-gradient(135deg,#6C63FF,#22d3ee);
                            -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                            background-clip:text;'>AI Lab Studio</div>
                <div style='font-size:.8rem;color:#475569;'>Interactive AI Learning Platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container():
            if st.session_state.auth_page == "signup":
                _signup_form()
            else:
                _login_form()


# ── Public: sidebar logout button ─────────────────────────────────────────────

def render_logout_button():
    """Render a styled logout button inside the sidebar."""
    user = st.session_state.get("user")
    is_guest = st.session_state.get("guest_mode", False)

    if is_guest:
        st.sidebar.markdown(
            "<div class='guest-badge'>👤 Guest Mode</div>",
            unsafe_allow_html=True,
        )
    elif user:
        st.sidebar.markdown(
            f"""
            <div style='background:#111827;border:1px solid #1e293b;border-radius:10px;
                        padding:10px 14px;margin-bottom:8px;font-size:.82rem;'>
                <span style='color:#a5b4fc;font-weight:600;'>👤 {user['username']}</span><br>
                <span style='color:#475569;font-size:.72rem;'>{user['email']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.sidebar.button("🚪 Logout", key="logout_btn", use_container_width=True):
        for key in ["authenticated", "user", "guest_mode", "auth_page"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
