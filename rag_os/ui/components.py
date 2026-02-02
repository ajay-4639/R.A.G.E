"""Reusable UI components for RAG OS Streamlit app."""

import streamlit as st
from typing import Any, Optional


def metric_card(title: str, value: Any, delta: Optional[str] = None):
    """Display a metric in a styled card."""
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="margin: 0; color: #666;">{title}</h4>
        <p style="font-size: 2rem; margin: 0; font-weight: bold;">{value}</p>
        {f'<p style="color: green; margin: 0;">{delta}</p>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)


def source_card(content: str, source: str, score: Optional[float] = None):
    """Display a source document card."""
    score_html = f'<span style="float: right; color: #1E88E5;">Score: {score:.2f}</span>' if score else ''
    st.markdown(f"""
    <div class="source-card">
        <p style="font-size: 0.8rem; color: #666; margin-bottom: 0.5rem;">
            <strong>{source}</strong> {score_html}
        </p>
        <p style="margin: 0;">{content[:300]}{'...' if len(content) > 300 else ''}</p>
    </div>
    """, unsafe_allow_html=True)


def status_badge(status: str):
    """Display a status badge."""
    colors = {
        "active": "#4CAF50",
        "healthy": "#4CAF50",
        "inactive": "#9E9E9E",
        "error": "#F44336",
        "warning": "#FF9800",
        "indexing": "#2196F3",
    }
    color = colors.get(status.lower(), "#9E9E9E")
    st.markdown(f"""
    <span style="
        background-color: {color};
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
    ">{status}</span>
    """, unsafe_allow_html=True)


def loading_spinner(message: str = "Loading..."):
    """Display a loading spinner with message."""
    return st.spinner(message)


def confirmation_dialog(title: str, message: str, confirm_label: str = "Confirm"):
    """Display a confirmation dialog."""
    with st.expander(title, expanded=True):
        st.warning(message)
        col1, col2 = st.columns(2)
        with col1:
            confirm = st.button(confirm_label, type="primary")
        with col2:
            cancel = st.button("Cancel")
        return confirm, cancel


def json_viewer(data: dict, title: Optional[str] = None):
    """Display JSON data with syntax highlighting."""
    if title:
        st.subheader(title)
    st.json(data)


def code_block(code: str, language: str = "python"):
    """Display a code block with syntax highlighting."""
    st.code(code, language=language)


def step_progress(steps: list[str], current_step: int):
    """Display a step progress indicator."""
    cols = st.columns(len(steps))
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i < current_step:
                st.markdown(f"✅ **{step}**")
            elif i == current_step:
                st.markdown(f"🔵 **{step}**")
            else:
                st.markdown(f"⚪ {step}")


def error_message(message: str, details: Optional[str] = None):
    """Display an error message with optional details."""
    st.error(message)
    if details:
        with st.expander("Error Details"):
            st.code(details)


def success_message(message: str, details: Optional[str] = None):
    """Display a success message with optional details."""
    st.success(message)
    if details:
        with st.expander("Details"):
            st.write(details)


def info_box(title: str, content: str):
    """Display an info box."""
    st.info(f"**{title}**\n\n{content}")


def pipeline_diagram(steps: list[dict]):
    """Display a simple pipeline diagram."""
    st.markdown("#### Pipeline Steps")

    for i, step in enumerate(steps):
        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            st.markdown(f"**{i + 1}**")

        with col2:
            st.markdown(f"**{step.get('id', 'unknown')}**")
            st.caption(f"Type: {step.get('type', 'unknown')}")

        with col3:
            if step.get("enabled", True):
                st.markdown("✅")
            else:
                st.markdown("⚪")

        if i < len(steps) - 1:
            st.markdown("↓")


def chat_message(role: str, content: str, timestamp: Optional[str] = None):
    """Display a chat message."""
    css_class = "user-message" if role == "user" else "assistant-message"
    icon = "👤" if role == "user" else "🤖"

    st.markdown(f"""
    <div class="chat-message {css_class}">
        <p style="margin-bottom: 0.5rem;"><strong>{icon} {role.title()}</strong>
        {f'<span style="float: right; color: #999; font-size: 0.8rem;">{timestamp}</span>' if timestamp else ''}
        </p>
        <p style="margin: 0;">{content}</p>
    </div>
    """, unsafe_allow_html=True)


def file_upload_area(
    label: str = "Upload files",
    file_types: list[str] = None,
    multiple: bool = True,
):
    """Display a file upload area."""
    if file_types is None:
        file_types = ["txt", "md", "pdf", "json"]

    return st.file_uploader(
        label,
        type=file_types,
        accept_multiple_files=multiple,
        help=f"Supported formats: {', '.join(file_types)}",
    )


def settings_toggle(
    label: str,
    key: str,
    default: bool = False,
    help_text: Optional[str] = None,
):
    """Display a settings toggle."""
    return st.toggle(label, value=default, key=key, help=help_text)


def settings_slider(
    label: str,
    min_val: int,
    max_val: int,
    default: int,
    key: str,
    help_text: Optional[str] = None,
):
    """Display a settings slider."""
    return st.slider(label, min_val, max_val, default, key=key, help=help_text)
