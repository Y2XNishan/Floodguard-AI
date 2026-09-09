"""Focused presentation overrides for the Streamlit application shell."""

import streamlit as st


def inject_ui_overrides() -> None:
    """Apply the final layer of UI styling after the legacy inline theme."""
    st.markdown(
        """
        <style>
        :root {
          --fg-canvas: #101827;
          --fg-surface: #182235;
          --fg-surface-raised: #202d43;
          --fg-border: #31425e;
          --fg-text: #e6edf7;
          --fg-muted: #a9b7ca;
          --fg-primary: #1cb5aa;
          --fg-danger: #ef6a70;
        }

        .stApp, .main, [data-testid="stAppViewContainer"] {
          background: var(--fg-canvas) !important;
          color: var(--fg-text) !important;
        }

        [data-testid="stMainBlockContainer"] {
          max-width: 1440px;
          padding: 2rem clamp(1rem, 3vw, 3rem) 3rem !important;
        }

        h1, h2, h3, h4, h5, h6 {
          color: var(--fg-text) !important;
          letter-spacing: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
