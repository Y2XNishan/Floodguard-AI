"""Clean internal-tool presentation overrides for Streamlit application shell."""

import streamlit as st


def inject_ui_overrides() -> None:
    """Apply clean, human-made developer tool styling to the application."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">

        <style>
        :root {
          --app-bg: #0f172a;
          --app-surface: #1e293b;
          --app-surface-hover: #243247;
          --app-border: #334155;
          --app-border-light: #475569;
          --app-text: #f8fafc;
          --app-text-muted: #94a3b8;
          --app-accent: #3b82f6;
          --app-accent-hover: #2563eb;
          --app-success: #10b981;
          --app-warning: #f59e0b;
          --app-danger: #ef4444;
          --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }

        /* Base Application Layout */
        body, .stApp, .main, [data-testid="stAppViewContainer"] {
          background-color: var(--app-bg) !important;
          color: var(--app-text) !important;
          font-family: var(--font-family) !important;
        }

        [data-testid="stMainBlockContainer"] {
          max-width: 1440px;
          padding: 1.5rem 2rem 3rem !important;
        }

        /* Clean Typography - No Gradients or Glows */
        h1, h2, h3, h4, h5, h6 {
          font-family: var(--font-family) !important;
          color: var(--app-text) !important;
          font-weight: 600 !important;
          letter-spacing: -0.01em !important;
          background: none !important;
          -webkit-text-fill-color: initial !important;
          text-shadow: none !important;
        }

        h1 { font-size: 1.875rem !important; line-height: 2.25rem !important; }
        h2 { font-size: 1.5rem !important; line-height: 2rem !important; }
        h3 { font-size: 1.25rem !important; line-height: 1.75rem !important; }
        h4, h5, h6 { font-size: 1rem !important; }

        p, span, div, label {
          font-family: var(--font-family);
        }

        /* Sidebar - Simple Flat Architecture */
        [data-testid="stSidebar"] {
          background: #0b1120 !important;
          border-right: 1px solid var(--app-border) !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] {
          background: var(--app-surface) !important;
          border: 1px solid var(--app-border) !important;
          border-radius: 6px !important;
        }

        [data-testid="stSidebar"] hr {
          border-color: var(--app-border) !important;
        }

        /* Tabs - Professional Flat Developer Tool Navigation */
        .stTabs [data-baseweb="tab-list"] {
          gap: 4px;
          background: transparent !important;
          border-bottom: 1px solid var(--app-border) !important;
          padding: 0 0 2px 0 !important;
          overflow-x: auto;
        }

        .stTabs [role="tab"] {
          font-family: var(--font-family) !important;
          font-weight: 500 !important;
          font-size: 0.875rem !important;
          color: var(--app-text-muted) !important;
          background: transparent !important;
          border: none !important;
          border-bottom: 2px solid transparent !important;
          border-radius: 0 !important;
          padding: 0.5rem 0.85rem !important;
          white-space: nowrap;
          transition: color 150ms ease, border-color 150ms ease !important;
        }

        .stTabs [role="tab"]:hover {
          color: var(--app-text) !important;
          background: transparent !important;
        }

        .stTabs [role="tab"][aria-selected="true"] {
          background: transparent !important;
          color: var(--app-accent) !important;
          border-bottom: 2px solid var(--app-accent) !important;
          font-weight: 600 !important;
          box-shadow: none !important;
        }

        [data-baseweb="tab-highlight"],
        [data-baseweb="tab-border"] {
          display: none !important;
        }

        /* Inputs, Selectboxes & Textareas */
        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input {
          background: #0b1120 !important;
          color: var(--app-text) !important;
          border: 1px solid var(--app-border) !important;
          border-radius: 6px !important;
          font-family: var(--font-family) !important;
          font-size: 0.875rem !important;
          transition: border-color 150ms ease !important;
        }

        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus,
        [data-baseweb="select"]:focus-within {
          border-color: var(--app-accent) !important;
          box-shadow: 0 0 0 1px var(--app-accent) !important;
          outline: none !important;
        }

        [data-testid="stTextInput"] label,
        [data-testid="stTextArea"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stSlider"] label {
          color: var(--app-text-muted) !important;
          font-family: var(--font-family) !important;
          font-size: 0.8125rem !important;
          font-weight: 500 !important;
          letter-spacing: 0 !important;
          text-transform: none !important;
        }

        /* Buttons - Clean Solid Blue Accent */
        .stButton > button[kind="primary"] {
          background: var(--app-accent) !important;
          color: #ffffff !important;
          font-family: var(--font-family) !important;
          font-weight: 600 !important;
          font-size: 0.875rem !important;
          border: 1px solid var(--app-accent-hover) !important;
          border-radius: 6px !important;
          padding: 0.5rem 1.1rem !important;
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
          transition: background-color 120ms ease !important;
        }

        .stButton > button[kind="primary"]:hover {
          background: var(--app-accent-hover) !important;
          transform: none !important;
          box-shadow: none !important;
        }

        .stButton > button:not([kind="primary"]),
        .stDownloadButton > button {

/* Remaining legacy overrides */
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

        .stTabs [data-baseweb="tab-list"] {
          gap: 0.25rem;
          overflow-x: auto;
          padding-bottom: 0.25rem;
        }

        .stTabs [role="tab"] {
          min-height: 2.5rem;
          padding: 0.5rem 0.75rem !important;
          border: 1px solid transparent !important;
          border-radius: 6px !important;
          white-space: nowrap;
        }

        .stTabs [role="tab"][aria-selected="true"] {
          background: rgba(28, 181, 170, 0.16) !important;
          border-color: rgba(28, 181, 170, 0.5) !important;
          border-left-width: 1px !important;
          border-radius: 6px !important;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-baseweb="select"] > div {
          background: var(--fg-surface) !important;
          color: var(--fg-text) !important;
          border-color: var(--fg-border) !important;
          border-radius: 6px !important;
        }

        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder {
          color: var(--fg-muted) !important;
          opacity: 1;
        }

        [data-testid="stTextInput"] label,
        [data-testid="stTextArea"] label,
        [data-testid="stSelectbox"] label {
          color: var(--fg-text) !important;
          font-weight: 600 !important;
        }

        .stButton > button,
        .stDownloadButton > button {
          min-height: 2.5rem;
          border-radius: 6px !important;
          font-weight: 700 !important;
          transition: background-color 120ms ease, border-color 120ms ease !important;
        }

        .stButton > button[kind="primary"] {
          background: var(--fg-primary) !important;
          color: #07201e !important;
          border-color: var(--fg-primary) !important;
        }

        .stButton > button:not([kind="primary"]),
        .stDownloadButton > button {
          background: transparent !important;
          color: #8ce4dc !important;
          border-color: #4c8c88 !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
          transform: none !important;
          filter: brightness(1.08);
        }

        [data-testid="stAlert"] {
          border-radius: 6px !important;
          border-width: 1px !important;
          padding: 0.8rem 1rem !important;
        }

        [data-testid="stAlert"] p {
          color: var(--fg-text) !important;
          margin: 0 !important;
        }

        [data-testid="stMetric"] {
          background: var(--fg-surface) !important;
          border: 1px solid var(--fg-border) !important;
          border-radius: 6px !important;
          min-height: 7.5rem;
          padding: 1rem !important;
        }

        [data-testid="stMetricLabel"] {
          color: var(--fg-muted) !important;
        }

        [data-testid="stMetricValue"] {
          color: var(--fg-text) !important;
          font-variant-numeric: tabular-nums;
        }

        [data-testid="stSidebar"] {
          background: #0c1320 !important;
          border-right-color: var(--fg-border) !important;
        }

        [data-testid="stSidebar"] [data-testid="stExpander"] {
          border-color: var(--fg-border) !important;
          border-radius: 6px !important;
        }

        [data-testid="stSidebar"] .stButton > button {
          min-height: 2.25rem;
          text-align: left;
        }

        [data-testid="stSidebar"] hr {
          border-color: var(--fg-border) !important;
        }

        .glass-card,
        .metric-card,
        .weather-card,
        .chat-window,
        .ai-summary-card,
        .rec-box,
        .stat-card {
          background: var(--fg-surface) !important;
          border-color: var(--fg-border) !important;
          border-radius: 6px !important;
          box-shadow: none !important;
        }

        .metric-card,
        .stat-card {
          min-width: 0;
          overflow-wrap: anywhere;
        }

        .weather-metrics {
          grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr)) !important;
        }

        button:focus-visible,
        input:focus-visible,
        textarea:focus-visible,
        [role="tab"]:focus-visible,
        [data-baseweb="select"]:focus-within {
          outline: 3px solid rgba(140, 228, 220, 0.7) !important;
          outline-offset: 2px !important;
          box-shadow: none !important;
        }

        [data-testid="stCheckbox"] label,
        [data-testid="stRadio"] label {
          color: var(--fg-text) !important;
        }

        @media (prefers-reduced-motion: reduce) {
          *, *::before, *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
          }
        }

        @media (max-width: 720px) {
          [data-testid="stMainBlockContainer"] {
            padding: 1rem 0.85rem 2rem !important;
          }

          .stTabs [role="tab"] {
            font-size: 0.85rem !important;
            padding: 0.45rem 0.65rem !important;
          }

          [data-testid="stMetric"] {
            min-height: 6.5rem;
            padding: 0.75rem !important;
          }

          .chat-bubble,
          .ai-bubble,
          .user-bubble {
            max-width: 92% !important;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
