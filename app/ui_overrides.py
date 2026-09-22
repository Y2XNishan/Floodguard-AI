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
          background: var(--app-surface) !important;
          color: var(--app-text) !important;
          font-family: var(--font-family) !important;
          font-weight: 500 !important;
          font-size: 0.875rem !important;
          border: 1px solid var(--app-border) !important;
          border-radius: 6px !important;
          padding: 0.5rem 1.1rem !important;
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
          transition: background-color 120ms ease, border-color 120ms ease !important;
        }

        .stButton > button:not([kind="primary"]):hover,
        .stDownloadButton > button:hover {
          background: var(--app-surface-hover) !important;
          border-color: var(--app-border-light) !important;
          transform: none !important;
          box-shadow: none !important;
        }

        /* Metric Cards - Simple, Non-Theatrical */
        [data-testid="stMetric"] {
          background: var(--app-surface) !important;
          border: 1px solid var(--app-border) !important;
          border-radius: 6px !important;
          padding: 1rem !important;
          box-shadow: none !important;
        }

        [data-testid="stMetricLabel"] {
          color: var(--app-text-muted) !important;
          font-family: var(--font-family) !important;
          font-size: 0.75rem !important;
          font-weight: 600 !important;
          text-transform: uppercase !important;
          letter-spacing: 0.04em !important;
        }

        [data-testid="stMetricValue"] {
          color: var(--app-text) !important;
          font-family: var(--font-family) !important;
          font-size: 1.5rem !important;
          font-weight: 700 !important;
        }

        /* Generic Container Cards */
        .glass-card,
        .metric-card,
        .weather-card,
        .chat-window,
        .ai-summary-card,
        .rec-box,
        .stat-card,
        .bento-card,
        .tactical-card {
          background: var(--app-surface) !important;
          border: 1px solid var(--app-border) !important;
          border-radius: 6px !important;
          box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
          backdrop-filter: none !important;
          -webkit-backdrop-filter: none !important;
          transform: none !important;
        }

        .glass-card:hover,
        .metric-card:hover,
        .weather-card:hover {
          border-color: var(--app-border-light) !important;
          transform: none !important;
          box-shadow: none !important;
        }

        /* Recommendation Boxes */
        .rec-box {
          border-radius: 6px;
          padding: 16px;
          margin: 10px 0;
          background: var(--app-surface);
          border: 1px solid var(--app-border);
          border-left-width: 3px !important;
        }
        .rec-safe { border-left-color: var(--app-success) !important; }
        .rec-mod { border-left-color: var(--app-warning) !important; }
        .rec-danger { border-left-color: var(--app-danger) !important; }

        /* Status Badges - Clean Flat Chips */
        .badge {
          display: inline-flex;
          align-items: center;
          padding: 2px 8px;
          border-radius: 4px;
          font-family: var(--font-family);
          font-weight: 600;
          font-size: 0.75rem;
          letter-spacing: 0;
          text-transform: uppercase;
        }
        .badge-safe {
          background: rgba(16, 185, 129, 0.12);
          color: #34d399;
          border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .badge-moderate {
          background: rgba(245, 158, 11, 0.12);
          color: #fbbf24;
          border: 1px solid rgba(245, 158, 11, 0.3);
        }
        .badge-danger {
          background: rgba(239, 68, 68, 0.12);
          color: #f87171;
          border: 1px solid rgba(239, 68, 68, 0.3);
        }

        /* Clean Scrollbars */
        ::-webkit-scrollbar {
          width: 6px;
          height: 6px;
        }
        ::-webkit-scrollbar-track {
          background: var(--app-bg);
        }
        ::-webkit-scrollbar-thumb {
          background: var(--app-border);
          border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
          background: var(--app-border-light);
        }

        /* Alerts & Status Strips */
        [data-testid="stAlert"] {
          background: var(--app-surface) !important;
          border-radius: 6px !important;
          border-width: 1px !important;
          border-color: var(--app-border) !important;
          padding: 0.75rem 1rem !important;
        }

        /* Clean Status Strip */
        .status-strip {
          background: var(--app-surface);
          border: 1px solid var(--app-border);
          border-radius: 6px;
          padding: 8px 14px;
          margin-bottom: 16px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 10px;
          font-size: 0.8125rem;
        }

        /* Clean Alert Banner */
        .system-alert-banner {
          background: rgba(239, 68, 68, 0.08);
          border: 1px solid rgba(239, 68, 68, 0.25);
          border-left: 3px solid var(--app-danger);
          border-radius: 6px;
          padding: 10px 14px;
          margin-bottom: 16px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-wrap: wrap;
          gap: 10px;
          font-size: 0.875rem;
        }

        /* Remove All Pulsing / Blinking / Shimmer Animations */
        *, *::before, *::after {
          animation-name: none !important;
          text-shadow: none !important;
        }

        /* Dividers */
        .clean-divider {
          height: 1px;
          background: var(--app-border);
          border: none;
          margin: 20px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
