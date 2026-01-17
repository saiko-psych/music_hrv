"""Streamlit-based GUI for RRational - HRV Analysis Toolkit."""

from __future__ import annotations

import streamlit as st
from pathlib import Path
import time
import re

from rrational.cleaning.rr import CleaningConfig
from rrational.io import DEFAULT_ID_PATTERN, load_recording, discover_recordings
from rrational.prep import load_hrv_logger_preview
from rrational.segments.section_normalizer import SectionNormalizer
from rrational.config.sections import SectionsConfig, SectionDefinition
from rrational.gui.persistence import (
    save_groups,
    load_groups,
    save_events,
    load_events,
    save_sections,
    load_sections,
    save_participants,
    load_participants,
    load_playlist_groups,
    save_playlist_groups,
    load_music_labels,
    load_settings,
    save_settings,
    DEFAULT_SETTINGS,
    migrate_legacy_config,
)
from rrational.gui.help_text import (
    ARTIFACT_CORRECTION_HELP,
    VNS_DATA_HELP,
)

# Lazy imports for heavy modules (saves ~0.5s+ on startup)
_pd = None
_render_setup_tab = None
_render_data_tab = None
_render_analysis_tab = None


def get_pandas():
    """Lazily import pandas to speed up app startup."""
    global _pd
    if _pd is None:
        import pandas as pd
        _pd = pd
    return _pd




def _get_render_setup_tab():
    """Lazily import setup tab."""
    global _render_setup_tab
    if _render_setup_tab is None:
        from rrational.gui.tabs.setup import render_setup_tab
        _render_setup_tab = render_setup_tab
    return _render_setup_tab


def _get_render_data_tab():
    """Lazily import data tab."""
    global _render_data_tab
    if _render_data_tab is None:
        from rrational.gui.tabs.data import render_data_tab
        _render_data_tab = render_data_tab
    return _render_data_tab


def _get_render_analysis_tab():
    """Lazily import analysis tab."""
    global _render_analysis_tab
    if _render_analysis_tab is None:
        from rrational.gui.tabs.analysis import render_analysis_tab
        _render_analysis_tab = render_analysis_tab
    return _render_analysis_tab

# Parse command line arguments (passed via: streamlit run app.py -- --test-mode)
import sys
TEST_MODE = "--test-mode" in sys.argv or "--test" in sys.argv

# Lazy import for neurokit2 and matplotlib (saves ~0.9s on startup)
NEUROKIT_AVAILABLE = True
_nk = None
_plt = None


def get_neurokit():
    """Lazily import neurokit2 to speed up app startup."""
    global _nk, NEUROKIT_AVAILABLE
    if _nk is None:
        try:
            import neurokit2 as nk
            _nk = nk
        except ImportError:
            NEUROKIT_AVAILABLE = False
            _nk = None
    return _nk


def get_matplotlib():
    """Lazily import matplotlib to speed up app startup."""
    global _plt
    if _plt is None:
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


# Lazy import for plotly (saves ~0.12s on startup)
PLOTLY_AVAILABLE = True
_go = None
_plotly_events = None


def get_plotly():
    """Lazily import plotly to speed up app startup."""
    global _go, _plotly_events, PLOTLY_AVAILABLE
    if _go is None:
        try:
            import plotly.graph_objects as go
            from streamlit_plotly_events import plotly_events
            _go = go
            _plotly_events = plotly_events
        except ImportError:
            PLOTLY_AVAILABLE = False
            _go = None
            _plotly_events = None
    return _go, _plotly_events


def get_current_theme_colors():
    """Get Plotly-compatible colors based on current theme setting.

    Reads from app_settings to determine if dark mode is enabled.
    Returns dict with colors for plot backgrounds, text, grid, etc.
    """
    # Check if dark theme is set in settings
    settings = st.session_state.get("app_settings", {})
    # Theme is stored in localStorage by JS, but we can check a session state flag
    # For now, check if user preference was saved
    is_dark = settings.get("theme", "light") == "dark"

    if is_dark:
        return {
            'bg': '#0E1117',
            'text': '#FAFAFA',
            'grid': 'rgba(255,255,255,0.1)',
            'line': '#3D3D4D',
        }
    else:
        return {
            'bg': '#FFFFFF',
            'text': '#31333F',
            'grid': 'rgba(0,0,0,0.1)',
            'line': '#E5E5E5',
        }


def get_plot_colors():
    """Get custom plot colors from settings.

    Returns dict with colors for RR line, artifacts, etc.
    """
    settings = st.session_state.get("app_settings", {})
    plot_opts = settings.get("plot_options", {})
    colors = plot_opts.get("colors", {})

    return {
        'line': colors.get("line", "#2E86AB"),  # Default: accent blue
        'artifact': colors.get("artifact", "#FF6B6B"),  # Default: red
    }


# Page configuration - load favicon
_favicon_path = Path(__file__).parent / "assets" / "favicon.svg"
_favicon = _favicon_path.read_text() if _favicon_path.exists() else "R"

st.set_page_config(
    page_title="RRational" + (" [TEST MODE]" if TEST_MODE else ""),
    page_icon=_favicon,
    layout="wide",
)

# Migrate legacy config from ~/.music_hrv to ~/.rrational (v0.7.0 rename)
# This runs once per session and migrates if legacy has more data than current
if "legacy_migration_done" not in st.session_state:
    if migrate_legacy_config():
        st.toast("Migrated settings from previous version", icon="info")
        # Clear session state keys so migrated data will be loaded fresh
        for key in ["groups", "all_events", "sections", "playlist_groups"]:
            if key in st.session_state:
                del st.session_state[key]
    st.session_state.legacy_migration_done = True

# Project management session state
if "current_project" not in st.session_state:
    st.session_state.current_project = None  # Path to current project or None
if "project_manager" not in st.session_state:
    st.session_state.project_manager = None  # ProjectManager instance
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True  # Show welcome screen on first load


def apply_custom_css():
    """Apply CSS-only theme system with instant switching.

    Uses CSS custom properties for colors and a class toggle for theme switching.
    No page reload required - themes switch instantly via JavaScript.
    """
    theme_css = """
    /* ============================================
       CSS-ONLY THEME SYSTEM FOR STREAMLIT
       ============================================ */

    /* CSS Custom Properties - Light Theme (default) */
    :root {
        --bg-primary: #FFFFFF;
        --bg-secondary: #F0F2F6;
        --bg-tertiary: #E6E9EF;
        --text-primary: #31333F;
        --text-secondary: #555867;
        --text-muted: #808495;
        --accent-primary: #2E86AB;
        --accent-hover: #236B8E;
        --border-color: #D3D3D3;
        --border-light: #E5E5E5;
        --input-bg: #FFFFFF;
        --input-border: #D3D3D3;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.07);
        --success-bg: #D4EDDA;
        --success-text: #155724;
        --warning-bg: #FFF3CD;
        --warning-text: #856404;
        --error-bg: #F8D7DA;
        --error-text: #721C24;
        --info-bg: #D1ECF1;
        --info-text: #0C5460;
        --sidebar-bg: #F0F2F6;
        --sidebar-text: #31333F;
        --tab-active-bg: #FFFFFF;
        --tab-hover-bg: #E6E9EF;
        --code-bg: #F5F5F5;
        --scrollbar-track: #F0F2F6;
        --scrollbar-thumb: #C1C1C1;
    }

    /* CSS Custom Properties - Dark Theme */
    :root.dark-theme {
        --bg-primary: #0E1117;
        --bg-secondary: #262730;
        --bg-tertiary: #1E1E2E;
        --text-primary: #FAFAFA;
        --text-secondary: #B8B8C0;
        --text-muted: #808495;
        --accent-primary: #4DA6C9;
        --accent-hover: #6BB8D6;
        --border-color: #3D3D4D;
        --border-light: #333340;
        --input-bg: #1A1A24;
        --input-border: #3D3D4D;
        --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.4);
        --success-bg: #1D3D2B;
        --success-text: #75D99A;
        --warning-bg: #3D3520;
        --warning-text: #E5C76B;
        --error-bg: #3D1D20;
        --error-text: #F5A0A8;
        --info-bg: #1D3540;
        --info-text: #7DCCE8;
        --sidebar-bg: #1A1A24;
        --sidebar-text: #FAFAFA;
        --tab-active-bg: #262730;
        --tab-hover-bg: #1E1E2E;
        --code-bg: #1A1A24;
        --scrollbar-track: #1A1A24;
        --scrollbar-thumb: #4A4A5A;
    }

    /* ============================================
       GLOBAL STYLES
       ============================================ */

    /* Main app container */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    /* Main content area */
    .main .block-container {
        background-color: var(--bg-primary) !important;
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: var(--text-primary) !important;
    }

    /* Paragraphs and text */
    p, span, div, label {
        color: var(--text-primary);
    }

    /* Links */
    a {
        color: var(--accent-primary) !important;
    }
    a:hover {
        color: var(--accent-hover) !important;
    }

    /* ============================================
       SIDEBAR
       ============================================ */

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: var(--sidebar-bg) !important;
    }

    [data-testid="stSidebar"] * {
        color: var(--sidebar-text);
    }

    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] label {
        color: var(--sidebar-text) !important;
    }

    /* Sidebar collapse button */
    [data-testid="stSidebar"] button[kind="header"] {
        color: var(--sidebar-text) !important;
    }

    /* ============================================
       BUTTONS
       ============================================ */

    /* Primary button */
    .stButton > button,
    button[kind="primary"] {
        background-color: var(--accent-primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover,
    button[kind="primary"]:hover {
        background-color: var(--accent-hover) !important;
        box-shadow: var(--shadow-md) !important;
    }

    /* Secondary button */
    .stButton > button[kind="secondary"],
    button[kind="secondary"] {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background-color: var(--bg-tertiary) !important;
    }

    /* ============================================
       INPUTS & FORMS
       ============================================ */

    /* Text inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea textarea {
        background-color: var(--input-bg) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--input-border) !important;
        border-radius: 6px !important;
    }

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea textarea:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 2px rgba(46, 134, 171, 0.2) !important;
    }

    /* Input placeholders */
    .stTextInput input::placeholder,
    .stNumberInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
        opacity: 0.7 !important;
    }

    /* Select boxes */
    [data-baseweb="select"] {
        border-radius: 6px !important;
    }

    [data-baseweb="select"] > div {
        background-color: var(--input-bg) !important;
        border-color: var(--input-border) !important;
    }

    [data-baseweb="select"] span {
        color: var(--text-primary) !important;
    }

    /* Dropdown menus */
    [data-baseweb="popover"] > div,
    [data-baseweb="menu"] {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
    }

    [data-baseweb="menu"] li {
        color: var(--text-primary) !important;
    }

    [data-baseweb="menu"] li:hover {
        background-color: var(--bg-tertiary) !important;
    }

    /* Select box dropdown arrow/icon */
    [data-baseweb="select"] svg {
        fill: var(--text-primary) !important;
        color: var(--text-primary) !important;
    }

    /* Select box clear button */
    [data-baseweb="select"] [data-baseweb="clear-icon"] {
        color: var(--text-muted) !important;
    }

    /* Checkboxes - MAXIMUM SPECIFICITY to override Streamlit defaults */
    .stCheckbox label span {
        color: var(--text-primary) !important;
    }

    /* Checkbox visual box - target Streamlit's span element with st-* classes */
    .stCheckbox label span[class*="st-"] {
        background-color: var(--input-bg) !important;
        border-color: var(--input-border) !important;
    }

    /* Checked checkbox - override the RED default (rgb(255, 75, 75)) */
    .stCheckbox input[aria-checked="true"] + span,
    .stCheckbox input:checked + span,
    .stCheckbox label span[class*="st-ch"],
    .stApp .stCheckbox label > span:first-child {
        background-color: var(--accent-primary) !important;
        border-color: var(--accent-primary) !important;
    }

    /* Target emotion-cache classes used by Streamlit for checkbox */
    [class*="emotion-cache"][class*="stCheckbox"] span:first-of-type,
    .stCheckbox span[class*="st-c"] {
        background-color: var(--accent-primary) !important;
    }

    /* Force override on any 16x16 colored span in checkbox (the visual box) */
    .stCheckbox label span {
        background-color: var(--accent-primary) !important;
    }

    /* Unchecked state */
    .stCheckbox input[aria-checked="false"] + span,
    .stCheckbox input:not(:checked) + span {
        background-color: var(--input-bg) !important;
        border-color: var(--input-border) !important;
    }

    /* Checkbox checkmark icon */
    .stCheckbox svg {
        fill: white !important;
        stroke: white !important;
    }

    /* Radio buttons */
    .stRadio label span {
        color: var(--text-primary) !important;
    }

    /* Radio button circle - HIGH SPECIFICITY */
    .stRadio > div > label > div:first-child,
    .stRadio [data-baseweb="radio"],
    .stApp .stRadio input[type="radio"] + div {
        background-color: var(--input-bg) !important;
        border-color: var(--input-border) !important;
    }

    .stRadio > div > label > div:first-child:hover,
    .stRadio [data-baseweb="radio"]:hover {
        border-color: var(--accent-primary) !important;
    }

    /* Selected radio - IMPORTANT: override default */
    .stRadio > div > label > div:first-child[aria-checked="true"],
    .stRadio [data-baseweb="radio"][aria-checked="true"],
    .stApp .stRadio input[type="radio"]:checked + div {
        background-color: var(--accent-primary) !important;
        border-color: var(--accent-primary) !important;
    }

    /* ============================================
       HORIZONTAL RADIO BUTTONS - Simple styling
       ============================================ */

    /* FORCE remove ALL backgrounds from radio buttons */
    .stRadio label,
    .stRadio label *,
    .stRadio [role="radiogroup"] label,
    .stRadio [role="radiogroup"] > div,
    .stRadio [role="radiogroup"] > div > label,
    .stRadio [role="radiogroup"] > div > label > div,
    .stRadio [data-baseweb="radio"],
    .stRadio [class*="st-"],
    .stApp .stRadio label,
    .stApp .stRadio [role="radiogroup"] label {
        background-color: transparent !important;
        background: none !important;
    }

    /* Unselected radio circle - black border */
    .stRadio [data-baseweb="radio"] > div:first-child {
        border-color: #31333F !important;
        background-color: transparent !important;
    }

    :root.dark-theme .stRadio [data-baseweb="radio"] > div:first-child {
        border-color: #FAFAFA !important;
    }

    /* Selected radio circle - accent color filled */
    .stRadio [data-baseweb="radio"][aria-checked="true"] > div:first-child {
        background-color: var(--accent-primary) !important;
        border-color: var(--accent-primary) !important;
    }

    /* Selected button - border highlight around the whole option */
    .stRadio [role="radiogroup"] > div:has([aria-checked="true"]) {
        outline: 2px solid var(--accent-primary) !important;
        outline-offset: 2px !important;
        border-radius: 4px !important;
    }

    /* ============================================
       TABS - COMPLETE STYLING
       ============================================ */

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: transparent !important;
        border-bottom: 1px solid var(--border-light) !important;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: var(--bg-secondary) !important;
        color: var(--text-secondary) !important;
        border-radius: 6px 6px 0 0 !important;
        border: 1px solid var(--border-light) !important;
        border-bottom: none !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--tab-hover-bg) !important;
        color: var(--text-primary) !important;
    }

    /* Selected tab - with accent color indicator */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--tab-active-bg) !important;
        color: var(--accent-primary) !important;
        border-color: var(--accent-primary) !important;
        border-bottom: 2px solid var(--accent-primary) !important;
        font-weight: 600 !important;
    }

    /* Tab highlight/underline indicator - override Streamlit default */
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        background-color: var(--accent-primary) !important;
    }

    .stTabs [data-baseweb="tab-panel"] {
        background-color: var(--bg-primary) !important;
        border: 1px solid var(--border-light) !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 1rem !important;
    }

    /* Tab button text */
    .stTabs button[role="tab"] {
        color: var(--text-secondary) !important;
    }

    .stTabs button[role="tab"][aria-selected="true"] {
        color: var(--accent-primary) !important;
    }

    /* ============================================
       EXPANDERS
       ============================================ */

    [data-testid="stExpander"] {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 8px !important;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] details > summary,
    [data-testid="stExpander"] summary[class*="emotion-cache"] {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        background-color: var(--bg-secondary) !important;
    }

    [data-testid="stExpander"] details,
    [data-testid="stExpander"] details > div,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background-color: var(--bg-secondary) !important;
    }

    /* ============================================
       DATA FRAMES & TABLES
       ============================================ */

    .stDataFrame {
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    .stDataFrame [data-testid="stDataFrameResizable"] {
        background-color: var(--bg-primary) !important;
    }

    /* Table headers */
    .stDataFrame th {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }

    /* Table cells */
    .stDataFrame td {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-light) !important;
    }

    /* Table row hover */
    .stDataFrame tr:hover td {
        background-color: var(--bg-secondary) !important;
    }

    /* Table container border */
    .stDataFrame > div {
        border: 1px solid var(--border-light) !important;
    }

    /* Dark mode: invert data grid canvas colors (canvas ignores CSS, needs filter) */
    :root.dark-theme .stDataFrame [data-testid="stDataFrameResizable"],
    :root.dark-theme [data-testid="glideDataEditor"] {
        filter: invert(0.93) hue-rotate(180deg);
    }

    /* Dark mode for Plotly charts: JavaScript handles color updates via Plotly.relayout()
       CSS filter removed to avoid double-inversion when JS updates colors.
       The MutationObserver in apply_custom_css() detects new plots and updates them. */

    /* For plotly_events component (iframe) - still needs CSS filter since JS can't access cross-origin */
    :root.dark-theme .stCustomComponentV1 {
        filter: invert(0.93) hue-rotate(180deg);
    }

    /* ============================================
       ALERTS & MESSAGES
       ============================================ */

    [data-testid="stAlert"] {
        border-radius: 8px !important;
    }

    .stSuccess, [data-testid="stAlert"][data-baseweb-type="positive"] {
        background-color: var(--success-bg) !important;
        color: var(--success-text) !important;
    }

    .stWarning, [data-testid="stAlert"][data-baseweb-type="warning"] {
        background-color: var(--warning-bg) !important;
        color: var(--warning-text) !important;
    }

    .stError, [data-testid="stAlert"][data-baseweb-type="negative"] {
        background-color: var(--error-bg) !important;
        color: var(--error-text) !important;
    }

    .stInfo, [data-testid="stAlert"][data-baseweb-type="info"] {
        background-color: var(--info-bg) !important;
        color: var(--info-text) !important;
    }

    /* ============================================
       METRICS
       ============================================ */

    [data-testid="stMetric"] {
        background-color: var(--bg-secondary) !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        border: 1px solid var(--border-light) !important;
    }

    [data-testid="stMetric"] label {
        color: var(--text-secondary) !important;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
    }

    /* ============================================
       CODE BLOCKS
       ============================================ */

    .stCodeBlock, pre, code {
        background-color: var(--code-bg) !important;
        color: var(--text-primary) !important;
        border-radius: 6px !important;
    }

    /* ============================================
       PROGRESS BARS
       ============================================ */

    .stProgress > div {
        background-color: var(--bg-tertiary) !important;
        border-radius: 4px !important;
    }

    .stProgress > div > div {
        background-color: var(--accent-primary) !important;
        border-radius: 4px !important;
    }

    /* ============================================
       SCROLLBARS
       ============================================ */

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--scrollbar-track);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--scrollbar-thumb);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }

    /* ============================================
       MISC ELEMENTS
       ============================================ */

    /* Dividers */
    hr {
        border-color: var(--border-light) !important;
    }

    /* Captions */
    .stCaption, figcaption {
        color: var(--text-muted) !important;
    }

    /* Tooltips */
    [data-baseweb="tooltip"] {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }

    /* Popovers */
    [data-testid="stPopover"] > div {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-color) !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: var(--bg-secondary) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: 8px !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-color: var(--accent-primary) transparent transparent transparent !important;
    }

    /* Toast messages */
    [data-testid="stToast"] {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }

    /* ============================================
       MULTI-SELECT
       ============================================ */

    /* Multi-select container */
    [data-baseweb="select"] [data-baseweb="tag"] {
        background-color: var(--accent-primary) !important;
        color: white !important;
    }

    /* Multi-select clear button */
    [data-baseweb="tag"] [data-baseweb="icon"] {
        color: white !important;
    }

    /* ============================================
       NUMBER INPUT
       ============================================ */

    .stNumberInput [data-baseweb="input"] {
        background-color: var(--input-bg) !important;
        border-color: var(--input-border) !important;
    }

    .stNumberInput [data-baseweb="input"] input {
        background-color: var(--input-bg) !important;
        color: var(--text-primary) !important;
        -webkit-text-fill-color: var(--text-primary) !important;
    }

    .stNumberInput > div > div {
        background-color: var(--input-bg) !important;
    }

    .stNumberInput button {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border-color: var(--input-border) !important;
    }

    /* Remove blue highlighting from number input */
    .stNumberInput [data-baseweb="input"] {
        border-color: var(--input-border) !important;
        background-color: var(--input-bg) !important;
    }

    .stNumberInput [data-baseweb="input"]:focus-within {
        border-color: var(--accent-primary) !important;
        box-shadow: none !important;
    }

    /* ============================================
       HELP/TOOLTIP ICONS
       ============================================ */

    /* Help icon circles next to labels - comprehensive styling */
    [data-testid="stTooltipIcon"],
    .stTooltipIcon,
    button[kind="tooltip"],
    [data-baseweb="tooltip"] button,
    .stCheckbox [data-testid="stTooltipIcon"],
    [data-testid="stWidgetLabel"] button,
    [data-testid="tooltipHoverTarget"],
    .st-emotion-cache-1inwz65,
    [class*="stTooltipIcon"] {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-muted) !important;
        border: none !important;
        border-radius: 50% !important;
    }

    [data-testid="stTooltipIcon"] svg,
    .stTooltipIcon svg,
    button[kind="tooltip"] svg,
    [data-testid="tooltipHoverTarget"] svg,
    [class*="stTooltipIcon"] svg {
        fill: var(--text-muted) !important;
        color: var(--text-muted) !important;
    }

    /* Tooltip popup content */
    [data-baseweb="tooltip"] > div,
    [role="tooltip"],
    [data-baseweb="popover"] [data-baseweb="tooltip"],
    .stTooltipContent {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }

    /* Popover container (help text popup) */
    [data-baseweb="popover"] > div {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }

    .stPopover button {
        background-color: transparent !important;
        color: var(--text-primary) !important;
        border: none !important;
    }

    .stPopover button:hover {
        background-color: var(--bg-tertiary) !important;
    }

    /* ============================================
       DATE/TIME INPUTS
       ============================================ */

    [data-baseweb="calendar"] {
        background-color: var(--bg-secondary) !important;
    }

    [data-baseweb="calendar"] * {
        color: var(--text-primary) !important;
    }

    [data-baseweb="datepicker"] {
        background-color: var(--input-bg) !important;
    }

    /* ============================================
       DOWNLOAD BUTTON
       ============================================ */

    .stDownloadButton > button {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
    }

    .stDownloadButton > button:hover {
        background-color: var(--bg-tertiary) !important;
        border-color: var(--accent-primary) !important;
    }

    /* ============================================
       COLUMN CONTAINERS
       ============================================ */

    [data-testid="column"] {
        background-color: transparent !important;
    }

    /* ============================================
       FORMS
       ============================================ */

    [data-testid="stForm"] {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }

    /* ============================================
       IFRAMES (components.html)
       ============================================ */

    iframe {
        background-color: transparent !important;
    }

    /* ============================================
       HEADER & TOOLBAR
       ============================================ */

    [data-testid="stHeader"] {
        background-color: var(--bg-primary) !important;
    }

    [data-testid="stToolbar"] {
        background-color: var(--bg-primary) !important;
    }

    [data-testid="stToolbar"] button {
        color: var(--text-primary) !important;
    }

    /* ============================================
       HELP TOOLTIPS
       ============================================ */

    .stTooltipIcon {
        color: var(--text-muted) !important;
    }

    /* ============================================
       EMPTY STATES
       ============================================ */

    .stEmpty {
        color: var(--text-muted) !important;
    }

    /* ============================================
       JSON VIEWER
       ============================================ */

    [data-testid="stJson"] {
        background-color: var(--code-bg) !important;
        border-radius: 6px !important;
    }

    [data-testid="stJson"] * {
        color: var(--text-primary) !important;
    }

    /* ============================================
       DIALOG/MODAL
       ============================================ */

    [data-testid="stModal"] > div {
        background-color: var(--bg-primary) !important;
        border: 1px solid var(--border-color) !important;
    }

    /* ============================================
       WIDGET LABELS
       ============================================ */

    .stSelectbox label,
    .stMultiSelect label,
    .stTextInput label,
    .stNumberInput label,
    .stTextArea label,
    .stDateInput label,
    .stTimeInput label,
    .stCheckbox label,
    .stRadio label,
    .stSlider label,
    .stFileUploader label {
        color: var(--text-primary) !important;
    }

    /* ============================================
       WIDGET HELP TEXT
       ============================================ */

    .stSelectbox [data-testid="stWidgetLabel"] small,
    .stMultiSelect [data-testid="stWidgetLabel"] small,
    .stTextInput [data-testid="stWidgetLabel"] small,
    .stNumberInput [data-testid="stWidgetLabel"] small {
        color: var(--text-muted) !important;
    }

    /* ============================================
       MARKDOWN ELEMENTS
       ============================================ */

    .stMarkdown code {
        background-color: var(--code-bg) !important;
        color: var(--text-primary) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }

    .stMarkdown blockquote {
        border-left: 3px solid var(--accent-primary) !important;
        padding-left: 1rem !important;
        color: var(--text-secondary) !important;
    }

    /* ============================================
       DROPDOWN MENUS (Selectbox, Multiselect)
       ============================================ */

    /* Dropdown menu container (popover) */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [data-baseweb="select"] [data-baseweb="popover"] {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-light) !important;
    }

    /* Dropdown menu list */
    [data-baseweb="menu"] ul,
    [data-baseweb="popover"] ul {
        background-color: var(--bg-secondary) !important;
    }

    /* Dropdown menu items */
    [data-baseweb="menu"] li,
    [data-baseweb="popover"] li,
    [role="option"] {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }

    /* Dropdown menu item hover */
    [data-baseweb="menu"] li:hover,
    [data-baseweb="popover"] li:hover,
    [role="option"]:hover,
    [data-highlighted="true"] {
        background-color: var(--bg-primary) !important;
    }

    /* Selected dropdown item */
    [aria-selected="true"],
    [data-baseweb="menu"] li[aria-selected="true"] {
        background-color: var(--accent-primary) !important;
        color: white !important;
    }

    /* ============================================
       PLOTLY SPECIFIC
       Note: Chart colors handled by JavaScript updatePlotlyTheme/updatePlotsForTheme
       We only style the container here, not internal SVG elements
       ============================================ */

    [data-testid="stPlotlyChart"] {
        background-color: transparent !important;
    }

    /* ============================================
       BOTTOM STATUS BAR / FOOTER
       ============================================ */

    [data-testid="stBottom"] {
        background-color: var(--bg-primary) !important;
        border-top: 1px solid var(--border-light) !important;
    }

    [data-testid="stStatusWidget"] {
        color: var(--text-muted) !important;
    }

    /* ============================================
       ICON BUTTONS
       ============================================ */

    [data-testid="baseButton-headerNoPadding"],
    [data-testid="baseButton-minimal"] {
        color: var(--text-primary) !important;
    }

    [data-testid="baseButton-headerNoPadding"]:hover,
    [data-testid="baseButton-minimal"]:hover {
        background-color: var(--bg-secondary) !important;
    }

    /* ============================================
       SPECIFIC TEXT ELEMENTS
       ============================================ */

    /* Ensure all small/caption text uses correct color */
    small, .caption, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
    }

    /* Widget instruction text */
    [data-testid="InputInstructions"] {
        color: var(--text-muted) !important;
    }

    /* Ensure SVG icons get correct color */
    .stApp svg:not([fill]) {
        fill: currentColor;
    }

    /* ============================================
       STREAMLIT NATIVE COMPONENTS
       ============================================ */

    /* Chat message styling */
    [data-testid="stChatMessage"] {
        background-color: var(--bg-secondary) !important;
        border: 1px solid var(--border-light) !important;
    }

    /* Status indicator */
    [data-testid="stStatusIndicator"] {
        background-color: var(--bg-secondary) !important;
    }

    /* Markdown container */
    [data-testid="stMarkdownContainer"] {
        color: var(--text-primary) !important;
    }

    /* Element container */
    [data-testid="element-container"] {
        color: var(--text-primary);
    }

    /* Vertical block */
    [data-testid="stVerticalBlock"] {
        background-color: transparent !important;
    }

    /* ============================================
       FIX: BASEWEB COMPONENTS OVERRIDES
       ============================================ */

    /* BaseWeb input containers */
    [data-baseweb="input"] {
        background-color: var(--input-bg) !important;
        border-color: var(--input-border) !important;
    }

    [data-baseweb="input"] input {
        color: var(--text-primary) !important;
        background-color: transparent !important;
    }

    /* BaseWeb base button overrides for non-primary buttons */
    [data-baseweb="button"]:not([kind="primary"]) {
        color: var(--text-primary) !important;
    }

    /* ============================================
       ENSURE SIDEBAR ELEMENTS
       ============================================ */

    /* Sidebar select boxes */
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: var(--input-bg) !important;
    }

    /* Sidebar inputs - comprehensive targeting */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-baseweb="input"] > div,
    [data-testid="stSidebar"] .stTextInput > div > div,
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] [data-testid="stExpander"] input,
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-testid="stExpander"] [data-baseweb="input"] > div {
        background-color: var(--input-bg) !important;
        color: var(--sidebar-text) !important;
        border-color: var(--input-border) !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background-color: var(--accent-primary) !important;
    }

    /* Sidebar checkboxes */
    [data-testid="stSidebar"] .stCheckbox span {
        color: var(--sidebar-text) !important;
    }

    /* ============================================
       LOADING INDICATOR OVERRIDE
       ============================================ */

    /* Hide the default Streamlit loading swimmer icon */
    [data-testid="stStatusWidget"] .StatusWidget-swimming-icon,
    [data-testid="stStatusWidget"] svg[class*="swimming"],
    .stStatusWidget svg,
    div[data-testid="stStatusWidget"] > div > svg {
        display: none !important;
    }

    /* Clean status widget - remove grey box, hide swimmer */
    [data-testid="stStatusWidget"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stStatusWidget"] svg {
        display: none !important;
    }
    """

    # Apply CSS
    st.markdown(f"<style>{theme_css}</style>", unsafe_allow_html=True)

    # JavaScript to apply saved theme and accent color on page load
    import streamlit.components.v1 as components
    theme_init_js = """
    <script>
        (function() {
            // Access parent document (Streamlit app)
            var parentDoc = window.parent.document;
            var root = parentDoc.documentElement;

            // Immediately set page title and favicon (before Streamlit loads)
            if (parentDoc.title === 'Streamlit' || parentDoc.title === '') {
                parentDoc.title = 'RRational';
            }
            // Update favicon if it's the default Streamlit one
            var existingFavicon = parentDoc.querySelector("link[rel*='icon']");
            if (existingFavicon && existingFavicon.href.includes('streamlit')) {
                existingFavicon.href = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">📊</text></svg>';
            }

            // Force Streamlit to use custom theme from config.toml on startup
            var savedTheme = window.parent.localStorage.getItem('music-hrv-theme') || 'light';
            var isDark = savedTheme === 'dark';

            // Check if we already attempted theme setup (prevent infinite reload)
            var themeSetupDone = window.parent.localStorage.getItem('music-hrv-theme-setup-done');

            if (!themeSetupDone) {
                // Set Streamlit's internal theme to use custom theme (from config.toml)
                var customTheme = {
                    name: 'Custom',
                    themeInput: isDark ? {
                        primaryColor: '#2E86AB',
                        backgroundColor: '#0E1117',
                        secondaryBackgroundColor: '#262730',
                        textColor: '#FAFAFA',
                        base: 'dark'
                    } : {
                        primaryColor: '#2E86AB',
                        backgroundColor: '#FFFFFF',
                        secondaryBackgroundColor: '#F0F2F6',
                        textColor: '#31333F',
                        base: 'light'
                    }
                };
                window.parent.localStorage.setItem('stActiveTheme-/-v1', JSON.stringify(customTheme));
                window.parent.localStorage.setItem('music-hrv-theme-setup-done', 'true');
                // One-time reload to apply custom theme
                window.parent.location.reload();
                return;
            }

            if (isDark) {
                root.classList.add('dark-theme');
            }

            // Apply saved accent color
            var savedAccent = window.parent.localStorage.getItem('music-hrv-accent') || '#2E86AB';
            root.style.setProperty('--accent-primary', savedAccent);

            // Calculate hover color (slightly darker)
            var num = parseInt(savedAccent.slice(1), 16);
            var r = Math.min(255, Math.max(0, (num >> 16) - 20));
            var g = Math.min(255, Math.max(0, ((num >> 8) & 0x00FF) - 20));
            var b = Math.min(255, Math.max(0, (num & 0x0000FF) - 20));
            var hoverColor = '#' + (0x1000000 + r * 0x10000 + g * 0x100 + b).toString(16).slice(1);
            root.style.setProperty('--accent-hover', hoverColor);

            // Inject dynamic CSS to override Streamlit's st-* classes
            function injectAccentCSS() {
                var styleId = 'music-hrv-accent-override';
                var existingStyle = parentDoc.getElementById(styleId);
                if (existingStyle) existingStyle.remove();

                var styleTag = parentDoc.createElement('style');
                styleTag.id = styleId;

                // Use CSS custom properties instead of hardcoded colors
                // This allows theme switching without re-injecting CSS
                styleTag.textContent = `
                    /* Dynamic accent color override for Streamlit components */
                    .stCheckbox label > span:first-child,
                    .stCheckbox span[class*="st-ch"],
                    .stCheckbox span[class*="st-c"]:first-child {
                        background-color: ${savedAccent} !important;
                        border-color: ${savedAccent} !important;
                    }
                    .stCheckbox input:not(:checked) + span,
                    .stCheckbox input[aria-checked="false"] + span {
                        background-color: var(--input-bg) !important;
                        border-color: var(--border-color) !important;
                    }
                    .stTabs [data-baseweb="tab"][aria-selected="true"],
                    .stTabs button[role="tab"][aria-selected="true"] {
                        color: ${savedAccent} !important;
                        border-bottom-color: ${savedAccent} !important;
                    }
                    .stTabs [data-baseweb="tab-highlight"] {
                        background-color: ${savedAccent} !important;
                    }
                    .stButton > button {
                        background-color: ${savedAccent} !important;
                    }
                    .stButton > button:hover {
                        background-color: ${hoverColor} !important;
                    }
                    /* Expanders - use CSS variables for theme switching */
                    [data-testid="stExpander"],
                    [data-testid="stExpander"] > details,
                    [data-testid="stExpander"] details[open],
                    [data-testid="stExpander"] details > summary,
                    [data-testid="stExpander"] details > div,
                    [data-testid="stExpander"] details[open] > div,
                    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
                        background-color: var(--bg-secondary) !important;
                        border-color: var(--border-color) !important;
                    }
                    [data-testid="stExpander"] summary span,
                    [data-testid="stExpander"] p,
                    [data-testid="stExpander"] label {
                        color: var(--text-primary) !important;
                    }
                    /* Dataframes / Tables - use CSS variables */
                    .stDataFrame,
                    .stDataFrame > div,
                    .stDataFrame > div > div,
                    [data-testid="stDataFrame"],
                    [data-testid="stTable"],
                    .stDataFrame [data-testid="stDataFrameResizable"] {
                        background-color: var(--bg-primary) !important;
                    }
                    .stDataFrame th,
                    .stDataFrame thead tr,
                    .stDataFrame thead {
                        background-color: var(--bg-secondary) !important;
                        color: var(--text-primary) !important;
                    }
                    .stDataFrame td,
                    .stDataFrame tbody tr,
                    .stDataFrame tbody {
                        background-color: var(--bg-primary) !important;
                        color: var(--text-primary) !important;
                    }
                    /* Plotly charts - only style container, JS handles chart internals */
                    [data-testid="stPlotlyChart"] {
                        background-color: transparent !important;
                    }
                    /* Data editor / Glide data grid */
                    [data-testid="stDataFrameResizable"],
                    [data-testid="glideDataEditor"],
                    [data-testid="glideDataEditor"] > div,
                    [data-testid="glideDataEditor"] canvas,
                    .dvn-scroller,
                    .dvn-scroller > div,
                    .gdg-cell,
                    [class*="dvn-underlay"],
                    [class*="dvn-scroll-inner"] {
                        background-color: var(--bg-primary) !important;
                    }
                    /* Data editor cells */
                    [data-testid="glideDataEditor"] [class*="dvn-cell"],
                    [data-testid="glideDataEditor"] [class*="cell"],
                    [class*="gdg-cell"] {
                        background-color: var(--bg-primary) !important;
                        color: var(--text-primary) !important;
                    }
                    /* Data editor header */
                    [data-testid="glideDataEditor"] [class*="header"],
                    .gdg-header,
                    [class*="gdg-header"] {
                        background-color: var(--bg-secondary) !important;
                        color: var(--text-primary) !important;
                    }
                    /* RADIO BUTTONS - remove all background highlighting */
                    .stRadio label,
                    .stRadio label span,
                    .stRadio label > div,
                    .stRadio [role="radiogroup"] label,
                    .stRadio [role="radiogroup"] > div > label,
                    .stRadio [data-baseweb="radio"],
                    .stRadio [class*="st-e"],
                    .stRadio [class*="st-f"],
                    .stRadio [class*="st-g"],
                    .stRadio [class*="st-h"] {
                        background-color: transparent !important;
                        background: none !important;
                    }
                    /* Radio circle - black when unselected */
                    .stRadio [data-baseweb="radio"] > div:first-child {
                        border-color: #31333F !important;
                        background-color: transparent !important;
                    }
                    /* Radio circle - blue when selected */
                    .stRadio [data-baseweb="radio"][aria-checked="true"] > div:first-child {
                        background-color: ${savedAccent} !important;
                        border-color: ${savedAccent} !important;
                    }
                    /* Selected option - outline border */
                    .stRadio [role="radiogroup"] > div:has([aria-checked="true"]) {
                        outline: 2px solid ${savedAccent} !important;
                        outline-offset: 2px !important;
                        border-radius: 4px !important;
                    }
                `;
                parentDoc.head.appendChild(styleTag);
            }

            // Inject CSS after a short delay to ensure DOM is ready
            setTimeout(injectAccentCSS, 100);

            // Style horizontal radio buttons (remove background highlight, show circles)
            function styleRadioButtons() {
                var isDark = root.classList.contains('dark-theme');
                var borderColor = isDark ? '#FAFAFA' : '#31333F';
                var radioLabels = parentDoc.querySelectorAll('.stRadio [role="radiogroup"] label');

                radioLabels.forEach(function(label) {
                    var input = label.querySelector('input[type="radio"]');
                    var textDiv = label.querySelector('[data-testid="stMarkdownContainer"]');
                    if (textDiv) textDiv = textDiv.parentElement;
                    var circleOuter = label.querySelector('div:first-child');

                    // Remove background from text
                    if (textDiv) {
                        textDiv.style.setProperty('background-color', 'transparent', 'important');
                        textDiv.style.setProperty('background', 'none', 'important');
                    }

                    // Style radio circle
                    if (circleOuter) {
                        circleOuter.style.setProperty('border', '2px solid ' + borderColor, 'important');
                        circleOuter.style.setProperty('border-radius', '50%', 'important');

                        if (input && input.checked) {
                            circleOuter.style.setProperty('background-color', savedAccent, 'important');
                            circleOuter.style.setProperty('border-color', savedAccent, 'important');
                            label.style.setProperty('outline', '2px solid ' + savedAccent, 'important');
                            label.style.setProperty('outline-offset', '4px', 'important');
                            label.style.setProperty('border-radius', '4px', 'important');
                        } else {
                            circleOuter.style.setProperty('background-color', 'transparent', 'important');
                            label.style.removeProperty('outline');
                        }
                    }
                });
            }

            // Run initially and observe for changes
            setTimeout(styleRadioButtons, 200);
            var radioObserver = new MutationObserver(function(mutations) {
                setTimeout(styleRadioButtons, 50);
            });
            radioObserver.observe(parentDoc.body, { childList: true, subtree: true, attributes: true });

            // Update Plotly charts for current theme (both dark AND light, including iframes)
            function updatePlotsForTheme() {
                // Check current theme state (not captured value)
                var currentIsDark = root.classList.contains('dark-theme');
                var bgColor = currentIsDark ? '#0E1117' : '#FFFFFF';
                var gridColor = currentIsDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.1)';
                var textColor = currentIsDark ? '#FAFAFA' : '#31333F';
                var lineColor = currentIsDark ? '#3D3D4D' : '#E5E5E5';

                function updatePlot(plot, Plotly) {
                    if (Plotly && plot.data) {
                        try {
                            Plotly.relayout(plot, {
                                'paper_bgcolor': bgColor,
                                'plot_bgcolor': bgColor,
                                'xaxis.gridcolor': gridColor,
                                'yaxis.gridcolor': gridColor,
                                'xaxis.linecolor': lineColor,
                                'yaxis.linecolor': lineColor,
                                'xaxis.tickfont.color': textColor,
                                'yaxis.tickfont.color': textColor,
                                'xaxis.title.font.color': textColor,
                                'yaxis.title.font.color': textColor,
                                'font.color': textColor,
                                'title.font.color': textColor,
                                'legend.font.color': textColor
                            });
                        } catch(e) {}
                    }
                }

                // Update plots in main document
                var plots = parentDoc.querySelectorAll('.js-plotly-plot');
                plots.forEach(function(plot) {
                    updatePlot(plot, window.parent.Plotly);
                });

                // Also update plots inside iframes (for plotly_events component)
                var iframes = parentDoc.querySelectorAll('iframe');
                iframes.forEach(function(iframe) {
                    try {
                        var iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                        var iframePlots = iframeDoc.querySelectorAll('.js-plotly-plot');
                        var iframePlotly = iframe.contentWindow.Plotly;
                        iframePlots.forEach(function(plot) {
                            updatePlot(plot, iframePlotly);
                        });
                    } catch(e) {} // Cross-origin iframes will throw
                });
            }

            // Initial update for any existing plots - multiple calls to catch async renders
            setTimeout(updatePlotsForTheme, 100);
            setTimeout(updatePlotsForTheme, 500);
            setTimeout(updatePlotsForTheme, 1000);

            // Debounced observer for new plots (avoid excessive updates)
            var plotUpdateTimeout = null;
            var observer = new MutationObserver(function(mutations) {
                // Check for any DOM changes that might include Plotly charts
                var hasPlotlyChange = mutations.some(function(m) {
                    if (m.addedNodes.length > 0) {
                        return Array.from(m.addedNodes).some(function(n) {
                            if (n.nodeType !== 1) return false;
                            // Check for Plotly-related classes
                            return n.classList?.contains('js-plotly-plot') ||
                                   n.classList?.contains('stPlotlyChart') ||
                                   n.querySelector?.('.js-plotly-plot') ||
                                   n.querySelector?.('[data-testid="stPlotlyChart"]');
                        });
                    }
                    return false;
                });
                if (hasPlotlyChange) {
                    clearTimeout(plotUpdateTimeout);
                    // Multiple updates to catch Plotly async rendering
                    plotUpdateTimeout = setTimeout(function() {
                        updatePlotsForTheme();
                        setTimeout(updatePlotsForTheme, 300);
                    }, 100);
                }
            });
            observer.observe(parentDoc.body, { childList: true, subtree: true });
        })();
    </script>
    """
    components.html(theme_init_js, height=0)


# Apply CSS styling (theme colors handled by Streamlit natively)
apply_custom_css()

# Restore session state from query params (after theme switch)
# Check for restore_participant param and apply it
if "restore_participant" in st.query_params:
    _restore_id = st.query_params["restore_participant"]
    if _restore_id:
        st.session_state["selected_participant"] = _restore_id
    # Clear the query param to clean up the URL
    del st.query_params["restore_participant"]


# Default canonical events for the Default Group
DEFAULT_CANONICAL_EVENTS = {
    "rest_pre_start": [],
    "rest_pre_end": [],
    "measurement_start": [],
    "pause_start": [],
    "pause_end": [],
    "measurement_end": [],
    "rest_post_start": [],
    "rest_post_end": [],
}


def create_gui_normalizer(gui_events_dict):
    """Create a custom SectionNormalizer that ONLY uses GUI-defined events.

    This function creates a normalizer that:
    - ONLY checks against events defined in the GUI (st.session_state.all_events)
    - Does NOT load from config/sections.yml
    - Returns None if no match found (strict mode)
    - Uses simple lowercase string matching for synonyms

    Args:
        gui_events_dict: Dictionary of {event_name: [synonyms]} from GUI

    Returns:
        SectionNormalizer configured with GUI events only
    """
    # Convert GUI events dictionary to SectionDefinition objects
    sections_dict = {}
    for event_name, synonyms in gui_events_dict.items():
        sections_dict[event_name] = SectionDefinition(
            name=event_name,
            synonyms=tuple(synonyms) if synonyms else (),  # Use GUI synonyms (must be tuple)
            required=False,  # GUI events are not required
            description=None,
            group=None
        )

    # Create a SectionsConfig with GUI events and canonical order
    config = SectionsConfig(
        version=1,
        canonical_order=tuple(gui_events_dict.keys()),
        sections=sections_dict,
        groups={}  # No groups needed for GUI normalizer
    )

    # Create normalizer with strict fallback (returns None for unmatched)
    return SectionNormalizer(config=config, fallback_label="unknown")

# Initialize session state with persistent storage
# Load app settings first (used for defaults below)
if "app_settings" not in st.session_state:
    st.session_state.app_settings = load_settings()

if "data_dir" not in st.session_state:
    if TEST_MODE:
        # In test mode, auto-load demo data for faster testing
        demo_path = Path(__file__).parent.parent.parent.parent / "data" / "demo" / "hrv_logger"
        if demo_path.exists():
            st.session_state.data_dir = str(demo_path)
        else:
            st.session_state.data_dir = None
    else:
        # Use saved default folder, or None to use file picker
        saved_folder = st.session_state.app_settings.get("data_folder", "")
        st.session_state.data_dir = saved_folder if saved_folder else None
if "summaries" not in st.session_state:
    st.session_state.summaries = []
    # Auto-load data in test mode OR if auto_load setting is enabled
    auto_load_enabled = st.session_state.app_settings.get("auto_load", False)
    should_auto_load = (TEST_MODE or auto_load_enabled) and st.session_state.data_dir
    if should_auto_load:
        from rrational.gui.shared import cached_load_hrv_logger_preview, cached_load_vns_preview
        config_dict = {"rr_min_ms": 200, "rr_max_ms": 2000, "sudden_change_pct": 100}
        summaries = []
        data_path = Path(st.session_state.data_dir)

        # Determine folders to scan: check for known subfolders, otherwise use root
        folders_to_scan = []
        hrv_subfolder = data_path / "hrv_logger"
        vns_subfolder = data_path / "vns"

        if hrv_subfolder.exists():
            folders_to_scan.append(("hrv", str(hrv_subfolder)))
        if vns_subfolder.exists():
            folders_to_scan.append(("vns", str(vns_subfolder)))

        # If no known subfolders, scan root directory with both formats
        if not folders_to_scan:
            folders_to_scan.append(("hrv", str(data_path)))
            folders_to_scan.append(("vns", str(data_path)))

        for format_type, folder_path in folders_to_scan:
            try:
                if format_type == "hrv":
                    folder_summaries = cached_load_hrv_logger_preview(
                        folder_path,
                        pattern=DEFAULT_ID_PATTERN,
                        config_dict=config_dict,
                        gui_events_dict={},
                    )
                else:  # vns
                    folder_summaries = cached_load_vns_preview(
                        folder_path,
                        pattern=DEFAULT_ID_PATTERN,
                        config_dict=config_dict,
                        gui_events_dict={},
                        use_corrected=False,
                    )
                # Only add summaries that aren't already loaded (by participant_id)
                existing_ids = {s.participant_id for s in summaries}
                for s in folder_summaries:
                    if s.participant_id not in existing_ids:
                        summaries.append(s)
            except Exception:
                pass

        if summaries:
            st.session_state.summaries = summaries
            # Auto-assign to Default group
            for s in summaries:
                if "participant_groups" not in st.session_state:
                    st.session_state.participant_groups = {}
                st.session_state.participant_groups[s.participant_id] = "Default"
if "participant_events" not in st.session_state:
    st.session_state.participant_events = {}
if "id_pattern" not in st.session_state:
    st.session_state.id_pattern = DEFAULT_ID_PATTERN
if "participant_randomizations" not in st.session_state:
    st.session_state.participant_randomizations = {}
if "randomization_labels" not in st.session_state:
    st.session_state.randomization_labels = {}
# Device/recording metadata
if "participant_devices" not in st.session_state:
    st.session_state.participant_devices = {}  # {pid: {"device": "...", "sampling_rate": N}}
if "default_device_settings" not in st.session_state:
    st.session_state.default_device_settings = {
        "recording_app": "HRV Logger",
        "device": "Polar H10",
        "sampling_rate": 1000  # Hz - Polar H10 native rate
    }
# Music item labels (e.g., music_1 -> "Brandenburg Concerto")
if "music_labels" not in st.session_state:
    loaded_music_labels = load_music_labels()
    st.session_state.music_labels = loaded_music_labels if loaded_music_labels else {}
# Load playlist groups at startup (defines valid randomization options)
if "playlist_groups" not in st.session_state:
    loaded_playlist = load_playlist_groups()
    if loaded_playlist:
        st.session_state.playlist_groups = loaded_playlist
    else:
        # Default playlist groups (playlist_01-05)
        st.session_state.playlist_groups = {
            "playlist_01": {"label": "Playlist 1", "music_order": ["music_1", "music_2", "music_3"]},
            "playlist_02": {"label": "Playlist 2", "music_order": ["music_1", "music_3", "music_2"]},
            "playlist_03": {"label": "Playlist 3", "music_order": ["music_2", "music_1", "music_3"]},
            "playlist_04": {"label": "Playlist 4", "music_order": ["music_2", "music_3", "music_1"]},
            "playlist_05": {"label": "Playlist 5", "music_order": ["music_3", "music_1", "music_2"]},
        }
# Note: normalizer will be created after all_events is loaded
if "cleaning_config" not in st.session_state:
    st.session_state.cleaning_config = CleaningConfig()

# Load persisted groups and events (use project path if available)
_project_path = st.session_state.get("current_project")
if "groups" not in st.session_state:
    loaded_groups = load_groups(_project_path)
    if not loaded_groups:
        # Initialize Default Group with canonical events
        st.session_state.groups = {
            "Default": {
                "label": "Default Group",
                "expected_events": DEFAULT_CANONICAL_EVENTS.copy(),
                "selected_sections": []  # ISSUE 7: Add sections selection
            }
        }
    else:
        # Ensure all groups have selected_sections field
        for group_name, group_data in loaded_groups.items():
            if "selected_sections" not in group_data:
                group_data["selected_sections"] = []
        st.session_state.groups = loaded_groups

if "all_events" not in st.session_state:
    loaded_events = load_events(_project_path)
    if not loaded_events:
        st.session_state.all_events = DEFAULT_CANONICAL_EVENTS.copy()
    else:
        st.session_state.all_events = loaded_events

# Initialize sections at startup (so Analysis tab can use them before Setup is visited)
if "sections" not in st.session_state:
    loaded_sections = load_sections(_project_path)
    if not loaded_sections:
        # Default sections - start_events/end_events are lists (any of these events can start/end the section)
        st.session_state.sections = {
            "rest_pre": {"label": "Pre-Rest", "description": "Baseline rest period", "start_events": ["rest_pre_start"], "end_events": ["rest_pre_end"]},
            "measurement": {"label": "Measurement", "description": "Main measurement period", "start_events": ["measurement_start"], "end_events": ["measurement_end"]},
            "pause": {"label": "Pause", "description": "Break between blocks", "start_events": ["pause_start"], "end_events": ["pause_end"]},
            "rest_post": {"label": "Post-Rest", "description": "Post-measurement rest", "start_events": ["rest_post_start"], "end_events": ["rest_post_end"]},
        }
    else:
        # Migrate old format (start_event/end_event) to new format (start_events/end_events)
        for section_data in loaded_sections.values():
            if "start_event" in section_data and "start_events" not in section_data:
                section_data["start_events"] = [section_data.pop("start_event")]
            if "end_event" in section_data and "end_events" not in section_data:
                section_data["end_events"] = [section_data.pop("end_event")]
        st.session_state.sections = loaded_sections

# Create normalizer from GUI events - only recreate when events change
_events_hash = hash(frozenset((k, tuple(v)) for k, v in st.session_state.all_events.items()))
if "normalizer" not in st.session_state or st.session_state.get("_events_hash") != _events_hash:
    st.session_state.normalizer = create_gui_normalizer(st.session_state.all_events)
    st.session_state._events_hash = _events_hash

# Load participant-specific data (groups, playlists, labels, event orders, manual events)
if "participant_groups" not in st.session_state or "event_order" not in st.session_state:
    loaded_participants = load_participants(_project_path)
    if loaded_participants:
        # Extract randomization labels if present
        if "_randomization_labels" in loaded_participants:
            st.session_state.randomization_labels = loaded_participants.pop("_randomization_labels")

        st.session_state.participant_groups = {
            pid: data.get("group", "Default")
            for pid, data in loaded_participants.items()
            if not pid.startswith("_")  # Skip special keys
        }
        st.session_state.participant_randomizations = {
            pid: data.get("randomization", "")
            for pid, data in loaded_participants.items()
            if not pid.startswith("_")
        }
        st.session_state.participant_playlists = {
            pid: data.get("playlist", "")
            for pid, data in loaded_participants.items()
            if not pid.startswith("_")
        }
        st.session_state.participant_labels = {
            pid: data.get("label", "")
            for pid, data in loaded_participants.items()
            if not pid.startswith("_")
        }
        st.session_state.event_order = {
            pid: data.get("event_order", [])
            for pid, data in loaded_participants.items()
            if not pid.startswith("_")
        }
        st.session_state.manual_events = {
            pid: data.get("manual_events", [])
            for pid, data in loaded_participants.items()
            if not pid.startswith("_")
        }
    else:
        st.session_state.participant_groups = {}
        st.session_state.participant_randomizations = {}
        st.session_state.participant_playlists = {}
        st.session_state.participant_labels = {}
        st.session_state.event_order = {}
        st.session_state.manual_events = {}


def save_all_config():
    """Save all configuration to persistent storage."""
    project_path = st.session_state.get("current_project")
    save_groups(st.session_state.groups, project_path)
    save_events(st.session_state.all_events, project_path)
    if hasattr(st.session_state, 'sections'):
        save_sections(st.session_state.sections, project_path)
    save_participant_data()


def save_participant_data():
    """Save participant-specific data (groups, randomizations, event orders, manual events)."""
    project_path = st.session_state.get("current_project")
    participants_data = {}
    all_participant_ids = set(
        list(st.session_state.participant_groups.keys()) +
        list(st.session_state.get("participant_randomizations", {}).keys()) +
        list(st.session_state.event_order.keys()) +
        list(st.session_state.manual_events.keys())
    )

    for pid in all_participant_ids:
        participants_data[pid] = {
            "group": st.session_state.participant_groups.get(pid, "Default"),
            "randomization": st.session_state.get("participant_randomizations", {}).get(pid, ""),
            "event_order": st.session_state.event_order.get(pid, []),
            "manual_events": st.session_state.manual_events.get(pid, []),
        }

    # Store randomization labels as a special entry
    if st.session_state.get("randomization_labels"):
        participants_data["_randomization_labels"] = st.session_state.randomization_labels

    save_participants(participants_data, project_path)


def update_normalizer():
    """Update the normalizer when events are added/removed in GUI.

    ISSUE 1 FIX: This ensures the normalizer always uses current GUI events.
    """
    st.session_state.normalizer = create_gui_normalizer(st.session_state.all_events)
    # Clear cache to force reloading with new normalizer
    cached_load_hrv_logger_preview.clear()


# Cached data loading function for better performance
@st.cache_data(show_spinner=False, ttl=300)
def cached_load_hrv_logger_preview(data_dir_str, pattern, config_dict, gui_events_dict):
    """Cached version of load_hrv_logger_preview for instant navigation.

    ISSUE 1 FIX: Uses GUI events dictionary to create normalizer (not sections.yml).
    """
    data_path = Path(data_dir_str)
    # Reconstruct config from dict (can't cache objects directly)
    config = CleaningConfig(
        rr_min_ms=config_dict["rr_min_ms"],
        rr_max_ms=config_dict["rr_max_ms"],
        sudden_change_pct=config_dict["sudden_change_pct"]
    )
    # ISSUE 1 FIX: Create normalizer from GUI events only
    normalizer = create_gui_normalizer(gui_events_dict)
    return load_hrv_logger_preview(data_path, pattern=pattern, config=config, normalizer=normalizer)


@st.cache_data(show_spinner=False, ttl=300)
def cached_load_participants():
    """Cached version of load_participants for faster access.

    TTL ensures cache is refreshed periodically to prevent memory accumulation.
    """
    return load_participants()


@st.cache_data(show_spinner=False, ttl=600)
def cached_discover_recordings(data_dir_str: str, pattern: str):
    """Cache discovery of recordings to avoid re-scanning directory."""
    data_path = Path(data_dir_str)
    return list(discover_recordings(data_path, pattern=pattern))


@st.cache_data(show_spinner=False, ttl=600)
def cached_load_recording(rr_paths_tuple, events_paths_tuple, participant_id: str):
    """Cache loaded recording data for instant access.

    Uses tuples for paths since lists aren't hashable for caching.
    Returns serializable data: (rr_data, events_data, raw_events)
    """
    from rrational.io.hrv_logger import RecordingBundle
    bundle = RecordingBundle(
        participant_id=participant_id,
        rr_paths=[Path(p) for p in rr_paths_tuple],
        events_paths=[Path(p) for p in events_paths_tuple]
    )
    recording, raw_events, _ = load_recording(bundle)
    # Return serializable data
    return {
        'rr_intervals': [(rr.timestamp, rr.rr_ms, rr.elapsed_ms) for rr in recording.rr_intervals],
        'events': [(e.label, e.timestamp) for e in recording.events],
        'raw_events': raw_events
    }


@st.cache_data(show_spinner=False, ttl=600)
def cached_load_vns_recording(vns_path_str: str, participant_id: str, use_corrected: bool = False):
    """Cache loaded VNS recording data for instant access."""
    from rrational.io.vns_analyse import VNSRecordingBundle, load_vns_recording
    bundle = VNSRecordingBundle(
        participant_id=participant_id,
        file_path=Path(vns_path_str),
    )
    recording = load_vns_recording(bundle, use_corrected=use_corrected)
    return {
        'rr_intervals': [(rr.timestamp, rr.rr_ms, rr.elapsed_ms) for rr in recording.rr_intervals],
        'events': [(e.label, e.timestamp) for e in recording.events],
        'raw_events': [],  # VNS doesn't have duplicate tracking
    }


@st.cache_data(show_spinner=False, ttl=300)
def cached_clean_rr_intervals(rr_data_tuple, config_dict, is_vns_data: bool = False):
    """Cache cleaned RR intervals to avoid recomputation.

    For VNS data, returns ALL intervals without filtering or flagging.
    Artifact detection is handled by NeuroKit2 at analysis time.

    Returns:
        tuple: (rr_data, stats, extra_info)
        - For HRV Logger: rr_data = [(timestamp, rr_ms), ...], cleaned data
        - For VNS: rr_data = [(timestamp, rr_ms, is_flagged=False), ...], ALL data, no flags
    """
    from rrational.cleaning.rr import clean_rr_intervals, CleaningStats, RRInterval

    # Reconstruct RR intervals from cached data
    rr_intervals = [RRInterval(timestamp=ts, rr_ms=rr, elapsed_ms=elapsed)
                    for ts, rr, elapsed in rr_data_tuple]

    if is_vns_data:
        # For VNS data: NO filtering, NO flagging
        # - All intervals are kept (timestamps are cumulative, can't remove any)
        # - No visual flagging (artifact detection done by NeuroKit2 at analysis time)
        result = [(rr.timestamp, rr.rr_ms, False) for rr in rr_intervals if rr.timestamp]
        stats = CleaningStats(
            total_samples=len(rr_intervals),
            retained_samples=len(rr_intervals),
            removed_samples=0,
            artifact_ratio=0.0,
            reasons={"out_of_range": 0, "sudden_change": 0}
        )
        return result, stats, {}
    else:
        # For HRV Logger, apply cleaning (real timestamps are independent of RR values)
        config = CleaningConfig(
            rr_min_ms=config_dict["rr_min_ms"],
            rr_max_ms=config_dict["rr_max_ms"],
            sudden_change_pct=config_dict["sudden_change_pct"]
        )
        cleaned, stats = clean_rr_intervals(rr_intervals, config)
        # HRV Logger: use original packet timestamps (align with events, ~1/sec resolution)
        # Multiple beats can share same timestamp - this is correct for plotting
        return [(rr.timestamp, rr.rr_ms) for rr in cleaned if rr.timestamp], stats, {}


@st.cache_data(show_spinner=False, ttl=300)
def cached_quality_analysis(rr_values_tuple, timestamps_tuple):
    """Cache quality changepoint detection results."""
    rr_list = list(rr_values_tuple)
    timestamps_list = list(timestamps_tuple)
    result = detect_quality_changepoints(rr_list, change_type="var")
    # Add timestamps to segment stats
    n_ts = len(timestamps_list)
    for seg_stats in result["segment_stats"]:
        start_idx = seg_stats["start_idx"]
        end_idx = min(seg_stats["end_idx"], n_ts - 1)
        seg_stats["start_time"] = timestamps_list[start_idx] if start_idx < n_ts else None
        seg_stats["end_time"] = timestamps_list[end_idx] if end_idx < n_ts else None
    return result


def generate_artifact_diagnostic_plots(rr_values: list[float]) -> bytes | None:
    """Generate NeuroKit2 diagnostic plots for artifact detection.

    This creates the same visualization as signal_fixpeaks(show=True):
    - Artifact types plot (heart period with marked artifacts by type)
    - Consecutive-difference criterion
    - Difference-from-median criterion
    - Subspace classification plots

    Args:
        rr_values: List of RR intervals in milliseconds

    Returns:
        PNG image bytes, or None if generation fails

    Raises:
        Exception: Re-raises any exception with detailed message for debugging
    """
    if not NEUROKIT_AVAILABLE:
        raise ValueError("NeuroKit2 not available for diagnostic plots")
    if len(rr_values) < 10:
        raise ValueError(f"Not enough RR values for diagnostic plots: {len(rr_values)}")

    import io
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt

    # Force Agg backend for figure generation (Streamlit may use different backend)
    original_backend = matplotlib.get_backend()
    matplotlib.use('Agg', force=True)

    # Store the current figure list
    existing_figs = set(plt.get_fignums())

    nk = get_neurokit()
    rr_array = np.array(rr_values, dtype=float)

    # Create peak indices from RR intervals
    peak_indices = np.cumsum(rr_array).astype(int)
    peak_indices = np.insert(peak_indices, 0, 0)

    # Temporarily disable interactive mode and plt.show()
    was_interactive = plt.isinteractive()
    plt.ioff()

    # Monkey-patch plt.show to prevent it from doing anything
    original_show = plt.show
    plt.show = lambda *args, **kwargs: None

    try:
        # Run signal_fixpeaks with show=True to generate the diagnostic figure
        info, corrected_peaks = nk.signal_fixpeaks(
            peak_indices,
            sampling_rate=1000,
            iterative=True,
            method="Kubios",
            show=True,
        )

        # Find the new figure(s) created by signal_fixpeaks
        new_figs = set(plt.get_fignums()) - existing_figs

        fig = None
        if new_figs:
            # Get the first new figure (should be the diagnostic plot)
            fig_num = min(new_figs)
            fig = plt.figure(fig_num)
        else:
            # Fallback: try to get current figure
            fig = plt.gcf()
            if not fig.get_axes():
                raise RuntimeError("NeuroKit2 signal_fixpeaks did not create any figures")

        # Convert figure to PNG bytes
        fig.set_size_inches(14, 10)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_bytes = buf.getvalue()

        # Clean up the figure
        plt.close(fig)

        return img_bytes

    finally:
        # Restore plt.show and interactive mode
        plt.show = original_show
        if was_interactive:
            plt.ion()
        # Restore original backend
        try:
            matplotlib.use(original_backend, force=True)
        except Exception:
            pass


@st.cache_data(show_spinner=False, ttl=300)
def cached_artifact_detection(rr_values_tuple, timestamps_tuple, method: str = "threshold",
                               threshold_pct: float = 0.20, segment_beats: int = 300):
    """Cache artifact detection results with indices AND corrected RR from NeuroKit2.

    Args:
        rr_values_tuple: Tuple of RR interval values in ms
        timestamps_tuple: Tuple of corresponding timestamps
        method: Detection method - "threshold" (simple), "kubios"/"lipponen2019" (single pass),
                or "kubios_segmented"/"lipponen2019_segmented" (segmented for long recordings)
        threshold_pct: For threshold method - max allowed change between beats (0.20 = 20%)
        segment_beats: For segmented methods - number of beats per segment (default: 300 = ~5 min)

    Returns dict with artifact indices mapped to timestamps for plotting.
    Note: "lipponen2019" uses NeuroKit2's Kubios method which implements
          Lipponen & Tarvainen (2019) artifact detection algorithm.

    The corrected_rr field contains RR intervals corrected by NeuroKit2's signal_fixpeaks
    (Kubios algorithm), NOT custom interpolation.
    """
    rr_list = list(rr_values_tuple)
    timestamps_list = list(timestamps_tuple)

    if len(rr_list) < 10:
        return {"artifact_indices": [], "artifact_timestamps": [], "artifact_rr": [],
                "total_artifacts": 0, "artifact_ratio": 0.0, "by_type": {},
                "indices_by_type": {},
                "method": method, "segment_stats": [], "corrected_rr": rr_list,
                "corrected_timestamps": timestamps_list}

    try:
        import numpy as np

        if method == "threshold":
            # Simple threshold-based detection (Malik method)
            # Good for long recordings - detects beats that change > threshold_pct from previous
            rr_array = np.array(rr_list, dtype=float)
            artifact_indices = []

            for i in range(1, len(rr_array)):
                # Check ratio to previous beat
                ratio = rr_array[i] / rr_array[i-1]
                if ratio < (1 - threshold_pct) or ratio > (1 + threshold_pct):
                    artifact_indices.append(i)

            by_type = {"threshold": len(artifact_indices)}
            indices_by_type = {}  # Threshold method doesn't categorize artifacts
            segment_stats = []  # No segments for threshold method

            # Use NeuroKit2's signal_fixpeaks for correction (even with threshold detection)
            corrected_rr = rr_list  # Default to original
            if NEUROKIT_AVAILABLE and artifact_indices:
                try:
                    nk = get_neurokit()
                    peak_indices = np.cumsum(rr_array).astype(int)
                    peak_indices = np.insert(peak_indices, 0, 0)
                    _, corrected_peaks = nk.signal_fixpeaks(
                        peak_indices, sampling_rate=1000, iterative=True, method="Kubios", show=False
                    )
                    corrected_rr = np.diff(corrected_peaks).tolist()
                except Exception:
                    pass  # Keep original if correction fails

        elif method in ("kubios_segmented", "lipponen2019_segmented"):
            # Segmented Lipponen/Kubios - process long recordings in chunks for better sensitivity
            # Lipponen2019 uses NeuroKit2's Kubios method (Lipponen & Tarvainen, 2019)
            if not NEUROKIT_AVAILABLE:
                return {"artifact_indices": [], "artifact_timestamps": [], "artifact_rr": [],
                        "total_artifacts": 0, "artifact_ratio": 0.0, "by_type": {},
                        "indices_by_type": {},
                        "method": method, "segment_stats": [], "corrected_rr": rr_list,
                        "corrected_timestamps": timestamps_list}

            nk = get_neurokit()
            rr_array = np.array(rr_list, dtype=float)
            n_beats = len(rr_array)

            # Initialize combined results
            artifact_indices_set = set()
            by_type = {"ectopic": 0, "missed": 0, "extra": 0, "longshort": 0}
            indices_by_type = {"ectopic": set(), "missed": set(), "extra": set(), "longshort": set()}
            segment_stats = []  # Track per-segment artifact percentages

            # Process in overlapping segments for continuity
            overlap = min(30, segment_beats // 10)  # 10% overlap or 30 beats
            start_idx = 0
            segment_num = 0

            while start_idx < n_beats:
                end_idx = min(start_idx + segment_beats, n_beats)
                segment_rr = rr_array[start_idx:end_idx]

                if len(segment_rr) < 10:
                    break

                # Create peak indices for this segment
                peak_indices = np.cumsum(segment_rr).astype(int)
                peak_indices = np.insert(peak_indices, 0, 0)

                segment_artifacts = 0
                try:
                    info, _ = nk.signal_fixpeaks(
                        peak_indices,
                        sampling_rate=1000,
                        iterative=True,
                        method="Kubios",
                        show=False,
                    )

                    # Collect artifacts and adjust indices to global position
                    segment_artifact_indices = set()
                    for artifact_type in ["ectopic", "missed", "extra", "longshort"]:
                        indices = info.get(artifact_type, [])
                        if isinstance(indices, np.ndarray):
                            indices = indices.tolist()
                        elif not isinstance(indices, list):
                            indices = []

                        # Adjust indices to global position
                        global_indices = [i + start_idx for i in indices if 0 <= i < len(segment_rr)]
                        by_type[artifact_type] += len(global_indices)
                        indices_by_type[artifact_type].update(global_indices)
                        artifact_indices_set.update(global_indices)
                        segment_artifact_indices.update(range(len(indices)))

                    segment_artifacts = len(segment_artifact_indices)

                except Exception:
                    pass  # Skip failed segments

                # Record segment statistics
                segment_num += 1
                segment_pct = (segment_artifacts / len(segment_rr) * 100) if len(segment_rr) > 0 else 0.0
                segment_stats.append({
                    "segment": segment_num,
                    "start_beat": start_idx,
                    "end_beat": end_idx,
                    "n_beats": len(segment_rr),
                    "n_artifacts": segment_artifacts,
                    "artifact_pct": round(segment_pct, 2),
                })

                # Move to next segment (with overlap)
                start_idx = end_idx - overlap if end_idx < n_beats else n_beats

            artifact_indices = sorted(artifact_indices_set)
            # Convert indices_by_type sets to sorted lists
            indices_by_type = {k: sorted(v) for k, v in indices_by_type.items()}

            # Get corrected RR from NeuroKit2 for the full recording
            corrected_rr = rr_list  # Default to original
            if artifact_indices:
                try:
                    peak_indices_full = np.cumsum(rr_array).astype(int)
                    peak_indices_full = np.insert(peak_indices_full, 0, 0)
                    _, corrected_peaks = nk.signal_fixpeaks(
                        peak_indices_full, sampling_rate=1000, iterative=True, method="Kubios", show=False
                    )
                    corrected_rr = np.diff(corrected_peaks).tolist()
                except Exception:
                    pass  # Keep original if correction fails

        elif method in ("kubios", "lipponen2019"):
            # Single-pass Lipponen/Kubios method (Lipponen & Tarvainen, 2019)
            if not NEUROKIT_AVAILABLE:
                return {"artifact_indices": [], "artifact_timestamps": [], "artifact_rr": [],
                        "total_artifacts": 0, "artifact_ratio": 0.0, "by_type": {},
                        "indices_by_type": {},
                        "method": method, "segment_stats": [], "corrected_rr": rr_list,
                        "corrected_timestamps": timestamps_list}

            nk = get_neurokit()
            rr_array = np.array(rr_list, dtype=float)
            peak_indices = np.cumsum(rr_array).astype(int)
            peak_indices = np.insert(peak_indices, 0, 0)

            # Use BOTH outputs: info for detection, corrected_peaks for correction
            info, corrected_peaks = nk.signal_fixpeaks(
                peak_indices,
                sampling_rate=1000,
                iterative=True,
                method="Kubios",
                show=False,
            )

            # Collect all artifact indices by type
            artifact_indices_set = set()
            by_type = {}
            indices_by_type = {}

            for artifact_type in ["ectopic", "missed", "extra", "longshort"]:
                indices = info.get(artifact_type, [])
                if isinstance(indices, np.ndarray):
                    indices = indices.tolist()
                elif not isinstance(indices, list):
                    indices = []
                by_type[artifact_type] = len(indices)
                indices_by_type[artifact_type] = sorted(indices)
                artifact_indices_set.update(indices)

            artifact_indices = sorted(artifact_indices_set)
            segment_stats = []  # No segments for single-pass methods

            # Get corrected RR from NeuroKit2's corrected peaks
            corrected_rr = np.diff(corrected_peaks).tolist()

        else:
            # Unknown method, fall back to threshold
            artifact_indices = []
            by_type = {}
            indices_by_type = {}
            segment_stats = []
            corrected_rr = rr_list  # No correction for unknown method

        # Filter to valid range and get timestamps/values
        valid_indices = [i for i in artifact_indices if 0 <= i < len(timestamps_list)]
        artifact_timestamps = [timestamps_list[i] for i in valid_indices]
        artifact_rr = [rr_list[i] for i in valid_indices]

        return {
            "artifact_indices": valid_indices,
            "artifact_timestamps": artifact_timestamps,
            "artifact_rr": artifact_rr,
            "total_artifacts": len(valid_indices),
            "artifact_ratio": len(valid_indices) / len(rr_list) if rr_list else 0.0,
            "by_type": by_type,
            "indices_by_type": indices_by_type,
            "method": method,
            "segment_stats": segment_stats,
            "corrected_rr": corrected_rr,
            "corrected_timestamps": timestamps_list,
        }
    except Exception:
        return {"artifact_indices": [], "artifact_timestamps": [], "artifact_rr": [],
                "total_artifacts": 0, "artifact_ratio": 0.0, "by_type": {},
                "indices_by_type": {},
                "method": method, "segment_stats": [], "corrected_rr": rr_list,
                "corrected_timestamps": timestamps_list}


def run_segmented_artifact_detection_at_gaps(
    rr_values: list,
    timestamps: list,
    gap_adjacent_indices: set,
    method: str = "lipponen2019_segmented",
    threshold_pct: float = 0.20,
    segment_beats: int = 300,
) -> dict:
    """Run artifact detection independently on each gap-separated segment.

    When gaps are treated as segment boundaries, this function:
    1. Splits the RR data at gap positions (gap_adjacent_indices mark the first beat after each gap)
    2. Runs artifact detection on each segment independently
    3. Merges results with correct index mapping back to the original data

    This is more scientifically rigorous than post-filtering because:
    - Each segment's median/statistics are computed independently
    - Artifacts near segment start/end don't influence other segments
    - The Lipponen algorithm uses local patterns which should be segment-specific

    Args:
        rr_values: List of RR intervals in ms
        timestamps: List of corresponding timestamps
        gap_adjacent_indices: Set of indices marking first beat after each gap
        method: Detection method (passed to cached_artifact_detection)
        threshold_pct: Threshold for threshold method
        segment_beats: Segment size for segmented methods

    Returns:
        Merged artifact result dict with all indices in original coordinate system
    """
    if not gap_adjacent_indices or len(rr_values) < 10:
        # No gaps - run normal detection
        return cached_artifact_detection(
            tuple(rr_values), tuple(timestamps),
            method=method, threshold_pct=threshold_pct, segment_beats=segment_beats
        )

    # Sort gap boundary indices
    sorted_boundaries = sorted(gap_adjacent_indices)

    # Build segments: each segment is (start_idx, end_idx) in original coordinates
    segments = []
    prev_end = 0
    for boundary_idx in sorted_boundaries:
        if boundary_idx > prev_end:
            segments.append((prev_end, boundary_idx))
        prev_end = boundary_idx
    # Add final segment
    if prev_end < len(rr_values):
        segments.append((prev_end, len(rr_values)))

    # Filter out segments that are too small (< 10 beats)
    segments = [(start, end) for start, end in segments if (end - start) >= 10]

    if not segments:
        # All segments too small - return empty result
        return {
            "artifact_indices": [],
            "artifact_timestamps": [],
            "artifact_rr": [],
            "total_artifacts": 0,
            "artifact_ratio": 0.0,
            "by_type": {"ectopic": 0, "missed": 0, "extra": 0, "longshort": 0},
            "indices_by_type": {"ectopic": [], "missed": [], "extra": [], "longshort": []},
            "method": method,
            "segment_stats": [],
            "corrected_rr": rr_values,
            "corrected_timestamps": timestamps,
            "gap_segments": segments,
            "segment_boundaries": sorted(gap_adjacent_indices),
        }

    # Run detection on each segment independently
    all_artifact_indices = []
    all_indices_by_type = {"ectopic": [], "missed": [], "extra": [], "longshort": []}
    all_by_type = {"ectopic": 0, "missed": 0, "extra": 0, "longshort": 0}
    all_segment_stats = []  # Lipponen method's internal segment stats
    gap_segment_stats = []  # Per-gap-segment summary stats
    all_corrected_rr = list(rr_values)  # Start with original values

    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        # Extract segment data
        seg_rr = rr_values[seg_start:seg_end]
        seg_ts = timestamps[seg_start:seg_end]

        # Run detection on this segment
        seg_result = cached_artifact_detection(
            tuple(seg_rr), tuple(seg_ts),
            method=method, threshold_pct=threshold_pct, segment_beats=segment_beats
        )

        # Map indices back to original coordinates
        seg_artifact_indices = [i + seg_start for i in seg_result.get("artifact_indices", [])]
        all_artifact_indices.extend(seg_artifact_indices)

        # Map indices_by_type back to original coordinates
        for artifact_type, indices in seg_result.get("indices_by_type", {}).items():
            if artifact_type in all_indices_by_type:
                mapped_indices = [i + seg_start for i in indices]
                all_indices_by_type[artifact_type].extend(mapped_indices)

        # Accumulate counts by type
        for artifact_type, count in seg_result.get("by_type", {}).items():
            if artifact_type in all_by_type:
                all_by_type[artifact_type] += count

        # Collect segment stats with offset info
        for stat in seg_result.get("segment_stats", []):
            stat_copy = dict(stat)
            stat_copy["global_offset"] = seg_start
            all_segment_stats.append(stat_copy)

        # Copy corrected RR values for this segment
        seg_corrected = seg_result.get("corrected_rr", seg_rr)
        if seg_corrected and len(seg_corrected) == (seg_end - seg_start):
            all_corrected_rr[seg_start:seg_end] = seg_corrected

        # Track per-gap-segment summary stats
        seg_n_artifacts = len(seg_artifact_indices)
        seg_n_beats = seg_end - seg_start
        seg_artifact_pct = (seg_n_artifacts / seg_n_beats * 100) if seg_n_beats > 0 else 0.0
        gap_segment_stats.append({
            "gap_segment": seg_idx + 1,
            "start_beat": seg_start,
            "end_beat": seg_end,
            "n_beats": seg_n_beats,
            "n_artifacts": seg_n_artifacts,
            "artifact_pct": round(seg_artifact_pct, 2),
            "by_type": dict(seg_result.get("by_type", {})),
        })

    # Build merged result
    total_artifacts = len(all_artifact_indices)
    artifact_ratio = total_artifacts / len(rr_values) if rr_values else 0.0

    return {
        "artifact_indices": sorted(all_artifact_indices),
        "artifact_timestamps": [timestamps[i] for i in sorted(all_artifact_indices) if 0 <= i < len(timestamps)],
        "artifact_rr": [rr_values[i] for i in sorted(all_artifact_indices) if 0 <= i < len(rr_values)],
        "total_artifacts": total_artifacts,
        "artifact_ratio": artifact_ratio,
        "by_type": all_by_type,
        "indices_by_type": all_indices_by_type,
        "method": method,
        "segment_stats": all_segment_stats,  # Lipponen method's internal segments
        "gap_segment_stats": gap_segment_stats,  # Per-gap-segment summary
        "corrected_rr": all_corrected_rr,
        "corrected_timestamps": timestamps,
        "gap_segments": segments,
        "segment_boundaries": sorted(gap_adjacent_indices),
        "independent_segment_analysis": True,  # Flag that this used true segment analysis
    }


@st.cache_data(show_spinner=False, ttl=300)
def _deprecated_cached_artifact_correction(rr_values_tuple, timestamps_tuple, artifact_indices_tuple):
    """Generate corrected NN intervals by interpolating artifacts.

    Uses cubic interpolation to replace artifact values with estimated values
    based on surrounding valid intervals. This gives a preview of what the
    cleaned data would look like.
    """
    import numpy as np
    from scipy.interpolate import interp1d

    rr_list = list(rr_values_tuple)
    timestamps_list = list(timestamps_tuple)
    artifact_indices = set(artifact_indices_tuple)

    if not artifact_indices or len(rr_list) < 4:
        return {
            "corrected_rr": rr_list,
            "corrected_timestamps": timestamps_list,
            "n_corrected": 0,
        }

    try:
        rr_array = np.array(rr_list, dtype=float)

        # Create mask for valid (non-artifact) indices
        valid_mask = np.array([i not in artifact_indices for i in range(len(rr_array))])
        valid_indices = np.where(valid_mask)[0]
        valid_rr = rr_array[valid_mask]

        if len(valid_indices) < 4:
            return {
                "corrected_rr": rr_list,
                "corrected_timestamps": timestamps_list,
                "n_corrected": 0,
            }

        # Cubic interpolation for smoother correction
        f = interp1d(valid_indices, valid_rr, kind='cubic', fill_value='extrapolate')
        corrected_rr = f(np.arange(len(rr_array)))

        # Keep original values for non-artifacts (only interpolate artifacts)
        corrected_rr[valid_mask] = rr_array[valid_mask]

        return {
            "corrected_rr": corrected_rr.tolist(),
            "corrected_timestamps": timestamps_list,
            "n_corrected": len(artifact_indices),
        }
    except Exception:
        return {
            "corrected_rr": rr_list,
            "corrected_timestamps": timestamps_list,
            "n_corrected": 0,
        }


@st.cache_data(show_spinner=False, ttl=300)
def cached_gap_detection(timestamps_tuple, rr_values_tuple, gap_threshold_s: float):
    """Cache gap detection results to avoid recalculation on every slider change."""
    import numpy as np

    timestamps = list(timestamps_tuple)
    rr_values = list(rr_values_tuple) if rr_values_tuple else None

    if len(timestamps) < 2:
        return {"gaps": [], "total_gaps": 0, "total_gap_duration_s": 0.0, "gap_ratio": 0.0}

    try:
        valid_mask = np.array([t is not None for t in timestamps])
        if not np.any(valid_mask):
            return {"gaps": [], "total_gaps": 0, "total_gap_duration_s": 0.0, "gap_ratio": 0.0}

        ts_seconds = np.array([t.timestamp() if t else np.nan for t in timestamps])
        ts_diff = np.diff(ts_seconds)

        if rr_values is not None and len(rr_values) == len(timestamps):
            rr_array = np.array(rr_values, dtype=float) / 1000.0
            expected_diff = rr_array[1:]
            unexplained_time = ts_diff - expected_diff
            gap_mask = unexplained_time > gap_threshold_s
        else:
            gap_mask = ts_diff > gap_threshold_s
            unexplained_time = ts_diff

        gap_indices = np.where(gap_mask)[0]
        gaps = []
        total_gap_duration = 0.0

        for idx in gap_indices:
            gap_duration = float(unexplained_time[idx]) if rr_values else float(ts_diff[idx])
            if gap_duration > 0:
                gaps.append({
                    "start_time": timestamps[idx],
                    "end_time": timestamps[idx + 1],
                    "duration_s": gap_duration,
                    "start_idx": int(idx),
                    "end_idx": int(idx + 1)
                })
                total_gap_duration += gap_duration

        recording_duration = float(ts_seconds[-1] - ts_seconds[0]) if not np.isnan(ts_seconds[0]) else 0.0

        return {
            "gaps": gaps,
            "total_gaps": len(gaps),
            "total_gap_duration_s": total_gap_duration,
            "gap_ratio": total_gap_duration / recording_duration if recording_duration > 0 else 0.0
        }
    except Exception:
        return {"gaps": [], "total_gaps": 0, "total_gap_duration_s": 0.0, "gap_ratio": 0.0}


@st.cache_data(show_spinner=False, ttl=600)
def cached_build_participant_table(summaries_data: tuple, participant_groups: dict, participant_randomizations: dict,
                                   group_labels: dict, randomization_labels: dict, loaded_participants_keys: tuple):
    """Cache the participant table data to avoid rebuilding on every rerun.

    Args:
        summaries_data: Tuple of tuples (serialized from RecordingSummary objects for hashing)
        participant_groups: Dict of participant -> group assignments
        participant_randomizations: Dict of participant -> randomization assignments
        group_labels: Dict of group_id -> label
        randomization_labels: Dict of rand_value -> label
        loaded_participants_keys: Tuple of saved participant IDs

    Returns:
        Tuple of (participants_data list, issues list)
    """
    # Convert tuples back to dicts for easier access
    summaries_data = [dict(t) for t in summaries_data]
    loaded_set = set(loaded_participants_keys)

    # Build status issues
    issues = []
    high_artifact = sum(1 for s in summaries_data if s["artifact_ratio"] > 0.15)
    if high_artifact:
        issues.append(f"[X] **{high_artifact}** participant(s) with high artifact rates (>15%)")

    with_duplicates = sum(1 for s in summaries_data if s["duplicate_rr_intervals"] > 0)
    if with_duplicates:
        issues.append(f"**{with_duplicates}** participant(s) with duplicate RR intervals")

    with_multi_files = sum(1 for s in summaries_data
                          if s["rr_file_count"] > 1 or s["events_file_count"] > 1)
    if with_multi_files:
        issues.append(f"**{with_multi_files}** participant(s) with multiple files (merged)")

    no_events = sum(1 for s in summaries_data if s["events_detected"] == 0)
    if no_events:
        issues.append(f"? **{no_events}** participant(s) with no events detected")

    # Build participant table data
    participants_data = []
    for s in summaries_data:
        recording_dt_str = s["recording_datetime_str"]

        rr_count = s["rr_file_count"]
        ev_count = s["events_file_count"]
        files_str = f"{rr_count}RR/{ev_count}Ev"
        if rr_count > 1 or ev_count > 1:
            files_str = f"{files_str}"

        quality_badge = get_quality_badge(100, s["artifact_ratio"])

        # Get group with label
        group_id = participant_groups.get(s["participant_id"], "Default")
        group_display = group_labels.get(group_id, group_id)

        # Get randomization with label
        rand_id = participant_randomizations.get(s["participant_id"], "")
        rand_display = randomization_labels.get(rand_id, rand_id) if rand_id else ""

        participants_data.append({
            "Participant": s["participant_id"],
            "Quality": quality_badge,
            "Saved": "Y" if s["participant_id"] in loaded_set else "N",
            "Files": files_str,
            "Date/Time": recording_dt_str,
            "Group": group_display,
            "_group_id": group_id,  # Hidden: actual group ID for saving
            "Randomization": rand_display,
            "_rand_id": rand_id,  # Hidden: actual rand ID for saving
            "Total Beats": s["total_beats"],
            "Retained": s["retained_beats"],
            "Duplicates": s["duplicate_rr_intervals"],
            "Duration (min)": f"{s['duration_s'] / 60:.1f}",
            "Events": s["events_detected"],
            "Total Events": s["events_detected"] + s["duplicate_events"],
            "Duplicate Events": s["duplicate_events"],
            "RR Range (ms)": f"{int(s['rr_min_ms'])}-{int(s['rr_max_ms'])}",
            "Mean RR (ms)": f"{s['rr_mean_ms']:.0f}",
        })

    return participants_data, issues


def serialize_summaries_for_cache():
    """Serialize summaries to a hashable tuple for caching (CACHED in session_state)."""
    if not st.session_state.summaries:
        return ()

    # Check if we already have cached serialization
    summaries = st.session_state.summaries
    cache_key = f"{len(summaries)}:{summaries[0].participant_id if summaries else ''}:{summaries[-1].participant_id if summaries else ''}"

    if st.session_state.get("_serialized_summaries_cache_key") == cache_key:
        return st.session_state._serialized_summaries

    # Build serialized data (only when summaries change)
    result = []
    for s in summaries:
        recording_dt_str = ""
        if s.recording_datetime:
            recording_dt_str = s.recording_datetime.strftime("%Y-%m-%d %H:%M")
        result.append({
            "participant_id": s.participant_id,
            "artifact_ratio": s.artifact_ratio,
            "duplicate_rr_intervals": s.duplicate_rr_intervals,
            "rr_file_count": getattr(s, 'rr_file_count', 1),
            "events_file_count": getattr(s, 'events_file_count', 1 if s.events_detected > 0 else 0),
            "events_detected": s.events_detected,
            "total_beats": s.total_beats,
            "retained_beats": s.retained_beats,
            "duration_s": s.duration_s,
            "duplicate_events": s.duplicate_events,
            "rr_min_ms": s.rr_min_ms,
            "rr_max_ms": s.rr_max_ms,
            "rr_mean_ms": s.rr_mean_ms,
            "recording_datetime_str": recording_dt_str,
        })

    # Cache it
    serialized = tuple(tuple(sorted(d.items())) for d in result)
    st.session_state._serialized_summaries = serialized
    st.session_state._serialized_summaries_cache_key = cache_key
    return serialized


def get_participant_list():
    """Get cached list of participant IDs (O(1) after first call per summaries change)."""
    if not st.session_state.summaries:
        return []
    # Use a simple cache key based on number of summaries and first/last IDs
    summaries = st.session_state.summaries
    cache_key = f"{len(summaries)}:{summaries[0].participant_id if summaries else ''}:{summaries[-1].participant_id if summaries else ''}"
    if st.session_state.get("_participant_list_cache_key") != cache_key:
        st.session_state._participant_list = [s.participant_id for s in summaries]
        st.session_state._participant_list_cache_key = cache_key
    return st.session_state._participant_list


def get_summary_dict():
    """Get cached dict mapping participant_id to summary (O(1) lookup after first call)."""
    if not st.session_state.summaries:
        return {}
    summaries = st.session_state.summaries
    cache_key = f"{len(summaries)}:{summaries[0].participant_id if summaries else ''}:{summaries[-1].participant_id if summaries else ''}"
    if st.session_state.get("_summary_dict_cache_key") != cache_key:
        st.session_state._summary_dict = {s.participant_id: s for s in summaries}
        st.session_state._summary_dict_cache_key = cache_key
    return st.session_state._summary_dict


@st.cache_data(show_spinner=False, ttl=300)
def cached_get_plot_data(timestamps_tuple, rr_values_tuple, participant_id: str, downsample_threshold: int = 5000, flags_tuple=None):
    """Cache processed plot data (NOT the figure - that's slow to serialize).

    Downsamples data if too many points for faster rendering.
    Returns the data needed to build the plot quickly.

    Also pre-calculates sequential timestamps (cumulative RR time) from FULL data
    before downsampling, for Signal Inspection mode.

    Note: Gap detection is NOT done here - it's done separately in the fragment
    using cached_gap_detection() with the user's configurable threshold.

    Args:
        flags_tuple: Optional tuple of booleans indicating flagged (problematic) intervals (VNS only)
    """
    from datetime import timedelta

    timestamps = list(timestamps_tuple)
    rr_values = list(rr_values_tuple)
    flags = list(flags_tuple) if flags_tuple else None
    n_points = len(timestamps)

    # Calculate sequential timestamps from FULL data (before downsampling)
    # Sequential time = cumulative RR time, removes gaps
    sequential_timestamps = []
    total_gap_time_ms = 0  # Computed from actual gaps later

    if timestamps and rr_values:
        base_ts = timestamps[0]
        cumulative_ms = 0

        for i, rr in enumerate(rr_values):
            seq_ts = base_ts + timedelta(milliseconds=cumulative_ms)
            sequential_timestamps.append(seq_ts)
            cumulative_ms += rr

    # Downsample if too many points (keeps every Nth point)
    step = 1
    original_indices = list(range(n_points))  # Maps displayed index -> original index
    if n_points > downsample_threshold:
        step = n_points // downsample_threshold
        timestamps = timestamps[::step]
        rr_values = rr_values[::step]
        sequential_timestamps = sequential_timestamps[::step]
        original_indices = original_indices[::step]  # Keep mapping after downsampling
        if flags:
            flags = flags[::step]

    y_min = min(rr_values)
    y_max = max(rr_values)
    y_range = y_max - y_min

    result = {
        'timestamps': timestamps,  # Original (real) timestamps
        'sequential_timestamps': sequential_timestamps,  # Cumulative RR time
        'rr_values': rr_values,
        'y_min': y_min,
        'y_max': y_max,
        'y_range': y_range,
        'n_original': n_points,
        'n_displayed': len(timestamps),
        'participant_id': participant_id,
        'original_indices': original_indices,  # Maps displayed index -> original index (for artifact marking)
        'downsample_step': step,  # For mapping gap indices after downsampling
        'total_rr_time_ms': cumulative_ms if timestamps else 0,  # Sum of all RR intervals
    }
    if flags:
        result['flags'] = flags
    return result


@st.fragment
def render_participant_table_fragment():
    """Fragment for participant table - prevents re-render when expanders change.

    IMPORTANT: This must be defined at module level for Streamlit to cache it properly.
    """
    if not st.session_state.summaries:
        return

    # Participants overview table
    st.subheader("Participants Overview")

    # Build group labels dict (group_id -> label)
    group_labels = {gid: gdata.get("label", gid) for gid, gdata in st.session_state.groups.items()}

    # Build randomization labels from playlist_groups (primary source)
    # Merge with any custom labels in randomization_labels (fallback for non-playlist values)
    randomization_labels = {}
    for pl_id, pl_data in st.session_state.get("playlist_groups", {}).items():
        randomization_labels[pl_id] = pl_data.get("label", pl_id)
    # Add any custom labels not in playlist_groups
    for rand_id, label in st.session_state.get("randomization_labels", {}).items():
        if rand_id not in randomization_labels:
            randomization_labels[rand_id] = label

    # Use CACHED participant table building (avoids expensive loops on every rerun)
    loaded_participants = cached_load_participants()
    participants_data, issues = cached_build_participant_table(
        serialize_summaries_for_cache(),
        dict(st.session_state.participant_groups),
        dict(st.session_state.participant_randomizations),
        group_labels,
        randomization_labels,
        tuple(loaded_participants.keys())
    )

    total_participants = len(st.session_state.summaries)

    # Display status summary (pre-computed in cached function)
    if issues:
        with st.container():
            st.markdown("**Issues Detected:**")
            for issue in issues:
                st.markdown(f"- {issue}")
            st.markdown("---")
    else:
        st.success(f"All {total_participants} participants look good! No issues detected.")

    # Cache DataFrame creation (avoid rebuilding on every rerun)
    df_cache_key = f"df_{len(participants_data)}_{participants_data[0]['Participant'] if participants_data else ''}"
    if st.session_state.get("_df_participants_cache_key") != df_cache_key:
        st.session_state._df_participants = get_pandas().DataFrame(participants_data)
        st.session_state._df_participants_cache_key = df_cache_key
    df_participants = st.session_state._df_participants

    # Build label -> ID lookup for saving
    label_to_group_id = {gdata.get("label", gid): gid for gid, gdata in st.session_state.groups.items()}
    group_label_options = list(label_to_group_id.keys())

    # Editable dataframe with better column config
    edited_df = st.data_editor(
        df_participants,
        column_config={
            "Participant": st.column_config.TextColumn(
                "Participant",
                disabled=True,
                width="medium",
            ),
            "Quality": st.column_config.TextColumn(
                "Quality",
                disabled=True,
                width="small",
                help="Good (<5% artifacts), Moderate (5-15%), Poor (>15%)",
            ),
            "Saved": st.column_config.TextColumn(
                "Saved",
                disabled=True,
                width="small",
            ),
            "Files": st.column_config.TextColumn(
                "Files",
                disabled=True,
                width="small",
                help="RR files / Events files. Indicates multiple files (merged from restarts)",
            ),
            "Group": st.column_config.SelectboxColumn(
                "Group",
                options=group_label_options,
                required=True,
                help="Assign participant to a group (changes save automatically)",
                width="medium",
            ),
            "_group_id": None,  # Hide this column
            "Randomization": st.column_config.TextColumn(
                "Randomization",
                help="Randomization group (e.g., R1, R2). Edit labels in 'Manage Labels' below.",
                width="small",
                disabled=True,  # Make read-only since we show labels
            ),
            "_rand_id": None,  # Hide this column
            "Total Beats": st.column_config.NumberColumn(
                "Total Beats",
                disabled=True,
                format="%d",
            ),
            "Retained": st.column_config.NumberColumn(
                "Retained",
                disabled=True,
                format="%d",
            ),
            "Total Events": st.column_config.NumberColumn(
                "Total Events",
                disabled=True,
                format="%d",
                help="Total number of events detected",
            ),
            "Duplicate Events": st.column_config.NumberColumn(
                "Duplicate Events",
                disabled=True,
                format="%d",
                help="Number of duplicate event occurrences",
            ),
        },
        width='stretch',
        hide_index=True,
        key="participants_table",
        disabled=["Participant", "Saved", "Date/Time", "Total Beats", "Retained", "Duplicates", "Duration (min)", "Events", "Total Events", "Duplicate Events", "RR Range (ms)", "Mean RR (ms)"]
    )

    # Auto-save group assignments when changed (map label back to group ID)
    edited_groups = dict(zip(edited_df["Participant"], edited_df["Group"]))
    groups_changed = False
    for pid, new_group_label in edited_groups.items():
        # Map label back to group ID (fall back to label if not found)
        new_group_id = label_to_group_id.get(new_group_label, new_group_label)
        if st.session_state.participant_groups.get(pid) != new_group_id:
            st.session_state.participant_groups[pid] = new_group_id
            groups_changed = True

    # Auto-save randomization assignments when changed
    edited_randomizations = dict(zip(edited_df["Participant"], edited_df["Randomization"]))
    randomizations_changed = False
    for pid, new_rand in edited_randomizations.items():
        current_rand = st.session_state.participant_randomizations.get(pid, "")
        if current_rand != new_rand:
            st.session_state.participant_randomizations[pid] = new_rand
            randomizations_changed = True

    # Auto-save if changes detected
    if groups_changed or randomizations_changed:
        save_participant_data()
        cached_load_participants.clear()
        if groups_changed and randomizations_changed:
            show_toast("Group and randomization assignments saved", icon="success")
        elif groups_changed:
            show_toast("Group assignments saved", icon="success")
        else:
            show_toast("Randomization assignments saved", icon="success")
        # Rerun to immediately reflect saved changes (fixes double-click issue)
        st.rerun()

    # Cache duplicate detection (only changes when data is reloaded)
    dup_cache_key = f"dup_{len(participants_data)}"
    if st.session_state.get("_high_duplicates_cache_key") != dup_cache_key:
        st.session_state._high_duplicates = [
            (p["Participant"], p["Duplicates"])
            for p in participants_data if p["Duplicates"] > 0
        ]
        st.session_state._high_duplicates_cache_key = dup_cache_key
    high_duplicates = st.session_state._high_duplicates

    if high_duplicates:
        st.warning(
            f"**Duplicate RR intervals detected!** "
            f"{len(high_duplicates)} participant(s) have duplicate RR intervals that were removed. "
            f"Check the 'Duplicates' column for details."
        )
        with st.expander("Show participants with duplicates"):
            for pid, dup_count in high_duplicates:
                st.text(f"• {pid}: {dup_count} duplicates removed")

    # CSV Import Section (recommended for bulk assignments)
    with st.expander("Import Assignments from CSV (Recommended)", expanded=False):
        st.markdown("""
        **Import group and randomization assignments** from your study's master CSV file.

        Default column names: `code` (participant ID), `group`, `playlist` (randomization)
        """)

        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="assignment_csv_upload")

        if uploaded_file is not None:
            try:
                import_df = get_pandas().read_csv(uploaded_file)
                columns = list(import_df.columns)
                st.write(f"Found {len(import_df)} rows. Columns: {columns}")

                # Smart defaults: look for common column names
                def find_default(options, defaults):
                    for d in defaults:
                        for col in options:
                            if col.lower() == d.lower():
                                return col
                    return ""

                default_id = find_default(columns, ["code", "id", "participant", "participant_id", "subject"])
                default_group = find_default(columns, ["group", "condition", "gruppe"])
                default_rand = find_default(columns, ["playlist", "randomization", "randomisation", "rand"])

                col1, col2, col3 = st.columns(3)
                with col1:
                    id_options = [""] + columns
                    id_idx = id_options.index(default_id) if default_id in id_options else 0
                    id_col = st.selectbox(
                        "Participant ID column",
                        options=id_options,
                        index=id_idx,
                        key="import_id_col"
                    )
                with col2:
                    group_options = ["(skip)"] + columns
                    group_idx = group_options.index(default_group) if default_group in group_options else 0
                    group_col = st.selectbox(
                        "Group column",
                        options=group_options,
                        index=group_idx,
                        key="import_group_col"
                    )
                with col3:
                    rand_options = ["(skip)"] + columns
                    rand_idx = rand_options.index(default_rand) if default_rand in rand_options else 0
                    rand_col = st.selectbox(
                        "Randomization column",
                        options=rand_options,
                        index=rand_idx,
                        key="import_rand_col"
                    )

                use_group = group_col and group_col != "(skip)"
                use_rand = rand_col and rand_col != "(skip)"

                if id_col and (use_group or use_rand):
                    # Preview matches
                    participant_ids = set(get_participant_list())
                    import_ids = set(import_df[id_col].astype(str))
                    matches = participant_ids & import_ids
                    missing = participant_ids - import_ids

                    st.success(f"Matching participants: **{len(matches)}** / {len(participant_ids)}")
                    if missing:
                        st.warning(f"Not found in CSV: {', '.join(sorted(missing)[:5])}{'...' if len(missing) > 5 else ''}")

                    # Show preview
                    if matches:
                        preview_data = []
                        for _, row in import_df.head(5).iterrows():
                            pid = str(row[id_col])
                            if pid in participant_ids:
                                preview_data.append({
                                    "ID": pid,
                                    "Group": str(row[group_col]) if use_group and get_pandas().notna(row[group_col]) else "-",
                                    "Randomization": str(row[rand_col]) if use_rand and get_pandas().notna(row[rand_col]) else "-"
                                })
                        if preview_data:
                            st.write("Preview (first matches):")
                            st.dataframe(get_pandas().DataFrame(preview_data), hide_index=True)

                    if st.button("Apply Assignments", type="primary", key="apply_csv_import"):
                        applied_groups = 0
                        applied_rands = 0
                        for _, row in import_df.iterrows():
                            pid = str(row[id_col])
                            if pid in participant_ids:
                                if use_group and get_pandas().notna(row[group_col]):
                                    new_group = str(row[group_col])
                                    # Create group if it doesn't exist
                                    if new_group not in st.session_state.groups:
                                        st.session_state.groups[new_group] = {
                                            "label": new_group,
                                            "expected_events": {},
                                            "selected_sections": []
                                        }
                                    st.session_state.participant_groups[pid] = new_group
                                    applied_groups += 1
                                if use_rand and get_pandas().notna(row[rand_col]):
                                    new_rand = str(row[rand_col])
                                    # Auto-create playlist group if it doesn't exist
                                    if new_rand and new_rand not in st.session_state.playlist_groups:
                                        st.session_state.playlist_groups[new_rand] = {
                                            "label": new_rand,
                                            "music_order": ["music_1", "music_2", "music_3"]
                                        }
                                    st.session_state.participant_randomizations[pid] = new_rand
                                    applied_rands += 1

                        # Save playlist groups if new ones were created
                        save_playlist_groups(st.session_state.playlist_groups)

                        # Save and clear all caches to force table rebuild
                        save_participant_data()
                        cached_load_participants.clear()
                        cached_build_participant_table.clear()
                        # Clear the DataFrame cache too
                        if "_df_participants_cache_key" in st.session_state:
                            del st.session_state._df_participants_cache_key

                        show_toast(f"Applied {applied_groups} group and {applied_rands} randomization assignments", icon="success")
                        st.rerun()
                elif id_col:
                    st.info("Select at least one column to import (Group or Randomization)")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # Manage Labels Section
    with st.expander("Manage Group & Randomization Labels", expanded=False):
        st.markdown("Add display labels for your groups and randomization conditions.")

        col_labels1, col_labels2 = st.columns(2)

        with col_labels1:
            st.markdown("**Group Labels**")
            # Show all groups with their labels
            groups_changed = False
            for group_name in list(st.session_state.groups.keys()):
                group_data = st.session_state.groups[group_name]
                current_label = group_data.get("label", group_name)
                new_label = st.text_input(
                    f"{group_name}",
                    value=current_label,
                    key=f"group_label_{group_name}",
                    label_visibility="visible"
                )
                if new_label != current_label:
                    st.session_state.groups[group_name]["label"] = new_label
                    groups_changed = True

            if groups_changed:
                auto_save_config()
                show_toast("Group labels saved", icon="success")

        with col_labels2:
            st.markdown("**Randomization Labels**")

            # Get all unique randomization values
            unique_randomizations = set(st.session_state.participant_randomizations.values())
            unique_randomizations.discard("")  # Remove empty string

            # Get playlist group IDs
            playlist_ids = set(st.session_state.get("playlist_groups", {}).keys())

            if unique_randomizations:
                # Show playlist-based randomizations (read-only, from Setup)
                playlist_values = sorted(unique_randomizations & playlist_ids)
                custom_values = sorted(unique_randomizations - playlist_ids)

                if playlist_values:
                    st.caption("From Playlist Groups (edit in Setup > Groups):")
                    for rand_value in playlist_values:
                        pl_label = st.session_state.playlist_groups.get(rand_value, {}).get("label", rand_value)
                        st.text_input(
                            f"{rand_value}",
                            value=pl_label,
                            key=f"rand_label_ro_{rand_value}",
                            disabled=True,
                            label_visibility="visible"
                        )

                if custom_values:
                    st.caption("Custom values (editable):")
                    rand_changed = False
                    for rand_value in custom_values:
                        current_label = st.session_state.get("randomization_labels", {}).get(rand_value, rand_value)
                        new_label = st.text_input(
                            f"{rand_value}",
                            value=current_label,
                            key=f"rand_label_{rand_value}",
                            label_visibility="visible"
                        )
                        if new_label != current_label:
                            if "randomization_labels" not in st.session_state:
                                st.session_state.randomization_labels = {}
                            st.session_state.randomization_labels[rand_value] = new_label
                            rand_changed = True

                    if rand_changed:
                        save_participant_data()
                        show_toast("Randomization labels saved", icon="success")
            else:
                st.caption("No randomization values assigned yet.")

    # Download button (save is now automatic)
    csv_participants = df_participants.to_csv(index=False)
    st.download_button(
        label="Download Participants CSV",
        data=csv_participants,
        file_name="participants_overview.csv",
        mime="text/csv",
        width='content',
    )
    st.caption("Group and randomization assignments save automatically when changed in the table.")


def show_toast(message, icon="success"):
    """Show a toast notification with auto-dismiss."""
    if icon == "success":
        st.toast(f"{message}", icon="✅")
    elif icon == "info":
        st.toast(f"{message}", icon="ℹ️")
    elif icon == "warning":
        st.toast(f"{message}", icon="⚠️")
    elif icon == "error":
        st.toast(f"{message}", icon="❌")
    else:
        st.toast(message)


def auto_save_config():
    """Auto-save configuration with non-intrusive feedback."""
    save_all_config()
    # Store save timestamp for UI feedback
    st.session_state.last_save_time = time.time()


def validate_regex_pattern(pattern):
    """Validate regex pattern and return error message if invalid."""
    try:
        re.compile(pattern)
        return None
    except re.error as e:
        return str(e)


def render_settings_panel():
    """Render the settings panel in the sidebar."""
    # Load current settings
    if "app_settings" not in st.session_state:
        st.session_state.app_settings = load_settings()

    settings = st.session_state.app_settings
    plot_opts = settings.get("plot_options", DEFAULT_SETTINGS["plot_options"])

    # Theme toggle - CSS-only instant switching (no page reload)
    st.caption("**Theme**")
    import streamlit.components.v1 as components

    # CSS-only theme toggle - instant switching via class toggle
    components.html("""
        <style>
            .theme-toggle-container {
                display: flex;
                gap: 8px;
            }
            .theme-btn {
                flex: 1;
                padding: 0.4rem 0.8rem;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-family: inherit;
                transition: all 0.2s;
            }
            .light-btn {
                background: #f0f2f6;
                border: 1px solid #ccc;
                color: #31333F;
            }
            .dark-btn {
                background: #262730;
                border: 1px solid #555;
                color: #fafafa;
            }
            .theme-btn:hover { opacity: 0.8; }
            .theme-btn.active {
                outline: 2px solid #2E86AB;
                outline-offset: 1px;
            }
        </style>
        <script>
            (function() {
                // Get the parent document (Streamlit app)
                var parentDoc = window.parent.document;
                var htmlEl = parentDoc.documentElement;

                // Check saved preference and apply immediately
                var savedTheme = window.parent.localStorage.getItem('music-hrv-theme');
                if (savedTheme === 'dark') {
                    htmlEl.classList.add('dark-theme');
                } else {
                    htmlEl.classList.remove('dark-theme');
                }

                // Update button active states
                function updateButtons() {
                    var isDark = htmlEl.classList.contains('dark-theme');
                    var lightBtn = document.querySelector('.light-btn');
                    var darkBtn = document.querySelector('.dark-btn');
                    if (lightBtn && darkBtn) {
                        lightBtn.classList.toggle('active', !isDark);
                        darkBtn.classList.toggle('active', isDark);
                    }
                }

                // Switch to light theme
                window.switchToLightTheme = function() {
                    htmlEl.classList.remove('dark-theme');
                    window.parent.localStorage.setItem('music-hrv-theme', 'light');
                    // Also set Streamlit's theme for data grids (use 'Custom' to keep using config.toml)
                    var lightTheme = {
                        name: 'Custom',
                        themeInput: {
                            primaryColor: '#2E86AB',
                            backgroundColor: '#FFFFFF',
                            secondaryBackgroundColor: '#F0F2F6',
                            textColor: '#31333F',
                            base: 'light'
                        }
                    };
                    window.parent.localStorage.setItem('stActiveTheme-/-v1', JSON.stringify(lightTheme));
                    updateButtons();
                    updatePlotlyTheme('light');
                };

                // Switch to dark theme
                window.switchToDarkTheme = function() {
                    htmlEl.classList.add('dark-theme');
                    window.parent.localStorage.setItem('music-hrv-theme', 'dark');
                    // Also set Streamlit's theme for data grids (use 'Custom' to keep using config.toml)
                    var darkTheme = {
                        name: 'Custom',
                        themeInput: {
                            primaryColor: '#2E86AB',
                            backgroundColor: '#0E1117',
                            secondaryBackgroundColor: '#262730',
                            textColor: '#FAFAFA',
                            base: 'dark'
                        }
                    };
                    window.parent.localStorage.setItem('stActiveTheme-/-v1', JSON.stringify(darkTheme));
                    updateButtons();
                    updatePlotlyTheme('dark');
                };

                // Update Plotly charts to match theme (including those in iframes)
                function updatePlotlyTheme(theme) {
                    var isDark = theme === 'dark';
                    var bgColor = isDark ? '#0E1117' : '#FFFFFF';
                    var gridColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
                    var textColor = isDark ? '#FAFAFA' : '#31333F';
                    var lineColor = isDark ? '#3D3D4D' : '#E5E5E5';

                    function updatePlot(plot, Plotly) {
                        if (Plotly && plot.data) {
                            try {
                                Plotly.relayout(plot, {
                                    'paper_bgcolor': bgColor,
                                    'plot_bgcolor': bgColor,
                                    'font.color': textColor,
                                    'title.font.color': textColor,
                                    'xaxis.gridcolor': gridColor,
                                    'xaxis.linecolor': lineColor,
                                    'xaxis.tickfont.color': textColor,
                                    'xaxis.title.font.color': textColor,
                                    'yaxis.gridcolor': gridColor,
                                    'yaxis.linecolor': lineColor,
                                    'yaxis.tickfont.color': textColor,
                                    'yaxis.title.font.color': textColor,
                                    'legend.font.color': textColor
                                });
                            } catch(e) {}
                        }
                    }

                    // Update plots in main document
                    var plots = parentDoc.querySelectorAll('.js-plotly-plot');
                    plots.forEach(function(plot) {
                        updatePlot(plot, window.parent.Plotly);
                    });

                    // Also update plots inside iframes (for plotly_events component)
                    var iframes = parentDoc.querySelectorAll('iframe');
                    iframes.forEach(function(iframe) {
                        try {
                            var iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                            var iframePlots = iframeDoc.querySelectorAll('.js-plotly-plot');
                            var iframePlotly = iframe.contentWindow.Plotly;
                            iframePlots.forEach(function(plot) {
                                updatePlot(plot, iframePlotly);
                            });
                        } catch(e) {} // Cross-origin iframes will throw
                    });
                }

                // Initialize on load
                setTimeout(function() {
                    updateButtons();
                    // Apply saved theme to any existing Plotly charts
                    var savedTheme = window.parent.localStorage.getItem('music-hrv-theme') || 'light';
                    updatePlotlyTheme(savedTheme);
                }, 100);
                // Note: MutationObserver for Plotly updates is in apply_custom_css() - no need to duplicate
            })();
        </script>
        <div class="theme-toggle-container">
            <button class="theme-btn light-btn" onclick="switchToLightTheme()">Light</button>
            <button class="theme-btn dark-btn" onclick="switchToDarkTheme()">Dark</button>
        </div>
    """, height=45)

    # Live accent color picker
    st.caption("**Accent Color**")
    saved_accent = settings.get("accent_color", "#2E86AB")

    # Color picker with live update via JavaScript
    components.html(f"""
        <style>
            .color-picker-container {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .color-input {{
                width: 50px;
                height: 32px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                padding: 0;
            }}
            .color-hex {{
                font-family: monospace;
                font-size: 14px;
                color: inherit;
                background: transparent;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 4px 8px;
                width: 80px;
            }}
        </style>
        <script>
            function updateAccentColor(color) {{
                var parentDoc = window.parent.document;
                var root = parentDoc.documentElement;

                // Update CSS custom properties
                root.style.setProperty('--accent-primary', color);

                // Calculate hover color (slightly darker)
                var hoverColor = adjustBrightness(color, -20);
                root.style.setProperty('--accent-hover', hoverColor);

                // Save to localStorage
                window.parent.localStorage.setItem('music-hrv-accent', color);

                // Update hex display
                var hexInput = document.getElementById('color-hex-input');
                if (hexInput) hexInput.value = color;

                // Inject dynamic CSS to override Streamlit's st-* classes
                var styleId = 'music-hrv-accent-override';
                var existingStyle = parentDoc.getElementById(styleId);
                if (existingStyle) existingStyle.remove();

                var styleTag = parentDoc.createElement('style');
                styleTag.id = styleId;
                styleTag.textContent = `
                    /* Dynamic accent color override for Streamlit components */
                    .stCheckbox label > span:first-child,
                    .stCheckbox span[class*="st-ch"],
                    .stCheckbox span[class*="st-c"]:first-child {{
                        background-color: ${{color}} !important;
                        border-color: ${{color}} !important;
                    }}
                    .stCheckbox input:not(:checked) + span,
                    .stCheckbox input[aria-checked="false"] + span {{
                        background-color: var(--input-bg, #fff) !important;
                        border-color: var(--input-border, #ccc) !important;
                    }}
                    .stTabs [data-baseweb="tab"][aria-selected="true"],
                    .stTabs button[role="tab"][aria-selected="true"] {{
                        color: ${{color}} !important;
                        border-bottom-color: ${{color}} !important;
                    }}
                    .stTabs [data-baseweb="tab-highlight"] {{
                        background-color: ${{color}} !important;
                    }}
                    .stButton > button {{
                        background-color: ${{color}} !important;
                    }}
                    .stButton > button:hover {{
                        background-color: ${{hoverColor}} !important;
                    }}
                `;
                parentDoc.head.appendChild(styleTag);

                // Update Plotly charts accent colors if present
                var plots = parentDoc.querySelectorAll('.js-plotly-plot');
                plots.forEach(function(plot) {{
                    if (window.parent.Plotly && plot.data) {{
                        // Update trace colors if they use the accent
                        plot.data.forEach(function(trace, i) {{
                            if (trace.marker && trace.marker.color === '#2E86AB') {{
                                window.parent.Plotly.restyle(plot, {{'marker.color': color}}, [i]);
                            }}
                        }});
                    }}
                }});
            }}

            function adjustBrightness(hex, percent) {{
                var num = parseInt(hex.slice(1), 16);
                var r = Math.min(255, Math.max(0, (num >> 16) + percent));
                var g = Math.min(255, Math.max(0, ((num >> 8) & 0x00FF) + percent));
                var b = Math.min(255, Math.max(0, (num & 0x0000FF) + percent));
                return '#' + (0x1000000 + r * 0x10000 + g * 0x100 + b).toString(16).slice(1);
            }}

            // Load saved accent color on init
            (function() {{
                var savedAccent = window.parent.localStorage.getItem('music-hrv-accent') || '{saved_accent}';
                updateAccentColor(savedAccent);
                document.getElementById('color-picker').value = savedAccent;
                document.getElementById('color-hex-input').value = savedAccent;
            }})();

            function onHexInput(e) {{
                var hex = e.target.value;
                if (/^#[0-9A-Fa-f]{{6}}$/.test(hex)) {{
                    document.getElementById('color-picker').value = hex;
                    updateAccentColor(hex);
                }}
            }}
        </script>
        <div class="color-picker-container">
            <input type="color" id="color-picker" class="color-input"
                   value="{saved_accent}"
                   onchange="updateAccentColor(this.value)"
                   oninput="updateAccentColor(this.value)">
            <input type="text" id="color-hex-input" class="color-hex"
                   value="{saved_accent}"
                   onchange="onHexInput(event)"
                   placeholder="#2E86AB">
        </div>
    """, height=45)

    st.caption("**Default Data Folder**")
    new_folder = st.text_input(
        "Data folder path",
        value=settings.get("data_folder", ""),
        key="settings_data_folder",
        placeholder="Leave empty for file picker",
        label_visibility="collapsed"
    )
    new_auto_load = st.checkbox(
        "Auto-load on startup",
        value=settings.get("auto_load", False),
        key="settings_auto_load",
        help="Automatically load data from the default folder when the app starts"
    )

    st.caption("**Plot Defaults**")
    new_resolution = st.slider(
        "Default resolution",
        min_value=1000,
        max_value=100000,
        value=settings.get("plot_resolution", 5000),
        step=1000,
        key="settings_resolution",
        help="Default number of points to show (higher values for long recordings)"
    )

    new_gap_threshold = st.slider(
        "Gap threshold (s)",
        min_value=1.0,
        max_value=60.0,
        value=float(plot_opts.get("gap_threshold", 15.0)),
        step=1.0,
        key="settings_gap_threshold"
    )

    st.caption("**Show by default**")
    col1, col2 = st.columns(2)
    with col1:
        new_show_events = st.checkbox("Events", value=plot_opts.get("show_events", True), key="settings_show_events")
        new_show_exclusions = st.checkbox("Exclusions", value=plot_opts.get("show_exclusions", True), key="settings_show_exclusions")
        new_show_gaps = st.checkbox("Gaps", value=plot_opts.get("show_gaps", True), key="settings_show_gaps")
    with col2:
        new_show_music_sec = st.checkbox("Sections", value=plot_opts.get("show_music_sections", True), key="settings_show_music_sec")
        new_show_artifacts = st.checkbox("Artifacts", value=plot_opts.get("show_artifacts", False), key="settings_show_artifacts")
        new_show_variability = st.checkbox("Variability", value=plot_opts.get("show_variability", False), key="settings_show_variability")

    st.caption("**Plot Colors**")
    plot_colors = plot_opts.get("colors", {})
    col1, col2 = st.columns(2)
    with col1:
        new_line_color = st.color_picker(
            "RR Line",
            value=plot_colors.get("line", "#2E86AB"),
            key="settings_line_color",
            help="Color for RR interval line"
        )
    with col2:
        new_artifact_color = st.color_picker(
            "Artifacts",
            value=plot_colors.get("artifact", "#FF6B6B"),
            key="settings_artifact_color",
            help="Color for flagged/artifact intervals"
        )

    # Save button
    if st.button("Save Settings", key="save_settings_btn", width="stretch"):
        # Validate auto_load requires folder to be set
        if new_auto_load and not new_folder:
            st.error("Auto-load requires a data folder to be set. Please enter a folder path above.")
            st.stop()

        # Validate folder exists if set
        if new_folder:
            folder_path = Path(new_folder)
            if not folder_path.exists():
                st.warning(f"Folder not found: {new_folder}. Auto-load will fail until the folder exists.")
            elif not folder_path.is_dir():
                st.error(f"Path is not a folder: {new_folder}")
                st.stop()

        new_settings = {
            "data_folder": new_folder,
            "auto_load": new_auto_load,
            "plot_resolution": new_resolution,
            "plot_options": {
                "show_events": new_show_events,
                "show_exclusions": new_show_exclusions,
                "show_music_sections": new_show_music_sec,
                "show_music_events": plot_opts.get("show_music_events", False),
                "show_artifacts": new_show_artifacts,
                "show_variability": new_show_variability,
                "show_gaps": new_show_gaps,
                "gap_threshold": new_gap_threshold,
                "colors": {
                    "line": new_line_color,
                    "artifact": new_artifact_color,
                },
            }
        }
        save_settings(new_settings)
        st.session_state.app_settings = new_settings
        st.toast("Settings saved!")


@st.fragment
def render_rr_plot_fragment(participant_id: str):
    """Render the RR interval plot as a fragment to prevent full page reruns.

    This fragment reads from session state:
    - plot_data_{participant_id}: Downsampled plot data
    - participant_events[participant_id]: Events to display
    - gaps_{participant_id}: Gap detection results (optional)

    When plot options change, only this fragment reruns, not the entire page.
    """
    plot_data_key = f"plot_data_{participant_id}"
    if plot_data_key not in st.session_state:
        st.warning("Plot data not loaded yet.")
        return

    plot_data = st.session_state[plot_data_key]
    stored_data = st.session_state.participant_events.get(participant_id, {})

    # Get plotly (lazy import)
    go, plotly_events = get_plotly()
    if go is None:
        st.warning("Plotly is not installed. Please install it with: `pip install plotly streamlit-plotly-events`")
        return

    # Always use Scattergl for performance
    ScatterType = go.Scattergl

    # Determine source_app (check plot_data first, then fall back to summary)
    source_app = plot_data.get('source_app')
    if not source_app:
        # Fall back to checking the summary
        summary = get_summary_dict().get(participant_id)
        source_app = getattr(summary, 'source_app', 'HRV Logger') if summary else 'HRV Logger'
    is_vns_data = (source_app == "VNS Analyse")

    # Check interaction mode for timestamp transformation
    plot_mode_key = f"plot_mode_{participant_id}"
    current_mode = st.session_state.get(plot_mode_key, "Add Events")
    use_sequential_timestamps = (current_mode == "Signal Inspection" and not is_vns_data)

    # Original timestamps are always available for event alignment
    original_timestamps = plot_data['timestamps']

    # Pre-calculated sequential timestamps from cached_get_plot_data
    # These are calculated from FULL data before downsampling
    sequential_timestamps = plot_data.get('sequential_timestamps', [])
    
    # Build real-to-sequential mapping for event alignment
    real_to_sequential_map = None
    if use_sequential_timestamps and sequential_timestamps:
        real_to_sequential_map = list(zip(original_timestamps, sequential_timestamps))
        # Use sequential timestamps for plotting
        plot_data = dict(plot_data)  # Make a copy
        plot_data['timestamps'] = sequential_timestamps
        plot_data['original_timestamps'] = original_timestamps

    # Show source info and clear any old gap data for VNS
    if is_vns_data:
        st.info(f"**Data source: {source_app}** - Gap detection disabled (timestamps synthesized from RR intervals)")
        # Force clear any old gap data for VNS participants
        st.session_state[f"gaps_{participant_id}"] = {
            "gaps": [], "total_gaps": 0, "total_gap_duration_s": 0.0, "gap_ratio": 0.0, "vns_note": True
        }

    # Keyboard shortcut for Signal Inspection mode (inside fragment for fast response)
    if current_mode == "Signal Inspection":
        zoom_key = f"inspection_zoom_{participant_id}"

        # Compact zoom controls
        col_zoom, col_auto, col_info = st.columns([2, 1, 3])
        with col_zoom:
            if st.button("Inspection Zoom", key=f"frag_zoom_btn_{participant_id}",
                        help="Reset Y-axis to 400-1200ms, X to 60s window (press I)"):
                st.session_state[zoom_key] = {
                    'y_min': 400,
                    'y_max': 1200,
                    'x_window_seconds': 60,
                    'center_on_mean': True
                }
        with col_auto:
            if zoom_key in st.session_state:
                if st.button("Auto", key=f"frag_clear_zoom_{participant_id}",
                            help="Return to auto-scaling"):
                    del st.session_state[zoom_key]
        with col_info:
            st.caption("Drag to pan, scroll to zoom, I / arrow keys")

        # Inject JavaScript for instant keyboard shortcuts (client-side, no server roundtrip)
        import streamlit.components.v1 as components
        components.html("""
        <script>
        (function() {
            // Only attach once per page load
            if (window._rrationalKeysAttached) return;
            window._rrationalKeysAttached = true;

            const PAN_SECONDS = 10000; // 10 seconds in milliseconds

            // Find Plotly plot in document or iframes
            const findPlot = () => {
                const findInDoc = (doc) => {
                    try {
                        const div = doc.querySelector('.js-plotly-plot');
                        if (div) return {div, Plotly: doc.defaultView.Plotly};
                    } catch(err) {}
                    return null;
                };

                let result = findInDoc(document);
                if (result) return result;

                // Check iframes (streamlit-plotly-events uses iframe)
                const iframes = document.querySelectorAll('iframe');
                for (const iframe of iframes) {
                    try {
                        const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                        result = findInDoc(iframeDoc);
                        if (result) return result;
                    } catch(err) {} // Cross-origin will fail
                }
                return null;
            };

            // Click button without scrolling
            const clickButtonNoScroll = (selector) => {
                const scrollPos = window.scrollY;
                const btn = document.querySelector(selector);
                if (btn) {
                    btn.click();
                    // Restore scroll position after Streamlit rerun
                    requestAnimationFrame(() => {
                        window.scrollTo(0, scrollPos);
                        setTimeout(() => window.scrollTo(0, scrollPos), 50);
                        setTimeout(() => window.scrollTo(0, scrollPos), 150);
                    });
                }
            };

            document.addEventListener('keydown', function(e) {
                // Don't trigger if typing in an input
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

                // Handle "I" key - click Inspection Zoom button
                if (e.key === 'i' || e.key === 'I') {
                    clickButtonNoScroll('button[data-testid="stBaseButton-secondary"]');
                    e.preventDefault();
                    return;
                }

                // Handle arrow keys - instant pan via Plotly
                if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;

                const plot = findPlot();
                if (!plot) return;

                const {div, Plotly} = plot;
                if (!div || !Plotly || !div.layout || !div.layout.xaxis) return;

                const xaxis = div.layout.xaxis;
                if (!xaxis.range || xaxis.range.length < 2) return;

                const shift = e.key === 'ArrowRight' ? PAN_SECONDS : -PAN_SECONDS;
                const newRange = [
                    new Date(new Date(xaxis.range[0]).getTime() + shift),
                    new Date(new Date(xaxis.range[1]).getTime() + shift)
                ];

                Plotly.relayout(div, {'xaxis.range': newRange});
                e.preventDefault();
            });
        })();
        </script>
        """, height=0)

        # Quick Save for Analysis expander
        with st.expander("Quick Save for Analysis", expanded=False):
            # Check for existing .rrational files
            from rrational.gui.rrational_export import find_rrational_files
            from rrational.gui.persistence import load_artifact_corrections, save_artifact_corrections
            data_dir_save = st.session_state.get("data_dir", "")
            project_path_save = st.session_state.get("current_project")
            existing_ready_files = find_rrational_files(participant_id, data_dir_save, project_path_save)

            if existing_ready_files:
                st.success(f"**{len(existing_ready_files)} .rrational file(s) saved** for this participant")
                with st.expander("View saved files", expanded=False):
                    for f in existing_ready_files:
                        segment = f.stem.replace(f"{participant_id}_", "") or "full"
                        mod_time = f.stat().st_mtime
                        from datetime import datetime
                        mod_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
                        st.caption(f"• **{segment}** - {f.name} ({mod_str})")

            # Save Artifact Corrections button (more accessible location)
            st.markdown("---")
            st.markdown("**Save Artifact Markings**")
            st.caption("Save your artifact markings to continue in future sessions")

            manual_artifacts_save = st.session_state.get(f"manual_artifacts_{participant_id}", [])
            artifact_exclusions_save = st.session_state.get(f"artifact_exclusions_{participant_id}", set())
            artifact_data_save = st.session_state.get(f"artifacts_{participant_id}", {})
            algo_indices_save = artifact_data_save.get('artifact_indices', [])

            n_manual_save = len(manual_artifacts_save)
            n_excluded_save = len(artifact_exclusions_save) if artifact_exclusions_save else 0
            n_algo_save = len(algo_indices_save)
            has_corrections_save = n_manual_save > 0 or n_excluded_save > 0 or n_algo_save > 0

            saved_corrections_check = load_artifact_corrections(participant_id, data_dir_save, project_path_save)
            has_saved_corrections = saved_corrections_check is not None

            if has_corrections_save:
                # Build summary text
                parts = []
                if n_algo_save > 0:
                    parts.append(f"{n_algo_save} algorithm")
                if n_manual_save > 0:
                    parts.append(f"{n_manual_save} manual")
                if n_excluded_save > 0:
                    parts.append(f"{n_excluded_save} excluded")
                st.write(" + ".join(parts))

                # Get section_key for display
                section_key_save = artifact_data_save.get('section_key', '_full')
                section_display = "Full recording" if section_key_save == "_full" else section_key_save
                if section_key_save.startswith("custom_"):
                    section_display = "Custom range"

                if section_key_save != "_full":
                    st.caption(f"Scope: {section_display}")

                if st.button("Save Artifact Corrections", key=f"sidebar_save_artifacts_{participant_id}",
                            type="primary", width="stretch"):
                    # Get algorithm method and threshold info
                    algo_method = artifact_data_save.get('method', None)
                    algo_threshold = artifact_data_save.get('threshold', None)
                    # Get scope (v1.2+ feature)
                    scope_save = artifact_data_save.get('scope', None)
                    # Get segment_beats for segmented methods
                    segment_beats_save = artifact_data_save.get('segment_beats', None)
                    # Get indices_by_type for artifact type categorization
                    indices_by_type_save = artifact_data_save.get('indices_by_type', None)

                    save_path = save_artifact_corrections(
                        participant_id,
                        manual_artifacts=manual_artifacts_save,
                        artifact_exclusions=artifact_exclusions_save,
                        data_dir=data_dir_save,
                        project_path=project_path_save,
                        algorithm_artifacts=algo_indices_save if algo_indices_save else None,
                        algorithm_method=algo_method,
                        algorithm_threshold=algo_threshold,
                        scope=scope_save,
                        section_key=section_key_save,
                        segment_beats=segment_beats_save,
                        indices_by_type=indices_by_type_save,
                    )

                    # Update loaded_info to reflect current saved state
                    from datetime import datetime
                    st.session_state[f"artifacts_loaded_info_{participant_id}"] = {
                        "saved_at": datetime.now().isoformat(),
                        "algorithm_method": algo_method,
                        "algorithm_threshold": algo_threshold,
                        "n_algorithm": n_algo_save,
                        "n_manual": n_manual_save,
                        "n_excluded": n_excluded_save,
                    }

                    st.success(f"✓ Saved to {save_path.name}")
            elif has_saved_corrections:
                st.success("Artifact corrections saved")
            else:
                st.caption("No artifact markings yet")

            # Export NN as CSV button (easily accessible)
            st.markdown("---")
            st.markdown("**Export Corrected NN Intervals**")
            st.caption("Download CSV with artifact-corrected (interpolated) NN intervals")

            # Get current plot data
            plot_data_nn = st.session_state.get(f"plot_data_{participant_id}", {})
            artifact_data_nn = st.session_state.get(f"artifacts_{participant_id}", {})

            if plot_data_nn and 'timestamps' in plot_data_nn:
                ts_nn = plot_data_nn['timestamps']
                rr_nn = plot_data_nn['rr_values']

                # Collect all artifact indices (algorithm + manual + exclusions reversed)
                algo_indices_nn = set(artifact_data_nn.get('artifact_indices', []))
                manual_indices_nn = set(art.get('plot_idx', -1) for art in manual_artifacts_save)
                # Remove excluded indices (user un-marked them)
                excluded_indices_nn = artifact_exclusions_save if artifact_exclusions_save else set()
                all_artifact_idx_nn = (algo_indices_nn | manual_indices_nn) - excluded_indices_nn

                n_artifacts_nn = len(all_artifact_idx_nn)
                st.write(f"{len(ts_nn)} beats | {n_artifacts_nn} artifacts to correct")

                if st.button("Export NN as CSV", key=f"sidebar_export_nn_{participant_id}",
                            width="stretch"):
                    # Use corrected RR from NeuroKit2's signal_fixpeaks (stored in artifact_data)
                    corrected_rr_nn = artifact_data_nn.get('corrected_rr', rr_nn)
                    if corrected_rr_nn is None or len(corrected_rr_nn) != len(rr_nn):
                        corrected_rr_nn = rr_nn

                    # Create DataFrame for export
                    import io
                    export_data_nn = []
                    for i, (ts, rr_orig, rr_corr) in enumerate(zip(ts_nn, rr_nn, corrected_rr_nn)):
                        ts_str = ts.strftime('%Y-%m-%d %H:%M:%S.%f') if hasattr(ts, 'strftime') else str(ts)
                        is_art = i in all_artifact_idx_nn
                        art_source = ''
                        if is_art:
                            if i in manual_indices_nn:
                                art_source = 'manual'
                            elif i in algo_indices_nn:
                                art_source = 'algorithm'
                        export_data_nn.append({
                            'timestamp': ts_str,
                            'rr_ms': rr_orig,
                            'nn_ms': round(rr_corr, 1),
                            'is_artifact': is_art,
                            'artifact_source': art_source
                        })

                    df_nn = get_pandas().DataFrame(export_data_nn)

                    # Store in session state for download
                    csv_buffer_nn = io.StringIO()
                    df_nn.to_csv(csv_buffer_nn, index=False)
                    st.session_state[f"nn_csv_data_{participant_id}"] = csv_buffer_nn.getvalue()
                    st.session_state[f"nn_csv_ready_{participant_id}"] = True
                    st.rerun()

                # Show download button if CSV is ready
                if st.session_state.get(f"nn_csv_ready_{participant_id}"):
                    csv_data_nn = st.session_state.get(f"nn_csv_data_{participant_id}", "")
                    st.download_button(
                        "Download NN CSV",
                        data=csv_data_nn,
                        file_name=f"{participant_id}_nn.csv",
                        mime="text/csv",
                        key=f"download_nn_{participant_id}",
                        type="primary",
                        width="stretch"
                    )
            else:
                st.caption("Load participant data first")

            # Get available sections
            sections = load_sections()
            section_names = ["Full Recording"] + [s.get("label", name) for name, s in sections.items() if s.get("start_event")]

            save_section = st.selectbox(
                "Segment to save",
                options=section_names,
                key=f"frag_save_section_{participant_id}",
                help="Select a section to export, or export full recording"
            )

            include_corrected = st.checkbox(
                "Include corrected NN intervals",
                value=False,
                key=f"frag_include_corrected_{participant_id}",
                help="Include interpolated NN intervals (artifact-corrected)"
            )

            if st.button("Save (.rrational)", key=f"frag_save_btn_{participant_id}",
                        help="Export data with audit trail for analysis"):
                # Import the export module
                from rrational.gui.rrational_export import (
                    RRationalExport, RRIntervalExport, SegmentDefinition,
                    ArtifactDetection, ManualArtifact, QualityMetrics, ProcessingStep, save_rrational,
                    build_export_filename, get_quality_grade, get_quigley_recommendation
                )
                from datetime import datetime

                # Get current artifact state
                artifacts_key = f"artifacts_{participant_id}"
                manual_artifacts_key = f"manual_artifacts_{participant_id}"
                exclusions_key = f"artifact_exclusions_{participant_id}"

                artifacts_data = st.session_state.get(artifacts_key, {})
                manual_artifacts_list = st.session_state.get(manual_artifacts_key, [])
                excluded_indices = list(st.session_state.get(exclusions_key, set()))

                # Get RR data from plot_data
                rr_values = plot_data.get('rr_values', [])
                timestamps = plot_data.get('timestamps', [])

                # Determine segment info
                segment_name = None
                if save_section != "Full Recording":
                    # Find the section key from the label
                    for name, s in sections.items():
                        if s.get("label", name) == save_section:
                            segment_name = name
                            break

                # Build export data
                now = datetime.now().isoformat()

                # Get source info from summary
                summary = get_summary_dict().get(participant_id)
                source_app = "HRV Logger"
                source_paths = []
                recording_dt = None
                if summary:
                    source_app = getattr(summary, 'source_app', 'HRV Logger')
                    source_paths = getattr(summary, 'rr_paths', []) or []
                    recording_dt = getattr(summary, 'recording_datetime', None)
                    if recording_dt:
                        recording_dt = recording_dt.isoformat() if hasattr(recording_dt, 'isoformat') else str(recording_dt)

                # Build RR interval exports
                rr_exports = []
                for i, (ts, rr) in enumerate(zip(timestamps, rr_values)):
                    ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
                    rr_exports.append(RRIntervalExport(
                        timestamp=ts_str,
                        rr_ms=int(rr),
                        original_idx=i
                    ))

                # Build artifact detection info
                artifact_detection = None
                if artifacts_data:
                    artifact_detection = ArtifactDetection(
                        method=artifacts_data.get('method', 'threshold'),
                        total=artifacts_data.get('total_artifacts', 0),
                        by_type=artifacts_data.get('by_type', {}),
                        indices=artifacts_data.get('artifact_indices', [])
                    )

                # Build manual artifacts
                manual_exports = []
                for ma in manual_artifacts_list:
                    ts_str = ma.get('timestamp', '')
                    if hasattr(ts_str, 'isoformat'):
                        ts_str = ts_str.isoformat()
                    manual_exports.append(ManualArtifact(
                        original_idx=ma.get('original_idx', 0),
                        timestamp=ts_str,
                        rr_value=ma.get('rr_value', 0),
                        marked_at=now
                    ))

                # Calculate final artifact indices
                detected_indices = set(artifacts_data.get('artifact_indices', []))
                manual_indices = set(ma.get('original_idx', 0) for ma in manual_artifacts_list)
                excluded_set = set(excluded_indices)
                final_artifacts = list((detected_indices | manual_indices) - excluded_set)

                # Build quality metrics
                artifact_rate = len(final_artifacts) / len(rr_values) if rr_values else 0
                quality = QualityMetrics(
                    artifact_rate_raw=artifacts_data.get('artifact_ratio', 0),
                    artifact_rate_final=artifact_rate,
                    beats_after_cleaning=len(rr_values),
                    quality_grade=get_quality_grade(artifact_rate),
                    quigley_recommendation=get_quigley_recommendation(artifact_rate, len(rr_values))
                )

                # Build segment definition
                segment_def = SegmentDefinition(
                    type="section" if segment_name else "full_recording",
                    section_name=segment_name,
                )

                # Build audit trail
                steps = [
                    ProcessingStep(step=1, action="export_ready_for_analysis",
                                  timestamp=now, details=f"Exported {len(rr_values)} beats")
                ]

                # Get software versions
                import neurokit2 as nk
                import sys
                software_versions = {
                    "rrational": "0.7.0",
                    "neurokit2": getattr(nk, '__version__', 'unknown'),
                    "python": sys.version.split()[0]
                }

                # Create export object
                export_data = RRationalExport(
                    participant_id=participant_id,
                    export_timestamp=now,
                    exported_by="RRational v0.7.0",
                    source_app=source_app,
                    source_file_paths=[str(p) for p in source_paths],
                    recording_datetime=recording_dt,
                    segment=segment_def,
                    n_beats=len(rr_exports),
                    rr_intervals=rr_exports,
                    artifact_detection=artifact_detection,
                    manual_artifacts=manual_exports,
                    excluded_detected_indices=excluded_indices,
                    final_artifact_indices=final_artifacts,
                    include_corrected=include_corrected,
                    quality=quality,
                    processing_steps=steps,
                    software_versions=software_versions
                )

                # Save to processed folder
                data_dir = st.session_state.get("data_dir")
                if data_dir:
                    from pathlib import Path
                    processed_dir = Path(data_dir).parent / "processed"
                    filename = build_export_filename(participant_id, segment_name)
                    filepath = processed_dir / filename
                    save_rrational(export_data, filepath)
                    st.success(f"Saved: {filepath}")
                else:
                    st.error("No data directory set - cannot save")

    # Plot display options - use saved defaults
    plot_defaults = st.session_state.get("app_settings", {}).get("plot_options", {})
    st.markdown("**Plot Options:**")
    col_opt1, col_opt2, col_opt3, col_opt4 = st.columns(4)
    with col_opt1:
        show_events = st.checkbox("Show events", value=plot_defaults.get("show_events", True),
                                  key=f"frag_show_events_{participant_id}",
                                  help="Show boundary events on plot")
        show_exclusions = st.checkbox("Show exclusions", value=plot_defaults.get("show_exclusions", True),
                                      key=f"frag_show_exclusions_{participant_id}",
                                      help="Show exclusion zones as red rectangles")
    with col_opt2:
        show_music_sections = st.checkbox("Show music sections", value=plot_defaults.get("show_music_sections", True),
                                          key=f"frag_show_music_sec_{participant_id}")
        show_music_events = st.checkbox("Show music events", value=plot_defaults.get("show_music_events", False),
                                        key=f"frag_show_music_evt_{participant_id}")
    with col_opt3:
        show_artifacts = st.checkbox("Show artifacts", value=plot_defaults.get("show_artifacts", True),
                                     key=f"frag_show_artifacts_{participant_id}",
                                     help="Show saved artifacts (manual, validated, algorithm)")
        # Artifact detection settings (only when enabled)
        if show_artifacts:
            # Check if we have loaded artifact settings to use as defaults
            loaded_info = st.session_state.get(f"artifacts_loaded_info_{participant_id}", {})
            loaded_method = loaded_info.get("algorithm_method")
            loaded_threshold = loaded_info.get("algorithm_threshold")

            # Check if we have saved/loaded artifacts
            saved_artifact_data = st.session_state.get(f"artifacts_{participant_id}", {})
            has_saved_artifacts = bool(saved_artifact_data.get("artifact_indices"))

            # Detection mode: saved (default) vs new detection
            detect_new_key = f"detect_new_artifacts_{participant_id}"
            run_new_detection = st.session_state.get(detect_new_key, False)

            # Track expander state separately - keep open while configuring
            expander_open_key = f"artifact_expander_open_{participant_id}"
            # Keep expander open if: explicitly set to open, detection requested, scope is not 'full', or method is segmented
            current_scope = st.session_state.get(f"frag_artifact_scope_{participant_id}", "full")
            current_method = st.session_state.get(f"frag_artifact_method_{participant_id}", "threshold")
            is_configuring = (
                current_scope != "full" or
                current_method in ("kubios_segmented", "lipponen2019_segmented")
            )
            expander_should_open = (
                st.session_state.get(expander_open_key, False) or
                run_new_detection or
                is_configuring
            )

            # Show saved artifact info if available (Clear button moved to results section)
            if has_saved_artifacts:
                n_saved = len(saved_artifact_data.get("artifact_indices", []))
                saved_method = saved_artifact_data.get("method", "unknown")
                st.caption(f"Loaded: {n_saved} artifacts ({saved_method})")

            # New detection settings (in expander to avoid clutter)
            with st.expander("Detect New Artifacts", expanded=expander_should_open):
                # Method display names for dropdown
                method_options = {
                    "threshold": "Threshold (Malik)",
                    "lipponen2019": "Lipponen 2019",
                    "lipponen2019_segmented": "Lipponen 2019 (segmented)",
                }

                # Determine default method index based on loaded settings
                method_keys = list(method_options.keys())
                default_method_idx = 2  # Default to lipponen2019_segmented
                if loaded_method and loaded_method in method_keys:
                    default_method_idx = method_keys.index(loaded_method)

                artifact_method = st.selectbox(
                    "Method",
                    options=method_keys,
                    format_func=lambda x: method_options[x],
                    index=default_method_idx,
                    key=f"frag_artifact_method_{participant_id}",
                    help="**Threshold**: Fast ratio check (>X% change). **Lipponen 2019**: State-of-the-art beat classification (=Kubios). **Segmented**: 5-min chunks for long recordings."
                )
                if artifact_method == "threshold":
                    # Use loaded threshold if available, convert to percentage (0.20 -> 20)
                    default_thresh_pct = 20
                    if loaded_threshold is not None and loaded_method == "threshold":
                        default_thresh_pct = int(loaded_threshold * 100)
                        # Clamp to valid range
                        default_thresh_pct = max(10, min(50, default_thresh_pct))

                    artifact_threshold = st.slider(
                        "Threshold %",
                        min_value=10, max_value=50, value=default_thresh_pct, step=5,
                        key=f"frag_artifact_thresh_{participant_id}",
                        help="Max allowed RR change between beats (20% = Malik method)"
                    ) / 100.0
                    segment_beats = 300  # Default, not used for threshold
                    is_segmented_method = False
                elif artifact_method in ("kubios_segmented", "lipponen2019_segmented"):
                    # For segmented methods, we need segment_beats - will be set after scope selection
                    artifact_threshold = 0.20  # Not used
                    # Placeholder - will be set after scope selection below
                    segment_beats = 300
                    is_segmented_method = True
                else:
                    # kubios or lipponen2019 single-pass methods
                    artifact_threshold = 0.20  # Not used
                    segment_beats = 300  # Not used
                    is_segmented_method = False

                # Gap-adjacent beat handling (only for HRV Logger data with gaps)
                gap_handling_options = {
                    "include": "Include in artifacts",
                    "exclude": "Exclude gap-adjacent beats",
                    "boundary": "Treat as segment boundaries (recommended)",
                }
                gap_handling = st.selectbox(
                    "Gap-adjacent beats",
                    options=list(gap_handling_options.keys()),
                    format_func=lambda x: gap_handling_options[x],
                    index=2,  # Default to boundary
                    key=f"frag_gap_handling_{participant_id}",
                    help="Beats immediately after signal gaps may show large RR changes.",
                    disabled=is_vns_data,
                )

                # Section-scoped detection
                st.markdown("---")
                st.markdown("**Detection Scope**")
                scope_options = ["full", "section", "custom"]
                scope_labels = {
                    "full": "Full recording",
                    "section": "Selected section",
                    "custom": "Custom time range",
                }

                # Check if sections are available for THIS participant
                # Filter to only sections where this participant has the required start/end events
                all_sections = st.session_state.get("sections", {})
                participant_events = st.session_state.participant_events.get(participant_id, {})
                event_list = participant_events.get("events", [])

                # Get canonical event names for this participant
                participant_event_names = set()
                for event in event_list:
                    if isinstance(event, dict):
                        if event.get("canonical"):
                            participant_event_names.add(event["canonical"])
                        if event.get("raw_label"):
                            participant_event_names.add(event["raw_label"])
                    else:
                        if getattr(event, "canonical", None):
                            participant_event_names.add(event.canonical)
                        if getattr(event, "raw_label", None):
                            participant_event_names.add(event.raw_label)

                # Filter sections to only those where participant has both start and end events
                available_sections = []
                for section_name, section_def in all_sections.items():
                    start_events = section_def.get("start_events", [])
                    if not start_events and "start_event" in section_def:
                        start_events = [section_def["start_event"]]
                    end_events = section_def.get("end_events", [])
                    if not end_events and "end_event" in section_def:
                        end_events = [section_def["end_event"]]

                    # Check if participant has at least one start and one end event
                    has_start = any(e in participant_event_names for e in start_events)
                    has_end = any(e in participant_event_names for e in end_events)
                    if has_start and has_end:
                        available_sections.append(section_name)

                has_sections = len(available_sections) > 0

                # Default to "section" if sections are available, otherwise "full"
                default_scope_idx = 1 if has_sections else 0  # 1 = section, 0 = full

                artifact_scope = st.radio(
                    "Scope",
                    options=scope_options,
                    format_func=lambda x: scope_labels[x],
                    horizontal=True,
                    index=default_scope_idx,
                    key=f"frag_artifact_scope_{participant_id}",
                    help="Run detection on full recording, a specific section, or custom time range.",
                )

                # Section selector (only shown when scope is "section")
                selected_section = None
                custom_start_time = None
                custom_end_time = None

                if artifact_scope == "section":
                    if has_sections:
                        selected_section = st.selectbox(
                            "Section",
                            options=available_sections,
                            key=f"frag_artifact_section_{participant_id}",
                            help="Select a section for artifact detection.",
                        )
                    else:
                        st.warning("No sections defined. Go to Setup tab to define sections.")
                        artifact_scope = "full"  # Fall back to full recording
                elif artifact_scope == "custom":
                    col_start, col_end = st.columns(2)
                    with col_start:
                        custom_start_time = st.text_input(
                            "Start (HH:MM:SS)",
                            value="00:00:00",
                            key=f"frag_custom_start_{participant_id}",
                            help="Start time relative to recording start",
                        )
                    with col_end:
                        custom_end_time = st.text_input(
                            "End (HH:MM:SS)",
                            value="00:10:00",
                            key=f"frag_custom_end_{participant_id}",
                            help="End time relative to recording start",
                        )

                # Store scope settings in session state for use later
                st.session_state[f"artifact_scope_settings_{participant_id}"] = {
                    "scope": artifact_scope,
                    "selected_section": selected_section,
                    "custom_start": custom_start_time,
                    "custom_end": custom_end_time,
                }

                # Calculate estimated beats for the selected scope (for adaptive sizing)
                plot_data_for_len = st.session_state.get(f"plot_data_{participant_id}", {})
                full_rr_values = plot_data_for_len.get('rr_values', [])
                full_timestamps = plot_data_for_len.get('timestamps', [])
                n_beats_full = len(full_rr_values)

                # Estimate scoped beats
                n_beats_scoped = n_beats_full
                scope_label = "full recording"

                if artifact_scope == "section" and selected_section:
                    # Estimate section beats from events
                    sections = st.session_state.get("sections", {})
                    section_def = sections.get(selected_section, {})
                    start_event_names = section_def.get("start_events", [])
                    if not start_event_names and "start_event" in section_def:
                        start_event_names = [section_def["start_event"]]
                    end_event_names = section_def.get("end_events", [])
                    if not end_event_names and "end_event" in section_def:
                        end_event_names = [section_def["end_event"]]

                    stored_data = st.session_state.participant_events.get(participant_id, {})
                    all_events = stored_data.get('events', []) + stored_data.get('manual', [])
                    start_ts = None
                    end_ts = None
                    for event in all_events:
                        canonical = getattr(event, "canonical", None)
                        timestamp = getattr(event, "first_timestamp", None)
                        if not timestamp:
                            continue
                        if canonical in start_event_names and start_ts is None:
                            start_ts = timestamp
                        elif canonical in end_event_names and end_ts is None:
                            end_ts = timestamp

                    if start_ts and end_ts and full_timestamps:
                        # Count beats in section
                        section_beats = sum(1 for ts in full_timestamps if start_ts <= ts <= end_ts)
                        if section_beats > 0:
                            n_beats_scoped = section_beats
                            scope_label = f"section '{selected_section}'"

                elif artifact_scope == "custom" and custom_start_time and custom_end_time:
                    # Estimate custom range beats
                    try:
                        from datetime import timedelta
                        def parse_time_offset(time_str):
                            parts = time_str.split(":")
                            if len(parts) == 3:
                                h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
                                return timedelta(hours=h, minutes=m, seconds=s)
                            return timedelta(0)

                        start_offset = parse_time_offset(custom_start_time)
                        end_offset = parse_time_offset(custom_end_time)
                        if full_timestamps:
                            recording_start = full_timestamps[0]
                            start_dt = recording_start + start_offset
                            end_dt = recording_start + end_offset
                            custom_beats = sum(1 for ts in full_timestamps if start_dt <= ts <= end_dt)
                            if custom_beats > 0:
                                n_beats_scoped = custom_beats
                                scope_label = f"range {custom_start_time}-{custom_end_time}"
                    except Exception:
                        pass  # Keep full recording estimate

                # Segment sizing (only for segmented methods, now with scope-aware adaptive)
                if is_segmented_method:
                    st.markdown("---")
                    st.markdown("**Segment Sizing**")

                    segment_mode = st.radio(
                        "Mode",
                        options=["adaptive", "preset", "manual"],
                        format_func=lambda x: {"adaptive": "Adaptive (recommended)", "preset": "Preset", "manual": "Manual"}[x],
                        horizontal=True,
                        key=f"frag_segment_mode_{participant_id}",
                        help="Adaptive adjusts based on data length. Presets offer quick selection. Manual gives full control."
                    )

                    scoped_minutes = n_beats_scoped / 60  # Approximate at 60 BPM

                    if segment_mode == "adaptive":
                        # Adaptive segment size based on scoped data length
                        if scoped_minutes < 15:
                            segment_beats = 150
                            adaptive_label = "Fine"
                        elif scoped_minutes < 60:
                            segment_beats = 250
                            adaptive_label = "Standard"
                        else:
                            segment_beats = 350
                            adaptive_label = "Robust"

                        st.info(f"**{adaptive_label}**: {segment_beats} beats/segment (~{segment_beats/60:.1f} min) for {n_beats_scoped} beats ({scoped_minutes:.0f} min {scope_label})")

                    elif segment_mode == "preset":
                        preset_options = {
                            "fine": ("Fine (150 beats)", 150, "More sensitive, for noisy data or short sections"),
                            "standard": ("Standard (300 beats)", 300, "Balanced sensitivity, recommended for most data"),
                            "robust": ("Robust (500 beats)", 500, "Less sensitive, for clean data with stable baseline"),
                        }
                        preset_choice = st.selectbox(
                            "Preset",
                            options=list(preset_options.keys()),
                            format_func=lambda x: preset_options[x][0],
                            index=1,
                            key=f"frag_segment_preset_{participant_id}",
                            help="\n".join([f"**{v[0]}**: {v[2]}" for v in preset_options.values()])
                        )
                        segment_beats = preset_options[preset_choice][1]
                        st.caption(f"{preset_options[preset_choice][2]} | {n_beats_scoped} beats in {scope_label}")

                    else:  # manual
                        segment_beats = st.slider(
                            "Segment size (beats)",
                            min_value=100, max_value=600, value=300, step=25,
                            key=f"frag_segment_beats_{participant_id}",
                            help="Number of beats per segment. Lower = more sensitive, Higher = more robust."
                        )
                        st.caption(f"~{segment_beats/60:.1f} min per segment | {n_beats_scoped} beats in {scope_label}")

                # Option to show diagnostic plots (like NeuroKit2's signal_fixpeaks visualization)
                # Use session state to persist checkbox value across reruns
                diag_plots_key = f"show_diagnostic_plots_{participant_id}"
                show_diagnostic_plots = st.checkbox(
                    "Show diagnostic plots",
                    value=st.session_state.get(diag_plots_key, True),  # Default to True
                    key=f"show_artifact_diagnostic_{participant_id}",
                    help="Show NeuroKit2 diagnostic plots: artifact types, criteria thresholds, and subspace classification"
                )
                # Store the checkbox value in session state for detection code to use
                st.session_state[diag_plots_key] = show_diagnostic_plots

                # Check if there are saved artifact corrections
                loaded_info_key = f"artifacts_loaded_info_{participant_id}"
                has_saved_corrections = loaded_info_key in st.session_state
                confirm_key = f"confirm_new_detection_{participant_id}"

                # Run detection button with warning for saved corrections
                if has_saved_corrections and not st.session_state.get(confirm_key, False):
                    # Show warning and require confirmation
                    st.warning("⚠️ You have saved artifact corrections. Running new detection will **replace** them.")
                    col_confirm, col_cancel = st.columns(2)
                    with col_confirm:
                        if st.button("Replace & Detect", key=f"confirm_detection_{participant_id}", type="primary"):
                            st.session_state[confirm_key] = True
                            st.session_state[detect_new_key] = True
                            # Store diagnostic plot preference for detection code
                            st.session_state[f"show_diagnostic_plots_{participant_id}"] = show_diagnostic_plots
                            if f"artifacts_{participant_id}" in st.session_state:
                                st.session_state[f"artifacts_{participant_id}"]["force_redetect"] = True
                            # Clear saved info since we're replacing
                            if loaded_info_key in st.session_state:
                                del st.session_state[loaded_info_key]
                            st.rerun()
                    with col_cancel:
                        if st.button("Cancel", key=f"cancel_detection_{participant_id}"):
                            st.session_state[confirm_key] = False
                else:
                    if st.button("Run Detection", key=f"run_artifact_detection_{participant_id}", type="primary",
                                width="stretch"):
                        st.session_state[detect_new_key] = True
                        # Store diagnostic plot preference for detection code
                        st.session_state[f"show_diagnostic_plots_{participant_id}"] = show_diagnostic_plots
                        # Clear saved artifacts to force new detection
                        if f"artifacts_{participant_id}" in st.session_state:
                            st.session_state[f"artifacts_{participant_id}"]["force_redetect"] = True
                        st.rerun()
                    # Reset confirm state when not in confirmation mode
                    if confirm_key in st.session_state:
                        del st.session_state[confirm_key]

            # Show corrected only when artifacts enabled
            show_corrected = st.checkbox("Show corrected (NN)", value=plot_defaults.get("show_corrected", False),
                                         key=f"frag_show_corrected_{participant_id}",
                                         help="Preview corrected NN intervals (artifacts interpolated)")
        else:
            artifact_method = "threshold"
            artifact_threshold = 0.20
            segment_beats = 300  # Default
            show_corrected = False  # Not available without artifacts
            gap_handling = "include"  # Default
        show_variability = st.checkbox("Show variability segments", value=plot_defaults.get("show_variability", False),
                                       key=f"frag_show_var_{participant_id}",
                                       help="Detect variance changepoints")
    with col_opt4:
        show_gaps = st.checkbox("Show time gaps", value=plot_defaults.get("show_gaps", True),
                                key=f"frag_show_gaps_{participant_id}",
                                disabled=is_vns_data)
        gap_threshold = st.number_input(
            "Gap threshold (s)",
            min_value=1.0, max_value=60.0, value=float(plot_defaults.get("gap_threshold", 15.0)), step=1.0,
            key=f"frag_gap_thresh_{participant_id}",
            help="Threshold for detecting gaps in data",
            disabled=is_vns_data
        )
        # Only show Help button here if NOT in Signal Inspection mode
        # (Signal Inspection has its own dedicated Help section)
        current_mode = st.session_state.get(f"plot_mode_{participant_id}", "Add Events")
        if current_mode != "Signal Inspection":
            with st.popover("Help"):
                if is_vns_data:
                    st.markdown(VNS_DATA_HELP)
                else:
                    st.markdown(ARTIFACT_CORRECTION_HELP)

    # Persistent notification when artifacts are loaded from saved session
    loaded_info_key = f"artifacts_loaded_info_{participant_id}"

    if loaded_info_key in st.session_state and show_artifacts:
        loaded_info = st.session_state[loaded_info_key]

        # Build info message
        info_parts = []
        if loaded_info.get("n_algorithm", 0) > 0:
            method_name = loaded_info.get("algorithm_method", "unknown")
            threshold = loaded_info.get("algorithm_threshold")
            if threshold is not None:
                info_parts.append(f"**{loaded_info['n_algorithm']}** algorithm ({method_name}, {threshold:.0%})")
            else:
                info_parts.append(f"**{loaded_info['n_algorithm']}** algorithm ({method_name})")
        if loaded_info.get("n_manual", 0) > 0:
            info_parts.append(f"**{loaded_info['n_manual']}** manual")
        if loaded_info.get("n_excluded", 0) > 0:
            info_parts.append(f"**{loaded_info['n_excluded']}** excluded")

        if info_parts:
            saved_at = loaded_info.get("saved_at", "")
            if saved_at:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(saved_at)
                    saved_str = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    saved_str = saved_at[:16] if len(saved_at) > 16 else saved_at
            else:
                saved_str = "unknown"

            st.info(f"**Loaded artifact corrections** (saved {saved_str}): " + " | ".join(info_parts))

    # Show downsampling info
    if plot_data['n_displayed'] < plot_data['n_original']:
        st.caption(f"Showing {plot_data['n_displayed']:,} of {plot_data['n_original']:,} points")

    # Build figure
    fig = go.Figure()

    # Get custom plot colors from settings
    plot_colors = get_plot_colors()
    line_color = plot_colors['line']
    artifact_color = plot_colors['artifact']

    # Check if we have flags (VNS data with flagged intervals)
    flags = plot_data.get('flags')
    if flags:
        # VNS data: Split into valid and flagged intervals
        timestamps = plot_data['timestamps']
        rr_values = plot_data['rr_values']

        good_ts, good_rr = [], []
        flagged_ts, flagged_rr = [], []

        for ts, rr, f in zip(timestamps, rr_values, flags):
            if f:
                flagged_ts.append(ts)
                flagged_rr.append(rr)
            else:
                good_ts.append(ts)
                good_rr.append(rr)

        # Count flagged intervals for info display
        n_flagged = len(flagged_ts)
        n_total = len(timestamps)
        if n_flagged > 0:
            flagged_time_ms = sum(flagged_rr)
            st.warning(f"**{n_flagged} intervals flagged** ({n_flagged/n_total*100:.1f}%) - "
                      f"shown in artifact color, excluded from HRV analysis. "
                      f"Total flagged time: {flagged_time_ms/1000:.1f}s")

        # Valid intervals (connected with lines)
        if good_ts:
            fig.add_trace(ScatterType(
                x=good_ts,
                y=good_rr,
                mode='markers+lines',
                name='RR Intervals (valid)',
                marker=dict(size=3, color=line_color),
                line=dict(width=1, color=line_color),
                hovertemplate='Time: %{x}<br>RR: %{y} ms<extra></extra>'
            ))

        # Flagged intervals (markers only, no lines to show discontinuity)
        if flagged_ts:
            fig.add_trace(ScatterType(
                x=flagged_ts,
                y=flagged_rr,
                mode='markers',
                name='RR Intervals (flagged)',
                marker=dict(size=5, color=artifact_color, symbol='x'),
                hovertemplate='Time: %{x}<br>RR: %{y} ms (FLAGGED)<extra></extra>'
            ))
    else:
        # HRV Logger: Already cleaned, show all with custom color
        fig.add_trace(ScatterType(
            x=plot_data['timestamps'],
            y=plot_data['rr_values'],
            mode='markers+lines',
            name='RR Intervals',
            marker=dict(size=3, color=line_color),
            line=dict(width=1, color=line_color),
            hovertemplate='Time: %{x}<br>RR: %{y} ms<extra></extra>'
        ))

    y_min, y_max = plot_data['y_min'], plot_data['y_max']
    y_range = plot_data['y_range']

    # Get theme colors for the plot
    theme = get_current_theme_colors()

    # Check if Signal Inspection section filter is active
    inspection_range = st.session_state.get(f"inspection_section_range_{participant_id}")

    # Configure x-axis based on mode
    if use_sequential_timestamps:
        # Signal Inspection mode: x-axis shows sequential beat time (each beat unique position)
        xaxis_config = dict(
            title=dict(text="Sequential Beat Time (gaps removed)", font=dict(color=theme['text'])),
            tickformat='%H:%M:%S',
            gridcolor=theme['grid'],
            linecolor=theme['line'],
            tickfont=dict(color=theme['text']),
            uirevision=True,
        )
    else:
        # Normal mode: x-axis shows real clock time (aligned with events)
        xaxis_config = dict(
            title=dict(text="Time", font=dict(color=theme['text'])),
            tickformat='%H:%M:%S',
            gridcolor=theme['grid'],
            linecolor=theme['line'],
            tickfont=dict(color=theme['text']),
            uirevision=True,  # Preserve x-axis zoom
        )
    # If a section is selected in Signal Inspection mode, zoom to that range
    if inspection_range and len(inspection_range) == 2:
        start_time, end_time = inspection_range
        xaxis_config['range'] = [start_time, end_time]

    # Build Y-axis config (check for inspection zoom)
    zoom_key = f"inspection_zoom_{participant_id}"
    inspection_zoom = st.session_state.get(zoom_key, None)
    yaxis_config = dict(
        title=dict(text="RR Interval (ms)", font=dict(color=theme['text'])),
        gridcolor=theme['grid'],
        linecolor=theme['line'],
        tickfont=dict(color=theme['text']),
        uirevision=True,  # Preserve y-axis zoom
    )

    # Apply inspection zoom if set
    if inspection_zoom:
        yaxis_config['range'] = [inspection_zoom['y_min'], inspection_zoom['y_max']]
        # Also set X-axis range if time window specified
        ts_for_zoom = plot_data.get('timestamps', [])
        if inspection_zoom.get('x_window_seconds') and ts_for_zoom:
            # Set initial X-axis window centered on data
            # User can then pan with arrow keys (client-side JS) or drag
            mid_idx = len(ts_for_zoom) // 2
            mid_time = get_pandas().to_datetime(ts_for_zoom[mid_idx])
            half_window = get_pandas().Timedelta(seconds=inspection_zoom['x_window_seconds'] / 2)
            xaxis_config['range'] = [mid_time - half_window, mid_time + half_window]

    # Set dragmode based on interaction mode
    # Signal Inspection: pan mode for instant drag-to-pan (client-side, no server roundtrip)
    # Other modes: zoom mode for selecting regions
    dragmode = 'pan' if current_mode == "Signal Inspection" else 'zoom'

    fig.update_layout(
        title=f"Tachogram - {participant_id}",
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top', font=dict(color=theme['text'])),
        uirevision=True,  # Preserve zoom/pan state across checkbox changes
        paper_bgcolor=theme['bg'],
        plot_bgcolor=theme['bg'],
        font=dict(color=theme['text']),
        dragmode=dragmode,  # Enable instant pan in Signal Inspection mode
    )

    # Add event markers (conditional on show_events)
    if show_events:
        events_list = stored_data.get('events', [])
        manual_list = stored_data.get('manual', [])
        if not isinstance(events_list, list):
            events_list = []
        if not isinstance(manual_list, list):
            manual_list = []
        current_events = events_list + manual_list

        distinct_colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b',
                           '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        event_by_canonical = {}

        for evt_status in current_events:
            if hasattr(evt_status, 'canonical'):
                canonical = evt_status.canonical
                timestamp = evt_status.first_timestamp
            else:
                canonical = st.session_state.normalizer.normalize(evt_status.label) if hasattr(evt_status, 'label') else None
                timestamp = evt_status.timestamp if hasattr(evt_status, 'timestamp') else None

            if canonical and canonical != "unmatched" and timestamp:
                if canonical not in event_by_canonical:
                    event_by_canonical[canonical] = []
                event_by_canonical[canonical].append(timestamp)

        # Helper function to map real timestamp to sequential position
        def map_to_sequential(real_ts, mapping):
            """Find the closest beat in real time and return its sequential position."""
            if not mapping:
                return real_ts
            # Binary search for closest match
            best_idx = 0
            best_diff = abs((mapping[0][0] - real_ts).total_seconds())
            for i, (real, seq) in enumerate(mapping):
                diff = abs((real - real_ts).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best_idx = i
            return mapping[best_idx][1]

        for idx, (event_name, event_times) in enumerate(event_by_canonical.items()):
            color = distinct_colors[idx % len(distinct_colors)]
            for event_time in event_times:
                # Map to sequential position if in Signal Inspection mode
                display_time = map_to_sequential(event_time, real_to_sequential_map) if real_to_sequential_map else event_time
                fig.add_shape(
                    type="line", x0=display_time, x1=display_time,
                    y0=y_min - 0.05 * y_range, y1=y_max + 0.05 * y_range,
                    line=dict(color=color, width=2, dash='dash'), opacity=0.7
                )
                fig.add_annotation(
                    x=display_time, y=y_max + 0.08 * y_range,
                    text=event_name, showarrow=False, textangle=-90,
                    font=dict(color=color, size=10)
                )

    # Gap detection (CACHED) - ALWAYS detect gaps from original timestamps
    # This provides consistent gap info for both Add Events and Signal Inspection modes
    timestamps_list = plot_data['timestamps']
    rr_list = plot_data['rr_values']
    if is_vns_data:
        # VNS data doesn't have meaningful timestamp gaps
        gap_result = {"gaps": [], "total_gaps": 0, "total_gap_duration_s": 0.0, "gap_ratio": 0.0, "vns_note": True}
        gap_adjacent_indices = set()
    else:
        # Detect gaps from ORIGINAL timestamps (not sequential), using user's threshold
        gap_result = cached_gap_detection(tuple(original_timestamps), tuple(rr_list), gap_threshold)
        # Compute gap-adjacent indices (beats immediately after gaps) for artifact handling
        gap_adjacent_indices = set()
        for gap in gap_result.get("gaps", []):
            gap_end_idx = gap.get("end_idx")
            if gap_end_idx is not None and gap_end_idx < len(rr_list):
                gap_adjacent_indices.add(gap_end_idx)
    st.session_state[f"gaps_{participant_id}"] = gap_result
    # Store gap-adjacent indices in plot_data for artifact handling
    plot_data['gap_adjacent_indices'] = gap_adjacent_indices

    # Get manual artifacts (always available, even if show_artifacts is False)
    manual_artifact_key = f"manual_artifacts_{participant_id}"
    manual_artifacts = st.session_state.get(manual_artifact_key, [])

    # Artifact detection (threshold or Kubios method)
    if show_artifacts:
        # Check if we should use saved artifacts or run new detection
        detect_new_key = f"detect_new_artifacts_{participant_id}"
        run_new_detection = st.session_state.get(detect_new_key, False)
        saved_artifact_data = st.session_state.get(f"artifacts_{participant_id}", {})
        has_saved_artifacts = bool(saved_artifact_data.get("artifact_indices"))
        force_redetect = saved_artifact_data.get("force_redetect", False)

        # Fast path for marker-only updates (user clicked to mark/unmark)
        # Skip heavy detection processing, just reuse existing artifact data
        marker_only_key = f"artifact_marker_only_{participant_id}"
        is_marker_only_update = st.session_state.pop(marker_only_key, False)

        if is_marker_only_update and has_saved_artifacts:
            # Fast path: reuse existing artifact result, only manual markers changed
            artifact_result = saved_artifact_data
        elif run_new_detection or force_redetect:
            # User explicitly requested new detection - run it
            # Get scope settings
            scope_settings = st.session_state.get(f"artifact_scope_settings_{participant_id}", {"scope": "full"})
            detection_scope = scope_settings.get("scope", "full")

            # Prepare RR data based on scope
            rr_for_detection = rr_list
            timestamps_for_detection = timestamps_list
            scope_offset = 0  # Index offset for mapping back to full recording
            scope_info = {"type": "full"}

            if detection_scope == "section" and scope_settings.get("selected_section"):
                # Get section time range from saved events
                section_name = scope_settings["selected_section"]
                sections = st.session_state.get("sections", {})
                section_def = sections.get(section_name)

                if section_def:
                    # Get start/end event names from section definition
                    start_event_names = section_def.get("start_events", [])
                    if not start_event_names and "start_event" in section_def:
                        start_event_names = [section_def["start_event"]]
                    end_event_names = section_def.get("end_events", [])
                    if not end_event_names and "end_event" in section_def:
                        end_event_names = [section_def["end_event"]]

                    # Find timestamps from participant events (same location as section validation)
                    stored_data = st.session_state.participant_events.get(participant_id, {})
                    all_events = stored_data.get('events', []) + stored_data.get('manual', [])
                    start_ts = None
                    end_ts = None

                    for event in all_events:
                        canonical = getattr(event, "canonical", None)
                        timestamp = getattr(event, "first_timestamp", None)
                        if not timestamp:
                            continue
                        if canonical in start_event_names and start_ts is None:
                            start_ts = timestamp
                        elif canonical in end_event_names and end_ts is None:
                            end_ts = timestamp

                    if start_ts and end_ts:
                        # Filter timestamps and RR values to section range
                        filtered_data = []
                        for idx, (ts, rr) in enumerate(zip(timestamps_list, rr_list)):
                            if start_ts <= ts <= end_ts:
                                if not filtered_data:
                                    scope_offset = idx
                                filtered_data.append((ts, rr))

                        if filtered_data:
                            timestamps_for_detection = [d[0] for d in filtered_data]
                            rr_for_detection = [d[1] for d in filtered_data]
                            scope_info = {"type": "section", "name": section_name, "offset": scope_offset}
                        else:
                            st.warning(f"No data found in section '{section_name}'. Using full recording.")
                    else:
                        st.warning(f"Could not find start/end events for section '{section_name}'. Using full recording.")

            elif detection_scope == "custom":
                # Parse custom time range
                custom_start = scope_settings.get("custom_start", "00:00:00")
                custom_end = scope_settings.get("custom_end", "00:10:00")

                try:
                    from datetime import timedelta

                    def parse_time_offset(time_str):
                        """Parse HH:MM:SS to timedelta."""
                        parts = time_str.split(":")
                        if len(parts) == 3:
                            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
                            return timedelta(hours=h, minutes=m, seconds=s)
                        return timedelta(0)

                    start_offset = parse_time_offset(custom_start)
                    end_offset = parse_time_offset(custom_end)

                    if timestamps_list:
                        recording_start = timestamps_list[0]
                        start_dt = recording_start + start_offset
                        end_dt = recording_start + end_offset

                        # Filter timestamps and RR values
                        filtered_data = []
                        for idx, (ts, rr) in enumerate(zip(timestamps_list, rr_list)):
                            if start_dt <= ts <= end_dt:
                                if not filtered_data:
                                    scope_offset = idx  # First match is the offset
                                filtered_data.append((ts, rr))

                        if filtered_data:
                            timestamps_for_detection = [d[0] for d in filtered_data]
                            rr_for_detection = [d[1] for d in filtered_data]
                            scope_info = {"type": "custom", "start": custom_start, "end": custom_end, "offset": scope_offset}
                        else:
                            st.warning(f"No data in time range {custom_start} - {custom_end}. Using full recording.")
                except Exception as e:
                    st.warning(f"Could not parse time range: {e}. Using full recording.")

            # Run artifact detection
            # Get gap handling setting from session state
            gap_handling_for_detection = st.session_state.get(f"frag_gap_handling_{participant_id}", "include")

            # Get gap_adjacent_indices for the detection scope
            gap_adjacent_for_scope = set()
            if gap_handling_for_detection == "boundary":
                # Get gap adjacent indices from plot_data
                all_gap_adjacent = plot_data.get('gap_adjacent_indices', set())
                if all_gap_adjacent:
                    # Filter to only indices within the detection scope
                    scope_offset = scope_info.get("offset", 0)
                    scope_length = len(rr_for_detection)
                    # Map global indices to scope-local indices
                    for global_idx in all_gap_adjacent:
                        local_idx = global_idx - scope_offset
                        if 0 <= local_idx < scope_length:
                            gap_adjacent_for_scope.add(local_idx)

            # Use segmented detection at gaps if boundary mode is selected AND there are gaps in scope
            if gap_handling_for_detection == "boundary" and gap_adjacent_for_scope:
                artifact_result = run_segmented_artifact_detection_at_gaps(
                    rr_for_detection, timestamps_for_detection,
                    gap_adjacent_for_scope,
                    method=artifact_method, threshold_pct=artifact_threshold,
                    segment_beats=segment_beats
                )
            else:
                artifact_result = cached_artifact_detection(
                    tuple(rr_for_detection), tuple(timestamps_for_detection),
                    method=artifact_method, threshold_pct=artifact_threshold,
                    segment_beats=segment_beats
                )
                # If boundary mode but no gaps in scope, mark as handled to skip legacy fallback
                if gap_handling_for_detection == "boundary":
                    artifact_result = dict(artifact_result)
                    artifact_result["independent_segment_analysis"] = True
                    artifact_result["segment_boundaries"] = []  # No gaps in this scope

            # Always make a copy before modifying (cached result may be immutable)
            artifact_result = dict(artifact_result)

            # Map artifact indices back to full recording if scope was not full
            if scope_info["type"] != "full" and scope_info.get("offset", 0) > 0:
                offset = scope_info["offset"]
                artifact_result["artifact_indices"] = [i + offset for i in artifact_result.get("artifact_indices", [])]
                # Also map indices_by_type to global indices
                if "indices_by_type" in artifact_result:
                    artifact_result["indices_by_type"] = {
                        k: [i + offset for i in v]
                        for k, v in artifact_result["indices_by_type"].items()
                    }
                # Also map segment_boundaries to global indices
                if "segment_boundaries" in artifact_result:
                    artifact_result["segment_boundaries"] = [i + offset for i in artifact_result["segment_boundaries"]]
                # Rebuild timestamps and RR from full recording indices
                artifact_result["artifact_timestamps"] = [timestamps_list[i] for i in artifact_result["artifact_indices"] if 0 <= i < len(timestamps_list)]
                artifact_result["artifact_rr"] = [rr_list[i] for i in artifact_result["artifact_indices"] if 0 <= i < len(rr_list)]
                # Note: corrected_rr is for the scope only, not full recording

            # Store scope info and detection parameters in result
            artifact_result["scope"] = scope_info
            artifact_result["segment_beats"] = segment_beats  # Store for segmented methods

            # Derive section_key from scope for section-scoped storage
            if scope_info["type"] == "section":
                section_key = scope_info.get("name", "_full")
            elif scope_info["type"] == "custom":
                # Use a unique key for custom time ranges
                section_key = f"custom_{scope_info.get('start', '0')}-{scope_info.get('end', '0')}".replace(":", "")
            else:
                section_key = "_full"
            artifact_result["section_key"] = section_key

            # Generate diagnostic plots if requested
            if st.session_state.get(f"show_diagnostic_plots_{participant_id}", False):
                # Use Lipponen methods for diagnostic plots (threshold method doesn't have NK2 diagnostics)
                if artifact_method in ("kubios", "lipponen2019", "kubios_segmented", "lipponen2019_segmented"):
                    with st.spinner("Generating diagnostic plots..."):
                        try:
                            # Check if we have gap-separated segments
                            gap_segments = artifact_result.get("gap_segments", [])
                            if gap_segments and len(gap_segments) > 1:
                                # Generate one diagnostic plot per gap-segment
                                diag_plots = []
                                for seg_idx, (seg_start, seg_end) in enumerate(gap_segments):
                                    seg_rr = rr_for_detection[seg_start:seg_end]
                                    if len(seg_rr) >= 10:
                                        seg_diag = generate_artifact_diagnostic_plots(seg_rr)
                                        if seg_diag:
                                            diag_plots.append({
                                                "segment": seg_idx + 1,
                                                "start": seg_start,
                                                "end": seg_end,
                                                "image": seg_diag,
                                            })
                                if diag_plots:
                                    st.session_state[f"artifact_diagnostic_fig_{participant_id}"] = diag_plots
                                    st.success(f"Diagnostic plots generated for {len(diag_plots)} gap-segments")
                                else:
                                    st.warning("No diagnostic plots could be generated (segments too small)")
                            else:
                                # Single diagnostic plot for entire scope
                                diag_result = generate_artifact_diagnostic_plots(rr_for_detection)
                                if diag_result is not None:
                                    st.session_state[f"artifact_diagnostic_fig_{participant_id}"] = diag_result
                                    st.success(f"Diagnostic plots generated: {len(diag_result)} bytes")
                                else:
                                    st.warning("Diagnostic plots could not be generated (function returned None)")
                        except Exception as e:
                            st.error(f"Error generating diagnostic plots: {e}")
                            import traceback
                            st.code(traceback.format_exc())
                else:
                    st.info(f"Diagnostic plots not available for method '{artifact_method}' (only Lipponen/Kubios methods)")

            # Clear the force_redetect flag
            if force_redetect and f"artifacts_{participant_id}" in st.session_state:
                st.session_state[f"artifacts_{participant_id}"].pop("force_redetect", None)
            # Reset detection flag
            st.session_state[detect_new_key] = False
        elif has_saved_artifacts:
            # Use saved/loaded artifacts (no new detection)
            artifact_result = {
                "artifact_indices": saved_artifact_data.get("artifact_indices", []),
                "artifact_timestamps": [timestamps_list[i] for i in saved_artifact_data.get("artifact_indices", []) if 0 <= i < len(timestamps_list)],
                "artifact_rr": [rr_list[i] for i in saved_artifact_data.get("artifact_indices", []) if 0 <= i < len(rr_list)],
                "total_artifacts": len(saved_artifact_data.get("artifact_indices", [])),
                "artifact_ratio": len(saved_artifact_data.get("artifact_indices", [])) / len(rr_list) if rr_list else 0.0,
                "method": saved_artifact_data.get("method", "loaded"),
                "by_type": saved_artifact_data.get("by_type", {}),
                "indices_by_type": saved_artifact_data.get("indices_by_type", {}),
                "scope": saved_artifact_data.get("scope"),
                "section_key": saved_artifact_data.get("section_key", "_full"),
                "segment_beats": saved_artifact_data.get("segment_beats"),
                "corrected_rr": saved_artifact_data.get("corrected_rr"),
                "restored_from_save": True,
            }
        else:
            # No saved artifacts and no detection requested - empty result
            artifact_result = {
                "artifact_indices": [],
                "artifact_timestamps": [],
                "artifact_rr": [],
                "total_artifacts": 0,
                "artifact_ratio": 0.0,
                "method": "none",
                "by_type": {},
                "no_detection_yet": True,
            }

        # Get gap-adjacent indices from plot_data (beats immediately after gaps)
        gap_adjacent_indices = plot_data.get('gap_adjacent_indices', set())

        # Filter out gap-adjacent beats based on gap_handling setting
        # Skip if we already did independent segment analysis (boundary mode with new detection)
        already_segmented = artifact_result.get("independent_segment_analysis", False)
        original_artifact_indices = artifact_result.get("artifact_indices", [])
        gap_adjacent_excluded = []  # Track how many were excluded
        segment_boundaries = artifact_result.get("segment_boundaries", [])  # May already be set

        if gap_handling == "exclude" and gap_adjacent_indices and not already_segmented:
            # Remove gap-adjacent beats from artifact detection
            filtered_indices = [i for i in original_artifact_indices if i not in gap_adjacent_indices]
            gap_adjacent_excluded = [i for i in original_artifact_indices if i in gap_adjacent_indices]

            # Update artifact_result with filtered data
            artifact_result = dict(artifact_result)  # Make a copy
            artifact_result["artifact_indices"] = filtered_indices
            artifact_result["artifact_timestamps"] = [timestamps_list[i] for i in filtered_indices if 0 <= i < len(timestamps_list)]
            artifact_result["artifact_rr"] = [rr_list[i] for i in filtered_indices if 0 <= i < len(rr_list)]
            artifact_result["total_artifacts"] = len(filtered_indices)
            artifact_result["artifact_ratio"] = len(filtered_indices) / len(rr_list) if rr_list else 0.0
            artifact_result["gap_adjacent_excluded"] = len(gap_adjacent_excluded)

        elif gap_handling == "boundary" and gap_adjacent_indices and not already_segmented:
            # Legacy fallback: Remove gap-adjacent beats AND mark as segment boundaries
            # (only used for saved/loaded artifacts that weren't analyzed with independent segments)
            filtered_indices = [i for i in original_artifact_indices if i not in gap_adjacent_indices]
            gap_adjacent_excluded = [i for i in original_artifact_indices if i in gap_adjacent_indices]
            segment_boundaries = sorted(gap_adjacent_indices)

            # Update artifact_result with filtered data
            artifact_result = dict(artifact_result)  # Make a copy
            artifact_result["artifact_indices"] = filtered_indices
            artifact_result["artifact_timestamps"] = [timestamps_list[i] for i in filtered_indices if 0 <= i < len(timestamps_list)]
            artifact_result["artifact_rr"] = [rr_list[i] for i in filtered_indices if 0 <= i < len(rr_list)]
            artifact_result["total_artifacts"] = len(filtered_indices)
            artifact_result["artifact_ratio"] = len(filtered_indices) / len(rr_list) if rr_list else 0.0
            artifact_result["gap_adjacent_excluded"] = len(gap_adjacent_excluded)
            artifact_result["segment_boundaries"] = segment_boundaries

        # Apply user exclusions (manually unmarked detected artifacts)
        artifact_exclusions_key = f"artifact_exclusions_{participant_id}"
        user_exclusions = st.session_state.get(artifact_exclusions_key, set())
        if user_exclusions:
            # Store original indices before filtering
            all_detected_indices = artifact_result.get("artifact_indices", [])
            # Filter out user-excluded artifacts
            active_indices = [i for i in all_detected_indices if i not in user_exclusions]
            excluded_indices = [i for i in all_detected_indices if i in user_exclusions]

            # Update artifact_result
            artifact_result = dict(artifact_result)  # Make a copy
            artifact_result["artifact_indices"] = active_indices
            artifact_result["artifact_timestamps"] = [timestamps_list[i] for i in active_indices if 0 <= i < len(timestamps_list)]
            artifact_result["artifact_rr"] = [rr_list[i] for i in active_indices if 0 <= i < len(rr_list)]
            artifact_result["total_artifacts"] = len(active_indices)
            artifact_result["artifact_ratio"] = len(active_indices) / len(rr_list) if rr_list else 0.0
            # Store excluded info for visualization
            artifact_result["user_excluded_indices"] = excluded_indices
            artifact_result["user_excluded_timestamps"] = [timestamps_list[i] for i in excluded_indices if 0 <= i < len(timestamps_list)]
            artifact_result["user_excluded_rr"] = [rr_list[i] for i in excluded_indices if 0 <= i < len(rr_list)]
            artifact_result["user_excluded_count"] = len(excluded_indices)

        st.session_state[f"artifacts_{participant_id}"] = artifact_result

        if artifact_result and artifact_result["total_artifacts"] > 0:
            # Show artifact summary based on method
            by_type = artifact_result["by_type"]
            method_used = artifact_result.get("method", "threshold")

            # Build summary message with gap handling and user exclusion info
            gap_excluded_count = artifact_result.get("gap_adjacent_excluded", 0)
            user_excluded_count = artifact_result.get("user_excluded_count", 0)
            gap_suffix = ""
            if gap_excluded_count > 0:
                gap_suffix = f" | {gap_excluded_count} gap-adjacent excluded"
            if user_excluded_count > 0:
                gap_suffix += f" | {user_excluded_count} manually unmarked"
            boundary_count = len(artifact_result.get("segment_boundaries", []))
            if boundary_count > 0:
                gap_suffix += f" | {boundary_count} segment boundaries"

            # Show scope info if not full recording
            scope_info = artifact_result.get("scope") or {"type": "full"}
            scope_prefix = ""
            if scope_info.get("type") == "section":
                scope_prefix = f"[Section: {scope_info.get('name', 'unknown')}] "
            elif scope_info.get("type") == "custom":
                scope_prefix = f"[Range: {scope_info.get('start', '?')}-{scope_info.get('end', '?')}] "

            # Format method display name
            method_display_names = {
                "threshold": f"Threshold ({artifact_threshold*100:.0f}%)",
                "lipponen2019": "Lipponen 2019",
                "lipponen2019_segmented": "Lipponen 2019 (segmented)",
                "kubios": "Kubios",
                "kubios_segmented": "Kubios (segmented)",
            }
            method_display = method_display_names.get(method_used, method_used)

            # Show artifact summary with Clear button
            col_summary, col_clear_btn = st.columns([5, 1])
            with col_summary:
                if method_used == "threshold":
                    st.info(f"{scope_prefix}**{artifact_result['total_artifacts']} artifacts detected** "
                           f"({artifact_result['artifact_ratio']*100:.1f}%) - "
                           f"Method: {method_display}{gap_suffix}")
                else:
                    st.info(f"{scope_prefix}**{artifact_result['total_artifacts']} artifacts detected** "
                           f"({artifact_result['artifact_ratio']*100:.1f}%) - "
                           f"Ectopic: {by_type.get('ectopic', 0)}, "
                           f"Missed: {by_type.get('missed', 0)}, "
                           f"Extra: {by_type.get('extra', 0)}, "
                       f"Long/Short: {by_type.get('longshort', 0)}{gap_suffix}")
            with col_clear_btn:
                if st.button("Clear", key=f"clear_artifacts_result_{participant_id}",
                            help="Clear algorithm-detected artifacts"):
                    if f"artifacts_{participant_id}" in st.session_state:
                        del st.session_state[f"artifacts_{participant_id}"]
                    loaded_info_key = f"artifacts_loaded_info_{participant_id}"
                    if loaded_info_key in st.session_state:
                        del st.session_state[loaded_info_key]
                    st.rerun()  # Full rerun to update Signal Inspection section

            # Display per-gap-segment stats (from independent segment analysis)
            gap_segment_stats = artifact_result.get("gap_segment_stats", [])
            if gap_segment_stats and len(gap_segment_stats) > 1:
                with st.expander(f"Gap-Separated Segments ({len(gap_segment_stats)} segments)", expanded=True):
                    st.caption("Each segment between gaps was analyzed independently:")
                    # Create display DataFrame
                    gap_df_data = []
                    for gs in gap_segment_stats:
                        by_type = gs.get("by_type", {})
                        gap_df_data.append({
                            "Segment": gs["gap_segment"],
                            "Beats": f"{gs['start_beat']}-{gs['end_beat']}",
                            "N Beats": gs["n_beats"],
                            "Artifacts": gs["n_artifacts"],
                            "Rate %": gs["artifact_pct"],
                            "Ectopic": by_type.get("ectopic", 0),
                            "Missed": by_type.get("missed", 0),
                            "Extra": by_type.get("extra", 0),
                            "Long/Short": by_type.get("longshort", 0),
                        })
                    gap_df = get_pandas().DataFrame(gap_df_data)

                    def highlight_high_artifacts_gap(row):
                        if row["Rate %"] > 10:
                            return ["background-color: #ffcccc"] * len(row)
                        elif row["Rate %"] > 5:
                            return ["background-color: #fff3cd"] * len(row)
                        return [""] * len(row)

                    st.dataframe(
                        gap_df.style.apply(highlight_high_artifacts_gap, axis=1).format({"Rate %": "{:.1f}"}),
                        use_container_width=True,
                        hide_index=True,
                    )

            # Display per-segment artifact percentages for segmented methods (Lipponen internal segments)
            segment_stats = artifact_result.get("segment_stats", [])
            if segment_stats:
                with st.expander(f"Lipponen Method Segment Details ({len(segment_stats)} segments)"):
                    # Filter to only the expected columns (remove extra fields like global_offset)
                    expected_keys = ["segment", "start_beat", "end_beat", "n_beats", "n_artifacts", "artifact_pct"]
                    filtered_stats = [{k: s.get(k) for k in expected_keys if k in s} for s in segment_stats]
                    df = get_pandas().DataFrame(filtered_stats)
                    df.columns = ["Segment", "Start Beat", "End Beat", "N Beats", "N Artifacts", "Artifact %"]

                    # Highlight segments with high artifact rates
                    def highlight_high_artifacts(row):
                        if row["Artifact %"] > 10:
                            return ["background-color: #ffcccc"] * len(row)  # Red for >10%
                        elif row["Artifact %"] > 5:
                            return ["background-color: #fff3cd"] * len(row)  # Yellow for >5%
                        return [""] * len(row)

                    st.dataframe(
                        df.style.apply(highlight_high_artifacts, axis=1).format({"Artifact %": "{:.1f}"}),
                        width="stretch",
                        hide_index=True,
                    )

                    # Summary statistics
                    avg_pct = df["Artifact %"].mean()
                    max_pct = df["Artifact %"].max()
                    high_segments = len(df[df["Artifact %"] > 10])
                    st.caption(f"Avg: {avg_pct:.1f}% | Max: {max_pct:.1f}% | Segments >10%: {high_segments}")

            # Display NeuroKit2 diagnostic plots if generated
            diag_fig_key = f"artifact_diagnostic_fig_{participant_id}"
            if diag_fig_key in st.session_state and st.session_state[diag_fig_key] is not None:
                diag_data = st.session_state[diag_fig_key]

                # Check if it's a list of segment plots or a single image
                if isinstance(diag_data, list):
                    # Multiple gap-segment plots
                    with st.expander(f"NeuroKit2 Diagnostic Plots ({len(diag_data)} gap-segments)", expanded=True):
                        st.caption(
                            "Each gap-separated segment was analyzed independently. "
                            "**Left column:** Artifact types (top), consecutive-difference criterion (middle), "
                            "difference-from-median criterion (bottom). "
                            "**Right column:** Subspace classification showing ectopic (red) and long/short (yellow/green) regions."
                        )
                        for plot_info in diag_data:
                            st.markdown(f"**Gap Segment {plot_info['segment']}** (beats {plot_info['start']}-{plot_info['end']})")
                            st.image(plot_info["image"], use_container_width=True)
                            st.markdown("---")
                        # Add button to close/clear the diagnostic plots
                        if st.button("Close diagnostic plots", key=f"close_diag_plots_{participant_id}"):
                            del st.session_state[diag_fig_key]
                            st.rerun()
                else:
                    # Single diagnostic plot (bytes)
                    with st.expander("NeuroKit2 Diagnostic Plots", expanded=True):
                        st.image(diag_data, use_container_width=True)
                        st.caption(
                            "**Left column:** Artifact types (top), consecutive-difference criterion (middle), "
                            "difference-from-median criterion (bottom). "
                            "**Right column:** Subspace classification showing ectopic (red) and long/short (yellow/green) regions."
                        )
                        # Add button to close/clear the diagnostic plots
                        if st.button("Close diagnostic plots", key=f"close_diag_plots_{participant_id}"):
                            del st.session_state[diag_fig_key]
                            st.rerun()

            # Add artifact markers to plot - show by type if available (like NeuroKit2)
            indices_by_type = artifact_result.get("indices_by_type", {})

            if indices_by_type and any(indices_by_type.values()):
                # Show artifacts by type with different colors (matching NeuroKit2 visualization)
                artifact_type_styles = {
                    "ectopic": {"color": "#FFD700", "symbol": "x", "name": "Ectopic"},  # Gold
                    "missed": {"color": "#FF4444", "symbol": "x", "name": "Missed (false neg)"},  # Red
                    "extra": {"color": "#FFEB3B", "symbol": "x", "name": "Extra (false pos)"},  # Light yellow
                    "longshort": {"color": "#FF00FF", "symbol": "x", "name": "Long/Short"},  # Magenta
                }

                for artifact_type, indices in indices_by_type.items():
                    if not indices:
                        continue
                    style = artifact_type_styles.get(artifact_type, {"color": "orange", "symbol": "x", "name": artifact_type})
                    # Get timestamps and RR values for these indices
                    type_ts = [timestamps_list[i] for i in indices if 0 <= i < len(timestamps_list)]
                    type_rr = [rr_list[i] for i in indices if 0 <= i < len(rr_list)]
                    if type_ts:
                        fig.add_trace(ScatterType(
                            x=type_ts,
                            y=type_rr,
                            mode='markers',
                            name=f'{style["name"]} ({len(type_ts)})',
                            marker=dict(size=10, color=style["color"], symbol=style["symbol"], line=dict(width=2)),
                            hovertemplate=f'Time: %{{x}}<br>RR: %{{y}} ms ({artifact_type.upper()})<extra></extra>'
                        ))
            elif artifact_result["artifact_timestamps"]:
                # Fallback: show all artifacts with single color (threshold method or no type info)
                fig.add_trace(ScatterType(
                    x=artifact_result["artifact_timestamps"],
                    y=artifact_result["artifact_rr"],
                    mode='markers',
                    name=f'Artifacts ({method_used})',
                    marker=dict(size=8, color='orange', symbol='x', line=dict(width=2)),
                    hovertemplate='Time: %{x}<br>RR: %{y} ms (ARTIFACT)<extra></extra>'
                ))

            # Add user-excluded artifact markers (dimmed gray circles with X)
            excluded_ts = artifact_result.get("user_excluded_timestamps", [])
            excluded_rr = artifact_result.get("user_excluded_rr", [])
            if excluded_ts:
                fig.add_trace(ScatterType(
                    x=excluded_ts,
                    y=excluded_rr,
                    mode='markers',
                    name=f'Excluded ({len(excluded_ts)})',
                    marker=dict(size=10, color='gray', symbol='circle-x-open', line=dict(width=1)),
                    opacity=0.5,
                    hovertemplate='Time: %{x}<br>RR: %{y} ms (EXCLUDED - click to re-enable)<extra></extra>'
                ))

            # Add segment boundary markers (cyan diamonds) when using boundary mode
            segment_boundaries = artifact_result.get("segment_boundaries", [])
            if segment_boundaries:
                boundary_ts = [timestamps_list[i] for i in segment_boundaries if 0 <= i < len(timestamps_list)]
                boundary_rr = [rr_list[i] for i in segment_boundaries if 0 <= i < len(rr_list)]
                if boundary_ts:
                    fig.add_trace(ScatterType(
                        x=boundary_ts,
                        y=boundary_rr,
                        mode='markers',
                        name=f'Segment Boundaries ({len(boundary_ts)})',
                        marker=dict(size=10, color='cyan', symbol='diamond-open', line=dict(width=2)),
                        hovertemplate='Time: %{x}<br>RR: %{y} ms (SEGMENT BOUNDARY)<extra></extra>'
                    ))

            # Collect manual artifact indices that are valid for current plot data (manual_artifacts already defined above)
            manual_artifact_indices = []
            for art in manual_artifacts:
                plot_idx = art.get('plot_idx', -1)
                if 0 <= plot_idx < len(rr_list):
                    manual_artifact_indices.append(plot_idx)

            # Show corrected preview using NeuroKit2's signal_fixpeaks correction
            # Note: Manual artifacts are not included in the NeuroKit2 correction
            algo_count = len(artifact_result["artifact_indices"])
            if show_corrected and algo_count > 0:
                corrected_rr = artifact_result.get("corrected_rr", rr_list)
                if corrected_rr and len(corrected_rr) == len(timestamps_list):
                    fig.add_trace(ScatterType(
                        x=timestamps_list,
                        y=corrected_rr,
                        mode='lines',
                        name='Corrected (NN)',
                        line=dict(width=2, color='green', dash='dot'),
                        opacity=0.7,
                        hovertemplate='Time: %{x}<br>NN: %{y:.0f} ms<extra></extra>'
                    ))
                    manual_count = len(manual_artifact_indices)
                    if manual_count > 0:
                        st.success(f"Correction preview: {algo_count} algorithm artifacts corrected (NeuroKit2 Kubios). {manual_count} manual artifacts marked but not in correction.")
                    else:
                        st.success(f"Correction preview: {algo_count} artifacts corrected (NeuroKit2 Kubios algorithm)")
        elif artifact_result.get("no_detection_yet"):
            # No detection has been run yet - show instructions
            st.info("No artifact detection yet. Use **Detect New Artifacts** to run detection.")
        elif artifact_result.get("restored_from_save") and artifact_result["total_artifacts"] == 0:
            # Loaded from save but had no artifacts
            st.info("No artifacts in saved data. Use **Detect New Artifacts** to run new detection.")

    # Display manual artifacts (purple diamond markers) - only when show_artifacts is enabled
    if show_artifacts and manual_artifacts:
        # Get timestamps and RR values for manual artifacts
        manual_ts = []
        manual_rr = []
        timestamps = plot_data['timestamps']
        rr_values = plot_data['rr_values']

        for art in manual_artifacts:
            plot_idx = art.get('plot_idx', 0)
            if 0 <= plot_idx < len(timestamps):
                manual_ts.append(timestamps[plot_idx])
                manual_rr.append(rr_values[plot_idx])

        if manual_ts:
            fig.add_trace(ScatterType(
                x=manual_ts,
                y=manual_rr,
                mode='markers',
                name=f'Manual Artifacts ({len(manual_ts)})',
                marker=dict(size=12, color='purple', symbol='diamond', line=dict(width=2, color='white')),
                hovertemplate='Time: %{x}<br>RR: %{y} ms (MANUAL)<extra></extra>'
            ))

    if show_variability:
        changepoint_result = cached_quality_analysis(tuple(rr_list), tuple(timestamps_list))
        st.session_state[f"changepoints_{participant_id}"] = changepoint_result

        if changepoint_result and changepoint_result.get("changepoint_indices"):
            n_ts = len(timestamps_list)
            for seg_stats in changepoint_result["segment_stats"]:
                start_idx = seg_stats["start_idx"]
                end_idx = min(seg_stats["end_idx"], n_ts - 1)
                if start_idx < n_ts and end_idx < n_ts:
                    cv = seg_stats.get("cv", 0)
                    if cv > 0.15:
                        fill_color = 'rgba(255, 0, 0, 0.1)'
                    elif cv > 0.10:
                        fill_color = 'rgba(255, 165, 0, 0.1)'
                    else:
                        fill_color = 'rgba(0, 255, 0, 0.05)'
                    fig.add_shape(
                        type="rect", x0=timestamps_list[start_idx], x1=timestamps_list[end_idx],
                        y0=y_min - 0.05 * y_range, y1=y_max + 0.05 * y_range,
                        fillcolor=fill_color, line=dict(width=0), layer="below"
                    )

    # Visualize gaps (skip for VNS data and Signal Inspection mode - timestamps are synthesized)
    # PERFORMANCE: Limit gaps shown to prevent plot slowdown
    MAX_GAPS_SHOWN = 50
    if show_gaps and gap_result.get("gaps") and not is_vns_data and not use_sequential_timestamps:
        gaps_to_show = gap_result["gaps"]
        total_gaps = len(gaps_to_show)
        if total_gaps > MAX_GAPS_SHOWN:
            # Show largest gaps only when there are too many
            gaps_to_show = sorted(gaps_to_show, key=lambda g: g["duration_s"], reverse=True)[:MAX_GAPS_SHOWN]
            st.caption(f"Showing {MAX_GAPS_SHOWN} largest gaps of {total_gaps} total (raise threshold to reduce)")

        for gap in gaps_to_show:
            fig.add_shape(
                type="rect", x0=gap["start_time"], x1=gap["end_time"],
                y0=y_min - 0.05 * y_range, y1=y_max + 0.05 * y_range,
                fillcolor='rgba(128, 128, 128, 0.3)',
                line=dict(color='rgba(128, 128, 128, 0.8)', width=2, dash='dot'),
                layer="below"
            )
            mid_time = gap["start_time"] + (gap["end_time"] - gap["start_time"]) / 2
            fig.add_annotation(
                x=mid_time, y=y_min - 0.1 * y_range,
                text=f"GAP: {gap['duration_s']:.1f}s",
                showarrow=False, font=dict(color='red', size=9),
                bgcolor='rgba(255,255,255,0.8)'
            )

    # In Signal Inspection mode, show gap markers at sequential positions
    # These mark where signal loss occurred in the original recording
    # Use the same gap_result from cached_gap_detection for consistency
    if use_sequential_timestamps and show_gaps and not is_vns_data:
        gaps_from_result = gap_result.get("gaps", [])
        MAX_GAPS_SHOWN = 50
        gaps_to_show = gaps_from_result[:MAX_GAPS_SHOWN]
        if len(gaps_from_result) > MAX_GAPS_SHOWN:
            st.caption(f"Showing {MAX_GAPS_SHOWN} of {len(gaps_from_result)} signal loss markers")

        for gap_info in gaps_to_show:
            # Get the end index of the gap (beat after gap)
            gap_end_idx = gap_info.get('end_idx', 0)
            gap_duration = gap_info.get('duration_s', 0)
            # Map to sequential timestamp position
            if gap_end_idx < len(sequential_timestamps):
                gap_ts = sequential_timestamps[gap_end_idx]
                # Add a vertical line at the gap position (dashed gray)
                fig.add_shape(
                    type="line", x0=gap_ts, x1=gap_ts,
                    y0=y_min - 0.05 * y_range, y1=y_max + 0.05 * y_range,
                    line=dict(color='rgba(128, 128, 128, 0.8)', width=2, dash='dot'),
                )
                fig.add_annotation(
                    x=gap_ts, y=y_min - 0.1 * y_range,
                    text=f"GAP: {gap_duration:.1f}s",
                    showarrow=False, font=dict(color='gray', size=8),
                    bgcolor='rgba(255,255,255,0.7)'
                )

    # Music sections
    music_events = stored_data.get('music_events', [])
    if show_music_sections and music_events:
        music_colors = {
            'music_1': 'rgba(65, 105, 225, 0.15)',
            'music_2': 'rgba(50, 205, 50, 0.15)',
            'music_3': 'rgba(255, 140, 0, 0.15)',
        }
        music_sections = {}
        for evt in music_events:
            # Handle both dict (from YAML) and object formats
            if isinstance(evt, dict):
                label = evt.get('raw_label') or str(evt)
                timestamp = evt.get('first_timestamp')
            else:
                label = evt.raw_label if hasattr(evt, 'raw_label') else str(evt)
                timestamp = evt.first_timestamp if hasattr(evt, 'first_timestamp') else None
            if not timestamp:
                continue
            # Convert string timestamps from YAML to datetime
            if isinstance(timestamp, str):
                from datetime import datetime
                timestamp = datetime.fromisoformat(timestamp)
            if label.endswith('_start'):
                music_type = label.replace('_start', '')
                if music_type not in music_sections:
                    music_sections[music_type] = []
                music_sections[music_type].append({'start': timestamp, 'end': None})
            elif label.endswith('_end'):
                music_type = label.replace('_end', '')
                if music_type in music_sections:
                    for sec in reversed(music_sections[music_type]):
                        if sec['end'] is None:
                            sec['end'] = timestamp
                            break

        for music_type, sections in music_sections.items():
            color = music_colors.get(music_type, 'rgba(128, 128, 128, 0.1)')
            for sec in sections:
                if sec['start'] and sec['end']:
                    fig.add_shape(
                        type="rect", x0=sec['start'], x1=sec['end'],
                        y0=y_min - 0.05 * y_range, y1=y_max + 0.05 * y_range,
                        fillcolor=color, line=dict(width=0), layer="below"
                    )
                    mid_time = sec['start'] + (sec['end'] - sec['start']) / 2
                    fig.add_annotation(
                        x=mid_time, y=y_max + 0.08 * y_range,
                        text=music_type.replace('_', ' ').title(),
                        showarrow=False, font=dict(size=8, color='gray')
                    )

    # Music event lines
    if show_music_events and music_events:
        music_line_colors = {
            'music_1': '#4169E1', 'music_2': '#32CD32', 'music_3': '#FF8C00',
        }
        for evt in music_events:
            # Handle both dict (from YAML) and object formats
            if isinstance(evt, dict):
                label = evt.get('raw_label') or str(evt)
                timestamp = evt.get('first_timestamp')
            else:
                label = evt.raw_label if hasattr(evt, 'raw_label') else str(evt)
                timestamp = evt.first_timestamp if hasattr(evt, 'first_timestamp') else None
            if timestamp:
                # Convert string timestamps from YAML to datetime
                if isinstance(timestamp, str):
                    from datetime import datetime
                    timestamp = datetime.fromisoformat(timestamp)
                music_type = label.replace('_start', '').replace('_end', '')
                color = music_line_colors.get(music_type, '#808080')
                fig.add_shape(
                    type="line", x0=timestamp, x1=timestamp,
                    y0=y_min - 0.05 * y_range, y1=y_max + 0.05 * y_range,
                    line=dict(color=color, width=1, dash='dot'), opacity=0.5
                )

    # Exclusion zones (red semi-transparent rectangles) - conditional on show_exclusions
    exclusion_zones = stored_data.get('exclusion_zones', [])
    if show_exclusions and exclusion_zones:
        for zone in exclusion_zones:
            zone_start = zone.get('start')
            zone_end = zone.get('end')
            if zone_start and zone_end:
                # Convert ISO strings back to datetime if needed
                if isinstance(zone_start, str):
                    zone_start = get_pandas().to_datetime(zone_start)
                if isinstance(zone_end, str):
                    zone_end = get_pandas().to_datetime(zone_end)

                # Draw exclusion zone as red rectangle
                fig.add_shape(
                    type="rect",
                    x0=zone_start, x1=zone_end,
                    y0=y_min - 0.05 * y_range, y1=y_max + 0.05 * y_range,
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(color='rgba(255, 0, 0, 0.8)', width=2),
                    layer="below"
                )
                # Add vertical label at start of exclusion zone (like event labels)
                reason = zone.get('reason', '')[:15]  # Truncate long reasons
                exclude_dur = zone.get('exclude_from_duration', True)
                label_text = reason if reason else "Excluded"
                if exclude_dur:
                    label_text += " [excl]"
                fig.add_annotation(
                    x=zone_start, y=y_max + 0.08 * y_range,  # Position at start
                    text=label_text,
                    showarrow=False, textangle=-90,  # Vertical like events
                    font=dict(color='darkred', size=10)
                )

    # Show pending exclusion click points on the plot (only in Add Exclusions mode)
    # Get current interaction mode to check if we should show exclusion markers
    plot_mode_key = f"plot_mode_{participant_id}"
    current_interaction_mode = st.session_state.get(plot_mode_key, "Add Events")

    exclusion_click_key = f"exclusion_clicks_{participant_id}"
    pending_clicks = st.session_state.get(exclusion_click_key, [])
    # Only show markers if in Add Exclusions mode to avoid visual clutter in other modes
    if pending_clicks and current_interaction_mode == "Add Exclusions":
        # Draw markers for pending exclusion points
        click_times = []
        click_y_values = []
        for click_ts in pending_clicks:
            click_times.append(click_ts)
            # Find closest RR value for y-position
            click_y_values.append(y_max + 0.02 * y_range)

        fig.add_trace(ScatterType(
            x=click_times,
            y=click_y_values,
            mode='markers',
            name='Exclusion Points',
            marker=dict(size=15, color='red', symbol='diamond'),
            hovertemplate='Exclusion point: %{x}<extra></extra>'
        ))

        # Draw vertical lines and labels
        if len(pending_clicks) == 1:
            # One point - show START label
            fig.add_shape(
                type="line",
                x0=pending_clicks[0], x1=pending_clicks[0],
                y0=y_min - 0.05 * y_range, y1=y_max + 0.05 * y_range,
                line=dict(color='red', width=2, dash='dash'),
                opacity=0.7
            )
            fig.add_annotation(
                x=pending_clicks[0], y=y_max + 0.12 * y_range,
                text="START",
                showarrow=False, font=dict(color='red', size=10, weight='bold'),
                bgcolor='rgba(255,255,255,0.9)'
            )
        elif len(pending_clicks) >= 2:
            # Two points - show START and END labels with shaded region
            sorted_clicks = sorted(pending_clicks[:2])
            for ts, label in zip(sorted_clicks, ["START", "END"]):
                fig.add_shape(
                    type="line",
                    x0=ts, x1=ts,
                    y0=y_min - 0.05 * y_range, y1=y_max + 0.05 * y_range,
                    line=dict(color='red', width=2, dash='dash'),
                    opacity=0.7
                )
                fig.add_annotation(
                    x=ts, y=y_max + 0.12 * y_range,
                    text=label,
                    showarrow=False, font=dict(color='red', size=10, weight='bold'),
                    bgcolor='rgba(255,255,255,0.9)'
                )
            # Draw shaded region
            fig.add_shape(
                type="rect",
                x0=sorted_clicks[0], x1=sorted_clicks[1],
                y0=y_min - 0.05 * y_range, y1=y_max + 0.05 * y_range,
                fillcolor="red",
                opacity=0.1,
                line=dict(width=0)
            )

    # Check current interaction mode (needed for conditional display below)
    plot_mode_key = f"plot_mode_{participant_id}"
    current_interaction_mode = st.session_state.get(plot_mode_key, "Add Events")

    # Check if in click-two-points exclusion mode (must also be in Add Exclusions mode)
    exclusion_method_key = f"exclusion_method_{participant_id}"
    is_exclusion_click_mode_check = (
        current_interaction_mode == "Add Exclusions" and
        exclusion_method_key in st.session_state and
        st.session_state[exclusion_method_key] == "Click two points on plot"
    )

    # Display interactive plot with click detection

    col_mode_info, col_refresh = st.columns([5, 1])
    with col_mode_info:
        if is_exclusion_click_mode_check:
            st.info("**Click two points** on the plot to define an exclusion zone (start → end)")
        elif current_interaction_mode == "Signal Inspection":
            st.info("Click on a beat to mark/unmark it as a manual artifact (purple diamonds)")
        elif current_interaction_mode == "Add Events":
            st.info("Click on the plot to add a new event at that timestamp")
    with col_refresh:
        if st.button("Refresh", key=f"refresh_plot_{participant_id}", help="Refresh plot to show new markers (resets zoom)"):
            st.rerun()

    # Store current zoom range in session state for potential restoration
    zoom_key = f"plot_zoom_{participant_id}"

    # Use a stable key to help preserve component state
    selected_points = plotly_events(
        fig,
        click_event=True,
        hover_event=False,
        select_event=False,
        override_height=600,
        key=f"plotly_events_{participant_id}"
    )

    # Handle click - check if we're in exclusion click mode
    exclusion_click_key = f"exclusion_clicks_{participant_id}"
    # Must verify BOTH that exclusion method is set AND we're in Add Exclusions mode
    is_exclusion_click_mode = (
        current_interaction_mode == "Add Exclusions" and
        exclusion_method_key in st.session_state and
        st.session_state[exclusion_method_key] == "Click two points on plot"
    )
    
    # Clear pending exclusion clicks if we switched away from Add Exclusions mode
    # (This prevents old clicks from blocking other modes)
    if current_interaction_mode != "Add Exclusions" and exclusion_click_key in st.session_state:
        if st.session_state[exclusion_click_key]:  # Only clear if there are pending clicks
            st.session_state[exclusion_click_key] = []

    # Handle click immediately
    # Track the last processed click to avoid reprocessing on rerun
    last_click_key = f"last_click_{participant_id}"

    if selected_points and len(selected_points) > 0:
        clicked_point = selected_points[0]
        if 'x' in clicked_point:
            clicked_ts = get_pandas().to_datetime(clicked_point['x'])
            clicked_time_str = clicked_ts.strftime("%H:%M:%S.%f")  # Include microseconds for uniqueness

            # Check if this is a new click (not a re-processed one)
            last_click = st.session_state.get(last_click_key)
            is_new_click = (last_click != clicked_time_str)

            # Check if we're in exclusion click mode
            if is_exclusion_click_mode:
                # Initialize clicks list if needed
                if exclusion_click_key not in st.session_state:
                    st.session_state[exclusion_click_key] = []

                current_clicks = st.session_state[exclusion_click_key]

                # Only add if this is a genuinely new click AND we have less than 2 points
                if is_new_click and len(current_clicks) < 2:
                    # Store this as the last processed click
                    st.session_state[last_click_key] = clicked_time_str

                    # Add this click to the list
                    st.session_state[exclusion_click_key].append(clicked_ts)
                    display_time = clicked_ts.strftime("%H:%M:%S")
                    st.toast(f"Exclusion point {len(st.session_state[exclusion_click_key])}: {display_time}")

                    # Only rerun for second point (to show confirmation form)
                    # First point: don't rerun to avoid zoom reset - marker shows on next interaction
                    if len(st.session_state[exclusion_click_key]) >= 2:
                        st.rerun(scope="fragment")

                # Always return early in exclusion mode (don't show event form)
                return

            # Check current interaction mode - only show event form in "Add Events" mode
            plot_mode_key = f"plot_mode_{participant_id}"
            current_mode = st.session_state.get(plot_mode_key, "Add Events")

            # Handle Signal Inspection mode - manual artifact marking
            if current_mode == "Signal Inspection":
                # Use a "just processed" key to prevent reprocessing the same click after rerun
                # This is simpler and more reliable than including artifact count in signature
                just_processed_key = f"artifact_just_processed_{participant_id}"

                # Check if this click was just processed (post-rerun)
                just_processed = st.session_state.get(just_processed_key)
                if just_processed == clicked_time_str:
                    # This is a post-rerun duplicate - clear the flag and skip
                    st.session_state[just_processed_key] = None
                    return

                # Find nearest beat index to clicked timestamp
                timestamps = plot_data['timestamps']
                rr_values = plot_data['rr_values']

                # Convert clicked timestamp to comparable format
                import numpy as np
                ts_array = get_pandas().to_datetime(timestamps)

                # Ensure both timestamps have matching timezone awareness
                if ts_array.tz is not None and clicked_ts.tzinfo is None:
                    clicked_ts = clicked_ts.tz_localize(ts_array.tz)
                elif ts_array.tz is None and clicked_ts.tzinfo is not None:
                    clicked_ts = clicked_ts.tz_localize(None)

                time_diffs = np.abs((ts_array - clicked_ts).total_seconds())
                nearest_idx = int(np.argmin(time_diffs))

                # Only mark if click is within 2 seconds of a beat
                if time_diffs[nearest_idx] > 2.0:
                    st.info(f"Click closer to a beat to mark it. Nearest beat is {time_diffs[nearest_idx]:.1f}s away.")
                    return

                # Store this click as processed BEFORE modifying state
                st.session_state[last_click_key] = clicked_time_str

                # Get artifact-related session state keys
                manual_artifact_key = f"manual_artifacts_{participant_id}"
                artifact_exclusions_key = f"artifact_exclusions_{participant_id}"

                if manual_artifact_key not in st.session_state:
                    st.session_state[manual_artifact_key] = []
                if artifact_exclusions_key not in st.session_state:
                    st.session_state[artifact_exclusions_key] = set()

                manual_artifacts = st.session_state[manual_artifact_key]
                artifact_exclusions = st.session_state[artifact_exclusions_key]

                # Get original index for this beat
                original_idx = plot_data.get('original_indices', list(range(len(timestamps))))[nearest_idx]
                clicked_rr = rr_values[nearest_idx]
                clicked_ts_str = get_pandas().to_datetime(timestamps[nearest_idx]).strftime('%H:%M:%S.%f')

                # Check if this beat is a detected artifact (from threshold detection)
                # Artifacts are stored in session state, not plot_data
                artifact_data = st.session_state.get(f"artifacts_{participant_id}", {})
                detected_artifact_indices = artifact_data.get('artifact_indices', [])
                is_detected_artifact = original_idx in detected_artifact_indices

                # Check if this beat is a manual artifact
                existing_manual_entry = None
                for entry in manual_artifacts:
                    if entry.get('original_idx') == original_idx:
                        existing_manual_entry = entry
                        break

                # Check if this detected artifact is currently excluded
                is_excluded = original_idx in artifact_exclusions

                # Toggle logic:
                # 1. If it's a detected artifact that's NOT excluded -> exclude it (unmark)
                # 2. If it's a detected artifact that IS excluded -> re-include it (re-mark)
                # 3. If it's a manual artifact -> remove it
                # 4. If it's a normal beat -> add as manual artifact

                if is_detected_artifact:
                    if is_excluded:
                        # Re-include detected artifact (was excluded, now re-mark it)
                        artifact_exclusions.discard(original_idx)
                        st.toast(f"✓ Re-enabled detected artifact at {clicked_ts_str} (RR={clicked_rr}ms)")
                    else:
                        # Exclude detected artifact (unmark it)
                        artifact_exclusions.add(original_idx)
                        st.toast(f"✗ Excluded detected artifact at {clicked_ts_str} (RR={clicked_rr}ms)")
                    st.session_state[artifact_exclusions_key] = artifact_exclusions
                elif existing_manual_entry:
                    # Remove from manual artifacts (toggle off)
                    manual_artifacts.remove(existing_manual_entry)
                    st.toast(f"Removed manual artifact at {clicked_ts_str} (RR={clicked_rr}ms)")
                    st.session_state[manual_artifact_key] = manual_artifacts
                else:
                    # Add to manual artifacts (toggle on)
                    manual_artifacts.append({
                        'original_idx': original_idx,
                        'plot_idx': nearest_idx,
                        'timestamp': clicked_ts_str,
                        'rr_value': clicked_rr,
                        'source': 'manual'
                    })
                    st.toast(f"Marked as artifact at {clicked_ts_str} (RR={clicked_rr}ms)")
                    st.session_state[manual_artifact_key] = manual_artifacts

                # Mark this click as just processed to prevent reprocessing after rerun
                st.session_state[just_processed_key] = clicked_time_str
                # Flag to skip heavy detection on rerun (we're just updating markers)
                st.session_state[f"artifact_marker_only_{participant_id}"] = True
                st.rerun()  # Full rerun to update Signal Inspection section

            # Only show event add form if in "Add Events" mode
            if current_mode != "Add Events":
                return

            # Show quick add form right here in the fragment
            display_time = clicked_ts.strftime("%H:%M:%S")
            st.success(f"**Clicked at {display_time}** - Add event below:")

            col_evt, col_custom, col_add = st.columns([2, 2, 1])
            with col_evt:
                quick_events = ["measurement_start", "measurement_end", "pause_start", "pause_end",
                               "rest_pre_start", "rest_pre_end", "rest_post_start", "rest_post_end",
                               "Custom..."]
                selected_evt = st.selectbox(
                    "Event type",
                    options=quick_events,
                    key=f"quick_evt_{participant_id}_{clicked_time_str}",
                    label_visibility="collapsed"
                )

            with col_custom:
                if selected_evt == "Custom...":
                    custom_evt_label = st.text_input(
                        "Custom label",
                        key=f"custom_evt_{participant_id}_{clicked_time_str}",
                        placeholder="Enter event name...",
                        label_visibility="collapsed"
                    )
                else:
                    custom_evt_label = None

            with col_add:
                if st.button("+ Add", key=f"add_click_{participant_id}_{clicked_time_str}", type="primary"):
                    from rrational.prep.summaries import EventStatus

                    # Determine label
                    event_label = custom_evt_label if selected_evt == "Custom..." else selected_evt
                    if not event_label:
                        st.error("Enter a custom event name")
                    else:
                        # Use clicked timestamp with proper timezone
                        if clicked_ts.tzinfo is None:
                            clicked_ts = clicked_ts.tz_localize('UTC')

                        new_event = EventStatus(
                            raw_label=event_label,
                            canonical=st.session_state.normalizer.normalize(event_label),
                            count=1,
                            first_timestamp=clicked_ts,
                            last_timestamp=clicked_ts,
                        )

                        if participant_id not in st.session_state.participant_events:
                            st.session_state.participant_events[participant_id] = {'events': [], 'manual': []}
                        st.session_state.participant_events[participant_id]['manual'].append(new_event)
                        st.toast(f"Added '{event_label}' at {clicked_time_str} - click 'Refresh Plot' or interact with plot to see marker")


def detect_quality_changepoints(rr_values: list[int], change_type: str = "var") -> dict:
    """Detect quality changepoints in RR interval data using NeuroKit2.

    Uses signal_changepoints() to find where signal properties change,
    which can indicate measurement issues, electrode problems, etc.

    Args:
        rr_values: List of RR interval values in ms
        change_type: Type of change to detect ("var", "mean", or "meanvar")

    Returns:
        dict with:
            - changepoint_indices: list of indices where changes occur
            - n_segments: number of segments detected
            - segment_stats: list of dicts with stats per segment
            - quality_score: 0-100 score (100 = no changepoints = stable)
    """
    if not NEUROKIT_AVAILABLE or len(rr_values) < 10:
        return {
            "changepoint_indices": [],
            "n_segments": 1,
            "segment_stats": [],
            "quality_score": 100,
        }

    try:
        import numpy as np
        rr_array = np.array(rr_values, dtype=float)

        # Detect changepoints in variance (most useful for quality issues)
        nk = get_neurokit()
        changepoints = nk.signal_changepoints(rr_array, change=change_type, show=False)

        # Calculate segment statistics
        segment_stats = []
        all_indices = [0] + list(changepoints) + [len(rr_array)]

        for i in range(len(all_indices) - 1):
            start_idx = all_indices[i]
            end_idx = all_indices[i + 1]
            segment = rr_array[start_idx:end_idx]

            if len(segment) > 0:
                segment_stats.append({
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "n_beats": len(segment),
                    "mean_rr": float(np.mean(segment)),
                    "std_rr": float(np.std(segment)),
                    "cv": float(np.std(segment) / np.mean(segment)) if np.mean(segment) > 0 else 0,
                })

        # Quality score: fewer changepoints = more stable = better quality
        # Penalize more for many changepoints
        n_changepoints = len(changepoints)
        if n_changepoints == 0:
            quality_score = 100
        elif n_changepoints <= 2:
            quality_score = 80
        elif n_changepoints <= 5:
            quality_score = 60
        else:
            quality_score = max(20, 100 - (n_changepoints * 10))

        return {
            "changepoint_indices": list(changepoints),
            "n_segments": len(segment_stats),
            "segment_stats": segment_stats,
            "quality_score": quality_score,
        }
    except Exception:
        return {
            "changepoint_indices": [],
            "n_segments": 1,
            "segment_stats": [],
            "quality_score": 100,
        }


def get_quality_badge(quality_score: float, artifact_ratio: float) -> str:
    """Return a quality badge emoji based on quality score and artifact ratio.

    Args:
        quality_score: 0-100 from changepoint detection
        artifact_ratio: 0-1 ratio of removed artifacts

    Returns:
        Emoji badge: [OK] (good), (moderate), [X] (poor)
    """
    # Combine changepoint quality and artifact ratio
    # Artifact ratio > 10% is concerning, > 20% is poor
    artifact_score = 100 - (artifact_ratio * 200)  # 10% artifacts = 80, 20% = 60
    artifact_score = max(0, min(100, artifact_score))

    combined = (quality_score + artifact_score) / 2

    if combined >= 75:
        return "[OK]"
    elif combined >= 50:
        return "[!]"
    else:
        return "[X]"


def detect_time_gaps(timestamps: list, rr_values: list = None, gap_threshold_s: float = 2.0) -> dict:
    """Detect time gaps (missing data) between consecutive RR intervals.

    HRV Logger Note: Timestamps are per-packet (~1s), not per-beat. Multiple RR
    intervals can share the same timestamp. A real gap is when the timestamp
    difference significantly exceeds what the RR intervals would predict.

    Detection method:
    - If RR values provided: gap = (timestamp_diff - expected_rr_sum) > threshold
    - If no RR values: gap = timestamp_diff > threshold (fallback)

    Args:
        timestamps: List of datetime timestamps for each RR interval
        rr_values: List of RR interval values in ms (optional, improves detection)
        gap_threshold_s: Minimum unexplained gap duration to flag (default: 2s)

    Returns:
        dict with gap details and statistics
    """
    import numpy as np

    if len(timestamps) < 2:
        return {"gaps": [], "total_gaps": 0, "total_gap_duration_s": 0.0, "gap_ratio": 0.0}

    try:
        # Convert to numpy for speed
        valid_mask = np.array([t is not None for t in timestamps])
        if not np.any(valid_mask):
            return {"gaps": [], "total_gaps": 0, "total_gap_duration_s": 0.0, "gap_ratio": 0.0}

        # Calculate timestamp differences in seconds (vectorized)
        ts_seconds = np.array([t.timestamp() if t else np.nan for t in timestamps])
        ts_diff = np.diff(ts_seconds)

        # If RR values provided, calculate expected time vs actual
        if rr_values is not None and len(rr_values) == len(timestamps):
            rr_array = np.array(rr_values, dtype=float) / 1000.0  # Convert ms to seconds
            # Expected time between consecutive beats = RR interval of the second beat
            expected_diff = rr_array[1:]  # RR[i] is duration before beat i
            # Gap = actual time diff - expected RR (unexplained time)
            unexplained_time = ts_diff - expected_diff
            gap_mask = unexplained_time > gap_threshold_s
        else:
            # Fallback: just use timestamp difference
            gap_mask = ts_diff > gap_threshold_s
            unexplained_time = ts_diff

        # Extract gap indices
        gap_indices = np.where(gap_mask)[0]

        gaps = []
        total_gap_duration = 0.0

        for idx in gap_indices:
            gap_duration = float(unexplained_time[idx]) if rr_values else float(ts_diff[idx])
            gap_info = {
                "start_idx": int(idx),
                "end_idx": int(idx + 1),
                "start_time": timestamps[idx],
                "end_time": timestamps[idx + 1],
                "duration_s": gap_duration,
                "timestamp_diff_s": float(ts_diff[idx]),
            }
            gaps.append(gap_info)
            total_gap_duration += gap_duration

        # Calculate total recording duration
        total_duration = ts_seconds[-1] - ts_seconds[0] if not np.isnan(ts_seconds[0]) else 0
        gap_ratio = total_gap_duration / total_duration if total_duration > 0 else 0

        return {
            "gaps": gaps,
            "total_gaps": len(gaps),
            "total_gap_duration_s": total_gap_duration,
            "gap_ratio": gap_ratio,
        }
    except Exception:
        return {"gaps": [], "total_gaps": 0, "total_gap_duration_s": 0.0, "gap_ratio": 0.0}


def detect_artifacts_fixpeaks(rr_values: list[int], sampling_rate: int = 1000) -> dict:
    """Detect and optionally correct artifacts using NeuroKit2's signal_fixpeaks.

    Uses the Kubios algorithm to identify ectopic beats, missed beats,
    extra beats, and long/short intervals.

    Args:
        rr_values: List of RR interval values in ms
        sampling_rate: Sampling rate (1000 for ms intervals)

    Returns:
        dict with:
            - artifacts: dict with counts by type (ectopic, missed, extra, longshort)
            - total_artifacts: total number of artifacts
            - artifact_ratio: ratio of artifacts to total beats
            - corrected_rr: corrected RR values (if correction was successful)
            - correction_applied: whether correction was applied
    """
    if not NEUROKIT_AVAILABLE or len(rr_values) < 10:
        return {
            "artifacts": {"ectopic": 0, "missed": 0, "extra": 0, "longshort": 0},
            "total_artifacts": 0,
            "artifact_ratio": 0.0,
            "corrected_rr": rr_values,
            "correction_applied": False,
        }

    try:
        import numpy as np

        # Convert RR intervals to peak indices (cumulative sum)
        rr_array = np.array(rr_values, dtype=float)
        peak_indices = np.cumsum(rr_array).astype(int)
        peak_indices = np.insert(peak_indices, 0, 0)  # Add starting point

        # Use signal_fixpeaks with Kubios method
        nk = get_neurokit()
        info, corrected_peaks = nk.signal_fixpeaks(
            peak_indices,
            sampling_rate=sampling_rate,
            iterative=True,
            method="Kubios",
            show=False,
        )

        # Extract artifact counts
        artifacts = {
            "ectopic": len(info.get("ectopic", [])) if isinstance(info.get("ectopic"), (list, np.ndarray)) else 0,
            "missed": len(info.get("missed", [])) if isinstance(info.get("missed"), (list, np.ndarray)) else 0,
            "extra": len(info.get("extra", [])) if isinstance(info.get("extra"), (list, np.ndarray)) else 0,
            "longshort": len(info.get("longshort", [])) if isinstance(info.get("longshort"), (list, np.ndarray)) else 0,
        }

        total_artifacts = sum(artifacts.values())
        artifact_ratio = total_artifacts / len(rr_values) if rr_values else 0

        # Convert corrected peaks back to RR intervals
        corrected_rr = list(np.diff(corrected_peaks))

        return {
            "artifacts": artifacts,
            "total_artifacts": total_artifacts,
            "artifact_ratio": artifact_ratio,
            "corrected_rr": corrected_rr,
            "correction_applied": total_artifacts > 0,
        }
    except Exception:
        return {
            "artifacts": {"ectopic": 0, "missed": 0, "extra": 0, "longshort": 0},
            "total_artifacts": 0,
            "artifact_ratio": 0.0,
            "corrected_rr": rr_values,
            "correction_applied": False,
        }


def _load_project(project_path: Path | str | None) -> None:
    """Load project configuration into session state.

    Args:
        project_path: Path to project directory, or None for temporary workspace
    """
    from rrational.gui.persistence import save_last_project

    if project_path is None or project_path == "":
        # Temporary workspace mode - use global config
        st.session_state.current_project = None
        st.session_state.project_manager = None
        save_last_project(None)  # Clear last project
        return

    from rrational.gui.project import ProjectManager, add_recent_project

    project_path = Path(project_path)
    pm = ProjectManager.open_project(project_path)
    st.session_state.current_project = project_path
    st.session_state.project_manager = pm

    # Update recent projects and save as last project for auto-load
    add_recent_project(project_path, pm.metadata.name if pm.metadata else project_path.name)
    save_last_project(project_path)

    # Clear and reload config from project
    for key in ["groups", "all_events", "sections", "playlist_groups",
                "participant_groups", "participant_randomizations",
                "participant_playlists", "participant_labels",
                "event_order", "manual_events"]:
        if key in st.session_state:
            del st.session_state[key]

    # Set data_dir to project's data/raw folder
    st.session_state.data_dir = str(pm.get_data_dir())

    # Clear loaded data - delete key entirely so auto-load can trigger
    if "summaries" in st.session_state:
        del st.session_state["summaries"]


def main():
    """Main Streamlit app."""
    import time as _time
    _script_start = _time.time()

    # Auto-load last project on startup (only on first run)
    if "startup_complete" not in st.session_state:
        st.session_state.startup_complete = True
        if not TEST_MODE:
            from rrational.gui.persistence import get_last_project
            last_project = get_last_project()
            if last_project:
                # Auto-load the last used project
                _load_project(Path(last_project))
                st.session_state.show_welcome = False
                st.rerun()

    # Project selection gate - show welcome screen if no project selected
    if st.session_state.get("show_welcome", True) and not st.session_state.get("current_project"):
        # Don't show welcome in TEST_MODE
        if not TEST_MODE:
            from rrational.gui.welcome import render_welcome_screen
            result = render_welcome_screen()
            if result is not None:
                if result == "":  # Temporary workspace
                    st.session_state.show_welcome = False
                else:
                    _load_project(Path(result))
                    st.session_state.show_welcome = False
                st.rerun()
            return  # Don't render main app until project selected
        else:
            # In test mode, skip welcome screen
            st.session_state.show_welcome = False

    if TEST_MODE:
        st.title("RRational [TEST MODE]")
        st.info("**Test mode active** - Using demo data from `data/demo/hrv_logger`")
    else:
        st.title("RRational")

    # Show current project name prominently below title
    _current_project = st.session_state.get("current_project")
    if _current_project:
        _pm = st.session_state.get("project_manager")
        _project_name = _pm.metadata.name if _pm and _pm.metadata else Path(_current_project).name
        st.markdown(f"#### Project: {_project_name}")
    else:
        st.markdown("#### Temporary Workspace")

    # Sidebar navigation using buttons (fast - only renders active page)
    # Initialize active page
    if "active_page" not in st.session_state:
        st.session_state.active_page = "Data"

    # CSS for compact sidebar navigation
    st.markdown("""
    <style>
    /* Compact sidebar navigation */
    section[data-testid="stSidebar"] .stButton button {
        margin: 0;
        padding: 0.4rem 0.8rem;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        # Navigation buttons - no emojis
        pages = ["Data", "Participants", "Setup", "Analysis"]

        for page_id in pages:
            # Highlight active page with primary button
            if st.session_state.active_page == page_id:
                st.button(page_id, key=f"nav_{page_id}", width='stretch', type="primary")
            else:
                if st.button(page_id, key=f"nav_{page_id}", width='stretch', type="secondary"):
                    st.session_state.active_page = page_id
                    st.session_state._scroll_to_top = True  # Scroll to top on tab switch
                    # Extra scroll trigger for Setup tab (content renders after main scroll)
                    if page_id == "Setup":
                        st.session_state._setup_scroll_to_top = True
                    st.rerun()

        st.markdown("---")

        # Project indicator
        if st.session_state.get("current_project"):
            pm = st.session_state.project_manager
            project_name = pm.metadata.name if pm and pm.metadata else "Project"
            st.caption(f"Project: **{project_name}**")
            if st.button("Switch Project", key="switch_project", width="stretch"):
                # Clear current project to show welcome screen
                st.session_state.current_project = None
                st.session_state.project_manager = None
                st.session_state.show_welcome = True
                st.rerun()
        else:
            st.caption("Temporary Workspace")
            if st.button("Open Project", key="open_project", width="stretch"):
                st.session_state.show_welcome = True
                st.rerun()

        st.markdown("---")

        # Show status in sidebar
        if st.session_state.summaries:
            st.caption(f"{len(st.session_state.summaries)} participants loaded")
        else:
            st.caption("No data loaded")

        # Show last save time if available
        if "last_save_time" in st.session_state:
            elapsed = time.time() - st.session_state.last_save_time
            if elapsed < 3:
                st.success("Saved")

        # Debug: Show script execution time
        if st.session_state.get("last_render_time"):
            st.caption(f"{st.session_state.last_render_time:.0f}ms render")

        # Settings section
        st.markdown("---")
        with st.expander("Settings", expanded=False):
            render_settings_panel()

    # Get selected page for content rendering
    selected_page = st.session_state.active_page

    # Scroll to top when switching tabs (applies to all pages)
    if st.session_state.get("_scroll_to_top", False):
        st.session_state._scroll_to_top = False
        st.components.v1.html(
            """
            <script>
                // Delay scroll to ensure page content is rendered
                setTimeout(function() {
                    // Try multiple selectors for different Streamlit versions
                    var mainSection = window.parent.document.querySelector('[data-testid="stMainBlockContainer"]');
                    if (!mainSection) mainSection = window.parent.document.querySelector('section.main');
                    if (!mainSection) mainSection = window.parent.document.querySelector('.main');
                    if (mainSection) {
                        mainSection.scrollTop = 0;
                    }
                    // Also scroll the whole document
                    window.parent.document.documentElement.scrollTop = 0;
                    window.parent.document.body.scrollTop = 0;

                    // Find and scroll the main scrollable container
                    var appViewContainer = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
                    if (appViewContainer) {
                        appViewContainer.scrollTop = 0;
                    }
                    var stMain = window.parent.document.querySelector('.stMain');
                    if (stMain) {
                        stMain.scrollTop = 0;
                    }
                }, 100);  // 100ms delay for content to render
            </script>
            """,
            height=0
        )

    # ================== PAGE: DATA ==================
    if selected_page == "Data":
        _get_render_data_tab()()


    # ================== TAB: PARTICIPANTS ==================
    elif selected_page == "Participants":
        st.header("Participant Details")

        if not st.session_state.summaries:
            st.info("Load data in the **Data** tab first to view participant details.")
        else:

            participant_list = get_participant_list()  # Cached for performance

            # Initialize selected participant index
            if "current_participant_idx" not in st.session_state:
                st.session_state.current_participant_idx = 0

            # Ensure index is valid
            if st.session_state.current_participant_idx >= len(participant_list):
                st.session_state.current_participant_idx = len(participant_list) - 1
            if st.session_state.current_participant_idx < 0:
                st.session_state.current_participant_idx = 0

            current_idx = st.session_state.current_participant_idx
            selected_participant = participant_list[current_idx] if participant_list else None

            # Navigation row
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                def on_select_change():
                    # Find index of selected participant
                    selected = st.session_state.participant_selector
                    if selected in participant_list:
                        st.session_state.current_participant_idx = participant_list.index(selected)
                        st.session_state.scroll_to_top_trigger = True

                st.selectbox(
                    "Select participant",
                    options=participant_list,
                    index=current_idx,
                    key="participant_selector",
                    label_visibility="collapsed",
                    on_change=on_select_change
                )

            with col2:
                def go_previous():
                    if st.session_state.current_participant_idx > 0:
                        st.session_state.current_participant_idx -= 1
                        # Sync selectbox key with new index
                        new_participant = participant_list[st.session_state.current_participant_idx]
                        st.session_state.participant_selector = new_participant
                        st.session_state.scroll_to_top_trigger = True

                st.button(
                    "Previous",
                    disabled=current_idx == 0,
                    key="prev_btn",
                    width='stretch',
                    on_click=go_previous
                )

            with col3:
                def go_next():
                    if st.session_state.current_participant_idx < len(participant_list) - 1:
                        st.session_state.current_participant_idx += 1
                        # Sync selectbox key with new index
                        new_participant = participant_list[st.session_state.current_participant_idx]
                        st.session_state.participant_selector = new_participant
                        st.session_state.scroll_to_top_trigger = True

                st.button(
                    "Next",
                    disabled=current_idx >= len(participant_list) - 1,
                    key="next_btn",
                    width='stretch',
                    on_click=go_next
                )

            # Scroll to top when navigating between participants
            if st.session_state.get("scroll_to_top_trigger", False):
                st.session_state.scroll_to_top_trigger = False
                st.components.v1.html("""
                    <script>
                        var streamlitDoc = window.parent.document;
                        var appContainer = streamlitDoc.querySelector('[data-testid="stAppViewContainer"]');
                        if (appContainer) appContainer.scrollTop = 0;
                    </script>
                """, height=0)

            # Update selected_participant from current index
            selected_participant = participant_list[st.session_state.current_participant_idx] if participant_list else None

            # Participant info header
            if selected_participant:
                summary = get_summary_dict().get(selected_participant)

                # Get group with label
                assigned_group = st.session_state.participant_groups.get(selected_participant, "Default")
                group_label = st.session_state.groups.get(assigned_group, {}).get("label", assigned_group)
                group_display = f"{group_label}" if group_label != assigned_group else assigned_group

                # Get randomization with label (check playlist_groups first, then custom labels)
                assigned_randomization = st.session_state.get("participant_randomizations", {}).get(selected_participant, "")
                if assigned_randomization:
                    # Try playlist_groups first, then custom randomization_labels
                    if assigned_randomization in st.session_state.get("playlist_groups", {}):
                        rand_label = st.session_state.playlist_groups[assigned_randomization].get("label", assigned_randomization)
                    else:
                        rand_label = st.session_state.get("randomization_labels", {}).get(assigned_randomization, assigned_randomization)
                    rand_display = f"{rand_label}" if rand_label != assigned_randomization else assigned_randomization
                else:
                    rand_display = "Not assigned"

                st.markdown(f"**{selected_participant}** | Group: {group_display} | Randomization: {rand_display} | ({current_idx + 1} of {len(participant_list)})")

                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Beats", summary.total_beats)
                with col2:
                    st.metric("Retained", summary.retained_beats)
                with col3:
                    st.metric("Duplicates", summary.duplicate_rr_intervals)
                with col4:
                    st.metric("Duration", f"{summary.duration_s / 60:.1f} min")

                # ISSUE 1 FIX: Show warning and expandable duplicate details if duplicates detected
                if summary.duplicate_rr_intervals > 0:
                    st.error(
                        f"**{summary.duplicate_rr_intervals} duplicate RR intervals** were detected and removed! "
                        f"This participant may have corrupted data."
                    )

                    # ISSUE 1 FIX: Display duplicate details in expandable section
                    if summary.duplicate_details:
                        with st.expander(f"Show Duplicate Details ({len(summary.duplicate_details)} duplicates)"):
                            # Show first 10 duplicates by default
                            num_to_show = min(10, len(summary.duplicate_details))

                            for i, dup in enumerate(summary.duplicate_details[:num_to_show]):
                                st.text(
                                    f"Line {dup.original_line} (original) duplicated at Line {dup.duplicate_line}: "
                                    f"date={dup.date_str}, rr={dup.rr_str}, elapsed={dup.elapsed_str}"
                                )

                            # Show remaining duplicates if user wants to see more
                            if len(summary.duplicate_details) > 10:
                                if st.button(f"Show all {len(summary.duplicate_details)} duplicates", key=f"show_all_dups_{selected_participant}"):
                                    st.markdown("**All Duplicates:**")
                                    for dup in summary.duplicate_details:
                                        st.text(
                                            f"Line {dup.original_line} (original) duplicated at Line {dup.duplicate_line}: "
                                            f"date={dup.date_str}, rr={dup.rr_str}, elapsed={dup.elapsed_str}"
                                        )

                # Recording date/time
                if summary.recording_datetime:
                    st.info(f" Recording Date: {summary.recording_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

                # Check for saved .rrational files and artifact corrections
                from rrational.gui.rrational_export import find_rrational_files
                from rrational.gui.persistence import load_artifact_corrections

                data_dir = st.session_state.get("data_dir", "")
                project_path = st.session_state.get("current_project")

                ready_files = find_rrational_files(selected_participant, data_dir, project_path)
                saved_artifacts = load_artifact_corrections(selected_participant, data_dir, project_path, section_key=None)

                # Show status badges
                status_parts = []
                if ready_files:
                    status_parts.append(f"**{len(ready_files)}** .rrational file(s)")
                if saved_artifacts:
                    # Handle v1.3+ format with sections
                    if "sections" in saved_artifacts:
                        sections = saved_artifacts.get("sections", {})
                        n_sections = len(sections)
                        total_algo = sum(len(s.get("algorithm_artifact_indices", [])) for s in sections.values())
                        total_manual = sum(len(s.get("manual_artifacts", [])) for s in sections.values())
                        total_excluded = sum(len(s.get("excluded_artifact_indices", [])) for s in sections.values())

                        if total_algo or total_manual or total_excluded:
                            art_parts = []
                            if total_algo:
                                art_parts.append(f"**{total_algo}** algorithm")
                            if total_manual:
                                art_parts.append(f"**{total_manual}** manual")
                            if total_excluded:
                                art_parts.append(f"**{total_excluded}** excluded")
                            section_info = f" ({n_sections} section{'s' if n_sections > 1 else ''})" if n_sections > 1 else ""
                            status_parts.append(" + ".join(art_parts) + f" artifacts saved{section_info}")
                    else:
                        # Legacy format (v1.2 and earlier)
                        n_algo = len(saved_artifacts.get("algorithm_artifact_indices", []))
                        n_manual = len(saved_artifacts.get("manual_artifacts", []))
                        n_excluded = len(saved_artifacts.get("excluded_artifact_indices", []))
                        if n_algo or n_manual or n_excluded:
                            art_parts = []
                            if n_algo:
                                art_parts.append(f"**{n_algo}** algorithm")
                            if n_manual:
                                art_parts.append(f"**{n_manual}** manual")
                            if n_excluded:
                                art_parts.append(f"**{n_excluded}** excluded")
                            status_parts.append(" + ".join(art_parts) + " artifacts saved")

                if status_parts:
                    st.success(f"Saved data: {' | '.join(status_parts)}")

                # Auto-load saved artifact corrections BEFORE plot is built
                # This ensures saved artifacts appear on the plot immediately
                artifacts_loaded_key = f"artifacts_loaded_{selected_participant}"
                if artifacts_loaded_key not in st.session_state:
                    st.session_state[artifacts_loaded_key] = True  # Mark as loaded

                    from rrational.gui.persistence import load_artifact_corrections, load_artifact_corrections_from_rrational, get_merged_artifacts_for_display
                    from rrational.gui.rrational_export import find_rrational_files

                    data_dir_load = str(st.session_state.get("data_dir", ""))
                    project_path_load = st.session_state.get("current_project")

                    # Try _artifacts.yml first - load all sections data
                    saved = load_artifact_corrections(
                        selected_participant,
                        data_dir=data_dir_load,
                        project_path=project_path_load,
                        section_key=None,  # Get all sections
                    )

                    if saved:
                        manual_artifact_key_load = f"manual_artifacts_{selected_participant}"
                        artifact_exclusions_key_load = f"artifact_exclusions_{selected_participant}"
                        artifacts_key_load = f"artifacts_{selected_participant}"

                        # Handle v1.3+ format with sections
                        if "sections" in saved:
                            # Get merged artifacts from all sections for display
                            merged = get_merged_artifacts_for_display(
                                selected_participant, data_dir_load, project_path_load
                            )
                            st.session_state[manual_artifact_key_load] = merged.get("manual_artifacts", [])
                            st.session_state[artifact_exclusions_key_load] = set(merged.get("excluded_artifact_indices", []))

                            # For algorithm artifacts, merge and track sections
                            if merged.get("algorithm_artifact_indices"):
                                sections_info = merged.get("sections_info", {})
                                # Get method from first section (they might differ)
                                first_section = list(sections_info.values())[0] if sections_info else {}
                                loaded_artifact_data = {
                                    "artifact_indices": merged.get("algorithm_artifact_indices", []),
                                    "method": first_section.get("method"),
                                    "restored_from_save": True,
                                    "sections_info": sections_info,  # Track per-section info
                                }
                                st.session_state[artifacts_key_load] = loaded_artifact_data

                            # Store info for persistent notification
                            total_algo = len(merged.get("algorithm_artifact_indices", []))
                            total_manual = len(merged.get("manual_artifacts", []))
                            total_excluded = len(merged.get("excluded_artifact_indices", []))
                            if total_algo or total_manual or total_excluded:
                                n_sections = len(saved.get("sections", {}))
                                st.session_state[f"artifacts_loaded_info_{selected_participant}"] = {
                                    "saved_at": saved.get("last_modified"),
                                    "n_algorithm": total_algo,
                                    "n_manual": total_manual,
                                    "n_excluded": total_excluded,
                                    "n_sections": n_sections,
                                }
                                section_label = f" from {n_sections} section{'s' if n_sections > 1 else ''}" if n_sections > 1 else ""
                                st.toast(f"Loaded artifact corrections{section_label}")
                        else:
                            # Legacy format (v1.2 and earlier)
                            st.session_state[manual_artifact_key_load] = saved.get("manual_artifacts", [])
                            st.session_state[artifact_exclusions_key_load] = set(saved.get("excluded_artifact_indices", []))

                            # Restore algorithm artifacts if present
                            if "algorithm_artifact_indices" in saved:
                                loaded_artifact_data = {
                                    "artifact_indices": saved.get("algorithm_artifact_indices", []),
                                    "method": saved.get("algorithm_method"),
                                    "threshold": saved.get("algorithm_threshold"),
                                    "restored_from_save": True,
                                    "section_key": saved.get("section_key", "_full"),
                                }
                                # Include scope and corrected_rr if present (v1.2+)
                                if "scope" in saved:
                                    loaded_artifact_data["scope"] = saved["scope"]
                                if "corrected_rr" in saved:
                                    loaded_artifact_data["corrected_rr"] = saved["corrected_rr"]
                                st.session_state[artifacts_key_load] = loaded_artifact_data

                            # Store info for persistent notification
                            has_any = (saved.get("manual_artifacts") or
                                      saved.get("excluded_artifact_indices") or
                                      saved.get("algorithm_artifact_indices"))
                            if has_any:
                                st.session_state[f"artifacts_loaded_info_{selected_participant}"] = {
                                    "saved_at": saved.get("saved_at"),
                                    "algorithm_method": saved.get("algorithm_method"),
                                    "algorithm_threshold": saved.get("algorithm_threshold"),
                                    "n_algorithm": len(saved.get("algorithm_artifact_indices", [])),
                                    "n_manual": len(saved.get("manual_artifacts", [])),
                                    "n_excluded": len(saved.get("excluded_artifact_indices", [])),
                                }
                                st.toast("Loaded artifact corrections from saved session")
                    else:
                        # Fallback: try .rrational file
                        ready_files = find_rrational_files(selected_participant, data_dir_load, project_path_load)
                        if ready_files:
                            from_rrational = load_artifact_corrections_from_rrational(ready_files[0])
                            if from_rrational:
                                manual_artifact_key_load = f"manual_artifacts_{selected_participant}"
                                artifact_exclusions_key_load = f"artifact_exclusions_{selected_participant}"
                                st.session_state[manual_artifact_key_load] = from_rrational.get("manual_artifacts", [])
                                st.session_state[artifact_exclusions_key_load] = set(from_rrational.get("excluded_artifact_indices", []))
                                source_name = Path(from_rrational.get("source_file", "")).name
                                st.toast(f"Restored artifact markings from {source_name}")
                                st.session_state[f"artifacts_from_rrational_{selected_participant}"] = source_name

                # RR Interval Plot with Event Markers
                st.markdown("---")
                st.subheader("RR Interval Visualization")

                try:
                    # Initialize interaction_mode with default (used for plot mode selection)
                    # This prevents UnboundLocalError if plot data loading fails or is skipped
                    interaction_mode = "Add Events"

                    # Load recording data based on source app (HRV Logger or VNS)
                    source_app = getattr(summary, 'source_app', 'HRV Logger')

                    if source_app == "VNS Analyse" and getattr(summary, 'vns_path', None):
                        # Load VNS recording using stored path
                        recording_data = cached_load_vns_recording(
                            str(summary.vns_path),
                            selected_participant,
                            use_corrected=st.session_state.get("vns_use_corrected", False),
                        )
                    elif getattr(summary, 'rr_paths', None):
                        # Load HRV Logger recording using stored paths
                        events_paths = getattr(summary, 'events_paths', []) or []
                        recording_data = cached_load_recording(
                            tuple(str(p) for p in summary.rr_paths),
                            tuple(str(p) for p in events_paths),
                            selected_participant
                        )
                    else:
                        # Fallback: re-discover recordings (for old cached summaries)
                        bundles = cached_discover_recordings(st.session_state.data_dir, st.session_state.id_pattern)
                        bundle = next(b for b in bundles if b.participant_id == selected_participant)
                        recording_data = cached_load_recording(
                            tuple(str(p) for p in bundle.rr_paths),
                            tuple(str(p) for p in bundle.events_paths),
                            selected_participant
                        )

                    # Initialize session state for event management (needed for plot)
                    if "participant_events" not in st.session_state:
                        st.session_state.participant_events = {}

                    # Store events in session state for this participant if not already there
                    if selected_participant not in st.session_state.participant_events:
                        # First check if we have saved events for this participant
                        from rrational.gui.persistence import load_participant_events
                        from rrational.prep.summaries import EventStatus
                        from datetime import datetime

                        saved_events = load_participant_events(selected_participant, st.session_state.data_dir)
                        if saved_events:
                            # Load from saved YAML - convert dicts back to EventStatus
                            def dict_to_event(d):
                                ts = d.get("first_timestamp")
                                if ts and isinstance(ts, str):
                                    ts = datetime.fromisoformat(ts)
                                last_ts = d.get("last_timestamp")
                                if last_ts and isinstance(last_ts, str):
                                    last_ts = datetime.fromisoformat(last_ts)
                                return EventStatus(
                                    raw_label=d.get("raw_label", ""),
                                    canonical=d.get("canonical"),
                                    first_timestamp=ts,
                                    last_timestamp=last_ts,
                                )

                            # Also load exclusion zones with datetime conversion
                            exclusion_zones = []
                            for zone in saved_events.get('exclusion_zones', []):
                                zone_copy = dict(zone)
                                # Convert ISO strings back to datetime
                                if zone_copy.get('start') and isinstance(zone_copy['start'], str):
                                    zone_copy['start'] = datetime.fromisoformat(zone_copy['start'])
                                if zone_copy.get('end') and isinstance(zone_copy['end'], str):
                                    zone_copy['end'] = datetime.fromisoformat(zone_copy['end'])
                                exclusion_zones.append(zone_copy)

                            st.session_state.participant_events[selected_participant] = {
                                'events': [dict_to_event(e) for e in saved_events.get('events', [])],
                                'manual': [dict_to_event(e) for e in saved_events.get('manual', [])],
                                'music_events': [dict_to_event(e) for e in saved_events.get('music_events', [])],
                                'exclusion_zones': exclusion_zones,
                            }
                        else:
                            # Load from original recording - use raw events, not grouped EventStatus
                            # Fix: Events with same label but different timestamps should be kept separate
                            from rrational.prep.summaries import EventStatus
                            raw_events = recording_data.get('events', [])
                            # Deduplicate by (timestamp, label) - keep unique combinations
                            seen = set()
                            unique_events = []
                            for label, ts in raw_events:
                                key = (ts.isoformat() if ts else '', label.strip().lower())
                                if key not in seen:
                                    seen.add(key)
                                    # Get canonical name from normalizer
                                    canonical = st.session_state.normalizer.normalize(label) if hasattr(st.session_state, 'normalizer') else None
                                    unique_events.append(EventStatus(
                                        raw_label=label,
                                        canonical=canonical,
                                        count=1,
                                        first_timestamp=ts,
                                        last_timestamp=ts,
                                    ))
                            st.session_state.participant_events[selected_participant] = {
                                'events': unique_events,
                                'manual': st.session_state.manual_events.get(selected_participant, []).copy(),
                                'music_events': [],
                                'exclusion_zones': [],
                            }

                    # Get cleaned RR intervals using CACHED function
                    config_dict = {
                        "rr_min_ms": st.session_state.cleaning_config.rr_min_ms,
                        "rr_max_ms": st.session_state.cleaning_config.rr_max_ms,
                        "sudden_change_pct": st.session_state.cleaning_config.sudden_change_pct
                    }
                    is_vns = (source_app == "VNS Analyse")
                    rr_with_timestamps, stats, _ = cached_clean_rr_intervals(
                        tuple(recording_data['rr_intervals']),
                        config_dict,
                        is_vns_data=is_vns
                    )

                    # Check plotly availability (triggers lazy import)
                    go, _ = get_plotly()
                    if go is None:
                        st.warning("Plotly is not installed. Please install it with: `pip install plotly streamlit-plotly-events`")
                    elif not rr_with_timestamps:
                        st.warning("No RR interval data available for visualization. The data may be empty or all intervals were filtered out.")
                    else:
                        # Unpack cached data - VNS has 3 elements (with flag), HRV Logger has 2
                        if is_vns:
                            timestamps, rr_values, flags = zip(*rr_with_timestamps)
                        else:
                            # Issue #11 fix: Use RAW data for visualization (not cleaned)
                            # Cleaning cascade can remove large portions of data after artifacts,
                            # but users need to see ALL data to understand measurement restarts
                            raw_rr_data = recording_data['rr_intervals']
                            timestamps = tuple(ts for ts, rr, _ in raw_rr_data if ts is not None)
                            rr_values = tuple(rr for ts, rr, _ in raw_rr_data if ts is not None)
                            flags = None

                        # Get plot resolution from session state (use saved settings as default)
                        resolution_key = f"plot_resolution_{selected_participant}"
                        saved_resolution = st.session_state.get("app_settings", {}).get("plot_resolution", 5000)
                        plot_resolution = st.session_state.get(resolution_key, saved_resolution)

                        # Get CACHED plot data and store in session state for fragment
                        plot_data = cached_get_plot_data(
                            tuple(timestamps),
                            tuple(rr_values),
                            selected_participant,
                            downsample_threshold=plot_resolution,
                            flags_tuple=tuple(flags) if flags else None
                        )
                        # Add source_app for gap detection logic
                        plot_data = dict(plot_data)  # Make mutable copy
                        plot_data['source_app'] = source_app
                        st.session_state[f"plot_data_{selected_participant}"] = plot_data

                        # Store FULL (non-downsampled) data for section validation
                        st.session_state[f"full_rr_data_{selected_participant}"] = {
                            'timestamps': list(timestamps),
                            'rr_values': list(rr_values),
                        }

                        # Mode selector for plot interaction (Events, Exclusions, or Signal Inspection)
                        st.markdown("---")
                        col_mode1, col_mode2, col_mode3 = st.columns([2, 1, 1])
                        with col_mode1:
                            interaction_mode = st.radio(
                                "Plot interaction",
                                ["Add Events", "Add Exclusions", "Signal Inspection"],
                                key=f"plot_mode_{selected_participant}",
                                horizontal=True,
                                label_visibility="collapsed"
                            )
                        with col_mode2:
                            # Signal Inspection mode info (controls moved to fragment for speed)
                            if interaction_mode == "Signal Inspection":
                                st.caption("Use I, ←, → keys")

                        with col_mode3:
                            # Plot resolution slider - allow up to all points
                            n_total = plot_data['n_original']
                            # Only show slider if dataset is large enough to benefit from downsampling
                            if n_total > 1000:
                                max_points = n_total  # Allow showing all points
                                # Signal Inspection mode: force max resolution for beat-level inspection
                                if interaction_mode == "Signal Inspection":
                                    # Force session state to max resolution
                                    st.session_state[resolution_key] = n_total
                                    default_points = n_total
                                elif n_total <= saved_resolution:
                                    default_points = n_total
                                else:
                                    default_points = saved_resolution
                                st.slider(
                                    "Plot resolution",
                                    min_value=1000,
                                    max_value=max_points,
                                    value=min(default_points, max_points),
                                    step=1000,
                                    key=resolution_key,
                                    help=f"Number of points to display ({n_total:,} total). Higher = more detail but slower."
                                )
                            else:
                                st.caption(f"Showing all {n_total:,} points")

                        # Render plot using fragment (click handling is inside the fragment)
                        render_rr_plot_fragment(selected_participant)

                except Exception as e:
                    st.warning(f"Could not generate RR plot: {e}")

                # ================== EXCLUSION ZONES (shown when in exclusion mode) ==================
                if interaction_mode == "Add Exclusions":
                    col_excl_title, col_excl_help = st.columns([4, 1])
                    with col_excl_title:
                        st.markdown("### Exclusion Zones")
                    with col_excl_help:
                        with st.popover("Help"):
                            from rrational.gui.help_text import EXCLUSION_ZONES_HELP
                            st.markdown(EXCLUSION_ZONES_HELP)

                    # Set exclusion method (click two points only)
                    st.session_state[f"exclusion_method_{selected_participant}"] = "Click two points on plot"

                    # Clear selection button - always visible when there are pending clicks
                    click_key = f"exclusion_clicks_{selected_participant}"
                    pending_clicks = st.session_state.get(click_key, [])

                    col_info, col_clear = st.columns([4, 1])
                    with col_info:
                        if len(pending_clicks) == 0:
                            st.caption("Click on the plot to set the **start point** of an exclusion zone.")
                        elif len(pending_clicks) == 1:
                            st.caption(f"Start: **{pending_clicks[0].strftime('%H:%M:%S')}** — Click to set **end point**.")
                    with col_clear:
                        if pending_clicks:
                            if st.button("X Clear", key=f"clear_selection_{selected_participant}", type="secondary"):
                                # Clear the pending clicks list, but keep last_click_key
                                # so the same click won't be re-added on rerun
                                st.session_state[click_key] = []
                                st.toast("Selection cleared")
                                st.rerun()

                    # Initialize exclusion zones in session state if needed
                    if 'exclusion_zones' not in st.session_state.participant_events.get(selected_participant, {}):
                        if selected_participant not in st.session_state.participant_events:
                            st.session_state.participant_events[selected_participant] = {'events': [], 'manual': [], 'exclusion_zones': []}
                        else:
                            st.session_state.participant_events[selected_participant]['exclusion_zones'] = []

                    exclusion_zones = st.session_state.participant_events[selected_participant].get('exclusion_zones', [])

                    # Check for pending click points (from click-two-points mode)
                    click_key = f"exclusion_clicks_{selected_participant}"
                    if click_key in st.session_state and len(st.session_state[click_key]) >= 2:
                        clicks = st.session_state[click_key]
                        start_click, end_click = sorted(clicks[:2])
                        st.success("Selected zone - adjust times below if needed:")

                        # Editable time inputs for the selected zone (HH:MM:SS format)
                        col_start, col_end = st.columns(2)
                        with col_start:
                            edited_start_str = st.text_input(
                                "Start time (HH:MM:SS)",
                                value=start_click.strftime("%H:%M:%S"),
                                key=f"excl_start_time_{selected_participant}",
                                help="Edit the start time in HH:MM:SS format"
                            )
                        with col_end:
                            edited_end_str = st.text_input(
                                "End time (HH:MM:SS)",
                                value=end_click.strftime("%H:%M:%S"),
                                key=f"excl_end_time_{selected_participant}",
                                help="Edit the end time in HH:MM:SS format"
                            )

                        # Parse edited times and combine with original date
                        import datetime
                        try:
                            edited_start_time = datetime.datetime.strptime(edited_start_str, "%H:%M:%S").time()
                        except ValueError:
                            st.error("Invalid start time format. Use HH:MM:SS")
                            edited_start_time = start_click.time()
                        try:
                            edited_end_time = datetime.datetime.strptime(edited_end_str, "%H:%M:%S").time()
                        except ValueError:
                            st.error("Invalid end time format. Use HH:MM:SS")
                            edited_end_time = end_click.time()

                        final_start = datetime.datetime.combine(start_click.date(), edited_start_time)
                        final_end = datetime.datetime.combine(end_click.date(), edited_end_time)
                        if final_start.tzinfo is None and start_click.tzinfo is not None:
                            final_start = final_start.replace(tzinfo=start_click.tzinfo)
                        if final_end.tzinfo is None and end_click.tzinfo is not None:
                            final_end = final_end.replace(tzinfo=end_click.tzinfo)

                        col_form1, col_form2 = st.columns(2)
                        with col_form1:
                            reason_click = st.text_input(
                                "Reason (optional)",
                                key=f"excl_reason_click_{selected_participant}",
                                placeholder="e.g., Bathroom break"
                            )
                        with col_form2:
                            exclude_dur_click = st.checkbox(
                                "Exclude from duration",
                                value=True,
                                key=f"excl_dur_click_{selected_participant}"
                            )

                        col_confirm, col_cancel = st.columns(2)
                        last_click_key = f"last_click_{selected_participant}"
                        with col_confirm:
                            if st.button("Add Exclusion Zone", key=f"confirm_excl_{selected_participant}", type="primary"):
                                new_zone = {
                                    'start': final_start,
                                    'end': final_end,
                                    'reason': reason_click,
                                    'exclude_from_duration': exclude_dur_click
                                }
                                st.session_state.participant_events[selected_participant]['exclusion_zones'].append(new_zone)
                                st.session_state[click_key] = []
                                # Clear last click to allow new selections
                                if last_click_key in st.session_state:
                                    del st.session_state[last_click_key]
                                show_toast("Exclusion zone added", icon="success")
                                st.rerun()
                        with col_cancel:
                            if st.button("Cancel", key=f"cancel_excl_{selected_participant}"):
                                st.session_state[click_key] = []
                                # Clear last click to allow new selections
                                if last_click_key in st.session_state:
                                    del st.session_state[last_click_key]
                                st.rerun()
                    elif click_key in st.session_state and len(st.session_state[click_key]) == 1:
                        st.warning(f"Start point set: **{st.session_state[click_key][0].strftime('%H:%M:%S')}** - Now click on plot to set **end point**")
                        if st.button("Cancel", key=f"cancel_click1_{selected_participant}"):
                            st.session_state[click_key] = []
                            last_click_key = f"last_click_{selected_participant}"
                            if last_click_key in st.session_state:
                                del st.session_state[last_click_key]
                            st.rerun()

                    # Display existing exclusion zones
                    if exclusion_zones:
                        col_zones_header, col_save = st.columns([3, 1])
                        with col_zones_header:
                            st.markdown("**Current Exclusion Zones:**")
                        with col_save:
                            if st.button("Save", key=f"save_exclusions_{selected_participant}", type="primary", help="Save exclusion zones to disk"):
                                from rrational.gui.persistence import save_participant_events
                                save_participant_events(selected_participant, st.session_state.participant_events[selected_participant], st.session_state.data_dir)
                                show_toast("Exclusion zones saved", icon="success")

                        for idx, zone in enumerate(exclusion_zones):
                            zone_start = zone.get('start', 'N/A')
                            zone_end = zone.get('end', 'N/A')
                            zone_reason = zone.get('reason', '')
                            exclude_duration = zone.get('exclude_from_duration', True)

                            # Format timestamps for display
                            if hasattr(zone_start, 'strftime'):
                                start_str = zone_start.strftime('%H:%M:%S')
                            elif isinstance(zone_start, str):
                                start_str = zone_start[:19]
                            else:
                                start_str = 'N/A'

                            if hasattr(zone_end, 'strftime'):
                                end_str = zone_end.strftime('%H:%M:%S')
                            elif isinstance(zone_end, str):
                                end_str = zone_end[:19]
                            else:
                                end_str = 'N/A'

                            col_zone, col_edit, col_del = st.columns([4, 1, 1])
                            with col_zone:
                                duration_icon = "[excl]" if exclude_duration else ""
                                reason_text = f" - {zone_reason}" if zone_reason else ""
                                st.write(f"{idx+1}. **{start_str}** → **{end_str}** {duration_icon}{reason_text}")
                            with col_edit:
                                edit_key = f"edit_zone_{selected_participant}_{idx}"
                                if st.button("Edit", key=f"btn_{edit_key}"):
                                    st.session_state[edit_key] = not st.session_state.get(edit_key, False)
                                    st.rerun()
                            with col_del:
                                if st.button("X", key=f"del_zone_{selected_participant}_{idx}"):
                                    exclusion_zones.pop(idx)
                                    st.rerun()

                            # Editable form for this zone
                            edit_key = f"edit_zone_{selected_participant}_{idx}"
                            if st.session_state.get(edit_key, False):
                                with st.container():
                                    st.markdown("---")
                                    import datetime
                                    col_e1, col_e2 = st.columns(2)
                                    with col_e1:
                                        new_start = st.text_input(
                                            "Start (HH:MM:SS)",
                                            value=start_str,
                                            key=f"edit_start_{selected_participant}_{idx}"
                                        )
                                    with col_e2:
                                        new_end = st.text_input(
                                            "End (HH:MM:SS)",
                                            value=end_str,
                                            key=f"edit_end_{selected_participant}_{idx}"
                                        )
                                    col_e3, col_e4 = st.columns(2)
                                    with col_e3:
                                        new_reason = st.text_input(
                                            "Reason",
                                            value=zone_reason,
                                            key=f"edit_reason_{selected_participant}_{idx}"
                                        )
                                    with col_e4:
                                        new_exclude_dur = st.checkbox(
                                            "Exclude from duration",
                                            value=exclude_duration,
                                            key=f"edit_excl_dur_{selected_participant}_{idx}"
                                        )
                                    col_save_edit, col_cancel_edit = st.columns(2)
                                    with col_save_edit:
                                        if st.button("Save Changes", key=f"save_edit_{selected_participant}_{idx}"):
                                            try:
                                                # Parse new times
                                                new_start_time = datetime.datetime.strptime(new_start, "%H:%M:%S").time()
                                                new_end_time = datetime.datetime.strptime(new_end, "%H:%M:%S").time()
                                                # Use original date
                                                orig_date = zone_start.date() if hasattr(zone_start, 'date') else datetime.date.today()
                                                new_start_dt = datetime.datetime.combine(orig_date, new_start_time)
                                                new_end_dt = datetime.datetime.combine(orig_date, new_end_time)
                                                # Preserve timezone if present
                                                if hasattr(zone_start, 'tzinfo') and zone_start.tzinfo:
                                                    new_start_dt = new_start_dt.replace(tzinfo=zone_start.tzinfo)
                                                    new_end_dt = new_end_dt.replace(tzinfo=zone_start.tzinfo)
                                                # Update zone
                                                zone['start'] = new_start_dt
                                                zone['end'] = new_end_dt
                                                zone['reason'] = new_reason
                                                zone['exclude_from_duration'] = new_exclude_dur
                                                st.session_state[edit_key] = False
                                                st.toast("Zone updated")
                                                st.rerun()
                                            except ValueError:
                                                st.error("Invalid time format. Use HH:MM:SS")
                                    with col_cancel_edit:
                                        if st.button("Cancel", key=f"cancel_edit_{selected_participant}_{idx}"):
                                            st.session_state[edit_key] = False
                                            st.rerun()
                                    st.markdown("---")
                    else:
                        st.info("No exclusion zones defined yet.")

                    st.markdown("---")
                    with st.expander("Manual Entry", expanded=False):
                        # Get first RR timestamp as reference
                        first_rr_time = None
                        if 'rr_intervals' in recording_data and recording_data['rr_intervals']:
                            first_rr_time = recording_data['rr_intervals'][0][0]

                        col_start, col_end = st.columns(2)
                        with col_start:
                            manual_start = st.text_input(
                                "Start time (HH:MM:SS)",
                                value=first_rr_time.strftime("%H:%M:%S") if first_rr_time and hasattr(first_rr_time, 'strftime') else "10:00:00",
                                key=f"manual_excl_start_{selected_participant}",
                                placeholder="HH:MM:SS"
                            )
                        with col_end:
                            manual_end = st.text_input(
                                "End time (HH:MM:SS)",
                                value="",
                                key=f"manual_excl_end_{selected_participant}",
                                placeholder="HH:MM:SS"
                            )

                        col_r, col_d = st.columns(2)
                        with col_r:
                            manual_reason = st.text_input(
                                "Reason (optional)",
                                key=f"manual_excl_reason_{selected_participant}",
                                placeholder="e.g., Extra break"
                            )
                        with col_d:
                            manual_exclude_dur = st.checkbox(
                                "Exclude from duration",
                                value=True,
                                key=f"manual_excl_dur_{selected_participant}"
                            )

                        def add_manual_exclusion():
                            import datetime as dt
                            try:
                                parts = manual_start.strip().split(":")
                                start_time = dt.time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)
                                parts = manual_end.strip().split(":")
                                end_time = dt.time(int(parts[0]), int(parts[1]), int(parts[2]) if len(parts) > 2 else 0)

                                if first_rr_time:
                                    start_dt = first_rr_time.replace(hour=start_time.hour, minute=start_time.minute, second=start_time.second)
                                    end_dt = first_rr_time.replace(hour=end_time.hour, minute=end_time.minute, second=end_time.second)
                                    new_zone = {
                                        'start': start_dt,
                                        'end': end_dt,
                                        'reason': manual_reason,
                                        'exclude_from_duration': manual_exclude_dur
                                    }
                                    st.session_state.participant_events[selected_participant]['exclusion_zones'].append(new_zone)
                                    st.toast("Exclusion zone added")
                            except (ValueError, IndexError):
                                st.error("Invalid time format. Use HH:MM:SS")

                        st.button("+ Add Exclusion Zone", key=f"add_manual_excl_{selected_participant}", on_click=add_manual_exclusion)

                # ================== SIGNAL INSPECTION (artifact correction mode) ==================
                if interaction_mode == "Signal Inspection":
                    col_sig_title, col_sig_help = st.columns([4, 1])
                    with col_sig_title:
                        st.markdown("### Signal Inspection")
                    with col_sig_help:
                        with st.popover("Help"):
                            st.markdown(ARTIFACT_CORRECTION_HELP)

                    # Section filter - focus on specific defined sections
                    st.markdown("##### Focus on Section")
                    sections = st.session_state.get("sections", {})
                    section_options = ["Full recording"] + list(sections.keys())
                    selected_section = st.selectbox(
                        "View section",
                        options=section_options,
                        index=0,
                        key=f"signal_inspection_section_{selected_participant}",
                        help="Focus the plot on a specific section for detailed inspection"
                    )

                    # Show section time range info if a section is selected
                    if selected_section != "Full recording" and selected_section in sections:
                        section_def = sections[selected_section]
                        participant_data = st.session_state.participant_events.get(selected_participant, {})

                        # Build event timestamp lookup from stored events
                        all_evts = participant_data.get('events', []) + participant_data.get('manual', [])
                        event_timestamps = {}
                        for evt in all_evts:
                            if hasattr(evt, 'canonical') and evt.canonical and hasattr(evt, 'first_timestamp') and evt.first_timestamp:
                                event_timestamps[evt.canonical] = evt.first_timestamp

                        start_event = section_def.get("start_event", "")
                        end_events = section_def.get("end_events", [])

                        start_time = event_timestamps.get(start_event) if start_event else None
                        end_time = None
                        for end_ev in end_events:
                            end_time = event_timestamps.get(end_ev)
                            if end_time:
                                break

                        if start_time and end_time:
                            # Calculate section stats
                            section_start_str = start_time.strftime('%H:%M:%S') if hasattr(start_time, 'strftime') else str(start_time)
                            section_end_str = end_time.strftime('%H:%M:%S') if hasattr(end_time, 'strftime') else str(end_time)
                            section_duration = (end_time - start_time).total_seconds() if hasattr(end_time, 'total_seconds') or hasattr(start_time, '__sub__') else 0

                            # Store for plot fragment to use
                            st.session_state[f"inspection_section_range_{selected_participant}"] = (start_time, end_time)

                            col_sec1, col_sec2, col_sec3 = st.columns(3)
                            with col_sec1:
                                st.caption(f"Start: {section_start_str}")
                            with col_sec2:
                                st.caption(f"End: {section_end_str}")
                            with col_sec3:
                                if section_duration:
                                    mins = int(section_duration // 60)
                                    secs = int(section_duration % 60)
                                    st.caption(f"Duration: {mins}:{secs:02d}")
                        else:
                            st.warning(f"Section '{selected_section}' events not found in this participant's data")
                            st.session_state[f"inspection_section_range_{selected_participant}"] = None
                    else:
                        # Clear section range when "Full recording" selected
                        st.session_state[f"inspection_section_range_{selected_participant}"] = None

                    st.markdown("---")

                    # Get artifact detection results from session state (computed in plot fragment)
                    artifact_key = f"artifacts_{selected_participant}"
                    artifact_result = st.session_state.get(artifact_key, {})

                    if artifact_result:
                        total_artifacts = artifact_result.get('total_artifacts', 0)
                        artifact_ratio = artifact_result.get('artifact_ratio', 0.0)
                        by_type = artifact_result.get('by_type', {})

                        # Include manual artifacts in total count
                        manual_artifact_key_stats = f"manual_artifacts_{selected_participant}"
                        manual_count = len(st.session_state.get(manual_artifact_key_stats, []))
                        total_with_manual = total_artifacts + manual_count
                        n_beats = len(rr_values) if rr_values else 1
                        artifact_ratio_with_manual = total_with_manual / n_beats if n_beats > 0 else 0

                        # Artifact statistics
                        col_a1, col_a2, col_a3 = st.columns(3)
                        with col_a1:
                            # Quality badge based on artifact percentage (including manual)
                            if artifact_ratio_with_manual < 0.02:
                                badge = "[OK]"
                            elif artifact_ratio_with_manual < 0.05:
                                badge = "[!]"
                            elif artifact_ratio_with_manual < 0.10:
                                badge = "[!!]"
                            else:
                                badge = "[X]"
                            display_count = f"{total_artifacts}" if manual_count == 0 else f"{total_artifacts}+{manual_count}"
                            st.metric("Artifacts Detected", f"{badge} {display_count}")
                        with col_a2:
                            st.metric("Artifact Rate", f"{artifact_ratio_with_manual*100:.2f}%")
                        with col_a3:
                            # Show artifact types breakdown
                            if by_type or manual_count > 0:
                                type_parts = [f"{k}: {v}" for k, v in by_type.items()]
                                if manual_count > 0:
                                    type_parts.append(f"manual: {manual_count}")
                                type_str = ", ".join(type_parts)
                                st.metric("Types", type_str[:40])  # Truncate if too long

                        # Quality recommendations (uses combined artifact ratio)
                        st.markdown("##### Artifact Quality Assessment")
                        if artifact_ratio_with_manual < 0.02:
                            st.success("**Excellent quality** - Less than 2% artifacts. Data is suitable for all HRV analyses.")
                        elif artifact_ratio_with_manual < 0.05:
                            st.info("**Good quality** - 2-5% artifacts. Suitable for most HRV analyses with correction.")
                        elif artifact_ratio_with_manual < 0.10:
                            st.warning("**Acceptable quality** - 5-10% artifacts. Consider excluding high-artifact segments or using robust methods.")
                        else:
                            st.error("**Poor quality** - >10% artifacts. Consider excluding this recording or identifying problematic sections.")

                        # Artifact details table
                        st.markdown("---")
                        st.markdown("##### Artifact Details")
                        artifact_indices = artifact_result.get('artifact_indices', [])
                        artifact_timestamps = artifact_result.get('artifact_timestamps', [])
                        artifact_rr = artifact_result.get('artifact_rr', [])
                        detection_method = artifact_result.get('method', 'unknown')

                        if artifact_indices and len(artifact_indices) > 0:
                            # Show first 20 artifacts in a table
                            with st.expander(f"View Artifact List ({len(artifact_indices)} artifacts)", expanded=False):
                                artifact_data = []
                                for i, (idx, ts, rr) in enumerate(zip(
                                    artifact_indices[:50],
                                    artifact_timestamps[:50] if artifact_timestamps else [None]*50,
                                    artifact_rr[:50] if artifact_rr else [None]*50
                                )):
                                    ts_str = ts.strftime('%H:%M:%S') if ts and hasattr(ts, 'strftime') else str(ts)[:8] if ts else "N/A"
                                    artifact_data.append({
                                        "#": i + 1,
                                        "Index": idx,
                                        "Time": ts_str,
                                        "RR (ms)": f"{rr:.0f}" if rr else "N/A",
                                        "Method": detection_method,
                                    })
                                if artifact_data:
                                    st.dataframe(get_pandas().DataFrame(artifact_data), hide_index=True, width="stretch")
                                if len(artifact_indices) > 50:
                                    st.caption(f"Showing first 50 of {len(artifact_indices)} artifacts")

                        # Manual artifact marking section
                        st.markdown("---")
                        st.markdown("##### Manual Artifact Marking")
                        st.caption("Click on a beat in the plot to toggle manual artifact marking (purple diamonds)")

                        # Manual artifacts session state keys (loaded automatically before plot)
                        manual_artifact_key = f"manual_artifacts_{selected_participant}"

                        if manual_artifact_key not in st.session_state:
                            st.session_state[manual_artifact_key] = []

                        manual_artifacts = st.session_state[manual_artifact_key]
                        if manual_artifacts:
                            col_man1, col_man2 = st.columns([3, 1])
                            with col_man1:
                                st.write(f"**{len(manual_artifacts)} manually marked artifacts** (purple diamonds)")
                            with col_man2:
                                if st.button("Clear all", key=f"clear_manual_art_{selected_participant}", type="secondary"):
                                    st.session_state[manual_artifact_key] = []
                                    st.rerun()

                            # Show table of manual artifacts
                            with st.expander(f"View Manual Artifacts ({len(manual_artifacts)})", expanded=False):
                                manual_data = []
                                for i, art in enumerate(manual_artifacts):
                                    manual_data.append({
                                        "#": i + 1,
                                        "Time": art.get('timestamp', 'N/A')[:8],
                                        "RR (ms)": f"{art.get('rr_value', 0):.0f}",
                                        "Index": art.get('original_idx', 'N/A'),
                                    })
                                st.dataframe(get_pandas().DataFrame(manual_data), hide_index=True, width="stretch")
                                st.caption("Click on marked beats in the plot to remove them")
                        else:
                            st.info("Click on beats in the plot to mark them as artifacts")

                        # Save corrected data section
                        st.markdown("---")
                        st.markdown("##### Export Corrected Data")
                        
                        # Get current plot data and artifact info
                        plot_data_export = st.session_state.get(f"plot_data_{selected_participant}", {})
                        artifact_result_export = st.session_state.get(f"artifacts_{selected_participant}", {})
                        
                        if plot_data_export and 'timestamps' in plot_data_export:
                            ts_export = plot_data_export['timestamps']
                            rr_export = plot_data_export['rr_values']
                            
                            # Collect all artifact indices (algorithm + manual)
                            algo_indices = set(artifact_result_export.get('artifact_indices', []))
                            manual_indices = set(art.get('plot_idx', -1) for art in manual_artifacts)
                            all_artifact_idx = algo_indices | manual_indices

                            # Use corrected RR from NeuroKit2's signal_fixpeaks (stored in artifact_result)
                            corrected_rr = artifact_result_export.get('corrected_rr', rr_export)
                            if corrected_rr is None or len(corrected_rr) != len(rr_export):
                                corrected_rr = rr_export
                            
                            # Create DataFrame for export
                            import io
                            export_data = []
                            for i, (ts, rr_orig, rr_corr) in enumerate(zip(ts_export, rr_export, corrected_rr)):
                                ts_str = ts.strftime('%Y-%m-%d %H:%M:%S.%f') if hasattr(ts, 'strftime') else str(ts)
                                is_artifact = i in all_artifact_idx
                                export_data.append({
                                    'timestamp': ts_str,
                                    'rr_ms': rr_orig,
                                    'nn_ms': round(rr_corr, 1),
                                    'is_artifact': is_artifact,
                                    'artifact_source': 'algorithm' if i in algo_indices else ('manual' if i in manual_indices else '')
                                })
                            
                            df_export = get_pandas().DataFrame(export_data)
                            csv_buffer = io.StringIO()
                            df_export.to_csv(csv_buffer, index=False)
                            csv_data = csv_buffer.getvalue()
                            
                            col_exp1, col_exp2 = st.columns([2, 1])
                            with col_exp1:
                                st.caption(f"{len(ts_export)} beats | {len(all_artifact_idx)} artifacts corrected")
                            with col_exp2:
                                st.download_button(
                                    "Download CSV",
                                    data=csv_data,
                                    file_name=f"{selected_participant}_corrected.csv",
                                    mime="text/csv",
                                    key=f"download_corrected_{selected_participant}",
                                    type="primary"
                                )
                        else:
                            st.caption("Load participant data to enable export")

                        # Artifact Corrections status section
                        st.markdown("---")
                        st.markdown("##### Artifact Status")

                        # Get exclusion data
                        artifact_exclusions_key = f"artifact_exclusions_{selected_participant}"
                        artifact_exclusions = st.session_state.get(artifact_exclusions_key, set())

                        # Get algorithm artifact data
                        artifact_data_insp_check = st.session_state.get(f"artifacts_{selected_participant}", {})
                        n_algo_insp = len(artifact_data_insp_check.get('artifact_indices', []))

                        # Count corrections
                        n_manual = len(manual_artifacts)
                        n_excluded = len(artifact_exclusions) if artifact_exclusions else 0
                        has_corrections = n_manual > 0 or n_excluded > 0 or n_algo_insp > 0

                        # Check if saved corrections exist
                        from rrational.gui.persistence import load_artifact_corrections
                        saved_corrections = load_artifact_corrections(
                            selected_participant,
                            data_dir=str(st.session_state.get("data_dir", "")),
                            project_path=st.session_state.get("current_project"),
                        )
                        has_saved = saved_corrections is not None

                        if has_corrections:
                            # Build summary text
                            corr_parts = []
                            if n_algo_insp > 0:
                                corr_parts.append(f"**{n_algo_insp}** algorithm-detected")
                            if n_manual > 0:
                                corr_parts.append(f"**{n_manual}** manually marked")
                            if n_excluded > 0:
                                corr_parts.append(f"**{n_excluded}** excluded")
                            st.write(" | ".join(corr_parts))

                            if has_saved:
                                st.success("Saved ✓")
                            else:
                                st.caption("Use **Save Artifact Corrections** in sidebar to save")

                            # Reset buttons
                            col_reset1, col_reset2 = st.columns(2)
                            with col_reset1:
                                # Reset to Original - keeps algorithm artifacts, clears manual changes
                                if n_manual > 0 or n_excluded > 0:
                                    if st.button("Reset to Original", key=f"reset_to_original_{selected_participant}",
                                                help="Clear manual markings and exclusions, keep algorithm detection"):
                                        st.session_state[manual_artifact_key] = []
                                        st.session_state[artifact_exclusions_key] = set()
                                        st.toast("Reset to original algorithm detection")
                                        st.rerun()
                            with col_reset2:
                                # Reset All - clears everything and deletes saved file
                                if st.button("Reset All", key=f"reset_artifacts_{selected_participant}", type="secondary",
                                            help="Clear ALL markings and delete saved file"):
                                    # Clear session state
                                    st.session_state[manual_artifact_key] = []
                                    st.session_state[artifact_exclusions_key] = set()
                                    # Clear loaded flag so fresh load can happen
                                    artifacts_loaded_key = f"artifacts_loaded_{selected_participant}"
                                    if artifacts_loaded_key in st.session_state:
                                        del st.session_state[artifacts_loaded_key]
                                    # Clear loaded info
                                    loaded_info_key = f"artifacts_loaded_info_{selected_participant}"
                                    if loaded_info_key in st.session_state:
                                        del st.session_state[loaded_info_key]
                                    # Clear rrational source indicator
                                    if f"artifacts_from_rrational_{selected_participant}" in st.session_state:
                                        del st.session_state[f"artifacts_from_rrational_{selected_participant}"]
                                    # Delete saved file
                                    from rrational.gui.persistence import delete_artifact_corrections
                                    delete_artifact_corrections(
                                        selected_participant,
                                        data_dir=str(st.session_state.get("data_dir", "")),
                                        project_path=st.session_state.get("current_project"),
                                    )
                                    st.toast("Reset all artifact corrections")
                                    st.rerun()
                        else:
                            st.caption("Click on beats in the plot to mark/unmark artifacts")

                        # Instructions
                        st.markdown("---")
                        st.markdown("##### Tips")
                        st.markdown("""
                        - **Orange X markers** show auto-detected artifacts
                        - **Purple diamonds** show manually marked artifacts
                        - **Green dotted line** shows corrected signal (enable "Show corrected (NN)")
                        - Adjust **threshold %** (higher = less sensitive) if too many false positives
                        - Try **kubios_segmented** method for long recordings
                        - Use **Add Exclusions** mode to exclude problematic regions
                        """)
                    else:
                        st.info("Enable 'Show artifacts' in plot options to see artifact detection results.")

                # ================== SIGNAL QUALITY & EVENTS (only in events mode) ==================
                if interaction_mode == "Add Events":
                    # Show quality analysis info if available
                    changepoint_key = f"changepoints_{selected_participant}"
                    gap_key = f"gaps_{selected_participant}"

                    # Time Gap Analysis Expander
                    if gap_key in st.session_state:
                        gap_info = st.session_state[gap_key]
                        # Determine badge for expander title
                        if gap_info.get('vns_note'):
                            gap_title = "Time Gap Analysis (N/A for VNS data)"
                        elif gap_info['total_gaps'] == 0:
                            gap_title = "Time Gap Analysis (No gaps)"
                        else:
                            gap_badge = "[!]" if gap_info['total_gaps'] <= 2 else "[X]"
                            gap_title = f"Time Gap Analysis ({gap_badge} {gap_info['total_gaps']} gaps)"

                        with st.expander(gap_title, expanded=False):
                            st.caption("Identifies time gaps >2 seconds between consecutive beats (recording interruptions, Bluetooth disconnections, device errors)")

                            # Check if VNS data (gap detection not applicable)
                            if gap_info.get('vns_note'):
                                st.info(
                                    "**Gap detection not applicable for VNS data.** "
                                    "VNS files only contain RR intervals without real timestamps. "
                                    "Timestamps are synthesized from cumulative RR values, so time gaps cannot be detected."
                                )
                            else:
                                col_g1, col_g2, col_g3 = st.columns(3)
                                with col_g1:
                                    gap_badge = "[OK]" if gap_info['total_gaps'] == 0 else ("[!]" if gap_info['total_gaps'] <= 2 else "[X]")
                                    st.metric("Gaps Detected", f"{gap_badge} {gap_info['total_gaps']}")
                                with col_g2:
                                    st.metric("Total Gap Time", f"{gap_info['total_gap_duration_s']:.1f}s")
                                with col_g3:
                                    st.metric("Gap Ratio", f"{gap_info['gap_ratio']*100:.2f}%")

                                if gap_info['gaps']:
                                    st.markdown("**Gap Details:**")
                                    gap_data = []
                                    for i, gap in enumerate(gap_info['gaps']):
                                        start_str = gap['start_time'].strftime('%H:%M:%S') if gap.get('start_time') else "?"
                                        end_str = gap['end_time'].strftime('%H:%M:%S') if gap.get('end_time') else "?"
                                        gap_data.append({
                                            "Gap #": i + 1,
                                            "Start Time": start_str,
                                            "End Time": end_str,
                                            "Duration (s)": f"{gap['duration_s']:.1f}",
                                            "Beat Index": f"{gap['start_idx']} → {gap['end_idx']}"
                                        })
                                    st.dataframe(get_pandas().DataFrame(gap_data), width='stretch', hide_index=True)

                                    # Recommendations for gaps
                                    st.markdown("##### Recommendations for Gaps:")
                                    st.markdown("""
                                    **What gaps mean:**
                                    - Recording was interrupted (Bluetooth disconnect, device error, or intentional pause)
                                    - Data during the gap is lost and cannot be recovered

                                    **What to do:**
                                    1. **Add boundary events** - Click on the plot at the gap start/end times to mark `gap_start` and `gap_end` events
                                    2. **Define sections around gaps** - In the Sections tab, create sections that exclude gap periods
                                    3. **In Analysis** - Select only valid sections for HRV computation (gaps will be automatically excluded if you use section boundaries)

                                    **When to exclude entire recording:**
                                    - If gaps occur during critical measurement periods (e.g., during music listening)
                                    - If total gap time exceeds 10% of recording duration
                                    """)

                                    # Auto-create gap events button
                                    st.markdown("##### Auto-Create Gap Events")
                                    col_gap_btn1, col_gap_btn2 = st.columns([1, 2])
                                    with col_gap_btn1:
                                        if st.button("Create Gap Events", key=f"auto_gap_{selected_participant}"):
                                            from rrational.prep.summaries import EventStatus
                                            events_added = 0
                                            for gap in gap_info['gaps']:
                                                # Create gap_start event
                                                if gap.get('start_time'):
                                                    gap_start_event = EventStatus(
                                                        raw_label="gap_start",
                                                        canonical="gap_start",
                                                        first_timestamp=gap['start_time'],
                                                        last_timestamp=gap['start_time']
                                                    )
                                                    st.session_state.participant_events[selected_participant]['manual'].append(gap_start_event)
                                                    events_added += 1

                                                # Create gap_end event
                                                if gap.get('end_time'):
                                                    gap_end_event = EventStatus(
                                                        raw_label="gap_end",
                                                        canonical="gap_end",
                                                        first_timestamp=gap['end_time'],
                                                        last_timestamp=gap['end_time']
                                                    )
                                                    st.session_state.participant_events[selected_participant]['manual'].append(gap_end_event)
                                                    events_added += 1

                                            show_toast(f"Created {events_added} gap boundary events", icon="success")
                                            st.rerun()
                                    with col_gap_btn2:
                                        st.caption("Creates `gap_start` and `gap_end` events for each detected gap. Use these to exclude gap periods from analysis.")
                                else:
                                    st.success("No time gaps detected - recording appears continuous")

                    # Variability Changepoint Analysis Expander
                    if changepoint_key in st.session_state:
                        cp_info = st.session_state[changepoint_key]
                        # Determine badge for expander title
                        high_var_count = sum(1 for s in cp_info.get('segment_stats', []) if s['cv'] > 0.15)
                        if high_var_count == 0:
                            var_title = f"Variability Analysis (Score: {cp_info['quality_score']}/100)"
                        else:
                            var_title = f"Variability Analysis ({high_var_count} high-CV segments)"

                        with st.expander(var_title, expanded=False):
                            st.caption("Uses NeuroKit2's signal_changepoints() with PELT algorithm to detect variance changes (movement artifacts, electrode issues, physiological changes)")

                            col_q1, col_q2, col_q3 = st.columns(3)
                            with col_q1:
                                st.metric("Quality Score", f"{cp_info['quality_score']}/100")
                            with col_q2:
                                st.metric("Segments Detected", cp_info['n_segments'])
                            with col_q3:
                                st.metric("Changepoints", len(cp_info['changepoint_indices']))

                            if cp_info['segment_stats']:
                                st.markdown("**Segment Details:**")
                                seg_data = []
                                for i, seg in enumerate(cp_info['segment_stats']):
                                    cv_pct = seg['cv'] * 100
                                    quality = "[OK] Good" if cv_pct < 10 else ("Moderate" if cv_pct < 15 else "[X] High")

                                    # Format timestamps
                                    start_str = seg.get('start_time').strftime('%H:%M:%S') if seg.get('start_time') else "?"
                                    end_str = seg.get('end_time').strftime('%H:%M:%S') if seg.get('end_time') else "?"

                                    seg_data.append({
                                        "Segment": i + 1,
                                        "Start Time": start_str,
                                        "End Time": end_str,
                                        "Beats": seg['n_beats'],
                                        "Mean RR (ms)": f"{seg['mean_rr']:.0f}",
                                        "Std (ms)": f"{seg['std_rr']:.1f}",
                                        "CV (%)": f"{cv_pct:.1f}",
                                        "Quality": quality
                                    })
                                st.dataframe(get_pandas().DataFrame(seg_data), width='stretch', hide_index=True)

                                # Check for high variability segments
                                high_var_segments = [s for s in cp_info['segment_stats'] if s['cv'] > 0.15]
                                if high_var_segments:
                                    st.markdown("##### Recommendations for High Variability:")
                                    st.markdown("""
                                    **What high variability (CV > 15%) may indicate:**
                                    - Movement artifacts (participant moved during recording)
                                    - Electrode contact issues (sensor shifted or loose)
                                    - Physiological response (stress, deep breathing, posture change)
                                    - Ectopic beats or arrhythmia

                                    **What to do:**
                                    1. **Check timing** - Does the high variability segment align with an expected event (e.g., task start)?
                                    2. **Add boundary events** - If it's artifact, mark `artifact_start` and `artifact_end` events
                                    3. **Use artifact correction** - In Analysis tab, enable "Apply artifact correction" to use the Kubios algorithm
                                    4. **Exclude if severe** - If CV > 25%, consider excluding that segment from analysis

                                    **When variability is expected:**
                                    - During music listening (emotional response)
                                    - During stress induction tasks
                                    - During breathing exercises
                                    """)

                                    # Auto-create variability boundary events
                                    st.markdown("##### Auto-Create Variability Events")
                                    col_var_thresh, col_var_btn, col_var_desc = st.columns([1, 1, 2])
                                    with col_var_thresh:
                                        cv_threshold = st.number_input(
                                            "CV threshold (%)",
                                            min_value=5.0,
                                            max_value=50.0,
                                            value=15.0,
                                            step=1.0,
                                            key=f"cv_thresh_{selected_participant}",
                                            help="Create boundary events for segments with CV above this threshold"
                                        )
                                    with col_var_btn:
                                        if st.button("Create Variability Events", key=f"auto_var_{selected_participant}"):
                                            from rrational.prep.summaries import EventStatus
                                            events_added = 0
                                            cv_thresh_decimal = cv_threshold / 100.0

                                            for seg in cp_info['segment_stats']:
                                                if seg['cv'] > cv_thresh_decimal:
                                                    # Create high_variability_start event
                                                    if seg.get('start_time'):
                                                        var_start_event = EventStatus(
                                                            raw_label="high_variability_start",
                                                            canonical="high_variability_start",
                                                            first_timestamp=seg['start_time'],
                                                            last_timestamp=seg['start_time']
                                                        )
                                                        st.session_state.participant_events[selected_participant]['manual'].append(var_start_event)
                                                        events_added += 1

                                                    # Create high_variability_end event
                                                    if seg.get('end_time'):
                                                        var_end_event = EventStatus(
                                                            raw_label="high_variability_end",
                                                            canonical="high_variability_end",
                                                            first_timestamp=seg['end_time'],
                                                            last_timestamp=seg['end_time']
                                                        )
                                                        st.session_state.participant_events[selected_participant]['manual'].append(var_end_event)
                                                        events_added += 1

                                            if events_added > 0:
                                                show_toast(f"Created {events_added} variability boundary events", icon="success")
                                            else:
                                                show_toast("No segments above threshold", icon="info")
                                            st.rerun()
                                    with col_var_desc:
                                        st.caption(f"Creates `high_variability_start` and `high_variability_end` events for segments with CV > {cv_threshold:.0f}%")

                            st.caption("""
                            **Legend**:
                            - **CV (Coefficient of Variation)** = Std / Mean × 100. Lower = more stable.
                            - [OK] Good: CV < 10% | Moderate: 10-15% | [X] High: > 15%
                            - **Gray regions** on plot = time gaps (missing data)
                            - **Colored regions** = variability segments (green=stable, orange=moderate, red=high)
                            """)

                    # Repetitive Event Generator (inside events mode)
                    with st.expander("Generate Repetitive Events", expanded=False):
                        st.markdown("""
                        **Auto-generate section events** at fixed time intervals within any time range.

                        Use this to mark repeating conditions, phases, or treatments that cycle at regular intervals
                        (e.g., music changes, stimuli presentations, condition blocks).
                        """)

                        # Ensure participant is initialized in events dict
                        if selected_participant not in st.session_state.participant_events:
                            st.session_state.participant_events[selected_participant] = {'events': [], 'manual': []}

                        # Get existing events for boundary selection
                        stored_data = st.session_state.participant_events.get(selected_participant, {'events': [], 'manual': []})
                        raw_events = stored_data.get('events', []) + stored_data.get('manual', []) + stored_data.get('generated_events', [])

                        # Helper to handle dicts (defensive for stale session state)
                        def get_event_attr(evt, attr, default=None):
                            """Get attribute from EventStatus or dict."""
                            if isinstance(evt, dict):
                                return evt.get(attr, default)
                            return getattr(evt, attr, default)

                        # Build list of available events with timestamps
                        available_events = {}
                        for evt in raw_events:
                            canonical = get_event_attr(evt, 'canonical')
                            first_ts = get_event_attr(evt, 'first_timestamp')
                            if canonical and first_ts:
                                available_events[canonical] = first_ts

                        st.markdown("**Time Range:**")

                        # Let user select start and end events OR enter manual times
                        range_mode = st.radio(
                            "Define range using:",
                            options=["Existing events", "Manual times"],
                            horizontal=True,
                            key=f"range_mode_{selected_participant}"
                        )

                        start_time = None
                        end_time = None

                        if range_mode == "Existing events":
                            event_list = list(available_events.keys())
                            if event_list:
                                col_start, col_end = st.columns(2)
                                with col_start:
                                    start_event = st.selectbox(
                                        "Start event",
                                        options=event_list,
                                        key=f"gen_start_event_{selected_participant}",
                                        help="Events will be generated starting from this time"
                                    )
                                    if start_event:
                                        start_time = available_events[start_event]
                                        st.caption(f"Time: {start_time.strftime('%H:%M:%S') if start_time else 'N/A'}")

                                with col_end:
                                    end_event = st.selectbox(
                                        "End event",
                                        options=event_list,
                                        key=f"gen_end_event_{selected_participant}",
                                        help="Events will be generated up to this time"
                                    )
                                    if end_event:
                                        end_time = available_events[end_event]
                                        st.caption(f"Time: {end_time.strftime('%H:%M:%S') if end_time else 'N/A'}")
                            else:
                                st.warning("No events detected yet. Add events first or use manual times.")

                        else:  # Manual times
                            col_start, col_end = st.columns(2)
                            with col_start:
                                start_time_str = st.text_input(
                                    "Start time (HH:MM:SS)",
                                    value="00:00:00",
                                    key=f"gen_start_manual_{selected_participant}"
                                )
                            with col_end:
                                end_time_str = st.text_input(
                                    "End time (HH:MM:SS)",
                                    value="01:00:00",
                                    key=f"gen_end_manual_{selected_participant}"
                                )

                            # Parse manual times (use recording start date as base)
                            try:
                                from datetime import datetime, timedelta
                                # Get recording base date from summary
                                summary = get_summary_dict().get(selected_participant)
                                base_date = summary.recording_datetime.date() if summary and summary.recording_datetime else datetime.now().date()

                                h, m, s = map(int, start_time_str.split(':'))
                                start_time = datetime.combine(base_date, datetime.min.time()) + timedelta(hours=h, minutes=m, seconds=s)

                                h, m, s = map(int, end_time_str.split(':'))
                                end_time = datetime.combine(base_date, datetime.min.time()) + timedelta(hours=h, minutes=m, seconds=s)
                            except Exception:
                                st.error("Invalid time format. Use HH:MM:SS")

                        st.markdown("---")
                        st.markdown("**Condition Settings:**")

                        col_interval, col_labels = st.columns(2)
                        with col_interval:
                            event_interval_min = st.number_input(
                                "Interval (minutes)",
                                min_value=1,
                                max_value=60,
                                value=5,
                                step=1,
                                key=f"event_interval_{selected_participant}",
                                help="Duration of each condition/phase"
                            )

                        with col_labels:
                            condition_labels = st.text_area(
                                "Condition labels (one per line)",
                                value="condition_1\ncondition_2\ncondition_3",
                                height=100,
                                key=f"condition_labels_{selected_participant}",
                                help="Labels for each condition that cycles"
                            )

                        # Parse condition labels
                        condition_label_list = [line.strip() for line in condition_labels.strip().split('\n') if line.strip()]
                        if not condition_label_list:
                            condition_label_list = ["condition_1", "condition_2", "condition_3"]

                        # Preview
                        st.markdown("**Preview:**")
                        if start_time and end_time:
                            duration_min = (end_time - start_time).total_seconds() / 60
                            num_segments = int(duration_min / event_interval_min)
                            st.write(f"Cycle: {' → '.join(condition_label_list)} → (repeat)")
                            st.caption(f"Duration: {duration_min:.1f} min → ~{num_segments} segments of {event_interval_min} min each")
                        else:
                            st.write(f"Cycle: {' → '.join(condition_label_list)} → (repeat)")
                            st.caption("Select start and end times above")

                        # Generate button
                        can_generate = start_time is not None and end_time is not None and start_time < end_time
                        if st.button("Generate Events", key=f"gen_events_{selected_participant}", disabled=not can_generate):
                            from rrational.prep.summaries import EventStatus
                            from datetime import timedelta

                            events_added = 0
                            interval_seconds = event_interval_min * 60
                            num_conditions = len(condition_label_list)

                            # Initialize generated_events list if not present
                            if 'generated_events' not in st.session_state.participant_events[selected_participant]:
                                st.session_state.participant_events[selected_participant]['generated_events'] = []
                            # Clear existing generated events before generating new ones
                            st.session_state.participant_events[selected_participant]['generated_events'] = []

                            # Generate events for the selected time range
                            current_time = start_time
                            condition_idx = 0

                            while current_time < end_time:
                                # Create condition start event
                                label = condition_label_list[condition_idx % num_conditions]
                                event_label = f"{label}_start"

                                new_event = EventStatus(
                                    raw_label=event_label,
                                    canonical=event_label,
                                    first_timestamp=current_time,
                                    last_timestamp=current_time
                                )
                                st.session_state.participant_events[selected_participant]['generated_events'].append(new_event)
                                events_added += 1

                                # Calculate end time for this condition segment
                                segment_end = current_time + timedelta(seconds=interval_seconds)
                                if segment_end > end_time:
                                    segment_end = end_time

                                # Create condition end event
                                end_event = EventStatus(
                                    raw_label=f"{label}_end",
                                    canonical=f"{label}_end",
                                    first_timestamp=segment_end,
                                    last_timestamp=segment_end
                                )
                                st.session_state.participant_events[selected_participant]['generated_events'].append(end_event)
                                events_added += 1

                                current_time = segment_end
                                condition_idx += 1

                            if events_added > 0:
                                show_toast(f"Created {events_added} section events", icon="success")
                            else:
                                show_toast("No events created - check time range", icon="warning")
                            st.rerun()

                        if not can_generate and start_time and end_time:
                            st.warning("End time must be after start time")

                        st.caption("""
                        **How it works:**
                        1. Select start/end time using existing events or enter manually
                        2. Set the interval duration (how long each condition lasts)
                        3. Define condition labels that will cycle
                        4. Generated events appear as `label_start` and `label_end` pairs
                        """)

                # ================== EVENTS MANAGEMENT (only in Add Events mode) ==================
                if interaction_mode == "Add Events":
                    st.markdown("---")

                    # Events table with reordering and inline editing
                    st.markdown("**Events Detected:**")

                    # Get events from session state (already initialized above for the plot)
                    stored_data = st.session_state.participant_events[selected_participant]

                    # Helper to ensure items are EventStatus objects (handles stale session state with dicts)
                    def ensure_event_status(item):
                        """Convert dict to EventStatus if needed."""
                        if isinstance(item, dict):
                            from rrational.prep.summaries import EventStatus
                            from datetime import datetime as dt
                            ts = item.get("first_timestamp")
                            if ts and isinstance(ts, str):
                                ts = dt.fromisoformat(ts)
                            last_ts = item.get("last_timestamp")
                            if last_ts and isinstance(last_ts, str):
                                last_ts = dt.fromisoformat(last_ts)
                            return EventStatus(
                                raw_label=item.get("raw_label", ""),
                                canonical=item.get("canonical"),
                                first_timestamp=ts,
                                last_timestamp=last_ts,
                            )
                        return item

                    # Ensure all events are EventStatus objects, not dicts
                    all_events = [ensure_event_status(e) for e in stored_data['events'] + stored_data['manual']]

                    if all_events:

                        # Helper function to safely compare datetimes (handle timezone-aware/naive mix)
                        def safe_compare_timestamps(ts1, ts2):
                            """Compare two timestamps, handling timezone-aware/naive mix."""
                            if ts1 is None or ts2 is None:
                                return 0  # Equal if either is None
                            # Make both timezone-aware or both timezone-naive
                            import datetime
                            if ts1.tzinfo is None and ts2.tzinfo is not None:
                                ts1 = ts1.replace(tzinfo=datetime.timezone.utc)
                            elif ts1.tzinfo is not None and ts2.tzinfo is None:
                                ts2 = ts2.replace(tzinfo=datetime.timezone.utc)
                            return 1 if ts1 > ts2 else (-1 if ts1 < ts2 else 0)

                        # Check timestamp order and display warning if needed
                        is_chronological = True
                        for i in range(len(all_events) - 1):
                            if all_events[i].first_timestamp and all_events[i+1].first_timestamp:
                                if safe_compare_timestamps(all_events[i].first_timestamp, all_events[i+1].first_timestamp) > 0:
                                    is_chronological = False
                                    break

                        if not is_chronological:
                            st.error("**Events are NOT in chronological order!** Click 'Auto-Sort by Timestamp' to fix.")

                    # Quick Add Event Section
                    st.markdown("### + Add Event")

                    # Get first RR timestamp as reference
                    first_rr_time = None
                    if 'rr_intervals' in recording_data and recording_data['rr_intervals']:
                        first_rr_time = recording_data['rr_intervals'][0][0]  # (timestamp, rr_ms, elapsed)

                    col_add1, col_add2, col_add3 = st.columns([2, 2, 1])

                    with col_add1:
                        # Quick event type selector
                        quick_events = ["measurement_start", "measurement_end", "pause_start", "pause_end",
                                       "rest_pre_start", "rest_pre_end", "rest_post_start", "rest_post_end",
                                       "Custom..."]
                        selected_quick_event = st.selectbox(
                            "Event type",
                            options=quick_events,
                            key=f"quick_event_type_{selected_participant}",
                            label_visibility="collapsed",
                            placeholder="Select event type..."
                        )

                        # Custom label input if "Custom..." selected
                        if selected_quick_event == "Custom...":
                            custom_label = st.text_input(
                                "Custom label",
                                key=f"custom_event_label_{selected_participant}",
                                placeholder="Enter event label..."
                            )
                        else:
                            custom_label = None

                    with col_add2:
                        # Time input with second precision using text input
                        import datetime as dt

                        # Default time from first RR timestamp
                        if first_rr_time and hasattr(first_rr_time, 'strftime'):
                            default_time_str = first_rr_time.strftime("%H:%M:%S")
                        else:
                            default_time_str = "10:00:00"

                        # Initialize session state key if not exists (don't use value= which resets on every render)
                        time_key = f"event_time_{selected_participant}"
                        if time_key not in st.session_state:
                            st.session_state[time_key] = default_time_str

                        event_time_str = st.text_input(
                            "Time (HH:MM:SS)",
                            key=time_key,
                            placeholder="HH:MM:SS",
                            help="Enter time as HH:MM:SS (e.g., 10:30:45)"
                        )

                        # Parse the time string
                        def parse_time_str(time_str):
                            """Parse HH:MM:SS or HH:MM string to time object."""
                            try:
                                parts = time_str.strip().split(":")
                                if len(parts) == 3:
                                    return dt.time(int(parts[0]), int(parts[1]), int(parts[2]))
                                elif len(parts) == 2:
                                    return dt.time(int(parts[0]), int(parts[1]), 0)
                                else:
                                    return None
                            except (ValueError, IndexError):
                                return None

                        # Optional: offset from measurement start
                        use_offset = st.checkbox(
                            "Use offset from start instead",
                            key=f"use_offset_{selected_participant}",
                            help="Enter time as offset (minutes:seconds) from recording start"
                        )

                        # Initialize offset values (prevents UnboundLocalError in callback)
                        offset_min = 0
                        offset_sec = 0
                        if use_offset:
                            col_min, col_sec = st.columns(2)
                            with col_min:
                                offset_min = st.number_input("Min", min_value=0, max_value=180, value=0,
                                                            key=f"offset_min_{selected_participant}")
                            with col_sec:
                                offset_sec = st.number_input("Sec", min_value=0, max_value=59, value=0,
                                                            key=f"offset_sec_{selected_participant}")

                    with col_add3:
                        def add_quick_event():
                            """Add event with selected type and time."""
                            from rrational.prep.summaries import EventStatus
                            import datetime as dt

                            # Read values from session state (not closure variables!)
                            # This ensures we get the CURRENT input values, not stale ones
                            curr_quick_event = st.session_state.get(f"quick_event_type_{selected_participant}")
                            curr_custom_label = st.session_state.get(f"custom_event_label_{selected_participant}")
                            curr_time_str = st.session_state.get(f"event_time_{selected_participant}", "")
                            curr_use_offset = st.session_state.get(f"use_offset_{selected_participant}", False)
                            curr_offset_min = st.session_state.get(f"offset_min_{selected_participant}", 0)
                            curr_offset_sec = st.session_state.get(f"offset_sec_{selected_participant}", 0)

                            # Determine label
                            label = curr_custom_label if curr_quick_event == "Custom..." else curr_quick_event
                            if not label:
                                show_toast("Please enter an event label", icon="error")
                                return

                            # Determine timestamp
                            if curr_use_offset and first_rr_time:
                                # Calculate from offset
                                offset_delta = dt.timedelta(minutes=curr_offset_min, seconds=curr_offset_sec)
                                event_timestamp = first_rr_time + offset_delta
                            else:
                                # Parse the time string
                                parsed_time = parse_time_str(curr_time_str)
                                if not parsed_time:
                                    show_toast("Invalid time format. Use HH:MM:SS", icon="error")
                                    return

                                # Use the parsed time with the recording date
                                if first_rr_time:
                                    event_timestamp = first_rr_time.replace(
                                        hour=parsed_time.hour,
                                        minute=parsed_time.minute,
                                        second=parsed_time.second,
                                        microsecond=0
                                    )
                                else:
                                    # Fallback to today's date
                                    event_timestamp = dt.datetime.combine(
                                        dt.date.today(),
                                        parsed_time,
                                        tzinfo=dt.timezone.utc
                                    )

                            # Normalize the label
                            canonical = st.session_state.normalizer.normalize(label)

                            # Check for duplicate events (same label/canonical and close timestamp)
                            if selected_participant in st.session_state.participant_events:
                                existing_data = st.session_state.participant_events[selected_participant]
                                existing_events = existing_data.get('events', []) + existing_data.get('manual', [])
                                for existing in existing_events:
                                    # Check if same label (raw or canonical)
                                    same_label = (
                                        existing.raw_label.lower() == label.lower() or
                                        (existing.canonical and existing.canonical == canonical)
                                    )
                                    # Check if timestamp within 1 second
                                    if same_label and existing.first_timestamp:
                                        time_diff = abs((existing.first_timestamp - event_timestamp).total_seconds())
                                        if time_diff < 1:
                                            show_toast(f"Event '{label}' already exists at this time", icon="warning")
                                            return

                            new_event = EventStatus(
                                raw_label=label,
                                canonical=canonical,
                                count=1,
                                first_timestamp=event_timestamp,
                                last_timestamp=event_timestamp,
                            )

                            # Add to participant events
                            if selected_participant not in st.session_state.participant_events:
                                st.session_state.participant_events[selected_participant] = {'events': [], 'manual': []}
                            st.session_state.participant_events[selected_participant]['manual'].append(new_event)
                            show_toast(f"Added '{label}' at {event_timestamp.strftime('%H:%M:%S')}", icon="success")

                        st.write("")  # Spacer
                        st.button(
                            "+ Add",
                            key=f"add_event_{selected_participant}",
                            on_click=add_quick_event,
                            type="primary",
                            width='stretch'
                        )

                    st.markdown("---")

                    # Section Validation - validates sections defined in Sections tab
                    with st.expander("Section Validation", expanded=True):
                        st.caption("Validates that all defined sections have required events and expected durations.")

                        # Get participant's events
                        stored_data = st.session_state.participant_events.get(selected_participant, {})
                        all_evts = stored_data.get('events', []) + stored_data.get('manual', [])

                        # Build event timestamp lookup
                        event_timestamps = {}
                        for evt in all_evts:
                            if evt.canonical and evt.first_timestamp:
                                event_timestamps[evt.canonical] = evt.first_timestamp

                        # Get sections from session state
                        sections = st.session_state.get("sections", {})

                        # Get FULL RR data for RR-based duration (HRV Logger only)
                        full_rr_key = f"full_rr_data_{selected_participant}"
                        full_rr_data = st.session_state.get(full_rr_key, {})
                        summary = get_summary_dict().get(selected_participant)
                        is_hrv_logger = (getattr(summary, 'source_app', 'HRV Logger') == "HRV Logger") if summary else True

                        # Prepare RR data for section calculation if HRV Logger
                        rr_timestamps = []
                        rr_values_ms = []
                        if is_hrv_logger and full_rr_data:
                            rr_timestamps = full_rr_data.get('timestamps', [])
                            rr_values_ms = full_rr_data.get('rr_values', [])

                        if not sections:
                            st.info("No sections defined. Define sections in the **Sections** tab (under Setup).")
                        else:
                            # Helper to normalize timestamps
                            def normalize_ts(ts):
                                if ts is None:
                                    return None
                                if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                                    return ts.replace(tzinfo=None)
                                return ts

                            # Get exclusion zones for duration calculation
                            participant_exclusion_zones = stored_data.get('exclusion_zones', [])

                            def calc_excluded_time(start_ts, end_ts):
                                """Calculate excluded time in seconds."""
                                if not participant_exclusion_zones or not start_ts or not end_ts:
                                    return 0.0
                                total = 0.0
                                for zone in participant_exclusion_zones:
                                    if not zone.get('exclude_from_duration', True):
                                        continue
                                    zs, ze = zone.get('start'), zone.get('end')
                                    if not zs or not ze:
                                        continue
                                    zs = normalize_ts(zs)
                                    ze = normalize_ts(ze)
                                    overlap_s = max(zs, normalize_ts(start_ts))
                                    overlap_e = min(ze, normalize_ts(end_ts))
                                    if overlap_s < overlap_e:
                                        total += (overlap_e - overlap_s).total_seconds()
                                return total

                            def calc_rr_duration(start_ts, end_ts):
                                """Calculate duration by summing RR intervals within time range."""
                                if not rr_timestamps or not rr_values_ms or not start_ts or not end_ts:
                                    return None
                                start_ts = normalize_ts(start_ts)
                                end_ts = normalize_ts(end_ts)
                                total_ms = 0
                                for ts, rr in zip(rr_timestamps, rr_values_ms):
                                    ts_norm = normalize_ts(ts) if hasattr(ts, 'tzinfo') else ts
                                    if start_ts <= ts_norm < end_ts:
                                        total_ms += rr
                                return total_ms / 1000 / 60  # Convert to minutes

                            # Get participant's group and filter sections by group-specific selections
                            participant_group = st.session_state.participant_groups.get(selected_participant, "Default")
                            group_data = st.session_state.groups.get(participant_group, {})
                            selected_sections = group_data.get("selected_sections", [])

                            # Filter sections to validate based on group's selected sections
                            # If no sections are selected for the group, validate all sections
                            if selected_sections:
                                sections_to_validate = {k: v for k, v in sections.items() if k in selected_sections}
                                st.caption(f"Showing sections for **{group_data.get('label', participant_group)}** group ({len(sections_to_validate)} of {len(sections)} sections)")
                            else:
                                sections_to_validate = sections
                                st.caption(f"No group-specific sections configured - showing all {len(sections)} sections")

                            # Validate each section
                            valid_count = 0
                            issue_count = 0

                            for section_code, section_data in sections_to_validate.items():
                                # Support both old (start_event/end_event) and new (start_events/end_events) format
                                start_evts = section_data.get("start_events", [])
                                if not start_evts and "start_event" in section_data:
                                    start_evts = [section_data["start_event"]]
                                end_evts = section_data.get("end_events", [])
                                if not end_evts and "end_event" in section_data:
                                    end_evts = [section_data["end_event"]]
                                label = section_data.get("label", section_code)
                                expected_dur = section_data.get("expected_duration_min", 0)
                                tolerance = section_data.get("tolerance_min", 1)

                                # Find the first matching start event
                                start_ts = None
                                matched_start_evt = None
                                for start_evt in start_evts:
                                    if start_evt in event_timestamps:
                                        start_ts = event_timestamps[start_evt]
                                        matched_start_evt = start_evt
                                        break

                                # Find the first matching end event
                                end_ts = None
                                matched_end_evt = None
                                for end_evt in end_evts:
                                    if end_evt in event_timestamps:
                                        end_ts = event_timestamps[end_evt]
                                        matched_end_evt = end_evt
                                        break

                                # Check event presence
                                start_evts_str = " | ".join(start_evts) if len(start_evts) > 1 else (start_evts[0] if start_evts else "none")
                                end_evts_str = " | ".join(end_evts) if len(end_evts) > 1 else (end_evts[0] if end_evts else "none")
                                if not start_ts and not end_ts:
                                    st.write(f"**{label}**: missing `{start_evts_str}` and `{end_evts_str}`")
                                    issue_count += 1
                                elif not start_ts:
                                    st.write(f"**{label}**: missing `{start_evts_str}`")
                                    issue_count += 1
                                elif not end_ts:
                                    st.write(f"**{label}**: missing `{end_evts_str}`")
                                    issue_count += 1
                                else:
                                    # Calculate event-based duration
                                    raw_dur = (normalize_ts(end_ts) - normalize_ts(start_ts)).total_seconds()
                                    excluded = calc_excluded_time(start_ts, end_ts)
                                    event_dur = (raw_dur - excluded) / 60

                                    # Calculate RR-based duration (HRV Logger only)
                                    rr_dur = calc_rr_duration(start_ts, end_ts) if is_hrv_logger else None

                                    excl_note = f" (excl: {excluded/60:.1f}m)" if excluded > 0 else ""
                                    start_evt_note = f"{matched_start_evt} → " if len(start_evts) > 1 else ""
                                    end_evt_note = f" → {matched_end_evt}" if len(end_evts) > 1 else ""

                                    # Build display string with both durations
                                    if rr_dur is not None:
                                        diff = event_dur - rr_dur
                                        diff_note = f" (Δ{diff:+.1f}m)" if abs(diff) > 0.1 else ""
                                        dur_display = f"{event_dur:.1f}m | RR: {rr_dur:.1f}m{diff_note}"
                                    else:
                                        dur_display = f"{event_dur:.1f}m"

                                    # Check if within tolerance (use event-based duration)
                                    evt_notes = f" ({start_evt_note}{end_evt_note.strip(' → ')})" if start_evt_note or end_evt_note else ""
                                    if expected_dur > 0 and abs(event_dur - expected_dur) > tolerance:
                                        st.write(f"**{label}**: {dur_display}{excl_note}{evt_notes} (expected {expected_dur:.0f}±{tolerance:.0f}m)")
                                        issue_count += 1
                                    else:
                                        st.write(f"**{label}**: {dur_display}{excl_note}{evt_notes}")
                                        valid_count += 1

                            # Summary
                            if issue_count == 0 and valid_count > 0:
                                st.success(f"All {valid_count} section(s) valid")
                            elif valid_count == 0 and issue_count > 0:
                                st.error(f"All {issue_count} section(s) have issues")

                        # RR+Gap Duration Validation (HRV Logger only)
                        # Compare event-based duration vs sum of RR intervals + gaps
                        plot_data_key = f"plot_data_{selected_participant}"
                        plot_data_for_validation = st.session_state.get(plot_data_key, {})

                        # Check if this is HRV Logger data (has real timestamps)
                        summary = get_summary_dict().get(selected_participant)
                        is_vns_data = (getattr(summary, 'source_app', 'HRV Logger') == "VNS Analyse") if summary else False

                        if not is_vns_data and plot_data_for_validation:
                            st.markdown("---")
                            st.markdown("**Data Integrity Check** (HRV Logger)")
                            st.caption("Compares event-based duration with RR+gap duration from recorded data")

                            total_rr_ms = plot_data_for_validation.get('total_rr_time_ms', 0)
                            # Get gap info from session state (computed during plot rendering)
                            gap_result_validation = st.session_state.get(f"gaps_{selected_participant}", {})
                            total_gap_ms = gap_result_validation.get('total_gap_duration_s', 0) * 1000
                            n_gaps = len(gap_result_validation.get('gaps', []))

                            # Convert to minutes for display
                            rr_duration_min = total_rr_ms / 1000 / 60
                            gap_duration_min = total_gap_ms / 1000 / 60
                            total_rr_gap_min = rr_duration_min + gap_duration_min

                            # Get event-based duration (first to last event)
                            if event_timestamps:
                                all_ts = [ts for ts in event_timestamps.values() if ts]
                                if len(all_ts) >= 2:
                                    first_event = min(all_ts)
                                    last_event = max(all_ts)
                                    event_duration_sec = (last_event - first_event).total_seconds()
                                    event_duration_min = event_duration_sec / 60

                                    # Compare with RR+gap duration
                                    diff_min = abs(event_duration_min - total_rr_gap_min)
                                    diff_pct = (diff_min / event_duration_min * 100) if event_duration_min > 0 else 0

                                    col_dur1, col_dur2, col_dur3 = st.columns(3)
                                    with col_dur1:
                                        st.metric("Event Duration", f"{event_duration_min:.1f} min", help="Time between first and last event")
                                    with col_dur2:
                                        st.metric("RR + Gap Duration", f"{total_rr_gap_min:.1f} min", help=f"Sum of all RR intervals ({rr_duration_min:.1f} min) + gaps ({gap_duration_min:.1f} min)")
                                    with col_dur3:
                                        delta_str = f"{diff_min:.1f} min ({diff_pct:.1f}%)"
                                        if diff_pct < 5:
                                            st.metric("Difference", delta_str, delta_color="off", help="Good data integrity")
                                        elif diff_pct < 15:
                                            st.metric("Difference", delta_str, delta_color="normal", help="Some data may be missing")
                                        else:
                                            st.metric("Difference", delta_str, delta_color="inverse", help="Significant data mismatch - check for missing data")

                                    if n_gaps > 0:
                                        st.caption(f"Detected {n_gaps} gap(s) totaling {gap_duration_min:.1f} min")
                                else:
                                    st.info("Need at least 2 events for duration comparison")
                            else:
                                st.info("No events to compare with RR duration")

                    st.markdown("---")

                    # Build available canonical events
                    available_canonical_events = list(st.session_state.all_events.keys())

                    # Section 1: Event Editing - Individual Cards
                    st.markdown("### Event Management")

                    # Initialize undo stack for this participant
                    undo_key = f"event_undo_stack_{selected_participant}"
                    if undo_key not in st.session_state:
                        st.session_state[undo_key] = []

                    def _save_undo_state(participant_id: str, action_desc: str):
                        """Save current state to undo stack before making changes."""
                        import copy
                        stored_data = st.session_state.participant_events.get(participant_id)
                        if stored_data:
                            # Deep copy the events
                            state = {
                                'events': copy.deepcopy(stored_data.get('events', [])),
                                'manual': copy.deepcopy(stored_data.get('manual', [])),
                                'action': action_desc
                            }
                            undo_stack = st.session_state.get(f"event_undo_stack_{participant_id}", [])
                            undo_stack.append(state)
                            # Keep only last 10 undo states
                            if len(undo_stack) > 10:
                                undo_stack = undo_stack[-10:]
                            st.session_state[f"event_undo_stack_{participant_id}"] = undo_stack

                    def _undo_last_action(participant_id: str):
                        """Undo the last event management action."""
                        undo_stack = st.session_state.get(f"event_undo_stack_{participant_id}", [])
                        if undo_stack:
                            prev_state = undo_stack.pop()
                            st.session_state[f"event_undo_stack_{participant_id}"] = undo_stack
                            # Restore the previous state
                            st.session_state.participant_events[participant_id]['events'] = prev_state['events']
                            st.session_state.participant_events[participant_id]['manual'] = prev_state['manual']
                            show_toast(f"Undid: {prev_state['action']}", icon="success")
                        else:
                            show_toast("Nothing to undo", icon="info")

                    # Undo button row
                    col_caption, col_undo = st.columns([4, 1])
                    with col_caption:
                        st.caption("Edit event details, match to canonical events, or delete events")
                    with col_undo:
                        undo_stack = st.session_state.get(undo_key, [])
                        st.button(
                            "↩ Undo",
                            key=f"undo_{selected_participant}",
                            on_click=_undo_last_action,
                            args=(selected_participant,),
                            disabled=len(undo_stack) == 0,
                            help=f"Undo last action ({len(undo_stack)} available)" if undo_stack else "No actions to undo"
                        )

                    # Define callbacks outside loop for better performance and to avoid stale closures
                    def _update_raw_label(participant_id: str, event_idx: int, evt_key: str):
                        """Callback to update raw label."""
                        key = f"raw_{evt_key}"
                        if key in st.session_state:
                            new_val = st.session_state[key]
                            stored_data = st.session_state.participant_events.get(participant_id)
                            if stored_data:
                                all_evts = stored_data['events'] + stored_data['manual']
                                if event_idx < len(all_evts):
                                    all_evts[event_idx].raw_label = new_val
                                    st.session_state.participant_events[participant_id]['events'] = all_evts
                                    st.session_state.participant_events[participant_id]['manual'] = []

                    if all_events:
                        events_to_delete = []

                        for idx, event in enumerate(all_events):
                            # Create unique key based on event content (not index) to avoid Streamlit widget caching issues
                            import hashlib
                            ts_str = event.first_timestamp.isoformat() if event.first_timestamp else "none"
                            event_hash = hashlib.md5(f"{event.raw_label}_{ts_str}".encode()).hexdigest()[:8]
                            event_key = f"{selected_participant}_{event_hash}"

                            with st.container():
                                # Create columns for this event
                                col_status, col_raw, col_canonical, col_syn, col_time, col_delete = st.columns([0.5, 2.5, 2.5, 1, 1.5, 0.5])

                                with col_status:
                                    # Show mapping status - only green check if canonical is valid
                                    if event.canonical and event.canonical != "unmatched" and event.canonical in st.session_state.all_events:
                                        st.markdown("*")
                                    else:
                                        st.markdown("[!]")

                                with col_raw:
                                    st.text_input(
                                        "Raw Label",
                                        value=event.raw_label,
                                        key=f"raw_{event_key}",
                                        label_visibility="collapsed",
                                        on_change=_update_raw_label,
                                        args=(selected_participant, idx, event_key)
                                    )

                                with col_canonical:
                                    # Canonical mapping dropdown with callback
                                    canonical_options = ["unmatched"] + available_canonical_events
                                    current_value = event.canonical if event.canonical else "unmatched"
                                    if current_value not in canonical_options:
                                        current_value = "unmatched"

                                    def update_canonical(participant_id, event_idx, evt_key):
                                        """Update canonical mapping and save to persistence."""
                                        key = f"canonical_{evt_key}"
                                        if key in st.session_state:
                                            new_val = st.session_state[key]
                                            stored_data = st.session_state.participant_events[participant_id]
                                            all_evts = stored_data['events'] + stored_data.get('manual', [])
                                            if event_idx < len(all_evts):
                                                all_evts[event_idx].canonical = new_val if new_val != "unmatched" else None
                                                st.session_state.participant_events[participant_id]['events'] = all_evts
                                                st.session_state.participant_events[participant_id]['manual'] = []
                                                # Save to persistence so changes persist across reruns
                                                from rrational.gui.persistence import save_participant_events
                                                save_participant_events(
                                                    participant_id,
                                                    st.session_state.participant_events[participant_id],
                                                    st.session_state.data_dir
                                                )

                                    st.selectbox(
                                        "Canonical",
                                        options=canonical_options,
                                        index=canonical_options.index(current_value),
                                        key=f"canonical_{event_key}",
                                        label_visibility="collapsed",
                                        on_change=update_canonical,
                                        args=(selected_participant, idx, event_key)
                                    )

                                with col_syn:
                                    # Add to synonyms button (only if canonical is selected)
                                    if event.canonical and event.canonical != "unmatched":
                                        def add_synonym(participant_id, event_idx, evt_key):
                                            """Add raw label as synonym to canonical event."""
                                            # Get the CURRENT canonical value from the selectbox
                                            canonical_key = f"canonical_{evt_key}"
                                            canonical_name = st.session_state.get(canonical_key)

                                            # Get the current event to get the raw label
                                            stored_data = st.session_state.participant_events[participant_id]
                                            current_events = stored_data['events'] + stored_data['manual']
                                            if event_idx >= len(current_events):
                                                show_toast("Event not found", icon="error")
                                                return

                                            raw_label = current_events[event_idx].raw_label

                                            if not canonical_name or canonical_name == "unmatched":
                                                show_toast("Please select a canonical event first", icon="warning")
                                                return

                                            if canonical_name not in st.session_state.all_events:
                                                # Debug info
                                                available = list(st.session_state.all_events.keys())
                                                show_toast(f"Event '{canonical_name}' not in list. Available: {', '.join(available[:5])}", icon="error")
                                                return

                                            raw_lower = raw_label.strip().lower()
                                            # Add to all_events synonyms for this canonical event
                                            if raw_lower not in st.session_state.all_events[canonical_name]:
                                                st.session_state.all_events[canonical_name].append(raw_lower)
                                                # Save to YAML using the save_events function
                                                save_events(st.session_state.all_events, st.session_state.get("current_project"))
                                                # Update normalizer after adding synonym
                                                update_normalizer()
                                                # Update the current event's canonical value in session state
                                                stored_data = st.session_state.participant_events[participant_id]
                                                current_events = stored_data['events'] + stored_data['manual']
                                                if event_idx < len(current_events):
                                                    current_events[event_idx].canonical = canonical_name
                                                    st.session_state.participant_events[participant_id]['events'] = current_events
                                                    st.session_state.participant_events[participant_id]['manual'] = []
                                                show_toast(f"Added '{raw_lower}' as synonym for {canonical_name}", icon="success")
                                            else:
                                                show_toast(f"'{raw_lower}' is already a synonym for {canonical_name}", icon="info")

                                        st.button("+Tag", key=f"syn_{event_key}",
                                                 on_click=add_synonym, args=(selected_participant, idx, event_key),
                                                 help="Add raw label as synonym")

                                with col_time:
                                    # Editable time input with second precision using text
                                    import datetime as dt
                                    current_time_str = event.first_timestamp.strftime("%H:%M:%S") if event.first_timestamp else "00:00:00"

                                    def update_event_time(participant_id, event_idx, original_ts, evt_key):
                                        """Update event timestamp from time text input."""
                                        key = f"time_{evt_key}"
                                        if key in st.session_state:
                                            time_str = st.session_state[key]
                                            # Parse HH:MM:SS
                                            try:
                                                parts = time_str.strip().split(":")
                                                if len(parts) >= 2:
                                                    h = int(parts[0])
                                                    m = int(parts[1])
                                                    s = int(parts[2]) if len(parts) > 2 else 0
                                                    if original_ts:
                                                        new_ts = original_ts.replace(hour=h, minute=m, second=s, microsecond=0)
                                                        stored = st.session_state.participant_events[participant_id]
                                                        all_evts = stored['events'] + stored['manual']
                                                        if event_idx < len(all_evts):
                                                            all_evts[event_idx].first_timestamp = new_ts
                                                            all_evts[event_idx].last_timestamp = new_ts
                                                            st.session_state.participant_events[participant_id]['events'] = all_evts
                                                            st.session_state.participant_events[participant_id]['manual'] = []
                                            except (ValueError, IndexError):
                                                pass  # Invalid format, ignore

                                    st.text_input(
                                        "Time",
                                        value=current_time_str,
                                        key=f"time_{event_key}",
                                        label_visibility="collapsed",
                                        on_change=update_event_time,
                                        args=(selected_participant, idx, event.first_timestamp, event_key),
                                        help="HH:MM:SS"
                                    )

                                with col_delete:
                                    # Delete button - use return value
                                    if st.button("X", key=f"delete_{event_key}", help=f"Delete '{event.raw_label}'"):
                                        events_to_delete.append(idx)

                                st.divider()

                        # Apply deletions after the loop (reverse order to preserve indices)
                        if events_to_delete:
                            for del_idx in sorted(events_to_delete, reverse=True):
                                _save_undo_state(selected_participant, f"Delete '{all_events[del_idx].raw_label}'")
                                del all_events[del_idx]
                            st.session_state.participant_events[selected_participant]['events'] = all_events
                            st.session_state.participant_events[selected_participant]['manual'] = []
                            from rrational.gui.persistence import save_participant_events
                            save_participant_events(
                                selected_participant,
                                st.session_state.participant_events[selected_participant],
                                st.session_state.data_dir
                            )
                            st.rerun()

                    else:
                        st.info("No events found. Click 'Add Event' to create one.")

                    st.markdown("---")

                    # Refresh all_events from session state
                    stored_data = st.session_state.participant_events[selected_participant]
                    all_events = [ensure_event_status(e) for e in stored_data['events'] + stored_data['manual']]

                    # Section 2: Event Order with Move Buttons
                    st.markdown("### Event Order")
                    st.caption("Use ↑↓ buttons to reorder events - changes reflect immediately in all sections")

                    # Helper to normalize timestamps for comparison (handle timezone-aware vs naive)
                    def get_sort_key(event):
                        ts = event.first_timestamp
                        if ts is None:
                            return get_pandas().Timestamp.max.tz_localize('UTC')
                        # Ensure all timestamps are timezone-aware for comparison
                        if hasattr(ts, 'tzinfo') and ts.tzinfo is None:
                            ts = get_pandas().Timestamp(ts).tz_localize('UTC')
                        return ts

                    # Auto-sort button - use return value instead of on_click for proper rerun
                    if st.button("Auto-Sort by Timestamp", key=f"auto_sort_{selected_participant}"):
                        all_events_copy = (st.session_state.participant_events[selected_participant]['events'] +
                                          st.session_state.participant_events[selected_participant]['manual'])
                        all_events_copy.sort(key=get_sort_key)
                        st.session_state.participant_events[selected_participant]['events'] = all_events_copy
                        st.session_state.participant_events[selected_participant]['manual'] = []
                        st.rerun()

                    if all_events:
                        # Check for pending move actions first
                        move_action = None

                        # Display compact event list
                        for idx, event in enumerate(all_events):
                            col_order, col_status, col_info, col_move = st.columns([0.5, 0.8, 5, 1.2])

                            with col_order:
                                st.text(f"{idx + 1}")

                            with col_status:
                                # Check if out of order
                                out_of_order = False
                                if idx > 0 and event.first_timestamp and all_events[idx-1].first_timestamp:
                                    if safe_compare_timestamps(all_events[idx-1].first_timestamp, event.first_timestamp) > 0:
                                        out_of_order = True

                                # Colored status indicators
                                if out_of_order:
                                    status_html = '<span style="color: #FF4B4B; font-weight: bold;">OUT OF ORDER</span>'
                                else:
                                    status_html = '<span style="color: #21C354;">OK</span>'

                                if not event.canonical:
                                    mapping_html = ' <span style="color: #FFA500; font-weight: bold;">UNMAPPED</span>'
                                else:
                                    mapping_html = ''

                                st.markdown(f"{status_html}{mapping_html}", unsafe_allow_html=True)

                            with col_info:
                                timestamp_str = event.first_timestamp.strftime("%H:%M:%S") if event.first_timestamp else "—"
                                canonical_str = event.canonical if event.canonical else "unmatched"
                                st.markdown(f"`{event.raw_label}` → **{canonical_str}** ({timestamp_str})")

                            with col_move:
                                m1, m2 = st.columns(2)
                                with m1:
                                    if idx > 0:
                                        if st.button("↑", key=f"up_{selected_participant}_{idx}"):
                                            move_action = ('up', idx)
                                with m2:
                                    if idx < len(all_events) - 1:
                                        if st.button("↓", key=f"dn_{selected_participant}_{idx}"):
                                            move_action = ('down', idx)

                        # Process move action after rendering all buttons
                        if move_action:
                            direction, idx = move_action
                            all_evts = st.session_state.participant_events[selected_participant]['events'] + \
                                      st.session_state.participant_events[selected_participant]['manual']
                            if direction == 'up' and idx > 0:
                                all_evts[idx], all_evts[idx-1] = all_evts[idx-1], all_evts[idx]
                            elif direction == 'down' and idx < len(all_evts) - 1:
                                all_evts[idx], all_evts[idx+1] = all_evts[idx+1], all_evts[idx]
                            st.session_state.participant_events[selected_participant]['events'] = all_evts
                            st.session_state.participant_events[selected_participant]['manual'] = []
                            st.rerun()
                    else:
                        st.info("No events to reorder.")

                    # Download button
                    events_data = []
                    for idx, event in enumerate(all_events):
                        events_data.append({
                            "Position": idx + 1,
                            "Raw Label": event.raw_label,
                            "Canonical": event.canonical or "unmatched",
                            "Timestamp": event.first_timestamp.strftime("%Y-%m-%d %H:%M:%S") if event.first_timestamp else "",
                            "Count": event.count,
                        })

                    # Create download button AFTER the loop (once)
                    if events_data:
                        df_events = get_pandas().DataFrame(events_data)
                        csv_events = df_events.to_csv(index=False)
                        st.download_button(
                            label=" Download Events CSV",
                            data=csv_events,
                            file_name=f"events_{selected_participant}.csv",
                            mime="text/csv",
                            width='stretch',
                            key=f"download_events_{selected_participant}",
                        )

                        # Show unmatched warning
                        unmatched_count = sum(1 for e in all_events if not e.canonical)
                        if unmatched_count > 0:
                            st.warning(f"{unmatched_count} unmatched event(s) - assign canonical mappings above")
                    else:
                        st.info("No events found for this participant")

                    # ISSUE 4 FIX: Show event mapping status (visible and working)
                    st.markdown("---")
                    st.markdown("**Event Mapping Status:**")

                    # Get expected events for this participant's group
                    participant_group = st.session_state.participant_groups.get(selected_participant, "Default")
                    expected_events = st.session_state.groups.get(participant_group, {}).get("expected_events", {})

                    # Get current events with raw labels
                    stored_data = st.session_state.participant_events[selected_participant]
                    current_events = stored_data['events'] + stored_data['manual']

                    if expected_events:
                        mapping_data = []
                        for event_name, synonyms in expected_events.items():
                            # Check if this canonical event exists in current session state events
                            matched = any(e.canonical == event_name for e in current_events)
                            # Find raw labels that matched this canonical event
                            matching_raw = [e.raw_label for e in current_events if e.canonical == event_name]
                            raw_labels_str = ", ".join(matching_raw) if matching_raw else "—"

                            mapping_data.append({
                                "Expected Event": event_name,
                                "Status": "Found" if matched else "Missing",
                                "Raw Labels": raw_labels_str,
                            })

                        df_mapping = get_pandas().DataFrame(mapping_data)
                        st.dataframe(df_mapping, width='stretch', hide_index=True)
                    else:
                        st.info(f"No expected events defined for group '{participant_group}'. Add them in the Event Mapping tab.")

                    # Save/Reset participant events
                    st.markdown("---")
                    from rrational.gui.persistence import (
                        save_participant_events,
                        load_participant_events,
                        delete_participant_events,
                        list_saved_participant_events,
                    )

                    col_save, col_reset, col_status = st.columns([1, 1, 2])

                    with col_save:
                        def save_events_to_yaml():
                            """Save participant events to YAML persistence."""
                            stored_data = st.session_state.participant_events.get(selected_participant, {})
                            save_participant_events(selected_participant, stored_data, st.session_state.data_dir)
                            show_toast(f"Saved events for {selected_participant}", icon="success")

                        st.button("Save Events",
                                 key=f"save_{selected_participant}",
                                 on_click=save_events_to_yaml,
                                 help="Save all event changes for this participant",
                                 type="primary")

                    with col_reset:
                        def reset_to_original():
                            """Reset participant events to original (from raw data file)."""
                            # Delete saved events YAML file (from both locations)
                            delete_participant_events(selected_participant, st.session_state.data_dir)
                            # Clear from session state so it reloads from original raw data
                            if selected_participant in st.session_state.participant_events:
                                del st.session_state.participant_events[selected_participant]
                            # Also clear any manual events
                            if selected_participant in st.session_state.get('manual_events', {}):
                                del st.session_state.manual_events[selected_participant]
                            # Clear undo stack for this participant
                            undo_key = f"event_undo_stack_{selected_participant}"
                            if undo_key in st.session_state:
                                st.session_state[undo_key] = []
                            show_toast(f"Reset {selected_participant} to original events from raw data", icon="success")

                        # Get list of saved participants for status display
                        saved_participants = list_saved_participant_events()

                        # Reset button is ALWAYS enabled - user can always reset to original raw data
                        st.button("Reset to Original",
                                 key=f"reset_{selected_participant}",
                                 on_click=reset_to_original,
                                 help="Discard all changes and reload original events from raw data file")

                    with col_status:
                        # Check if participant has saved events
                        if selected_participant in saved_participants:
                            st.caption("Has saved event edits")
                        else:
                            st.caption("Not yet saved")

                    # Export for Analysis section
                    st.markdown("---")
                    with st.expander("Export for Analysis", expanded=False):
                        st.caption("Export data with corrections and audit trail")

                        # Export type selection
                        export_type = st.radio(
                            "Export type",
                            ["Full Recording", "Selected Sections", "Custom Time Range"],
                            key=f"export_type_{selected_participant}",
                            horizontal=True
                        )

                        # Get sections for selection
                        sections = load_sections()
                        section_options = {name: s.get("label", name) for name, s in sections.items() if s.get("start_event")}

                        selected_sections = []
                        custom_start = None
                        custom_end = None
                        custom_label = ""

                        if export_type == "Selected Sections":
                            selected_sections = st.multiselect(
                                "Sections to export",
                                options=list(section_options.keys()),
                                format_func=lambda x: section_options.get(x, x),
                                key=f"export_sections_{selected_participant}",
                                help="Select one or more sections to export"
                            )
                        elif export_type == "Custom Time Range":
                            # Get recording time range for defaults
                            summary = get_summary_dict().get(selected_participant)
                            default_start = summary.first_timestamp if summary else None
                            default_end = summary.last_timestamp if summary else None

                            col_start, col_end = st.columns(2)
                            with col_start:
                                custom_start = st.text_input(
                                    "Start time (HH:MM:SS)",
                                    value=default_start.strftime("%H:%M:%S") if default_start else "09:00:00",
                                    key=f"export_start_{selected_participant}"
                                )
                            with col_end:
                                custom_end = st.text_input(
                                    "End time (HH:MM:SS)",
                                    value=default_end.strftime("%H:%M:%S") if default_end else "12:00:00",
                                    key=f"export_end_{selected_participant}"
                                )
                            custom_label = st.text_input(
                                "Segment label",
                                value="custom_selection",
                                key=f"export_label_{selected_participant}",
                                help="Name for this custom segment"
                            )

                        # Export options
                        st.markdown("**Include in export:**")
                        col_opt1, col_opt2 = st.columns(2)
                        with col_opt1:
                            include_artifacts = st.checkbox("Artifact detection", value=True,
                                                           key=f"exp_artifacts_{selected_participant}")
                            include_manual = st.checkbox("Manual markings", value=True,
                                                        key=f"exp_manual_{selected_participant}")
                        with col_opt2:
                            include_corrected = st.checkbox("Corrected NN intervals", value=False,
                                                           key=f"exp_corrected_{selected_participant}")
                            include_audit = st.checkbox("Full audit trail", value=True,
                                                       key=f"exp_audit_{selected_participant}")

                        # Export button
                        if st.button("Export (.rrational)", key=f"export_btn_{selected_participant}",
                                    type="primary", help="Export to processed folder"):
                            from rrational.gui.rrational_export import (
                                RRationalExport, RRIntervalExport, SegmentDefinition,
                                ArtifactDetection, ManualArtifact, QualityMetrics, ProcessingStep, save_rrational,
                                build_export_filename, get_quality_grade, get_quigley_recommendation
                            )
                            from datetime import datetime
                            # Path is already imported at module level

                            # Get data directory
                            data_dir = st.session_state.get("data_dir")
                            if not data_dir:
                                st.error("No data directory set")
                            else:
                                # Get artifact state
                                artifacts_key = f"artifacts_{selected_participant}"
                                manual_artifacts_key = f"manual_artifacts_{selected_participant}"
                                exclusions_key = f"artifact_exclusions_{selected_participant}"

                                artifacts_data = st.session_state.get(artifacts_key, {})
                                manual_artifacts_list = st.session_state.get(manual_artifacts_key, [])
                                excluded_indices = list(st.session_state.get(exclusions_key, set()))

                                # Get plot data
                                plot_data_key = f"plot_data_{selected_participant}"
                                plot_data = st.session_state.get(plot_data_key, {})
                                rr_values = plot_data.get('rr_values', [])
                                timestamps = plot_data.get('timestamps', [])

                                if not rr_values:
                                    st.error("No RR data loaded - view participant first")
                                else:
                                    now = datetime.now().isoformat()
                                    processed_dir = Path(data_dir).parent / "processed"

                                    # Get source info
                                    summary = get_summary_dict().get(selected_participant)
                                    source_app = getattr(summary, 'source_app', 'HRV Logger') if summary else 'HRV Logger'
                                    source_paths = getattr(summary, 'rr_paths', []) if summary else []
                                    recording_dt = getattr(summary, 'recording_datetime', None) if summary else None
                                    if recording_dt and hasattr(recording_dt, 'isoformat'):
                                        recording_dt = recording_dt.isoformat()

                                    # Determine what to export
                                    exports_to_make = []
                                    if export_type == "Full Recording":
                                        exports_to_make.append(("full", None, None))
                                    elif export_type == "Selected Sections":
                                        for sec_name in selected_sections:
                                            exports_to_make.append(("section", sec_name, None))
                                    else:  # Custom Time Range
                                        exports_to_make.append(("custom", custom_label, (custom_start, custom_end)))

                                    export_count = 0
                                    for exp_type, exp_name, exp_range in exports_to_make:
                                        # Build RR interval exports
                                        rr_exports = []
                                        for i, (ts, rr) in enumerate(zip(timestamps, rr_values)):
                                            ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)
                                            rr_exports.append(RRIntervalExport(
                                                timestamp=ts_str,
                                                rr_ms=int(rr),
                                                original_idx=i
                                            ))

                                        # Build artifact detection
                                        artifact_detection = None
                                        if include_artifacts and artifacts_data:
                                            artifact_detection = ArtifactDetection(
                                                method=artifacts_data.get('method', 'threshold'),
                                                total=artifacts_data.get('total_artifacts', 0),
                                                by_type=artifacts_data.get('by_type', {}),
                                                indices=artifacts_data.get('artifact_indices', [])
                                            )

                                        # Build manual artifacts
                                        manual_exports = []
                                        if include_manual:
                                            for ma in manual_artifacts_list:
                                                ts_str = ma.get('timestamp', '')
                                                if hasattr(ts_str, 'isoformat'):
                                                    ts_str = ts_str.isoformat()
                                                manual_exports.append(ManualArtifact(
                                                    original_idx=ma.get('original_idx', 0),
                                                    timestamp=ts_str,
                                                    rr_value=ma.get('rr_value', 0),
                                                    marked_at=now
                                                ))

                                        # Calculate final artifacts
                                        detected_indices = set(artifacts_data.get('artifact_indices', [])) if include_artifacts else set()
                                        manual_indices = set(ma.get('original_idx', 0) for ma in manual_artifacts_list) if include_manual else set()
                                        excluded_set = set(excluded_indices)
                                        final_artifacts = list((detected_indices | manual_indices) - excluded_set)

                                        # Quality metrics
                                        artifact_rate = len(final_artifacts) / len(rr_values) if rr_values else 0
                                        quality = QualityMetrics(
                                            artifact_rate_raw=artifacts_data.get('artifact_ratio', 0) if include_artifacts else 0,
                                            artifact_rate_final=artifact_rate,
                                            beats_after_cleaning=len(rr_values),
                                            quality_grade=get_quality_grade(artifact_rate),
                                            quigley_recommendation=get_quigley_recommendation(artifact_rate, len(rr_values))
                                        )

                                        # Segment definition
                                        segment_def = SegmentDefinition(
                                            type=exp_type if exp_type != "custom" else "manual_range",
                                            section_name=exp_name if exp_type == "section" else None,
                                            time_range={"start": exp_range[0], "end": exp_range[1], "label": exp_name} if exp_type == "custom" else None
                                        )

                                        # Audit trail
                                        steps = []
                                        if include_audit:
                                            steps = [
                                                ProcessingStep(step=1, action="export_ready_for_analysis",
                                                              timestamp=now, details=f"Exported {len(rr_values)} beats ({exp_type})")
                                            ]

                                        # Software versions
                                        import neurokit2 as nk
                                        import sys
                                        software_versions = {
                                            "rrational": "0.7.0",
                                            "neurokit2": getattr(nk, '__version__', 'unknown'),
                                            "python": sys.version.split()[0]
                                        } if include_audit else {}

                                        # Create export
                                        export_data = RRationalExport(
                                            participant_id=selected_participant,
                                            export_timestamp=now,
                                            exported_by="RRational v0.7.0",
                                            source_app=source_app,
                                            source_file_paths=[str(p) for p in source_paths],
                                            recording_datetime=recording_dt,
                                            segment=segment_def,
                                            n_beats=len(rr_exports),
                                            rr_intervals=rr_exports,
                                            artifact_detection=artifact_detection,
                                            manual_artifacts=manual_exports,
                                            excluded_detected_indices=excluded_indices if include_manual else [],
                                            final_artifact_indices=final_artifacts,
                                            include_corrected=include_corrected,
                                            quality=quality,
                                            processing_steps=steps,
                                            software_versions=software_versions
                                        )

                                        # Save
                                        filename = build_export_filename(selected_participant, exp_name)
                                        filepath = processed_dir / filename
                                        save_rrational(export_data, filepath)
                                        export_count += 1

                                    st.success(f"Exported {export_count} file(s) to {processed_dir}")

            # Bottom navigation buttons (duplicate for convenience)
            st.markdown("---")
            col_nav1, col_nav2, col_nav3 = st.columns([3, 1, 1])
            with col_nav1:
                st.caption(f"Participant {st.session_state.current_participant_idx + 1} of {len(participant_list)}")
            with col_nav2:
                def go_previous_bottom():
                    if st.session_state.current_participant_idx > 0:
                        st.session_state.current_participant_idx -= 1
                        new_participant = participant_list[st.session_state.current_participant_idx]
                        st.session_state.participant_selector = new_participant
                        st.session_state._scroll_to_top = True

                st.button(
                    "Previous",
                    disabled=st.session_state.current_participant_idx == 0,
                    key="prev_btn_bottom",
                    width='stretch',
                    on_click=go_previous_bottom
                )
            with col_nav3:
                def go_next_bottom():
                    if st.session_state.current_participant_idx < len(participant_list) - 1:
                        st.session_state.current_participant_idx += 1
                        new_participant = participant_list[st.session_state.current_participant_idx]
                        st.session_state.participant_selector = new_participant
                        st.session_state._scroll_to_top = True

                st.button(
                    "Next",
                    disabled=st.session_state.current_participant_idx >= len(participant_list) - 1,
                    key="next_btn_bottom",
                    width='stretch',
                    on_click=go_next_bottom
                )

    # ================== TAB: SETUP ==================
    elif selected_page == "Setup":
        _get_render_setup_tab()()

    # ================== TAB: ANALYSIS ==================
    elif selected_page == "Analysis":
        _get_render_analysis_tab()()

    # Record render time for debugging
    st.session_state.last_render_time = (_time.time() - _script_start) * 1000


if __name__ == "__main__":
    main()
