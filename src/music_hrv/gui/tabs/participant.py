"""Participant tab - Individual participant inspection and event management.

This module contains the render function for the Participant tab.
Due to the complexity and tight integration with session state,
the main implementation remains in app.py and is called from here.
"""

from __future__ import annotations

import streamlit as st


def render_participant_tab():
    """Render the Participant tab content.

    This tab contains:
    - Individual participant view (plot, events, quality metrics)
    - Protocol & Timing configuration (auto-fill, validation)
    - Event management and exclusion zones

    Note: The implementation is in app.py's main() function due to
    complex session state dependencies.
    """
    st.header("Participant Details")

    if not st.session_state.summaries:
        st.info("Load data in the **Data** tab first to view participant details.")
        return

    # Placeholder - actual implementation in app.py
    st.warning("Implementation pending - content being migrated from app.py")
