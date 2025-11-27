"""Streamlit-based GUI for the Music HRV Toolkit."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from pathlib import Path

from music_hrv.cleaning.rr import CleaningConfig
from music_hrv.io import DEFAULT_ID_PATTERN, load_recording, discover_recordings
from music_hrv.prep import load_hrv_logger_preview
from music_hrv.segments.section_normalizer import SectionNormalizer
from music_hrv.config.sections import SectionsConfig, SectionDefinition
import time
import re
from music_hrv.gui.persistence import (
    save_groups,
    load_groups,
    save_events,
    load_events,
    save_sections,
    load_sections,
    save_participants,
    load_participants,
)

try:
    import neurokit2 as nk
    import matplotlib.pyplot as plt
    NEUROKIT_AVAILABLE = True
except ImportError:
    NEUROKIT_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from streamlit_plotly_events import plotly_events
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None
    plotly_events = None

# Page configuration
st.set_page_config(
    page_title="Music HRV Toolkit",
    page_icon="🎵",
    layout="wide",
)

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
if "data_dir" not in st.session_state:
    st.session_state.data_dir = None
if "summaries" not in st.session_state:
    st.session_state.summaries = []
# Note: normalizer will be created after all_events is loaded
if "cleaning_config" not in st.session_state:
    st.session_state.cleaning_config = CleaningConfig()

# Load persisted groups and events
if "groups" not in st.session_state:
    loaded_groups = load_groups()
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
    loaded_events = load_events()
    if not loaded_events:
        st.session_state.all_events = DEFAULT_CANONICAL_EVENTS.copy()
    else:
        st.session_state.all_events = loaded_events

# ISSUE 1 FIX: Create normalizer from GUI events only (not from sections.yml)
if "normalizer" not in st.session_state:
    st.session_state.normalizer = create_gui_normalizer(st.session_state.all_events)

# Load participant-specific data (groups, event orders, manual events)
if "participant_groups" not in st.session_state or "event_order" not in st.session_state:
    loaded_participants = load_participants()
    if loaded_participants:
        st.session_state.participant_groups = {
            pid: data.get("group", "Default")
            for pid, data in loaded_participants.items()
        }
        st.session_state.event_order = {
            pid: data.get("event_order", [])
            for pid, data in loaded_participants.items()
        }
        st.session_state.manual_events = {
            pid: data.get("manual_events", [])
            for pid, data in loaded_participants.items()
        }
    else:
        st.session_state.participant_groups = {}
        st.session_state.event_order = {}
        st.session_state.manual_events = {}


def save_all_config():
    """Save all configuration to persistent storage."""
    save_groups(st.session_state.groups)
    save_events(st.session_state.all_events)
    if hasattr(st.session_state, 'sections'):
        save_sections(st.session_state.sections)
    save_participant_data()


def save_participant_data():
    """Save participant-specific data (groups, event orders, manual events)."""
    participants_data = {}
    all_participant_ids = set(
        list(st.session_state.participant_groups.keys()) +
        list(st.session_state.event_order.keys()) +
        list(st.session_state.manual_events.keys())
    )

    for pid in all_participant_ids:
        participants_data[pid] = {
            "group": st.session_state.participant_groups.get(pid, "Default"),
            "event_order": st.session_state.event_order.get(pid, []),
            "manual_events": st.session_state.manual_events.get(pid, []),
        }

    save_participants(participants_data)


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


@st.cache_data(show_spinner=False)
def cached_load_participants():
    """Cached version of load_participants for faster access."""
    return load_participants()


def show_toast(message, icon="success"):
    """Show a toast notification with auto-dismiss."""
    if icon == "success":
        st.toast(f"✅ {message}", icon="✅")
    elif icon == "info":
        st.toast(f"ℹ️ {message}", icon="ℹ️")
    elif icon == "warning":
        st.toast(f"⚠️ {message}", icon="⚠️")
    elif icon == "error":
        st.toast(f"❌ {message}", icon="❌")
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


def extract_section_rr_intervals(recording, section_def, normalizer):
    """Extract RR intervals for a specific section based on start/end events.

    Args:
        recording: HRVLoggerRecording object
        section_def: dict with 'start_event' and 'end_event' keys
        normalizer: SectionNormalizer instance

    Returns:
        list of RRInterval objects for the section, or None if events not found
    """

    start_event_name = section_def.get("start_event")
    end_event_name = section_def.get("end_event")

    if not start_event_name or not end_event_name:
        return None

    # Find start and end event timestamps
    start_ts = None
    end_ts = None

    for event in recording.events:
        canonical = normalizer.normalize(event.label)
        if canonical == start_event_name and event.timestamp:
            start_ts = event.timestamp
        elif canonical == end_event_name and event.timestamp:
            end_ts = event.timestamp

    if not start_ts or not end_ts:
        return None

    # Extract RR intervals between start and end timestamps
    section_rr = []
    for rr in recording.rr_intervals:
        if rr.timestamp and start_ts <= rr.timestamp <= end_ts:
            section_rr.append(rr)

    return section_rr if section_rr else None


def main():
    """Main Streamlit app."""
    st.title("🎵 Music HRV Toolkit")
    st.markdown("### HRV Analysis Pipeline for Music Psychology Research")

    # Minimal sidebar - just status info
    with st.sidebar:
        st.header("🎵 Music HRV Toolkit")

        # Show last save time if available
        if "last_save_time" in st.session_state:
            elapsed = time.time() - st.session_state.last_save_time
            if elapsed < 3:
                st.success("💾 Saved ✓")
            else:
                st.markdown("**💾 Auto-save enabled**")
        else:
            st.markdown("**💾 Auto-save enabled**")

        st.markdown("---")
        st.caption("Configure data import settings in the **Data & Groups** tab.")

    # Main content tabs with Analysis tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Data & Groups",
        "🎭 Event Mapping",
        "👥 Group Management",
        "📐 Sections",
        "📊 Analysis"
    ])

    # ================== TAB 1: Data & Groups ==================
    with tab1:
        st.header("Data Import & Participants")

        # Import Settings section
        with st.expander("⚙️ Import Settings", expanded=False):
            col_cfg1, col_cfg2 = st.columns(2)

            with col_cfg1:
                st.markdown("**Participant ID Pattern**")
                id_pattern = st.text_input(
                    "Regex pattern",
                    value=DEFAULT_ID_PATTERN,
                    help="Regex pattern with named group 'participant'. Default: 4 digits + 4 uppercase letters",
                    key="id_pattern_input",
                )

                # Real-time validation for regex pattern
                pattern_error = validate_regex_pattern(id_pattern)
                if pattern_error:
                    st.error(f"⚠️ Invalid regex: {pattern_error}")
                elif "(?P<participant>" not in id_pattern:
                    st.warning("⚠️ Pattern should include named group '(?P<participant>...)'")

            with col_cfg2:
                st.markdown("**RR Cleaning Thresholds**")

                def update_cleaning_config():
                    """Callback to update cleaning config and clear cache."""
                    st.session_state.cleaning_config = CleaningConfig(
                        rr_min_ms=st.session_state.rr_min_input,
                        rr_max_ms=st.session_state.rr_max_input,
                        sudden_change_pct=st.session_state.sudden_change_input,
                    )
                    cached_load_hrv_logger_preview.clear()

                col_rr1, col_rr2 = st.columns(2)
                with col_rr1:
                    st.number_input(
                        "Min RR (ms)",
                        min_value=200,
                        max_value=1000,
                        value=st.session_state.cleaning_config.rr_min_ms,
                        step=10,
                        key="rr_min_input",
                        on_change=update_cleaning_config,
                    )
                with col_rr2:
                    st.number_input(
                        "Max RR (ms)",
                        min_value=1000,
                        max_value=3000,
                        value=st.session_state.cleaning_config.rr_max_ms,
                        step=10,
                        key="rr_max_input",
                        on_change=update_cleaning_config,
                    )
                st.slider(
                    "Sudden change threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.cleaning_config.sudden_change_pct,
                    step=0.05,
                    format="%.2f",
                    key="sudden_change_input",
                    on_change=update_cleaning_config,
                )

        # Data directory input
        col1, col2 = st.columns([3, 1])
        with col1:
            data_dir_input = st.text_input(
                "Data directory path",
                value=st.session_state.data_dir or "data/raw/hrv_logger",
                help="Path to folder containing HRV Logger RR and Events CSV files",
            )
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("🔄 Load Data", type="primary", use_container_width=True):
                data_path = Path(data_dir_input).expanduser()
                if data_path.exists():
                    st.session_state.data_dir = str(data_path)

                    # Use status context for multi-step feedback
                    with st.status("Loading recordings...", expanded=True) as status:
                        try:
                            st.write("🔍 Discovering recordings...")

                            # Use cached version for faster loading
                            config_dict = {
                                "rr_min_ms": st.session_state.cleaning_config.rr_min_ms,
                                "rr_max_ms": st.session_state.cleaning_config.rr_max_ms,
                                "sudden_change_pct": st.session_state.cleaning_config.sudden_change_pct,
                            }
                            # ISSUE 1 FIX: Pass GUI events to cached function
                            summaries = cached_load_hrv_logger_preview(
                                str(data_path),
                                pattern=id_pattern,
                                config_dict=config_dict,
                                gui_events_dict=st.session_state.all_events,
                            )

                            st.write(f"📊 Processing {len(summaries)} participant(s)...")
                            st.session_state.summaries = summaries

                            # Auto-assign to Default group if not assigned
                            for summary in summaries:
                                if summary.participant_id not in st.session_state.participant_groups:
                                    st.session_state.participant_groups[summary.participant_id] = "Default"

                            status.update(label=f"✅ Loaded {len(summaries)} participant(s)", state="complete")
                            show_toast(f"Loaded {len(summaries)} participant(s)", icon="success")
                        except Exception as e:
                            status.update(label="❌ Error loading data", state="error")
                            st.error(f"Error loading data: {e}")
                else:
                    st.error(f"Directory not found: {data_path}")

        if st.session_state.summaries:
            st.markdown("---")

            # Participants overview table
            st.subheader("📋 Participants Overview")

            # Create editable dataframe
            participants_data = []
            loaded_participants = cached_load_participants()

            for summary in st.session_state.summaries:
                recording_dt_str = ""
                if summary.recording_datetime:
                    recording_dt_str = summary.recording_datetime.strftime("%Y-%m-%d %H:%M")

                # Show file counts (highlight if multiple files)
                files_str = f"{summary.rr_file_count}RR/{summary.events_file_count}Ev"
                if summary.has_multiple_files:
                    files_str = f"⚠️ {files_str}"

                participants_data.append({
                    "Participant": summary.participant_id,
                    "Saved": "✅" if summary.participant_id in loaded_participants else "❌",
                    "Files": files_str,
                    "Date/Time": recording_dt_str,
                    "Group": st.session_state.participant_groups.get(summary.participant_id, "Default"),
                    "Total Beats": summary.total_beats,
                    "Retained": summary.retained_beats,
                    "Duplicates": summary.duplicate_rr_intervals,
                    "Artifacts (%)": f"{summary.artifact_ratio * 100:.1f}",
                    "Duration (min)": f"{summary.duration_s / 60:.1f}",
                    "Events": summary.events_detected,
                    # ISSUE 1 FIX: Total Events = events_detected + duplicate_events (raw count)
                    "Total Events": summary.events_detected + summary.duplicate_events,
                    "Duplicate Events": summary.duplicate_events,
                    "RR Range (ms)": f"{int(summary.rr_min_ms)}–{int(summary.rr_max_ms)}",
                    "Mean RR (ms)": f"{summary.rr_mean_ms:.0f}",
                })

            df_participants = pd.DataFrame(participants_data)

            # Editable dataframe with better column config
            edited_df = st.data_editor(
                df_participants,
                column_config={
                    "Participant": st.column_config.TextColumn(
                        "Participant",
                        disabled=True,
                        width="medium",
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
                        help="RR files / Events files. ⚠️ indicates multiple files (merged from restarts)",
                    ),
                    "Group": st.column_config.SelectboxColumn(
                        "Group",
                        options=list(st.session_state.groups.keys()),
                        required=True,
                        help="Assign participant to a group (changes save automatically)",
                        width="medium",
                    ),
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
                    "Artifacts (%)": st.column_config.TextColumn(
                        "Artifacts (%)",
                        disabled=True,
                        width="small",
                    ),
                    # ISSUE 2 FIX: Add column config for new event columns
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
                use_container_width=True,
                hide_index=True,
                key="participants_table",
                disabled=["Participant", "Saved", "Date/Time", "Total Beats", "Retained", "Duplicates", "Artifacts (%)", "Duration (min)", "Events", "Total Events", "Duplicate Events", "RR Range (ms)", "Mean RR (ms)"]
            )

            # Auto-save group assignments when changed
            groups_changed = False
            for idx, row in edited_df.iterrows():
                participant_id = row["Participant"]
                new_group = row["Group"]
                old_group = st.session_state.participant_groups.get(participant_id)
                if old_group != new_group:
                    st.session_state.participant_groups[participant_id] = new_group
                    groups_changed = True

            # Auto-save if changes detected
            if groups_changed:
                save_participant_data()
                cached_load_participants.clear()
                show_toast("Group assignments saved", icon="success")

            # Show warning if any participant has duplicate RR intervals
            high_duplicates = [
                (row["Participant"], row["Duplicates"])
                for _, row in df_participants.iterrows()
                if row["Duplicates"] > 0
            ]
            if high_duplicates:
                st.warning(
                    f"⚠️ **Duplicate RR intervals detected!** "
                    f"{len(high_duplicates)} participant(s) have duplicate RR intervals that were removed. "
                    f"Check the 'Duplicates' column for details."
                )
                with st.expander("Show participants with duplicates"):
                    for pid, dup_count in high_duplicates:
                        st.text(f"• {pid}: {dup_count} duplicates removed")

            # Download button (save is now automatic)
            csv_participants = df_participants.to_csv(index=False)
            st.download_button(
                label="📥 Download Participants CSV",
                data=csv_participants,
                file_name="participants_overview.csv",
                mime="text/csv",
                use_container_width=False,
            )
            st.info("💡 **Tip:** Group assignments save automatically when you change them in the table above.")

            st.markdown("---")

            # Participant selector with easier navigation
            st.subheader("🔍 Participant Details")

            # ISSUE 2 FIX: Initialize and manage participant selection with bounds checking
            if "current_participant_idx" not in st.session_state:
                st.session_state.current_participant_idx = 0

            participant_list = [s.participant_id for s in st.session_state.summaries]

            # ISSUE 2 FIX: Ensure index is always valid BEFORE using it
            if st.session_state.current_participant_idx < 0:
                st.session_state.current_participant_idx = 0
            elif st.session_state.current_participant_idx >= len(participant_list):
                st.session_state.current_participant_idx = max(0, len(participant_list) - 1)

            # Callback for navigation buttons with toast
            def navigate_prev():
                st.session_state.current_participant_idx -= 1
                show_toast(f"Viewing participant {st.session_state.current_participant_idx + 1} of {len(participant_list)}", icon="info")

            def navigate_next():
                st.session_state.current_participant_idx += 1
                show_toast(f"Viewing participant {st.session_state.current_participant_idx + 1} of {len(participant_list)}", icon="info")

            col1, col2, col3 = st.columns([2, 1, 1])

            with col2:
                st.button(
                    "⬅️ Previous",
                    disabled=st.session_state.current_participant_idx == 0,
                    key="prev_btn",
                    on_click=navigate_prev,
                    use_container_width=True,
                )

            with col3:
                st.button(
                    "➡️ Next",
                    disabled=st.session_state.current_participant_idx == len(participant_list) - 1,
                    key="next_btn",
                    on_click=navigate_next,
                    use_container_width=True,
                )

            with col1:
                selected_participant = st.selectbox(
                    "Select participant",
                    options=participant_list,
                    index=st.session_state.current_participant_idx,
                    key="selected_participant_dropdown",
                    on_change=lambda: setattr(st.session_state, 'current_participant_idx',
                                              participant_list.index(st.session_state.selected_participant_dropdown))
                )

            # Show participant position with keyboard hints
            st.info(
                f"👤 **Viewing participant {st.session_state.current_participant_idx + 1} of {len(participant_list)}** | "
                f"Use ⬅️ Previous / Next ➡️ buttons or the dropdown to navigate"
            )

            if selected_participant:
                summary = next(
                    s for s in st.session_state.summaries
                    if s.participant_id == selected_participant
                )

                # Metrics row
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    st.metric("Total Beats", summary.total_beats)
                with col2:
                    st.metric("Retained", summary.retained_beats)
                with col3:
                    st.metric("Duplicates", summary.duplicate_rr_intervals)
                with col4:
                    st.metric("Artifacts", f"{summary.artifact_ratio * 100:.1f}%")
                with col5:
                    st.metric("Duration", f"{summary.duration_s / 60:.1f} min")
                with col6:
                    assigned_group = st.session_state.participant_groups.get(selected_participant, "Default")
                    st.metric("Group", assigned_group)

                # ISSUE 1 FIX: Show warning and expandable duplicate details if duplicates detected
                if summary.duplicate_rr_intervals > 0:
                    st.error(
                        f"⚠️ **{summary.duplicate_rr_intervals} duplicate RR intervals** were detected and removed! "
                        f"This participant may have corrupted data."
                    )

                    # ISSUE 1 FIX: Display duplicate details in expandable section
                    if summary.duplicate_details:
                        with st.expander(f"🔍 Show Duplicate Details ({len(summary.duplicate_details)} duplicates)"):
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
                    st.info(f"📅 Recording Date: {summary.recording_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

                # RR Interval Plot with Event Markers
                st.markdown("---")
                st.markdown("**📈 RR Interval Visualization with Event Markers:**")

                try:
                    # Load the recording to get full RR data and events
                    data_path = Path(st.session_state.data_dir)
                    bundles = discover_recordings(data_path, pattern=id_pattern)
                    bundle = next(b for b in bundles if b.participant_id == selected_participant)
                    recording, _, _ = load_recording(bundle)

                    from music_hrv.cleaning.rr import clean_rr_intervals

                    # Initialize session state for event management (needed for plot)
                    if "participant_events" not in st.session_state:
                        st.session_state.participant_events = {}

                    # Store events in session state for this participant if not already there
                    if selected_participant not in st.session_state.participant_events:
                        st.session_state.participant_events[selected_participant] = {
                            'events': list(summary.events),
                            'manual': st.session_state.manual_events.get(selected_participant, []).copy()
                        }

                    # Get cleaned RR intervals
                    cleaned_rr, stats = clean_rr_intervals(
                        recording.rr_intervals,
                        st.session_state.cleaning_config
                    )

                    if cleaned_rr and PLOTLY_AVAILABLE:
                        # Prepare data for plotting with REAL timestamps
                        # Only include RR intervals that have timestamps
                        rr_with_timestamps = [(rr.timestamp, rr.rr_ms) for rr in cleaned_rr if rr.timestamp]

                        # Only plot if we have timestamps
                        if not rr_with_timestamps:
                            st.warning("No timestamp data available for plotting")
                        else:
                            timestamps, rr_values = zip(*rr_with_timestamps)

                            # Create Plotly figure
                            fig = go.Figure()

                            # Add RR interval scatter plot (shows gaps automatically)
                            fig.add_trace(go.Scatter(
                                x=timestamps,
                                y=rr_values,
                                mode='markers+lines',
                                name='RR Intervals',
                                marker=dict(size=3, color='blue'),
                                line=dict(width=1, color='blue'),
                                hovertemplate='Time: %{x}<br>RR: %{y} ms<extra></extra>'
                            ))

                            # Get events from session state
                            stored_data = st.session_state.participant_events[selected_participant]
                            events_list = stored_data.get('events', [])
                            manual_list = stored_data.get('manual', [])
                            # Ensure both are lists before concatenating
                            if not isinstance(events_list, list):
                                events_list = []
                            if not isinstance(manual_list, list):
                                manual_list = []
                            current_events = events_list + manual_list

                            # Add event markers as vertical lines
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

                            # Draw event lines using shapes (add_vline doesn't work well with datetime)
                            y_min = min(rr_values)
                            y_max = max(rr_values)
                            y_range = y_max - y_min
                            for idx, (event_name, event_times) in enumerate(event_by_canonical.items()):
                                color = distinct_colors[idx % len(distinct_colors)]
                                for event_time in event_times:
                                    # Add vertical line as a shape
                                    fig.add_shape(
                                        type="line",
                                        x0=event_time, x1=event_time,
                                        y0=y_min - 0.05 * y_range, y1=y_max + 0.05 * y_range,
                                        line=dict(color=color, width=2, dash='dash'),
                                        opacity=0.7
                                    )
                                    # Add annotation at the top
                                    fig.add_annotation(
                                        x=event_time,
                                        y=y_max + 0.08 * y_range,
                                        text=event_name,
                                        showarrow=False,
                                        textangle=-90,
                                        font=dict(color=color, size=10)
                                    )

                            # Update layout for interactivity
                            fig.update_layout(
                                title=f"RR Intervals - {selected_participant} (Click to add events)",
                                xaxis=dict(
                                    title="Time",
                                    tickformat='%H:%M:%S',
                                    hoverformat='%H:%M:%S'
                                ),
                                yaxis=dict(title="RR Interval (ms)"),
                                hovermode='closest',
                                height=600,
                                showlegend=True,
                                legend=dict(x=1.02, y=1, xanchor='left', yanchor='top')
                            )

                            # Display interactive plot with click events
                            st.info("💡 Click on the plot to add a new event at that timestamp")
                            selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=600)

                            # Handle click events to add new event markers
                            if selected_points and len(selected_points) > 0:
                                clicked_point = selected_points[0]
                                if 'x' in clicked_point:
                                    # Convert x value back to timestamp (make timezone-aware like original data)
                                    from datetime import datetime, timezone
                                    clicked_timestamp = pd.to_datetime(clicked_point['x'])
                                    # Make timezone-aware (UTC) to match original event timestamps
                                    if clicked_timestamp.tzinfo is None:
                                        clicked_timestamp = clicked_timestamp.tz_localize('UTC')

                                    # Show form to add event
                                    with st.form(key=f"add_event_from_plot_{selected_participant}"):
                                        st.write(f"Add event at: {clicked_timestamp.strftime('%H:%M:%S')}")
                                        new_event_label = st.text_input("Event label:")
                                        submitted = st.form_submit_button("Add Event")

                                        if submitted and new_event_label:
                                            # Add to manual events
                                            from music_hrv.prep.summaries import EventStatus
                                            new_event = EventStatus(
                                                raw_label=new_event_label,
                                                canonical=st.session_state.normalizer.normalize(new_event_label),
                                                first_timestamp=clicked_timestamp,
                                                last_timestamp=clicked_timestamp
                                            )

                                            if selected_participant not in st.session_state.participant_events:
                                                st.session_state.participant_events[selected_participant] = {'events': [], 'manual': []}

                                            st.session_state.participant_events[selected_participant]['manual'].append(new_event)
                                            show_toast(f"✅ Added event '{new_event_label}' at {clicked_timestamp.strftime('%H:%M:%S')}", icon="success")
                                            st.rerun()

                except Exception as e:
                    st.warning(f"Could not generate RR plot: {e}")

                st.markdown("---")

                # Events table with reordering and inline editing
                st.markdown("**Events Detected:**")

                # Get events from session state (already initialized above for the plot)
                stored_data = st.session_state.participant_events[selected_participant]
                all_events = stored_data['events'] + stored_data['manual']

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
                        st.error("⚠️ **Events are NOT in chronological order!** Click 'Auto-Sort by Timestamp' to fix.")

                    # Add new event button
                    def add_new_event():
                        """Add a new manual event for this participant."""
                        from music_hrv.prep.summaries import EventStatus
                        import datetime
                        # Use timezone-aware datetime to match existing events
                        now = datetime.datetime.now(datetime.timezone.utc)
                        new_event = EventStatus(
                            raw_label="New Event",
                            canonical=None,
                            count=1,
                            first_timestamp=now,
                            last_timestamp=now,
                        )
                        # Add to participant events
                        st.session_state.participant_events[selected_participant]['manual'].append(new_event)
                        show_toast("New event added", icon="success")

                    st.button(
                        "➕ Add Event",
                        key=f"add_event_{selected_participant}",
                        on_click=add_new_event,
                    )

                    st.markdown("---")

                    # Build available canonical events
                    available_canonical_events = list(st.session_state.all_events.keys())

                    # Section 1: Event Editing - Individual Cards
                    st.markdown("### 📝 Event Management")
                    st.caption("Edit event details, match to canonical events, or delete events")

                    if all_events:
                        # Track if any changes were made
                        changes_made = False
                        events_to_delete = []

                        for idx, event in enumerate(all_events):
                            with st.container():
                                # Create columns for this event
                                col_status, col_raw, col_canonical, col_syn, col_time, col_delete = st.columns([0.5, 2.5, 2.5, 1, 1.5, 0.5])

                                with col_status:
                                    # Show mapping status - only green check if canonical is valid
                                    if event.canonical and event.canonical != "unmatched" and event.canonical in st.session_state.all_events:
                                        st.markdown("✓")
                                    else:
                                        st.markdown("🟡")

                                with col_raw:
                                    # Editable raw label with callback
                                    def update_raw_label():
                                        key = f"raw_{selected_participant}_{idx}"
                                        if key in st.session_state:
                                            new_val = st.session_state[key]
                                            stored_data = st.session_state.participant_events[selected_participant]
                                            all_evts = stored_data['events'] + stored_data['manual']
                                            all_evts[idx].raw_label = new_val
                                            st.session_state.participant_events[selected_participant]['events'] = all_evts
                                            st.session_state.participant_events[selected_participant]['manual'] = []

                                    st.text_input(
                                        "Raw Label",
                                        value=event.raw_label,
                                        key=f"raw_{selected_participant}_{idx}",
                                        label_visibility="collapsed",
                                        on_change=update_raw_label
                                    )

                                with col_canonical:
                                    # Canonical mapping dropdown with callback
                                    canonical_options = ["unmatched"] + available_canonical_events
                                    current_value = event.canonical if event.canonical else "unmatched"
                                    if current_value not in canonical_options:
                                        current_value = "unmatched"

                                    def update_canonical():
                                        key = f"canonical_{selected_participant}_{idx}"
                                        if key in st.session_state:
                                            new_val = st.session_state[key]
                                            stored_data = st.session_state.participant_events[selected_participant]
                                            all_evts = stored_data['events'] + stored_data['manual']
                                            all_evts[idx].canonical = new_val if new_val != "unmatched" else None
                                            st.session_state.participant_events[selected_participant]['events'] = all_evts
                                            st.session_state.participant_events[selected_participant]['manual'] = []

                                    st.selectbox(
                                        "Canonical",
                                        options=canonical_options,
                                        index=canonical_options.index(current_value),
                                        key=f"canonical_{selected_participant}_{idx}",
                                        label_visibility="collapsed",
                                        on_change=update_canonical
                                    )

                                with col_syn:
                                    # Add to synonyms button (only if canonical is selected)
                                    if event.canonical and event.canonical != "unmatched":
                                        def add_synonym(participant_id, event_idx):
                                            """Add raw label as synonym to canonical event."""
                                            # Get the CURRENT canonical value from the selectbox
                                            canonical_key = f"canonical_{participant_id}_{event_idx}"
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
                                                save_events(st.session_state.all_events)
                                                # Update normalizer after adding synonym
                                                update_normalizer()
                                                # Update the current event's canonical value in session state
                                                stored_data = st.session_state.participant_events[participant_id]
                                                current_events = stored_data['events'] + stored_data['manual']
                                                if event_idx < len(current_events):
                                                    current_events[event_idx].canonical = canonical_name
                                                    st.session_state.participant_events[participant_id]['events'] = current_events
                                                    st.session_state.participant_events[participant_id]['manual'] = []
                                                show_toast(f"✅ Added '{raw_lower}' as synonym for {canonical_name}", icon="success")
                                            else:
                                                show_toast(f"'{raw_lower}' is already a synonym for {canonical_name}", icon="info")

                                        st.button("➕🔖", key=f"syn_{selected_participant}_{idx}",
                                                 on_click=add_synonym, args=(selected_participant, idx),
                                                 help="Add raw label as synonym")

                                with col_time:
                                    # Display timestamp
                                    timestamp_str = event.first_timestamp.strftime("%H:%M:%S") if event.first_timestamp else "—"
                                    st.text_input(
                                        "Time",
                                        value=timestamp_str,
                                        key=f"time_{selected_participant}_{idx}",
                                        disabled=True,
                                        label_visibility="collapsed"
                                    )

                                with col_delete:
                                    # Delete button
                                    if st.button("🗑️", key=f"delete_{selected_participant}_{idx}", help="Delete this event"):
                                        events_to_delete.append(idx)
                                        changes_made = True

                                st.divider()

                        # Apply deletions
                        if events_to_delete:
                            for idx in sorted(events_to_delete, reverse=True):
                                del all_events[idx]
                            st.session_state.participant_events[selected_participant]['events'] = all_events
                            st.session_state.participant_events[selected_participant]['manual'] = []
                            st.rerun()

                    else:
                        st.info("No events found. Click 'Add Event' to create one.")

                    st.markdown("---")

                    # Refresh all_events from session state
                    stored_data = st.session_state.participant_events[selected_participant]
                    all_events = stored_data['events'] + stored_data['manual']

                    # Section 2: Event Order with Move Buttons
                    st.markdown("### 🔄 Event Order")
                    st.caption("Use ↑↓ buttons to reorder events - changes reflect immediately in all sections")

                    # Helper to normalize timestamps for comparison (handle timezone-aware vs naive)
                    def get_sort_key(event):
                        ts = event.first_timestamp
                        if ts is None:
                            return pd.Timestamp.max.tz_localize('UTC')
                        # Ensure all timestamps are timezone-aware for comparison
                        if hasattr(ts, 'tzinfo') and ts.tzinfo is None:
                            ts = pd.Timestamp(ts).tz_localize('UTC')
                        return ts

                    # Auto-sort button - use return value instead of on_click for proper rerun
                    if st.button("🔄 Auto-Sort by Timestamp", key=f"auto_sort_{selected_participant}"):
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

                                status_icon = "❌" if out_of_order else "✅"
                                mapping_badge = "🟡" if not event.canonical else "✓"
                                st.text(f"{status_icon}{mapping_badge}")

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

                    df_events = pd.DataFrame(events_data)
                    csv_events = df_events.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Events CSV",
                        data=csv_events,
                        file_name=f"events_{selected_participant}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    # Show unmatched warning
                    unmatched_count = sum(1 for e in all_events if not e.canonical)
                    if unmatched_count > 0:
                        st.warning(f"⚠️ {unmatched_count} unmatched event(s) - assign canonical mappings above")
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
                            "Status": "✅ Found" if matched else "❌ Missing",
                            "Raw Labels": raw_labels_str,
                        })

                    df_mapping = pd.DataFrame(mapping_data)
                    st.dataframe(df_mapping, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No expected events defined for group '{participant_group}'. Add them in the Event Mapping tab.")

                # Save participant button
                st.markdown("---")
                col_save, col_status = st.columns([1, 2])
                with col_save:
                    def save_participant_data():
                        """Save participant events to disk."""
                        # Get the current recording with updated events
                        stored_data = st.session_state.participant_events[selected_participant]
                        all_evts = stored_data['events'] + stored_data['manual']

                        # Find the original summary to get the recording
                        orig_summary = next((s for s in st.session_state.summaries if s.participant_id == selected_participant), None)
                        if orig_summary:
                            # Update summary events with current state
                            orig_summary.events = all_evts

                            # Save to output directory
                            output_dir = Path("output/participants")
                            output_dir.mkdir(parents=True, exist_ok=True)

                            # Convert EventStatus back to EventMarker for saving
                            from music_hrv.io.hrv_logger import EventMarker
                            event_markers = []
                            for evt in all_evts:
                                event_markers.append(EventMarker(
                                    label=evt.raw_label,
                                    timestamp=evt.first_timestamp
                                ))

                            # Get RR intervals from original recording
                            bundle = next((b for b in discover_recordings(st.session_state.data_path) if b.participant_id == selected_participant), None)
                            if bundle:
                                recording, _, _ = load_recording(bundle)
                                recording.events = event_markers

                                # Save
                                save_path = output_dir / f"{selected_participant}.pkl"
                                import pickle
                                with open(save_path, 'wb') as f:
                                    pickle.dump(recording, f)

                                show_toast(f"Saved {selected_participant} to {save_path}", icon="success")

                    st.button("💾 Save Participant Data",
                             key=f"save_{selected_participant}",
                             on_click=save_participant_data,
                             help="Save all event changes for this participant",
                             type="primary")

                with col_status:
                    # Check if participant is saved
                    saved_path = Path("output/participants") / f"{selected_participant}.pkl"
                    if saved_path.exists():
                        from datetime import datetime
                        mod_time = datetime.fromtimestamp(saved_path.stat().st_mtime)
                        st.caption(f"✅ Last saved: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        st.caption("⚠️ Not yet saved")

    # ================== TAB 2: Event Mapping ==================
    with tab2:
        st.header("🎭 Event Mapping")
        st.markdown("Define and manage all available events. These events can then be assigned to groups.")
        st.info("ℹ️ All event matching is done in **lowercase** automatically to reduce the number of synonyms needed.")

        # Create new event
        with st.expander("➕ Create New Event"):
            new_event_name = st.text_input("Event Name (canonical)", key="new_event_name_global")
            new_event_synonyms = st.text_area(
                "Synonyms (one per line, regex patterns supported)",
                key="new_event_synonyms_global",
                help="Enter regex patterns, one per line. All matching is lowercase. Example: ruhe[ _-]?pre[ _-]?start"
            )

            # Real-time validation of event name
            if new_event_name:
                if new_event_name in st.session_state.all_events:
                    st.warning(f"⚠️ Event '{new_event_name}' already exists")
                elif not new_event_name.replace("_", "").isalnum():
                    st.warning("⚠️ Event name should be alphanumeric with underscores")

            # Validate synonyms as regex patterns
            if new_event_synonyms:
                invalid_patterns = []
                for line in new_event_synonyms.split("\n"):
                    if line.strip():
                        error = validate_regex_pattern(line.strip())
                        if error:
                            invalid_patterns.append(f"'{line.strip()}': {error}")
                if invalid_patterns:
                    st.error("⚠️ Invalid regex patterns:\n" + "\n".join(invalid_patterns))

            def create_event():
                """Callback to create new event."""
                if new_event_name and new_event_name not in st.session_state.all_events:
                    synonyms_list = [s.strip().lower() for s in new_event_synonyms.split("\n") if s.strip()]
                    st.session_state.all_events[new_event_name] = synonyms_list
                    auto_save_config()
                    # ISSUE 1 FIX: Update normalizer when events change
                    update_normalizer()
                    show_toast(f"Created event '{new_event_name}'", icon="success")
                elif new_event_name in st.session_state.all_events:
                    show_toast(f"Event '{new_event_name}' already exists", icon="error")
                else:
                    show_toast("Please enter an event name", icon="error")

            st.button("Create Event", key="create_event_btn_global", on_click=create_event, type="primary")

        st.markdown("---")

        # ISSUE 5 & 6 FIX: Show all events with inline editing and synonym management
        st.subheader("📋 All Available Events")

        # Show count of events
        st.info(f"📊 **{len(st.session_state.all_events)} event(s) defined**")

        if st.session_state.all_events:
            for event_name, synonyms in list(st.session_state.all_events.items()):
                with st.expander(f"Event: {event_name} ({len(synonyms)} synonym(s))", expanded=False):
                    # Editable event name
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        new_event_name = st.text_input(
                            "Event Name",
                            value=event_name,
                            key=f"edit_event_name_{event_name}"
                        )

                    # Real-time validation
                    name_valid = True
                    if new_event_name != event_name:
                        if new_event_name in st.session_state.all_events:
                            st.warning(f"⚠️ Event '{new_event_name}' already exists")
                            name_valid = False
                        elif not new_event_name.replace("_", "").isalnum():
                            st.warning("⚠️ Event name should be alphanumeric with underscores")
                            name_valid = False

                    with col2:
                        def rename_event(old_name, new_name):
                            """Callback to rename event."""
                            if new_name != old_name and new_name not in st.session_state.all_events:
                                # Rename event
                                st.session_state.all_events[new_name] = st.session_state.all_events.pop(old_name)
                                # Update in all groups
                                for group_data in st.session_state.groups.values():
                                    if old_name in group_data.get("expected_events", {}):
                                        group_data["expected_events"][new_name] = group_data["expected_events"].pop(old_name)
                                auto_save_config()
                                # ISSUE 1 FIX: Update normalizer when events change
                                update_normalizer()
                                show_toast(f"Renamed to '{new_name}'", icon="success")
                            elif new_name == old_name:
                                show_toast("Name unchanged", icon="info")
                            else:
                                show_toast(f"Event '{new_name}' already exists", icon="error")

                        st.button(
                            "💾 Save Name",
                            key=f"save_event_name_{event_name}",
                            on_click=rename_event,
                            args=(event_name, new_event_name),
                            disabled=not name_valid or new_event_name == event_name,
                            use_container_width=True,
                        )

                    # Show used in groups
                    used_in_groups = [
                        gname for gname, gdata in st.session_state.groups.items()
                        if event_name in gdata.get("expected_events", {})
                    ]
                    if used_in_groups:
                        st.info(f"Used in groups: {', '.join(used_in_groups)}")
                    else:
                        st.info("Not used in any groups yet")

                    st.markdown("---")
                    st.markdown("**Synonyms:**")

                    # Display and manage synonyms
                    if synonyms:
                        for syn_idx, synonym in enumerate(synonyms):
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.text(synonym)
                            with col2:
                                # Optimized synonym deletion with callback
                                def delete_synonym(evt_name, idx):
                                    """Callback to delete synonym."""
                                    syn_list = st.session_state.all_events[evt_name]
                                    syn_list.pop(idx)
                                    st.session_state.all_events[evt_name] = syn_list
                                    # Update in all groups that use this event
                                    for group_data in st.session_state.groups.values():
                                        if evt_name in group_data.get("expected_events", {}):
                                            group_data["expected_events"][evt_name] = syn_list.copy()
                                    auto_save_config()
                                    # ISSUE 1 FIX: Update normalizer when synonyms change
                                    update_normalizer()
                                    show_toast("Synonym deleted", icon="success")

                                st.button(
                                    "🗑️",
                                    key=f"delete_syn_{event_name}_{syn_idx}",
                                    on_click=delete_synonym,
                                    args=(event_name, syn_idx),
                                    help="Delete this synonym",
                                )
                    else:
                        st.info("No synonyms defined")

                    # Add new synonym
                    st.markdown("**Add New Synonym:**")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        new_synonym = st.text_input(
                            "New synonym (regex pattern)",
                            key=f"new_syn_{event_name}",
                            placeholder="e.g., rest[ _-]?start"
                        )

                        # Real-time validation of synonym
                        if new_synonym:
                            error = validate_regex_pattern(new_synonym.strip())
                            if error:
                                st.error(f"⚠️ Invalid regex: {error}")
                            elif new_synonym.strip().lower() in synonyms:
                                st.warning("⚠️ This synonym already exists")

                    with col2:
                        st.write("")  # Spacer
                        st.write("")  # Spacer

                        # Optimized synonym addition with callback
                        def add_synonym(evt_name, new_syn):
                            """Callback to add synonym."""
                            synonym_lower = new_syn.strip().lower()
                            syn_list = st.session_state.all_events[evt_name]
                            if synonym_lower and synonym_lower not in syn_list:
                                syn_list.append(synonym_lower)
                                st.session_state.all_events[evt_name] = syn_list
                                # Update in all groups that use this event
                                for group_data in st.session_state.groups.values():
                                    if evt_name in group_data.get("expected_events", {}):
                                        group_data["expected_events"][evt_name] = syn_list.copy()
                                auto_save_config()
                                # ISSUE 1 FIX: Update normalizer when synonyms change
                                update_normalizer()
                                show_toast(f"Added '{synonym_lower}'", icon="success")
                            elif synonym_lower in syn_list:
                                show_toast("Synonym already exists", icon="warning")
                            else:
                                show_toast("Please enter a synonym", icon="error")

                        st.button(
                            "➕ Add",
                            key=f"add_syn_btn_{event_name}",
                            on_click=add_synonym,
                            args=(event_name, new_synonym),
                            type="primary",
                            disabled=not new_synonym or validate_regex_pattern(new_synonym.strip()) is not None,
                        )

                    # Delete event
                    st.markdown("---")

                    def delete_event(evt_name):
                        """Callback to delete event."""
                        # Remove from all events
                        del st.session_state.all_events[evt_name]
                        # Remove from all groups
                        for group_data in st.session_state.groups.values():
                            if evt_name in group_data.get("expected_events", {}):
                                del group_data["expected_events"][evt_name]
                        auto_save_config()
                        # ISSUE 1 FIX: Update normalizer when events change
                        update_normalizer()
                        show_toast(f"Deleted event '{evt_name}'", icon="success")

                    st.button(
                        f"🗑️ Delete Event '{event_name}'",
                        key=f"delete_event_{event_name}",
                        on_click=delete_event,
                        args=(event_name,),
                        type="secondary",
                    )
        else:
            st.info("No events defined yet. Create events above.")

    # ================== TAB 3: Group Management ==================
    with tab3:
        st.header("👥 Group Management")
        st.markdown("Create groups, edit/rename/delete them, and assign events from the Event Mapping tab.")

        # Create new group
        with st.expander("➕ Create New Group"):
            new_group_name = st.text_input("Group Name (internal ID)", key="new_group_name")
            new_group_label = st.text_input("Group Label (display name)", key="new_group_label")

            # Real-time validation
            if new_group_name:
                if new_group_name in st.session_state.groups:
                    st.warning(f"⚠️ Group '{new_group_name}' already exists")
                elif not new_group_name.replace("_", "").replace("-", "").isalnum():
                    st.warning("⚠️ Group name should be alphanumeric with underscores/hyphens")

            def create_group():
                """Callback to create new group."""
                if new_group_name and new_group_name not in st.session_state.groups:
                    st.session_state.groups[new_group_name] = {
                        "label": new_group_label or new_group_name,
                        "expected_events": {},
                        "selected_sections": []
                    }
                    auto_save_config()
                    show_toast(f"Created group '{new_group_name}'", icon="success")
                elif new_group_name in st.session_state.groups:
                    show_toast(f"Group '{new_group_name}' already exists", icon="error")
                else:
                    show_toast("Please enter a group name", icon="error")

            st.button("Create Group", key="create_group_btn", on_click=create_group, type="primary")

        st.markdown("---")

        # Manage existing groups
        st.subheader("Existing Groups")

        for group_name, group_data in list(st.session_state.groups.items()):
            with st.expander(f"📂 {group_name} - {group_data['label']}", expanded=(group_name == "Default")):

                # Edit group name and label
                st.markdown("**Edit Group:**")
                col1, col2 = st.columns(2)
                with col1:
                    new_name = st.text_input(
                        "Group Name (ID)",
                        value=group_name,
                        key=f"edit_name_{group_name}"
                    )
                with col2:
                    new_label = st.text_input(
                        "Group Label",
                        value=group_data["label"],
                        key=f"edit_label_{group_name}"
                    )

                def save_group_changes(old_name, new_name_val, new_label_val):
                    """Callback to save group changes."""
                    # Update group data
                    current_name = old_name
                    if new_name_val != old_name:
                        # Rename group
                        st.session_state.groups[new_name_val] = st.session_state.groups.pop(old_name)
                        # Update participant assignments
                        for pid, gname in st.session_state.participant_groups.items():
                            if gname == old_name:
                                st.session_state.participant_groups[pid] = new_name_val
                        current_name = new_name_val

                    st.session_state.groups[current_name]["label"] = new_label_val
                    auto_save_config()
                    show_toast(f"Saved changes to '{current_name}'", icon="success")

                st.button(
                    f"💾 Save Changes to {group_name}",
                    key=f"save_group_{group_name}",
                    on_click=save_group_changes,
                    args=(group_name, new_name, new_label),
                    type="primary",
                )

                st.markdown("---")

                # Show group info
                participant_count = sum(1 for g in st.session_state.participant_groups.values() if g == group_name)
                st.markdown(f"**Participants in this group:** {participant_count}")

                # ISSUE 7 FIX: Add sections selection for each group with auto-save
                st.markdown("**Select Sections for Analysis:**")
                available_sections = list(st.session_state.sections.keys()) if hasattr(st.session_state, 'sections') else []
                if available_sections:
                    def update_sections():
                        """Callback to update sections selection."""
                        st.session_state.groups[group_name]["selected_sections"] = st.session_state[f"sections_select_{group_name}"]
                        auto_save_config()
                        show_toast(f"Sections updated for {group_name}", icon="success")

                    selected_sections = st.multiselect(
                        "Sections to use in analysis",
                        options=available_sections,
                        default=group_data.get("selected_sections", []),
                        key=f"sections_select_{group_name}",
                        help="Choose which sections to analyze for participants in this group (saves automatically)",
                        on_change=update_sections,
                    )
                else:
                    st.info("No sections defined yet. Create sections in the Sections tab first.")

                st.markdown("---")

                # Expected events for this group with auto-save
                st.markdown("**Select Expected Events:**")

                expected_events = group_data.get("expected_events", {})

                # Show checkboxes for all available events
                st.markdown("*Click events to add/remove from this group (saves automatically):*")

                # Create columns for better layout
                num_cols = 3
                cols = st.columns(num_cols)

                available_event_names = list(st.session_state.all_events.keys())
                for idx, event_name in enumerate(available_event_names):
                    col_idx = idx % num_cols
                    with cols[col_idx]:
                        is_selected = event_name in expected_events

                        def toggle_event(grp_name, evt_name, currently_selected):
                            """Callback to toggle event selection."""
                            exp_events = st.session_state.groups[grp_name]["expected_events"]
                            if st.session_state[f"event_select_{grp_name}_{evt_name}"]:
                                # Event selected - add to group
                                if evt_name not in exp_events:
                                    exp_events[evt_name] = st.session_state.all_events[evt_name].copy()
                                    st.session_state.groups[grp_name]["expected_events"] = exp_events
                                    auto_save_config()
                                    show_toast(f"Added '{evt_name}' to {grp_name}", icon="success")
                            else:
                                # Event deselected - remove from group
                                if evt_name in exp_events:
                                    del exp_events[evt_name]
                                    st.session_state.groups[grp_name]["expected_events"] = exp_events
                                    auto_save_config()
                                    show_toast(f"Removed '{evt_name}' from {grp_name}", icon="info")

                        st.checkbox(
                            event_name,
                            value=is_selected,
                            key=f"event_select_{group_name}_{event_name}",
                            on_change=toggle_event,
                            args=(group_name, event_name, is_selected),
                        )

                st.markdown("---")

                # Show currently selected events with synonyms
                if expected_events:
                    st.markdown("**Currently Selected Events:**")
                    events_list = []
                    for event_name, synonyms in expected_events.items():
                        events_list.append({
                            "Event Name": event_name,
                            "Synonyms": ", ".join(synonyms[:3]) + ("..." if len(synonyms) > 3 else "") if synonyms else "No synonyms",
                        })

                    df_group_events = pd.DataFrame(events_list)
                    st.dataframe(df_group_events, use_container_width=True, hide_index=True)

                    # Download group events
                    csv_group_events = df_group_events.to_csv(index=False)
                    st.download_button(
                        label=f"📥 Download Events for {group_name}",
                        data=csv_group_events,
                        file_name=f"group_events_{group_name}.csv",
                        mime="text/csv",
                        key=f"download_group_{group_name}"
                    )
                else:
                    st.info("No events selected for this group yet. Select events above.")

                # Delete group button
                st.markdown("---")

                def delete_group(grp_name):
                    """Callback to delete group."""
                    # Reassign participants to Default
                    for pid, gname in st.session_state.participant_groups.items():
                        if gname == grp_name:
                            st.session_state.participant_groups[pid] = "Default"
                    del st.session_state.groups[grp_name]
                    auto_save_config()
                    show_toast(f"Deleted group '{grp_name}' and reassigned participants to Default", icon="success")

                st.button(
                    f"🗑️ Delete Group '{group_name}'",
                    key=f"delete_group_{group_name}",
                    on_click=delete_group,
                    args=(group_name,),
                    type="secondary",
                )

        # Info about auto-save at bottom
        st.markdown("---")
        st.info("💡 **All changes save automatically** when you modify group settings, select events, or assign participants.")

    # ================== TAB 4: Sections ==================
    with tab4:
        st.header("📐 Sections")
        st.markdown("Define time ranges (sections) between events for analysis. Each section has a start and end event.")

        # Initialize sections if not present
        if "sections" not in st.session_state:
            loaded_sections = load_sections()
            if not loaded_sections:
                # Initialize default sections
                st.session_state.sections = {
                    "rest_pre": {
                        "label": "Pre-Rest",
                        "start_event": "rest_pre_start",
                        "end_event": "rest_pre_end",
                    },
                    "measurement": {
                        "label": "Measurement",
                        "start_event": "measurement_start",
                        "end_event": "measurement_end",
                    },
                    "pause": {
                        "label": "Pause",
                        "start_event": "pause_start",
                        "end_event": "pause_end",
                    },
                    "rest_post": {
                        "label": "Post-Rest",
                        "start_event": "rest_post_start",
                        "end_event": "rest_post_end",
                    },
                }
            else:
                st.session_state.sections = loaded_sections

        # Create new section
        with st.expander("➕ Create New Section"):
            new_section_name = st.text_input("Section Name (internal ID)", key="new_section_name")
            new_section_label = st.text_input("Section Label (display name)", key="new_section_label")

            col1, col2 = st.columns(2)
            with col1:
                available_events = list(st.session_state.all_events.keys())
                start_event = st.selectbox(
                    "Start Event",
                    options=available_events,
                    key="new_section_start"
                )
            with col2:
                end_event = st.selectbox(
                    "End Event",
                    options=available_events,
                    key="new_section_end"
                )

            # Real-time validation
            if new_section_name:
                if new_section_name in st.session_state.sections:
                    st.warning(f"⚠️ Section '{new_section_name}' already exists")
                elif not new_section_name.replace("_", "").isalnum():
                    st.warning("⚠️ Section name should be alphanumeric with underscores")

            def create_section():
                """Callback to create section."""
                if new_section_name and new_section_name not in st.session_state.sections:
                    st.session_state.sections[new_section_name] = {
                        "label": new_section_label or new_section_name,
                        "start_event": start_event,
                        "end_event": end_event,
                    }
                    auto_save_config()
                    show_toast(f"Created section '{new_section_name}'", icon="success")
                elif new_section_name in st.session_state.sections:
                    show_toast(f"Section '{new_section_name}' already exists", icon="error")
                else:
                    show_toast("Please enter a section name", icon="error")

            st.button("Create Section", key="create_section_btn", on_click=create_section, type="primary")

        st.markdown("---")

        # Show all sections
        st.subheader("📋 All Defined Sections")

        if st.session_state.sections:
            sections_list = []
            for section_name, section_data in st.session_state.sections.items():
                sections_list.append({
                    "Section Name": section_name,
                    "Label": section_data.get("label", section_name),
                    "Start Event": section_data.get("start_event", ""),
                    "End Event": section_data.get("end_event", ""),
                })

            df_sections = pd.DataFrame(sections_list)

            available_events = list(st.session_state.all_events.keys())
            edited_sections = st.data_editor(
                df_sections,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                key="sections_table",
                column_config={
                    "Start Event": st.column_config.SelectboxColumn(
                        "Start Event",
                        options=available_events,
                        required=True,
                    ),
                    "End Event": st.column_config.SelectboxColumn(
                        "End Event",
                        options=available_events,
                        required=True,
                    ),
                }
            )

            # Save changes button with callback
            def save_section_changes():
                """Callback to save section changes."""
                updated_sections = {}
                for _, row in edited_sections.iterrows():
                    section_name = row["Section Name"]
                    updated_sections[section_name] = {
                        "label": row["Label"],
                        "start_event": row["Start Event"],
                        "end_event": row["End Event"],
                    }

                st.session_state.sections = updated_sections
                auto_save_config()
                show_toast("Saved section changes", icon="success")

            st.button(
                "💾 Save Section Changes",
                key="save_sections_btn",
                on_click=save_section_changes,
                type="primary",
            )

            # Download sections
            csv_sections = df_sections.to_csv(index=False)
            st.download_button(
                label="📥 Download Sections CSV",
                data=csv_sections,
                file_name="sections.csv",
                mime="text/csv",
                key="download_sections"
            )
        else:
            st.info("No sections defined yet. Create sections above.")

    # ================== TAB 5: Analysis ==================
    with tab5:
        st.header("📊 HRV Analysis with NeuroKit2")

        if not NEUROKIT_AVAILABLE:
            st.error("❌ NeuroKit2 is not installed. Please install it to use HRV analysis features.")
            st.code("uv add neurokit2")
            return

        if not st.session_state.summaries:
            st.info("📊 Load data from the 'Data & Groups' tab to perform analysis")
        else:
            st.markdown("Select a participant, choose multiple sections, and analyze HRV metrics for each section individually and combined.")

            # Initialize analysis results in session state
            if "analysis_results" not in st.session_state:
                st.session_state.analysis_results = {}

            # Selection mode
            analysis_mode = st.radio(
                "Analysis Mode",
                options=["Single Participant", "Group Analysis"],
                horizontal=True,
            )

            if analysis_mode == "Single Participant":
                # Participant selection
                participant_list = [s.participant_id for s in st.session_state.summaries]
                selected_participant = st.selectbox(
                    "Select Participant",
                    options=participant_list,
                    key="analysis_participant"
                )

                # Section selection
                available_sections = list(st.session_state.sections.keys())
                if not available_sections:
                    st.warning("⚠️ No sections defined. Please define sections in the Sections tab first.")
                else:
                    selected_sections = st.multiselect(
                        "Select Sections to Analyze",
                        options=available_sections,
                        default=[available_sections[0]] if available_sections else [],
                        key="analysis_sections_single"
                    )

                    if st.button("🔬 Analyze HRV", key="analyze_single_btn", type="primary"):
                        if not selected_sections:
                            st.error("Please select at least one section")
                        else:
                            # Use status context for multi-step analysis
                            with st.status("Analyzing HRV for selected sections...", expanded=True) as status:
                                try:
                                    st.write("📂 Loading recording data...")
                                    progress = st.progress(0)
                                    # Load the recording
                                    data_path = Path(st.session_state.data_dir)
                                    bundles = discover_recordings(data_path, pattern=id_pattern)
                                    bundle = next(b for b in bundles if b.participant_id == selected_participant)
                                    recording, _, _ = load_recording(bundle)
                                    progress.progress(20)

                                    from music_hrv.cleaning.rr import clean_rr_intervals

                                    # Store results for each section
                                    section_results = {}
                                    combined_rr = []

                                    st.write(f"🔬 Analyzing {len(selected_sections)} section(s)...")

                                    # Analyze each section individually
                                    for idx, section_name in enumerate(selected_sections):
                                        progress.progress(20 + int((idx / len(selected_sections)) * 60))
                                        st.write(f"  • Processing section: {section_name}")

                                        section_def = st.session_state.sections[section_name]
                                        section_rr = extract_section_rr_intervals(
                                            recording, section_def, st.session_state.normalizer
                                        )

                                        if section_rr:
                                            # Clean RR intervals for this section
                                            cleaned_section_rr, stats = clean_rr_intervals(
                                                section_rr, st.session_state.cleaning_config
                                            )

                                            if cleaned_section_rr:
                                                rr_ms = [rr.rr_ms for rr in cleaned_section_rr]
                                                combined_rr.extend(rr_ms)

                                                # Calculate HRV metrics
                                                peaks = nk.intervals_to_peaks(rr_ms, sampling_rate=1000)
                                                hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
                                                hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)
                                                hrv_results = pd.concat([hrv_time, hrv_freq], axis=1)

                                                section_results[section_name] = {
                                                    "hrv_results": hrv_results,
                                                    "rr_intervals": rr_ms,
                                                    "n_beats": len(rr_ms),
                                                    "label": section_def.get("label", section_name),
                                                }
                                        else:
                                            st.write(f"  ⚠️ Could not find events for section '{section_name}'")

                                    # Analyze combined sections if multiple selected
                                    if len(selected_sections) > 1 and combined_rr:
                                        progress.progress(80)
                                        st.write("📊 Computing combined analysis...")
                                        peaks = nk.intervals_to_peaks(combined_rr, sampling_rate=1000)
                                        hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
                                        hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)
                                        combined_hrv = pd.concat([hrv_time, hrv_freq], axis=1)
                                        section_results["_combined"] = {
                                            "hrv_results": combined_hrv,
                                            "rr_intervals": combined_rr,
                                            "n_beats": len(combined_rr),
                                            "label": "Combined Sections",
                                        }

                                    # Store in session state
                                    progress.progress(100)
                                    st.session_state.analysis_results[selected_participant] = section_results

                                    status.update(label=f"✅ Analysis complete for {len(section_results)} section(s)!", state="complete")
                                    show_toast(f"Analysis complete for {len(section_results)} section(s)", icon="success")

                                except Exception as e:
                                    status.update(label="❌ Error during analysis", state="error")
                                    st.error(f"Error during analysis: {e}")
                                    import traceback
                                    st.code(traceback.format_exc())

                    # Display results if available
                    if selected_participant in st.session_state.analysis_results:
                        st.markdown("---")
                        st.subheader(f"📊 Results for {selected_participant}")

                        section_results = st.session_state.analysis_results[selected_participant]

                        for section_name, result_data in section_results.items():
                            section_label = result_data["label"]
                            hrv_results = result_data["hrv_results"]
                            rr_intervals = result_data["rr_intervals"]
                            n_beats = result_data["n_beats"]

                            with st.expander(f"📈 {section_label} ({n_beats} beats)", expanded=True):
                                # Key metrics
                                if not hrv_results.empty:
                                    metrics_to_show = {
                                        "HRV_RMSSD": "RMSSD (ms)",
                                        "HRV_SDNN": "SDNN (ms)",
                                        "HRV_pNN50": "pNN50 (%)",
                                        "HRV_HF": "HF Power",
                                        "HRV_LF": "LF Power",
                                        "HRV_LFHF": "LF/HF Ratio",
                                    }

                                    cols = st.columns(3)
                                    for idx, (col_name, display_name) in enumerate(metrics_to_show.items()):
                                        if col_name in hrv_results.columns:
                                            value = hrv_results[col_name].iloc[0]
                                            with cols[idx % 3]:
                                                st.metric(display_name, f"{value:.2f}")

                                    # Full results table
                                    st.markdown("**Full HRV Metrics:**")
                                    st.dataframe(hrv_results.T, use_container_width=True)

                                    # Download button for this section
                                    csv_hrv = hrv_results.to_csv(index=True)
                                    st.download_button(
                                        label=f"📥 Download {section_label} Results",
                                        data=csv_hrv,
                                        file_name=f"hrv_{selected_participant}_{section_name}.csv",
                                        mime="text/csv",
                                        key=f"download_{selected_participant}_{section_name}",
                                    )

                                    # RR interval plot for this section
                                    st.markdown("**RR Interval Plot:**")
                                    fig, ax = plt.subplots(figsize=(12, 4))
                                    ax.plot(rr_intervals, marker='o', markersize=2, linestyle='-', linewidth=0.5)
                                    ax.set_xlabel("Beat Index")
                                    ax.set_ylabel("RR Interval (ms)")
                                    ax.set_title(f"RR Intervals - {section_label}")
                                    ax.grid(True, alpha=0.3)
                                    st.pyplot(fig)
                                    plt.close(fig)

            else:  # Group Analysis
                # Group selection
                group_list = list(st.session_state.groups.keys())
                selected_group = st.selectbox(
                    "Select Group",
                    options=group_list,
                    key="analysis_group"
                )

                # Section selection
                available_sections = list(st.session_state.sections.keys())
                if not available_sections:
                    st.warning("⚠️ No sections defined. Please define sections in the Sections tab first.")
                else:
                    selected_sections = st.multiselect(
                        "Select Sections to Analyze",
                        options=available_sections,
                        default=[available_sections[0]] if available_sections else [],
                        key="analysis_sections_group"
                    )

                    if st.button("🔬 Analyze Group HRV", key="analyze_group_btn", type="primary"):
                        if not selected_sections:
                            st.error("Please select at least one section")
                        else:
                            # Get participants in selected group
                            group_participants = [
                                pid for pid, gname in st.session_state.participant_groups.items()
                                if gname == selected_group
                            ]

                            if not group_participants:
                                st.warning(f"No participants assigned to group '{selected_group}'")
                            else:
                                # Use status context for group analysis
                                with st.status(f"Analyzing {len(group_participants)} participants...", expanded=True) as status:
                                    from music_hrv.cleaning.rr import clean_rr_intervals
                                    data_path = Path(st.session_state.data_dir)
                                    bundles = discover_recordings(data_path, pattern=id_pattern)

                                    # Results organized by section
                                    results_by_section = {section: [] for section in selected_sections}
                                    if len(selected_sections) > 1:
                                        results_by_section["_combined"] = []

                                    progress = st.progress(0)
                                    total_steps = len(group_participants)

                                    for idx, participant_id in enumerate(group_participants):
                                        st.write(f"📊 Processing {participant_id} ({idx + 1}/{total_steps})")
                                        progress.progress(int((idx / total_steps) * 100))
                                        try:
                                            bundle = next(b for b in bundles if b.participant_id == participant_id)
                                            recording, _, _ = load_recording(bundle)

                                            combined_rr = []

                                            # Analyze each section
                                            for section_name in selected_sections:
                                                section_def = st.session_state.sections[section_name]
                                                section_rr = extract_section_rr_intervals(
                                                    recording, section_def, st.session_state.normalizer
                                                )

                                                if section_rr:
                                                    cleaned_rr, stats = clean_rr_intervals(
                                                        section_rr, st.session_state.cleaning_config
                                                    )

                                                    if cleaned_rr:
                                                        rr_ms = [rr.rr_ms for rr in cleaned_rr]
                                                        combined_rr.extend(rr_ms)

                                                        peaks = nk.intervals_to_peaks(rr_ms, sampling_rate=1000)
                                                        hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
                                                        hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)
                                                        hrv_results = pd.concat([hrv_time, hrv_freq], axis=1)

                                                        if not hrv_results.empty:
                                                            result_row = {"participant_id": participant_id}
                                                            for col in hrv_results.columns:
                                                                result_row[col] = hrv_results[col].iloc[0]
                                                            results_by_section[section_name].append(result_row)

                                            # Combined analysis
                                            if len(selected_sections) > 1 and combined_rr:
                                                peaks = nk.intervals_to_peaks(combined_rr, sampling_rate=1000)
                                                hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
                                                hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)
                                                combined_hrv = pd.concat([hrv_time, hrv_freq], axis=1)

                                                if not combined_hrv.empty:
                                                    result_row = {"participant_id": participant_id}
                                                    for col in combined_hrv.columns:
                                                        result_row[col] = combined_hrv[col].iloc[0]
                                                    results_by_section["_combined"].append(result_row)

                                        except Exception as e:
                                            st.write(f"  ⚠️ Could not analyze {participant_id}: {e}")

                                    # Complete
                                    progress.progress(100)
                                    status.update(label="✅ Group analysis complete!", state="complete")
                                    show_toast(f"Group analysis complete for {len(group_participants)} participants", icon="success")

                                    # Display results by section
                                    st.subheader(f"Group HRV Results - {selected_group}")

                                    for section_name, results in results_by_section.items():
                                        if results:
                                            section_label = (
                                                "Combined Sections"
                                                if section_name == "_combined"
                                                else st.session_state.sections[section_name].get("label", section_name)
                                            )

                                            with st.expander(f"📊 {section_label} ({len(results)} participants)", expanded=True):
                                                df_results = pd.DataFrame(results)

                                                # Summary statistics
                                                st.markdown("**Summary Statistics:**")
                                                st.dataframe(df_results.describe(), use_container_width=True)

                                                # Individual results
                                                st.markdown("**Individual Results:**")
                                                st.dataframe(df_results, use_container_width=True)

                                                # Download
                                                csv_data = df_results.to_csv(index=False)
                                                st.download_button(
                                                    label=f"📥 Download {section_label} Results",
                                                    data=csv_data,
                                                    file_name=f"hrv_group_{selected_group}_{section_name}.csv",
                                                    mime="text/csv",
                                                    key=f"download_group_{section_name}",
                                                )


if __name__ == "__main__":
    main()
