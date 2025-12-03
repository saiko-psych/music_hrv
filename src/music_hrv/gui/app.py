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
    load_playlist_groups,
    save_playlist_groups,
    load_music_labels,
)
from music_hrv.gui.tabs.setup import render_setup_tab
from music_hrv.gui.tabs.data import render_data_tab

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

try:
    import plotly.graph_objects as go
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

# Initialize sections at startup (so Analysis tab can use them before Setup is visited)
if "sections" not in st.session_state:
    loaded_sections = load_sections()
    if not loaded_sections:
        st.session_state.sections = {
            "rest_pre": {"label": "Pre-Rest", "description": "Baseline rest period", "start_event": "rest_pre_start", "end_event": "rest_pre_end"},
            "measurement": {"label": "Measurement", "description": "Main measurement period", "start_event": "measurement_start", "end_event": "measurement_end"},
            "pause": {"label": "Pause", "description": "Break between blocks", "start_event": "pause_start", "end_event": "pause_end"},
            "rest_post": {"label": "Post-Rest", "description": "Post-measurement rest", "start_event": "rest_post_start", "end_event": "rest_post_end"},
        }
    else:
        st.session_state.sections = loaded_sections

# ISSUE 1 FIX: Create normalizer from GUI events only (not from sections.yml)
if "normalizer" not in st.session_state:
    st.session_state.normalizer = create_gui_normalizer(st.session_state.all_events)

# Load participant-specific data (groups, randomizations, event orders, manual events)
if "participant_groups" not in st.session_state or "event_order" not in st.session_state:
    loaded_participants = load_participants()
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
    """Save participant-specific data (groups, randomizations, event orders, manual events)."""
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
    from music_hrv.io.hrv_logger import RecordingBundle
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


@st.cache_data(show_spinner=False, ttl=300)
def cached_clean_rr_intervals(rr_data_tuple, config_dict):
    """Cache cleaned RR intervals to avoid recomputation."""
    from music_hrv.cleaning.rr import clean_rr_intervals, RRInterval
    # Reconstruct RR intervals from cached data
    rr_intervals = [RRInterval(timestamp=ts, rr_ms=rr, elapsed_ms=elapsed)
                    for ts, rr, elapsed in rr_data_tuple]
    config = CleaningConfig(
        rr_min_ms=config_dict["rr_min_ms"],
        rr_max_ms=config_dict["rr_max_ms"],
        sudden_change_pct=config_dict["sudden_change_pct"]
    )
    cleaned, stats = clean_rr_intervals(rr_intervals, config)
    # Return as serializable tuples
    return [(rr.timestamp, rr.rr_ms) for rr in cleaned if rr.timestamp], stats


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
        issues.append(f"🔴 **{high_artifact}** participant(s) with high artifact rates (>15%)")

    with_duplicates = sum(1 for s in summaries_data if s["duplicate_rr_intervals"] > 0)
    if with_duplicates:
        issues.append(f"⚠️ **{with_duplicates}** participant(s) with duplicate RR intervals")

    with_multi_files = sum(1 for s in summaries_data
                          if s["rr_file_count"] > 1 or s["events_file_count"] > 1)
    if with_multi_files:
        issues.append(f"📁 **{with_multi_files}** participant(s) with multiple files (merged)")

    no_events = sum(1 for s in summaries_data if s["events_detected"] == 0)
    if no_events:
        issues.append(f"❓ **{no_events}** participant(s) with no events detected")

    # Build participant table data
    participants_data = []
    for s in summaries_data:
        recording_dt_str = s["recording_datetime_str"]

        rr_count = s["rr_file_count"]
        ev_count = s["events_file_count"]
        files_str = f"{rr_count}RR/{ev_count}Ev"
        if rr_count > 1 or ev_count > 1:
            files_str = f"⚠️ {files_str}"

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
            "Artifacts (%)": f"{s['artifact_ratio'] * 100:.1f}",
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
def cached_get_plot_data(timestamps_tuple, rr_values_tuple, participant_id: str, downsample_threshold: int = 5000):
    """Cache processed plot data (NOT the figure - that's slow to serialize).

    Downsamples data if too many points for faster rendering.
    Returns the data needed to build the plot quickly.
    """
    timestamps = list(timestamps_tuple)
    rr_values = list(rr_values_tuple)

    # Downsample if too many points (keeps every Nth point)
    n_points = len(timestamps)
    if n_points > downsample_threshold:
        step = n_points // downsample_threshold
        timestamps = timestamps[::step]
        rr_values = rr_values[::step]

    y_min = min(rr_values)
    y_max = max(rr_values)
    y_range = y_max - y_min

    return {
        'timestamps': timestamps,
        'rr_values': rr_values,
        'y_min': y_min,
        'y_max': y_max,
        'y_range': y_range,
        'n_original': n_points,
        'n_displayed': len(timestamps),
        'participant_id': participant_id
    }


@st.fragment
def render_participant_table_fragment():
    """Fragment for participant table - prevents re-render when expanders change.

    IMPORTANT: This must be defined at module level for Streamlit to cache it properly.
    """
    if not st.session_state.summaries:
        return

    # Participants overview table
    st.subheader("📋 Participants Overview")

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
            st.markdown("**⚠️ Issues Detected:**")
            for issue in issues:
                st.markdown(f"- {issue}")
            st.markdown("---")
    else:
        st.success(f"✅ All {total_participants} participants look good! No issues detected.")

    # Cache DataFrame creation (avoid rebuilding on every rerun)
    df_cache_key = f"df_{len(participants_data)}_{participants_data[0]['Participant'] if participants_data else ''}"
    if st.session_state.get("_df_participants_cache_key") != df_cache_key:
        st.session_state._df_participants = pd.DataFrame(participants_data)
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
            "Artifacts (%)": st.column_config.TextColumn(
                "Artifacts (%)",
                disabled=True,
                width="small",
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
        use_container_width=True,
        hide_index=True,
        key="participants_table",
        disabled=["Participant", "Saved", "Date/Time", "Total Beats", "Retained", "Duplicates", "Artifacts (%)", "Duration (min)", "Events", "Total Events", "Duplicate Events", "RR Range (ms)", "Mean RR (ms)"]
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
            f"⚠️ **Duplicate RR intervals detected!** "
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
                import_df = pd.read_csv(uploaded_file)
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
                                    "Group": str(row[group_col]) if use_group and pd.notna(row[group_col]) else "-",
                                    "Randomization": str(row[rand_col]) if use_rand and pd.notna(row[rand_col]) else "-"
                                })
                        if preview_data:
                            st.write("Preview (first matches):")
                            st.dataframe(pd.DataFrame(preview_data), hide_index=True)

                    if st.button("Apply Assignments", type="primary", key="apply_csv_import"):
                        applied_groups = 0
                        applied_rands = 0
                        for _, row in import_df.iterrows():
                            pid = str(row[id_col])
                            if pid in participant_ids:
                                if use_group and pd.notna(row[group_col]):
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
                                if use_rand and pd.notna(row[rand_col]):
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
        use_container_width=False,
    )
    st.caption("Group and randomization assignments save automatically when changed in the table.")


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

    # Plot display options
    st.markdown("**Plot Options:**")
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        show_variability = st.checkbox("Show variability segments", value=False,
                                       key=f"frag_show_var_{participant_id}",
                                       help="Variability analysis is computationally intensive.")
        show_music_sections = st.checkbox("Show music sections", value=True,
                                          key=f"frag_show_music_sec_{participant_id}")
    with col_opt2:
        show_gaps = st.checkbox("Show time gaps", value=True,
                                key=f"frag_show_gaps_{participant_id}")
        show_music_events = st.checkbox("Show music events", value=False,
                                        key=f"frag_show_music_evt_{participant_id}")
    with col_opt3:
        gap_threshold = st.number_input(
            "Gap threshold (s)",
            min_value=1.0, max_value=60.0, value=15.0, step=1.0,
            key=f"frag_gap_thresh_{participant_id}",
            help="Threshold for detecting gaps in data"
        )

    # Show downsampling info
    if plot_data['n_displayed'] < plot_data['n_original']:
        st.caption(f"Showing {plot_data['n_displayed']:,} of {plot_data['n_original']:,} points")

    # Build figure
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=plot_data['timestamps'],
        y=plot_data['rr_values'],
        mode='markers+lines',
        name='RR Intervals',
        marker=dict(size=3, color='blue'),
        line=dict(width=1, color='blue'),
        hovertemplate='Time: %{x}<br>RR: %{y} ms<extra></extra>'
    ))

    y_min, y_max = plot_data['y_min'], plot_data['y_max']
    y_range = plot_data['y_range']

    fig.update_layout(
        title=f"RR Intervals - {participant_id}",
        xaxis=dict(title="Time", tickformat='%H:%M:%S'),
        yaxis=dict(title="RR Interval (ms)"),
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top')
    )

    # Add event markers
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

    for idx, (event_name, event_times) in enumerate(event_by_canonical.items()):
        color = distinct_colors[idx % len(distinct_colors)]
        for event_time in event_times:
            fig.add_shape(
                type="line", x0=event_time, x1=event_time,
                y0=y_min - 0.05 * y_range, y1=y_max + 0.05 * y_range,
                line=dict(color=color, width=2, dash='dash'), opacity=0.7
            )
            fig.add_annotation(
                x=event_time, y=y_max + 0.08 * y_range,
                text=event_name, showarrow=False, textangle=-90,
                font=dict(color=color, size=10)
            )

    # Gap detection (fast)
    timestamps_list = list(plot_data['timestamps'])
    rr_list = list(plot_data['rr_values'])
    gap_result = detect_time_gaps(timestamps_list, rr_values=rr_list, gap_threshold_s=gap_threshold)
    st.session_state[f"gaps_{participant_id}"] = gap_result

    # Variability analysis (slow - only if enabled)
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

    # Visualize gaps
    if show_gaps and gap_result.get("gaps"):
        for gap in gap_result["gaps"]:
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
            label = evt.raw_label if hasattr(evt, 'raw_label') else str(evt)
            timestamp = evt.first_timestamp if hasattr(evt, 'first_timestamp') else None
            if not timestamp:
                continue
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
            label = evt.raw_label if hasattr(evt, 'raw_label') else str(evt)
            timestamp = evt.first_timestamp if hasattr(evt, 'first_timestamp') else None
            if timestamp:
                music_type = label.replace('_start', '').replace('_end', '')
                color = music_line_colors.get(music_type, '#808080')
                fig.add_shape(
                    type="line", x0=timestamp, x1=timestamp,
                    y0=y_min - 0.05 * y_range, y1=y_max + 0.05 * y_range,
                    line=dict(color=color, width=1, dash='dot'), opacity=0.5
                )

    # Display interactive plot
    st.info("💡 Click on the plot to add a new event at that timestamp")
    selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=600)

    # Store click result for parent to handle
    if selected_points and len(selected_points) > 0:
        st.session_state[f"plot_click_{participant_id}"] = selected_points[0]


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
        Emoji badge: 🟢 (good), 🟡 (moderate), 🔴 (poor)
    """
    # Combine changepoint quality and artifact ratio
    # Artifact ratio > 10% is concerning, > 20% is poor
    artifact_score = 100 - (artifact_ratio * 200)  # 10% artifacts = 80, 20% = 60
    artifact_score = max(0, min(100, artifact_score))

    combined = (quality_score + artifact_score) / 2

    if combined >= 75:
        return "🟢"
    elif combined >= 50:
        return "🟡"
    else:
        return "🔴"


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


def main():
    """Main Streamlit app."""
    import time as _time
    _script_start = _time.time()

    st.title("🎵 Music HRV Toolkit")
    st.markdown("### HRV Analysis Pipeline for Music Psychology Research")

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
                st.button(page_id, key=f"nav_{page_id}", use_container_width=True, type="primary")
            else:
                if st.button(page_id, key=f"nav_{page_id}", use_container_width=True, type="secondary"):
                    st.session_state.active_page = page_id
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

    # Get selected page for content rendering
    selected_page = st.session_state.active_page

    # ================== PAGE: DATA ==================
    if selected_page == "Data":
        render_data_tab()


    # ================== TAB: PARTICIPANTS ==================
    elif selected_page == "Participants":
        st.header("Participant Details")

        # Scroll to top if triggered by bottom navigation buttons
        if st.session_state.get("_scroll_to_top", False):
            st.session_state._scroll_to_top = False
            st.components.v1.html(
                """
                <script>
                    // Try multiple selectors for different Streamlit versions
                    var mainSection = window.parent.document.querySelector('[data-testid="stMainBlockContainer"]');
                    if (!mainSection) mainSection = window.parent.document.querySelector('section.main');
                    if (!mainSection) mainSection = window.parent.document.querySelector('.main');
                    if (mainSection) {
                        mainSection.scrollTo({top: 0, behavior: 'instant'});
                    }
                    // Also try scrolling the whole document
                    window.parent.document.documentElement.scrollTo({top: 0, behavior: 'instant'});
                    window.parent.scrollTo({top: 0, behavior: 'instant'});
                </script>
                """,
                height=0
            )

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

                st.button(
                    "Previous",
                    disabled=current_idx == 0,
                    key="prev_btn",
                    use_container_width=True,
                    on_click=go_previous
                )

            with col3:
                def go_next():
                    if st.session_state.current_participant_idx < len(participant_list) - 1:
                        st.session_state.current_participant_idx += 1
                        # Sync selectbox key with new index
                        new_participant = participant_list[st.session_state.current_participant_idx]
                        st.session_state.participant_selector = new_participant

                st.button(
                    "Next",
                    disabled=current_idx >= len(participant_list) - 1,
                    key="next_btn",
                    use_container_width=True,
                    on_click=go_next
                )

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
                col1, col2, col3, col4, col5 = st.columns(5)
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
                st.subheader("RR Interval Visualization")

                try:
                    # Load the recording using CACHED functions for instant access
                    bundles = cached_discover_recordings(st.session_state.data_dir, st.session_state.id_pattern)
                    bundle = next(b for b in bundles if b.participant_id == selected_participant)

                    # Get cached recording data (uses tuples for hashability)
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
                        st.session_state.participant_events[selected_participant] = {
                            'events': list(summary.events),
                            'manual': st.session_state.manual_events.get(selected_participant, []).copy()
                        }

                    # Get cleaned RR intervals using CACHED function
                    config_dict = {
                        "rr_min_ms": st.session_state.cleaning_config.rr_min_ms,
                        "rr_max_ms": st.session_state.cleaning_config.rr_max_ms,
                        "sudden_change_pct": st.session_state.cleaning_config.sudden_change_pct
                    }
                    rr_with_timestamps, stats = cached_clean_rr_intervals(
                        tuple(recording_data['rr_intervals']),
                        config_dict
                    )

                    if rr_with_timestamps and PLOTLY_AVAILABLE:
                        # Unpack cached data
                        timestamps, rr_values = zip(*rr_with_timestamps)

                        # Get CACHED plot data and store in session state for fragment
                        plot_data = cached_get_plot_data(
                            tuple(timestamps),
                            tuple(rr_values),
                            selected_participant
                        )
                        st.session_state[f"plot_data_{selected_participant}"] = plot_data

                        # Render plot using fragment (prevents full page reruns when toggling options)
                        render_rr_plot_fragment(selected_participant)

                        # Handle click events from fragment
                        click_key = f"plot_click_{selected_participant}"
                        if click_key in st.session_state:
                            clicked_point = st.session_state[click_key]
                            del st.session_state[click_key]  # Clear to prevent repeated triggers

                            if 'x' in clicked_point:
                                from datetime import datetime
                                clicked_timestamp = pd.to_datetime(clicked_point['x'])
                                if clicked_timestamp.tzinfo is None:
                                    clicked_timestamp = clicked_timestamp.tz_localize('UTC')

                                with st.form(key=f"add_event_from_plot_{selected_participant}"):
                                    st.write(f"Add event at: {clicked_timestamp.strftime('%H:%M:%S')}")
                                    new_event_label = st.text_input("Event label:")
                                    submitted = st.form_submit_button("Add Event")

                                    if submitted and new_event_label:
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
                                        show_toast(f"Added event '{new_event_label}' at {clicked_timestamp.strftime('%H:%M:%S')}", icon="success")
                                        st.rerun()

                except Exception as e:
                    st.warning(f"Could not generate RR plot: {e}")

                # Show quality analysis info if available
                changepoint_key = f"changepoints_{selected_participant}"
                gap_key = f"gaps_{selected_participant}"

                if changepoint_key in st.session_state or gap_key in st.session_state:
                    with st.expander("📊 Signal Quality Analysis", expanded=False):
                        # Documentation section
                        st.markdown("""
                        #### How Quality is Assessed

                        **1. Variability Analysis (signal_changepoints)**
                        Uses NeuroKit2's `signal_changepoints()` with the PELT algorithm to detect where
                        signal variance changes significantly. High variance segments may indicate:
                        - Movement artifacts
                        - Electrode contact issues
                        - Physiological changes (stress, exercise)

                        **2. Gap Detection (Missing Data)**
                        Identifies time gaps >2 seconds between consecutive beats, which indicate:
                        - Recording interruptions
                        - Bluetooth disconnections
                        - Device errors

                        ---
                        """)

                        # Gap Detection Results
                        if gap_key in st.session_state:
                            gap_info = st.session_state[gap_key]
                            st.markdown("##### ⏱️ Time Gap Analysis (Missing Data)")

                            col_g1, col_g2, col_g3 = st.columns(3)
                            with col_g1:
                                gap_badge = "🟢" if gap_info['total_gaps'] == 0 else ("🟡" if gap_info['total_gaps'] <= 2 else "🔴")
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
                                st.dataframe(pd.DataFrame(gap_data), use_container_width=True, hide_index=True)

                                # Recommendations for gaps
                                st.markdown("##### 💡 Recommendations for Gaps:")
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
                                st.markdown("##### 🤖 Auto-Create Gap Events")
                                col_gap_btn1, col_gap_btn2 = st.columns([1, 2])
                                with col_gap_btn1:
                                    if st.button("Create Gap Events", key=f"auto_gap_{selected_participant}"):
                                        from music_hrv.prep.summaries import EventStatus
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

                                        show_toast(f"✅ Created {events_added} gap boundary events", icon="success")
                                        st.rerun()
                                with col_gap_btn2:
                                    st.caption("Creates `gap_start` and `gap_end` events for each detected gap. Use these to exclude gap periods from analysis.")
                            else:
                                st.success("✅ No time gaps detected - recording appears continuous")

                        st.markdown("---")

                        # Changepoint Results
                        if changepoint_key in st.session_state:
                            cp_info = st.session_state[changepoint_key]
                            st.markdown("##### 📈 Variability Changepoint Analysis")

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
                                    quality = "🟢 Good" if cv_pct < 10 else ("🟡 Moderate" if cv_pct < 15 else "🔴 High")

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
                                st.dataframe(pd.DataFrame(seg_data), use_container_width=True, hide_index=True)

                                # Check for high variability segments
                                high_var_segments = [s for s in cp_info['segment_stats'] if s['cv'] > 0.15]
                                if high_var_segments:
                                    st.markdown("##### 💡 Recommendations for High Variability:")
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
                                    st.markdown("##### 🤖 Auto-Create Variability Events")
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
                                            from music_hrv.prep.summaries import EventStatus
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
                                                show_toast(f"✅ Created {events_added} variability boundary events", icon="success")
                                            else:
                                                show_toast("ℹ️ No segments above threshold", icon="info")
                                            st.rerun()
                                    with col_var_desc:
                                        st.caption(f"Creates `high_variability_start` and `high_variability_end` events for segments with CV > {cv_threshold:.0f}%")

                        st.caption("""
                        **Legend**:
                        - **CV (Coefficient of Variation)** = Std / Mean × 100. Lower = more stable.
                        - 🟢 Good: CV < 10% | 🟡 Moderate: 10-15% | 🔴 High: > 15%
                        - **Gray regions** on plot = time gaps (missing data)
                        - **Colored regions** = variability segments (green=stable, orange=moderate, red=high)
                        """)

                # Music Change Event Generator
                with st.expander("Generate Music Change Events", expanded=False):
                    st.markdown("""
                    **Auto-generate music section boundaries** based on timing intervals and playlist group.

                    Music changes every 5 minutes in a cycling pattern based on the participant's randomization group.
                    """)

                    # Ensure participant is initialized in events dict
                    if selected_participant not in st.session_state.participant_events:
                        st.session_state.participant_events[selected_participant] = {'events': [], 'manual': []}

                    # Get existing events to find measurement boundaries
                    stored_data = st.session_state.participant_events.get(selected_participant, {'events': [], 'manual': []})
                    all_current_events = stored_data.get('events', []) + stored_data.get('manual', [])

                    # Find all relevant boundary events for music generation
                    boundary_events = {}
                    for evt in all_current_events:
                        canonical = evt.canonical if hasattr(evt, 'canonical') else None
                        if canonical in ['measurement_start', 'measurement_end', 'pause_start', 'pause_end']:
                            if evt.first_timestamp:
                                boundary_events[canonical] = evt.first_timestamp

                    st.markdown("**Music Change Settings:**")

                    # Initialize playlist groups if needed
                    if "playlist_groups" not in st.session_state:
                        st.session_state.playlist_groups = {}
                    if "participant_playlists" not in st.session_state:
                        st.session_state.participant_playlists = {}

                    # Playlist group selection for this participant
                    playlist_options = ["(None - use custom order)"] + list(st.session_state.playlist_groups.keys())
                    current_playlist = st.session_state.participant_playlists.get(selected_participant, "(None - use custom order)")
                    if current_playlist not in playlist_options:
                        current_playlist = "(None - use custom order)"

                    selected_playlist = st.selectbox(
                        "Playlist Group (Randomization)",
                        options=playlist_options,
                        index=playlist_options.index(current_playlist) if current_playlist in playlist_options else 0,
                        key=f"playlist_select_{selected_participant}",
                        help="Select the randomization group for this participant"
                    )

                    # Save playlist assignment
                    if selected_playlist != "(None - use custom order)":
                        if st.session_state.participant_playlists.get(selected_participant) != selected_playlist:
                            st.session_state.participant_playlists[selected_participant] = selected_playlist
                        playlist_data = st.session_state.playlist_groups.get(selected_playlist, {})
                        music_label_list = playlist_data.get("music_order", ["music_1", "music_2", "music_3"])
                        st.success(f"Using **{selected_playlist}** music order: {' → '.join(music_label_list)}")
                    else:
                        if selected_participant in st.session_state.participant_playlists:
                            del st.session_state.participant_playlists[selected_participant]

                        # Custom order input
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            music_interval_min = st.number_input(
                                "Interval (minutes)",
                                min_value=1,
                                max_value=30,
                                value=5,
                                step=1,
                                key=f"music_interval_{selected_participant}",
                                help="How often the music changes (in minutes)"
                            )
                        with col_m2:
                            music_labels = st.text_area(
                                "Music type labels (one per line)",
                                value="music_1\nmusic_2\nmusic_3",
                                height=100,
                                key=f"music_labels_{selected_participant}",
                                help="Custom labels for each music type"
                            )

                        # Parse music labels
                        music_label_list = [line.strip() for line in music_labels.strip().split('\n') if line.strip()]
                        if not music_label_list:
                            music_label_list = ["music_1", "music_2", "music_3"]

                    # Always show interval setting when using playlist group
                    if selected_playlist != "(None - use custom order)":
                        music_interval_min = st.number_input(
                            "Interval (minutes)",
                            min_value=1,
                            max_value=30,
                            value=5,
                            step=1,
                            key=f"music_interval_pl_{selected_participant}",
                            help="How often the music changes (in minutes)"
                        )

                    st.markdown("**Preview:**")
                    st.write(f"Music cycle: {' → '.join(music_label_list)} → (repeat)")

                    # Generate button
                    if st.button("🎵 Generate Music Events", key=f"gen_music_{selected_participant}"):
                        from music_hrv.prep.summaries import EventStatus
                        from datetime import timedelta

                        events_added = 0
                        interval_seconds = music_interval_min * 60
                        num_music_types = len(music_label_list)

                        # Initialize music_events list if not present
                        if 'music_events' not in st.session_state.participant_events[selected_participant]:
                            st.session_state.participant_events[selected_participant]['music_events'] = []
                        # Clear existing music events before generating new ones
                        st.session_state.participant_events[selected_participant]['music_events'] = []

                        # Generate for pre-pause period (measurement_start to pause_start)
                        if 'measurement_start' in boundary_events:
                            start_time = boundary_events['measurement_start']
                            end_time = boundary_events.get('pause_start') or boundary_events.get('measurement_end')

                            if end_time:
                                current_time = start_time
                                music_idx = 0

                                while current_time < end_time:
                                    # Create music section start event
                                    label = music_label_list[music_idx % num_music_types]
                                    event_label = f"{label}_start"

                                    new_event = EventStatus(
                                        raw_label=event_label,
                                        canonical=event_label,
                                        first_timestamp=current_time,
                                        last_timestamp=current_time
                                    )
                                    st.session_state.participant_events[selected_participant]['music_events'].append(new_event)
                                    events_added += 1

                                    # Calculate end time for this music segment
                                    segment_end = current_time + timedelta(seconds=interval_seconds)
                                    if segment_end > end_time:
                                        segment_end = end_time

                                    # Create music section end event
                                    end_event = EventStatus(
                                        raw_label=f"{label}_end",
                                        canonical=f"{label}_end",
                                        first_timestamp=segment_end,
                                        last_timestamp=segment_end
                                    )
                                    st.session_state.participant_events[selected_participant]['music_events'].append(end_event)
                                    events_added += 1

                                    current_time = segment_end
                                    music_idx += 1

                        # Generate for post-pause period (pause_end to measurement_end)
                        if 'pause_end' in boundary_events and 'measurement_end' in boundary_events:
                            start_time = boundary_events['pause_end']
                            end_time = boundary_events['measurement_end']

                            current_time = start_time
                            music_idx = 0  # Reset cycle after pause

                            while current_time < end_time:
                                label = music_label_list[music_idx % num_music_types]
                                event_label = f"{label}_start"

                                new_event = EventStatus(
                                    raw_label=event_label,
                                    canonical=event_label,
                                    first_timestamp=current_time,
                                    last_timestamp=current_time
                                )
                                st.session_state.participant_events[selected_participant]['music_events'].append(new_event)
                                events_added += 1

                                segment_end = current_time + timedelta(seconds=interval_seconds)
                                if segment_end > end_time:
                                    segment_end = end_time

                                end_event = EventStatus(
                                    raw_label=f"{label}_end",
                                    canonical=f"{label}_end",
                                    first_timestamp=segment_end,
                                    last_timestamp=segment_end
                                )
                                st.session_state.participant_events[selected_participant]['music_events'].append(end_event)
                                events_added += 1

                                current_time = segment_end
                                music_idx += 1

                        if events_added > 0:
                            show_toast(f"✅ Created {events_added} music section events", icon="success")
                        else:
                            show_toast("⚠️ No events created - check boundary events", icon="warning")
                        st.rerun()

                    st.caption("""
                    **How it works:**
                    1. Uses the participant's playlist group to determine music order
                    2. Finds `measurement_start`, `pause_start`, `pause_end`, `measurement_end` events
                    3. Creates music section events every 5 minutes between these boundaries
                    4. Restarts cycle after the pause
                    """)

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

                # Timing validation section
                st.markdown("---")
                st.markdown("##### ⏱️ Timing Validation")

                # Find boundary events from current events
                boundary_events = {}
                for evt in current_events:
                    canonical = evt.canonical if hasattr(evt, 'canonical') else None
                    if canonical in ['measurement_start', 'measurement_end', 'pause_start', 'pause_end',
                                    'rest_pre_start', 'rest_pre_end', 'rest_post_start', 'rest_post_end']:
                        if evt.first_timestamp:
                            boundary_events[canonical] = evt.first_timestamp

                if boundary_events:
                    # Show detected boundaries
                    with st.expander("Detected Boundary Events", expanded=False):
                        for name, ts in sorted(boundary_events.items(), key=lambda x: x[1] if x[1] else ""):
                            st.write(f"- {name}: {ts.strftime('%H:%M:%S') if ts else 'N/A'}")

                    # Calculate and validate durations
                    validation_issues = []
                    validation_ok = []

                    # Rest Pre duration (should be 3-5 min)
                    if 'rest_pre_start' in boundary_events and 'rest_pre_end' in boundary_events:
                        rest_pre_dur = (boundary_events['rest_pre_end'] - boundary_events['rest_pre_start']).total_seconds() / 60
                        if rest_pre_dur < 3:
                            validation_issues.append(f"⚠️ Rest Pre: {rest_pre_dur:.1f} min (should be ≥3 min)")
                        elif rest_pre_dur < 5:
                            validation_ok.append(f"🟡 Rest Pre: {rest_pre_dur:.1f} min (OK, but 5 min recommended)")
                        else:
                            validation_ok.append(f"✅ Rest Pre: {rest_pre_dur:.1f} min")

                    # Rest Post duration (should be 3-5 min)
                    if 'rest_post_start' in boundary_events and 'rest_post_end' in boundary_events:
                        rest_post_dur = (boundary_events['rest_post_end'] - boundary_events['rest_post_start']).total_seconds() / 60
                        if rest_post_dur < 3:
                            validation_issues.append(f"⚠️ Rest Post: {rest_post_dur:.1f} min (should be ≥3 min)")
                        elif rest_post_dur < 5:
                            validation_ok.append(f"🟡 Rest Post: {rest_post_dur:.1f} min (OK, but 5 min recommended)")
                        else:
                            validation_ok.append(f"✅ Rest Post: {rest_post_dur:.1f} min")

                    # Pre-pause measurement (should be ~90 min / 1.5 hours)
                    if 'measurement_start' in boundary_events and 'pause_start' in boundary_events:
                        pre_pause_dur = (boundary_events['pause_start'] - boundary_events['measurement_start']).total_seconds() / 60
                        expected_segments = int(pre_pause_dur / 5)
                        if abs(pre_pause_dur - 90) > 10:
                            validation_issues.append(f"⚠️ Pre-pause measurement: {pre_pause_dur:.1f} min (expected ~90 min)")
                        else:
                            validation_ok.append(f"✅ Pre-pause measurement: {pre_pause_dur:.1f} min ({expected_segments} × 5-min segments)")

                        # Check if 5-min intervals fit evenly
                        remainder = pre_pause_dur % 5
                        if remainder > 0.5 and remainder < 4.5:
                            validation_issues.append(f"⚠️ Pre-pause: {remainder:.1f} min leftover after 5-min segments")

                    # Post-pause measurement (should be ~90 min / 1.5 hours)
                    if 'pause_end' in boundary_events and 'measurement_end' in boundary_events:
                        post_pause_dur = (boundary_events['measurement_end'] - boundary_events['pause_end']).total_seconds() / 60
                        expected_segments = int(post_pause_dur / 5)
                        if abs(post_pause_dur - 90) > 10:
                            validation_issues.append(f"⚠️ Post-pause measurement: {post_pause_dur:.1f} min (expected ~90 min)")
                        else:
                            validation_ok.append(f"✅ Post-pause measurement: {post_pause_dur:.1f} min ({expected_segments} × 5-min segments)")

                        # Check if 5-min intervals fit evenly
                        remainder = post_pause_dur % 5
                        if remainder > 0.5 and remainder < 4.5:
                            validation_issues.append(f"⚠️ Post-pause: {remainder:.1f} min leftover after 5-min segments")

                    # Display validation results
                    for msg in validation_ok:
                        st.write(msg)
                    for msg in validation_issues:
                        st.warning(msg)

                    if not validation_ok and not validation_issues:
                        st.info("Waiting for more boundary events to validate timing...")
                else:
                    st.info("No boundary events found yet. Events will be validated once measurement boundaries are detected.")

                # Save participant button
                st.markdown("---")
                col_save, col_status = st.columns([1, 2])
                with col_save:
                    def save_participant_to_disk():
                        """Save participant events to disk."""
                        # Get the current recording with updated events
                        stored_data = st.session_state.participant_events[selected_participant]
                        all_evts = stored_data['events'] + stored_data['manual']

                        # Find the original summary to get the recording
                        orig_summary = get_summary_dict().get(selected_participant)  # O(1) cached lookup
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
                             on_click=save_participant_to_disk,
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
                    use_container_width=True,
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
                    use_container_width=True,
                    on_click=go_next_bottom
                )

    # ================== TAB: SETUP ==================
    elif selected_page == "Setup":
        render_setup_tab()

    # ================== TAB: ANALYSIS ==================
    elif selected_page == "Analysis":
        st.header("HRV Analysis")

        with st.expander("❓ Help - HRV Analysis", expanded=False):
            st.markdown("""
            ### What is HRV Analysis?

            Heart Rate Variability (HRV) analysis quantifies the variation in time between heartbeats.
            Higher HRV generally indicates better cardiovascular health and autonomic function.

            ### Key Metrics

            **Time Domain:**
            - **RMSSD**: Root Mean Square of Successive Differences - sensitive to parasympathetic activity
            - **SDNN**: Standard Deviation of NN intervals - overall HRV
            - **pNN50**: Percentage of successive differences > 50ms

            **Frequency Domain:**
            - **HF (High Frequency)**: 0.15-0.4 Hz - parasympathetic activity
            - **LF (Low Frequency)**: 0.04-0.15 Hz - mixed sympathetic/parasympathetic
            - **LF/HF Ratio**: Sympathetic/parasympathetic balance

            ### Analysis Modes

            - **Single Participant**: Analyze one participant at a time, compare different sections
            - **Group Analysis**: Compare HRV across all participants in a group

            ### Best Practices

            - Ensure clean data (check for gaps and artifacts first)
            - Use sections of at least 5 minutes for frequency domain analysis
            - Compare equivalent sections across participants (e.g., all rest_pre periods)
            """)

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
                participant_list = get_participant_list()  # Cached for performance
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

                    # Artifact correction options
                    with st.expander("🔧 Artifact Correction (signal_fixpeaks)", expanded=False):
                        st.markdown("""
                        Uses NeuroKit2's `signal_fixpeaks()` with the **Kubios algorithm** to detect and correct:
                        - **Ectopic beats** (premature/delayed beats)
                        - **Missed beats** (undetected R-peaks)
                        - **Extra beats** (false positive detections)
                        - **Long/short intervals** (physiologically implausible)
                        """)
                        apply_artifact_correction = st.checkbox(
                            "Apply artifact correction before HRV analysis",
                            value=False,
                            key="apply_artifact_correction",
                            help="Recommended for data with known quality issues"
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
                                    # Load the recording using CACHED functions
                                    bundles = cached_discover_recordings(st.session_state.data_dir, st.session_state.id_pattern)
                                    bundle = next(b for b in bundles if b.participant_id == selected_participant)
                                    recording_data = cached_load_recording(
                                        tuple(str(p) for p in bundle.rr_paths),
                                        tuple(str(p) for p in bundle.events_paths),
                                        selected_participant
                                    )
                                    # Reconstruct recording object from cached data
                                    from music_hrv.io.hrv_logger import HRVLoggerRecording, EventMarker
                                    from music_hrv.cleaning.rr import RRInterval
                                    rr_intervals = [RRInterval(timestamp=ts, rr_ms=rr, elapsed_ms=elapsed)
                                                    for ts, rr, elapsed in recording_data['rr_intervals']]
                                    events = [EventMarker(label=label, timestamp=ts, offset_s=None)
                                              for label, ts in recording_data['events']]
                                    recording = HRVLoggerRecording(
                                        participant_id=selected_participant,
                                        rr_intervals=rr_intervals,
                                        events=events
                                    )
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

                                                # Apply artifact correction if enabled
                                                artifact_info = None
                                                if apply_artifact_correction:
                                                    st.write("    🔧 Applying artifact correction...")
                                                    artifact_result = detect_artifacts_fixpeaks(rr_ms)
                                                    if artifact_result["correction_applied"]:
                                                        rr_ms = artifact_result["corrected_rr"]
                                                        artifact_info = artifact_result
                                                        st.write(f"    ✓ Corrected {artifact_result['total_artifacts']} artifacts")

                                                combined_rr.extend(rr_ms)

                                                # Calculate HRV metrics
                                                nk = get_neurokit()
                                                peaks = nk.intervals_to_peaks(rr_ms, sampling_rate=1000)
                                                hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
                                                hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)
                                                hrv_results = pd.concat([hrv_time, hrv_freq], axis=1)

                                                section_results[section_name] = {
                                                    "hrv_results": hrv_results,
                                                    "rr_intervals": rr_ms,
                                                    "n_beats": len(rr_ms),
                                                    "label": section_def.get("label", section_name),
                                                    "artifact_info": artifact_info,
                                                }
                                        else:
                                            st.write(f"  ⚠️ Could not find events for section '{section_name}'")

                                    # Analyze combined sections if multiple selected
                                    if len(selected_sections) > 1 and combined_rr:
                                        progress.progress(80)
                                        st.write("📊 Computing combined analysis...")
                                        nk = get_neurokit()
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
                            artifact_info = result_data.get("artifact_info")

                            with st.expander(f"📈 {section_label} ({n_beats} beats)", expanded=True):
                                # Show artifact correction info if applied
                                if artifact_info:
                                    st.info(f"🔧 **Artifact Correction Applied**: {artifact_info['total_artifacts']} artifacts corrected "
                                           f"({artifact_info['artifact_ratio']*100:.1f}% of beats)")
                                    art = artifact_info['artifacts']
                                    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                                    with col_a1:
                                        st.metric("Ectopic", art['ectopic'])
                                    with col_a2:
                                        st.metric("Missed", art['missed'])
                                    with col_a3:
                                        st.metric("Extra", art['extra'])
                                    with col_a4:
                                        st.metric("Long/Short", art['longshort'])

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
                                    plt = get_matplotlib()
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
                                    from music_hrv.cleaning.rr import clean_rr_intervals, RRInterval
                                    from music_hrv.io.hrv_logger import HRVLoggerRecording, EventMarker
                                    bundles = cached_discover_recordings(st.session_state.data_dir, st.session_state.id_pattern)

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
                                            # Use CACHED loading
                                            recording_data = cached_load_recording(
                                                tuple(str(p) for p in bundle.rr_paths),
                                                tuple(str(p) for p in bundle.events_paths),
                                                participant_id
                                            )
                                            # Reconstruct recording object
                                            rr_intervals = [RRInterval(timestamp=ts, rr_ms=rr, elapsed_ms=elapsed)
                                                            for ts, rr, elapsed in recording_data['rr_intervals']]
                                            events = [EventMarker(label=label, timestamp=ts, offset_s=None)
                                                      for label, ts in recording_data['events']]
                                            recording = HRVLoggerRecording(
                                                participant_id=participant_id,
                                                rr_intervals=rr_intervals,
                                                events=events
                                            )

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

                                                        nk = get_neurokit()
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
                                                nk = get_neurokit()
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

    # Record render time for debugging
    st.session_state.last_render_time = (_time.time() - _script_start) * 1000


if __name__ == "__main__":
    main()
