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
    load_settings,
    save_settings,
    DEFAULT_SETTINGS,
)
from music_hrv.gui.tabs.setup import render_setup_tab
from music_hrv.gui.tabs.data import render_data_tab
from music_hrv.gui.tabs.analysis import render_analysis_tab
from music_hrv.gui.help_text import (
    ARTIFACT_CORRECTION_HELP,
    VNS_DATA_HELP,
)

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
    page_title="Music HRV Toolkit" + (" [TEST MODE]" if TEST_MODE else ""),
    page_icon="M" if TEST_MODE else "M",
    layout="wide",
)


def apply_custom_css():
    """Apply custom CSS for professional styling.

    Theme colors are handled by Streamlit's native theming.
    Light mode: config.toml defaults
    Dark mode: localStorage override via theme toggle buttons
    """
    # Base styling that works with any theme - NO color overrides
    base_css = """
    /* Professional styling improvements */
    [data-testid="stMetric"] { border-radius: 8px; padding: 12px 16px; }
    [data-testid="stExpander"] details { border-radius: 8px; }
    [data-testid="stExpander"] summary { font-weight: 500; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    .stButton button { border-radius: 6px; font-weight: 500; }
    [data-testid="stAlert"] { border-radius: 8px; }
    [data-baseweb="select"] { border-radius: 6px; }
    .stProgress > div > div { border-radius: 4px; }
    """
    st.markdown(f"<style>{base_css}</style>", unsafe_allow_html=True)


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
    # Auto-load data in test mode
    if TEST_MODE and st.session_state.data_dir:
        from music_hrv.gui.shared import cached_load_hrv_logger_preview
        config_dict = {"rr_min_ms": 200, "rr_max_ms": 2000, "sudden_change_pct": 100}
        try:
            summaries = cached_load_hrv_logger_preview(
                st.session_state.data_dir,
                pattern=DEFAULT_ID_PATTERN,
                config_dict=config_dict,
                gui_events_dict={},
            )
            st.session_state.summaries = summaries
            # Auto-assign to Default group
            for s in summaries:
                if "participant_groups" not in st.session_state:
                    st.session_state.participant_groups = {}
                st.session_state.participant_groups[s.participant_id] = "Default"
        except Exception:
            pass  # Silently fail - user can load manually
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
        # Default sections - end_events is a list (any of these events can end the section)
        st.session_state.sections = {
            "rest_pre": {"label": "Pre-Rest", "description": "Baseline rest period", "start_event": "rest_pre_start", "end_events": ["rest_pre_end"]},
            "measurement": {"label": "Measurement", "description": "Main measurement period", "start_event": "measurement_start", "end_events": ["measurement_end"]},
            "pause": {"label": "Pause", "description": "Break between blocks", "start_event": "pause_start", "end_events": ["pause_end"]},
            "rest_post": {"label": "Post-Rest", "description": "Post-measurement rest", "start_event": "rest_post_start", "end_events": ["rest_post_end"]},
        }
    else:
        # Migrate old format (end_event) to new format (end_events)
        for section_data in loaded_sections.values():
            if "end_event" in section_data and "end_events" not in section_data:
                section_data["end_events"] = [section_data.pop("end_event")]
        st.session_state.sections = loaded_sections

# Create normalizer from GUI events - always recreate to pick up code/config changes
st.session_state.normalizer = create_gui_normalizer(st.session_state.all_events)

# Load participant-specific data (groups, playlists, labels, event orders, manual events)
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


@st.cache_data(show_spinner=False, ttl=600)
def cached_load_vns_recording(vns_path_str: str, participant_id: str, use_corrected: bool = False):
    """Cache loaded VNS recording data for instant access."""
    from music_hrv.io.vns_analyse import VNSRecordingBundle, load_vns_recording
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
    from music_hrv.cleaning.rr import clean_rr_intervals, CleaningStats, RRInterval

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


@st.cache_data(show_spinner=False, ttl=300)
def cached_artifact_detection(rr_values_tuple, timestamps_tuple):
    """Cache NeuroKit2 artifact detection results with indices.

    Returns dict with artifact indices mapped to timestamps for plotting.
    """
    if not NEUROKIT_AVAILABLE:
        return {"artifact_indices": [], "artifact_timestamps": [], "artifact_rr": [],
                "total_artifacts": 0, "artifact_ratio": 0.0, "by_type": {}}

    rr_list = list(rr_values_tuple)
    timestamps_list = list(timestamps_tuple)

    if len(rr_list) < 10:
        return {"artifact_indices": [], "artifact_timestamps": [], "artifact_rr": [],
                "total_artifacts": 0, "artifact_ratio": 0.0, "by_type": {}}

    try:
        import numpy as np
        nk = get_neurokit()

        rr_array = np.array(rr_list, dtype=float)
        peak_indices = np.cumsum(rr_array).astype(int)
        peak_indices = np.insert(peak_indices, 0, 0)

        info, _ = nk.signal_fixpeaks(
            peak_indices,
            sampling_rate=1000,
            iterative=True,
            method="Kubios",
            show=False,
        )

        # Collect all artifact indices
        artifact_indices = set()
        by_type = {}

        for artifact_type in ["ectopic", "missed", "extra", "longshort"]:
            indices = info.get(artifact_type, [])
            if isinstance(indices, np.ndarray):
                indices = indices.tolist()
            elif not isinstance(indices, list):
                indices = []
            by_type[artifact_type] = len(indices)
            artifact_indices.update(indices)

        # Filter to valid range and get timestamps/values
        valid_indices = sorted([i for i in artifact_indices if 0 <= i < len(timestamps_list)])
        artifact_timestamps = [timestamps_list[i] for i in valid_indices]
        artifact_rr = [rr_list[i] for i in valid_indices]

        return {
            "artifact_indices": valid_indices,
            "artifact_timestamps": artifact_timestamps,
            "artifact_rr": artifact_rr,
            "total_artifacts": len(valid_indices),
            "artifact_ratio": len(valid_indices) / len(rr_list) if rr_list else 0.0,
            "by_type": by_type,
        }
    except Exception:
        return {"artifact_indices": [], "artifact_timestamps": [], "artifact_rr": [],
                "total_artifacts": 0, "artifact_ratio": 0.0, "by_type": {}}


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
def cached_get_plot_data(timestamps_tuple, rr_values_tuple, participant_id: str, downsample_threshold: int = 5000, flags_tuple=None):
    """Cache processed plot data (NOT the figure - that's slow to serialize).

    Downsamples data if too many points for faster rendering.
    Returns the data needed to build the plot quickly.

    Args:
        flags_tuple: Optional tuple of booleans indicating flagged (problematic) intervals (VNS only)
    """
    timestamps = list(timestamps_tuple)
    rr_values = list(rr_values_tuple)
    flags = list(flags_tuple) if flags_tuple else None

    # Downsample if too many points (keeps every Nth point)
    n_points = len(timestamps)
    if n_points > downsample_threshold:
        step = n_points // downsample_threshold
        timestamps = timestamps[::step]
        rr_values = rr_values[::step]
        if flags:
            flags = flags[::step]

    y_min = min(rr_values)
    y_max = max(rr_values)
    y_range = y_max - y_min

    result = {
        'timestamps': timestamps,
        'rr_values': rr_values,
        'y_min': y_min,
        'y_max': y_max,
        'y_range': y_range,
        'n_original': n_points,
        'n_displayed': len(timestamps),
        'participant_id': participant_id
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
        width='stretch',
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

    # Theme toggle - uses Streamlit's native theming via localStorage
    st.caption("**Theme**")
    import streamlit.components.v1 as components

    # Get current participant to preserve across theme switch
    current_participant = st.session_state.get("selected_participant", "")

    # JavaScript to switch themes and preserve state
    components.html(f"""
        <style>
            .theme-btn {{
                flex: 1;
                padding: 0.4rem 0.8rem;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-family: inherit;
            }}
            .light-btn {{
                background: #f0f2f6;
                border: 1px solid #ccc;
                color: #31333F;
            }}
            .dark-btn {{
                background: #262730;
                border: 1px solid #555;
                color: #fafafa;
            }}
            .theme-btn:hover {{ opacity: 0.8; }}
        </style>
        <div style="display: flex; gap: 8px;">
            <button class="theme-btn light-btn" onclick="
                window.parent.localStorage.removeItem('stActiveTheme-/-v1');
                var url = new URL(window.parent.location.href);
                url.searchParams.set('restore_participant', '{current_participant}');
                window.parent.location.href = url.toString();
            ">Light</button>
            <button class="theme-btn dark-btn" onclick="
                var darkTheme = {{
                    name: 'Dark',
                    themeInput: {{
                        primaryColor: '#2E86AB',
                        backgroundColor: '#0E1117',
                        secondaryBackgroundColor: '#262730',
                        textColor: '#FAFAFA',
                        base: 'dark'
                    }}
                }};
                window.parent.localStorage.setItem('stActiveTheme-/-v1', JSON.stringify(darkTheme));
                var url = new URL(window.parent.location.href);
                url.searchParams.set('restore_participant', '{current_participant}');
                window.parent.location.href = url.toString();
            ">Dark</button>
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
        new_show_music_sec = st.checkbox("Music sections", value=plot_opts.get("show_music_sections", True), key="settings_show_music_sec")
        new_show_artifacts = st.checkbox("Artifacts", value=plot_opts.get("show_artifacts", False), key="settings_show_artifacts")
        new_show_variability = st.checkbox("Variability", value=plot_opts.get("show_variability", False), key="settings_show_variability")

    # Save button
    if st.button("Save Settings", key="save_settings_btn", use_container_width=True):
        new_settings = {
            "data_folder": new_folder,
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

    # Always use Scattergl for performance
    ScatterType = go.Scattergl

    # Determine source_app (check plot_data first, then fall back to summary)
    source_app = plot_data.get('source_app')
    if not source_app:
        # Fall back to checking the summary
        summary = get_summary_dict().get(participant_id)
        source_app = getattr(summary, 'source_app', 'HRV Logger') if summary else 'HRV Logger'
    is_vns_data = (source_app == "VNS Analyse")

    # Show source info and clear any old gap data for VNS
    if is_vns_data:
        st.info(f"**Data source: {source_app}** - Gap detection disabled (timestamps synthesized from RR intervals)")
        # Force clear any old gap data for VNS participants
        st.session_state[f"gaps_{participant_id}"] = {
            "gaps": [], "total_gaps": 0, "total_gap_duration_s": 0.0, "gap_ratio": 0.0, "vns_note": True
        }

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
        show_artifacts = st.checkbox("Show artifacts (NeuroKit2)", value=plot_defaults.get("show_artifacts", False),
                                     key=f"frag_show_artifacts_{participant_id}",
                                     help="Detect ectopic, missed, extra beats using Kubios algorithm")
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
        with st.popover("Help"):
            if is_vns_data:
                st.markdown(VNS_DATA_HELP)
            else:
                st.markdown(ARTIFACT_CORRECTION_HELP)

    # Show downsampling info
    if plot_data['n_displayed'] < plot_data['n_original']:
        st.caption(f"Showing {plot_data['n_displayed']:,} of {plot_data['n_original']:,} points")

    # Build figure
    fig = go.Figure()

    # Check if we have flags (VNS data with flagged intervals)
    flags = plot_data.get('flags')
    if flags:
        # VNS data: Split into valid (blue) and flagged (red) intervals
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
                      f"shown in RED, excluded from HRV analysis. "
                      f"Total flagged time: {flagged_time_ms/1000:.1f}s")

        # Valid intervals in blue (connected with lines)
        if good_ts:
            fig.add_trace(ScatterType(
                x=good_ts,
                y=good_rr,
                mode='markers+lines',
                name='RR Intervals (valid)',
                marker=dict(size=3, color='blue'),
                line=dict(width=1, color='blue'),
                hovertemplate='Time: %{x}<br>RR: %{y} ms<extra></extra>'
            ))

        # Flagged intervals in red (markers only, no lines to show discontinuity)
        if flagged_ts:
            fig.add_trace(ScatterType(
                x=flagged_ts,
                y=flagged_rr,
                mode='markers',
                name='RR Intervals (flagged)',
                marker=dict(size=5, color='red', symbol='x'),
                hovertemplate='Time: %{x}<br>RR: %{y} ms (FLAGGED)<extra></extra>'
            ))
    else:
        # HRV Logger: Already cleaned, show all in blue
        fig.add_trace(ScatterType(
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
        title=f"Tachogram - {participant_id}",
        xaxis=dict(title="Time", tickformat='%H:%M:%S'),
        yaxis=dict(title="RR Interval (ms)"),
        hovermode='closest',
        height=600,
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top'),
        uirevision=participant_id  # Preserve zoom/pan state across updates
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

    # Gap detection (CACHED) - skip for VNS data (synthesized timestamps)
    timestamps_list = plot_data['timestamps']
    rr_list = plot_data['rr_values']
    if is_vns_data:
        # VNS timestamps are synthesized from RR intervals, so gaps are meaningless
        gap_result = {"gaps": [], "total_gaps": 0, "total_gap_duration_s": 0.0, "gap_ratio": 0.0, "vns_note": True}
    else:
        # Use cached version to avoid recalculation on every threshold change
        gap_result = cached_gap_detection(tuple(timestamps_list), tuple(rr_list), gap_threshold)
    st.session_state[f"gaps_{participant_id}"] = gap_result

    # Variability analysis (slow - only if enabled)
    # Artifact detection using NeuroKit2 (Kubios algorithm)
    if show_artifacts:
        artifact_result = cached_artifact_detection(tuple(rr_list), tuple(timestamps_list))
        st.session_state[f"artifacts_{participant_id}"] = artifact_result

        if artifact_result and artifact_result["total_artifacts"] > 0:
            # Show artifact summary
            by_type = artifact_result["by_type"]
            st.info(f"**{artifact_result['total_artifacts']} artifacts detected** "
                   f"({artifact_result['artifact_ratio']*100:.1f}%) - "
                   f"Ectopic: {by_type.get('ectopic', 0)}, "
                   f"Missed: {by_type.get('missed', 0)}, "
                   f"Extra: {by_type.get('extra', 0)}, "
                   f"Long/Short: {by_type.get('longshort', 0)}")

            # Add artifact markers to plot (orange X markers)
            if artifact_result["artifact_timestamps"]:
                fig.add_trace(ScatterType(
                    x=artifact_result["artifact_timestamps"],
                    y=artifact_result["artifact_rr"],
                    mode='markers',
                    name='Artifacts (NeuroKit2)',
                    marker=dict(size=8, color='orange', symbol='x', line=dict(width=2)),
                    hovertemplate='Time: %{x}<br>RR: %{y} ms (ARTIFACT)<extra></extra>'
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

    # Visualize gaps (skip for VNS data - timestamps are synthesized)
    # PERFORMANCE: Limit gaps shown to prevent plot slowdown
    MAX_GAPS_SHOWN = 50
    if show_gaps and gap_result.get("gaps") and not is_vns_data:
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
                    zone_start = pd.to_datetime(zone_start)
                if isinstance(zone_end, str):
                    zone_end = pd.to_datetime(zone_end)

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

    # Show pending exclusion click points on the plot
    exclusion_click_key = f"exclusion_clicks_{participant_id}"
    pending_clicks = st.session_state.get(exclusion_click_key, [])
    if pending_clicks:
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

    # Check if in click-two-points exclusion mode
    exclusion_method_key = f"exclusion_method_{participant_id}"
    is_exclusion_click_mode_check = (
        exclusion_method_key in st.session_state and
        st.session_state[exclusion_method_key] == "Click two points on plot"
    )

    # Display interactive plot with click detection
    col_mode_info, col_refresh = st.columns([5, 1])
    with col_mode_info:
        if is_exclusion_click_mode_check:
            st.info("**Click two points** on the plot to define an exclusion zone (start → end)")
        else:
            st.info("Click on the plot to add a new event at that timestamp")
    with col_refresh:
        if st.button("Refresh", key=f"refresh_plot_{participant_id}", help="Refresh plot to show new markers (resets zoom)"):
            st.rerun()

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
    is_exclusion_click_mode = (
        exclusion_method_key in st.session_state and
        st.session_state[exclusion_method_key] == "Click two points on plot"
    )

    # Handle click immediately
    # Track the last processed click to avoid reprocessing on rerun
    last_click_key = f"last_click_{participant_id}"

    if selected_points and len(selected_points) > 0:
        clicked_point = selected_points[0]
        if 'x' in clicked_point:
            clicked_ts = pd.to_datetime(clicked_point['x'])
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
                        st.rerun()

                # Always return early in exclusion mode (don't show event form)
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
                    from music_hrv.prep.summaries import EventStatus

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


def main():
    """Main Streamlit app."""
    import time as _time
    _script_start = _time.time()

    if TEST_MODE:
        st.title("Music HRV Toolkit [TEST MODE]")
        st.info("**Test mode active** - Using demo data from `data/demo/hrv_logger`")
    else:
        st.title("Music HRV Toolkit")
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
                st.button(page_id, key=f"nav_{page_id}", width='stretch', type="primary")
            else:
                if st.button(page_id, key=f"nav_{page_id}", width='stretch', type="secondary"):
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

        # Settings section
        st.markdown("---")
        with st.expander("Settings", expanded=False):
            render_settings_panel()

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

                # RR Interval Plot with Event Markers
                st.markdown("---")
                st.subheader("RR Interval Visualization")

                try:
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
                        from music_hrv.gui.persistence import load_participant_events
                        from music_hrv.prep.summaries import EventStatus
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
                            # Load from original recording - VNS events already have correct
                            # timestamps from the VNS loader (based on cumulative RR intervals
                            # from filename datetime)
                            st.session_state.participant_events[selected_participant] = {
                                'events': list(summary.events),
                                'manual': st.session_state.manual_events.get(selected_participant, []).copy(),
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

                    if rr_with_timestamps and PLOTLY_AVAILABLE:
                        # Unpack cached data - VNS has 3 elements (with flag), HRV Logger has 2
                        if is_vns:
                            timestamps, rr_values, flags = zip(*rr_with_timestamps)
                        else:
                            timestamps, rr_values = zip(*rr_with_timestamps)
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

                        # Mode selector for plot interaction (Events vs Exclusions)
                        st.markdown("---")
                        col_mode1, col_mode2, col_mode3 = st.columns([1, 2, 1])
                        with col_mode1:
                            interaction_mode = st.radio(
                                "Plot interaction",
                                ["Add Events", "Add Exclusions"],
                                key=f"plot_mode_{selected_participant}",
                                horizontal=True,
                                label_visibility="collapsed"
                            )
                        with col_mode2:
                            if interaction_mode != "Add Exclusions":
                                # Clear exclusion method when not in exclusion mode
                                if f"exclusion_method_{selected_participant}" in st.session_state:
                                    del st.session_state[f"exclusion_method_{selected_participant}"]
                        with col_mode3:
                            # Plot resolution slider - allow up to all points
                            n_total = plot_data['n_original']
                            # Only show slider if dataset is large enough to benefit from downsampling
                            if n_total > 1000:
                                max_points = n_total  # Allow showing all points
                                # Use saved resolution as default, but show all for small datasets
                                if n_total <= saved_resolution:
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
                            from music_hrv.gui.help_text import EXCLUSION_ZONES_HELP
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
                                from music_hrv.gui.persistence import save_participant_events
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

                # ================== SIGNAL QUALITY & EVENTS (only in events mode) ==================
                if interaction_mode != "Add Exclusions":
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
                                    st.dataframe(pd.DataFrame(gap_data), width='stretch', hide_index=True)

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
                                st.dataframe(pd.DataFrame(seg_data), width='stretch', hide_index=True)

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

                    # Music Change Event Generator (inside events mode)
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
                        raw_events = stored_data.get('events', []) + stored_data.get('manual', [])

                        # Helper to handle dicts (defensive for stale session state)
                        def get_event_attr(evt, attr, default=None):
                            """Get attribute from EventStatus or dict."""
                            if isinstance(evt, dict):
                                return evt.get(attr, default)
                            return getattr(evt, attr, default)

                        # Find all relevant boundary events for music generation
                        boundary_events = {}
                        for evt in raw_events:
                            canonical = get_event_attr(evt, 'canonical')
                            if canonical in ['measurement_start', 'measurement_end', 'pause_start', 'pause_end']:
                                first_ts = get_event_attr(evt, 'first_timestamp')
                                if first_ts:
                                    boundary_events[canonical] = first_ts

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
                        if st.button("Generate Music Events", key=f"gen_music_{selected_participant}"):
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
                                show_toast(f"Created {events_added} music section events", icon="success")
                            else:
                                show_toast("No events created - check boundary events", icon="warning")
                            st.rerun()

                        st.caption("""
                        **How it works:**
                        1. Uses the participant's playlist group to determine music order
                        2. Finds `measurement_start`, `pause_start`, `pause_end`, `measurement_end` events
                        3. Creates music section events every 5 minutes between these boundaries
                        4. Restarts cycle after the pause
                        """)

                # ================== EVENTS MANAGEMENT (only in events mode) ==================
                if interaction_mode != "Add Exclusions":
                    st.markdown("---")

                    # Events table with reordering and inline editing
                    st.markdown("**Events Detected:**")

                    # Get events from session state (already initialized above for the plot)
                    stored_data = st.session_state.participant_events[selected_participant]

                    # Helper to ensure items are EventStatus objects (handles stale session state with dicts)
                    def ensure_event_status(item):
                        """Convert dict to EventStatus if needed."""
                        if isinstance(item, dict):
                            from music_hrv.prep.summaries import EventStatus
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

                        event_time_str = st.text_input(
                            "Time (HH:MM:SS)",
                            value=default_time_str,
                            key=f"event_time_{selected_participant}",
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
                            from music_hrv.prep.summaries import EventStatus
                            import datetime as dt

                            # Determine label
                            label = custom_label if selected_quick_event == "Custom..." else selected_quick_event
                            if not label:
                                show_toast("Please enter an event label", icon="error")
                                return

                            # Determine timestamp
                            if use_offset and first_rr_time:
                                # Calculate from offset
                                offset_delta = dt.timedelta(minutes=offset_min, seconds=offset_sec)
                                event_timestamp = first_rr_time + offset_delta
                            else:
                                # Parse the time string
                                parsed_time = parse_time_str(event_time_str)
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

                            # Validate each section
                            valid_count = 0
                            issue_count = 0

                            for section_code, section_data in sections.items():
                                start_evt = section_data.get("start_event", "")
                                # Support both old (end_event) and new (end_events) format
                                end_evts = section_data.get("end_events", [])
                                if not end_evts and "end_event" in section_data:
                                    end_evts = [section_data["end_event"]]
                                label = section_data.get("label", section_code)
                                expected_dur = section_data.get("expected_duration_min", 0)
                                tolerance = section_data.get("tolerance_min", 1)

                                start_ts = event_timestamps.get(start_evt)

                                # Find the first matching end event
                                end_ts = None
                                matched_end_evt = None
                                for end_evt in end_evts:
                                    if end_evt in event_timestamps:
                                        end_ts = event_timestamps[end_evt]
                                        matched_end_evt = end_evt
                                        break

                                # Check event presence
                                end_evts_str = " | ".join(end_evts) if len(end_evts) > 1 else (end_evts[0] if end_evts else "none")
                                if not start_ts and not end_ts:
                                    st.write(f"**{label}**: missing `{start_evt}` and `{end_evts_str}`")
                                    issue_count += 1
                                elif not start_ts:
                                    st.write(f"**{label}**: missing `{start_evt}`")
                                    issue_count += 1
                                elif not end_ts:
                                    st.write(f"**{label}**: missing any of `{end_evts_str}`")
                                    issue_count += 1
                                else:
                                    # Calculate duration
                                    raw_dur = (normalize_ts(end_ts) - normalize_ts(start_ts)).total_seconds()
                                    excluded = calc_excluded_time(start_ts, end_ts)
                                    actual_dur = (raw_dur - excluded) / 60

                                    excl_note = f" (excl: {excluded/60:.1f}m)" if excluded > 0 else ""
                                    end_evt_note = f" → {matched_end_evt}" if len(end_evts) > 1 else ""

                                    # Check if within tolerance
                                    if expected_dur > 0 and abs(actual_dur - expected_dur) > tolerance:
                                        st.write(f"**{label}**: {actual_dur:.1f}m{excl_note}{end_evt_note} (expected {expected_dur:.0f}±{tolerance:.0f}m)")
                                        issue_count += 1
                                    else:
                                        st.write(f"**{label}**: {actual_dur:.1f}m{excl_note}{end_evt_note}")
                                        valid_count += 1

                            # Summary
                            if issue_count == 0 and valid_count > 0:
                                st.success(f"All {valid_count} section(s) valid")
                            elif valid_count == 0 and issue_count > 0:
                                st.error(f"All {issue_count} section(s) have issues")

                    st.markdown("---")

                    # Build available canonical events
                    available_canonical_events = list(st.session_state.all_events.keys())

                    # Section 1: Event Editing - Individual Cards
                    st.markdown("### Event Management")
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
                                        st.markdown("*")
                                    else:
                                        st.markdown("[!]")

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
                                                show_toast(f"Added '{raw_lower}' as synonym for {canonical_name}", icon="success")
                                            else:
                                                show_toast(f"'{raw_lower}' is already a synonym for {canonical_name}", icon="info")

                                        st.button("+Tag", key=f"syn_{selected_participant}_{idx}",
                                                 on_click=add_synonym, args=(selected_participant, idx),
                                                 help="Add raw label as synonym")

                                with col_time:
                                    # Editable time input with second precision using text
                                    import datetime as dt
                                    current_time_str = event.first_timestamp.strftime("%H:%M:%S") if event.first_timestamp else "00:00:00"

                                    def update_event_time(participant_id, event_idx, original_ts):
                                        """Update event timestamp from time text input."""
                                        key = f"time_{participant_id}_{event_idx}"
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
                                        key=f"time_{selected_participant}_{idx}",
                                        label_visibility="collapsed",
                                        on_change=update_event_time,
                                        args=(selected_participant, idx, event.first_timestamp),
                                        help="HH:MM:SS"
                                    )

                                with col_delete:
                                    # Delete button
                                    if st.button("X", key=f"delete_{selected_participant}_{idx}", help="Delete this event"):
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
                    all_events = [ensure_event_status(e) for e in stored_data['events'] + stored_data['manual']]

                    # Section 2: Event Order with Move Buttons
                    st.markdown("### Event Order")
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

                                status_icon = "❌" if out_of_order else "✅"
                                mapping_badge = "[!]" if not event.canonical else "*"
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

                    # Create download button AFTER the loop (once)
                    if events_data:
                        df_events = pd.DataFrame(events_data)
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

                        df_mapping = pd.DataFrame(mapping_data)
                        st.dataframe(df_mapping, width='stretch', hide_index=True)
                    else:
                        st.info(f"No expected events defined for group '{participant_group}'. Add them in the Event Mapping tab.")

                    # Save/Reset participant events
                    st.markdown("---")
                    from music_hrv.gui.persistence import (
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
                            """Reset participant events to original (from file)."""
                            # Delete saved events
                            delete_participant_events(selected_participant)
                            # Clear from session state so it reloads from original
                            if selected_participant in st.session_state.participant_events:
                                del st.session_state.participant_events[selected_participant]
                            show_toast(f"Reset {selected_participant} to original events", icon="success")

                        # Only show reset if there are saved events
                        saved_participants = list_saved_participant_events()
                        if selected_participant in saved_participants:
                            st.button("Reset to Original",
                                     key=f"reset_{selected_participant}",
                                     on_click=reset_to_original,
                                     help="Discard saved changes and reload original events")

                    with col_status:
                        # Check if participant has saved events
                        if selected_participant in saved_participants:
                            st.caption("Has saved event edits")
                        else:
                            st.caption("Not yet saved")

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
        render_setup_tab()

    # ================== TAB: ANALYSIS ==================
    elif selected_page == "Analysis":
        render_analysis_tab()

    # Record render time for debugging
    st.session_state.last_render_time = (_time.time() - _script_start) * 1000


if __name__ == "__main__":
    main()
