"""Shared helpers, constants, and caching for the Music HRV GUI."""

from __future__ import annotations

import re
import time
from pathlib import Path

import streamlit as st

from music_hrv.cleaning.rr import CleaningConfig
from music_hrv.io import DEFAULT_ID_PATTERN, PREDEFINED_PATTERNS, load_recording, discover_recordings
from music_hrv.prep.summaries import load_hrv_logger_preview, load_vns_preview
from music_hrv.segments.section_normalizer import SectionNormalizer
from music_hrv.config.sections import SectionsConfig, SectionDefinition
from music_hrv.gui.persistence import (
    save_groups,
    load_groups,
    save_events,
    load_events,
    save_sections,
    save_participants,
    load_participants,
)

# Re-export for convenience
__all__ = [
    # Constants
    "DEFAULT_CANONICAL_EVENTS",
    "DEFAULT_ID_PATTERN",
    "PREDEFINED_PATTERNS",
    "NEUROKIT_AVAILABLE",
    # Functions
    "get_neurokit",
    "get_matplotlib",
    "create_gui_normalizer",
    "save_all_config",
    "save_participant_data",
    "update_normalizer",
    "show_toast",
    "auto_save_config",
    "validate_regex_pattern",
    "extract_section_rr_intervals",
    "filter_exclusion_zones",
    "detect_quality_changepoints",
    "get_quality_badge",
    "detect_time_gaps",
    "detect_artifacts_fixpeaks",
    "scroll_to_top",
    "get_participant_list",
    "get_summary_dict",
    # Cached functions
    "cached_load_hrv_logger_preview",
    "cached_load_vns_preview",
    "cached_load_participants",
    "cached_discover_recordings",
    "cached_load_recording",
    "cached_load_vns_recording",
    "cached_clean_rr_intervals",
    "cached_quality_analysis",
    "cached_get_plot_data",
    # Session state
    "init_session_state",
]

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


def create_gui_normalizer(gui_events_dict):
    """Create a custom SectionNormalizer that ONLY uses GUI-defined events."""
    sections_dict = {}
    for event_name, synonyms in gui_events_dict.items():
        sections_dict[event_name] = SectionDefinition(
            name=event_name,
            synonyms=tuple(synonyms) if synonyms else (),
            required=False,
            description=None,
            group=None
        )

    config = SectionsConfig(
        version=1,
        canonical_order=tuple(gui_events_dict.keys()),
        sections=sections_dict,
        groups={}
    )

    return SectionNormalizer(config=config, fallback_label="unknown")


def init_session_state():
    """Initialize all session state variables."""
    if "data_dir" not in st.session_state:
        st.session_state.data_dir = None
    if "summaries" not in st.session_state:
        st.session_state.summaries = []
    if "cleaning_config" not in st.session_state:
        st.session_state.cleaning_config = CleaningConfig()

    # Load persisted groups
    if "groups" not in st.session_state:
        loaded_groups = load_groups()
        if not loaded_groups:
            st.session_state.groups = {
                "Default": {
                    "label": "Default Group",
                    "expected_events": DEFAULT_CANONICAL_EVENTS.copy(),
                    "selected_sections": []
                }
            }
        else:
            for group_name, group_data in loaded_groups.items():
                if "selected_sections" not in group_data:
                    group_data["selected_sections"] = []
            st.session_state.groups = loaded_groups

    # Load persisted events
    if "all_events" not in st.session_state:
        loaded_events = load_events()
        if not loaded_events:
            st.session_state.all_events = DEFAULT_CANONICAL_EVENTS.copy()
        else:
            st.session_state.all_events = loaded_events

    # Create normalizer from GUI events
    if "normalizer" not in st.session_state:
        st.session_state.normalizer = create_gui_normalizer(st.session_state.all_events)

    # Load participant-specific data
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
    """Save participant-specific data (groups, playlists, labels, event orders, manual events)."""
    participants_data = {}
    all_participant_ids = set(
        list(st.session_state.participant_groups.keys()) +
        list(st.session_state.get("participant_playlists", {}).keys()) +
        list(st.session_state.get("participant_labels", {}).keys()) +
        list(st.session_state.event_order.keys()) +
        list(st.session_state.manual_events.keys())
    )

    for pid in all_participant_ids:
        participants_data[pid] = {
            "group": st.session_state.participant_groups.get(pid, "Default"),
            "playlist": st.session_state.get("participant_playlists", {}).get(pid, ""),
            "label": st.session_state.get("participant_labels", {}).get(pid, ""),
            "event_order": st.session_state.event_order.get(pid, []),
            "manual_events": st.session_state.manual_events.get(pid, []),
        }

    save_participants(participants_data)


def update_normalizer():
    """Update the normalizer when events are added/removed in GUI."""
    st.session_state.normalizer = create_gui_normalizer(st.session_state.all_events)
    cached_load_hrv_logger_preview.clear()
    cached_load_vns_preview.clear()


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
    st.session_state.last_save_time = time.time()


def validate_regex_pattern(pattern):
    """Validate regex pattern and return error message if invalid."""
    try:
        re.compile(pattern)
        return None
    except re.error as e:
        return str(e)


def extract_section_rr_intervals(recording, section_def, normalizer):
    """Extract RR intervals for a specific section based on start/end events."""
    start_event_name = section_def.get("start_event")
    # Support both old (end_event) and new (end_events) format
    end_event_names = section_def.get("end_events", [])
    if not end_event_names and "end_event" in section_def:
        end_event_names = [section_def["end_event"]]

    if not start_event_name or not end_event_names:
        return None

    start_ts = None
    end_ts = None

    for event in recording.events:
        label = event.label
        # First check if label is already a canonical name (for manual events)
        if label == start_event_name and event.timestamp:
            start_ts = event.timestamp
        elif label in end_event_names and event.timestamp:
            if end_ts is None:
                end_ts = event.timestamp
        else:
            # Try normalizing for raw labels from file
            canonical = normalizer.normalize(label)
            if canonical == start_event_name and event.timestamp:
                start_ts = event.timestamp
            elif canonical in end_event_names and event.timestamp:
                if end_ts is None:
                    end_ts = event.timestamp

    if not start_ts or not end_ts:
        return None

    section_rr = []
    for rr in recording.rr_intervals:
        if rr.timestamp and start_ts <= rr.timestamp <= end_ts:
            section_rr.append(rr)

    return section_rr if section_rr else None


def filter_exclusion_zones(rr_intervals, exclusion_zones: list[dict]) -> tuple[list, dict]:
    """Filter RR intervals to exclude specified time zones.

    Args:
        rr_intervals: List of RRInterval objects (with .timestamp and .rr_ms attributes)
        exclusion_zones: List of dicts with 'start' and 'end' datetime keys

    Returns:
        Tuple of (filtered_rr_intervals, stats_dict)
        stats_dict contains: n_original, n_excluded, n_remaining, excluded_duration_ms
    """
    import pandas as pd

    if not exclusion_zones or not rr_intervals:
        return rr_intervals, {
            "n_original": len(rr_intervals) if rr_intervals else 0,
            "n_excluded": 0,
            "n_remaining": len(rr_intervals) if rr_intervals else 0,
            "excluded_duration_ms": 0,
            "zones_applied": 0
        }

    # Parse exclusion zone timestamps
    parsed_zones = []
    for zone in exclusion_zones:
        try:
            start = zone.get('start')
            end = zone.get('end')

            # Convert string to datetime if needed
            if isinstance(start, str):
                start = pd.to_datetime(start)
            if isinstance(end, str):
                end = pd.to_datetime(end)

            if start and end:
                parsed_zones.append((start, end))
        except Exception:
            continue

    if not parsed_zones:
        return rr_intervals, {
            "n_original": len(rr_intervals),
            "n_excluded": 0,
            "n_remaining": len(rr_intervals),
            "excluded_duration_ms": 0,
            "zones_applied": 0
        }

    # Filter RR intervals
    filtered = []
    excluded_duration_ms = 0
    n_excluded = 0

    for rr in rr_intervals:
        ts = rr.timestamp
        if ts is None:
            filtered.append(rr)
            continue

        # Make timezone-aware comparison safe
        if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
            ts_naive = ts.replace(tzinfo=None)
        else:
            ts_naive = ts

        is_excluded = False
        for zone_start, zone_end in parsed_zones:
            # Make zone timestamps naive for comparison
            if hasattr(zone_start, 'tzinfo') and zone_start.tzinfo is not None:
                zone_start = zone_start.replace(tzinfo=None)
            if hasattr(zone_end, 'tzinfo') and zone_end.tzinfo is not None:
                zone_end = zone_end.replace(tzinfo=None)

            if zone_start <= ts_naive <= zone_end:
                is_excluded = True
                excluded_duration_ms += rr.rr_ms
                n_excluded += 1
                break

        if not is_excluded:
            filtered.append(rr)

    return filtered, {
        "n_original": len(rr_intervals),
        "n_excluded": n_excluded,
        "n_remaining": len(filtered),
        "excluded_duration_ms": excluded_duration_ms,
        "zones_applied": len(parsed_zones)
    }


def detect_quality_changepoints(rr_values: list[int], change_type: str = "var") -> dict:
    """Detect quality changepoints in RR interval data using NeuroKit2."""
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

        nk = get_neurokit()
        changepoints = nk.signal_changepoints(rr_array, change=change_type, show=False)

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
    """Return a quality badge emoji based on quality score and artifact ratio."""
    artifact_score = 100 - (artifact_ratio * 200)
    artifact_score = max(0, min(100, artifact_score))

    combined = (quality_score + artifact_score) / 2

    if combined >= 75:
        return "🟢"
    elif combined >= 50:
        return "🟡"
    else:
        return "🔴"


def detect_time_gaps(timestamps: list, rr_values: list = None, gap_threshold_s: float = 2.0) -> dict:
    """Detect time gaps (missing data) between consecutive RR intervals."""
    import numpy as np

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
    """Detect and optionally correct artifacts using NeuroKit2's signal_fixpeaks."""
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

        rr_array = np.array(rr_values, dtype=float)
        peak_indices = np.cumsum(rr_array).astype(int)
        peak_indices = np.insert(peak_indices, 0, 0)

        nk = get_neurokit()
        info, corrected_peaks = nk.signal_fixpeaks(
            peak_indices,
            sampling_rate=sampling_rate,
            iterative=True,
            method="Kubios",
            show=False,
        )

        artifacts = {
            "ectopic": len(info.get("ectopic", [])) if isinstance(info.get("ectopic"), (list, np.ndarray)) else 0,
            "missed": len(info.get("missed", [])) if isinstance(info.get("missed"), (list, np.ndarray)) else 0,
            "extra": len(info.get("extra", [])) if isinstance(info.get("extra"), (list, np.ndarray)) else 0,
            "longshort": len(info.get("longshort", [])) if isinstance(info.get("longshort"), (list, np.ndarray)) else 0,
        }

        total_artifacts = sum(artifacts.values())
        artifact_ratio = total_artifacts / len(rr_values) if rr_values else 0

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


# ================== Cached Functions ==================

@st.cache_data(show_spinner=False, ttl=300)
def cached_load_hrv_logger_preview(data_dir_str, pattern, config_dict, gui_events_dict):
    """Cached version of load_hrv_logger_preview for instant navigation."""
    data_path = Path(data_dir_str)
    config = CleaningConfig(
        rr_min_ms=config_dict["rr_min_ms"],
        rr_max_ms=config_dict["rr_max_ms"],
        sudden_change_pct=config_dict["sudden_change_pct"]
    )
    normalizer = create_gui_normalizer(gui_events_dict)
    return load_hrv_logger_preview(data_path, pattern=pattern, config=config, normalizer=normalizer)


@st.cache_data(show_spinner=False, ttl=300)
def cached_load_vns_preview(data_dir_str, pattern, config_dict, gui_events_dict, use_corrected=False):
    """Cached version of load_vns_preview for VNS Analyse data."""
    data_path = Path(data_dir_str)
    config = CleaningConfig(
        rr_min_ms=config_dict["rr_min_ms"],
        rr_max_ms=config_dict["rr_max_ms"],
        sudden_change_pct=config_dict["sudden_change_pct"]
    )
    normalizer = create_gui_normalizer(gui_events_dict)
    return load_vns_preview(data_path, pattern=pattern, config=config, normalizer=normalizer, use_corrected=use_corrected)


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
    """Cache loaded recording data for instant access."""
    from music_hrv.io.hrv_logger import RecordingBundle
    bundle = RecordingBundle(
        participant_id=participant_id,
        rr_paths=[Path(p) for p in rr_paths_tuple],
        events_paths=[Path(p) for p in events_paths_tuple]
    )
    recording, raw_events, _ = load_recording(bundle)
    return {
        'rr_intervals': [(rr.timestamp, rr.rr_ms, rr.elapsed_ms) for rr in recording.rr_intervals],
        'events': [(e.label, e.timestamp) for e in recording.events],
        'raw_events': raw_events
    }


@st.cache_data(show_spinner=False, ttl=300)
def cached_clean_rr_intervals(rr_data_tuple, config_dict):
    """Cache cleaned RR intervals to avoid recomputation."""
    from music_hrv.cleaning.rr import clean_rr_intervals, RRInterval
    rr_intervals = [RRInterval(timestamp=ts, rr_ms=rr, elapsed_ms=elapsed)
                    for ts, rr, elapsed in rr_data_tuple]
    config = CleaningConfig(
        rr_min_ms=config_dict["rr_min_ms"],
        rr_max_ms=config_dict["rr_max_ms"],
        sudden_change_pct=config_dict["sudden_change_pct"]
    )
    cleaned, stats = clean_rr_intervals(rr_intervals, config)
    return [(rr.timestamp, rr.rr_ms) for rr in cleaned if rr.timestamp], stats


@st.cache_data(show_spinner=False, ttl=300)
def cached_quality_analysis(rr_values_tuple, timestamps_tuple):
    """Cache quality changepoint detection results."""
    rr_list = list(rr_values_tuple)
    timestamps_list = list(timestamps_tuple)
    result = detect_quality_changepoints(rr_list, change_type="var")
    n_ts = len(timestamps_list)
    for seg_stats in result["segment_stats"]:
        start_idx = seg_stats["start_idx"]
        end_idx = min(seg_stats["end_idx"], n_ts - 1)
        seg_stats["start_time"] = timestamps_list[start_idx] if start_idx < n_ts else None
        seg_stats["end_time"] = timestamps_list[end_idx] if end_idx < n_ts else None
    return result


@st.cache_data(show_spinner=False, ttl=300)
def cached_get_plot_data(timestamps_tuple, rr_values_tuple, participant_id: str, downsample_threshold: int = 5000):
    """Cache processed plot data (NOT the figure - that's slow to serialize)."""
    timestamps = list(timestamps_tuple)
    rr_values = list(rr_values_tuple)

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


def scroll_to_top():
    """Inject JavaScript to scroll the page to the top.

    This is useful when navigating between participants or sections.
    """
    js = """
    <script>
        var streamlitDoc = window.parent.document;
        streamlitDoc.querySelector('[data-testid="stAppViewContainer"]').scrollTop = 0;
    </script>
    """
    st.components.v1.html(js, height=0)


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
