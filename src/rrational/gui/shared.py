"""Shared helpers, constants, and caching for the Music HRV GUI."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

from rrational.cleaning.rr import CleaningConfig
from rrational.io import DEFAULT_ID_PATTERN, PREDEFINED_PATTERNS, load_recording, discover_recordings
from rrational.prep.summaries import load_hrv_logger_preview, load_vns_preview
from rrational.segments.section_normalizer import SectionNormalizer
from rrational.config.sections import SectionsConfig, SectionDefinition, load_sections_config, DEFAULT_SECTIONS_PATH
from rrational.gui.persistence import (
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
    # ValidatedSection System
    "EventCandidate",
    "ValidatedSection",
    "SectionValidationResult",
    "find_event_candidates",
    "validate_section_for_participant",
    "get_validated_sections_for_participant",
    "save_section_selection",
    "save_full_section_validations",
    "load_and_restore_section_validations",
    "get_section_time_range",
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


# ================== ValidatedSection System ==================
# Centralized section validation to ensure consistent boundaries across all features


@dataclass
class EventCandidate:
    """A candidate event that could be used as a section boundary."""
    canonical_name: str
    raw_label: str
    timestamp: datetime
    index: int  # Position in the events list (for disambiguation)

    def display_label(self) -> str:
        """Human-readable label for UI display."""
        time_str = self.timestamp.strftime("%H:%M:%S") if self.timestamp else "?"
        return f"{self.raw_label} @ {time_str}"


@dataclass
class ValidatedSection:
    """A validated section with explicit event references.

    This is the single source of truth for section boundaries.
    All features (Analysis, Artifact Detection, Signal Inspection) should use this.
    """
    section_name: str  # e.g., "rest_pre", "measurement"
    start_event: EventCandidate  # The selected start event
    end_event: EventCandidate  # The selected end event
    beat_count: int = 0  # Number of RR intervals in this section
    duration_s: float = 0.0  # Duration in seconds
    is_user_selected: bool = False  # True if user disambiguated multiple candidates


@dataclass
class SectionValidationResult:
    """Result of attempting to validate a section for a participant."""
    section_name: str
    is_valid: bool
    validated_section: Optional[ValidatedSection] = None
    # For disambiguation UI when multiple candidates exist
    start_candidates: list[EventCandidate] = field(default_factory=list)
    end_candidates: list[EventCandidate] = field(default_factory=list)
    needs_disambiguation: bool = False
    error_message: Optional[str] = None


def find_event_candidates(
    events: list,
    target_canonical_names: list[str],
    normalizer,
) -> list[EventCandidate]:
    """Find all events that match the target canonical names.

    Args:
        events: List of event objects (EventStatus, dict, or raw Event)
        target_canonical_names: List of canonical names to match (e.g., ["rest_pre_start"])
        normalizer: SectionNormalizer for mapping raw labels

    Returns:
        List of EventCandidate objects, sorted by timestamp
    """
    candidates = []

    for idx, event in enumerate(events):
        # Handle different event formats
        if isinstance(event, dict):
            canonical = event.get("canonical")
            raw_label = event.get("raw_label", event.get("label", ""))
            timestamp = event.get("first_timestamp") or event.get("timestamp")
        else:
            # Object with attributes (EventStatus, Event, etc.)
            canonical = getattr(event, "canonical", None)
            raw_label = getattr(event, "raw_label", None) or getattr(event, "label", "")
            timestamp = getattr(event, "first_timestamp", None) or getattr(event, "timestamp", None)

            # If no canonical, normalize the raw label
            if not canonical and raw_label and normalizer:
                canonical = normalizer.normalize(raw_label)

        if not timestamp:
            continue

        # Check if this event matches any target
        if canonical in target_canonical_names or raw_label in target_canonical_names:
            candidates.append(EventCandidate(
                canonical_name=canonical or raw_label,
                raw_label=raw_label,
                timestamp=timestamp,
                index=idx,
            ))

    # Sort by timestamp
    candidates.sort(key=lambda c: c.timestamp)
    return candidates


def validate_section_for_participant(
    section_def: dict,
    events: list,
    normalizer,
    rr_intervals: list = None,
    user_selection: dict = None,
) -> SectionValidationResult:
    """Validate a section for a participant, finding matching events.

    This is the SINGLE centralized function for determining section boundaries.
    All features should use this instead of implementing their own logic.

    Args:
        section_def: Section definition with start_events/end_events lists
        events: Participant's events (saved/edited events from session state)
        normalizer: SectionNormalizer for mapping labels
        rr_intervals: Optional list of RR intervals to calculate beat count
        user_selection: Optional dict with {"start_index": int, "end_index": int}
                        specifying which candidate to use when multiple exist

    Returns:
        SectionValidationResult with validation status and any candidates for disambiguation
    """
    section_name = section_def.get("name", "unknown")

    # Get target event names (support both old and new format)
    start_event_names = section_def.get("start_events", [])
    if not start_event_names and "start_event" in section_def:
        start_event_names = [section_def["start_event"]]
    end_event_names = section_def.get("end_events", [])
    if not end_event_names and "end_event" in section_def:
        end_event_names = [section_def["end_event"]]

    if not start_event_names or not end_event_names:
        return SectionValidationResult(
            section_name=section_name,
            is_valid=False,
            error_message="Section definition missing start or end events",
        )

    # Find all matching candidates
    start_candidates = find_event_candidates(events, start_event_names, normalizer)
    end_candidates = find_event_candidates(events, end_event_names, normalizer)

    if not start_candidates:
        return SectionValidationResult(
            section_name=section_name,
            is_valid=False,
            error_message=f"No start event found ({', '.join(start_event_names)})",
        )

    if not end_candidates:
        return SectionValidationResult(
            section_name=section_name,
            is_valid=False,
            error_message=f"No end event found ({', '.join(end_event_names)})",
        )

    # Determine if disambiguation is needed
    needs_disambiguation = len(start_candidates) > 1 or len(end_candidates) > 1

    # Select which candidate to use
    if user_selection:
        # User has explicitly selected
        start_idx = user_selection.get("start_index", 0)
        end_idx = user_selection.get("end_index", 0)
        start_idx = min(start_idx, len(start_candidates) - 1)
        end_idx = min(end_idx, len(end_candidates) - 1)
        is_user_selected = True
    else:
        # Default: use first candidate
        start_idx = 0
        end_idx = 0
        is_user_selected = False

    selected_start = start_candidates[start_idx]
    selected_end = end_candidates[end_idx]

    # Validate that start comes before end
    if selected_start.timestamp >= selected_end.timestamp:
        return SectionValidationResult(
            section_name=section_name,
            is_valid=False,
            start_candidates=start_candidates,
            end_candidates=end_candidates,
            needs_disambiguation=needs_disambiguation,
            error_message="Start event must come before end event",
        )

    # Calculate beat count and duration if RR intervals provided
    beat_count = 0
    duration_s = 0.0
    if rr_intervals:
        for rr in rr_intervals:
            ts = getattr(rr, "timestamp", None)
            if ts and selected_start.timestamp <= ts <= selected_end.timestamp:
                beat_count += 1
                duration_s += getattr(rr, "rr_ms", 0) / 1000.0

    validated = ValidatedSection(
        section_name=section_name,
        start_event=selected_start,
        end_event=selected_end,
        beat_count=beat_count,
        duration_s=duration_s,
        is_user_selected=is_user_selected,
    )

    return SectionValidationResult(
        section_name=section_name,
        is_valid=True,
        validated_section=validated,
        start_candidates=start_candidates,
        end_candidates=end_candidates,
        needs_disambiguation=needs_disambiguation,
    )


def get_validated_sections_for_participant(
    participant_id: str,
    sections_config: dict,
    normalizer,
    rr_intervals: list = None,
) -> dict[str, SectionValidationResult]:
    """Get all validated sections for a participant.

    This retrieves saved events from session state and validates all sections.
    User selections are loaded from session state if available.

    Args:
        participant_id: The participant ID
        sections_config: Dict of section definitions (from st.session_state.sections)
        normalizer: SectionNormalizer for mapping labels
        rr_intervals: Optional list of RR intervals for beat counting

    Returns:
        Dict mapping section_name to SectionValidationResult
    """
    # Get saved events for this participant
    # Events are stored in participant_events[participant_id] dict with 'events' and 'manual' keys
    participant_events = st.session_state.get("participant_events", {})
    participant_data = participant_events.get(participant_id, {})
    saved_events = participant_data.get("events", []) + participant_data.get("manual", [])

    if not saved_events:
        # No saved events - return empty results
        return {
            name: SectionValidationResult(
                section_name=name,
                is_valid=False,
                error_message="No events available for participant",
            )
            for name in sections_config.keys()
        }

    # Get any user selections for disambiguation
    user_selections_key = f"section_selections_{participant_id}"
    user_selections = st.session_state.get(user_selections_key, {})

    results = {}
    for section_name, section_def in sections_config.items():
        # Add name to def for convenience
        section_def_with_name = {**section_def, "name": section_name}

        # Get user selection for this section if exists
        user_selection = user_selections.get(section_name)

        result = validate_section_for_participant(
            section_def=section_def_with_name,
            events=saved_events,
            normalizer=normalizer,
            rr_intervals=rr_intervals,
            user_selection=user_selection,
        )
        results[section_name] = result

    return results


def save_section_selection(
    participant_id: str,
    section_name: str,
    start_index: int,
    end_index: int,
):
    """Save user's disambiguation choice for a section.

    Args:
        participant_id: The participant ID
        section_name: Name of the section (e.g., "rest_pre")
        start_index: Index of selected start candidate
        end_index: Index of selected end candidate
    """
    key = f"section_selections_{participant_id}"
    if key not in st.session_state:
        st.session_state[key] = {}

    st.session_state[key][section_name] = {
        "start_index": start_index,
        "end_index": end_index,
    }


def save_full_section_validations(participant_id: str):
    """Save the full section validation state for a participant to disk.

    This persists all section validation data explicitly, including:
    - Group membership
    - For each section: validity, events, disambiguation choices, etc.

    Call this whenever section validations change to ensure data is persisted.
    """
    from rrational.gui.persistence import save_section_validations

    # Get participant's group
    group = st.session_state.participant_groups.get(participant_id, "Default")

    # Get the group's sections config
    group_data = st.session_state.groups.get(group, {})
    if isinstance(group_data, dict):
        selected_sections = group_data.get("selected_sections", [])
    else:
        selected_sections = []

    # Get sections config
    sections_config = st.session_state.get("sections", {})
    normalizer = st.session_state.get("normalizer")

    if not normalizer or not sections_config:
        return

    # Get validation results for this participant
    validation_results = get_validated_sections_for_participant(
        participant_id,
        sections_config,
        normalizer,
        selected_sections=selected_sections if selected_sections else None,
    )

    # Build explicit section validation state
    section_validations = {}

    for section_name, result in validation_results.items():
        section_data = {
            "is_valid": result.is_valid,
            "needs_disambiguation": result.needs_disambiguation,
            "error_message": result.error_message,
            "start_candidates_count": len(result.start_candidates),
            "end_candidates_count": len(result.end_candidates),
            "missing_start": len(result.start_candidates) == 0,
            "missing_end": len(result.end_candidates) == 0,
        }

        # Add validated section details if valid
        if result.validated_section:
            vs = result.validated_section
            section_data["start_event"] = {
                "canonical": vs.start_event.canonical_name,
                "raw_label": vs.start_event.raw_label,
                "timestamp": vs.start_event.timestamp.isoformat() if vs.start_event.timestamp else None,
                "index": vs.start_event.index,
            }
            section_data["end_event"] = {
                "canonical": vs.end_event.canonical_name,
                "raw_label": vs.end_event.raw_label,
                "timestamp": vs.end_event.timestamp.isoformat() if vs.end_event.timestamp else None,
                "index": vs.end_event.index,
            }
            section_data["manually_selected"] = vs.is_user_selected
            section_data["duration_s"] = vs.duration_s
            section_data["beat_count"] = vs.beat_count

        # Store user's selection indices
        selections_key = f"section_selections_{participant_id}"
        user_selections = st.session_state.get(selections_key, {})
        if section_name in user_selections:
            section_data["selected_start_index"] = user_selections[section_name].get("start_index", 0)
            section_data["selected_end_index"] = user_selections[section_name].get("end_index", 0)

        section_validations[section_name] = section_data

    # Save to disk
    project_path = st.session_state.get("current_project")
    data_dir = st.session_state.get("data_dir")

    save_section_validations(
        participant_id=participant_id,
        group=group,
        section_validations=section_validations,
        data_dir=data_dir,
        project_path=project_path,
    )


def load_and_restore_section_validations(participant_id: str) -> bool:
    """Load saved section validations and restore to session state.

    This restores the user's section selection indices from the explicit
    validation file, ensuring disambiguation choices are preserved.

    Returns:
        True if validations were loaded, False if none existed.
    """
    from rrational.gui.persistence import load_section_validations

    project_path = st.session_state.get("current_project")
    data_dir = st.session_state.get("data_dir")

    saved = load_section_validations(
        participant_id=participant_id,
        data_dir=data_dir,
        project_path=project_path,
    )

    if not saved:
        return False

    # Restore section selections to session state
    selections_key = f"section_selections_{participant_id}"
    if selections_key not in st.session_state:
        st.session_state[selections_key] = {}

    sections = saved.get("sections", {})
    for section_name, section_data in sections.items():
        start_idx = section_data.get("selected_start_index")
        end_idx = section_data.get("selected_end_index")

        if start_idx is not None or end_idx is not None:
            st.session_state[selections_key][section_name] = {
                "start_index": start_idx if start_idx is not None else 0,
                "end_index": end_idx if end_idx is not None else 0,
            }

    return True


def get_section_time_range(
    participant_id: str,
    section_name: str,
    sections_config: dict,
    normalizer,
) -> tuple[Optional[datetime], Optional[datetime]]:
    """Convenience function to get start/end timestamps for a section.

    This is the ONLY function that should be used to get section boundaries.
    Replaces all ad-hoc implementations throughout the codebase.

    Args:
        participant_id: The participant ID
        section_name: Name of the section
        sections_config: Dict of section definitions
        normalizer: SectionNormalizer

    Returns:
        Tuple of (start_timestamp, end_timestamp), or (None, None) if invalid
    """
    if section_name not in sections_config:
        return None, None

    results = get_validated_sections_for_participant(
        participant_id=participant_id,
        sections_config=sections_config,
        normalizer=normalizer,
    )

    result = results.get(section_name)
    if not result or not result.is_valid or not result.validated_section:
        return None, None

    section = result.validated_section
    return section.start_event.timestamp, section.end_event.timestamp


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
    """Create a SectionNormalizer that merges default patterns with GUI-defined events.

    The normalizer uses patterns from sections.yml as the base, then adds any
    additional synonyms defined in the GUI. This ensures German labels like
    'messung start' are properly matched even if not explicitly configured.

    GUI synonyms are treated as EXACT matches (escaped for regex, full-string match).
    Default patterns from sections.yml are treated as regex patterns.
    """
    # Load default patterns from sections.yml
    default_config = load_sections_config(DEFAULT_SECTIONS_PATH)

    # Build canonical order: start with default order, then add GUI-only events
    canonical_order = list(default_config.canonical_order)
    for name in gui_events_dict.keys():
        if name not in canonical_order:
            canonical_order.append(name)

    # Build sections_dict in canonical order (order matters for pattern matching!)
    sections_dict = {}
    for event_name in canonical_order:
        # Start with default patterns if available
        default_def = default_config.sections.get(event_name)
        default_synonyms = list(default_def.synonyms) if default_def else []

        # Get GUI-defined synonyms and convert to exact-match patterns
        # GUI synonyms are user-entered literal strings, not regex
        gui_synonyms_raw = gui_events_dict.get(event_name, [])
        gui_synonyms = [f"^{re.escape(s)}$" for s in gui_synonyms_raw if s]

        # Merge: GUI exact-match patterns first (higher priority), then default regex patterns
        merged_synonyms = gui_synonyms + [s for s in default_synonyms if s not in gui_synonyms]

        sections_dict[event_name] = SectionDefinition(
            name=event_name,
            synonyms=tuple(merged_synonyms),
            required=default_def.required if default_def else False,
            description=default_def.description if default_def else None,
            group=default_def.group if default_def else None
        )

    config = SectionsConfig(
        version=1,
        canonical_order=tuple(canonical_order),
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

    # Get project path for loading config
    project_path = st.session_state.get("current_project")

    # Load persisted groups
    if "groups" not in st.session_state:
        loaded_groups = load_groups(project_path)
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
        loaded_events = load_events(project_path)
        if not loaded_events:
            st.session_state.all_events = DEFAULT_CANONICAL_EVENTS.copy()
        else:
            st.session_state.all_events = loaded_events

    # Create normalizer from GUI events - always recreate to pick up code/config changes
    st.session_state.normalizer = create_gui_normalizer(st.session_state.all_events)

    # Load participant-specific data
    if "participant_groups" not in st.session_state or "event_order" not in st.session_state:
        loaded_participants = load_participants(project_path)
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
            # Load section selections (user disambiguation choices for section boundaries)
            for pid, data in loaded_participants.items():
                section_selections = data.get("section_selections", {})
                if section_selections:
                    st.session_state[f"section_selections_{pid}"] = section_selections
        else:
            st.session_state.participant_groups = {}
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
    """Save participant-specific data (groups, playlists, labels, event orders, manual events, section selections)."""
    project_path = st.session_state.get("current_project")
    participants_data = {}

    # Collect all participant IDs that have any data
    all_participant_ids = set(
        list(st.session_state.participant_groups.keys()) +
        list(st.session_state.get("participant_playlists", {}).keys()) +
        list(st.session_state.get("participant_labels", {}).keys()) +
        list(st.session_state.event_order.keys()) +
        list(st.session_state.manual_events.keys())
    )

    # Also include any participants with section selections
    for key in st.session_state.keys():
        if key.startswith("section_selections_"):
            pid = key[len("section_selections_"):]
            all_participant_ids.add(pid)

    for pid in all_participant_ids:
        # Get section selections for this participant
        section_selections_key = f"section_selections_{pid}"
        section_selections = st.session_state.get(section_selections_key, {})

        participants_data[pid] = {
            "group": st.session_state.participant_groups.get(pid, "Default"),
            "playlist": st.session_state.get("participant_playlists", {}).get(pid, ""),
            "label": st.session_state.get("participant_labels", {}).get(pid, ""),
            "event_order": st.session_state.event_order.get(pid, []),
            "manual_events": st.session_state.manual_events.get(pid, []),
            "section_selections": section_selections,  # User-selected section boundaries
        }

    save_participants(participants_data, project_path)


def update_normalizer():
    """Update the normalizer when events are added/removed in GUI."""
    st.session_state.normalizer = create_gui_normalizer(st.session_state.all_events)
    cached_load_hrv_logger_preview.clear()
    cached_load_vns_preview.clear()


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
    st.session_state.last_save_time = time.time()


def validate_regex_pattern(pattern):
    """Validate regex pattern and return error message if invalid."""
    try:
        re.compile(pattern)
        return None
    except re.error as e:
        return str(e)


def extract_section_rr_intervals(recording, section_def, normalizer, saved_events=None, participant_id=None):
    """Extract RR intervals for a specific section based on start/end events.

    Args:
        recording: Recording object with rr_intervals and events
        section_def: Section definition dict with start_events/end_events (lists) or
                     start_event/end_event (legacy single values). Must include "name" key
                     for centralized validation.
        normalizer: SectionNormalizer for mapping labels to canonical names
        saved_events: Optional list of saved/edited events (EventStatus objects or dicts).
                     If provided, uses these instead of recording.events.
                     This allows using user-edited events from session state.
        participant_id: Optional participant ID. If provided, uses centralized validation
                        which respects user disambiguation selections.
    """
    section_name = section_def.get("name")

    # If we have participant_id and section_name, use centralized validation for consistency
    if participant_id and section_name:
        sections_config = st.session_state.get("sections", {})
        if section_name in sections_config:
            start_ts, end_ts = get_section_time_range(
                participant_id=participant_id,
                section_name=section_name,
                sections_config=sections_config,
                normalizer=normalizer,
            )
            if start_ts and end_ts:
                section_rr = []
                for rr in recording.rr_intervals:
                    if rr.timestamp and start_ts <= rr.timestamp <= end_ts:
                        section_rr.append(rr)
                return section_rr if section_rr else None

    # Fallback: use legacy logic (for backwards compatibility)
    # Support both old (start_event/end_event) and new (start_events/end_events) format
    start_event_names = section_def.get("start_events", [])
    if not start_event_names and "start_event" in section_def:
        start_event_names = [section_def["start_event"]]
    end_event_names = section_def.get("end_events", [])
    if not end_event_names and "end_event" in section_def:
        end_event_names = [section_def["end_event"]]

    if not start_event_names or not end_event_names:
        return None

    start_ts = None
    end_ts = None

    # Use saved events if provided, otherwise fall back to recording.events
    if saved_events:
        # Saved events are EventStatus objects or dicts with canonical/first_timestamp
        for event in saved_events:
            # Handle both EventStatus objects and dicts
            if isinstance(event, dict):
                canonical = event.get("canonical")
                timestamp = event.get("first_timestamp")
                raw_label = event.get("raw_label", "")
            else:
                canonical = getattr(event, "canonical", None)
                timestamp = getattr(event, "first_timestamp", None)
                raw_label = getattr(event, "raw_label", "")

            if not timestamp:
                continue

            # Check canonical name (already normalized in saved events)
            if canonical in start_event_names:
                start_ts = timestamp
            elif canonical in end_event_names:
                if end_ts is None:
                    end_ts = timestamp
            # Also check raw label as fallback
            elif raw_label in start_event_names:
                start_ts = timestamp
            elif raw_label in end_event_names:
                if end_ts is None:
                    end_ts = timestamp
    else:
        # Fall back to recording.events (raw events from file)
        for event in recording.events:
            label = event.label
            canonical = normalizer.normalize(label)

            # First check if label is already a canonical name (for manual events)
            if label in start_event_names and event.timestamp:
                start_ts = event.timestamp
            elif label in end_event_names and event.timestamp:
                if end_ts is None:
                    end_ts = event.timestamp
            elif canonical in start_event_names and event.timestamp:
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
        return "[OK]"
    elif combined >= 50:
        return "[!]"
    else:
        return "[X]"


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
    """Cache loaded recording data for instant access."""
    from rrational.io.hrv_logger import RecordingBundle
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
    from rrational.cleaning.rr import clean_rr_intervals, RRInterval
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
def cached_load_vns_recording(vns_paths_tuple: tuple, participant_id: str, use_corrected: bool = False):
    """Cache loaded VNS recording data for instant access.

    Args:
        vns_paths_tuple: Tuple of VNS file path strings (for cache key hashability)
        participant_id: Participant identifier
        use_corrected: Whether to use corrected RR values from VNS files
    """
    from rrational.io.vns_analyse import VNSRecordingBundle, load_vns_recording
    bundle = VNSRecordingBundle(
        participant_id=participant_id,
        file_paths=[Path(p) for p in vns_paths_tuple],
    )
    recording = load_vns_recording(bundle, use_corrected=use_corrected)

    # Serialize file segments for caching
    file_segments = None
    if recording.file_segments:
        file_segments = [
            {
                'file_name': seg.file_path.name,
                'start_time': seg.start_time,
                'end_time': seg.end_time,
                'duration_ms': seg.duration_ms,
                'beat_count': seg.beat_count,
            }
            for seg in recording.file_segments
        ]

    # Serialize gaps
    gaps = None
    if recording.gaps:
        gaps = [
            {
                'after_file': gap.after_file.name,
                'before_file': gap.before_file.name,
                'gap_start': gap.gap_start,
                'gap_end': gap.gap_end,
                'gap_duration_s': gap.gap_duration_s,
            }
            for gap in recording.gaps
        ]

    # Serialize overlaps
    overlaps = None
    if recording.overlaps:
        overlaps = [
            {
                'file1': ov.file1.name,
                'file2': ov.file2.name,
                'overlap_start': ov.overlap_start,
                'overlap_end': ov.overlap_end,
                'overlap_duration_s': ov.overlap_duration_s,
            }
            for ov in recording.overlaps
        ]

    return {
        'rr_intervals': [(rr.timestamp, rr.rr_ms, rr.elapsed_ms) for rr in recording.rr_intervals],
        'events': [(e.label, e.timestamp) for e in recording.events],
        'raw_events': [],  # VNS doesn't have duplicate tracking
        'file_segments': file_segments,
        'gaps': gaps,
        'overlaps': overlaps,
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
