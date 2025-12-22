"""Persistence layer for GUI configuration (groups, events, sections)."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any


CONFIG_DIR = Path.home() / ".music_hrv"
GROUPS_FILE = CONFIG_DIR / "groups.yml"
EVENTS_FILE = CONFIG_DIR / "events.yml"
SECTIONS_FILE = CONFIG_DIR / "sections.yml"
PARTICIPANTS_FILE = CONFIG_DIR / "participants.yml"
PLAYLIST_GROUPS_FILE = CONFIG_DIR / "playlist_groups.yml"
MUSIC_LABELS_FILE = CONFIG_DIR / "music_labels.yml"
PROTOCOL_FILE = CONFIG_DIR / "protocol.yml"
PARTICIPANT_EVENTS_FILE = CONFIG_DIR / "participant_events.yml"
SETTINGS_FILE = CONFIG_DIR / "settings.yml"

# Default settings
DEFAULT_SETTINGS = {
    "data_folder": "",  # Empty = use file picker
    "plot_resolution": 5000,
    "plot_options": {
        "show_events": True,
        "show_exclusions": True,
        "show_music_sections": True,
        "show_music_events": False,
        "show_artifacts": False,
        "show_variability": False,
        "show_gaps": True,
        "gap_threshold": 15.0,
    },
}


def ensure_config_dir() -> None:
    """Create config directory if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def save_groups(groups: dict[str, Any]) -> None:
    """Save groups configuration to YAML."""
    ensure_config_dir()
    with open(GROUPS_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(groups, f, default_flow_style=False, allow_unicode=True)


def load_groups() -> dict[str, Any]:
    """Load groups configuration from YAML."""
    if not GROUPS_FILE.exists():
        return {}
    with open(GROUPS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_events(events: dict[str, list[str]]) -> None:
    """Save events configuration to YAML."""
    ensure_config_dir()
    with open(EVENTS_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(events, f, default_flow_style=False, allow_unicode=True)


def load_events() -> dict[str, list[str]]:
    """Load events configuration from YAML."""
    if not EVENTS_FILE.exists():
        return {}
    with open(EVENTS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_sections(sections: dict[str, Any]) -> None:
    """Save sections configuration to YAML."""
    ensure_config_dir()
    with open(SECTIONS_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(sections, f, default_flow_style=False, allow_unicode=True)


def load_sections() -> dict[str, Any]:
    """Load sections configuration from YAML."""
    if not SECTIONS_FILE.exists():
        return {}
    with open(SECTIONS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_participants(participants_data: dict[str, Any]) -> None:
    """Save participants configuration to YAML.

    Format:
    {
        "participant_id": {
            "group": "group_name",
            "event_order": ["event1", "event2", ...],
            "manual_events": [...]
        }
    }
    """
    ensure_config_dir()
    with open(PARTICIPANTS_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(participants_data, f, default_flow_style=False, allow_unicode=True)


def load_participants() -> dict[str, Any]:
    """Load participants configuration from YAML."""
    if not PARTICIPANTS_FILE.exists():
        return {}
    with open(PARTICIPANTS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_playlist_groups(playlist_groups: dict[str, Any]) -> None:
    """Save playlist/randomization groups configuration to YAML.

    Format:
    {
        "R1": {
            "label": "Randomization 1",
            "music_order": ["music_1", "music_3", "music_2"]
        },
        "R2": {
            "label": "Randomization 2",
            "music_order": ["music_2", "music_1", "music_3"]
        }
    }
    """
    ensure_config_dir()
    with open(PLAYLIST_GROUPS_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(playlist_groups, f, default_flow_style=False, allow_unicode=True)


def load_playlist_groups() -> dict[str, Any]:
    """Load playlist/randomization groups configuration from YAML."""
    if not PLAYLIST_GROUPS_FILE.exists():
        return {}
    with open(PLAYLIST_GROUPS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_music_labels(music_labels: dict[str, Any]) -> None:
    """Save music labels configuration to YAML.

    Format:
    {
        "music_1": {
            "label": "Music 1",
            "description": "Brandenburg Concerto No. 3 - Bach"
        },
        "music_2": {
            "label": "Music 2",
            "description": "Moonlight Sonata - Beethoven"
        }
    }
    """
    ensure_config_dir()
    with open(MUSIC_LABELS_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(music_labels, f, default_flow_style=False, allow_unicode=True)


def load_music_labels() -> dict[str, Any]:
    """Load music labels configuration from YAML."""
    if not MUSIC_LABELS_FILE.exists():
        return {}
    with open(MUSIC_LABELS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_protocol(protocol: dict[str, Any]) -> None:
    """Save protocol configuration to YAML.

    Format:
    {
        "expected_duration_min": 90.0,
        "section_length_min": 5.0,
        "pre_pause_sections": 9,
        "post_pause_sections": 9,
        "min_section_duration_min": 4.0,
        "min_section_beats": 100,
        "mismatch_strategy": "flag_only"
    }
    """
    ensure_config_dir()
    with open(PROTOCOL_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(protocol, f, default_flow_style=False, allow_unicode=True)


def load_protocol() -> dict[str, Any]:
    """Load protocol configuration from YAML."""
    if not PROTOCOL_FILE.exists():
        return {
            "expected_duration_min": 90.0,
            "section_length_min": 5.0,
            "pre_pause_sections": 9,
            "post_pause_sections": 9,
            "min_section_duration_min": 4.0,
            "min_section_beats": 100,
            "mismatch_strategy": "flag_only",
        }
    with open(PROTOCOL_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_participant_events(participant_id: str, events_data: dict[str, Any]) -> None:
    """Save participant events (edited events) to YAML.

    Format per participant:
    {
        "participant_id": {
            "events": [
                {"raw_label": "...", "canonical": "...", "timestamp": "ISO8601", ...},
                ...
            ],
            "manual": [...],
            "music_events": [...],
            "exclusion_zones": [
                {"start": "ISO8601", "end": "ISO8601", "reason": "...", "exclude_from_duration": true},
                ...
            ]
        }
    }
    """
    ensure_config_dir()

    # Load existing data
    all_events = {}
    if PARTICIPANT_EVENTS_FILE.exists():
        with open(PARTICIPANT_EVENTS_FILE, "r", encoding="utf-8") as f:
            all_events = yaml.safe_load(f) or {}

    # Convert EventStatus objects to serializable dicts
    serialized = {"events": [], "manual": [], "music_events": [], "exclusion_zones": []}

    for key in ["events", "manual", "music_events"]:
        for evt in events_data.get(key, []):
            evt_dict = {
                "raw_label": getattr(evt, "raw_label", str(evt)),
                "canonical": getattr(evt, "canonical", None),
                "first_timestamp": getattr(evt, "first_timestamp", None),
                "last_timestamp": getattr(evt, "last_timestamp", None),
            }
            # Convert datetime to ISO string for YAML
            if evt_dict["first_timestamp"]:
                evt_dict["first_timestamp"] = evt_dict["first_timestamp"].isoformat()
            if evt_dict["last_timestamp"]:
                evt_dict["last_timestamp"] = evt_dict["last_timestamp"].isoformat()
            serialized[key].append(evt_dict)

    # Handle exclusion zones (already dicts with serializable data)
    for zone in events_data.get("exclusion_zones", []):
        zone_dict = {
            "start": zone.get("start"),
            "end": zone.get("end"),
            "reason": zone.get("reason", ""),
            "exclude_from_duration": zone.get("exclude_from_duration", True),
        }
        # Convert datetime to ISO string if needed
        if zone_dict["start"] and hasattr(zone_dict["start"], "isoformat"):
            zone_dict["start"] = zone_dict["start"].isoformat()
        if zone_dict["end"] and hasattr(zone_dict["end"], "isoformat"):
            zone_dict["end"] = zone_dict["end"].isoformat()
        serialized["exclusion_zones"].append(zone_dict)

    all_events[participant_id] = serialized

    with open(PARTICIPANT_EVENTS_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(all_events, f, default_flow_style=False, allow_unicode=True)


def load_participant_events(participant_id: str) -> dict[str, Any] | None:
    """Load saved participant events from YAML.

    Returns None if no saved events exist for this participant.
    """
    if not PARTICIPANT_EVENTS_FILE.exists():
        return None

    with open(PARTICIPANT_EVENTS_FILE, "r", encoding="utf-8") as f:
        all_events = yaml.safe_load(f) or {}

    if participant_id not in all_events:
        return None

    return all_events[participant_id]


def delete_participant_events(participant_id: str) -> bool:
    """Delete saved events for a participant (reset to original).

    Returns True if events were deleted, False if none existed.
    """
    if not PARTICIPANT_EVENTS_FILE.exists():
        return False

    with open(PARTICIPANT_EVENTS_FILE, "r", encoding="utf-8") as f:
        all_events = yaml.safe_load(f) or {}

    if participant_id not in all_events:
        return False

    del all_events[participant_id]

    with open(PARTICIPANT_EVENTS_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(all_events, f, default_flow_style=False, allow_unicode=True)

    return True


def list_saved_participant_events() -> list[str]:
    """List all participant IDs that have saved events."""
    if not PARTICIPANT_EVENTS_FILE.exists():
        return []

    with open(PARTICIPANT_EVENTS_FILE, "r", encoding="utf-8") as f:
        all_events = yaml.safe_load(f) or {}

    return list(all_events.keys())


def save_settings(settings: dict[str, Any]) -> None:
    """Save application settings to YAML.

    Format:
    {
        "data_folder": "/path/to/data",
        "plot_resolution": 5000,
        "plot_options": {
            "show_events": true,
            "show_exclusions": true,
            ...
        }
    }
    """
    ensure_config_dir()
    # Merge with defaults to ensure all keys exist
    merged = {**DEFAULT_SETTINGS, **settings}
    if "plot_options" in settings:
        merged["plot_options"] = {**DEFAULT_SETTINGS["plot_options"], **settings["plot_options"]}
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, default_flow_style=False, allow_unicode=True)


def load_settings() -> dict[str, Any]:
    """Load application settings from YAML, with defaults for missing keys."""
    if not SETTINGS_FILE.exists():
        return DEFAULT_SETTINGS.copy()

    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
        saved = yaml.safe_load(f) or {}

    # Merge with defaults to handle missing keys
    result = {**DEFAULT_SETTINGS, **saved}
    if "plot_options" in saved:
        result["plot_options"] = {**DEFAULT_SETTINGS["plot_options"], **saved.get("plot_options", {})}
    else:
        result["plot_options"] = DEFAULT_SETTINGS["plot_options"].copy()

    return result


def get_setting(key: str, default: Any = None) -> Any:
    """Get a single setting value."""
    settings = load_settings()
    if "." in key:
        # Support nested keys like "plot_options.show_events"
        parts = key.split(".")
        value = settings
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return default
        return value if value is not None else default
    return settings.get(key, default)
