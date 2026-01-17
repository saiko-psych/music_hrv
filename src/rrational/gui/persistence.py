"""Persistence layer for GUI configuration (groups, events, sections).

Supports both global config (~/.rrational/) and project-based config.
When project_path is provided, config is stored in project/config/ folder.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any


CONFIG_DIR = Path.home() / ".rrational"
LEGACY_CONFIG_DIR = Path.home() / ".music_hrv"  # Pre-v0.7.0 config directory
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
    "auto_load": False,  # Auto-load from default folder on startup
    "accent_color": "#2E86AB",  # UI accent color
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
        "colors": {
            "line": "#2E86AB",  # RR interval line color
            "artifact": "#FF6B6B",  # Artifact marker color
        },
    },
    "recent_projects": [],  # List of recently opened projects
    "max_recent_projects": 10,
    "last_project": "",  # Path to last used project (auto-load on startup)
}


def ensure_config_dir() -> None:
    """Create config directory if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _get_config_path(filename: str, project_path: Path | str | None = None) -> Path:
    """Get the path for a config file, supporting project-based storage.

    Args:
        filename: Name of the config file (e.g., 'groups.yml')
        project_path: If provided (Path or str), returns project/config/{filename}
                      Otherwise returns ~/.rrational/{filename}

    Returns:
        Path to the config file
    """
    if project_path:
        # Ensure project_path is a Path object (handles both Path and str)
        config_dir = Path(project_path) / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / filename
    return CONFIG_DIR / filename


def migrate_legacy_config() -> bool:
    """Migrate configuration from legacy ~/.music_hrv to ~/.rrational.

    This handles the v0.7.0 rename from music_hrv to rrational.
    Migrates files when:
    - New file doesn't exist, OR
    - Legacy file has more content (larger size) than new file
      (indicates new file only has defaults, legacy has real user data)

    Returns:
        True if migration was performed, False if not needed
    """
    import shutil

    if not LEGACY_CONFIG_DIR.exists():
        return False  # No legacy config to migrate

    # List of files to migrate
    legacy_files = [
        "groups.yml",
        "events.yml",
        "sections.yml",
        "participants.yml",
        "playlist_groups.yml",
        "music_labels.yml",
        "protocol.yml",
        "participant_events.yml",
        "settings.yml",
    ]

    migrated_any = False
    ensure_config_dir()

    for filename in legacy_files:
        legacy_file = LEGACY_CONFIG_DIR / filename
        new_file = CONFIG_DIR / filename

        if not legacy_file.exists():
            continue

        # Migrate if: new file doesn't exist OR legacy is larger (has more user data)
        should_migrate = False
        if not new_file.exists():
            should_migrate = True
        else:
            # If legacy file is significantly larger, it likely has real user data
            # while new file only has defaults
            legacy_size = legacy_file.stat().st_size
            new_size = new_file.stat().st_size
            if legacy_size > new_size:
                should_migrate = True

        if should_migrate:
            try:
                shutil.copy2(legacy_file, new_file)
                migrated_any = True
            except Exception:
                pass  # Silently continue if copy fails

    return migrated_any


# --- Groups ---

def save_groups(groups: dict[str, Any], project_path: Path | None = None) -> None:
    """Save groups configuration to YAML.

    Args:
        groups: Groups configuration dict
        project_path: If provided, saves to project/config/groups.yml
    """
    if not project_path:
        ensure_config_dir()
    target = _get_config_path("groups.yml", project_path)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(groups, f, default_flow_style=False, allow_unicode=True)


def load_groups(project_path: Path | None = None) -> dict[str, Any]:
    """Load groups configuration from YAML.

    Args:
        project_path: If provided, loads from project/config/groups.yml

    Returns:
        Groups configuration dict, or empty dict if not found
    """
    target = _get_config_path("groups.yml", project_path)
    if not target.exists():
        return {}
    with open(target, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --- Events ---

def save_events(events: dict[str, list[str]], project_path: Path | None = None) -> None:
    """Save events configuration to YAML.

    Args:
        events: Events configuration dict (canonical -> synonyms)
        project_path: If provided, saves to project/config/events.yml
    """
    if not project_path:
        ensure_config_dir()
    target = _get_config_path("events.yml", project_path)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(events, f, default_flow_style=False, allow_unicode=True)


def load_events(project_path: Path | None = None) -> dict[str, list[str]]:
    """Load events configuration from YAML.

    Args:
        project_path: If provided, loads from project/config/events.yml

    Returns:
        Events configuration dict, or empty dict if not found
    """
    target = _get_config_path("events.yml", project_path)
    if not target.exists():
        return {}
    with open(target, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --- Sections ---

def save_sections(sections: dict[str, Any], project_path: Path | None = None) -> None:
    """Save sections configuration to YAML.

    Args:
        sections: Sections configuration dict
        project_path: If provided, saves to project/config/sections.yml
    """
    if not project_path:
        ensure_config_dir()
    target = _get_config_path("sections.yml", project_path)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(sections, f, default_flow_style=False, allow_unicode=True)


def load_sections(project_path: Path | None = None) -> dict[str, Any]:
    """Load sections configuration from YAML.

    Args:
        project_path: If provided, loads from project/config/sections.yml

    Returns:
        Sections configuration dict, or empty dict if not found
    """
    target = _get_config_path("sections.yml", project_path)
    if not target.exists():
        return {}
    with open(target, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --- Participants ---

def save_participants(participants_data: dict[str, Any], project_path: Path | None = None) -> None:
    """Save participants configuration to YAML.

    Args:
        participants_data: Participant data dict
        project_path: If provided, saves to project/config/participants.yml

    Format:
    {
        "participant_id": {
            "group": "group_name",
            "event_order": ["event1", "event2", ...],
            "manual_events": [...]
        }
    }
    """
    if not project_path:
        ensure_config_dir()
    target = _get_config_path("participants.yml", project_path)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(participants_data, f, default_flow_style=False, allow_unicode=True)


def load_participants(project_path: Path | None = None) -> dict[str, Any]:
    """Load participants configuration from YAML.

    Args:
        project_path: If provided, loads from project/config/participants.yml

    Returns:
        Participants configuration dict, or empty dict if not found
    """
    target = _get_config_path("participants.yml", project_path)
    if not target.exists():
        return {}
    with open(target, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --- Playlist Groups ---

def save_playlist_groups(playlist_groups: dict[str, Any], project_path: Path | None = None) -> None:
    """Save playlist/randomization groups configuration to YAML.

    Args:
        playlist_groups: Playlist groups configuration dict
        project_path: If provided, saves to project/config/playlist_groups.yml

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
    if not project_path:
        ensure_config_dir()
    target = _get_config_path("playlist_groups.yml", project_path)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(playlist_groups, f, default_flow_style=False, allow_unicode=True)


def load_playlist_groups(project_path: Path | None = None) -> dict[str, Any]:
    """Load playlist/randomization groups configuration from YAML.

    Args:
        project_path: If provided, loads from project/config/playlist_groups.yml

    Returns:
        Playlist groups configuration dict, or empty dict if not found
    """
    target = _get_config_path("playlist_groups.yml", project_path)
    if not target.exists():
        return {}
    with open(target, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --- Music Labels ---

def save_music_labels(music_labels: dict[str, Any], project_path: Path | None = None) -> None:
    """Save music labels configuration to YAML.

    Args:
        music_labels: Music labels configuration dict
        project_path: If provided, saves to project/config/music_labels.yml

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
    if not project_path:
        ensure_config_dir()
    target = _get_config_path("music_labels.yml", project_path)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(music_labels, f, default_flow_style=False, allow_unicode=True)


def load_music_labels(project_path: Path | None = None) -> dict[str, Any]:
    """Load music labels configuration from YAML.

    Args:
        project_path: If provided, loads from project/config/music_labels.yml

    Returns:
        Music labels configuration dict, or empty dict if not found
    """
    target = _get_config_path("music_labels.yml", project_path)
    if not target.exists():
        return {}
    with open(target, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --- Protocol ---

def save_protocol(protocol: dict[str, Any], project_path: Path | None = None) -> None:
    """Save protocol configuration to YAML.

    Args:
        protocol: Protocol configuration dict
        project_path: If provided, saves to project/config/protocol.yml

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
    if not project_path:
        ensure_config_dir()
    target = _get_config_path("protocol.yml", project_path)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(protocol, f, default_flow_style=False, allow_unicode=True)


def load_protocol(project_path: Path | None = None) -> dict[str, Any]:
    """Load protocol configuration from YAML.

    Args:
        project_path: If provided, loads from project/config/protocol.yml

    Returns:
        Protocol configuration dict with defaults for missing keys
    """
    target = _get_config_path("protocol.yml", project_path)
    if not target.exists():
        return {
            "expected_duration_min": 90.0,
            "section_length_min": 5.0,
            "pre_pause_sections": 9,
            "post_pause_sections": 9,
            "min_section_duration_min": 4.0,
            "min_section_beats": 100,
            "mismatch_strategy": "flag_only",
        }
    with open(target, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# --- Participant Events ---

def save_participant_events(
    participant_id: str,
    events_data: dict[str, Any],
    data_dir: str | None = None,
    project_path: Path | None = None,
) -> None:
    """Save participant events (edited events) to YAML.

    Storage priority:
    1. If project_path provided: saves to project/processed/{participant_id}_events.yml
    2. If data_dir provided: saves to {data_dir}/../processed/{participant_id}_events.yml
    3. Otherwise: saves to ~/.rrational/participant_events.yml (fallback)

    This keeps event data with the project for portability.
    """
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

    # Determine save location (priority: project > data_dir > global)
    if project_path:
        processed_dir = Path(project_path) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        participant_file = processed_dir / f"{participant_id}_events.yml"

        output_data = {
            "participant_id": participant_id,
            "format_version": "1.0",
            "source_type": "rrational_toolkit",
            **serialized
        }

        with open(participant_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(output_data, f, default_flow_style=False, allow_unicode=True)

    elif data_dir:
        # Save to processed folder (portable with project)
        data_path = Path(data_dir)
        processed_dir = data_path.parent / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        participant_file = processed_dir / f"{participant_id}_events.yml"

        output_data = {
            "participant_id": participant_id,
            "format_version": "1.0",
            "source_type": "rrational_toolkit",
            **serialized
        }

        with open(participant_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(output_data, f, default_flow_style=False, allow_unicode=True)
    else:
        # Fallback: save to app config (no project folder available)
        ensure_config_dir()

        all_events = {}
        if PARTICIPANT_EVENTS_FILE.exists():
            with open(PARTICIPANT_EVENTS_FILE, "r", encoding="utf-8") as f:
                all_events = yaml.safe_load(f) or {}

        all_events[participant_id] = serialized

        with open(PARTICIPANT_EVENTS_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(all_events, f, default_flow_style=False, allow_unicode=True)


def load_participant_events(
    participant_id: str,
    data_dir: str | None = None,
    project_path: Path | None = None,
) -> dict[str, Any] | None:
    """Load saved participant events from YAML.

    Storage priority:
    1. If project_path provided: checks project/processed/{participant_id}_events.yml
    2. If data_dir provided: checks {data_dir}/../processed/{participant_id}_events.yml
    3. Falls back to ~/.rrational/participant_events.yml

    Returns None if no saved events exist for this participant.
    """
    # First, try to load from project folder
    if project_path:
        processed_dir = Path(project_path) / "processed"
        participant_file = processed_dir / f"{participant_id}_events.yml"

        if participant_file.exists():
            with open(participant_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            # Remove metadata fields before returning
            data.pop("participant_id", None)
            data.pop("format_version", None)
            data.pop("source_type", None)
            return data

    # Try to load from data_dir processed folder
    if data_dir:
        data_path = Path(data_dir)
        processed_dir = data_path.parent / "processed"
        participant_file = processed_dir / f"{participant_id}_events.yml"

        if participant_file.exists():
            with open(participant_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            # Remove metadata fields before returning
            data.pop("participant_id", None)
            data.pop("format_version", None)
            data.pop("source_type", None)
            return data

    # Fall back to app config
    if not PARTICIPANT_EVENTS_FILE.exists():
        return None

    with open(PARTICIPANT_EVENTS_FILE, "r", encoding="utf-8") as f:
        all_events = yaml.safe_load(f) or {}

    if participant_id not in all_events:
        return None

    return all_events[participant_id]


def delete_participant_events(
    participant_id: str,
    data_dir: str | None = None,
    project_path: Path | None = None,
) -> bool:
    """Delete saved events for a participant (reset to original).

    Deletes from all locations for backwards compatibility:
    - project/processed/{participant_id}_events.yml (project folder)
    - {data_dir}/../processed/{participant_id}_events.yml (data folder)
    - ~/.rrational/participant_events.yml (app config)

    Returns True if events were deleted from any location, False if none existed.
    """
    deleted_any = False

    # Delete from project folder
    if project_path:
        processed_dir = Path(project_path) / "processed"
        participant_file = processed_dir / f"{participant_id}_events.yml"

        if participant_file.exists():
            participant_file.unlink()
            deleted_any = True

    # Delete from data_dir processed folder
    if data_dir:
        data_path = Path(data_dir)
        processed_dir = data_path.parent / "processed"
        participant_file = processed_dir / f"{participant_id}_events.yml"

        if participant_file.exists():
            participant_file.unlink()
            deleted_any = True

    # Delete from app config
    if PARTICIPANT_EVENTS_FILE.exists():
        with open(PARTICIPANT_EVENTS_FILE, "r", encoding="utf-8") as f:
            all_events = yaml.safe_load(f) or {}

        if participant_id in all_events:
            del all_events[participant_id]

            with open(PARTICIPANT_EVENTS_FILE, "w", encoding="utf-8") as f:
                yaml.safe_dump(all_events, f, default_flow_style=False, allow_unicode=True)

            deleted_any = True

    return deleted_any


def list_saved_participant_events() -> list[str]:
    """List all participant IDs that have saved events."""
    if not PARTICIPANT_EVENTS_FILE.exists():
        return []

    with open(PARTICIPANT_EVENTS_FILE, "r", encoding="utf-8") as f:
        all_events = yaml.safe_load(f) or {}

    return list(all_events.keys())


# --- Settings (always global) ---

def save_settings(settings: dict[str, Any]) -> None:
    """Save application settings to YAML.

    Settings are always saved to ~/.rrational/settings.yml (global).

    Format:
    {
        "data_folder": "/path/to/data",
        "plot_resolution": 5000,
        "plot_options": {...},
        "recent_projects": [...]
    }
    """
    ensure_config_dir()
    # Merge with defaults to ensure all keys exist
    merged = {**DEFAULT_SETTINGS, **settings}
    if "plot_options" in settings:
        merged["plot_options"] = {**DEFAULT_SETTINGS["plot_options"], **settings["plot_options"]}
        # Also merge nested colors within plot_options
        if "colors" in settings["plot_options"]:
            merged["plot_options"]["colors"] = {
                **DEFAULT_SETTINGS["plot_options"]["colors"],
                **settings["plot_options"]["colors"]
            }
        else:
            merged["plot_options"]["colors"] = DEFAULT_SETTINGS["plot_options"]["colors"].copy()
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
        # Also merge nested colors within plot_options
        saved_colors = saved.get("plot_options", {}).get("colors", {})
        result["plot_options"]["colors"] = {
            **DEFAULT_SETTINGS["plot_options"]["colors"],
            **saved_colors
        }
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


def save_last_project(project_path: str | Path | None) -> None:
    """Save the last used project path for auto-load on startup.

    Args:
        project_path: Path to the project, or None/empty to clear
    """
    settings = load_settings()
    settings["last_project"] = str(project_path) if project_path else ""
    save_settings(settings)


def get_last_project() -> str | None:
    """Get the last used project path.

    Returns:
        Path to last project, or None if not set or doesn't exist
    """
    last = get_setting("last_project", "")
    if last and Path(last).exists():
        return last
    return None


# --- Processed Directory ---

def get_processed_dir(
    data_dir: str | Path | None = None,
    project_path: Path | None = None,
) -> Path:
    """Get the processed directory path for storing .rrational files and artifacts.

    Args:
        data_dir: The data directory path (e.g., project/data/raw)
        project_path: Project path (takes priority if provided)

    Priority:
    1. If project_path: returns project/data/processed/
    2. If data_dir: returns {data_dir}/../processed/ (sibling to raw folder)
    3. Otherwise: returns ~/.rrational/exports/

    Returns:
        Path to the processed directory (created if needed)
    """
    if project_path:
        processed_dir = Path(project_path) / "data" / "processed"
    elif data_dir:
        processed_dir = Path(data_dir).parent / "processed"
    else:
        processed_dir = CONFIG_DIR / "exports"

    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir


def list_ready_files_for_participant(
    participant_id: str,
    data_dir: str | Path | None = None,
    project_path: Path | None = None,
) -> list[Path]:
    """List all .rrational ready files for a participant.

    Convenience wrapper around rrational_export.find_rrational_files()
    for API consistency with other persistence functions.

    Args:
        participant_id: The participant ID to find files for
        data_dir: Optional data directory to search
        project_path: Project path (takes priority if provided)

    Returns:
        List of .rrational file paths, sorted by modification time (newest first)
    """
    from rrational.gui.rrational_export import find_rrational_files
    return find_rrational_files(participant_id, data_dir, project_path)


# --- Artifact Corrections ---

def save_artifact_corrections(
    participant_id: str,
    manual_artifacts: list[dict],
    artifact_exclusions: set[int] | list[int],
    data_dir: str | None = None,
    project_path: Path | None = None,
    algorithm_artifacts: list[int] | None = None,
    algorithm_method: str | None = None,
    algorithm_threshold: float | None = None,
) -> Path:
    """Save artifact corrections (algorithm-detected, manual markings, and exclusions) to YAML.

    Storage priority:
    1. If project_path provided: saves to project/data/processed/{participant_id}_artifacts.yml
    2. If data_dir provided: saves to {data_dir}/../processed/{participant_id}_artifacts.yml
    3. Otherwise: saves to ~/.rrational/{participant_id}_artifacts.yml (fallback)

    Args:
        participant_id: The participant ID
        manual_artifacts: List of manually marked artifacts (dicts with original_idx, timestamp, rr_value, etc.)
        artifact_exclusions: Set of original indices for excluded detected artifacts
        data_dir: Optional data directory
        project_path: Project path (takes priority if provided)
        algorithm_artifacts: List of indices detected by algorithm (optional)
        algorithm_method: Detection method used (e.g., "threshold", "malik")
        algorithm_threshold: Threshold value used for detection

    Returns:
        Path to the saved file
    """
    from datetime import datetime

    # Serialize data
    output_data = {
        "participant_id": participant_id,
        "format_version": "1.1",
        "source_type": "rrational_toolkit",
        "saved_at": datetime.now().isoformat(),
        "manual_artifacts": manual_artifacts,
        "excluded_artifact_indices": list(artifact_exclusions) if artifact_exclusions else [],
    }

    # Add algorithm-detected artifacts if provided
    if algorithm_artifacts is not None:
        output_data["algorithm_artifact_indices"] = list(algorithm_artifacts)
        if algorithm_method:
            output_data["algorithm_method"] = algorithm_method
        if algorithm_threshold is not None:
            output_data["algorithm_threshold"] = algorithm_threshold

    # Use the same processed directory as .rrational files
    processed_dir = get_processed_dir(data_dir=data_dir, project_path=project_path)
    artifact_file = processed_dir / f"{participant_id}_artifacts.yml"

    with open(artifact_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(output_data, f, default_flow_style=False, allow_unicode=True)

    return artifact_file


def load_artifact_corrections(
    participant_id: str,
    data_dir: str | None = None,
    project_path: Path | None = None,
) -> dict[str, Any] | None:
    """Load saved artifact corrections from YAML.

    Uses the same processed directory as .rrational files (via get_processed_dir).
    Also checks ~/.rrational/ as fallback for legacy files.

    Returns:
        Dict with 'manual_artifacts' (list), 'excluded_artifact_indices' (list),
        and optionally 'algorithm_artifact_indices' (list), 'algorithm_method' (str),
        'algorithm_threshold' (float), 'saved_at' (str), or None if no saved corrections exist.
    """
    # Primary location: same as .rrational files
    processed_dir = get_processed_dir(data_dir=data_dir, project_path=project_path)
    artifact_file = processed_dir / f"{participant_id}_artifacts.yml"

    # Also check legacy location as fallback
    search_paths = [artifact_file, CONFIG_DIR / f"{participant_id}_artifacts.yml"]

    for artifact_file in search_paths:
        if artifact_file.exists():
            with open(artifact_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            result = {
                "manual_artifacts": data.get("manual_artifacts", []),
                "excluded_artifact_indices": data.get("excluded_artifact_indices", []),
                "saved_at": data.get("saved_at"),
            }

            # Include algorithm artifacts if present (format version 1.1+)
            if "algorithm_artifact_indices" in data:
                result["algorithm_artifact_indices"] = data.get("algorithm_artifact_indices", [])
                result["algorithm_method"] = data.get("algorithm_method")
                result["algorithm_threshold"] = data.get("algorithm_threshold")

            return result

    return None


def delete_artifact_corrections(
    participant_id: str,
    data_dir: str | None = None,
    project_path: Path | None = None,
) -> bool:
    """Delete saved artifact corrections for a participant.

    Returns True if corrections were deleted, False if none existed.
    """
    deleted_any = False

    # Primary location: same as .rrational files
    processed_dir = get_processed_dir(data_dir=data_dir, project_path=project_path)
    delete_paths = [
        processed_dir / f"{participant_id}_artifacts.yml",
        CONFIG_DIR / f"{participant_id}_artifacts.yml",  # Legacy fallback
    ]

    for artifact_file in delete_paths:
        if artifact_file.exists():
            artifact_file.unlink()
            deleted_any = True

    return deleted_any


def load_artifact_corrections_from_rrational(
    rrational_file: Path,
) -> dict[str, Any] | None:
    """Load artifact markings from a .rrational export file.

    This allows restoring artifact markings from a previously exported file
    when no _artifacts.yml exists (fallback).

    Args:
        rrational_file: Path to the .rrational file

    Returns:
        Dict with 'manual_artifacts' (list) and 'excluded_artifact_indices' (list),
        or None if file doesn't exist or has no artifact data.
    """
    if not rrational_file.exists():
        return None

    try:
        with open(rrational_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        processing = data.get("processing", {})
        manual_artifacts = processing.get("manual_artifacts", [])
        excluded_indices = processing.get("excluded_detected_indices", [])

        # Convert ManualArtifact format to session state format
        converted_manual = []
        for ma in manual_artifacts:
            converted_manual.append({
                "original_idx": ma.get("original_idx", 0),
                "timestamp": ma.get("timestamp", ""),
                "rr_value": ma.get("rr_value", 0),
                "source": ma.get("source", "manual"),
                "plot_idx": ma.get("original_idx", 0),  # Use original_idx as plot_idx
            })

        if converted_manual or excluded_indices:
            return {
                "manual_artifacts": converted_manual,
                "excluded_artifact_indices": excluded_indices,
                "source_file": str(rrational_file),
            }

        return None

    except Exception:
        return None
