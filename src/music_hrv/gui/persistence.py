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
