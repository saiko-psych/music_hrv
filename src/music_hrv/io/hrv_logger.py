"""HRV Logger CSV ingestion utilities."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_ID_PATTERN = r"(?P<participant>[A-Za-z0-9]+)"
_RR_REQUIRED_COLUMNS = ("date", "rr")
_EVENT_REQUIRED_COLUMNS = ("annotation", "timestamp")


def _prepare_reader(path: Path) -> csv.DictReader:
    """Return a CSV reader with stripped headers/body."""

    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return csv.DictReader(io.StringIO(text))


def _normalise_row(row: dict[str, str | None]) -> dict[str, str]:
    """Lowercase + strip CSV keys/values for tolerant parsing."""

    return {
        (key.strip().lower() if key else ""): (value or "").strip()
        for key, value in row.items()
    }


@dataclass(slots=True)
class RRInterval:
    """Single beat from the HRV Logger RR CSV."""

    timestamp: datetime | None
    rr_ms: int
    elapsed_ms: int | None


@dataclass(slots=True)
class DuplicateInfo:
    """Information about a detected duplicate RR interval."""

    original_line: int  # Line number where first occurrence was seen
    duplicate_line: int  # Line number where duplicate was found
    date_str: str
    rr_str: str
    elapsed_str: str


@dataclass(slots=True)
class EventMarker:
    """Marker/annotation extracted from the Events CSV."""

    label: str
    timestamp: datetime | None
    offset_s: float | None


@dataclass(slots=True)
class RecordingBundle:
    """Paired RR + Events file for one participant."""

    participant_id: str
    rr_path: Path
    events_path: Path | None = None


@dataclass(slots=True)
class HRVLoggerRecording:
    """Full HRV Logger recording (RR beats + optional events)."""

    participant_id: str
    rr_intervals: list[RRInterval]
    events: list[EventMarker]


def extract_participant_id(name: str, pattern: str = DEFAULT_ID_PATTERN) -> str:
    """Return the participant identifier derived from file names."""

    stem = Path(name).stem
    regex = re.compile(pattern)
    matches = list(regex.finditer(stem))
    for match in reversed(matches):
        participant = match.groupdict().get("participant")
        if participant:
            return participant
    tokens = re.findall(r"[A-Za-z0-9]+", stem)
    if tokens:
        return tokens[-1]
    return "unknown"


def discover_recordings(
    root: Path, *, pattern: str = DEFAULT_ID_PATTERN
) -> list[RecordingBundle]:
    """Discover RR/Events pairs under the provided root folder."""

    root = root.expanduser().resolve()
    rr_index: dict[str, list[Path]] = {}
    events_index: dict[str, list[Path]] = {}

    for rr_path in sorted(root.rglob("*RR*.csv")):
        participant = extract_participant_id(rr_path.name, pattern)
        rr_index.setdefault(participant, []).append(rr_path)

    for events_path in sorted(root.rglob("*Events*.csv")):
        participant = extract_participant_id(events_path.name, pattern)
        events_index.setdefault(participant, []).append(events_path)

    bundles: list[RecordingBundle] = []
    for participant, paths in sorted(rr_index.items()):
        rr_path = paths[0]
        events_candidates = events_index.get(participant) or []
        bundles.append(
            RecordingBundle(
                participant_id=participant,
                rr_path=rr_path,
                events_path=events_candidates[0] if events_candidates else None,
            )
        )
    return bundles


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None


def _parse_int(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def load_rr_intervals(rr_path: Path) -> tuple[list[RRInterval], int, list[DuplicateInfo]]:
    """Parse RR CSV rows and detect exact duplicate rows.

    A duplicate is defined as:
    - Entire row is identical (date + rr + since_start values all match)

    Returns:
        tuple: (list of unique RR intervals, count of duplicates removed, list of duplicate details)
    """

    intervals: list[RRInterval] = []
    seen_rows: dict[tuple, int] = {}  # (timestamp_str, rr_str, elapsed_str) -> line number
    duplicates_list: list[DuplicateInfo] = []
    line_num = 0  # Track CSV line number (including header)

    for row in _prepare_reader(rr_path):
        line_num += 1

        cleaned = _normalise_row(row)
        if not all(col in cleaned for col in _RR_REQUIRED_COLUMNS):
            continue

        # Parse values
        date_str = cleaned.get("date", "")
        rr_str = cleaned.get("rr", "")
        elapsed_str = cleaned.get("since start") or cleaned.get("since_start", "")

        # Create fingerprint of entire row
        row_fingerprint = (date_str, rr_str, elapsed_str)

        # Check if this EXACT row was seen before
        if row_fingerprint in seen_rows:
            original_line = seen_rows[row_fingerprint]
            duplicates_list.append(
                DuplicateInfo(
                    original_line=original_line,
                    duplicate_line=line_num + 1,  # +1 because line 1 is header
                    date_str=date_str,
                    rr_str=rr_str,
                    elapsed_str=elapsed_str,
                )
            )
            continue  # Skip duplicate

        seen_rows[row_fingerprint] = line_num + 1  # +1 because line 1 is header

        # Now parse the actual values
        timestamp = _parse_datetime(date_str)
        rr_ms = _parse_int(rr_str)
        elapsed_ms = _parse_int(elapsed_str)

        if rr_ms is None:
            continue

        intervals.append(
            RRInterval(
                timestamp=timestamp,
                rr_ms=rr_ms,
                elapsed_ms=elapsed_ms,
            )
        )

    return intervals, len(duplicates_list), duplicates_list


def load_events(events_path: Path) -> list[EventMarker]:
    """Parse HRV Logger Events CSV rows."""

    markers: list[EventMarker] = []
    for row in _prepare_reader(events_path):
        cleaned = _normalise_row(row)
        if not all(col in cleaned for col in _EVENT_REQUIRED_COLUMNS):
            continue
        label = cleaned.get("annotation") or ""
        if not label:
            continue
        timestamp = _parse_datetime(cleaned.get("date"))
        offset_s: float | None = None
        ts_value = cleaned.get("timestamp")
        if ts_value:
            try:
                offset_s = float(ts_value)
            except ValueError:
                offset_s = None
        markers.append(
            EventMarker(
                label=label,
                timestamp=timestamp,
                offset_s=offset_s,
            )
        )
    return markers


def load_recording(bundle: RecordingBundle) -> tuple[HRVLoggerRecording, int, list[DuplicateInfo]]:
    """Load RR + events content for a discovered bundle.

    Returns:
        tuple: (HRVLoggerRecording, duplicate_count, duplicate_details)
    """

    rr_intervals, duplicates, duplicate_details = load_rr_intervals(bundle.rr_path)
    events: list[EventMarker] = []
    if bundle.events_path and bundle.events_path.exists():
        events = load_events(bundle.events_path)
    recording = HRVLoggerRecording(
        participant_id=bundle.participant_id,
        rr_intervals=rr_intervals,
        events=events,
    )
    return recording, duplicates, duplicate_details


def load_recordings_from_directory(
    root: Path, *, pattern: str = DEFAULT_ID_PATTERN
) -> list[tuple[HRVLoggerRecording, int]]:
    """Convenience helper returning fully parsed recordings.

    Returns:
        list of tuples: [(HRVLoggerRecording, duplicate_count), ...]
    """

    bundles = discover_recordings(root, pattern=pattern)
    return [load_recording(bundle) for bundle in bundles]


__all__ = [
    "DEFAULT_ID_PATTERN",
    "RRInterval",
    "EventMarker",
    "RecordingBundle",
    "HRVLoggerRecording",
    "discover_recordings",
    "extract_participant_id",
    "load_events",
    "load_recording",
    "load_recordings_from_directory",
    "load_rr_intervals",
]
