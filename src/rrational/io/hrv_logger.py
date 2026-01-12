"""HRV Logger CSV ingestion utilities."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Pattern for participant ID: 4 digits followed by 4 uppercase letters (e.g., 0123ABCD)
DEFAULT_ID_PATTERN = r"(?P<participant>\d{4}[A-Z]{4})"

# Predefined patterns for common ID formats
PREDEFINED_PATTERNS = {
    "4 digits + 4 uppercase (e.g., 0123ABCD)": r"(?P<participant>\d{4}[A-Z]{4})",
    "4 digits + 4 letters (case insensitive)": r"(?P<participant>\d{4}[A-Za-z]{4})",
    "Any alphanumeric ID": r"(?P<participant>[A-Za-z0-9]+)",
    "Digits only (e.g., 001, 0123)": r"(?P<participant>\d+)",
    "Letters + digits (e.g., P001, SUB123)": r"(?P<participant>[A-Za-z]+\d+)",
    "Underscore separated (e.g., sub_001)": r"(?P<participant>[A-Za-z]+_\d+)",
}
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
    """Paired RR + Events files for one participant (supports multiple files from restarts)."""

    participant_id: str
    rr_paths: list[Path]  # All RR files for this participant (sorted by name/timestamp)
    events_paths: list[Path]  # All Events files for this participant

    @property
    def rr_path(self) -> Path:
        """First RR file (for backward compatibility)."""
        return self.rr_paths[0] if self.rr_paths else None

    @property
    def events_path(self) -> Path | None:
        """First Events file (for backward compatibility)."""
        return self.events_paths[0] if self.events_paths else None

    @property
    def has_multiple_files(self) -> bool:
        """Check if this participant has multiple RR or Events files."""
        return len(self.rr_paths) > 1 or len(self.events_paths) > 1


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
    """Discover RR/Events pairs under the provided root folder.

    Supports multiple files per participant (e.g., from measurement restarts).
    Files are sorted by name which typically reflects timestamp order.
    """

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
    for participant, rr_paths in sorted(rr_index.items()):
        events_paths = events_index.get(participant) or []
        bundles.append(
            RecordingBundle(
                participant_id=participant,
                rr_paths=rr_paths,  # All RR files for this participant
                events_paths=events_paths,  # All Events files
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

    Supports multiple RR/Events files per participant (merges all files).
    Data is sorted by timestamp after merging.

    Returns:
        tuple: (HRVLoggerRecording, duplicate_count, duplicate_details)
    """

    all_rr_intervals: list[RRInterval] = []
    all_duplicates = 0
    all_duplicate_details: list[DuplicateInfo] = []

    # Load all RR files and merge
    for rr_path in bundle.rr_paths:
        rr_intervals, duplicates, duplicate_details = load_rr_intervals(rr_path)
        all_rr_intervals.extend(rr_intervals)
        all_duplicates += duplicates
        all_duplicate_details.extend(duplicate_details)

    # Sort RR intervals by timestamp (handles merged files from restarts)
    all_rr_intervals.sort(key=lambda x: x.timestamp if x.timestamp else datetime.min.replace(tzinfo=None))

    # Load all Events files and merge
    all_events: list[EventMarker] = []
    for events_path in bundle.events_paths:
        if events_path.exists():
            events = load_events(events_path)
            all_events.extend(events)

    # Sort events by timestamp
    all_events.sort(key=lambda x: x.timestamp if x.timestamp else datetime.min.replace(tzinfo=None))

    recording = HRVLoggerRecording(
        participant_id=bundle.participant_id,
        rr_intervals=all_rr_intervals,
        events=all_events,
    )
    return recording, all_duplicates, all_duplicate_details


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
