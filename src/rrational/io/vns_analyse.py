"""VNS Analyse TXT ingestion utilities.

VNS Analyse exports a single .txt file per participant containing:
- Header sections with parameter names and values (tab-separated)
- Two RR intervals sections:
  - "RR-Intervalle - Rohwerte (Nicht aktiv)" = Raw/Uncorrected values
  - "RR-Intervalle - Korrigierte Werte (Aktiv)" = Corrected values
- RR values are in SECONDS (not milliseconds!)
- Each line: RR_value<tab> or RR_value<tab>Notiz: <note text>

Filename format: "dd.mm.yyyy hh.mm <word> xh xxmin.txt"
- dd.mm.yyyy = recording date
- hh.mm = recording start time
- xh xxmin = recording duration (informational)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from rrational.io.hrv_logger import (
    DEFAULT_ID_PATTERN,
    EventMarker,
    RRInterval,
    extract_participant_id,
)

# VNS section identifiers
VNS_RAW_SECTION = "RR-Intervalle - Rohwerte"
VNS_CORRECTED_SECTION = "RR-Intervalle - Korrigierte Werte"

# Regex to parse VNS filename: "dd.mm.yyyy hh.mm ..."
VNS_FILENAME_PATTERN = re.compile(r"(\d{2})\.(\d{2})\.(\d{4})\s+(\d{1,2})\.(\d{2})")

# Pattern to extract participant ID from VNS filename after date/time
# Format: "dd.mm.yyyy hh.mm <participant_id> xh xxmin.txt"
# The participant ID is typically between the time and duration (e.g., "VP01", "P001", "SUB123")
VNS_PARTICIPANT_PATTERN = re.compile(
    r"\d{2}\.\d{2}\.\d{4}\s+\d{1,2}\.\d{2}\s+(?P<participant>[A-Za-z0-9_-]+?)(?:\s+\d+h|\s+\d+min|\.txt)",
    re.IGNORECASE
)


def extract_vns_participant_id(filename: str, fallback_pattern: str = DEFAULT_ID_PATTERN) -> str:
    """Extract participant ID from VNS filename.

    VNS filename format: "dd.mm.yyyy hh.mm <participant_id> xh xxmin.txt"
    Example: "05.12.2024 14.30 VP01 2h 15min.txt" -> "VP01"

    Falls back to generic extraction if VNS-specific pattern fails.
    """
    # Try VNS-specific pattern first
    match = VNS_PARTICIPANT_PATTERN.search(filename)
    if match:
        participant = match.group("participant")
        if participant:
            return participant

    # Fallback to generic extraction
    return extract_participant_id(filename, fallback_pattern)


def parse_vns_filename_datetime(filename: str) -> datetime | None:
    """Parse recording date and time from VNS filename.

    Filename format: "dd.mm.yyyy hh.mm <word> xh xxmin.txt"
    Example: "05.12.2024 14.30 VP01 2h 15min.txt" -> datetime(2024, 12, 5, 14, 30)

    Returns None if parsing fails.
    """
    match = VNS_FILENAME_PATTERN.search(filename)
    if match:
        day, month, year, hour, minute = match.groups()
        try:
            return datetime(
                year=int(year),
                month=int(month),
                day=int(day),
                hour=int(hour),
                minute=int(minute),
            )
        except ValueError:
            pass
    return None


@dataclass(slots=True)
class VNSRecordingBundle:
    """VNS files for one participant (supports multiple files from restarts)."""

    participant_id: str
    file_paths: list[Path]  # All VNS files for this participant (sorted by timestamp)

    @property
    def file_path(self) -> Path:
        """First file (for backward compatibility)."""
        return self.file_paths[0] if self.file_paths else None

    @property
    def has_multiple_files(self) -> bool:
        """Check if this participant has multiple VNS files."""
        return len(self.file_paths) > 1


@dataclass(slots=True)
class VNSFileSegment:
    """Information about a single VNS file segment."""

    file_path: Path
    start_time: datetime | None  # Parsed from filename
    end_time: datetime | None  # Calculated from start + duration
    duration_ms: int  # Total duration in milliseconds
    beat_count: int


@dataclass(slots=True)
class VNSRecordingGap:
    """Gap between two VNS recording segments."""

    after_file: Path  # File before the gap
    before_file: Path  # File after the gap
    gap_start: datetime
    gap_end: datetime
    gap_duration_s: float  # Duration in seconds


@dataclass(slots=True)
class VNSRecordingOverlap:
    """Overlap between two VNS recording segments."""

    file1: Path
    file2: Path
    overlap_start: datetime
    overlap_end: datetime
    overlap_duration_s: float  # Duration in seconds


@dataclass(slots=True)
class VNSRecording:
    """Full VNS Analyse recording (RR beats + optional notes as events)."""

    participant_id: str
    rr_intervals: list[RRInterval]
    events: list[EventMarker]
    header_info: dict[str, str]  # Key-value pairs from header section
    # Multi-file metadata
    file_segments: list[VNSFileSegment] | None = None
    gaps: list[VNSRecordingGap] | None = None
    overlaps: list[VNSRecordingOverlap] | None = None


def discover_vns_recordings(
    root: Path, *, pattern: str = DEFAULT_ID_PATTERN
) -> list[VNSRecordingBundle]:
    """Discover VNS Analyse files under the provided root folder.

    Supports multiple files per participant (e.g., from measurement restarts).
    Files are sorted by parsed timestamp from filename.
    """
    root = root.expanduser().resolve()
    file_index: dict[str, list[Path]] = {}

    # Look for .txt files (VNS Analyse exports)
    for file_path in sorted(root.rglob("*.txt")):
        name = file_path.name.lower()
        # Skip files that look like HRV Logger files
        if "rr_" in name or "_rr" in name or "events_" in name or "_events" in name:
            continue

        # Use VNS-specific extraction which handles VNS filename format better
        participant = extract_vns_participant_id(file_path.name, pattern)
        file_index.setdefault(participant, []).append(file_path)

    # Build bundles with all files per participant, sorted by timestamp
    bundles: list[VNSRecordingBundle] = []
    for participant, file_paths in sorted(file_index.items()):
        # Sort files by parsed datetime from filename (earliest first)
        sorted_paths = sorted(
            file_paths,
            key=lambda p: parse_vns_filename_datetime(p.name) or datetime.min,
        )
        bundles.append(
            VNSRecordingBundle(
                participant_id=participant,
                file_paths=sorted_paths,
            )
        )

    return bundles


def _load_single_vns_file(
    path: Path,
    *,
    use_corrected: bool = False,
) -> tuple[list[RRInterval], list[EventMarker], dict[str, str]]:
    """Load a single VNS Analyse file.

    Returns:
        tuple: (rr_intervals, events, header_info)
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    header_info: dict[str, str] = {}
    rr_intervals: list[RRInterval] = []
    events: list[EventMarker] = []

    # Track cumulative time for timestamps
    # Parse actual recording date/time from filename (format: "dd.mm.yyyy hh.mm ...")
    # Fall back to fixed date if parsing fails
    cumulative_ms = 0
    base_time = parse_vns_filename_datetime(path.name)
    if base_time is None:
        # Fallback to fixed date if filename doesn't match expected format
        base_time = datetime(2000, 1, 1, 0, 0, 0, 0)

    # Track which section we're in
    current_section: str | None = None
    target_section = VNS_CORRECTED_SECTION if use_corrected else VNS_RAW_SECTION

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for section headers
        if VNS_RAW_SECTION in line:
            current_section = VNS_RAW_SECTION
            continue
        elif VNS_CORRECTED_SECTION in line:
            current_section = VNS_CORRECTED_SECTION
            continue
        elif line.startswith("RR-Intervalle"):
            # Some other RR section we don't recognize
            current_section = None
            continue

        # Check for other section headers (end RR section)
        if current_section and not line[0].isdigit():
            # New section starting, stop reading RR
            current_section = None
            continue

        # Only parse RR values from the target section
        if current_section != target_section:
            # Try to parse as header line
            if "\t" in line:
                key_val = line.split("\t", 1)
                if len(key_val) == 2:
                    header_info[key_val[0].strip()] = key_val[1].strip()
            continue

        # Try to parse as RR interval line
        # Format: "0.719\t" or "0.863\tNotiz: Ende Ruhe"
        parts = line.split("\t")
        if parts:
            try:
                # Try to parse first part as float (RR in seconds)
                rr_seconds = float(parts[0].strip())
                # Convert to milliseconds
                rr_ms = int(rr_seconds * 1000)

                # Load ALL intervals - filtering happens at display/analysis time
                # Calculate timestamp based on cumulative RR values
                timestamp = base_time + timedelta(milliseconds=cumulative_ms)

                rr_intervals.append(RRInterval(
                    timestamp=timestamp,
                    rr_ms=rr_ms,
                    elapsed_ms=cumulative_ms,
                ))

                # Check for note (Notiz:)
                if len(parts) > 1:
                    note_text = parts[1].strip()
                    if note_text.startswith("Notiz:"):
                        note_label = note_text[6:].strip()  # Remove "Notiz:" prefix
                        if note_label:
                            events.append(EventMarker(
                                label=note_label,
                                timestamp=timestamp,
                                offset_s=cumulative_ms / 1000.0,
                            ))

                cumulative_ms += rr_ms

            except ValueError:
                pass  # Not a valid RR line

    return rr_intervals, events, header_info


def load_vns_recording(
    bundle: VNSRecordingBundle,
    *,
    use_corrected: bool = False,
) -> VNSRecording:
    """Load VNS Analyse file(s) and return parsed data.

    Supports multiple files per participant (merges all files).
    Data is sorted by timestamp after merging.
    Detects gaps and overlaps between files.

    VNS format:
    - Header sections with tab-separated key-value pairs
    - Two RR sections: Raw (Rohwerte) and Corrected (Korrigierte Werte)
    - RR interval lines: "0.719<tab>" or "0.863<tab>Notiz: Ende Ruhe"
    - RR values are in SECONDS (converted to ms internally)

    Args:
        bundle: VNS recording bundle with file path(s)
        use_corrected: If True, use corrected RR values. Default False (raw values).
    """
    all_rr_intervals: list[RRInterval] = []
    all_events: list[EventMarker] = []
    merged_header: dict[str, str] = {}
    file_segments: list[VNSFileSegment] = []

    # Load all files and collect segment info
    for path in bundle.file_paths:
        rr_intervals, events, header_info = _load_single_vns_file(
            path, use_corrected=use_corrected
        )

        # Calculate segment info
        start_time = parse_vns_filename_datetime(path.name)
        duration_ms = sum(rr.rr_ms for rr in rr_intervals)
        end_time = None
        if start_time and duration_ms > 0:
            end_time = start_time + timedelta(milliseconds=duration_ms)

        file_segments.append(VNSFileSegment(
            file_path=path,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            beat_count=len(rr_intervals),
        ))

        all_rr_intervals.extend(rr_intervals)
        all_events.extend(events)
        # Merge header info (later files overwrite earlier)
        merged_header.update(header_info)

    # Detect gaps and overlaps between segments
    gaps: list[VNSRecordingGap] = []
    overlaps: list[VNSRecordingOverlap] = []

    if len(file_segments) > 1:
        # Sort segments by start time
        sorted_segments = sorted(
            file_segments,
            key=lambda s: s.start_time if s.start_time else datetime.min
        )

        for i in range(len(sorted_segments) - 1):
            seg1 = sorted_segments[i]
            seg2 = sorted_segments[i + 1]

            if seg1.end_time and seg2.start_time:
                if seg1.end_time < seg2.start_time:
                    # Gap detected
                    gap_duration = (seg2.start_time - seg1.end_time).total_seconds()
                    gaps.append(VNSRecordingGap(
                        after_file=seg1.file_path,
                        before_file=seg2.file_path,
                        gap_start=seg1.end_time,
                        gap_end=seg2.start_time,
                        gap_duration_s=gap_duration,
                    ))
                elif seg1.end_time > seg2.start_time:
                    # Overlap detected
                    overlap_end = min(seg1.end_time, seg2.end_time) if seg2.end_time else seg1.end_time
                    overlap_duration = (overlap_end - seg2.start_time).total_seconds()
                    overlaps.append(VNSRecordingOverlap(
                        file1=seg1.file_path,
                        file2=seg2.file_path,
                        overlap_start=seg2.start_time,
                        overlap_end=overlap_end,
                        overlap_duration_s=overlap_duration,
                    ))

    # Sort RR intervals by timestamp (handles merged files from restarts)
    all_rr_intervals.sort(
        key=lambda x: x.timestamp if x.timestamp else datetime.min.replace(tzinfo=None)
    )

    # Sort events by timestamp
    all_events.sort(
        key=lambda x: x.timestamp if x.timestamp else datetime.min.replace(tzinfo=None)
    )

    return VNSRecording(
        participant_id=bundle.participant_id,
        rr_intervals=all_rr_intervals,
        events=all_events,
        header_info=merged_header,
        file_segments=file_segments if len(file_segments) > 1 else None,
        gaps=gaps if gaps else None,
        overlaps=overlaps if overlaps else None,
    )


def load_vns_recordings_from_directory(
    root: Path, *, pattern: str = DEFAULT_ID_PATTERN
) -> list[VNSRecording]:
    """Load all VNS recordings from a directory."""
    bundles = discover_vns_recordings(root, pattern=pattern)
    return [load_vns_recording(bundle) for bundle in bundles]
