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
    """Single VNS file for one participant."""

    participant_id: str
    file_path: Path


@dataclass(slots=True)
class VNSRecording:
    """Full VNS Analyse recording (RR beats + optional notes as events)."""

    participant_id: str
    rr_intervals: list[RRInterval]
    events: list[EventMarker]
    header_info: dict[str, str]  # Key-value pairs from header section


def discover_vns_recordings(
    root: Path, *, pattern: str = DEFAULT_ID_PATTERN
) -> list[VNSRecordingBundle]:
    """Discover VNS Analyse files under the provided root folder.

    VNS files are typically .txt files with "VNS" in the name.
    """
    root = root.expanduser().resolve()
    bundles: list[VNSRecordingBundle] = []
    seen_participants: dict[str, VNSRecordingBundle] = {}

    # Look for .txt files (VNS Analyse exports)
    for file_path in sorted(root.rglob("*.txt")):
        name = file_path.name.lower()
        # Skip files that look like HRV Logger files
        if "rr_" in name or "_rr" in name or "events_" in name or "_events" in name:
            continue

        participant = extract_participant_id(file_path.name, pattern)
        if participant not in seen_participants:
            bundle = VNSRecordingBundle(
                participant_id=participant,
                file_path=file_path,
            )
            bundles.append(bundle)
            seen_participants[participant] = bundle

    return bundles


def load_vns_recording(
    bundle: VNSRecordingBundle,
    *,
    use_corrected: bool = False,
) -> VNSRecording:
    """Load a VNS Analyse file and return parsed data.

    VNS format:
    - Header sections with tab-separated key-value pairs
    - Two RR sections: Raw (Rohwerte) and Corrected (Korrigierte Werte)
    - RR interval lines: "0.719<tab>" or "0.863<tab>Notiz: Ende Ruhe"
    - RR values are in SECONDS (converted to ms internally)

    Args:
        bundle: VNS recording bundle with file path
        use_corrected: If True, use corrected RR values. Default False (raw values).
    """
    path = bundle.file_path
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

    return VNSRecording(
        participant_id=bundle.participant_id,
        rr_intervals=rr_intervals,
        events=events,
        header_info=header_info,
    )


def load_vns_recordings_from_directory(
    root: Path, *, pattern: str = DEFAULT_ID_PATTERN
) -> list[VNSRecording]:
    """Load all VNS recordings from a directory."""
    bundles = discover_vns_recordings(root, pattern=pattern)
    return [load_vns_recording(bundle) for bundle in bundles]
