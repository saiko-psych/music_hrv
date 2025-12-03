"""VNS Analyse TXT ingestion utilities.

VNS Analyse exports a single .txt file per participant containing:
- Header sections with parameter names and values (tab-separated)
- RR intervals section with:
  - RR values in SECONDS (not milliseconds!)
  - Each line: RR_value<tab> or RR_value<tab>Notiz: <note text>
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from music_hrv.io.hrv_logger import (
    DEFAULT_ID_PATTERN,
    EventMarker,
    RRInterval,
    extract_participant_id,
)


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


def load_vns_recording(bundle: VNSRecordingBundle) -> VNSRecording:
    """Load a VNS Analyse file and return parsed data.

    VNS format:
    - Header sections with tab-separated key-value pairs
    - RR interval lines: "0.719<tab>" or "0.863<tab>Notiz: Ende Ruhe"
    - RR values are in SECONDS (converted to ms internally)
    """
    path = bundle.file_path
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    header_info: dict[str, str] = {}
    rr_intervals: list[RRInterval] = []
    events: list[EventMarker] = []

    # Track cumulative time for timestamps
    cumulative_ms = 0
    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    in_rr_section = False
    beat_index = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for section headers
        if line.startswith("RR-Intervalle"):
            in_rr_section = True
            continue

        # Check for other section headers (end RR section)
        if in_rr_section and not line[0].isdigit():
            # New section starting, stop reading RR
            in_rr_section = False
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

                # Validate RR range (200ms to 3000ms)
                if 200 <= rr_ms <= 3000:
                    # Calculate timestamp
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
                    beat_index += 1

            except ValueError:
                # Not a valid RR line, might be header
                if "\t" in line and not in_rr_section:
                    # Could be a header line
                    key_val = line.split("\t", 1)
                    if len(key_val) == 2:
                        header_info[key_val[0].strip()] = key_val[1].strip()

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
