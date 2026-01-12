"""Music section analysis module.

Handles protocol-based music section extraction, validation, and analysis.
Supports both HRV Logger (real timestamps) and VNS Analyse (synthetic timestamps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence

from rrational.io.hrv_logger import RRInterval


@dataclass(slots=True)
class ProtocolConfig:
    """Protocol configuration for music section analysis.

    Defines the expected structure of a recording session.
    """
    expected_duration_min: float = 90.0  # Total expected duration in minutes
    section_length_min: float = 5.0  # Length of each music section
    pre_pause_sections: int = 9  # Number of sections before pause
    post_pause_sections: int = 9  # Number of sections after pause
    min_section_duration_min: float = 4.0  # Minimum valid section duration
    min_section_beats: int = 100  # Minimum beats for valid section

    @property
    def total_sections(self) -> int:
        return self.pre_pause_sections + self.post_pause_sections

    @property
    def expected_pre_pause_min(self) -> float:
        return self.pre_pause_sections * self.section_length_min

    @property
    def expected_post_pause_min(self) -> float:
        return self.post_pause_sections * self.section_length_min


@dataclass(slots=True)
class MusicSection:
    """A single music section with its data and validation status."""
    index: int  # 0-based index (0-8 pre-pause, 9-17 post-pause)
    music_type: str  # e.g., "music_1", "music_2", "music_3"
    phase: str  # "pre_pause" or "post_pause"

    # Time boundaries
    expected_start: datetime | None = None
    expected_end: datetime | None = None
    actual_start: datetime | None = None
    actual_end: datetime | None = None

    # Data
    rr_intervals: list[RRInterval] = field(default_factory=list)

    # Validation
    expected_duration_s: float = 300.0  # 5 min default
    actual_duration_s: float = 0.0
    beat_count: int = 0
    artifact_count: int = 0
    is_valid: bool = True
    validation_warnings: list[str] = field(default_factory=list)

    @property
    def duration_ratio(self) -> float:
        """Ratio of actual to expected duration."""
        if self.expected_duration_s <= 0:
            return 0.0
        return self.actual_duration_s / self.expected_duration_s

    @property
    def artifact_rate(self) -> float:
        """Artifact rate as percentage."""
        if self.beat_count <= 0:
            return 0.0
        return self.artifact_count / self.beat_count

    @property
    def label(self) -> str:
        """Human-readable label for this section."""
        phase_label = "Pre" if self.phase == "pre_pause" else "Post"
        section_num = self.index + 1 if self.phase == "pre_pause" else self.index - 8
        return f"{phase_label}-{section_num}: {self.music_type}"


@dataclass(slots=True)
class MusicSectionAnalysis:
    """Results of music section extraction and validation."""
    protocol: ProtocolConfig
    sections: list[MusicSection] = field(default_factory=list)

    # Duration analysis
    expected_total_duration_s: float = 0.0
    actual_total_duration_s: float = 0.0
    duration_mismatch_s: float = 0.0

    # Validation summary
    valid_sections: int = 0
    incomplete_sections: int = 0
    warnings: list[str] = field(default_factory=list)

    # Source info
    source_app: str = "Unknown"
    has_real_timestamps: bool = True


class DurationMismatchStrategy:
    """Strategy for handling duration mismatches."""
    STRICT = "strict"  # Exclude incomplete sections
    PROPORTIONAL = "proportional"  # Scale all sections equally
    FLAG_ONLY = "flag_only"  # Include all, mark incomplete


def extract_music_sections(
    rr_intervals: Sequence[RRInterval],
    events: dict[str, datetime],  # canonical_name -> timestamp
    music_order: list[str],  # e.g., ["music_1", "music_2", "music_3"]
    protocol: ProtocolConfig | None = None,
    mismatch_strategy: str = DurationMismatchStrategy.FLAG_ONLY,
) -> MusicSectionAnalysis:
    """Extract music sections from RR intervals based on protocol and events.

    Args:
        rr_intervals: List of RR intervals with timestamps
        events: Dictionary mapping canonical event names to timestamps
        music_order: Order of music types for this participant's playlist
        protocol: Protocol configuration (uses defaults if None)
        mismatch_strategy: How to handle duration mismatches

    Returns:
        MusicSectionAnalysis with extracted sections and validation info
    """
    protocol = protocol or ProtocolConfig()
    analysis = MusicSectionAnalysis(protocol=protocol)

    if not rr_intervals:
        analysis.warnings.append("No RR intervals provided")
        return analysis

    # Get boundary events
    measurement_start = events.get("measurement_start")
    pause_start = events.get("pause_start")
    pause_end = events.get("pause_end")
    measurement_end = events.get("measurement_end")

    if not measurement_start:
        analysis.warnings.append("Missing measurement_start event")
        return analysis

    # Pre-pause sections
    pre_pause_start = measurement_start
    pre_pause_end = pause_start or (measurement_start + timedelta(minutes=protocol.expected_pre_pause_min))

    # Post-pause sections
    post_pause_start = pause_end or pre_pause_end
    post_pause_end = measurement_end or (post_pause_start + timedelta(minutes=protocol.expected_post_pause_min))

    # Calculate actual durations
    actual_pre_pause_duration = (pre_pause_end - pre_pause_start).total_seconds()
    actual_post_pause_duration = (post_pause_end - post_pause_start).total_seconds()

    analysis.expected_total_duration_s = protocol.expected_duration_min * 60
    analysis.actual_total_duration_s = actual_pre_pause_duration + actual_post_pause_duration
    analysis.duration_mismatch_s = analysis.expected_total_duration_s - analysis.actual_total_duration_s

    # Determine section boundaries based on strategy
    if mismatch_strategy == DurationMismatchStrategy.PROPORTIONAL:
        # Scale sections proportionally
        pre_section_duration = actual_pre_pause_duration / protocol.pre_pause_sections
        post_section_duration = actual_post_pause_duration / protocol.post_pause_sections
    else:
        # Use expected duration (strict or flag_only)
        pre_section_duration = protocol.section_length_min * 60
        post_section_duration = protocol.section_length_min * 60

    # Extract pre-pause sections
    for i in range(protocol.pre_pause_sections):
        music_idx = i % len(music_order)
        music_type = music_order[music_idx]

        expected_start = pre_pause_start + timedelta(seconds=i * pre_section_duration)
        expected_end = expected_start + timedelta(seconds=pre_section_duration)

        # Clamp to actual boundaries
        actual_start = max(expected_start, pre_pause_start)
        actual_end = min(expected_end, pre_pause_end)

        section = MusicSection(
            index=i,
            music_type=music_type,
            phase="pre_pause",
            expected_start=expected_start,
            expected_end=expected_end,
            actual_start=actual_start,
            actual_end=actual_end,
            expected_duration_s=pre_section_duration,
        )

        # Extract RR intervals for this section
        section.rr_intervals = [
            rr for rr in rr_intervals
            if rr.timestamp and actual_start <= rr.timestamp <= actual_end
        ]
        section.beat_count = len(section.rr_intervals)
        section.actual_duration_s = sum(rr.rr_ms for rr in section.rr_intervals) / 1000

        # Validate section
        _validate_section(section, protocol, mismatch_strategy)

        analysis.sections.append(section)

    # Extract post-pause sections
    for i in range(protocol.post_pause_sections):
        music_idx = i % len(music_order)
        music_type = music_order[music_idx]

        expected_start = post_pause_start + timedelta(seconds=i * post_section_duration)
        expected_end = expected_start + timedelta(seconds=post_section_duration)

        # Clamp to actual boundaries
        actual_start = max(expected_start, post_pause_start)
        actual_end = min(expected_end, post_pause_end)

        section = MusicSection(
            index=protocol.pre_pause_sections + i,
            music_type=music_type,
            phase="post_pause",
            expected_start=expected_start,
            expected_end=expected_end,
            actual_start=actual_start,
            actual_end=actual_end,
            expected_duration_s=post_section_duration,
        )

        # Extract RR intervals for this section
        section.rr_intervals = [
            rr for rr in rr_intervals
            if rr.timestamp and actual_start <= rr.timestamp <= actual_end
        ]
        section.beat_count = len(section.rr_intervals)
        section.actual_duration_s = sum(rr.rr_ms for rr in section.rr_intervals) / 1000

        # Validate section
        _validate_section(section, protocol, mismatch_strategy)

        analysis.sections.append(section)

    # Summarize validation
    analysis.valid_sections = sum(1 for s in analysis.sections if s.is_valid)
    analysis.incomplete_sections = len(analysis.sections) - analysis.valid_sections

    if analysis.duration_mismatch_s > 120:  # More than 2 min mismatch
        analysis.warnings.append(
            f"Duration mismatch: expected {protocol.expected_duration_min:.1f} min, "
            f"actual {analysis.actual_total_duration_s/60:.1f} min "
            f"({analysis.duration_mismatch_s/60:.1f} min short)"
        )

    return analysis


def _validate_section(
    section: MusicSection,
    protocol: ProtocolConfig,
    strategy: str,
) -> None:
    """Validate a single music section."""
    section.validation_warnings = []
    section.is_valid = True

    # Check duration
    min_duration_s = protocol.min_section_duration_min * 60
    if section.actual_duration_s < min_duration_s:
        section.validation_warnings.append(
            f"Duration {section.actual_duration_s/60:.1f} min < {protocol.min_section_duration_min} min minimum"
        )
        if strategy == DurationMismatchStrategy.STRICT:
            section.is_valid = False

    # Check beat count
    if section.beat_count < protocol.min_section_beats:
        section.validation_warnings.append(
            f"Only {section.beat_count} beats < {protocol.min_section_beats} minimum"
        )
        if strategy == DurationMismatchStrategy.STRICT:
            section.is_valid = False

    # Check if section is completely empty
    if section.beat_count == 0:
        section.validation_warnings.append("No RR intervals in section")
        section.is_valid = False


def get_sections_by_music_type(
    analysis: MusicSectionAnalysis,
    valid_only: bool = True,
) -> dict[str, list[MusicSection]]:
    """Group sections by music type for comparison analysis.

    Args:
        analysis: The music section analysis results
        valid_only: If True, only include valid sections

    Returns:
        Dictionary mapping music_type -> list of sections
    """
    result: dict[str, list[MusicSection]] = {}

    for section in analysis.sections:
        if valid_only and not section.is_valid:
            continue

        if section.music_type not in result:
            result[section.music_type] = []
        result[section.music_type].append(section)

    return result


def get_sections_by_phase(
    analysis: MusicSectionAnalysis,
    valid_only: bool = True,
) -> dict[str, list[MusicSection]]:
    """Group sections by phase (pre/post pause) for comparison.

    Args:
        analysis: The music section analysis results
        valid_only: If True, only include valid sections

    Returns:
        Dictionary mapping phase -> list of sections
    """
    result = {"pre_pause": [], "post_pause": []}

    for section in analysis.sections:
        if valid_only and not section.is_valid:
            continue
        result[section.phase].append(section)

    return result
