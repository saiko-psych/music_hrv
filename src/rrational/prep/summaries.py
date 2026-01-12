"""Summaries for data preparation previews surfaced in the GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from rrational.cleaning.rr import CleaningConfig, clean_rr_intervals, rr_summary
from rrational.io.hrv_logger import HRVLoggerRecording, discover_recordings, load_recording
from rrational.io.vns_analyse import discover_vns_recordings, load_vns_recording
from rrational.segments.section_normalizer import SectionNormalizer


@dataclass(slots=True)
class EventStatus:
    """Mapping result for a raw event label."""

    raw_label: str
    canonical: str | None
    count: int = 1
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None


@dataclass(slots=True)
class PreparationSummary:
    """Aggregate metrics for a cleaned recording."""

    participant_id: str
    recording_datetime: datetime | None  # Timestamp of first event (preferably rest_pre_start)
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    total_beats: int
    retained_beats: int
    removed_beats: int
    artifact_ratio: float
    duration_s: float
    events_detected: int
    duplicate_events: int
    duplicate_rr_intervals: int  # Count of duplicate RR intervals removed
    duplicate_details: list  # List of DuplicateInfo objects
    rr_min_ms: float
    rr_max_ms: float
    rr_mean_ms: float
    artifact_reasons: dict[str, int]
    events: list[EventStatus]
    present_sections: set[str]
    # Multiple file support
    rr_file_count: int = 1  # Number of RR files for this participant
    events_file_count: int = 0  # Number of Events files for this participant
    source_app: str = "Unknown"  # Recording app (HRV Logger, VNS Analyse, etc.)
    # File paths for reloading data in participant view
    rr_paths: list[Path] | None = None  # HRV Logger RR file paths
    events_paths: list[Path] | None = None  # HRV Logger Events file paths
    vns_path: Path | None = None  # VNS Analyse file path

    @property
    def has_multiple_files(self) -> bool:
        """Check if this participant has multiple recording files."""
        return self.rr_file_count > 1 or self.events_file_count > 1

    def as_row(self) -> tuple[str, ...]:
        """Return human readable values for tables."""

        return (
            self.participant_id,
            str(self.total_beats),
            str(self.retained_beats),
            f"{self.artifact_ratio * 100:.1f}%",
            f"{self.duration_s / 60:.1f} min",
            str(self.events_detected),
            f"{int(self.rr_min_ms)}–{int(self.rr_max_ms)} ms",
            f"{self.rr_mean_ms:.0f} ms",
        )


def summarize_recording(
    recording: HRVLoggerRecording,
    *,
    duplicate_rr_count: int = 0,
    duplicate_details: list = None,
    config: CleaningConfig | None = None,
    normalizer: SectionNormalizer | None = None,
    rr_file_count: int = 1,
    events_file_count: int = 0,
    source_app: str = "Unknown",
    rr_paths: list[Path] | None = None,
    events_paths: list[Path] | None = None,
    vns_path: Path | None = None,
) -> PreparationSummary:
    """Clean one participant recording and collect descriptive stats."""

    if duplicate_details is None:
        duplicate_details = []

    cleaned, stats = clean_rr_intervals(recording.rr_intervals, config)
    rr_stats = rr_summary(cleaned or recording.rr_intervals)
    normalizer = normalizer or SectionNormalizer.from_yaml()
    by_label: dict[str, EventStatus] = {}
    present_sections: set[str] = set()
    for marker in recording.events:
        key = marker.label.strip().lower()
        canonical = normalizer.normalize(marker.label)
        if key in by_label:
            by_label[key].count += 1
            by_label[key].last_timestamp = max(
                (ts for ts in (by_label[key].last_timestamp, marker.timestamp) if ts), default=None
            )
        else:
            by_label[key] = EventStatus(
                raw_label=marker.label,
                canonical=canonical,
                first_timestamp=marker.timestamp,
                last_timestamp=marker.timestamp,
            )
        if canonical:
            present_sections.add(canonical)
    event_statuses = list(by_label.values())
    duplicate_events = sum(max(event.count - 1, 0) for event in event_statuses)
    timestamps = [rr.timestamp for rr in recording.rr_intervals if rr.timestamp]
    first_ts = min(timestamps) if timestamps else None
    last_ts = max(timestamps) if timestamps else None

    # Find recording datetime from first event, preferably rest_pre_start
    recording_dt = None
    # Try to find rest_pre_start first
    for event in event_statuses:
        if event.canonical == "rest_pre_start" and event.first_timestamp:
            recording_dt = event.first_timestamp
            break
    # Fallback to first event with timestamp
    if not recording_dt:
        for event in event_statuses:
            if event.first_timestamp:
                recording_dt = event.first_timestamp
                break

    return PreparationSummary(
        participant_id=recording.participant_id,
        recording_datetime=recording_dt,
        first_timestamp=first_ts,
        last_timestamp=last_ts,
        total_beats=stats.total_samples,
        retained_beats=stats.retained_samples,
        removed_beats=stats.removed_samples,
        artifact_ratio=stats.artifact_ratio,
        duration_s=rr_stats["duration_s"],
        events_detected=len(event_statuses),
        duplicate_events=duplicate_events,
        duplicate_rr_intervals=duplicate_rr_count,
        duplicate_details=duplicate_details,
        rr_min_ms=rr_stats["min"],
        rr_max_ms=rr_stats["max"],
        rr_mean_ms=rr_stats["mean"],
        artifact_reasons=stats.reasons,
        events=event_statuses,
        present_sections=present_sections,
        rr_file_count=rr_file_count,
        events_file_count=events_file_count,
        source_app=source_app,
        rr_paths=rr_paths,
        events_paths=events_paths,
        vns_path=vns_path,
    )


def load_hrv_logger_preview(
    root: Path,
    *,
    pattern: str,
    config: CleaningConfig | None = None,
    normalizer: SectionNormalizer | None = None,
) -> list[PreparationSummary]:
    """Return summaries for each HRV Logger participant under root."""

    bundles = discover_recordings(root, pattern=pattern)
    normalizer = normalizer or SectionNormalizer.from_yaml()
    summaries: list[PreparationSummary] = []
    for bundle in bundles:
        recording, duplicate_count, duplicate_details = load_recording(bundle)
        summaries.append(
            summarize_recording(
                recording,
                duplicate_rr_count=duplicate_count,
                duplicate_details=duplicate_details,
                config=config,
                normalizer=normalizer,
                rr_file_count=len(bundle.rr_paths),
                events_file_count=len(bundle.events_paths),
                source_app="HRV Logger",
                rr_paths=list(bundle.rr_paths),
                events_paths=list(bundle.events_paths),
            )
        )
    return summaries


def load_vns_preview(
    root: Path,
    *,
    pattern: str,
    config: CleaningConfig | None = None,
    normalizer: SectionNormalizer | None = None,
    use_corrected: bool = False,
) -> list[PreparationSummary]:
    """Return summaries for each VNS Analyse participant under root.

    Args:
        use_corrected: If True, use corrected RR values from VNS files.
                       Default False = raw values.
    """

    bundles = discover_vns_recordings(root, pattern=pattern)
    normalizer = normalizer or SectionNormalizer.from_yaml()
    summaries: list[PreparationSummary] = []

    for bundle in bundles:
        vns_recording = load_vns_recording(bundle, use_corrected=use_corrected)

        # Convert VNS recording to HRVLoggerRecording format for reuse of summarize_recording
        hrv_recording = HRVLoggerRecording(
            participant_id=vns_recording.participant_id,
            rr_intervals=vns_recording.rr_intervals,
            events=vns_recording.events,
        )

        summaries.append(
            summarize_recording(
                hrv_recording,
                duplicate_rr_count=0,  # VNS doesn't track duplicates the same way
                duplicate_details=[],
                config=config,
                normalizer=normalizer,
                rr_file_count=1,  # VNS always has one file per participant
                events_file_count=1 if vns_recording.events else 0,
                source_app="VNS Analyse",
                vns_path=bundle.file_path,
            )
        )
    return summaries
