"""Summaries for data preparation previews surfaced in the GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from music_hrv.cleaning.rr import CleaningConfig, clean_rr_intervals, rr_summary
from music_hrv.io.hrv_logger import HRVLoggerRecording, discover_recordings, load_recording
from music_hrv.segments.section_normalizer import SectionNormalizer


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
            )
        )
    return summaries
