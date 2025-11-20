"""Summaries for data preparation previews surfaced in the GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from music_hrv.cleaning.rr import CleaningConfig, clean_rr_intervals, rr_summary
from music_hrv.io.hrv_logger import HRVLoggerRecording, discover_recordings, load_recording
from music_hrv.segments.section_normalizer import SectionNormalizer


@dataclass(slots=True)
class EventStatus:
    """Mapping result for a raw event label."""

    raw_label: str
    canonical: str | None


@dataclass(slots=True)
class PreparationSummary:
    """Aggregate metrics for a cleaned recording."""

    participant_id: str
    total_beats: int
    retained_beats: int
    removed_beats: int
    artifact_ratio: float
    duration_s: float
    events_detected: int
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
    config: CleaningConfig | None = None,
    normalizer: SectionNormalizer | None = None,
) -> PreparationSummary:
    """Clean one participant recording and collect descriptive stats."""

    cleaned, stats = clean_rr_intervals(recording.rr_intervals, config)
    rr_stats = rr_summary(cleaned or recording.rr_intervals)
    normalizer = normalizer or SectionNormalizer.from_yaml()
    event_statuses: list[EventStatus] = []
    present_sections: set[str] = set()
    for marker in recording.events:
        canonical = normalizer.normalize(marker.label)
        event_statuses.append(EventStatus(raw_label=marker.label, canonical=canonical))
        if canonical:
            present_sections.add(canonical)
    return PreparationSummary(
        participant_id=recording.participant_id,
        total_beats=stats.total_samples,
        retained_beats=stats.retained_samples,
        removed_beats=stats.removed_samples,
        artifact_ratio=stats.artifact_ratio,
        duration_s=rr_stats["duration_s"],
        events_detected=len(recording.events),
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
        recording = load_recording(bundle)
        summaries.append(summarize_recording(recording, config=config, normalizer=normalizer))
    return summaries
