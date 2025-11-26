"""RR-interval cleaning helpers."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Iterable, Sequence

from music_hrv.io.hrv_logger import RRInterval


@dataclass(slots=True)
class CleaningConfig:
    """Thresholds used for RR filtering."""

    rr_min_ms: int = 300
    rr_max_ms: int = 2200
    sudden_change_pct: float = 0.2


@dataclass(slots=True)
class CleaningStats:
    """Bookkeeping for removed beats."""

    total_samples: int
    retained_samples: int
    removed_samples: int
    artifact_ratio: float
    reasons: dict[str, int]

    @property
    def percent_removed(self) -> float:
        return self.artifact_ratio * 100


def _artifact_ratio(total: int, removed: int) -> float:
    if total <= 0:
        return 0.0
    return removed / total


def clean_rr_intervals(
    samples: Sequence[RRInterval] | Iterable[RRInterval],
    config: CleaningConfig | None = None,
) -> tuple[list[RRInterval], CleaningStats]:
    """Filter implausible beats and sudden jumps."""

    cfg = config or CleaningConfig()
    if not isinstance(samples, Sequence):
        samples = list(samples)
    cleaned: list[RRInterval] = []
    reasons = {"out_of_range": 0, "sudden_change": 0}

    previous_rr: int | None = None

    for sample in samples:
        rr = sample.rr_ms
        if rr < cfg.rr_min_ms or rr > cfg.rr_max_ms:
            reasons["out_of_range"] += 1
            continue

        if previous_rr is not None and previous_rr > 0:
            delta = abs(rr - previous_rr) / previous_rr
            if delta > cfg.sudden_change_pct:
                reasons["sudden_change"] += 1
                continue

        cleaned.append(sample)
        previous_rr = rr

    total = len(samples)
    stats = CleaningStats(
        total_samples=total,
        retained_samples=len(cleaned),
        removed_samples=sum(reasons.values()),
        artifact_ratio=_artifact_ratio(total, sum(reasons.values())),
        reasons=reasons,
    )
    return cleaned, stats


def rr_summary(samples: Sequence[RRInterval]) -> dict[str, float]:
    """Compute descriptive stats for a RR series.

    Duration is calculated by summing all RR intervals (physiological duration).
    Timestamps are available for validation but not used for duration calculation.
    """

    if not samples:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "duration_s": 0.0,
        }
    rr_values = [sample.rr_ms for sample in samples]
    min_ms = min(rr_values)
    max_ms = max(rr_values)
    mean_ms = fmean(rr_values)

    # Calculate duration by summing all RR intervals
    # This gives the actual physiological duration based on heartbeats
    duration_s = sum(rr_values) / 1000

    return {
        "min": float(min_ms),
        "max": float(max_ms),
        "mean": float(mean_ms),
        "duration_s": duration_s,
    }


__all__ = ["CleaningConfig", "CleaningStats", "clean_rr_intervals", "rr_summary"]
