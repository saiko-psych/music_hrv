from music_hrv.cleaning.rr import CleaningConfig, clean_rr_intervals, clean_rr_intervals_with_flags, rr_summary
from music_hrv.io.hrv_logger import RRInterval


def build_samples():
    return [
        RRInterval(timestamp=None, rr_ms=650, elapsed_ms=0),
        RRInterval(timestamp=None, rr_ms=120, elapsed_ms=650),
        RRInterval(timestamp=None, rr_ms=640, elapsed_ms=1290),
        RRInterval(timestamp=None, rr_ms=8000, elapsed_ms=1930),
        RRInterval(timestamp=None, rr_ms=630, elapsed_ms=2560),
    ]


def test_clean_rr_intervals_filters_outliers():
    samples = build_samples()
    cleaned, stats = clean_rr_intervals(samples, CleaningConfig(rr_min_ms=300, rr_max_ms=2000))

    assert len(cleaned) == 3
    assert stats.total_samples == 5
    assert stats.removed_samples == 2
    assert stats.reasons["out_of_range"] == 2
    assert stats.reasons["sudden_change"] == 0


def test_rr_summary_reports_min_max_mean_duration():
    samples = build_samples()
    cleaned, _ = clean_rr_intervals(samples, CleaningConfig(rr_min_ms=300, rr_max_ms=2000))
    summary = rr_summary(cleaned)

    assert summary["min"] == 630.0
    assert summary["max"] == 650.0
    # Duration calculated by summing RR intervals: 650 + 640 + 630 = 1920 ms = 1.92 s
    assert summary["duration_s"] == (650 + 640 + 630) / 1000


def test_clean_rr_intervals_with_flags_keeps_all():
    """Test that clean_rr_intervals_with_flags keeps all intervals but flags problematic ones."""
    samples = build_samples()
    flagged, stats = clean_rr_intervals_with_flags(samples, CleaningConfig(rr_min_ms=300, rr_max_ms=2000))

    # All 5 intervals should be returned (not removed)
    assert len(flagged) == 5

    # Stats should match regular clean_rr_intervals
    assert stats.total_samples == 5
    assert stats.removed_samples == 2
    assert stats.retained_samples == 3
    assert stats.reasons["out_of_range"] == 2

    # Check which ones are flagged
    flags = [f.is_flagged for f in flagged]
    assert flags == [False, True, False, True, False]  # 120 and 8000 are out of range

    # Flagged intervals should have reasons
    assert flagged[1].flag_reason == "out_of_range"
    assert flagged[3].flag_reason == "out_of_range"

    # Non-flagged should have no reason
    assert flagged[0].flag_reason is None
    assert flagged[2].flag_reason is None
