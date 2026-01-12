from pathlib import Path

from rrational.cleaning.rr import CleaningConfig
from rrational.prep import load_hrv_logger_preview

FIXTURES = Path("tests/fixtures/hrv_logger")


def test_load_hrv_logger_preview_returns_summary(tmp_path):
    for filename in ("sample_RR_0001TEST.csv", "sample_Events_0001TEST.csv"):
        (tmp_path / filename).write_text(
            (FIXTURES / filename).read_text(encoding="utf-8"), encoding="utf-8"
        )

    summaries = load_hrv_logger_preview(
        tmp_path, pattern=r"(?P<participant>\d{4}TEST)", config=CleaningConfig(rr_min_ms=300, rr_max_ms=2000)
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.participant_id == "0001TEST"
    assert summary.total_beats == 7
    assert summary.retained_beats == 5
    assert summary.events_detected == 2
    assert summary.artifact_ratio > 0
    assert summary.events
    assert any(status.canonical for status in summary.events)
