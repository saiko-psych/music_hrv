from pathlib import Path

from music_hrv.io import discover_recordings, load_recording, load_rr_intervals

FIXTURES = Path("tests/fixtures/hrv_logger")


def test_discover_recordings_pairs_rr_and_events(tmp_path):
    for filename in ("sample_RR_0001TEST.csv", "sample_Events_0001TEST.csv"):
        (tmp_path / filename).write_text(
            (FIXTURES / filename).read_text(encoding="utf-8"), encoding="utf-8"
        )

    bundles = discover_recordings(tmp_path)
    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle.participant_id == "0001TEST"
    assert bundle.rr_path.name.startswith("sample_RR")
    assert bundle.events_path and bundle.events_path.name.startswith("sample_Events")


def test_load_rr_intervals_parses_numeric_values(tmp_path):
    rr_path = tmp_path / "sample_RR_0001TEST.csv"
    rr_path.write_text(
        (FIXTURES / "sample_RR_0001TEST.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    intervals, duplicates, duplicate_details = load_rr_intervals(rr_path)
    assert len(intervals) == 7
    assert duplicates == 0  # No duplicates in test fixture
    assert len(duplicate_details) == 0  # No duplicate details
    assert intervals[0].rr_ms == 665
    assert intervals[1].rr_ms == 150  # intentionally implausible for cleaning tests


def test_load_recording_combines_rr_and_events(tmp_path):
    for filename in ("sample_RR_0001TEST.csv", "sample_Events_0001TEST.csv"):
        (tmp_path / filename).write_text(
            (FIXTURES / filename).read_text(encoding="utf-8"), encoding="utf-8"
        )

    bundle = discover_recordings(tmp_path)[0]
    recording, duplicates, duplicate_details = load_recording(bundle)

    assert recording.participant_id == "0001TEST"
    assert len(recording.rr_intervals) == 7
    assert len(recording.events) == 2
    assert duplicates == 0  # No duplicates in test fixture
    assert len(duplicate_details) == 0  # No duplicate details
