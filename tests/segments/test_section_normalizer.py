from pathlib import Path

from music_hrv.segments import SectionNormalizer


def build_normalizer() -> SectionNormalizer:
    return SectionNormalizer.from_yaml(Path("config/sections.yml"))


def test_normalize_ruhs_variations():
    normalizer = build_normalizer()
    assert normalizer.normalize("Ruhe pre start") == "rest_pre_start"
    assert normalizer.normalize("RUHE_PRE_ENDE") == "rest_pre_end"
    assert normalizer.normalize("Start Ruhe post ") == "rest_post_start"
    assert normalizer.normalize("ende   ruhe post") == "rest_post_end"


def test_normalize_measurement_aliases():
    normalizer = build_normalizer()
    assert normalizer.normalize("Messung start") == "measurement_start"
    assert normalizer.normalize("Messung Ende") == "measurement_end"
    assert normalizer.normalize("Pause start") == "pause_start"
    assert normalizer.normalize("PAUSE ENDE") == "pause_end"


def test_unknown_labels_return_fallback():
    normalizer = build_normalizer()
    assert normalizer.normalize("completely new label") == "unknown"
    assert normalizer.normalize(None) == "unknown"
    assert normalizer.normalize("", strict=False) == "unknown"
    assert normalizer.normalize("", strict=True) is None


def test_summary_tracks_raw_labels():
    normalizer = build_normalizer()
    summary = normalizer.summarize_labels(
        ["ruhe pre start", "unknown block", "Messung Ende"]
    )
    assert "rest_pre_start" in summary
    assert summary["rest_pre_start"] == {"ruhe pre start"}
    assert "measurement_end" in summary
    assert summary["measurement_end"] == {"Messung Ende"}
    assert summary["unknown"] == {"unknown block"}
