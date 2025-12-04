"""Tests for music section extraction and validation."""

from datetime import datetime, timedelta

import pytest

from music_hrv.analysis.music_sections import (
    ProtocolConfig,
    DurationMismatchStrategy,
    extract_music_sections,
    get_sections_by_music_type,
)
from music_hrv.io.hrv_logger import RRInterval


def create_test_rr_intervals(
    start_time: datetime,
    duration_minutes: float,
    rr_ms: int = 800,
) -> list[RRInterval]:
    """Create test RR intervals for a given duration."""
    intervals = []
    current_time = start_time
    cumulative_ms = 0

    total_ms = duration_minutes * 60 * 1000
    while cumulative_ms < total_ms:
        intervals.append(RRInterval(
            timestamp=current_time,
            rr_ms=rr_ms,
            elapsed_ms=cumulative_ms,
        ))
        cumulative_ms += rr_ms
        current_time += timedelta(milliseconds=rr_ms)

    return intervals


def test_extract_music_sections_basic():
    """Test basic music section extraction."""
    start = datetime(2024, 1, 1, 10, 0, 0)

    # Create 90 minutes of RR data
    rr_intervals = create_test_rr_intervals(start, 90)

    events = {
        "measurement_start": start,
        "pause_start": start + timedelta(minutes=45),
        "pause_end": start + timedelta(minutes=45),
        "measurement_end": start + timedelta(minutes=90),
    }

    protocol = ProtocolConfig(
        expected_duration_min=90.0,
        section_length_min=5.0,
        pre_pause_sections=9,
        post_pause_sections=9,
    )

    analysis = extract_music_sections(
        rr_intervals=rr_intervals,
        events=events,
        music_order=["music_1", "music_2", "music_3"],
        protocol=protocol,
    )

    assert len(analysis.sections) == 18
    assert analysis.valid_sections == 18
    assert analysis.incomplete_sections == 0


def test_extract_music_sections_short_recording():
    """Test extraction with shorter than expected recording."""
    start = datetime(2024, 1, 1, 10, 0, 0)

    # Create only 80 minutes of RR data (10 min short)
    rr_intervals = create_test_rr_intervals(start, 80)

    events = {
        "measurement_start": start,
        "pause_start": start + timedelta(minutes=40),
        "pause_end": start + timedelta(minutes=40),
        "measurement_end": start + timedelta(minutes=80),
    }

    protocol = ProtocolConfig(
        expected_duration_min=90.0,
        section_length_min=5.0,
        pre_pause_sections=9,
        post_pause_sections=9,
        min_section_duration_min=4.0,
    )

    analysis = extract_music_sections(
        rr_intervals=rr_intervals,
        events=events,
        music_order=["music_1", "music_2", "music_3"],
        protocol=protocol,
        mismatch_strategy=DurationMismatchStrategy.FLAG_ONLY,
    )

    # Should have warnings about duration mismatch
    assert len(analysis.warnings) > 0
    assert analysis.duration_mismatch_s > 0


def test_sections_by_music_type():
    """Test grouping sections by music type."""
    start = datetime(2024, 1, 1, 10, 0, 0)
    rr_intervals = create_test_rr_intervals(start, 30)

    events = {
        "measurement_start": start,
        "measurement_end": start + timedelta(minutes=30),
    }

    protocol = ProtocolConfig(
        expected_duration_min=30.0,
        section_length_min=5.0,
        pre_pause_sections=6,
        post_pause_sections=0,
    )

    analysis = extract_music_sections(
        rr_intervals=rr_intervals,
        events=events,
        music_order=["music_1", "music_2", "music_3"],
        protocol=protocol,
    )

    by_type = get_sections_by_music_type(analysis)

    # Should have 2 sections of each type (6 sections / 3 types)
    assert "music_1" in by_type
    assert "music_2" in by_type
    assert "music_3" in by_type
    assert len(by_type["music_1"]) == 2
    assert len(by_type["music_2"]) == 2
    assert len(by_type["music_3"]) == 2


def test_protocol_config_properties():
    """Test ProtocolConfig computed properties."""
    protocol = ProtocolConfig(
        expected_duration_min=90.0,
        section_length_min=5.0,
        pre_pause_sections=9,
        post_pause_sections=9,
    )

    assert protocol.total_sections == 18
    assert protocol.expected_pre_pause_min == 45.0
    assert protocol.expected_post_pause_min == 45.0
