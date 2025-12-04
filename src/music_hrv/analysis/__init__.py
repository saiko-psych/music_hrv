"""Analysis modules for Music HRV."""

from music_hrv.analysis.music_sections import (
    ProtocolConfig,
    MusicSection,
    MusicSectionAnalysis,
    DurationMismatchStrategy,
    extract_music_sections,
    get_sections_by_music_type,
    get_sections_by_phase,
)

__all__ = [
    "ProtocolConfig",
    "MusicSection",
    "MusicSectionAnalysis",
    "DurationMismatchStrategy",
    "extract_music_sections",
    "get_sections_by_music_type",
    "get_sections_by_phase",
]
