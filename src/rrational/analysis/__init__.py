"""Analysis modules for Music HRV."""

from rrational.analysis.music_sections import (
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
