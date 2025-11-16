"""Section segmentation helpers driven by markers and YAML mappings."""

from music_hrv.segments.section_normalizer import SectionNormalizer
from music_hrv.segments.probe import SectionReport, scan_sections

__all__: list[str] = ["SectionNormalizer", "SectionReport", "scan_sections"]
