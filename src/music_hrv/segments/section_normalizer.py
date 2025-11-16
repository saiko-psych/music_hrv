"""Normalise free-text section labels to canonical identifiers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from music_hrv.config.sections import (
    DEFAULT_SECTIONS_PATH,
    SectionDefinition,
    SectionsConfig,
    load_sections_config,
)

_DEFAULT_FALLBACK = "unknown"


def _slugify(value: str) -> str:
    normalized = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii", "ignore")
    )
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


@dataclass(slots=True)
class SectionNormalizer:
    """Apply regex-based mappings to raw labels."""

    config: SectionsConfig
    fallback_label: str = _DEFAULT_FALLBACK
    _pattern_cache: dict[str, tuple[re.Pattern[str], ...]] = field(init=False)

    def __post_init__(self) -> None:
        self._pattern_cache = {
            definition.name: tuple(
                re.compile(pattern, re.IGNORECASE) for pattern in definition.synonyms
            )
            for definition in self.config.sections.values()
        }

    @classmethod
    def from_yaml(
        cls, path: Path | None = None, *, fallback_label: str = _DEFAULT_FALLBACK
    ) -> "SectionNormalizer":
        """Convenience constructor loading the shared YAML template."""

        config = load_sections_config(path or DEFAULT_SECTIONS_PATH)
        return cls(config=config, fallback_label=fallback_label)

    def normalize(self, label: str | None, *, strict: bool = False) -> str | None:
        """Return the canonical section name or the fallback/None."""

        if not label:
            return None if strict else self.fallback_label
        slug = _slugify(label)
        if not slug:
            return None if strict else self.fallback_label

        for canonical, patterns in self._pattern_cache.items():
            for pattern in patterns:
                if pattern.search(slug):
                    return canonical

        return None if strict else self.fallback_label

    def summarize_labels(self, labels: Iterable[str]) -> dict[str, set[str]]:
        """Map canonical names to the raw labels that matched them."""

        summary: dict[str, set[str]] = {}
        for raw_label in labels:
            canonical = self.normalize(raw_label)
            summary.setdefault(canonical or "unmatched", set()).add(raw_label)
        return summary

    def ordered_sections(self) -> tuple[str, ...]:
        """Return canonical sections in their expected order."""

        return self.config.canonical_order

    def definition_for(self, name: str) -> SectionDefinition | None:
        """Fetch the definition for a canonical section."""

        return self.config.sections.get(name)


__all__ = ["SectionNormalizer"]
