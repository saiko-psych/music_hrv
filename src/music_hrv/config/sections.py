"""Utilities for loading the section normalisation template."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml

DEFAULT_SECTIONS_PATH = Path("config/sections.yml")


@dataclass(slots=True)
class SectionDefinition:
    """Describes one canonical section and its matching rules."""

    name: str
    synonyms: tuple[str, ...]
    required: bool
    description: str | None = None
    group: str | None = None


@dataclass(slots=True)
class SectionsConfig:
    """Container with loaded section definitions."""

    version: int
    canonical_order: tuple[str, ...]
    sections: Mapping[str, SectionDefinition]

    def iter_definitions(self) -> Iterable[SectionDefinition]:
        """Yield section definitions in canonical order."""
        seen: set[str] = set()
        for name in self.canonical_order:
            definition = self.sections.get(name)
            if definition:
                seen.add(name)
                yield definition
        for name, definition in self.sections.items():
            if name not in seen:
                yield definition


def _normalize_synonyms(raw_synonyms: Iterable[str] | None) -> tuple[str, ...]:
    if not raw_synonyms:
        return ()
    return tuple(pattern for pattern in raw_synonyms if pattern)


def load_sections_config(
    path: Path | None = None, *, strict: bool = True
) -> SectionsConfig:
    """Read the YAML config file describing canonical sections."""

    source = path or DEFAULT_SECTIONS_PATH
    if not source.exists():
        message = f"Section config not found: {source}"
        if strict:
            raise FileNotFoundError(message)
        return SectionsConfig(
            version=1,
            canonical_order=(),
            sections={},
        )

    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid sections config format: {source}")

    canonical_order = tuple(str(name) for name in data.get("canonical_order", ()))
    sections: dict[str, SectionDefinition] = {}

    def _register_section(
        name: str, node: Mapping[str, object], *, group: str | None = None
    ) -> None:
        description = node.get("description")
        synonyms = _normalize_synonyms(node.get("synonyms"))
        sections[name] = SectionDefinition(
            name=name,
            synonyms=synonyms,
            required=bool(node.get("required", False)),
            description=str(description) if description is not None else None,
            group=group,
        )

    for name, node in (data.get("sections") or {}).items():
        if not isinstance(node, Mapping):
            continue
        _register_section(str(name), node)

    for name, node in (data.get("group_sections") or {}).items():
        if not isinstance(node, Mapping):
            continue
        _register_section(str(name), node, group="group")

    version = int(data.get("version", 1))
    return SectionsConfig(
        version=version,
        canonical_order=canonical_order,
        sections=sections,
    )


__all__ = ["SectionDefinition", "SectionsConfig", "load_sections_config"]
