"""Helpers to inspect raw files and list discovered section labels."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from rrational.segments.section_normalizer import SectionNormalizer


@dataclass(slots=True)
class SectionReport:
    """Summary of raw labels and their canonical mapping."""

    path: Path
    source: str
    raw_labels: Sequence[str]
    normalized: dict[str | None, list[str]]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "source": self.source,
            "raw_labels": list(self.raw_labels),
            "normalized": {key or "unmatched": sorted(values) for key, values in self.normalized.items()},
        }


def _clean_csv_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\r\n", "\n")
    if "\r" in text:
        text = text.replace("\r", "\n")
    return text


def _extract_vns_labels(path: Path) -> list[str]:
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Notiz:" not in line:
            continue
        note = line.split("Notiz:", 1)[-1].strip()
        if note:
            labels.append(note)
    return labels


def _extract_hrv_logger_events(path: Path) -> list[str]:
    labels: list[str] = []
    content = _clean_csv_text(path)
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        raw = (row.get("annotation") or row.get(" annotation") or "").strip()
        if raw:
            labels.append(raw)
    return labels


def scan_sections(
    root: Path, *, config_path: Path | None = None, normalizer: SectionNormalizer | None = None
) -> list[SectionReport]:
    """Traverse raw directories and summarise discovered labels."""

    root = root.expanduser().resolve()
    if normalizer is None:
        normalizer = SectionNormalizer.from_yaml(config_path)

    reports: list[SectionReport] = []

    def _build_report(source: str, path: Path, labels: Iterable[str]) -> None:
        labels_list = list(labels)
        if not labels_list:
            return
        summary = {
            canonical: sorted(values)
            for canonical, values in normalizer.summarize_labels(labels_list).items()
        }
        reports.append(
            SectionReport(
                path=path.relative_to(root),
                source=source,
                raw_labels=sorted(set(labels_list)),
                normalized=summary,
            )
        )

    for txt_file in sorted(root.rglob("*.txt")):
        _build_report("vns", txt_file, _extract_vns_labels(txt_file))

    for events_file in sorted(root.rglob("*Events*.csv")):
        _build_report("hrv_logger", events_file, _extract_hrv_logger_events(events_file))

    return reports


__all__ = ["SectionReport", "scan_sections"]
