"""Command-line surface for the HRV processing pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from music_hrv.segments import scan_sections


def build_parser() -> argparse.ArgumentParser:
    """Create a reusable argument parser for scripts and tests."""
    parser = argparse.ArgumentParser(
        prog="music-hrv",
        description="Batch ingest RR files, clean beats, and export HRV metrics.",
    )
    parser.add_argument(
        "--config",
        default=Path("config/pipeline.yml"),
        type=Path,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--inputs",
        default=Path("data/raw"),
        type=Path,
        help="Directory containing participant recordings (HRV Logger/VNS).",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("data/processed"),
        type=Path,
        help="Destination folder for aggregated metrics and reports.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and inputs without running computations.",
    )
    parser.add_argument(
        "--inspect-sections",
        action="store_true",
        help="List raw labels per file and show their canonical counterparts.",
    )
    parser.add_argument(
        "--sections-config",
        default=Path("config/sections.yml"),
        type=Path,
        help="Alternative sections.yml path for custom studies.",
    )
    parser.add_argument(
        "--sections-format",
        choices=("text", "json"),
        default="text",
        help="Output format when using --inspect-sections.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    """Entry point invoked by `python -m music_hrv.cli` or the uv script hook."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.inspect_sections:
        reports = scan_sections(args.inputs, config_path=args.sections_config)
        if args.sections_format == "json":
            payload = [report.to_dict() for report in reports]
            print(json.dumps(payload, indent=2))
            return
        if not reports:
            print(f"No section labels found under {args.inputs}")
            return
        for report in reports:
            print(f"[{report.source}] {report.path}")
            for canonical, raw_values in sorted(report.normalized.items()):
                joined = ", ".join(sorted(raw_values))
                print(f"  - {canonical}: {joined}")
        return

    if args.dry_run:
        print(
            "Dry run successful. Config: %s, inputs: %s"
            % (args.config.resolve(), args.inputs.resolve())
        )
        return

    print(
        "Pipeline stub: would process %s using %s → %s"
        % (args.inputs.resolve(), args.config.resolve(), args.output_dir.resolve())
    )


if __name__ == "__main__":  # pragma: no cover
    main()
