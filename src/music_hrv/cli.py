"""Command-line surface for the HRV processing pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


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
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    """Entry point invoked by `python -m music_hrv.cli` or the uv script hook."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

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
