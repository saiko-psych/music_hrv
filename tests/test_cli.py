"""Smoke tests for the command-line interface."""

from pathlib import Path

from music_hrv import cli


def test_parser_defaults() -> None:
    parser = cli.build_parser()
    args = parser.parse_args([])

    assert args.config == Path("config/pipeline.yml")
    assert args.inputs == Path("data/raw")
    assert args.output_dir == Path("data/processed")
    assert args.dry_run is False
