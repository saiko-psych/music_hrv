"""Smoke tests for the command-line interface."""

from pathlib import Path

from rrational import cli


def test_parser_defaults() -> None:
    parser = cli.build_parser()
    args = parser.parse_args([])

    assert args.config == Path("config/pipeline.yml")
    assert args.inputs == Path("data/raw")
    assert args.output_dir == Path("data/processed")
    assert args.dry_run is False
    assert args.inspect_sections is False
    assert args.sections_config == Path("config/sections.yml")
    assert args.sections_format == "text"


def test_inspect_sections_empty(tmp_path, capsys) -> None:
    cli.main(["--inspect-sections", "--inputs", str(tmp_path)])
    out = capsys.readouterr().out
    assert "No section labels found" in out
