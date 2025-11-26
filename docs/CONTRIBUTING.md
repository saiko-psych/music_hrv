# Contributing to Music HRV Toolkit

This document explains how to work on the project, keep data protected, and publish trustworthy analyses. Treat it as the canonical guide for maintainers and contributors.

## Repository Layout
- `src/io`, `src/cleaning`, `src/segments`, `src/metrics`, `src/gui`, `src/reports`: modular Python packages for ingestion, RR preprocessing, session segmentation, metric calculation, GUI widgets, and report builders.
- `config/pipeline.yml`, `config/sections.yml`: configuration defaults for directory structure, thresholding, and section mappings.
- `data/raw/`, `data/processed/`: anonymised inputs/outputs that never leave the machine (both ignored by git). Keep larger validation recordings in `local_test_data/`.
- `docs/`: specifications (`HRV_project_spec.md`), HRV Logger manuals, and design notes (this file).
- `tests/`: pytest suites plus `tests/fixtures/` with minimal CSV snippets that mimic HRV Logger and VNS exports.

## Development Workflow
1. Install dependencies using `uv sync --group dev` (Python 3.11+).
2. Run the CLI with `uv run music-hrv --dry-run` or `uv run python -m music_hrv.cli --help` to explore options.
3. Launch the Streamlit GUI using `uv run streamlit run src/music_hrv/gui/app.py`.
4. Run tests with `uv run pytest` or scope to specific modules (e.g., `pytest tests/segments/test_section_normalizer.py -k baseline`).
5. Keep the sample data anonymised and store sensitive vendor dumps under `local_test_data/`.

## Coding Standards
- Target Python 3.11+, format with `black` (line length 88) and lint with `ruff` before committing.
- Use `snake_case` for modules/functions/variables and `PascalCase` for classes.
- Prefer explicit type hints and `dataclasses` for structured configuration; use `pathlib.Path` for filesystem work.
- Document every public function with NumPy-style docstrings describing inputs, outputs, HRV-specific assumptions (RR bounds, window length, etc.).

## Testing Expectations
- Run `pytest` for the full suite; scope to specific modules when iterating (e.g., `pytest tests/segments/test_section_builder.py -k baseline`).
- Maintain dedicated fixtures in `tests/fixtures` with anonymised CSV snippets.
- Target ≥85 % statement coverage and compare metric outputs against neurokit2 within acceptable tolerances.
- Add regression tests whenever YAML configs or GUI callbacks change behaviour.

## Version Control & Releases
- Follow Conventional Commits (`feat: add HRV Logger ingestion`, `fix: clamp RR artefact threshold`).
- Pull requests must summarize scope, reference design notes or specs, list manual test commands, and attach screenshots/gifs for GUI changes.
- Confirm that generated artefacts only contain anonymised data before sharing outside the lab.
