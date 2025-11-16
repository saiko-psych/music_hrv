# Repository Guidelines

## Project Structure & Module Organization
`docs/` holds the working specifications (`HRV_project_spec.md`, `manual_HRV_logger.md`) and should stay source-of-truth for terminology and requirements. Place Python code under `src/` with clear subpackages such as `src/io` (file ingestion), `src/cleaning` (RR sanitising utilities), `src/segments` (marker parsing), and `src/metrics` (neurokit2 wraps). GUI assets (Qt/Kivy/Flet) belong in `src/gui`. Keep reusable configuration files (`config/pipeline.yml`, `config/sections.yml`) under `config/`, and store anonymised sample recordings inside `data/raw` (input) and `data/processed` (outputs). Mirror the code tree inside `tests/` so every module ships with tests and synthetic fixtures.

## Build, Test, and Development Commands
Create a virtual environment and install dependencies with `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`. Run the batch CLI using `python -m src.cli --config config/pipeline.yml --inputs data/raw` to produce participant and group CSVs. Launch the GUI prototype via `python -m src.gui.app` to validate non-programmer workflows. Use `pytest` for the full suite, or narrow the focus (`pytest tests/segments/test_section_builder.py -k baseline`). Regenerate artefact reports with `python -m src.reports.summarise data/processed`.

## Coding Style & Naming Conventions
Target Python 3.11+, format with `black` (line length 88) and lint with `ruff` prior to committing. Modules, files, functions, and variables use `snake_case`; classes keep `PascalCase`. Prefer explicit type hints, `dataclasses` for config blocks, and `Path` objects for filesystem work. All public functions need NumPy-style docstrings describing inputs, outputs, and HRV-specific assumptions (e.g., RR bounds, window length). use ruff and uv for better coding.

## Testing Guidelines
`pytest` is the primary framework; store fixtures in `tests/fixtures` with short, anonymised CSV snippets derived from HRV Logger and VNS exports. Name test files `test_<module>.py` and keep scenario-focused methods such as `test_segment_builder_flags_short_baseline`. Aim for ≥85 % statement coverage and validate that calculated metrics match neurokit2 within tolerances. Include regression tests for YAML config parsing and GUI callbacks whenever behaviour changes.

## Commit & Pull Request Guidelines
With no established history yet, adopt Conventional Commits (`feat: add HRV Logger ingestion`, `fix: clamp RR artefact threshold`). Keep commits focused and reference the relevant spec section in the body when changing requirements. Pull requests must summarise scope, link to design notes or issues, list manual test commands (`python -m src.cli …`, `pytest`), and attach screenshots/gifs for GUI changes. Request at least one review and confirm sample data remains de-identified.
