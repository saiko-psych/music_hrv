# Music HRV Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Workflow: pytest](https://img.shields.io/badge/tests-pytest-informational.svg)](https://docs.pytest.org/)

Python-based pipeline for ingesting HRV Logger and VNS Analyse exports, cleaning RR intervals, segmenting sessions, and producing per-participant + group metrics powered by `neurokit2`. The project targets both command-line users and a future GUI for researchers with no coding background.

## Project Layout
- `src/music_hrv/` – modular codebase with `io`, `cleaning`, `segments`, `metrics`, `gui`, and `reports` packages.
- `config/pipeline.yml` – default configuration for directories, section mappings, cleaning thresholds, and QC gates.
- `data/raw/` and `data/processed/` – local-only input/output folders (ignored by git). Drop anonymised recordings under `data/raw` and inspect results under `data/processed`.
- `local_test_data/` – stash sensitive vendor-provided or large validation files here; directory is ignored so nothing leaks to GitHub.
- `docs/` – project specification and HRV Logger manual.
- `tests/` – pytest suite plus `tests/fixtures/` for lightweight synthetic CSV snippets.

Refer to `docs/CONTRIBUTING.md` for contributor guidelines, coding standards, and review expectations.

## Getting Started
```bash
uv sync --group dev      # install runtime + dev dependencies (Python 3.11)
uv run music-hrv --dry-run
uv run pytest
uv run streamlit run src/music_hrv/gui/app.py  # launches Streamlit GUI
```
`music-hrv --dry-run` validates configs/paths. Omit `--dry-run` once ingestion and analysis modules are implemented. Use `uv run python -m music_hrv.cli --help` to explore extra flags.

## Version Control
Repository initialised with git (`main` branch). Add your GitHub remote (e.g., `git remote add origin git@github.com:<org>/music_hrv.git`) and push the baseline once secrets are configured. The `.gitignore` prevents raw/test data and build artefacts from leaving your machine; keep participant files anonymised.

## Documentation
- `docs/CONTRIBUTING.md` — contributor workflow, coding standards, and review expectations.
- `docs/HRV_project_spec.md` — pipeline specification, terminology, and study assumptions.
- `docs/manual_HRV_logger.md` — HRV Logger reference for ingestion and QA workflows.

## Next Steps
1. Implement IO loaders for HRV Logger RR + Events and VNS exports.
2. Build RR cleaning + QC utilities aligned with `config/pipeline.yml` thresholds.
3. Wire segmentation + metrics modules into the CLI, then add GUI/reporting surfaces.
