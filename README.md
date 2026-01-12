# RRational

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.7.0-green.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-18%20passing-brightgreen.svg)](tests/)
[![NeuroKit2](https://img.shields.io/badge/powered%20by-NeuroKit2-orange.svg)](https://neuropsychology.github.io/NeuroKit/)

**A rational approach to Heart Rate Variability analysis**

*Free, open-source HRV toolkit - like Kubios, but free*

</div>

---

## Overview

RRational is a free, open-source HRV analysis toolkit built for researchers. It provides a user-friendly Streamlit GUI for analyzing Heart Rate Variability data, with support for multiple data formats, interactive visualization, and publication-ready metrics following current scientific guidelines.

### Key Capabilities

- **Multi-format Import**: HRV Logger CSV and VNS Analyse TXT files
- **Interactive Visualization**: WebGL-accelerated tachograms with click-to-add events
- **Section-Based Analysis**: Define time segments with start/end events and duration validation
- **Research Protocol Support**: Group management, randomization groups, auto-generated section boundaries
- **Scientific Rigor**: Follows 2024 Quigley guidelines for artifact handling and reporting
- **Export Ready**: CSV export for statistical analysis

---

## Quick Start

### Prerequisites

1. **Python 3.11, 3.12, or 3.13** - [Download from python.org](https://www.python.org/downloads/)
   - Python 3.14 is **not yet supported** (pyarrow lacks wheels)
   - Check your version: `python --version`

2. **uv** (recommended package manager) - [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
   ```bash
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

### Installation

```bash
# Clone the repository
git clone https://github.com/saiko-psych/rrational.git
cd rrational

# If you have Python 3.14, install a compatible version first:
uv python install 3.11

# Install dependencies (uv will use the correct Python version)
uv sync

# Or with pip (requires correct Python version already active)
pip install -e .
```

> **Troubleshooting:** If you see `Failed to build pyarrow`, you likely have Python 3.14.
> Run `uv python install 3.11` first, then `uv sync` again.

### Launch the GUI

```bash
uv run streamlit run src/rrational/gui/app.py

# Or with test mode (auto-loads demo data)
uv run streamlit run src/rrational/gui/app.py -- --test-mode
```

### Try the Demo Data

The repository includes simulated demo data for testing:

```bash
# Demo data is in data/demo/
# - data/demo/hrv_logger/  (HRV Logger format)
# - data/demo/vns_analyse/ (VNS Analyse format)
```

Load the demo data in the GUI to explore all features without real participant data.

---

## Features

### Data Import & Management

| Feature | Description |
|---------|-------------|
| **HRV Logger** | Automatic pairing of `*_RR_*.csv` and `*_Events_*.csv` files |
| **VNS Analyse** | Single `.txt` file with date/time from filename |
| **Multi-file Merge** | Automatically combines multiple files per participant |
| **ID Extraction** | 6 predefined patterns + custom regex support |
| **Group Assignment** | Assign participants to study groups with CSV import |

### Interactive Tachogram

- **Fast Rendering**: Plotly Scattergl with intelligent 5000-point downsampling
- **Click-to-Add**: Add event markers by clicking on the plot
- **Overlay Toggles**: Show/hide variability, exclusion zones, sections
- **Zoom & Pan**: Full interactive exploration
- **Exclusion Zones**: Define and edit time ranges to exclude from analysis

### Section-Based Validation

```
Section: "Baseline Rest"
├── Start Event: rest_pre_start
├── End Events: rest_pre_end | measurement_start  (any can end section)
├── Expected Duration: 5 minutes
├── Tolerance: ±1 minute
└── Status: Valid (5.2 min)
```

- Define sections with flexible start/end event patterns
- Automatic duration validation with configurable tolerance
- Visual status indicators for quick quality checks

### HRV Metrics (via NeuroKit2)

| Domain | Metrics |
|--------|---------|
| **Time Domain** | RMSSD, SDNN, pNN50, Mean HR, HR Range |
| **Frequency Domain** | LF, HF, LF/HF ratio, Total Power |
| **Nonlinear** | SD1, SD2, DFA Alpha 1 |

---

## Scientific Best Practices

This toolkit follows current HRV research guidelines:

| Guideline | Implementation |
|-----------|----------------|
| **Quigley et al. (2024)** | Artifact rates dictate valid metrics |
| **Minimum Data** | 100 beats (time domain), 300 beats (frequency domain) |
| **Artifact Threshold** | Automatic exclusion at >10% artifacts |
| **Transparency** | Always reports artifact rates, beat counts, section boundaries |
| **Correction Method** | NeuroKit2 Kubios algorithm for 2-10% artifact rates |

---

## GUI Tabs

| Tab | Purpose |
|-----|---------|
| **Participants** | View tachogram, manage events, validate sections, define exclusion zones |
| **Setup > Events** | Define expected events with synonym patterns (regex support) |
| **Setup > Groups** | Create/edit study groups, assign expected sections |
| **Setup > Sections** | Define time ranges with duration and tolerance |
| **Analysis** | Run HRV analysis, view metrics, export CSV |

---

## Data Formats

### HRV Logger (CSV)

```
data/raw/hrv_logger/
├── 2025-03-15_RR_0001CTRL.csv      # RR intervals
└── 2025-03-15_Events_0001CTRL.csv  # Event markers
```

**RR File Format:**
```csv
date,rr,since start
2025-03-15 09:00:15.123,823,0
2025-03-15 09:00:15.946,812,823
...
```

**Events File Format:**
```csv
date,timestamp,annotation,manual
2025-03-15 09:00:15,0,Start Ruhe,false
2025-03-15 09:05:30,315000,Ruhe Ende,false
...
```

### VNS Analyse (TXT)

```
data/raw/vns_analyse/
└── VNS - 0001VNST, 0001VNST (0) - 15.03.2025 09.07 Langzeit, 1h 46min KORRIGIERT.txt
```

**File Format:**
```
Korrektur	Aktiv

Hauptparameter der VNS Analyse - Rohwerte (Nicht aktiv)
...

RR-Intervalle - Rohwerte (Nicht aktiv)
0.807	Notiz: Start Ruhe
0.838
0.851
...

RR-Intervalle - Korrigierte Werte (Aktiv)
0.807
0.838
...
```

---

## Configuration & Data Storage

### App Settings (`~/.rrational/`)

Global settings persist across sessions:

```
~/.rrational/
├── groups.yml              # Study group definitions
├── events.yml              # Event types and synonyms
├── sections.yml            # Section definitions
├── participants.yml        # Participant group assignments
├── participant_events.yml  # Backup of saved events
├── playlist_groups.yml     # Randomization groups
├── settings.yml            # App settings (data folder, plot options)
└── music_labels.yml        # Section labels
```

### Processed Data (`data/processed/`)

Edited events are saved alongside your data for portability:

```
data/processed/
├── 0001CTRL_events.yml     # Per-participant event files
├── 0002EXPR_events.yml     # Standardized format (v1.0)
└── ...
```

This allows sharing processed events with collaborators without sharing raw data.

---

## Project Structure

```
rrational/
├── src/rrational/
│   ├── gui/
│   │   ├── app.py           # Main Streamlit app (~3700 lines)
│   │   ├── tabs/            # Tab modules (data, setup, analysis)
│   │   ├── shared.py        # Caching and utilities
│   │   └── persistence.py   # YAML storage
│   ├── io/
│   │   ├── hrv_logger.py    # HRV Logger CSV parser
│   │   └── vns_analyse.py   # VNS Analyse TXT parser
│   ├── cleaning/
│   │   └── rr.py            # RR interval cleaning
│   └── segments/
│       └── section_normalizer.py
├── tests/                   # 18 pytest tests
├── data/
│   ├── demo/               # Demo data (included in repo)
│   │   ├── hrv_logger/     # Simulated HRV Logger files
│   │   └── vns_analyse/    # Simulated VNS files
│   ├── processed/          # Saved events per participant (git-ignored)
│   └── raw/                # Your data here (git-ignored)
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
│   └── generate_demo_data.py
└── QUICKSTART.md           # User guide
```

---

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest -v

# Lint and format
uv run ruff check src/ tests/ --fix
```

### Performance Notes

- **Lazy imports**: NeuroKit2 and Matplotlib loaded on demand
- **Plot downsampling**: 5000 points max for smooth rendering
- **Caching**: `@st.cache_data` for expensive operations
- **Startup time**: ~0.7s, participant switch ~200ms

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| **v0.7.0** | 2026-01 | Renamed to RRational, smart power formatting |
| v0.6.8 | 2026-01 | Professional analysis plots with reference values, data quality warnings |
| v0.6.7 | 2026-01 | Processed folder for events, --test-mode flag, Analysis tab fixes |
| v0.6.6 | 2025-12 | Settings panel, plot resolution slider, performance fixes |
| v0.6.5 | 2025-12 | Demo data, VNS event alignment fix, tachogram naming |

---

## Roadmap

- [ ] Standalone executable (PyInstaller/Nuitka)
- [ ] Tutorial videos
- [ ] Group comparison visualization
- [ ] Batch export for all participants
- [ ] PDF/HTML report generation

---

## References

- **Quigley, K. S., et al. (2024)** - [Publication guidelines for human heart rate and heart rate variability studies in psychophysiology](https://doi.org/10.1111/psyp.14604) - *Psychophysiology*, 61(9), 1-63.
- **Lipponen & Tarvainen (2019)** - [A robust algorithm for heart rate variability time series artefact correction](https://doi.org/10.1088/1361-6579/ab3c96) - *Physiological Measurement*, 40(10).
- **NeuroKit2** - [neuropsychology.github.io/NeuroKit](https://neuropsychology.github.io/NeuroKit/) - Open-source Python toolbox for neurophysiological signal processing.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**RRational - A rational approach to HRV**

*Free and open-source. If you use this tool in your research, please cite appropriately.*

</div>
