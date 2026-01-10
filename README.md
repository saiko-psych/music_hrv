# Music HRV Toolkit

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.6.7-green.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-18%20passing-brightgreen.svg)](tests/)
[![NeuroKit2](https://img.shields.io/badge/powered%20by-NeuroKit2-orange.svg)](https://neuropsychology.github.io/NeuroKit/)

**A research-grade HRV analysis toolkit for music intervention studies**

*Streamlit-based GUI for researchers with no coding background*

</div>

---

## Overview

Music HRV Toolkit is a Python-based pipeline for analyzing Heart Rate Variability (HRV) data in the context of music intervention research. It supports data from **HRV Logger** and **VNS Analyse**, provides automatic event detection and section-based analysis, and produces publication-ready metrics following current scientific guidelines.

### Key Capabilities

- **Multi-format Import**: HRV Logger CSV and VNS Analyse TXT files
- **Interactive Visualization**: WebGL-accelerated tachograms with click-to-add events
- **Section-Based Analysis**: Define time segments with start/end events and duration validation
- **Music Protocol Support**: Randomization groups, auto-generated music section boundaries
- **Scientific Rigor**: Follows 2024 Quigley guidelines for artifact handling and reporting
- **Export Ready**: CSV export for statistical analysis

---

## Quick Start

### Installation

**Requirements:** Python 3.11, 3.12, or 3.13 (Python 3.14 is not yet supported due to pyarrow compatibility)

```bash
# Clone the repository
git clone https://github.com/saiko-psych/music_hrv.git
cd music_hrv

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Launch the GUI

```bash
uv run streamlit run src/music_hrv/gui/app.py

# Or with test mode (auto-loads demo data)
uv run streamlit run src/music_hrv/gui/app.py -- --test-mode
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
- **Overlay Toggles**: Show/hide variability, exclusion zones, music sections
- **Zoom & Pan**: Full interactive exploration
- **Exclusion Zones**: Define and edit time ranges to exclude from analysis

### Section-Based Validation

```
Section: "Baseline Rest"
├── Start Event: rest_pre_start
├── End Events: rest_pre_end | measurement_start  (any can end section)
├── Expected Duration: 5 minutes
├── Tolerance: ±1 minute
└── Status: ✅ Valid (5.2 min)
```

- Define sections with flexible start/end event patterns
- Automatic duration validation with configurable tolerance
- Visual status indicators for quick quality checks

### Music Section Analysis

- **Playlist Groups**: Define randomization groups with different music orders
- **Auto-Generate Events**: Create music boundaries at configurable intervals
- **Per-Style Analysis**: Analyze HRV separately for each music type
- **Cross-Group Comparison**: Compare effects across randomization groups

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
| **Setup > Playlists** | Manage music randomization groups and labels |
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

### App Settings (`~/.music_hrv/`)

Global settings persist across sessions:

```
~/.music_hrv/
├── groups.yml              # Study group definitions
├── events.yml              # Event types and synonyms
├── sections.yml            # Section definitions
├── participants.yml        # Participant group assignments
├── participant_events.yml  # Backup of saved events
├── playlist_groups.yml     # Music randomization groups
├── settings.yml            # App settings (data folder, plot options)
└── music_labels.yml        # Music item labels
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
music_hrv/
├── src/music_hrv/
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
| **v0.6.7** | 2026-01 | Processed folder for events, --test-mode flag, Analysis tab fixes |
| v0.6.6 | 2025-12 | Settings panel, plot resolution slider, performance fixes |
| v0.6.5 | 2025-12 | Demo data, VNS event alignment fix, tachogram naming |
| v0.6.4 | 2025-12 | Multiple end events, VNS timestamp parsing |
| v0.6.3 | 2025-12 | Section-based validation with duration/tolerance |
| v0.6.2 | 2025-12 | Editable exclusion zones |
| v0.6.1 | 2025-12 | Auto-fill boundary events, click-to-add events |
| v0.6.0 | 2025-12 | Music Section Analysis mode |

---

## Roadmap

- [ ] Standalone executable (PyInstaller/Nuitka)
- [ ] Tutorial videos
- [ ] Playlist group comparison visualization
- [ ] Batch export for all participants

---

## References

- **Quigley, K. S., et al. (2024)** - Guidelines for reporting heart rate variability
- **Khandoker, A. H., et al. (2020)** - Artifact tolerance thresholds in HRV
- **NeuroKit2** - [neuropsychology.github.io/NeuroKit](https://neuropsychology.github.io/NeuroKit/)

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Made for HRV researchers**

*If you use this tool in your research, please cite appropriately.*

</div>
