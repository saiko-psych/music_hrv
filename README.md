# RRational

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11-3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.7.2-green.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-18%20passing-brightgreen.svg)](tests/)
[![NeuroKit2](https://img.shields.io/badge/powered%20by-NeuroKit2-orange.svg)](https://neuropsychology.github.io/NeuroKit/)

**A rational approach to Heart Rate Variability analysis**

*Free, open-source HRV toolkit - like Kubios, but free*

</div>

---

## Overview

RRational is a free, open-source HRV analysis toolkit built for researchers. It provides a user-friendly Streamlit GUI for analyzing Heart Rate Variability data, with support for multiple data formats, interactive visualization, and publication-ready metrics following current scientific guidelines.

### Key Capabilities

- **Project Management**: Self-contained project folders with data, config, and results
- **Multi-format Import**: HRV Logger CSV and VNS Analyse TXT files
- **Interactive Visualization**: WebGL-accelerated tachograms with click-to-add events
- **Section-Based Analysis**: Define time segments with start/end events and duration validation
- **Research Protocol Support**: Group management, randomization groups, auto-generated section boundaries
- **Scientific Rigor**: Follows 2024 Quigley guidelines for artifact handling and reporting
- **Ready for Analysis Export**: Save inspected data as `.rrational` files with full audit trail
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

### Updating

Already have RRational installed? Update to the latest version:

```bash
cd rrational

# Pull latest changes
git pull origin main

# Sync dependencies
uv sync
```

> **Troubleshooting:** If you see `failed to canonicalize script path` when launching the app,
> your virtual environment may be corrupted. Delete it and re-sync:
> ```bash
> # Windows
> rmdir /s /q .venv
> uv sync
>
> # macOS/Linux
> rm -rf .venv
> uv sync
> ```

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

## Project Management

RRational uses a **project-based workflow** where each study is a self-contained folder.

### Creating a Project

1. Launch RRational - the welcome screen appears
2. Click **"Create New Project"**
3. Choose a location and enter project details
4. Select your data sources (HRV Logger, VNS Analyse)
5. Click "Create Project"

### Project Structure

```
MyStudy/
├── project.rrational          # Project metadata (YAML)
├── data/
│   ├── raw/                   # Your original HRV files
│   │   ├── hrv_logger/        # Place HRV Logger CSV files here
│   │   └── vns/               # Place VNS Analyse TXT files here
│   └── processed/             # Exported .rrational files, saved events
├── config/                    # Project-specific configuration
│   ├── groups.yml             # Study groups
│   ├── events.yml             # Event definitions
│   ├── sections.yml           # Section definitions
│   └── ...                    # Other settings
└── analysis/                  # Future: analysis results
```

### Why Projects?

| Feature | Project | Temporary Workspace |
|---------|---------|---------------------|
| **Settings saved** | In project folder | In `~/.rrational/` |
| **Portable** | Yes - share the folder | No - tied to your computer |
| **Auto-load** | Remembers last project | No persistence |
| **Best for** | Real research | Quick testing |

### Auto-Load

RRational remembers your last used project and automatically loads it on startup.
Click "Switch Project" in the sidebar to choose a different project.

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

## GUI Overview

### Welcome Screen

On launch, choose how to work:
- **Recent Projects**: Quick access to previously opened projects
- **Open Existing Project**: Browse for a project folder
- **Create New Project**: Set up a new study with organized folder structure
- **Continue Without Project**: Use temporary workspace for quick testing

### Main Tabs

| Tab | Purpose |
|-----|---------|
| **Data** | Import data, view participant overview, load sources |
| **Participants** | View tachogram, manage events, validate sections, define exclusion zones |
| **Setup > Events** | Define expected events with synonym patterns (regex support) |
| **Setup > Groups** | Create/edit study groups, assign expected sections |
| **Setup > Sections** | Define time ranges with duration and tolerance |
| **Analysis** | Run HRV analysis, view metrics, export CSV |

### Sidebar

- **Project indicator**: Shows current project name
- **Switch Project**: Return to welcome screen
- **Settings**: Data folder, plot options, resolution
- **Save/Reset**: Persist or reset configuration

---

## Data Formats

RRational supports data from two popular HRV recording apps:

| App | Platform | Format | Link |
|-----|----------|--------|------|
| **HRV Logger** | iOS/Android | CSV | [hrv.tools](https://www.hrv.tools/hrv-logger-faq.html) |
| **VNS Analyse** | iOS | TXT | [App Store](https://apps.apple.com/de/app/vns-analyse/id990667927) |

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

### Project-Based Storage (Recommended)

When using a project, all configuration is stored **inside the project folder**:

```
MyProject/
├── config/                    # Project configuration
│   ├── groups.yml             # Study group definitions
│   ├── events.yml             # Event types and synonyms
│   ├── sections.yml           # Section definitions
│   ├── participants.yml       # Participant group assignments
│   ├── playlist_groups.yml    # Randomization groups
│   └── music_labels.yml       # Section labels
└── data/processed/            # Saved events and exports
    ├── 0001CTRL_events.yml    # Per-participant events
    └── 0001CTRL_baseline.rrational  # Ready for Analysis exports
```

### Global Settings (`~/.rrational/`)

Some settings are always stored globally:

```
~/.rrational/
├── settings.yml            # App settings (last project, plot options)
└── [config files]          # Fallback when not using a project
```

### Sharing Projects

To share a study with collaborators:
1. Copy the entire project folder
2. Recipient opens it with "Open Existing Project"
3. All configuration and processed data is included

> **Note**: Raw HRV data files are in `data/raw/` - include or exclude as needed.

---

## Code Structure

```
rrational/
├── src/rrational/
│   ├── gui/
│   │   ├── app.py           # Main Streamlit app (~3800 lines)
│   │   ├── project.py       # Project management (ProjectManager class)
│   │   ├── welcome.py       # Welcome screen and project wizard
│   │   ├── tabs/            # Tab modules (data, setup, analysis)
│   │   ├── shared.py        # Caching and utilities
│   │   ├── persistence.py   # YAML storage and settings
│   │   └── rrational_export.py  # .rrational export format
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
| **v0.7.2** | 2026-01 | Project management system (welcome screen, project folders, auto-load) |
| v0.7.1 | 2026-01 | Ready for Analysis export (.rrational files with audit trail) |
| v0.7.0 | 2026-01 | Renamed to RRational, smart power formatting |
| v0.6.8 | 2026-01 | Professional analysis plots with reference values, data quality warnings |
| v0.6.7 | 2026-01 | Processed folder for events, --test-mode flag, Analysis tab fixes |
| v0.6.6 | 2025-12 | Settings panel, plot resolution slider, performance fixes |
| v0.6.5 | 2025-12 | Demo data, VNS event alignment fix, tachogram naming |

---

## Roadmap

- [ ] Group comparison visualization (compare HRV across study groups)
- [ ] PDF/HTML report generation (publication-ready methods sections)
- [ ] Manual beat editing (click to mark/unmark individual artifacts)
- [ ] Standalone executable (PyInstaller/Nuitka)
- [ ] Tutorial videos

---

## References

- **Quigley, K. S., et al. (2024)** - [Publication guidelines for human heart rate and heart rate variability studies in psychophysiology](https://doi.org/10.1111/psyp.14604) - *Psychophysiology*, 61(9), 1-63.
- **Lipponen & Tarvainen (2019)** - [A robust algorithm for heart rate variability time series artefact correction](https://doi.org/10.1088/1361-6579/ab3c96) - *Physiological Measurement*, 40(10).
- **NeuroKit2** - [neuropsychology.github.io/NeuroKit](https://neuropsychology.github.io/NeuroKit/) - Open-source Python toolbox for neurophysiological signal processing.

---

## Contributing & Reporting Issues

Found a bug or have a feature request?

1. **Check existing issues** at [Issues](https://github.com/saiko-psych/rrational/issues)
2. **Create a new issue** using our templates - they guide you through what information to include
3. **Include screenshots** if possible - very helpful for GUI issues!

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for detailed guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**RRational - A rational approach to HRV**

*Free and open-source. If you use this tool in your research, please cite appropriately.*

</div>
