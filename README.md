# Music HRV Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)
[![Version](https://img.shields.io/badge/version-0.3.0-green.svg)](pyproject.toml)

Python-based pipeline for ingesting HRV Logger exports, cleaning RR intervals, segmenting sessions by music style, and producing per-participant + group HRV metrics powered by `neurokit2`. Features a Streamlit GUI for researchers with no coding background.

## Quick Start

```bash
# Install dependencies
uv sync --group dev

# Launch the GUI
uv run streamlit run src/music_hrv/gui/app.py

# Run tests
uv run pytest
```

## Features (v0.3.0)

### Data Import & Management
- **HRV Logger CSV Import**: Load RR intervals and event markers from HRV Logger exports
- **Multi-file Support**: Automatically merges multiple files per participant (handles measurement restarts)
- **Participant ID Patterns**: 6 predefined patterns + custom regex for extracting IDs from filenames
- **Group Assignment**: Assign participants to study groups with expected events

### Interactive Visualization
- **WebGL-Accelerated Plots**: Fast rendering with Plotly Scattergl
- **Click-to-Add Events**: Click on the plot to add manual event markers
- **Toggle Overlays**: Show/hide variability segments, time gaps, music sections
- **Zoom & Pan**: Full interactive exploration of RR interval data

### Quality Detection
- **Gap Detection**: Automatically detect time gaps in recordings (default: >15s threshold)
- **Variability Detection**: Flag high-variability segments using coefficient of variation
- **Auto-Create Events**: Generate `gap_start/end` and `high_variability_start/end` events

### Music Section Analysis
- **Playlist Groups**: Define randomization groups (R1-R6) with different music orders
- **Auto-Generate Music Events**: Create music section boundaries at 5-minute intervals
- **Per-Music-Style Analysis**: Analyze HRV separately for each music type (music_1, music_2, music_3)
- **Separate Event Category**: Music events are distinct from protocol events

### Timing Validation
- **Rest Period Checks**: Validates rest periods are ≥3 min (5 min recommended)
- **Measurement Duration**: Validates ~90 min measurement sections
- **5-Minute Segment Fit**: Warns about leftover time after music segments

### HRV Analysis
- **NeuroKit2 Integration**: Full HRV analysis (time domain, frequency domain)
- **Section-Based Analysis**: Analyze specific time segments
- **Group Analysis**: Compare HRV across participant groups
- **CSV Export**: Download results for further analysis

## GUI Tabs

| Tab | Purpose |
|-----|---------|
| **Data & Groups** | Import data, assign participants to groups, view/edit events, interactive RR plot |
| **Event Mapping** | Define expected events with synonyms (auto-lowercase matching) |
| **Group Management** | Create/edit/delete groups, manage playlist randomization groups |
| **Sections** | Define time ranges between events (e.g., rest_pre = rest_pre_start → rest_pre_end) |
| **Analysis** | Run NeuroKit2 HRV analysis, view metrics and plots |

## Data Storage

Configuration persists across sessions in `~/.music_hrv/`:
- `groups.yml` - Study group definitions
- `events.yml` - Event types and synonyms
- `sections.yml` - Section definitions
- `participants.yml` - Participant assignments
- `playlist_groups.yml` - Music randomization groups

## Project Structure

```
music_hrv/
├── src/music_hrv/
│   ├── gui/
│   │   ├── app.py           # Main Streamlit application
│   │   └── persistence.py   # YAML storage helpers
│   ├── io/
│   │   └── hrv_logger.py    # HRV Logger CSV parsing
│   ├── cleaning/
│   │   └── rr.py            # RR interval cleaning
│   ├── config/
│   │   └── sections.py      # Section configuration loader
│   ├── prep/
│   │   └── summaries.py     # Data preparation summaries
│   └── segments/
│       └── section_normalizer.py  # Event normalization
├── config/
│   └── sections.yml         # Canonical section definitions
├── tests/                   # Pytest suite (13 tests)
├── docs/                    # Documentation
└── data/raw/hrv_logger/     # Place HRV Logger exports here
```

## Configuration

### Section Configuration (`config/sections.yml`)

Defines canonical events and their synonyms:
- **sections**: Protocol events (rest_pre_start, measurement_start, etc.)
- **quality_markers**: Auto-generated quality events (gap_start, high_variability_start)
- **music_sections**: Music change events (music_1_start, music_2_end, etc.)
- **groups**: Protocol templates with required sections

### Cleaning Thresholds

- **RR Range**: 300-2000 ms (default)
- **Gap Threshold**: 15 seconds (HRV Logger packet-based)
- **Variability CV**: 20% (coefficient of variation)

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Lint code
uv run ruff check src/ tests/ --fix

# Format code
uv run black src/ tests/
```

## Future Roadmap

### High Priority
- [ ] Better explanations and help text in GUI
- [ ] Performance optimization (background processing, reduce loading indicators)
- [ ] Batch processing: apply predefined settings to all participants automatically
- [ ] Only prompt user interaction when issues are detected

### Medium Priority
- [ ] Improve UI layout (spacing, element sizing, visual polish)
- [ ] Plot customization (colors, titles, axis labels)
- [ ] Alternative visualization libraries evaluation

### Low Priority
- [ ] VNS Analyse loader implementation
- [ ] BIDS export format
- [ ] Statistical comparison tools

## Documentation

- `CLAUDE.md` - Quick reference for development
- `MEMORY.md` - Detailed session history and implementation notes
- `docs/HRV_project_spec.md` - Full pipeline specification
- `docs/manual_HRV_logger.md` - HRV Logger device reference

## License

MIT License - see LICENSE file for details.

---

*Version 0.3.0 | Last updated: 2025-11-27*
