# CLAUDE.md

Quick reference for Claude Code working in this repository. **For detailed session history, see MEMORY.md**.

## Working Style with David

1. **GUI-first development**: David works with the Streamlit GUI. Describe what changes in the interface, not code details.
2. **Minimal code exposure**: Only show code when necessary.
3. **Track progress**: Use TodoWrite tool actively.
4. **Be proactive**: Describe what user will see differently in GUI.
5. **Update MEMORY.md**: Add detailed notes to MEMORY.md, keep this file slim.
6. **ALWAYS TEST**: Run `uv run pytest` and test imports before delivering code changes!

## Current Status (2025-11-27)

**Version**: `v0.2.3-patterns`

**GUI**: 5-tab Streamlit app with persistent storage
- Tab 1: Data & Groups (import with pattern selector, multi-file support, interactive Plotly plot)
- Tab 2: Event Mapping (define events, synonyms auto-lowercase)
- Tab 3: Group Management (create/edit/rename/delete groups)
- Tab 4: Sections (define time ranges between events)
- Tab 5: Analysis (NeuroKit2 HRV analysis, plots, metrics)

**Key Features**:
- Interactive Plotly RR visualization with click-to-add events
- Multiple RR/Events files per participant (auto-merge from restarts)
- Predefined ID pattern dropdown (6 formats + custom)
- Visible event sorting with immediate UI updates
- Timezone-aware timestamp handling

**Storage**: `~/.music_hrv/*.yml` (groups, events, sections persist across sessions)

**Status**: All tests passing (13/13), no linting errors

## Quick Commands

```bash
# Launch GUI
uv run streamlit run src/music_hrv/gui/app.py

# Run tests (ALWAYS DO THIS BEFORE DELIVERING!)
uv run pytest

# Lint
uv run ruff check src/ tests/ --fix
```

## Project Overview

Music HRV Toolkit analyzes heart rate variability from HRV Logger exports. Pipeline: ingest CSV → clean RR intervals → normalize events → compute HRV metrics (neurokit2) → export results.

## Architecture Essentials

**Data Flow**: CSVs → `discover_recordings()` → `RecordingBundle` (multi-file) → `load_recording()` → merge & sort → `PreparationSummary` → Streamlit session_state

**Key Files**:
- `src/music_hrv/gui/app.py` - Main Streamlit app (~2200 lines)
- `src/music_hrv/gui/persistence.py` - YAML storage helpers
- `src/music_hrv/io/hrv_logger.py` - CSV loading, multi-file support, ID patterns
- `src/music_hrv/cleaning/rr.py` - RR interval cleaning
- `src/music_hrv/prep/summaries.py` - Data summaries with file counts

**Predefined ID Patterns** (in `hrv_logger.py`):
- `\d{4}[A-Z]{4}` - 4 digits + 4 uppercase (default, e.g., 0123ABCD)
- `\d{4}[A-Za-z]{4}` - 4 digits + 4 letters (case insensitive)
- `[A-Za-z0-9]+` - Any alphanumeric
- `\d+` - Digits only
- `[A-Za-z]+\d+` - Letters + digits (e.g., P001)
- `[A-Za-z]+_\d+` - Underscore separated (e.g., sub_001)

## Important Patterns

**Multi-File Support**:
```python
# RecordingBundle stores all files for a participant
bundle.rr_paths: list[Path]  # All RR files
bundle.events_paths: list[Path]  # All Events files
bundle.has_multiple_files  # True if >1 file of either type
```

**Plotly Event Lines** (use `add_shape()` not `add_vline()` for datetime):
```python
fig.add_shape(type="line", x0=event_time, x1=event_time, ...)
```

**Timezone Handling**:
```python
# Make timestamps UTC-aware for comparison
if ts.tzinfo is None:
    ts = pd.Timestamp(ts).tz_localize('UTC')
```

## Version Tags

- `v0.2.3-patterns` - Predefined ID patterns, multi-file detection fix
- `v0.2.2-multi-files` - Multiple files per participant
- `v0.2.1-sorting-fix` - Timezone handling and visible sorting
- `v0.2.0-plotly-viz` - Interactive Plotly visualization

## Known Limitations / Next Steps

- [ ] Missing/error data detection and handling
- [ ] Section-based HRV analysis (currently analyzes whole recording)
- [ ] VNS Analyse loader not implemented

## References

- **MEMORY.md** - Detailed session history, implementation notes
- **SESSION_NOTES.md** - Technical documentation for v0.2.0
- **QUICKSTART.md** - User quick start guide
- `docs/HRV_project_spec.md` - Full specification

---

*Last updated: 2025-11-27 | Keep this file concise - add details to MEMORY.md*
