# CLAUDE.md

Quick reference for Claude Code working in this repository. **For detailed session history, see MEMORY.md**.

## Working Style with David

1. **GUI-first development**: David works with the Streamlit GUI. Describe what changes in the interface, not code details.
2. **Minimal code exposure**: Only show code when necessary.
3. **Track progress**: Use TodoWrite tool actively.
4. **Be proactive**: Describe what user will see differently in GUI.
5. **Update MEMORY.md**: Add detailed notes to MEMORY.md, keep this file slim.

## Current Status (2025-11-26)

**GUI**: 5-tab Streamlit app with persistent storage
- Tab 1: Data & Groups (import, assign, reorder events)
- Tab 2: Event Mapping (define events, synonyms auto-lowercase)
- Tab 3: Group Management (create/edit/rename/delete groups)
- Tab 4: Sections (define time ranges between events)
- Tab 5: Analysis (NeuroKit2 HRV analysis, plots, metrics)

**Storage**: `~/.music_hrv/*.yml` (groups, events, sections persist across sessions)

**Status**: All tests passing (13/13), no linting errors, ready for user testing

## Quick Commands

```bash
# Launch GUI
uv run streamlit run src/music_hrv/gui/app.py

# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/ --fix
```

## Project Overview

Music HRV Toolkit analyzes heart rate variability from HRV Logger/VNS Analyse exports. Pipeline: ingest CSV → clean RR intervals → normalize events → compute HRV metrics (neurokit2) → export results.

## Architecture Essentials

**Data Flow**: CSVs → `load_hrv_logger_preview()` → `PreparationSummary` → Streamlit session_state → editable dataframes → YAML storage

**Key Files**:
- `src/music_hrv/gui/app.py` - Main Streamlit app (977 lines)
- `src/music_hrv/gui/persistence.py` - YAML storage helpers
- `src/music_hrv/cleaning/rr.py` - RR interval cleaning
- `src/music_hrv/prep/summaries.py` - Data summaries
- `src/music_hrv/segments/section_normalizer.py` - Event normalization (lowercase regex)

**Config Files**:
- `config/sections.yml` - Canonical section definitions
- `~/.music_hrv/groups.yml` - User's group configurations
- `~/.music_hrv/events.yml` - User's event definitions
- `~/.music_hrv/sections.yml` - User's section definitions

## Important Patterns

**Session State Keys**: `data_dir`, `summaries`, `groups`, `all_events`, `participant_groups`, `event_order`, `sections`

**Persistence Pattern**:
```python
# Load on startup
if "groups" not in st.session_state:
    st.session_state.groups = load_groups() or default_groups

# Save after changes
save_all_config()  # Saves groups, events, sections to YAML
```

**Lowercase Event Matching**: All event synonyms converted to `.lower()` automatically

**Duration Calculation**: Sum RR intervals (`sum(rr_values) / 1000`), NOT timestamp difference

## Conventions

- **File Paths**: Always use `pathlib.Path`
- **Time Units**: RR intervals in ms (int), durations in seconds (float)
- **Data Classes**: `@dataclass(slots=True)` for efficiency
- **Testing**: `pytest`, target ≥85% coverage
- **Code Style**: Black (line 88), Ruff, Python 3.11+
- **Type Hints**: Explicit, use `from __future__ import annotations`

## Known Limitations

- Section boundaries not yet used in HRV analysis (analyzes whole recording)
- VNS Analyse loader not implemented
- Metrics/visualizations/reports modules stubbed out

## References

- **MEMORY.md** - Detailed session history, implementation notes, user feedback
- `docs/HRV_project_spec.md` - Full specification
- `docs/manual_HRV_logger.md` - Device-specific details
- `config/sections.yml` - Section definitions and synonyms

---

*Last updated: 2025-11-26 | Keep this file concise - add details to MEMORY.md*
