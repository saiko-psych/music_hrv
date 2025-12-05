# CLAUDE.md

Quick reference for Claude Code. **Detailed history: MEMORY.md**

## How to Maintain This File

**CLAUDE.md**: Keep SHORT (<70 lines). Only current state, no history.
- Update version/status when releasing
- Update TODOs when completing/adding tasks
- Update performance timings if they change significantly

**MEMORY.md**: Add detailed session notes there, including:
- What was changed and why
- Code snippets and patterns discovered
- Profiling results and learnings
- Version history

**Rule**: If adding more than 5 lines here, it probably belongs in MEMORY.md.

---

## Working Style

1. **GUI-first**: Describe interface changes, not code
2. **Minimal code**: Only show when necessary
3. **Track progress**: Use TodoWrite actively
4. **Update MEMORY.md**: Add detailed session notes there
5. **ALWAYS TEST**: `uv run pytest` before delivering!
6. **OPTIMIZE FOR SPEED**: Profile before optimizing

## Scientific Best Practices (CRITICAL)

This is a **scientific research tool**. ALL implementations MUST follow current HRV guidelines:

1. **Artifact handling**: Follow 2024 Quigley guidelines - artifact rates dictate valid metrics
2. **Data requirements**: Min 100 beats (time domain), 300 beats (frequency domain)
3. **Transparency**: Always report artifact rates, beat counts, section boundaries
4. **Correction**: Use NeuroKit2 Kubios algorithm for artifact correction (2-10% artifacts)
5. **Exclusion**: Exclude segments with >10% artifacts

**References** (see `help_text.py` for full documentation):
- Quigley et al. (2024) - Publication guidelines for HRV studies
- Khandoker et al. (2020) - Artifact tolerance thresholds

## Quick Commands

```bash
uv run streamlit run src/music_hrv/gui/app.py  # Launch GUI
uv run pytest                                   # Run tests
uv run ruff check src/ tests/ --fix            # Lint
```

## Current Status

**Version**: `v0.6.2` | **Tests**: 18/18 passing

**GUI**: 5-tab Streamlit app
- Tab 1: Data & Groups (import, plot, quality detection, batch processing, CSV import)
- Tab 2: Event Mapping (define events, synonyms)
- Tab 3: Group Management (groups + playlist groups)
- Tab 4: Sections (time ranges between events)
- Tab 5: Analysis (NeuroKit2 HRV metrics)

**Storage**: `~/.music_hrv/*.yml`

**Data Sources**: HRV Logger (CSV) and VNS Analyse (TXT) supported
- VNS loader: Only imports one RR section (raw by default, `use_corrected` option in Import Settings)
- VNS display: ALL intervals shown (no filtering) - cleaning only at analysis time
- Participant view works for both HRV Logger and VNS data
- Exclusion zones: Editable, vertical labels, auto-applied in analysis

## Performance Rules (CRITICAL)

1. **Lazy imports**: Use `get_neurokit()`, `get_matplotlib()` - NOT direct imports
2. **Downsample plots**: 5000 points max
3. **Cache data**: Use `@st.cache_data` - cache raw data, not objects
4. **NEVER use Plotly JSON serialization** - extremely slow

**Current timings**: Startup ~0.7s, participant switch ~200ms

## Key Files & Code Organization

- `src/music_hrv/gui/app.py` - Main app + Participants tab (~3700 lines)
- `src/music_hrv/gui/tabs/` - Tab modules (data.py, setup.py, analysis.py)
- `src/music_hrv/gui/shared.py` - Shared utilities, caching, helpers
- `src/music_hrv/gui/persistence.py` - YAML storage

**IMPORTANT**: Keep `app.py` lean! Add new features to appropriate tab modules, not app.py.

## TODOs

**Next up:**
- [ ] Fix batch processing
- [ ] Remove redundant code/UI elements (cleanup)
- [ ] Flexible section end events (end with any event, or selection of events)
- [ ] Standalone app (no Python required) - PyInstaller/Nuitka
- [ ] Example simulated data for testing/demo
- [ ] Tutorial videos

**Future:**
- [ ] Playlist group comparison (compare music types across playlists)

**Done:**
- [x] ~~Editable exclusion zones~~ (v0.6.2)
- [x] ~~Vertical exclusion labels~~ (v0.6.2)
- [x] ~~Exclusion zones affect timing validation~~ (v0.6.2)
- [x] ~~Auto-fill boundary events~~ (v0.6.1)
- [x] ~~Custom events from plot click~~ (v0.6.1)
- [x] ~~Music Section Analysis mode~~ (v0.6.0)

## References

- **MEMORY.md** - Session history, implementation details
- **QUICKSTART.md** - User guide
- `docs/HRV_project_spec.md` - Full spec
