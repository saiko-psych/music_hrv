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

**Version**: `v0.6.0` | **Tests**: 18/18 passing

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
- Scroll-to-top when navigating participants

## Performance Rules (CRITICAL)

1. **Lazy imports**: Use `get_neurokit()`, `get_matplotlib()` - NOT direct imports
2. **Downsample plots**: 5000 points max
3. **Cache data**: Use `@st.cache_data` - cache raw data, not objects
4. **NEVER use Plotly JSON serialization** - extremely slow

**Current timings**: Startup ~0.7s, participant switch ~200ms

## Key Files

- `src/music_hrv/gui/app.py` - Main Streamlit app
- `src/music_hrv/gui/persistence.py` - YAML storage
- `src/music_hrv/io/hrv_logger.py` - CSV loading

## TODOs

- [ ] Playlist group comparison (compare music types across playlists)
- [x] ~~Music Section Analysis mode~~ (DONE v0.6.0)
- [x] ~~Section-based HRV analysis~~ (DONE v0.6.0)
- [x] ~~Add `use_corrected` option in UI for VNS data~~ (DONE v0.5.0)
- [x] ~~Participant section works for VNS data~~ (DONE v0.5.0)
- [x] ~~Scroll to top on navigation~~ (DONE v0.5.0)
- [x] ~~VNS loader fix: only imports one section~~ (DONE v0.4.2)
- [x] ~~App column shows recording source~~ (DONE v0.4.2)
- [x] ~~CSV import for groups/playlists~~ (DONE v0.3.4)

## References

- **MEMORY.md** - Session history, implementation details
- **QUICKSTART.md** - User guide
- `docs/HRV_project_spec.md` - Full spec
