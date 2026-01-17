# CLAUDE.md

Quick reference for Claude Code. **Detailed history: MEMORY.md**

## Rules
- **CLAUDE.md**: Keep SHORT (<70 lines). Current state only.
- **MEMORY.md**: Session notes, version history, implementation details.
- **ALWAYS TEST**: `uv run pytest` before delivering!

## Quick Commands
```bash
uv run streamlit run src/rrational/gui/app.py  # Launch GUI
uv run pytest                                   # Run tests
uv run ruff check src/ tests/ --fix            # Lint
```

## Current Status

**Version**: `v0.7.4` | **Tests**: 18/18 passing

**GUI**: 5-tab Streamlit app (Data, Participants, Setup, Sections, Analysis)

**Storage**: Project-based (`MyProject/config/*.yml`) or global fallback (`~/.rrational/`)

**Data Sources**: [HRV Logger](https://www.hrv.tools/hrv-logger-faq.html) (CSV) and [VNS Analyse](https://apps.apple.com/de/app/vns-analyse/id990667927) (TXT)

## Scientific Best Practices (CRITICAL)

This is a **scientific research tool**. Follow HRV guidelines:
1. **Artifact handling**: 2024 Quigley guidelines - artifact rates dictate valid metrics
2. **Data requirements**: Min 100 beats (time), 300 beats (frequency)
3. **Correction**: NeuroKit2 Kubios algorithm (2-10% artifacts)
4. **Exclusion**: Exclude segments with >10% artifacts

## Performance Rules
1. **Lazy imports**: Use `get_neurokit()`, `get_matplotlib()`
2. **Downsample plots**: 5000 points max
3. **Cache data**: `@st.cache_data` - cache raw data, not objects
4. **NEVER use Plotly JSON serialization** - extremely slow

## Key Files
- `app.py` - Main app + Participants tab
- `tabs/` - data.py, setup.py, analysis.py
- `shared.py` - Utilities, caching
- `persistence.py` - YAML storage
- `project.py` - ProjectManager
- `welcome.py` - Welcome screen

## TODOs

**High Priority:**
- [ ] Playlist group comparison
- [ ] Setup section rework
- [ ] R-R power spectrum plot
- [ ] Batch processing / groupwise analysis
- [ ] Report generation (PDF/HTML)

**Low Priority:**
- [ ] Standalone app (PyInstaller/Nuitka)
- [ ] Tutorial videos

**Known limitations:**
- Plot zoom doesn't auto-load detail (use resolution slider)

## References
- **MEMORY.md** - Session history, version changelog
- **QUICKSTART.md** - User guide
- `docs/HRV_project_spec.md` - Full spec
