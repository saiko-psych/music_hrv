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
uv run streamlit run src/rrational/gui/app.py  # Launch GUI
uv run pytest                                   # Run tests
uv run ruff check src/ tests/ --fix            # Lint
```

## Current Status

**Version**: `v0.7.3` (RRational) | **Tests**: 18/18 passing

**GUI**: 5-tab Streamlit app
- Tab 1: Data & Groups (import, plot, quality detection, CSV import)
- Tab 2: Event Mapping (define events, synonyms)
- Tab 3: Group Management (groups + playlist groups)
- Tab 4: Sections (time ranges between events)
- Tab 5: Analysis (NeuroKit2 HRV metrics)

**Storage**:
- Project-based: `MyProject/config/*.yml` (groups, events, sections, etc.)
- Global fallback: `~/.rrational/*.yml` (for temporary workspace)
- Settings always in: `~/.rrational/settings.yml` (includes last_project for auto-load)

**Settings**: Sidebar expander with persistent defaults for:
- Default data folder
- Plot resolution (1000-20000 points)
- Plot options (events, exclusions, gaps, artifacts, etc.)

**Data Sources**: HRV Logger (CSV) and VNS Analyse (TXT) supported
- VNS loader: Parses date/time from filename (`dd.mm.yyyy hh.mm ...`), raw RR by default
- VNS display: ALL intervals shown (no filtering) - cleaning only at analysis time
- Sections: Support multiple end events (any can end section)
- Exclusion zones: Editable, vertical labels, auto-applied in analysis

## Performance Rules (CRITICAL)

1. **Lazy imports**: Use `get_neurokit()`, `get_matplotlib()` - NOT direct imports
2. **Downsample plots**: 5000 points max
3. **Cache data**: Use `@st.cache_data` - cache raw data, not objects
4. **NEVER use Plotly JSON serialization** - extremely slow

**Current timings**: Startup ~0.7s, participant switch ~200ms

## Key Files & Code Organization

- `src/rrational/gui/app.py` - Main app + Participants tab (~3800 lines)
- `src/rrational/gui/tabs/` - Tab modules (data.py, setup.py, analysis.py)
- `src/rrational/gui/shared.py` - Shared utilities, caching, helpers
- `src/rrational/gui/persistence.py` - YAML storage + settings
- `src/rrational/gui/project.py` - Project management (ProjectManager class)
- `src/rrational/gui/welcome.py` - Welcome screen + project creation wizard
- `src/rrational/gui/rrational_export.py` - .rrational export format handling

**IMPORTANT**: Keep `app.py` lean! Add new features to appropriate tab modules, not app.py.

## TODOs

**High Priority:**
- [ ] Playlist group comparison (compare music types across playlists)
- [ ] Setup section rework (better saving, design improvements)
- [ ] Evaluate hrv-analysis library for preprocessing (multiple ectopic methods, cleaner RR→NN pipeline)
- [ ] R-R power spectrum plot
- [ ] Generalized event system (music events with single border, extensible for other event types)
- [ ] Speed/loading improvements
- [ ] Batch processing / groupwise analysis (fix/finish, add group comparison)
- [ ] Report generation (PDF/HTML, scientific quality for methods sections)

**Low Priority:**
- [ ] Add nice banner at top (RR-plot style)
- [ ] Keyboard shortcuts for faster workflow
- [ ] Standalone app (no Python required) - PyInstaller/Nuitka
- [ ] Tutorial videos

**Known limitations:**
- Plot zoom doesn't auto-load detail (Streamlit/Plotly limitation - no relayout events)
  - Workaround: Resolution slider (1k-20k points), auto-shows all for <10k datasets

**Done:**
- [x] ~~Manual artifact demarking + improved detection workflow~~ (v0.7.3)
- [x] ~~Project management system (welcome screen, project creation, auto-load last project)~~ (v0.7.2)
- [x] ~~Ready for Analysis export (.rrational files with audit trail)~~ (v0.7.1)
- [x] ~~Kubios-like artifact workflow (method selector, threshold, correction preview)~~ (v0.7.0)
- [x] ~~Favicon (Poincaré plot style)~~ (v0.7.0)
- [x] ~~Project renamed to RRational~~ (v0.7.0)
- [x] ~~Smart power formatting (no more "0 ms²")~~ (v0.7.0)
- [x] ~~Light/dark theme switching~~ (v0.6.8)
- [x] ~~Event canonical mapping persistence fix~~ (v0.6.8)
- [x] ~~Professional analysis plots with reference values~~ (v0.6.8)
- [x] ~~Demo data for testing~~ (v0.6.5)

## References

- **MEMORY.md** - Session history, implementation details
- **QUICKSTART.md** - User guide
- `docs/HRV_project_spec.md` - Full spec
