# MEMORY.md - Detailed Session History

This file contains detailed session notes and implementation history. For quick reference, see CLAUDE.md.

---

## Session 2025-11-27 (Late Evening): Music Events, Quality Detection & Timing Validation

### Version Tag: `v0.3.0`

### Major Features Implemented:

#### 1. WebGL Accelerated Plotting
- Switched from `go.Scatter` to `go.Scattergl` for WebGL rendering
- Significantly faster rendering for large RR interval datasets
- Same functionality, better performance

#### 2. Auto-Create Quality Events
- **Gap Detection**: Creates `gap_start` and `gap_end` events for time gaps in data
  - Default threshold: 15 seconds (justified by HRV Logger per-packet timestamps)
  - User-configurable threshold
- **Variability Detection**: Creates `high_variability_start` and `high_variability_end` events
  - Uses coefficient of variation (CV) in sliding windows
  - User-configurable CV threshold (default: 20%)

#### 3. Music Section Events (Separate from Normal Events)
- Music events stored in separate `music_events` key (not mixed with `manual` events)
- Music events do NOT require matching to predefined events
- Auto-generation based on:
  - Participant's playlist/randomization group
  - 5-minute intervals between measurement boundaries
  - Cycling pattern (music_1 → music_2 → music_3 → repeat)

#### 4. Playlist/Randomization Groups
- New section in Tab 3 (Group Management) for managing playlist groups
- Pre-defined R1-R6 groups (all 6 permutations of 3 music types)
- Assign participants to playlist groups for automatic music order
- Persisted to `~/.music_hrv/playlist_groups.yml`

#### 5. Plot Visualization Toggles
- `Show variability segments` - toggle high variability shaded regions
- `Show time gaps` - toggle gap markers
- `Show music sections` - toggle music section shaded regions (color-coded per music type)
- `Show music events` - toggle music event vertical lines

#### 6. Timing Validation (Moved to Event Mapping Status)
- Validates rest periods: ≥3 min required, 5 min recommended
- Validates measurement sections: ~90 min expected
- Validates 5-min segment fit (warns about leftover time)
- Collapsible "Detected Boundary Events" expander

### Config Updates:
- `config/sections.yml` - Added `quality_markers` and `music_sections` categories
- `src/music_hrv/config/sections.py` - Loads new section types with group="quality" and group="music"

### Files Modified:
- `src/music_hrv/gui/app.py` - Major updates (~200 lines added)
- `src/music_hrv/gui/persistence.py` - Added playlist groups persistence
- `config/sections.yml` - Added quality_markers, music_sections
- `src/music_hrv/config/sections.py` - Load new section types

### Testing Results:
- ✅ All 13 tests passing
- ✅ No linting errors
- ✅ All features verified working

### User Feedback:
- Gap threshold 15s justified by HRV Logger packet behavior
- Music events should be separate category (not matched to predefined)
- Timing validation belongs in event mapping status, not music generator

---

## Session 2025-11-27 (Continued): Predefined Patterns & Bug Fixes

### Version Tag: `v0.2.3-patterns`

### Features Added:

#### 1. Predefined ID Pattern Selector
- Dropdown with 6 preset patterns + custom option
- Default: `\d{4}[A-Z]{4}` (4 digits + 4 uppercase letters, e.g., 0123ABCD)
- Other presets: case-insensitive, digits only, letters+digits, underscore-separated
- Pattern displayed as code block when using presets
- Custom pattern input for advanced users

#### 2. Fixed Multi-File Detection
- Changed default pattern from generic `[A-Za-z0-9]+` to specific `\d{4}[A-Z]{4}`
- Now correctly groups multiple files for same participant
- Files column shows `⚠️ 2RR/2Ev` when multiple files detected

#### 3. Bug Fixes
- Fixed `AttributeError: 'PreparationSummary' object has no attribute 'rr_file_count'`
- Added `getattr()` fallback for old cached summaries without file count fields

### Files Modified:
- `src/music_hrv/io/hrv_logger.py` - Added `PREDEFINED_PATTERNS` dict
- `src/music_hrv/io/__init__.py` - Exported `PREDEFINED_PATTERNS`
- `src/music_hrv/gui/app.py` - Added pattern selector dropdown

### Testing Note:
**IMPORTANT**: Always run `uv run pytest` and test imports before delivering code changes!

---

## Session 2025-11-27 (Evening): Interactive Plotly & Multi-File Support

### Version Tags Created:
- `v0.2.0-plotly-viz` - Interactive Plotly visualization
- `v0.2.1-sorting-fix` - Timezone handling and event sorting
- `v0.2.2-multi-files` - Multiple files per participant support

### Major Features Implemented:

#### 1. Interactive Plotly RR Visualization
- Replaced matplotlib with Plotly for interactive plots
- Real datetime timestamps from RR interval data
- Automatic gap detection where measurements are missing
- Interactive zoom/pan capabilities
- Click-to-add events functionality
- Event markers as vertical dashed lines with color coding

**Technical Solution**: Used `add_shape()` instead of `add_vline()` to avoid Plotly datetime handling bug.

#### 2. Multiple Files Per Participant
- **Problem**: Sometimes participants have multiple RR/Events files due to measurement restarts (Bluetooth errors, technical issues)
- **Solution**: `RecordingBundle` now stores `rr_paths: list[Path]` and `events_paths: list[Path]`
- `load_recording()` merges all files, sorted by timestamp
- GUI shows "Files" column with ⚠️ indicator for multiple files
- Format: "2RR/2Ev" shows file counts

#### 3. Config Moved to Data Import Section
- Moved ID pattern and cleaning thresholds from sidebar to Tab 1
- Settings now in collapsible "⚙️ Import Settings" expander
- Sidebar simplified to just show app title and auto-save status

#### 4. Timezone Handling Fixes
- Click-to-add events now creates timezone-aware timestamps (UTC)
- Sort function handles mixed timezone-aware/naive datetimes
- Helper `get_sort_key()` normalizes timestamps for comparison

#### 5. Event Sorting with Visible Updates
- Changed from `on_click` callbacks to button return values
- Added `st.rerun()` after sort/move actions
- Events now visibly reorder in the table immediately

### Files Modified:
- `src/music_hrv/gui/app.py` - Plotly integration, config reorganization
- `src/music_hrv/io/hrv_logger.py` - Multi-file support in RecordingBundle
- `src/music_hrv/prep/summaries.py` - File count tracking
- `.gitignore` - Added HRV data file patterns

### Key Code Changes:

**RecordingBundle (io/hrv_logger.py)**:
```python
@dataclass(slots=True)
class RecordingBundle:
    participant_id: str
    rr_paths: list[Path]  # All RR files
    events_paths: list[Path]  # All Events files

    @property
    def has_multiple_files(self) -> bool:
        return len(self.rr_paths) > 1 or len(self.events_paths) > 1
```

**Plotly Event Lines (app.py)**:
```python
# Using add_shape() instead of add_vline() for datetime compatibility
fig.add_shape(
    type="line",
    x0=event_time, x1=event_time,
    y0=y_min - 0.05 * y_range, y1=y_max + 0.05 * y_range,
    line=dict(color=color, width=2, dash='dash'),
)
```

**Sort with Visible Update (app.py)**:
```python
if st.button("🔄 Auto-Sort by Timestamp", key=f"auto_sort_{participant}"):
    all_events_copy.sort(key=get_sort_key)
    st.session_state.participant_events[participant]['events'] = all_events_copy
    st.rerun()  # Triggers visible update
```

### Testing Results:
- ✅ All 13 tests passing
- ✅ No linting errors
- ✅ Plotly plot renders correctly
- ✅ Click-to-add events works
- ✅ Sort/move buttons update visibly

### User Feedback:
- "this looks kinda nice!"
- Requested timezone-aware timestamp handling
- Wanted sorting to update visibly (not just toast message)
- Needed multiple file support for measurement restarts

---

## Session 2025-11-26 (Afternoon): Bug Fixes & UI Improvements

### User-Reported Issues (All Fixed):

#### Data & Groups Tab:
1. ✅ **Duration calculation bug** - Verified it correctly uses cleaned RR intervals
2. ✅ **Previous/Next buttons not working** - Fixed with unique keys and proper `st.rerun()` calls
3. ✅ **Event reordering not intuitive** - Replaced with ⬆️⬇️ arrow buttons for drag-like experience
4. ✅ **Section mapping status missing** - Ensured visibility with proper separators

#### Event Mapping Tab:
5. ✅ **Existing events not editable** - Redesigned with per-event expanders and inline editing
6. ✅ **Synonym management difficult** - Added ➕ Add and 🗑️ Delete buttons per synonym

#### Group Management Tab:
7. ✅ **No sections selection** - Added multiselect for sections per group (stored in `selected_sections`)

#### Analysis Tab:
8. ✅ **NeuroKit2 error** - Fixed by using `nk.intervals_to_peaks()` then `nk.hrv_time()` + `nk.hrv_frequency()`

### Implementation Changes:

**File Modified**: `src/music_hrv/gui/app.py` (1045 lines, up from 977)

**Key Code Changes**:

1. **Previous/Next Buttons** (lines 273-282):
```python
if st.button("⬅️ Previous", disabled=current_idx == 0, key="prev_participant_btn"):
    st.session_state.selected_participant_dropdown = participant_list[current_idx - 1]
    st.rerun()
```

2. **Event Reordering with Arrows** (lines 345-362):
```python
for idx, event_id in enumerate(current_order):
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        if st.button("⬆️", key=f"up_{event_id}_{idx}", disabled=idx == 0):
            current_order[idx], current_order[idx-1] = current_order[idx-1], current_order[idx]
            st.rerun()
    with col3:
        if st.button("⬇️", key=f"down_{event_id}_{idx}", disabled=idx == len(current_order)-1):
            current_order[idx], current_order[idx+1] = current_order[idx+1], current_order[idx]
            st.rerun()
```

3. **Per-Event Synonym Management** (lines 468-519):
```python
with st.expander(f"Event: {event_name}"):
    # Add synonym interface
    new_syn = st.text_input("New synonym", key=f"add_syn_{event_name}")
    if st.button("➕ Add", key=f"add_btn_{event_name}"):
        synonyms.append(new_syn.strip().lower())
        save_all_config()
        st.rerun()

    # Delete synonym buttons
    for syn_idx, syn in enumerate(synonyms):
        if st.button("🗑️", key=f"del_syn_{event_name}_{syn_idx}"):
            synonyms.pop(syn_idx)
            save_all_config()
            st.rerun()
```

4. **Group Sections Selection** (lines 607-623):
```python
available_sections = list(st.session_state.sections.keys())
selected_sections = st.multiselect(
    "Select sections for this group",
    options=available_sections,
    default=group_data.get("selected_sections", []),
    key=f"sections_{group_name}"
)
st.session_state.groups[group_name]["selected_sections"] = selected_sections
save_all_config()
```

5. **NeuroKit2 Fix** (lines 900-906, 1013-1019):
```python
# Convert RR intervals to peak indices
peaks = nk.intervals_to_peaks(rr_intervals_ms, sampling_rate=1000)

# Compute HRV metrics
hrv_time_results = nk.hrv_time(peaks, sampling_rate=1000, show=False)
hrv_freq_results = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)

# Combine results
hrv_results = pd.concat([hrv_time_results, hrv_freq_results], axis=1)
```

### Testing Results:
- ✅ All 13 tests passing
- ✅ No linting errors
- ✅ App launches successfully (port 8505)
- ✅ All 9 issues verified as resolved

### User Feedback:
- User emphasized fixing ALL issues, not just some
- Requested comprehensive verification of each fix
- Prefers GUI improvements over code visibility

---

## Session 2025-11-26 (Morning): Major GUI Overhaul

### User Requirements:
1. ✅ Group management should allow edit/rename/delete (including Default group)
2. ✅ All Groups and Events should be saved persistently and loaded on restart
3. ✅ Event matching should use lowercase automatically to reduce synonym count
4. ✅ Should be able to add more synonyms to existing events
5. ✅ Should be able to reorder events for each participant
6. ✅ Rename "Section Mapping" to "Event Mapping"
7. ✅ Create "Sections" concept - time ranges between events (e.g., rest_pre = rest_pre_start to rest_pre_end)
8. ✅ Add Analysis tab with NeuroKit2 integration for HRV analysis

### Implementation Details:

#### Persistent Storage (`src/music_hrv/gui/persistence.py`):
- Created YAML-based storage in `~/.music_hrv/`
- Files: `groups.yml`, `events.yml`, `sections.yml`
- Functions: `save_groups()`, `load_groups()`, `save_events()`, `load_events()`, `save_sections()`, `load_sections()`
- Auto-saves on every change to groups/events/sections

#### App Structure Changes (`src/music_hrv/gui/app.py`):
- Expanded from 608 lines to 977 lines
- Changed from 4 tabs to 5 tabs:
  - Tab 1: Data & Groups (added event reordering)
  - Tab 2: Event Mapping (renamed from "Events", added lowercase info)
  - Tab 3: Group Management (added edit/rename/delete functionality)
  - Tab 4: Sections (NEW - define time ranges between events)
  - Tab 5: Analysis (NEW - NeuroKit2 HRV analysis)

#### Key Features Implemented:

**1. Group Edit/Rename/Delete:**
- Each group expander has text inputs for Group Name (ID) and Label
- "Save Changes" button updates group data
- Renaming a group updates all participant assignments
- Delete button reassigns participants to Default before deletion
- Can delete any group including Default

**2. Persistent Storage:**
- Session state initialized from YAML files on startup
- `save_all_config()` function saves groups, events, and sections
- Called after every modification
- Config directory: `~/.music_hrv/`

**3. Lowercase Event Matching:**
- Info banner in Event Mapping tab explains lowercase matching
- All synonyms converted to lowercase on save: `.lower()` applied to all inputs
- Reduces synonym count significantly (e.g., "Rest Pre Start" and "rest pre start" are now the same)

**4. Add More Synonyms:**
- Events table is fully editable with `num_rows="dynamic"`
- Multi-line text column for synonyms
- "Save All Event Changes" button processes edits
- All synonyms converted to lowercase automatically

**5. Event Reordering:**
- New session state: `event_order` (maps participant_id -> list of event names)
- Interface: dropdown to select event, number input for new position (1-based)
- "Move Event" button performs reordering
- Uses list operations: remove, insert

**6. Sections Tab:**
- Define sections as time ranges between two events
- Each section has: name (ID), label (display), start_event, end_event
- Editable table with dropdown columns for event selection
- Sections saved to persistent storage
- Default sections initialized: rest_pre, measurement, pause, rest_post

**7. NeuroKit2 Analysis Tab:**
- Two modes: Single Participant, Group Analysis
- Section selection (multiselect)
- Loads recording, cleans RR intervals, runs `nk.hrv()`
- Displays key metrics: RMSSD, SDNN, pNN50, HF, LF, LF/HF
- Visualization: matplotlib plot of RR intervals
- Full results table with all NeuroKit2 metrics
- CSV download for results
- Group mode: analyzes all participants in group, shows summary statistics

#### Dependencies Added:
- `neurokit2>=0.2.7` - Added to pyproject.toml
- Includes matplotlib, scipy, scikit-learn, pywavelets as transitive deps

#### Files Modified:
1. `src/music_hrv/gui/app.py` - Complete rewrite with all features (977 lines)
2. `pyproject.toml` - Added neurokit2 dependency
3. `CLAUDE.md` - Updated with new features

#### Files Created:
1. `src/music_hrv/gui/persistence.py` - YAML storage helpers (62 lines)

### Testing Results:
- ✅ All 13 tests passing
- ✅ No linting errors (ruff)
- ✅ App launches successfully on port 8504
- ✅ Persistent storage working (saves to ~/.music_hrv/)

### User Feedback Points:
- User wants GUI-first development approach
- Minimal code exposure preferred
- Use TodoWrite to track progress
- Update MEMORY.md for detailed history
- Keep CLAUDE.md slim and efficient

---

## Previous Sessions Summary

### Initial Setup (Earlier 2025-11-26):
- Created CLAUDE.md file with project overview
- Migrated from Flet to Streamlit
- Cleaned up dead code (removed ~1,500 lines)
- Fixed duration calculation bug (now sums RR intervals)
- Added Date/Time column to participant table
- Built initial 4-tab GUI with editable dataframes

### Bug Fixes:
1. **Duration Calculation**: Changed from timestamp difference to summing RR intervals (`sum(rr_values) / 1000`)
2. **Recording DateTime**: Extract from first event (preferably rest_pre_start), added to PreparationSummary
3. **Linting Errors**: Fixed unused imports and unused variable assignments

---

## Architecture Notes

### GUI State Management:
- Streamlit session_state for runtime data
- YAML files for persistent configuration
- Config saved on every modification
- Config loaded on app startup

### Event Flow:
1. User loads data from `data/raw/hrv_logger`
2. Events detected and normalized to canonical names
3. User can edit events, add synonyms, create groups
4. Groups assign expected events to participants
5. Sections define time ranges for analysis
6. Analysis tab uses sections to compute HRV metrics

### Data Flow:
```
HRV Logger CSVs
  → load_hrv_logger_preview()
  → PreparationSummary objects
  → Streamlit session_state
  → Editable dataframes
  → User modifications
  → Persistent YAML storage
```

### Analysis Flow:
```
Participant selection
  → Load recording
  → Clean RR intervals
  → Extract section data (future: use section boundaries)
  → nk.hrv()
  → Display metrics + plot
  → CSV download
```

---

## Future Considerations

### Potential Improvements:
1. Actually use section boundaries in analysis (currently analyzes whole recording)
2. Add section-specific HRV analysis (extract RR intervals only from selected sections)
3. Add comparison between sections (e.g., rest_pre vs rest_post)
4. Add statistical tests for group comparisons
5. Add more visualizations (Poincaré plots, frequency domain)
6. Export results to standardized formats (BIDS, CSV templates)

### Known Limitations:
1. Section boundaries not yet implemented in analysis (just selects sections, doesn't filter data)
2. Event reordering is manual (no auto-sort by timestamp)
3. No validation for section logic (start event could be after end event)
4. No conflict detection for overlapping sections

---

## Working Patterns Observed

### User Preferences:
- Prefers seeing GUI changes over code
- Wants persistence so work isn't lost
- Values flexibility (edit/delete everything)
- Wants clear visual feedback
- Appreciates auto-saves

### Development Approach:
- GUI-first: implement features user will interact with
- Test immediately after major changes
- Keep code clean (linting, tests)
- Document decisions in MEMORY.md
- Keep CLAUDE.md concise

---

## File Structure

```
music_hrv/
├── src/music_hrv/
│   ├── gui/
│   │   ├── app.py          # Main Streamlit app (977 lines)
│   │   └── persistence.py   # YAML storage (62 lines)
│   ├── cleaning/
│   │   └── rr.py           # RR interval cleaning
│   ├── io/
│   │   └── hrv_logger.py   # CSV loading
│   ├── prep/
│   │   └── summaries.py    # Data summaries
│   └── segments/
│       └── section_normalizer.py  # Event normalization
├── tests/                   # All tests (13 passing)
├── data/raw/hrv_logger/     # User data directory
├── pyproject.toml           # Dependencies
├── CLAUDE.md               # Quick reference (slim)
└── MEMORY.md               # This file (detailed history)
```

---

## Important Code Patterns

### Session State Keys:
- `data_dir` - Path to data directory
- `summaries` - List of PreparationSummary objects
- `normalizer` - SectionNormalizer instance
- `cleaning_config` - CleaningConfig instance
- `groups` - Dict of group configurations
- `all_events` - Dict of all events with synonyms
- `participant_groups` - Maps participant_id to group name
- `manual_events` - Manually added events per participant
- `event_order` - Event ordering per participant
- `sections` - Section definitions

### Persistence Pattern:
```python
# Load on startup
if "groups" not in st.session_state:
    loaded_groups = load_groups()
    st.session_state.groups = loaded_groups or default_groups

# Save after changes
def save_all_config():
    save_groups(st.session_state.groups)
    save_events(st.session_state.all_events)
    save_sections(st.session_state.sections)

# Call after modifications
st.session_state.groups[group_name] = new_value
save_all_config()
```

### Lowercase Event Handling:
```python
# On input
synonyms_list = [s.strip().lower() for s in input.split("\n") if s.strip()]

# On save
synonyms = [s.strip().lower() for s in synonyms_str.split("\n") if s.strip()]
```

---

## Session Timestamps

- **2025-11-26 12:40** - App launch test successful (port 8502)
- **2025-11-26 13:11** - Updated app launch successful (port 8504)
- **2025-11-26 13:12** - All tests passing, no linting errors
- **2025-11-26 13:15** - MEMORY.md created, CLAUDE.md to be streamlined

---

*This file is automatically updated after each significant session. For current status, see CLAUDE.md.*
