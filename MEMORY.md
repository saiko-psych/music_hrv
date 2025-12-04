# MEMORY.md - Detailed Session History

This file contains detailed session notes and implementation history. For quick reference, see CLAUDE.md.

---

## Session 2025-12-04: Music Section Analysis & Event UI Improvements

### Version Tags: `v0.6.0`, `v0.6.1`

### Major Features Implemented:

#### 1. Music Section Analysis Mode (v0.6.0)
New dedicated analysis mode for protocol-based 5-minute music sections:

- **Protocol Configuration**:
  - Expected duration (default 90 min)
  - Section length (default 5 min)
  - Pre-pause sections (default 9)
  - Post-pause sections (default 9)
  - Min section duration and beats for validity

- **Duration Mismatch Strategies**:
  - `strict`: Fail if duration doesn't match expected
  - `proportional`: Scale section durations to fit actual recording
  - `flag_only`: Create sections as-is with warnings (default)

- **Section Extraction**: Creates MusicSection objects with:
  - Section index, music type, phase (pre/post pause)
  - Start/end times and RR intervals
  - Duration, beat count, validity flags

- **HRV by Section**: Computes time-domain HRV metrics per section using NeuroKit2

#### 2. Event Adding UI Improvements
- **Plot Click to Add Events**: Click on RR plot to set timestamp, quick-add form appears immediately
- **Custom Events**: Added "Custom..." option in quick-add dropdown for arbitrary event names
- **Second-level Time Precision**: Changed from Streamlit time_input (60s min) to text_input with HH:MM:SS parsing
- **Immediate Feedback**: Moved add-event form inside fragment for instant click response

#### 3. Auto-fill Boundary Events
- Shows which boundary events are present (✅) or missing (❌)
- Configurable pre-pause and post-pause durations (default 45 min each)
- "Auto-fill X missing event(s)" button calculates timestamps from existing events
- Works forwards from measurement_start or backwards from measurement_end

#### 4. Participant Events Persistence
- Events saved to `~/.music_hrv/participant_events.yml`
- Save/load/delete functions for participant-specific events
- "Reset to Original" button to clear saved edits

### Bug Fixes (v0.6.1):

1. **HRV Computation Error**: NeuroKit2 expects peaks not RR intervals
   - Fixed by using `nk.intervals_to_peaks(rr_values, sampling_rate=1000)`

2. **EventMarker Missing offset_s**: Added `offset_s=None` to constructor call

3. **Timezone-aware vs Naive Datetime Comparison**:
   - Added `sort_key()` and `normalize_ts()` helpers
   - Normalizes timestamps by removing tzinfo before comparison/arithmetic
   - Fixed in timing validation section

### Files Created:
- `src/music_hrv/analysis/music_sections.py` - Protocol-based music section extraction
- `tests/analysis/test_music_sections.py` - 4 tests for music section extraction
- `tests/analysis/__init__.py` - Test module init

### Files Modified:
- `src/music_hrv/gui/app.py` - Music section UI, event adding improvements, auto-fill
- `src/music_hrv/gui/persistence.py` - Protocol and participant events persistence

### Testing Results:
- ✅ All 18 tests passing
- ✅ No linting errors

---

## Session 2025-12-04 (Continued): VNS Flagging + NeuroKit2 Artifact Detection

### Version Tag: `v0.5.1`

### Changes Made:

#### 1. New Default Cleaning Thresholds
- **Min RR**: 200ms (was 300ms)
- **Max RR**: 2000ms (was 2200ms)
- **Sudden Change**: 100% = disabled (was 20%)
- Rationale: Sudden change threshold interferes with normal HRV. Use NeuroKit2 artifact detection instead.

#### 2. VNS Loader - No Hardcoded Filter
- Removed `if 200 <= rr_ms <= 3000:` filter from `load_vns_recording()`
- ALL RR intervals now loaded regardless of value

#### 3. VNS Display with Flagging
- VNS data shows ALL intervals with correct timestamps
- Intervals outside min/max thresholds flagged in RED
- Uses `clean_rr_intervals_with_flags()` for display
- Cleaning only applied at analysis time

#### 4. NeuroKit2 Artifact Detection in Participant View
- New checkbox: "Show artifacts (NeuroKit2)"
- Uses Kubios algorithm via `signal_fixpeaks`
- Detects: ectopic, missed, extra, long/short beats
- Artifacts shown as orange X markers on plot
- Info bar shows count by artifact type

### Files Modified:
- `src/music_hrv/cleaning/rr.py` - New defaults, `clean_rr_intervals_with_flags()`
- `src/music_hrv/io/vns_analyse.py` - Removed hardcoded RR filter
- `src/music_hrv/gui/app.py` - VNS flagging, `cached_artifact_detection()`, artifact display
- `src/music_hrv/gui/tabs/data.py` - Updated slider to show percentage

### Testing Results:
- ✅ All 14 tests passing
- ✅ No linting errors

---

## Session 2025-12-04: VNS UI Support & Navigation Improvements

### Version Tag: `v0.5.0`

### Features Implemented:

#### 1. VNS `use_corrected` Option in UI
- Added checkbox in Import Settings → VNS Analyse Settings
- Toggle between raw and corrected RR values from VNS files
- Clears VNS cache when setting changes to reload data
- Stored in `st.session_state.vns_use_corrected`

#### 2. Participant View Works for VNS Data
- Added `rr_paths`, `events_paths`, and `vns_path` fields to `PreparationSummary`
- Paths are stored when loading preview data
- Participant view checks `source_app` and uses appropriate loader:
  - VNS Analyse: Uses `cached_load_vns_recording()` with stored `vns_path`
  - HRV Logger: Uses `cached_load_recording()` with stored `rr_paths`/`events_paths`
  - Fallback: Re-discovers recordings for old cached summaries

#### 3. Scroll to Top on Navigation
- Added `scroll_to_top_trigger` session state flag
- Set in Previous/Next button callbacks and dropdown on_change
- Injects JavaScript to scroll `stAppViewContainer` to top

### Files Modified:
- `src/music_hrv/prep/summaries.py` - Added path fields to PreparationSummary, use_corrected param
- `src/music_hrv/gui/shared.py` - Added cached_load_vns_recording, scroll_to_top helper
- `src/music_hrv/gui/tabs/data.py` - Added VNS use_corrected checkbox
- `src/music_hrv/gui/app.py` - VNS-aware participant loading, scroll trigger

### Testing Results:
- ✅ All 13 tests passing
- ✅ No linting errors

---

## Session 2025-12-03 (Continued): VNS Loader Fix & App Column

### Version Tag: `v0.4.2`

### Issues Fixed:

#### 1. VNS Loader Importing Data Twice
**Problem**: VNS Analyse files contain TWO RR sections:
- `RR-Intervalle - Rohwerte (Nicht aktiv)` = Raw/Uncorrected values
- `RR-Intervalle - Korrigierte Werte (Aktiv)` = Corrected values

The loader was importing ALL RR values, essentially doubling the data.

**Solution**:
- Added section tracking with `current_section` variable
- Added `use_corrected` parameter (default: False = raw values)
- Only parse RR values when `current_section == target_section`

```python
VNS_RAW_SECTION = "RR-Intervalle - Rohwerte"
VNS_CORRECTED_SECTION = "RR-Intervalle - Korrigierte Werte"

def load_vns_recording(bundle, *, use_corrected=False):
    target_section = VNS_CORRECTED_SECTION if use_corrected else VNS_RAW_SECTION
    current_section = None

    for line in lines:
        if VNS_RAW_SECTION in line:
            current_section = VNS_RAW_SECTION
        elif VNS_CORRECTED_SECTION in line:
            current_section = VNS_CORRECTED_SECTION

        if current_section != target_section:
            continue
        # ... parse RR values
```

#### 2. App Column Showing "Unknown"
**Problem**: Even with `source_app` field added to PreparationSummary, cached data still showed "Unknown".

**Solution**: Explicitly set `source_app` after loading to handle cached data:
```python
# Ensure source_app is set (handles old cached data)
for s in summaries:
    if not hasattr(s, 'source_app') or s.source_app == "Unknown":
        object.__setattr__(s, 'source_app', app_name)
```

### Files Modified:
- `src/music_hrv/io/vns_analyse.py` - Section tracking, use_corrected parameter
- `src/music_hrv/gui/tabs/data.py` - Explicit source_app setting

### Future Tasks (noted for next session):
- [ ] Add UI option for `use_corrected` (raw vs corrected VNS data)
- [ ] Participant section should work for VNS data
- [ ] Scroll to top when clicking Next/Previous or switching sections

### Testing Results:
- ✅ All 13 tests passing
- ✅ Changes committed and tagged v0.4.2

---

## Session 2025-12-03: CSV Import & UI Improvements

### Version Tags: `v0.3.3`, `v0.3.4`

### Major Features Implemented:

#### 1. CSV Import for Group/Playlist Assignments (v0.3.3, v0.3.4)
- **Value→Label Mappings**: Define what CSV values mean (e.g., group "5" = "MAR", playlist "1" = "R1")
- **Column Mapping**: Configurable column names (default: `code`, `group`, `playlist`)
- **Auto-create Groups/Playlists**: Creates groups and playlist entries with labels when importing
- **CSV Status Column**: Shows which participants have imported data (✓✓=Both, G=Group, P=Playlist, —=None)
- **Scrollable Preview**: CSV preview limited to 200px height with priority columns first

#### 2. App Column in Participants Table (v0.3.3)
- Shows recording app source (HRV Logger, VNS Analyse)
- `source_app` field added to `PreparationSummary` dataclass
- Set during data loading based on which loader is used

#### 3. Label-Only Display (v0.3.3)
- Group and Playlist columns show only labels, not "code (label)" format
- Cleaner table presentation
- Mapping from label→code maintained for saving

#### 4. CSV Import UX Fixes (v0.3.4)
- Expander stays open when file is uploaded or labels are added/removed
- Used session state to track expander state
- Check for uploaded file before expander renders

### Files Modified:
- `src/music_hrv/gui/tabs/data.py` - CSV import UI, participants table columns
- `src/music_hrv/prep/summaries.py` - Added `source_app` field

### Project Structure Review:
The project is well-organized with clean separation:
```
src/music_hrv/
├── cleaning/        # RR interval cleaning
├── cli.py           # Command-line interface
├── config/          # Configuration (sections)
├── gui/             # Streamlit GUI
│   ├── app.py       # Main app
│   ├── persistence.py  # YAML storage
│   ├── shared.py    # Shared utilities, cached functions
│   └── tabs/        # Tab implementations
├── io/              # Data loading (HRV Logger, VNS Analyse)
├── prep/            # Data preparation
└── segments/        # Segment handling
```

### Testing Results:
- ✅ All 13 tests passing
- ✅ No linting errors

---

## Session 2025-12-03: Startup Performance & Lazy Imports

### Changes Made:

#### 1. Project Structure Cleanup
- Deleted `nul` artifact file (empty Windows device file)
- Verified project organization is clean and well-structured
- All tests passing (13/13), no linting errors

#### 2. Lazy Import for neurokit2 (Major Performance Win)
**Problem:** `import neurokit2` takes ~0.9s, blocking app startup even when not needed.

**Solution:** Created `get_neurokit()` function for lazy loading:
```python
_nk = None

def get_neurokit():
    global _nk, NEUROKIT_AVAILABLE
    if _nk is None:
        try:
            import neurokit2 as nk
            _nk = nk
        except ImportError:
            NEUROKIT_AVAILABLE = False
    return _nk
```

Replaced all `nk.` calls with `nk = get_neurokit()` followed by `nk.method()`.

#### 3. Lazy Import for matplotlib
Same pattern - `get_matplotlib()` function to defer import until needed.

### Performance Results:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| App startup | ~1.6s | ~0.7s | **56% faster** |
| app.py import | ~1.0s | ~0.16s | 84% faster |
| neurokit2 | at startup | on-demand | Deferred |
| matplotlib | at startup | on-demand | Deferred |

### Files Modified:
- `src/music_hrv/gui/app.py` - Added `get_neurokit()`, `get_matplotlib()`, updated all call sites
- `CLAUDE.md` - Updated performance guidelines

### Profiling Results (for reference):
- `pandas` import: 0.36s (unavoidable)
- `streamlit` import: 0.37s (unavoidable)
- `neurokit2` import: 0.88s (now deferred)
- `signal_changepoints(variance)`: 0.75s (cached)
- `signal_changepoints(mean)`: 26s (!!! - never use this)
- Plot rendering with downsampling: 9ms

### Key Learnings:
- Heavy scientific libraries (neurokit2, scipy) are slow to import
- Lazy loading is essential for responsive UIs
- Profile before optimizing - identify actual bottlenecks

---

## Session 2025-11-28: Performance Optimization & Batch Processing

### Version Tag: `v0.3.1`

### Major Improvements:

#### 1. Performance Optimization (CRITICAL)
Profiled the app and found major bottlenecks:

| Bottleneck | Before | After | Fix |
|------------|--------|-------|-----|
| JSON Deserialize | 1496ms | 0ms | Removed (was making things SLOWER) |
| Changepoint Analysis | 1186ms | 0ms (lazy) | Only runs when variability view enabled |
| Plotly Scattergl | 125ms | 13ms | Downsample to 5000 points |

**Key Learnings:**
- Plotly JSON serialization/deserialization is VERY slow - avoid it!
- Always profile before optimizing - my first caching approach made things worse
- Lazy loading expensive operations saves significant time

**Performance Results:**
- First participant load: ~500ms (was ~3s)
- Toggling plot options: Near-instant
- Switching participants: ~200ms (cached data)

#### 2. Caching Functions Added
- `cached_discover_recordings()` - Cache directory scanning
- `cached_load_recording()` - Cache file loading per participant
- `cached_clean_rr_intervals()` - Cache RR cleaning results
- `cached_quality_analysis()` - Cache changepoint detection
- `cached_get_plot_data()` - Cache downsampled plot data

#### 3. Batch Processing
New "Batch Processing" expander in Tab 1:
- **Auto-Generate Music Events**: For all participants in a playlist group
- **Auto-Create Quality Events**: Detect gaps/variability for all participants
- Progress bars and status updates during processing

#### 4. Help Text Added
- Tab 1: "Quick Help - Getting Started" with workflow overview
- Tab 2: "Help - Event Mapping" explaining synonyms and regex
- Tab 3: "Help - Groups & Playlists" explaining study groups
- Tab 5: "Help - HRV Analysis" explaining HRV metrics

#### 5. Smart Status Summary
- Shows issue count at top of participant table
- Only displays warnings when issues exist
- Shows success message when all participants look good

### Files Modified:
- `src/music_hrv/gui/app.py` - Performance optimizations, batch processing, help text

### Testing Results:
- All 13 tests passing
- No linting errors
- Performance verified via profiling script

### Important Notes for Future Development:
- **ALWAYS profile before optimizing** - use timer context managers
- **Avoid Plotly JSON serialization** - it's extremely slow for large figures
- **Downsample large datasets** - 5000 points is visually sufficient
- **Lazy load expensive operations** - only compute when user requests it
- **Cache at the right level** - cache data, not serialized objects

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
