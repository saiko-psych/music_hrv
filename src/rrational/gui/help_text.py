"""Help text and documentation for the RRational HRV GUI.

This module contains markdown text for help sections displayed in the UI.

References:
- Quigley et al. (2024) Publication guidelines for HR/HRV studies in psychophysiology
- Khandoker et al. (2020) HRV Analysis: How Much Artifact Can We Remove?
- German Journal Sports Medicine (2024) HRV Methods and Analysis
"""

# =============================================================================
# GETTING STARTED
# =============================================================================

GETTING_STARTED_HELP = """
## Getting Started with RRational

### What is RRational?

RRational is a free, open-source HRV analysis toolkit - like Kubios, but free.
It follows scientific best practices (2024 Quigley guidelines) and provides
a complete workflow from raw data to publication-ready metrics.

---

### Typical Workflow

```
1. Create/Open Project  →  2. Import Data  →  3. Review & Clean  →  4. Analyze
```

#### Step 1: Project Setup
- **New Study**: Create a project with a folder structure for your data
- **Existing Study**: Open an existing project folder
- **Quick Test**: Use "Continue Without Project" for temporary work

#### Step 2: Import Data
- Place HRV files in `YourProject/data/raw/hrv_logger/` or `data/raw/vns/`
- Click "Load Selected Sources" in the Data tab
- Review participant overview table

#### Step 3: Review & Clean
- Use **Participants** tab to inspect each recording
- Enable artifact detection to identify problems
- Define exclusion zones for bad segments
- Save events and exclusions

#### Step 4: Analyze
- Go to **Analysis** tab
- Select participant and section
- Enable artifact correction if needed
- View metrics and export results

---

### Quick Tips

- **Keyboard**: Use Previous/Next buttons or dropdown to navigate participants
- **Saving**: Click "Save" in sidebar to persist changes
- **Help**: Look for ℹ️ icons and expandable help sections throughout the app
- **Export**: Use "Ready for Analysis" export to save inspected data with audit trail
"""

PROJECT_HELP = """
## Project Management

### Why Projects?

Projects keep your HRV studies organized. Each project is a self-contained folder
with your data, configuration, and results. This makes it easy to:

- **Share** studies with collaborators
- **Archive** completed research
- **Reproduce** your analysis workflow

---

### Project Structure

When you create a project, RRational sets up this folder structure:

```
MyStudy/
├── project.rrational          # Project metadata (name, author, notes)
├── data/
│   ├── raw/                   # Your original HRV files
│   │   ├── hrv_logger/        # HRV Logger CSV files go here
│   │   └── vns/               # VNS Analyse TXT files go here
│   └── processed/             # Exported .rrational files, saved events
├── config/                    # Project-specific settings
│   ├── groups.yml             # Study groups
│   ├── events.yml             # Event definitions
│   ├── sections.yml           # Section definitions
│   └── ...                    # Other configuration
└── analysis/                  # Future: analysis results
```

---

### Creating a New Project

1. Click **"Create New Project"** on the welcome screen
2. **Choose location**: Browse to where you want the project folder
3. **Enter details**: Name, description, author (optional)
4. **Select data sources**: HRV Logger, VNS Analyse, or both
5. **Review and create**: Check the summary and click "Create Project"

The project folder will be created with all necessary subfolders.
Copy your HRV data files into `data/raw/hrv_logger/` or `data/raw/vns/`.

---

### Opening an Existing Project

1. Click **"Open Existing Project"** on the welcome screen
2. Browse to a folder containing `project.rrational`
3. The project will load with all your saved settings

**Recent Projects**: Previously opened projects appear in the list for quick access.

---

### Project vs Temporary Workspace

| Aspect | Project | Temporary Workspace |
|--------|---------|---------------------|
| **Settings saved** | In project folder | In `~/.rrational/` |
| **Portable** | Yes, share the folder | No, tied to your computer |
| **Auto-load** | Remembers last project | No persistence |
| **Best for** | Real research | Quick testing |

---

### Switching Projects

- Click **"Switch Project"** in the sidebar to return to the welcome screen
- Your last used project is remembered and will auto-load next time

---

### Migrating Existing Configuration

If you have configuration from previous RRational sessions (in `~/.rrational/`),
the Create Project wizard offers to import it into your new project.
"""

WELCOME_SCREEN_HELP = """
## Welcome Screen

The welcome screen appears when you launch RRational.

### Options

| Option | Description |
|--------|-------------|
| **Recent Projects** | Click any project to open it immediately |
| **Open Existing Project** | Browse for a project folder (contains `project.rrational`) |
| **Create New Project** | Start a new study with organized folder structure |
| **Continue Without Project** | Use temporary workspace (settings saved globally) |

### Auto-Load

RRational remembers your last used project and offers to load it automatically.
Click "Continue" to load, or "Choose Different" to see all options.

### First Time Users

1. Click **"Create New Project"** to set up your first study
2. Follow the wizard to name your project and select data sources
3. Copy your HRV data files into the `data/raw/` subfolders
4. Start analyzing!
"""

DATA_SOURCES_HELP = """
## Supported Data Sources

RRational supports data from two popular HRV recording apps:

| App | Platform | Format | Link |
|-----|----------|--------|------|
| **HRV Logger** | iOS/Android | CSV | [hrv.tools](https://www.hrv.tools/hrv-logger-faq.html) |
| **VNS Analyse** | iOS | TXT | [App Store](https://apps.apple.com/de/app/vns-analyse/id990667927) |

### HRV Logger
A cross-platform mobile app for recording RR intervals via Bluetooth heart rate monitors.
Exports paired CSV files: one with RR intervals and timestamps, one with event markers.

### VNS Analyse
An iOS app by BioSign GmbH for heart rate variability analysis.
Exports TXT files with both raw and corrected RR interval sections.
"""

DATA_CORRECTION_WORKFLOW = """
## Data Correction Workflow

### Overview

RRational processes RR interval data through three stages:
**Import → Display → Analysis**. Each stage handles data differently depending
on the source (HRV Logger vs VNS Analyse).

---

### 1. Import Stage

#### HRV Logger (CSV files)
- **What's loaded**: All RR intervals with their original timestamps
- **Duplicates**: Automatically detected and removed (shown in summary)
- **No filtering**: All intervals preserved, even physiologically impossible ones
- **Timestamps**: Real measurement times from the device

#### VNS Analyse (TXT files)
- **Section selection**: Choose between Raw or Corrected RR values (Import Settings)
- **No filtering**: All intervals loaded regardless of value
- **Timestamps**: **Synthesized** from cumulative RR intervals (not real times)
- **Events**: "Notiz:" annotations converted to event markers

> **Important**: VNS timestamps are calculated as `base_time + sum(RR intervals)`.
> This means the "time" shown is physiological duration, not wall-clock time.

---

### 2. Display Stage

#### Data Tab (Participant Overview)
- Shows summary statistics for each participant
- **Quality badge**: Based on artifact ratio and duplicate count
- **No cleaning applied** - shows raw data characteristics

#### Participant View (Plot)

**HRV Logger Data:**
- All intervals shown in **blue**
- Gap detection enabled (finds recording interruptions >15s)
- Intervals already filtered by Import Settings thresholds

**VNS Data:**
- All intervals shown, with **flagged intervals in RED**
- Flagged = outside Min/Max RR thresholds (Import Settings)
- Gap detection **disabled** (timestamps are synthetic)
- Warning shows count and total time of flagged intervals

#### Plot Options
- **Show artifacts (NeuroKit2)**: Detect ectopic/missed/extra beats using Kubios algorithm
- **Show variability segments**: Detect high-variance regions (potential movement artifacts)
- **Show time gaps**: Highlight recording interruptions (HRV Logger only)
- **Show music sections**: Display defined section boundaries
- **Show music events**: Display individual event markers

---

### 3. Analysis Stage

When you click "Analyze HRV", the following steps occur:

1. **Section Extraction**
   - Finds start/end events for the selected section
   - Extracts only RR intervals within that time range

2. **Exclusion Zone Filtering** (NEW)
   - Removes all RR intervals falling within user-defined exclusion zones
   - Exclusion zones are set in the Participants tab under "Add Exclusions" mode
   - See "Exclusion Zones" section below for details

3. **Threshold Filtering** (Import Settings)
   - Removes intervals outside Min/Max RR range (default: 200-2000ms)
   - Sudden change filter (default: disabled at 100%)

4. **Artifact Correction** (Optional)
   - Enable "Apply artifact correction" checkbox
   - Uses NeuroKit2's Kubios algorithm to detect and correct:
     - **Ectopic beats**: Premature/delayed contractions
     - **Missed beats**: Undetected R-peaks
     - **Extra beats**: False positive detections
     - **Long/short intervals**: Physiologically implausible

5. **HRV Metrics Computation**
   - Time domain: RMSSD, SDNN, pNN50, etc.
   - Frequency domain: HF power, LF power, LF/HF ratio

---

### Recommended Workflow

#### For VNS Data:
1. **Import**: Use "Raw" values (unless you trust VNS's correction)
2. **Review**: Check participant plot - RED intervals are flagged
3. **Adjust thresholds**: If too many flagged, adjust Min/Max RR in Import Settings
4. **Enable artifact detection**: Check "Show artifacts (NeuroKit2)" to see Kubios results
5. **Analyze**: Enable "Apply artifact correction" for HRV analysis

#### For HRV Logger Data:
1. **Import**: Data loaded as-is
2. **Review**: Check for gaps (gray shaded regions) - may indicate Bluetooth issues
3. **Enable artifact detection**: Check "Show artifacts (NeuroKit2)"
4. **Define sections**: Avoid sections that span gaps
5. **Analyze**: Enable "Apply artifact correction" if artifacts detected

---

### Determining Segment Validity

A segment may be **invalid** or **questionable** if:

1. **High artifact count** (>5% flagged intervals)
   - Check with "Show artifacts (NeuroKit2)" option
   - Consider excluding segment or using artifact correction

2. **Contains gaps** (HRV Logger only)
   - Gray shaded regions indicate recording interruptions
   - Define sections to exclude gap periods

3. **High variability segments** (red/orange shading)
   - May indicate movement or electrode issues
   - Consider excluding or noting in analysis

4. **Too few beats** (<100 beats for time domain, <300 for frequency)
   - NeuroKit2 requires minimum data for reliable metrics

#### Exclusion Strategy:
- **Don't select** problematic sections for analysis
- **Define custom sections** that avoid problem periods
- **Use artifact correction** to salvage borderline segments

---

### Scientific Best Practices (2024 Guidelines)

Based on current research and guidelines:

#### Artifact Thresholds by Metric Type

| Metric Type | Max Artifact % | Notes |
|-------------|---------------|-------|
| **RMSSD, SDNN** | ~36% | Most robust to artifacts |
| **pNN50** | ~4% | Sensitive to beat timing shifts |
| **HF, LF, LF/HF** | ~2% | Most sensitive - use with caution |

> **Recommendation**: For frequency domain analysis (HF, LF), ensure artifact
> rate is below 2%. For time domain only (RMSSD, SDNN), up to 5-10% is acceptable.

#### Minimum Data Requirements

| Analysis Type | Minimum Beats | Minimum Duration |
|---------------|--------------|------------------|
| Time domain | ~100 beats | ~2 minutes |
| Frequency domain | ~300 beats | ~5 minutes |
| Ultra-short | 60 beats | 1 minute (RMSSD only) |

#### Recommended Workflow (Scientific Standard)

1. **Visual inspection** of RR plot before analysis
2. **Report artifact rates** in any publication
3. **Use artifact correction** for rates 2-10%
4. **Exclude segments** with >10% artifacts
5. **Prefer time domain** metrics if artifact rates uncertain

#### References
- [Quigley et al. (2024) Publication guidelines](https://pubmed.ncbi.nlm.nih.gov/38873876/)
- [Khandoker et al. (2020) Artifact tolerance study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7538246/)
- [German J Sports Med (2024) HRV methods](https://www.germanjournalsportsmedicine.com/archive/archive-2024/issue-3/)

---

### Current Implementation Status

This app follows scientific best practices:

**Implemented:**
- NeuroKit2 Kubios algorithm for artifact detection/correction
- Threshold-based detection (Malik method) for long recordings
- Visual display of flagged/artifact intervals
- Signal Inspection mode for beat-level review
- Correction preview (green dotted line)
- Artifact correction optional at analysis time
- Time and frequency domain HRV metrics

**User Responsibility:**
- Visual inspection of plots before analysis
- Checking artifact rates displayed in UI
- Deciding whether to correct or exclude segments
- Ensuring sufficient beat count for analysis type

**Reported in Results:**
- Artifact counts (when correction enabled)
- Number of beats used
- Section boundaries
"""

CLEANING_THRESHOLDS_HELP = """
### Cleaning Thresholds

These thresholds filter RR intervals before analysis:

| Threshold | Default | Purpose |
|-----------|---------|---------|
| **Min RR** | 200ms | Removes impossibly fast beats (>300 BPM) |
| **Max RR** | 2000ms | Removes impossibly slow beats (<30 BPM) |
| **Sudden Change** | 100% (disabled) | Would flag beats that change >X% from previous |

> **Note**: Sudden change is disabled by default because normal heart rate
> variability can exceed 20% between beats. Use NeuroKit2 artifact correction
> instead for proper artifact detection.

For **VNS data**: Thresholds flag intervals (shown in RED) but keep them for
correct timestamp display. Flagged intervals are excluded from HRV analysis.

For **HRV Logger data**: Thresholds remove intervals entirely (timestamps
are independent of RR values, so removal doesn't affect timing).
"""

ARTIFACT_CORRECTION_HELP = """
### Signal Inspection and Artifact Correction

RRational provides multiple artifact detection methods:

#### Detection Methods

| Method | Algorithm | Best For |
|--------|-----------|----------|
| **Threshold (Malik)** | Simple ratio check - flags beats differing >X% from previous | Quick screening, very long recordings (>1 hour) |
| **Lipponen 2019** | State-of-the-art beat classification algorithm | Standard recordings (<10 min), most accurate detection |
| **Lipponen 2019 (segmented)** | Same algorithm, processes in ~5-min chunks | Long recordings (>10 min) - better sensitivity across time |

> **Note**: This is the same algorithm that NeuroKit2 calls "Kubios". We use the scientific name
> (Lipponen & Tarvainen, 2019) for clarity and proper attribution.

#### Which Method Should I Use?

| Recording Length | Recommended Method | Why |
|-----------------|-------------------|-----|
| **< 10 min** | Lipponen 2019 | Full context for beat classification |
| **10-60 min** | Lipponen 2019 (segmented) | Better sensitivity in later segments |
| **> 60 min** | Threshold (Malik) or Lipponen 2019 (segmented) | Fast screening or segment-by-segment analysis |
| **Quick screening** | Threshold (Malik) | Fast, adjustable sensitivity via slider |

#### Segmented Mode Explained

For long recordings, artifact characteristics may change over time (fatigue, movement, electrode drift).
Segmented mode divides the recording into ~5-minute chunks (300 beats) and runs detection independently
on each segment. This provides:

- **Better sensitivity**: Local thresholds adapt to each segment
- **Per-segment statistics**: See artifact % for each 5-min chunk (expand "Segment Artifact Details")
- **Identify problem periods**: Segments >10% artifacts should be excluded

#### Artifact Types Detected (Lipponen/Kubios methods):

| Type | Description | How Corrected |
|------|-------------|---------------|
| **Ectopic** | Premature/delayed beat (PVC, PAC) | Interpolated from neighbors |
| **Missed** | Undetected R-peak | Interval split |
| **Extra** | False positive detection | Beat removed |
| **Long/Short** | Physiologically implausible | Interpolated |

#### Threshold Method:
- Simple: flags beats where `RR[i] / RR[i-1]` differs by > threshold %
- Default 20% - adjust slider to tune sensitivity
- No beat classification, just ratio check

#### Signal Inspection Mode:
1. Select **"Signal Inspection"** mode in participant view
2. Resolution auto-increases for beat-level inspection
3. Orange X markers show detected artifacts on the plot
4. Purple diamond markers show manually marked artifacts
5. Gray circle-x markers show excluded (demarked) artifacts
6. Green dotted line shows corrected NN intervals (preview)
7. Quality assessment shows artifact rate and recommendations
8. For segmented methods, expand **"Segment Artifact Details"** to see per-segment statistics

#### Artifact Detection Workflow:
1. **Show artifacts** checkbox displays saved/loaded artifacts (no auto-detection)
2. **Detect New Artifacts** expander - configure method and click "Run Detection"
3. **Clear** button removes algorithm-detected artifacts

#### Manual Artifact Marking (Signal Inspection mode):
- **Click on algorithm artifact** → Exclude it (gray circle-x)
- **Click on excluded artifact** → Re-enable it (orange X)
- **Click on normal beat** → Mark as manual artifact (purple diamond)
- **Click on manual artifact** → Remove marking
- Changes are saved automatically and persist across sessions

#### Quality Guidelines (Quigley et al. 2024):

| Artifact Rate | Quality | Recommended Action |
|---------------|---------|-------------------|
| < 2% | Excellent | All metrics valid |
| 2-5% | Good | Use with correction |
| 5-10% | Acceptable | Prefer time-domain metrics |
| > 10% | Poor | Exclude segment or use only RMSSD/SDNN |

#### Recommended Workflow:
1. **Load participant** - saved artifacts auto-load if available
2. **Visual inspection** in Signal Inspection mode
3. **Run detection** if no saved artifacts (choose method in expander)
4. **Manual review** - click to mark/unmark individual artifacts
5. **Save corrections** using sidebar "Save Artifact Corrections" button
6. **Define exclusions** for persistently bad regions (switch to Add Exclusions mode)
7. **Apply correction** during analysis (Analysis tab checkbox)

#### Reference:
Lipponen, J. A., & Tarvainen, M. P. (2019). A robust algorithm for heart rate variability
time series artefact correction using novel beat classification. *J Med Eng Technol*, 43(3), 173-181.
https://doi.org/10.1088/1361-6579/ab3c96
"""

EXCLUSION_ZONES_HELP = """
### Exclusion Zones

Exclusion zones allow you to **exclude specific time periods** from HRV analysis.
This is useful for removing known artifacts, bathroom breaks, or other disruptions.

#### Creating Exclusion Zones (Participants Tab):

1. Select a participant and switch to **"Add Exclusions"** mode
2. **Click two points** on the plot to define start and end
3. After clicking both points, you'll see:
   - Red markers showing START and END boundaries
   - Shaded red region highlighting the excluded area
   - Confirmation form to add reason and confirm
4. Fill in the optional **Reason** (e.g., "Bathroom break")
5. Check **"Exclude from duration"** if this time should not count toward timing validation
6. Click **"Confirm"** to add the exclusion zone

#### Viewing Exclusion Zones:

- Exclusion zones appear as **red shaded rectangles** on the plot
- **Vertical label** shows the reason (truncated) at the zone start
- "[excl]" suffix indicates zone is excluded from duration calculations
- Toggle visibility with **"Show exclusions"** checkbox

#### Effect on Analysis:

When you run HRV analysis:
1. All RR intervals **within exclusion zones are automatically removed**
2. Analysis reports how many intervals were excluded (e.g., "Excluded 45 intervals from 2 zones")
3. Remaining intervals are then cleaned and analyzed
4. Timing validation excludes marked time (if "Exclude from duration" enabled)

#### Editing Exclusion Zones:

- Click the **Edit** button next to any zone to modify it
- Edit start/end times in HH:MM:SS format
- Change the reason or "Exclude from duration" setting
- Click **"Save"** to apply changes

#### Managing Exclusion Zones:

- View existing zones in the **"Current Exclusion Zones"** list
- Edit zones with the Edit button
- Delete zones with the X button
- Click **"Save"** to persist zones to disk
- Zones are restored when you reload the participant

#### Manual Entry:

If you know exact times, use **"Add manually"** instead:
1. Enter Start time (HH:MM:SS)
2. Enter End time (HH:MM:SS)
3. Add optional reason
4. Click "Add Zone"

#### Best Practices:

**Do:**
- Exclude known disruptions (bathroom, phone, talking)
- Exclude movement artifacts visible in the plot
- Add reasons for documentation and reproducibility
- Use "Exclude from duration" for timing validation accuracy

**Don't:**
- Exclude data just because HRV values look unexpected
- Exclude too much data (reduces statistical power)
"""

VNS_DATA_HELP = """
### VNS Analyse Data Specifics

VNS Analyse exports RR intervals differently from HRV Logger:

#### Key Differences:
| Aspect | HRV Logger | VNS Analyse |
|--------|------------|-------------|
| **Timestamps** | Real measurement times | Synthesized from RR sum |
| **RR Format** | Milliseconds | Seconds (converted to ms) |
| **Sections** | Single data stream | Raw + Corrected sections |
| **Events** | Separate Events.csv | "Notiz:" annotations in file |

#### Timestamp Implications:
- VNS timestamps = `base_time + cumulative_RR`
- If intervals are removed, timestamps shift
- That's why VNS shows flagged intervals instead of removing them
- Gap detection is disabled (gaps would be artifacts of filtering)

#### Raw vs Corrected:
- **Raw**: Original RR intervals as measured
- **Corrected**: VNS software's artifact correction applied
- Select in Import Settings → VNS Analyse Settings
- Recommendation: Start with Raw, apply NeuroKit2 correction if needed

#### Why Flagging Instead of Filtering:
For VNS data, removing intervals would compress the timeline incorrectly.
Instead:
1. All intervals shown with correct (synthesized) timestamps
2. Problematic intervals flagged in RED
3. Flagged intervals excluded only at analysis time
4. Total recording duration remains accurate
"""

SECTIONS_HELP = """
### Sections and Events

#### How Sections Work:
1. **Events** mark time points (e.g., "music_start", "music_end")
2. **Sections** define time ranges between events
3. **Analysis** processes RR intervals within section boundaries

#### Defining Sections (Sections Tab):
1. Select **Start Event** (e.g., "measurement_start")
2. Select **End Event** (e.g., "measurement_end")
3. Give section a **Name** and optional **Label**
4. Repeat for each segment of interest

#### Event Mapping (Events Tab):
- Define canonical event names
- Add synonyms for fuzzy matching
- Example: "music_start" matches "Musik Start", "music starts", etc.

#### Analyzing Sections:
1. Go to **Analysis** tab
2. Select participant
3. Select one or more sections
4. Click **Analyze HRV**
5. Results shown per section + combined if multiple selected

#### Section Validity Tips:
- Ensure start event occurs before end event
- Check that events are correctly detected (Events tab preview)
- Avoid sections that span recording gaps
- Minimum ~100 beats for reliable time-domain metrics
- Minimum ~300 beats for frequency-domain metrics
"""

ANALYSIS_HELP = """
### What is HRV Analysis?

Heart Rate Variability (HRV) analysis quantifies the variation in time between heartbeats.
Higher HRV generally indicates better cardiovascular health and autonomic function.

### Key Metrics

**Time Domain:**
- **RMSSD**: Root Mean Square of Successive Differences - sensitive to parasympathetic activity
- **SDNN**: Standard Deviation of NN intervals - overall HRV
- **pNN50**: Percentage of successive differences > 50ms

**Frequency Domain:**
- **HF (High Frequency)**: 0.15-0.4 Hz - parasympathetic activity
- **LF (Low Frequency)**: 0.04-0.15 Hz - mixed sympathetic/parasympathetic
- **LF/HF Ratio**: Sympathetic/parasympathetic balance

---

### How Time Gaps Are Handled

**Important**: HRV analysis uses the **sum of RR intervals**, not wall-clock time:

| Aspect | How It Works |
|--------|--------------|
| **Section extraction** | Selects RR intervals by timestamp between events |
| **Gap handling** | Gaps (missing data) are simply not present |
| **Duration basis** | Sum of RR intervals = physiological time |
| **Beat sequence** | All intervals treated as consecutive beats |

**Example**: A 5-minute section with a 30-second gap will analyze ~4.5 minutes of actual RR data.

**What this means**:
- Check the **beat count** to verify sufficient data (expect ~350 beats for 5 min at 70 BPM)
- Significantly fewer beats indicates data loss or gaps
- Missing data is NOT interpolated - this is scientifically correct

---

### Scientific Best Practices (2024 Guidelines)

Based on current research and guidelines:

#### Artifact Thresholds by Metric Type

| Metric Type | Max Artifact % | Notes |
|-------------|---------------|-------|
| **RMSSD, SDNN** | ~36% | Most robust to artifacts |
| **pNN50** | ~4% | Sensitive to beat timing shifts |
| **HF, LF, LF/HF** | ~2% | Most sensitive - use with caution |

> **Recommendation**: For frequency domain analysis (HF, LF), ensure artifact
> rate is below 2%. For time domain only (RMSSD, SDNN), up to 5-10% is acceptable.

#### Minimum Data Requirements

| Analysis Type | Minimum Beats | Minimum Duration |
|---------------|--------------|------------------|
| Time domain | ~100 beats | ~2 minutes |
| Frequency domain | ~300 beats | ~5 minutes |
| Ultra-short | 60 beats | 1 minute (RMSSD only) |

#### Recommended Workflow (Scientific Standard)

1. **Visual inspection** of RR plot before analysis
2. **Report artifact rates** in any publication
3. **Use artifact correction** for rates 2-10%
4. **Exclude segments** with >10% artifacts
5. **Prefer time domain** metrics if artifact rates uncertain

#### References
- [Quigley et al. (2024) Publication guidelines](https://pubmed.ncbi.nlm.nih.gov/38873876/)
- [Khandoker et al. (2020) Artifact tolerance study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7538246/)
- [German J Sports Med (2024) HRV methods](https://www.germanjournalsportsmedicine.com/archive/archive-2024/issue-3/)
"""
