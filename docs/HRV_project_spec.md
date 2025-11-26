# HRV Analysis Pipeline with `neurokit2`  
**Project Specification & Requirements (including GUI & Standalone App)**

---

## 1. Purpose of This Document

This markdown file defines the **goals and requirements** for a Python-based HRV analysis pipeline built on top of **`neurokit2`**.

It is written so that code assistants (e.g. Claude Code, GitHub Copilot, OpenAI Codex, etc.) can use it as a **high-level specification**.  
Every requirement should be interpretable as something that can be implemented as:

- Functions / modules in Python  
- A GUI application for non-programmers  
- A final **standalone desktop tool** for end users

---

## 2. Project Overview

### 2.1 Main Idea

Build a **robust, reproducible HRV analysis tool** that:

1. Reads **RR interval data** from:
   - **HRV Logger** (RR + Events files)
   - **VNS Analyse** (single export file with RR + Notes)
2. Prepares and cleans these RR intervals.
3. Segments the data into **meaningful sections** (e.g. baseline, tasks, recovery) using markers/notes.
4. Computes **HRV metrics** for each section using `neurokit2`.
5. Produces:
   - Per-participant summary files
   - A **single group-level CSV** across all participants and sections
   - Visualisations on both **individual** and **group** level
6. Is usable by non-programmers via a **graphical user interface** (GUI).
7. Can be distributed as a **standalone application** (no Python knowledge required for end users).

### 2.2 Target Users

- Researchers in **music psychology**, psychophysiology, sports science, etc.
- Students and clinicians with **little or no programming experience**.
- Power users who want to customize or extend the pipeline via code.

---

## 3. Data Sources & Structures

### 3.1 HRV Logger (Input Type A)

**Expected files per participant:**

1. **RR file**  
   - Contains raw R–R intervals in ms  
   - Contains timestamps (absolute or relative)  
   - May contain additional info (HR, artefact flags etc.)

2. **Events file**  
   - Contains timestamps of markers  
   - Contains event labels / codes (e.g. “baseline_start”, “task1_start”)

3. (Optional) HRV summary file  
   - Exported HRV indices that can be used for validation (not required for the pipeline).

### 3.2 VNS Analyse (Input Type B)

**Expected file per participant:**

- A **single text/CSV file** that includes:
  - Header section with summary statistics for the whole recording
  - Table with:
    - RR interval values (ms)
    - Time or sample index
    - A **Note** column that may contain markers or comments

### 3.3 Participant Identification

- Participant IDs are typically encoded in file names  
  - Example: `2025-3-10_RR_0405BIPE.csv` → ID: `0405BIPE`
- Requirement:  
  - Implement a **configurable function** for extracting participant IDs from filenames.

---

## 4. Functional Requirements (Core Logic)

### 4.1 Data Ingestion

The tool must:

- **Batch-process multiple participants** in one run.
- Support:
  - A folder containing HRV Logger files (`RR_*.csv` + `Events_*.csv`).
  - A folder containing VNS files (`*.txt` or `*.csv`).
- Use a **configuration file** (YAML/JSON) to define:
  - Input directories
  - File patterns
  - Participant ID extraction rules
  - Section definitions (mapping markers → section names)

### 4.2 Data Preparation & Cleaning

For each participant and recording:

1. **Load raw RR data**
   - Read RR intervals and timestamps into a DataFrame.
   - For HRV Logger: join with Events data where needed.
   - For VNS: parse header, then main RR table.

2. **Convert timestamps**
   - Convert to a unified representation (e.g. seconds from recording start or absolute `datetime`).

3. **Clean RR intervals**
   - Remove or correct:
     - Physiologically implausible RR values (too small/too large).
     - Sudden jumps indicating missing or extra beats.
   - Implement a **parameterised cleaning function**:
     - Thresholds for min/max RR
     - Threshold for sudden change (e.g. percentage difference)
   - Store artefact information:
     - Number of removed/corrected beats
     - Proportion of artefacts per section

4. **Quality control rules**
   - Mark sections as **invalid** if:
     - Duration is below a minimum threshold (e.g. < 2 minutes, configurable).
     - Percent artefacts exceeds a defined threshold.

### 4.3 Section Definition & Segmentation

The tool must:

- Use markers to define **sections**:
  - For HRV Logger: use Events file (timestamp + label).
  - For VNS: use Note column as markers.
- Create a mapping from raw markers to **canonical section names** (configured by user).
  - Example:
    - `"BaselineStart"` → `baseline`
    - `"Music1"` → `music_block_1`
- For each section:
  - Determine start and end time.
  - Extract corresponding RR intervals.
  - Store metadata: start, end, duration, number of beats, artefact percentage.

### 4.4 HRV Computation (Using `neurokit2`)

For each valid section:

- Compute **time-domain HRV metrics**, e.g.:
  - Mean RR (AVNN)
  - SDNN
  - RMSSD
  - pNN50

- Compute **frequency-domain metrics**, e.g.:
  - LF power
  - HF power
  - LF/HF ratio

- Compute **non-linear metrics** (as available and appropriate), e.g.:
  - DFA α1 (if possible and meaningful)
  - Optional: sample entropy, Poincaré plot indices

**Implementation requirements:**

- Encapsulate HRV calculations in a function like:
  - `compute_hrv_metrics(rr_series, sampling_rate, params) -> dict`
- Ensure unit consistency:
  - Define clearly whether RR is in ms or seconds at each stage.
- Allow configurable parameters:
  - Frequency bands
  - Detrending options
  - Window sizes

### 4.5 Output Data Structures

**Per participant output (CSV):**

- One row per **(participant, section)**.
- Columns include:
  - `participant_id`
  - `device_type` (e.g. `HRV_Logger`, `VNS`)
  - `section_name`
  - `section_start_time`
  - `section_end_time`
  - `section_duration_sec`
  - `n_beats`
  - `artifact_ratio`
  - HRV metrics (e.g. `mean_rr`, `sdnn`, `rmssd`, `pnn50`, `lf`, `hf`, `lf_hf`, `dfa_alpha1`, …)

**Group-level output (CSV):**

- Single file combining all participants:
  - Same columns as above.
  - Can be directly imported into R, Python, SPSS, etc.

---

## 5. Visualisation Requirements

### 5.1 Individual-Level Visuals

For each participant:

- Plot **RR time series** or heart rate over full recording with **section boundaries** marked.
- For each section:
  - Bar or point plots of HRV metrics (e.g. RMSSD, DFA α1) per section.
- Optional:
  - Sliding-window plots of HRV metrics over time.

### 5.2 Group-Level Visuals

- Boxplots / violin plots of key HRV metrics by section, e.g.:
  - RMSSD baseline vs. task vs. recovery.
- Optional:
  - Line plots showing individual trajectories (spaghetti plots) over sections.
  - Group means ± confidence intervals.

### 5.3 Export

- Allow saving plots as:
  - PNG
  - PDF
  - Possibly SVG

---

## 6. GUI Requirements

The project must include a **user-friendly graphical interface** suitable for users with **no programming knowledge**.

### 6.1 General GUI Requirements

- Platform: **Streamlit** web-based GUI (browser-based interface).
- Must be **cross-platform** (Windows, macOS, Linux).
- Simple, clear language (no technical jargon).
- Consistent layout using Streamlit's native components.

### 6.2 Main GUI Screens / Views

#### 6.2.1 Home Screen

- Elements:
  - Project title
  - Short description of what the tool does
  - Buttons:
    - “Start New Analysis”
    - “Load Previous Results”
    - “Help / Documentation”

#### 6.2.2 Data Import Screen

- Let the user:
  - Select **input type** (HRV Logger / VNS).
  - Choose **input folder(s)** via file dialog.
  - Optionally preview a list of detected participants/files.
- Show:
  - Count of participants detected
  - Example filenames
- Provide:
  - Button “Next: Sections / Markers”

#### 6.2.3 Section Definition Screen

- Allow the user to:
  - Define mappings from raw event labels/notes to canonical **section names**.
    - Example: Text fields or a small table to map input markers → section labels.
  - Specify:
    - Minimum section duration (seconds)
    - Maximum allowed artefact ratio
- Provide default templates:
  - “Baseline / Task / Recovery” scheme as a starting point.

#### 6.2.4 Analysis Run Screen

- Show:
  - Summary of settings (input folders, number of participants, section definitions).
- Provide:
  - “Run Analysis” button.
  - Progress indicator (progress bar, log area with messages).
- Display:
  - Success / error messages per participant.
  - Basic statistics: how many participants successfully processed, sections discarded etc.

#### 6.2.5 Results & Visualisation Screen

- Allow user to:
  - View **individual participant** results:
    - Dropdown to select participant
    - Table of HRV metrics per section
    - Plots for that participant
  - View **group-level** results:
    - Summary table
    - Boxplots/violin plots by section
- Provide export options:
  - “Export group-level CSV”
  - “Export figures”
  - “Open results folder”

### 6.3 Usability & Accessibility

- Clear labels and tooltips explaining:
  - What RR intervals are
  - What “sections” mean
  - What key HRV metrics roughly represent
- Provide basic error handling:
  - Informative messages when files are missing, wrong format, or parsing fails.
- Avoid overwhelming users with technical parameters; advanced settings can be hidden under an “Advanced” section.

---

## 7. Standalone Application Requirements

The final tool should be **usable without installing Python manually**.

### 7.1 Packaging

- Use a packaging method (e.g. PyInstaller, cx_Freeze, Briefcase, or similar) to create:
  - A Windows executable (`.exe`)
  - Ideally macOS app bundle as well

### 7.2 Standalone Behaviour

- End user workflow:
  1. Download the app (or zip).
  2. Double-click the executable.
  3. GUI opens and is fully functional.
  4. All libraries (`neurokit2`, etc.) are bundled inside.

- The standalone version should:
  - Create a configuration directory (if needed).
  - Log errors to a simple log file (`logs/` directory) for debugging.

### 7.3 Updates

- Ensure that:
  - New versions of the app can be released by rebuilding the package.
  - The codebase remains the single source of truth for the app behaviour.

---

## 8. Scientific Good Practice & Reproducibility

### 8.1 Transparency

- Every major step (import, cleaning, segmentation, HRV computation) must be:
  - Implemented in **separate, well-named functions**.
  - Documented with docstrings explaining:
    - Purpose
    - Parameters
    - Returns
    - Important assumptions (e.g. units, thresholds)

- Maintain markdown documentation, for example:
  - `METHODS_HRV_ANALYSIS.md` – suitable for copy-pasting into a methods section.
  - `DATA_FORMATS.md` – describing input/output formats.

### 8.2 Reproducible Environments

- Provide:
  - `requirements.txt` or `environment.yml` describing Python environment.
- Pin versions of key libraries (including `neurokit2`) to ensure consistent results over time.

### 8.3 Code Quality

- Follow **PEP8** style where reasonable.
- Use clear, descriptive names (`clean_rr_intervals`, `segment_by_events`, `compute_neurokit_hrv`).
- Avoid magic numbers in code; use configuration variables.

### 8.4 Version Control (Git / GitHub)

- Use **Git** for version control.
- Host project on **GitHub** (or similar).
- Recommended repo structure:

  ```text
  hrv-project/
  ├─ src/
  │  ├─ hrv_io.py
  │  ├─ hrv_cleaning.py
  │  ├─ hrv_segmentation.py
  │  ├─ hrv_metrics.py
  │  ├─ hrv_visualization.py
  │  └─ gui_app.py
  ├─ config/
  │  └─ example_config.yml
  ├─ examples/
  │  └─ example_data/  (anonymised or synthetic)
  ├─ docs/
  │  ├─ METHODS_HRV_ANALYSIS.md
  │  └─ DATA_FORMATS.md
  ├─ tests/
  ├─ README.md
  ├─ requirements.txt
  ├─ .gitignore
  └─ LICENSE
  ```

- Commit often with **informative messages** (e.g. “Implement RR cleaning and artefact logging”).

### 8.5 Testing

- Add basic tests (e.g. with `pytest`) for:
  - Parsing of HRV Logger and VNS files.
  - RR cleaning behaviour on synthetic data.
  - Segmentation correctness for simple artificial events.
  - HRV metric calculation output shape and basic sanity checks.

### 8.6 Data Privacy

- **Never commit raw participant data** to the public repo.
- Use `.gitignore` to exclude:
  - `data/` or similar directories.
  - Any files containing non-anonymised participant information.
- Participant ID ↔ real identity mapping must remain outside of the codebase.

---

## 9. Implementation Roadmap

This section breaks the project into steps that can be implemented one by one.

1. **Set up project skeleton and environment**
   - Create repo structure.
   - Add `requirements.txt`.
   - Install `neurokit2`, `pandas`, `numpy`, `matplotlib`, GUI framework.

2. **Implement data loading**
   - Functions to parse HRV Logger RR file.
   - Functions to parse HRV Logger Events file.
   - Functions to parse VNS Analyse file.
   - Unit tests using small example files.

3. **Implement RR cleaning module**
   - Cleaning function with configurable thresholds.
   - Artefact statistics per section.
   - Simple tests on synthetic RR sequences.

4. **Implement section segmentation**
   - Using Events (HRV Logger) and Notes (VNS).
   - Configurable mapping from event labels to section names.
   - Handling of too-short sections.
   - Tests for segmentation logic.

5. **Implement HRV metric computation**
   - Time-domain metrics.
   - Frequency-domain metrics.
   - Non-linear metrics (e.g. DFA α1 if feasible).
   - Wrapper function that outputs a dict / DataFrame row.

6. **Implement data export**
   - Per-participant CSV.
   - Group-level CSV combining all participants.

7. **Implement visualisation functions**
   - Individual time-series and per-section plots.
   - Group-level distributions.
   - Save plots to files.

8. **Build GUI**
   - Implement screens described in section 6.
   - Connect GUI actions to backend functions.

9. **Create standalone builds**
   - Configure packaging tool (e.g. PyInstaller).
   - Create Windows (and optionally macOS) standalone executables.
   - Test on clean systems.

10. **Finalize documentation and examples**
    - Update README and docs.
    - Provide example config and sample data.
    - Ensure that another person can reproduce the full workflow.

---

## 10. Summary

This document defines a **full HRV analysis pipeline** based on `neurokit2`, starting from **HRV Logger** and **VNS Analyse** RR exports, and ending with:

- Cleaned, segmented RR data
- HRV metrics per section and participant
- Group-level CSV for statistics
- Individual and group visualisations
- A user-friendly GUI
- A standalone executable for non-programmers
- use UV and rust!!!!!!

The specification emphasises **good scientific practice**, **clear coding**, and **reproducibility**, and is structured so that humans and code assistants can implement the project step by step.
