# manual_HRV_logger.md  
**Practical Guide for Using HRV Logger Data in the HRV Neurokit2 Pipeline**

---

## 1. What Is HRV Logger?

Heart Rate Variability (HRV) Logger is a mobile app (iOS and Android) designed to:

- Record heart rate and **RR intervals** using a Bluetooth Low Energy (BLE) chest strap  
- Compute and plot **time-domain**, **frequency-domain**, and **non-linear** HRV features in real time  
- Export HRV data for offline analysis in tools like Python, R, spreadsheets, etc.

Typical use cases:

- Morning / resting HRV measurements  
- HRV during exercise (e.g. using DFA α1 to approximate aerobic threshold)  
- Self-experiments and research studies

The app requires a **BLE chest strap**, with **Polar H7/H10** commonly recommended.  

Main feature groups (relevant for this project):

- HR & RR: continuous heart rate and RR intervals
- HRV features: AVNN, SDNN, rMSSD, pNN50, LF, HF, LF/HF, DFA α1, etc.
- Events / experience sampling: timestamped markers you can set during recording
- Activity & location: optional context information (steps, accelerometer, GPS)

---

## 2. Files Exported by HRV Logger

When you export a recording, HRV Logger typically provides **four files** (exact naming may differ):

1. **Heart Rate file**
2. **RR Intervals file**
3. **Features file**
4. **Events file**

### 2.1 Heart Rate File

- Contains heart rate (beats per minute) over time.
- Values are usually averages over a short time window (e.g. 15–30 seconds).
- Useful for:
  - Quick visualisation of intensity over time
  - Checking overall recording quality and trends
- In this project, HRV will mostly be calculated from **RR intervals** rather than this averaged HR stream.

### 2.2 RR Intervals File

This is the **most important file** for our pipeline.

- Contains all RR intervals in **milliseconds**, in the order they were recorded.
- Includes timestamps (absolute clock time or relative time from recording start).
- If **artifact correction** is enabled in the app, only the RR intervals marked as “clean” are exported.
  - Ectopic beats and obvious artifacts may be removed or corrected at the app level.

In our Python pipeline, this file is used as the **primary source** to:

- Rebuild the RR time series  
- Align with events  
- Compute HRV features using `neurokit2`  

### 2.3 Features File

- Contains pre-computed HRV indices over a configurable **time window**.
- Typical window lengths: **30 seconds, 1 minute, 2 minutes, 5 minutes**.
- For each window, the file may include:
  - Time (center or start of the window)
  - Time-domain features (AVNN, SDNN, rMSSD, pNN50, etc.)
  - Frequency-domain features (LF, HF, LF/HF)
  - Non-linear features (DFA α1, etc.)

Usage in this project:

- As a **reference** to validate our own `neurokit2` HRV calculations on a subset of data.
- For quick visual inspection of HRV trends (especially DFA α1 during exercise).

However, the main HRV calculations in the project should be done from the **RR file**, so that:

- HRV metrics are computed consistently across HRV Logger and VNS data.
- All pre-processing decisions (artifact handling, segmentation) are under our control.

### 2.4 Events File

- Contains a list of **annotated events**, each with:
  - Timestamp
  - Event label or description
- Events are created via **experience sampling / markers** during the recording.
  - Example: pressing a button when a music block starts or ends.

In our pipeline, this file is used to:

- Define **sections** such as:
  - `baseline_rest`
  - `music_block_1`
  - `music_block_2`
  - `recovery`
- Segment the RR series into these sections based on event timestamps.
- Align sections across participants for group-level analysis.

---

## 3. Recommended HRV Logger Settings for Data Collection

To ensure that HRV Logger recordings work well with our Python analysis, it is useful to standardise a few settings and procedures.

### 3.1 Hardware

- Use a reliable **BLE chest strap** (e.g. Polar H7/H10).
- Make sure:
  - The strap is moistened and placed correctly.
  - Bluetooth connection is stable.
  - Battery is not low.

### 3.2 Feature Computation Window

Within HRV Logger, you can choose the **time window** over which features are computed (for the Features file and live view):

- 30 seconds  
- 1 minute  
- 2 minutes  
- 5 minutes  

Recommendations:

- For **exercise intensity / DFA α1** work:
  - Use **2-minute windows** for more stable DFA α1 estimates.
- For **resting HRV**:
  - 1–5 minute windows are typical; choose a value that matches your study design.
- In the Python pipeline:
  - You can use similar or different windows, but you should document and keep them consistent across participants.

### 3.3 Artifact Correction

HRV Logger provides RR-interval correction options:

- Artifact correction can be **enabled** or **disabled**.
- For exercise / noisy signals, the app offers a more aggressive **“workout”** correction mode.
- When correction is enabled:
  - RR artifacts and ectopic beats are removed or corrected before HRV features are computed.
  - The RR export file will include only the “clean” beats after correction.

Recommendations:

- Decide **once** for your study whether to:
  - Use the app’s artifact correction and treat the exported RR as already cleaned.
  - Or disable correction and perform all artifact handling yourself in Python.
- For **aerobic threshold / DFA α1** tests:
  - Using the “workout” artefact correction and a 2-minute window is a common suggestion so that high-noise windows can be excluded later.
- In the Python pipeline, it is still helpful to:
  - Inspect the RR series for remaining artifacts.
  - Apply additional cleaning if needed.
  - Log the number and percentage of removed beats per section.

### 3.4 Using Events as Experimental Markers

Good practice when collecting data:

- Define in advance which events you need:
  - `BASELINE_START`, `BASELINE_END`
  - `MUSIC1_START`, `MUSIC1_END`
  - `MUSIC2_START`, `MUSIC2_END`
  - `RECOVERY_START`, `RECOVERY_END`
- Use these markers consistently for all participants.
- During analysis, the Python pipeline will:
  - Read these events from the Events file.
  - Map them to canonical section names in a configuration file.
  - Extract RR intervals between each `_START` and `_END` event pair.

---

## 4. Using HRV Logger for Aerobic Threshold Estimation (DFA α1)

One popular use of HRV Logger is to estimate the **aerobic threshold** during exercise using **DFA α1**:

- DFA α1 is a non-linear HRV index based on detrended fluctuation analysis.
- During a ramp-like exercise or steady effort:
  - Higher DFA α1 values are associated with lower intensity / more correlated RR dynamics.
  - As intensity increases, DFA α1 tends to drop.
- A commonly used practical rule is:
  - The **aerobic threshold** occurs around **DFA α1 ≈ 0.75**.
  - This is an approximation and should not be treated as an exact universal threshold.

Practical guidance for recording:

1. Set artifact correction to a mode suitable for exercise (e.g., “workout”).
2. Set the feature computation window to **2 minutes**.
3. Use a **Polar chest strap** or other high-quality BLE belt.
4. Perform a workout with:
   - Either a steady easy pace,
   - Or a gradual ramp in intensity.
   - Avoid frequent sudden intensity changes; DFA α1 needs some stability within each window.
5. After recording:
   - Inspect DFA α1 values across time.
   - Identify when they cross roughly 0.75 to approximate the aerobic threshold.

Important notes:

- The DFA α1 method is still an active research topic.
- There is evidence that a **single universal threshold (0.75)** does not work perfectly for every individual.
- For your project, DFA α1 values can be exported and analysed in context, but avoid over-claiming precision.

---

## 5. How HRV Logger Exports Map to the Python Pipeline

Here is how each HRV Logger export file will be used in the planned `neurokit2` pipeline.

### 5.1 RR File → Core Input

- Read RR intervals (ms) and timestamps into a pandas DataFrame.
- Optionally re-apply or refine artifact correction (e.g., removing implausible beats).
- Use this series to compute all HRV features with `neurokit2` for each section.

### 5.2 Events File → Section Segmentation

- Read event timestamps and labels.
- Define a mapping (in a YAML/JSON config) from event labels to **section names**, for example:

  ```yaml
  marker_to_section:
    BASELINE_START: baseline
    MUSIC1_START: music_block_1
    MUSIC2_START: music_block_2
    RECOVERY_START: recovery
  ```

- For each section:
  - Identify start and end times.
  - Extract the corresponding part of the RR series.
  - Compute HRV metrics for that section.

### 5.3 Features File → Validation and Visual QC

- Optionally load the Features file for:
  - Comparing HRV Logger’s pre-computed metrics with our `neurokit2` results (sanity check).
  - Quick plotting of DFA α1, LF/HF, etc. over time.
- If there are discrepancies:
  - Check window definitions, artefact correction, and frequency bands.

### 5.4 Heart Rate File → Supplementary Information

- Use the heart rate file to:
  - Plot heart rate over time with section boundaries.
  - Provide context for HRV patterns (e.g., increasing HR during a task).
- HRV itself will still be computed primarily from the RR file.

---

## 6. Recommended Workflow for HRV Logger Data in This Project

1. **Recordings**
   - Design a clear protocol (baseline, tasks, recovery).
   - Use consistent event markers for all participants.
   - Use a Polar or similar chest strap and stable app settings.

2. **Export from HRV Logger**
   - Export all four files for each session:
     - HR
     - RR
     - Features
     - Events
   - Store them in participant-specific folders.

3. **Import into Python**
   - The pipeline reads the RR and Events files as primary inputs.
   - Optionally loads Features and HR files for validation and plots.

4. **Preprocessing & Segmentation**
   - Clean RR data, log artefacts.
   - Segment recordings into sections using Events.

5. **HRV Analysis**
   - Use `neurokit2` to compute time-, frequency-, and non-linear HRV for each section.
   - Optionally re-create DFA α1 and compare to HRV Logger’s values.

6. **Output & Visualisation**
   - Per-participant CSV with sections and HRV metrics.
   - Group-level CSV combining all participants.
   - Individual plots (time series + per-section metrics).
   - Group-level plots (boxplots/violin plots by section).

---

## 7. Summary

HRV Logger is a flexible app that:

- Records high-resolution RR data
- Computes HRV features in real time
- Allows event annotation during recordings
- Exports data in a format that works well with Python workflows

For this project:

- The **RR file** and **Events file** are the key inputs.
- The **Features file** is useful for validation.
- Standardising recording and export settings (window length, artefact correction, markers) will make the `neurokit2`-based analysis much easier, more reproducible, and scientifically transparent.
