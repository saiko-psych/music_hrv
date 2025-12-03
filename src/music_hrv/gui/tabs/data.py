"""Data tab - Import settings and participant table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from music_hrv.cleaning.rr import CleaningConfig
from music_hrv.io import DEFAULT_ID_PATTERN, PREDEFINED_PATTERNS
from music_hrv.gui.shared import (
    cached_load_hrv_logger_preview,
    cached_load_vns_preview,
    cached_load_participants,
    save_participant_data,
    show_toast,
    validate_regex_pattern,
    get_quality_badge,
)


# Mapping of folder name patterns to recording app info
RECORDING_APP_DETECTION = {
    "hrv_logger": {"name": "HRV Logger", "device": "Polar H10", "sampling_rate": 1000},
    "hrv-logger": {"name": "HRV Logger", "device": "Polar H10", "sampling_rate": 1000},
    "vns_analyse": {"name": "VNS Analyse", "device": "Unknown", "sampling_rate": 1000},
    "vns-analyse": {"name": "VNS Analyse", "device": "Unknown", "sampling_rate": 1000},
    "vns": {"name": "VNS Analyse", "device": "Unknown", "sampling_rate": 1000},
    "elite_hrv": {"name": "Elite HRV", "device": "Unknown", "sampling_rate": 1000},
    "elite-hrv": {"name": "Elite HRV", "device": "Unknown", "sampling_rate": 1000},
    "elitehrv": {"name": "Elite HRV", "device": "Unknown", "sampling_rate": 1000},
}


def analyze_folder_structure(root_path: Path) -> dict:
    """Analyze the folder structure to find data sources.

    Expected structure:
    raw/
    ├── hrv_logger/
    │   ├── RR_0001TEST.csv
    │   └── Events_0001TEST.csv
    ├── vns/
    │   └── participant1.txt
    └── elite_hrv/
        └── participant2.csv

    Returns:
        Dictionary with:
        - 'sources': list of detected data sources with their info
        - 'tree': string representation of folder structure
    """
    if not root_path.exists():
        return {"sources": [], "tree": "Directory not found"}

    sources = []
    tree_lines = [f"📁 {root_path.name}/"]

    # Check immediate subdirectories
    subdirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

    for subdir in subdirs:
        folder_name = subdir.name.lower()
        app_info = None

        # Check if folder matches a known app
        for pattern, info in RECORDING_APP_DETECTION.items():
            if pattern == folder_name or folder_name.startswith(pattern):
                app_info = info.copy()
                app_info["folder"] = subdir.name
                app_info["path"] = str(subdir)
                break

        if app_info:
            # Count files in this folder
            csv_files = list(subdir.glob("*.csv"))
            txt_files = list(subdir.glob("*.txt"))
            all_files = csv_files + txt_files

            app_info["file_count"] = len(all_files)
            sources.append(app_info)

            tree_lines.append(f"├── 📂 {subdir.name}/ ({len(all_files)} files) → {app_info['name']}")

            # Show first few files as preview
            for i, f in enumerate(sorted(all_files)[:3]):
                prefix = "│   ├──" if i < min(2, len(all_files) - 1) else "│   └──"
                tree_lines.append(f"{prefix} 📄 {f.name}")
            if len(all_files) > 3:
                tree_lines.append(f"│   └── ... and {len(all_files) - 3} more files")
        else:
            # Unknown folder - just count files
            all_files = list(subdir.glob("*.*"))
            data_files = [f for f in all_files if f.suffix.lower() in ('.csv', '.txt')]
            if data_files:
                tree_lines.append(f"├── 📂 {subdir.name}/ ({len(data_files)} data files) → Unknown format")
            else:
                tree_lines.append(f"├── 📂 {subdir.name}/")

    # Also check if root itself contains data files (flat structure)
    root_csv = list(root_path.glob("*.csv"))
    root_txt = list(root_path.glob("*.txt"))
    root_files = root_csv + root_txt

    if root_files and not subdirs:
        # Flat structure - try to detect from folder name
        app_info = None
        for pattern, info in RECORDING_APP_DETECTION.items():
            if pattern in root_path.name.lower():
                app_info = info.copy()
                break

        if not app_info:
            # Try to detect from file patterns
            has_rr = any("RR" in f.name.upper() for f in root_files)
            has_events = any("Events" in f.name for f in root_files)
            if has_rr and has_events:
                app_info = RECORDING_APP_DETECTION["hrv_logger"].copy()
            else:
                app_info = {"name": "Unknown", "device": "Unknown", "sampling_rate": 1000}

        app_info["folder"] = root_path.name
        app_info["path"] = str(root_path)
        app_info["file_count"] = len(root_files)
        sources.append(app_info)

        for i, f in enumerate(sorted(root_files)[:5]):
            tree_lines.append(f"├── 📄 {f.name}")
        if len(root_files) > 5:
            tree_lines.append(f"└── ... and {len(root_files) - 5} more files")

    return {
        "sources": sources,
        "tree": "\n".join(tree_lines),
    }


def detect_recording_app(data_path: Path) -> dict:
    """Detect recording app from folder name in the data path.

    Args:
        data_path: Path to the data directory

    Returns:
        Dictionary with 'name', 'device', and 'sampling_rate' keys
    """
    # Check the folder name directly first
    folder_name = data_path.name.lower()
    for folder_pattern, app_info in RECORDING_APP_DETECTION.items():
        if folder_pattern == folder_name or folder_name.startswith(folder_pattern):
            return app_info.copy()

    # Check each part of the path (normalized to lowercase)
    path_parts = [p.lower() for p in data_path.parts]

    for folder_pattern, app_info in RECORDING_APP_DETECTION.items():
        if folder_pattern in path_parts:
            return app_info.copy()

    # Default if no pattern matched
    return {"name": "Unknown", "device": "Unknown", "sampling_rate": 1000}


def render_data_tab():
    """Render the Data tab content."""
    st.header("Data Import")

    # Quick help section
    with st.expander("Help - Getting Started", expanded=False):
        st.markdown("""
        ### Workflow Overview

        1. **Import Data**: Select your HRV Logger data folder below. The app will automatically
           detect all RR interval and event files.

        2. **Assign Groups**: Use the participant table to assign each participant to a study group.
           Groups define which events are expected for each participant.

        3. **Review Events**: Go to the Participant tab to see individual RR interval plots and events.
           You can add manual events by clicking on the plot.

        4. **Quality Check**: The app automatically detects gaps in data and high variability segments.
           Use "Batch Processing" to detect issues across all participants at once.

        ---

        **Key Terms:**
        - **RR Interval**: Time between consecutive heartbeats (in milliseconds)
        - **Canonical Event**: Standardized event name (e.g., `measurement_start`)
        - **Synonym**: Alternative label that maps to a canonical event
        - **Gap**: Period where data is missing (>15s between timestamps by default)
        """)

    # Import Settings section
    with st.expander("Import Settings", expanded=False):
        col_cfg1, col_cfg2 = st.columns(2)

        with col_cfg1:
            st.markdown("**Participant ID Pattern**")

            # Predefined pattern dropdown
            pattern_options = list(PREDEFINED_PATTERNS.keys()) + ["Custom pattern..."]
            selected_pattern_name = st.selectbox(
                "Select pattern format",
                options=pattern_options,
                index=0,
                key="pattern_selector",
                help="Choose a predefined pattern or select 'Custom pattern...' to enter your own",
            )

            # Get the pattern based on selection
            if selected_pattern_name == "Custom pattern...":
                id_pattern = st.text_input(
                    "Custom regex pattern",
                    value=DEFAULT_ID_PATTERN,
                    help="Regex pattern with named group 'participant'",
                    key="id_pattern_input",
                )
            else:
                id_pattern = PREDEFINED_PATTERNS[selected_pattern_name]
                st.code(id_pattern, language=None)

            # Real-time validation for regex pattern
            pattern_error = validate_regex_pattern(id_pattern)
            if pattern_error:
                st.error(f"Invalid regex: {pattern_error}")
            elif "(?P<participant>" not in id_pattern:
                st.warning("Pattern should include named group '(?P<participant>...)'")

        with col_cfg2:
            st.markdown("**RR Cleaning Thresholds**")

            def update_cleaning_config():
                """Callback to update cleaning config and clear cache."""
                st.session_state.cleaning_config = CleaningConfig(
                    rr_min_ms=st.session_state.rr_min_input,
                    rr_max_ms=st.session_state.rr_max_input,
                    sudden_change_pct=st.session_state.sudden_change_input,
                )
                cached_load_hrv_logger_preview.clear()

            col_rr1, col_rr2 = st.columns(2)
            with col_rr1:
                st.number_input(
                    "Min RR (ms)",
                    min_value=200,
                    max_value=1000,
                    value=st.session_state.cleaning_config.rr_min_ms,
                    step=10,
                    key="rr_min_input",
                    on_change=update_cleaning_config,
                )
            with col_rr2:
                st.number_input(
                    "Max RR (ms)",
                    min_value=1000,
                    max_value=3000,
                    value=st.session_state.cleaning_config.rr_max_ms,
                    step=10,
                    key="rr_max_input",
                    on_change=update_cleaning_config,
                )
            st.slider(
                "Sudden change threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.cleaning_config.sudden_change_pct,
                step=0.05,
                format="%.2f",
                key="sudden_change_input",
                on_change=update_cleaning_config,
            )

    # Data directory input
    col1, col2 = st.columns([3, 1])
    with col1:
        data_dir_input = st.text_input(
            "Raw data directory path",
            value=st.session_state.data_dir or "data/raw",
            help="Path to your raw data folder (should contain subfolders like hrv_logger, vns, etc.)",
        )
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        analyze_clicked = st.button("Analyze Folder", type="secondary", use_container_width=True)

    # Analyze folder structure when button clicked or path changes
    data_path = Path(data_dir_input).expanduser()

    if analyze_clicked and data_path.exists():
        folder_analysis = analyze_folder_structure(data_path)
        st.session_state.folder_analysis = folder_analysis
        st.session_state.analyzed_path = str(data_path)

    # Show folder analysis if available
    if "folder_analysis" in st.session_state and st.session_state.get("analyzed_path") == str(data_path):
        folder_analysis = st.session_state.folder_analysis

        # Show tree diagram
        with st.expander("Folder Structure", expanded=True):
            st.code(folder_analysis["tree"], language=None)

        # Show detected sources and let user select
        sources = folder_analysis["sources"]
        if sources:
            st.markdown("**Detected Data Sources:**")

            # Create source selection
            source_options = []
            for src in sources:
                label = f"{src['folder']} → {src['name']} ({src['file_count']} files)"
                source_options.append(label)

            if len(sources) == 1:
                # Auto-select if only one source
                selected_idx = 0
                st.info(f"Found: **{sources[0]['name']}** in `{sources[0]['folder']}/`")
            else:
                selected_idx = st.selectbox(
                    "Select data source to load",
                    options=range(len(source_options)),
                    format_func=lambda i: source_options[i],
                    key="source_selector",
                )

            selected_source = sources[selected_idx]

            # Show source info
            col_src1, col_src2, col_src3 = st.columns(3)
            with col_src1:
                st.metric("App", selected_source["name"])
            with col_src2:
                st.metric("Device", selected_source["device"])
            with col_src3:
                st.metric("Files", selected_source["file_count"])

            # Check if the selected app is supported
            if selected_source["name"] == "Elite HRV":
                st.warning(
                    f"**{selected_source['name']}** detected but not yet supported. "
                    "HRV Logger and VNS Analyse are currently supported."
                )

            # Load button
            if st.button("Load Selected Source", type="primary", use_container_width=True):
                load_path = Path(selected_source["path"])
                st.session_state.data_dir = str(load_path)
                st.session_state.default_device_settings = {
                    "recording_app": selected_source["name"],
                    "device": selected_source["device"],
                    "sampling_rate": selected_source["sampling_rate"],
                }

                with st.status("Loading recordings...", expanded=True) as status:
                    try:
                        st.write(f"Loading from: {selected_source['name']}")
                        st.write("Discovering recordings...")

                        config_dict = {
                            "rr_min_ms": st.session_state.cleaning_config.rr_min_ms,
                            "rr_max_ms": st.session_state.cleaning_config.rr_max_ms,
                            "sudden_change_pct": st.session_state.cleaning_config.sudden_change_pct,
                        }

                        # Use the appropriate loader based on detected app
                        if selected_source["name"] == "VNS Analyse":
                            summaries = cached_load_vns_preview(
                                str(load_path),
                                pattern=id_pattern,
                                config_dict=config_dict,
                                gui_events_dict=st.session_state.all_events,
                            )
                        else:
                            # Default to HRV Logger format (also used for unknown)
                            summaries = cached_load_hrv_logger_preview(
                                str(load_path),
                                pattern=id_pattern,
                                config_dict=config_dict,
                                gui_events_dict=st.session_state.all_events,
                            )

                        st.write(f"Processing {len(summaries)} participant(s)...")
                        st.session_state.summaries = summaries

                        # Auto-assign to Default group if not assigned
                        for summary in summaries:
                            if summary.participant_id not in st.session_state.participant_groups:
                                st.session_state.participant_groups[summary.participant_id] = "Default"

                        status.update(label=f"Loaded {len(summaries)} participant(s)", state="complete")
                        show_toast(f"Loaded {len(summaries)} participant(s)", icon="success")
                    except Exception as e:
                        status.update(label="Error loading data", state="error")
                        st.error(f"Error loading data: {e}")
        else:
            st.warning("No supported data sources found. Check that your folder structure matches the expected format.")

    # Store id_pattern for use by other tabs
    st.session_state.id_pattern = id_pattern

    if st.session_state.summaries:
        st.markdown("---")
        _render_participants_table()


def _render_participants_table():
    """Render the participants overview table."""
    st.subheader("Participants Overview")

    # Smart status summary - only show issues if they exist
    issues = []
    total_participants = len(st.session_state.summaries)

    # Check for high artifact rates
    high_artifact = [s for s in st.session_state.summaries if s.artifact_ratio > 0.15]
    if high_artifact:
        issues.append(f"**{len(high_artifact)}** participant(s) with high artifact rates (>15%)")

    # Check for duplicates
    with_duplicates = [s for s in st.session_state.summaries if s.duplicate_rr_intervals > 0]
    if with_duplicates:
        issues.append(f"**{len(with_duplicates)}** participant(s) with duplicate RR intervals")

    # Check for multiple files
    with_multi_files = [s for s in st.session_state.summaries
                       if getattr(s, 'rr_file_count', 1) > 1 or getattr(s, 'events_file_count', 0) > 1]
    if with_multi_files:
        issues.append(f"**{len(with_multi_files)}** participant(s) with multiple files (merged)")

    # Check for missing events
    no_events = [s for s in st.session_state.summaries if s.events_detected == 0]
    if no_events:
        issues.append(f"**{len(no_events)}** participant(s) with no events detected")

    # Display status summary
    if issues:
        with st.container():
            st.markdown("**Issues Detected:**")
            for issue in issues:
                st.markdown(f"- {issue}")
            st.markdown("---")
    else:
        st.success(f"All {total_participants} participants look good! No issues detected.")

    # Create editable dataframe
    participants_data = []
    loaded_participants = cached_load_participants()

    for summary in st.session_state.summaries:
        recording_dt_str = ""
        if summary.recording_datetime:
            recording_dt_str = summary.recording_datetime.strftime("%Y-%m-%d %H:%M")

        # Show file counts
        rr_count = getattr(summary, 'rr_file_count', 1)
        ev_count = getattr(summary, 'events_file_count', 1 if summary.events_detected > 0 else 0)
        files_str = f"{rr_count}RR/{ev_count}Ev"
        if rr_count > 1 or ev_count > 1:
            files_str = f"* {files_str}"

        quality_badge = get_quality_badge(100, summary.artifact_ratio)

        # Get device info from session state
        device_settings = st.session_state.get("default_device_settings", {})
        device_name = device_settings.get("device", "Unknown")

        participants_data.append({
            "Participant": summary.participant_id,
            "Quality": quality_badge,
            "Saved": "Y" if summary.participant_id in loaded_participants else "N",
            "Device": device_name,
            "Files": files_str,
            "Date/Time": recording_dt_str,
            "Group": st.session_state.participant_groups.get(summary.participant_id, "Default"),
            "Total Beats": summary.total_beats,
            "Retained": summary.retained_beats,
            "Duplicates": summary.duplicate_rr_intervals,
            "Artifacts (%)": f"{summary.artifact_ratio * 100:.1f}",
            "Duration (min)": f"{summary.duration_s / 60:.1f}",
            "Events": summary.events_detected,
            "Total Events": summary.events_detected + summary.duplicate_events,
            "Duplicate Events": summary.duplicate_events,
            "RR Range (ms)": f"{int(summary.rr_min_ms)}-{int(summary.rr_max_ms)}",
            "Mean RR (ms)": f"{summary.rr_mean_ms:.0f}",
        })

    df_participants = pd.DataFrame(participants_data)

    # Editable dataframe
    edited_df = st.data_editor(
        df_participants,
        column_config={
            "Participant": st.column_config.TextColumn("Participant", disabled=True, width="medium"),
            "Quality": st.column_config.TextColumn("Quality", disabled=True, width="small",
                help="Green=Good (<5% artifacts), Yellow=Moderate (5-15%), Red=Poor (>15%)"),
            "Saved": st.column_config.TextColumn("Saved", disabled=True, width="small"),
            "Device": st.column_config.TextColumn("Device", disabled=True, width="small",
                help="Recording device used (e.g., Polar H10)"),
            "Files": st.column_config.TextColumn("Files", disabled=True, width="small",
                help="RR files / Events files. * indicates multiple files (merged)"),
            "Group": st.column_config.SelectboxColumn("Group", options=list(st.session_state.groups.keys()),
                required=True, help="Assign participant to a group", width="medium"),
            "Total Beats": st.column_config.NumberColumn("Total Beats", disabled=True, format="%d"),
            "Retained": st.column_config.NumberColumn("Retained", disabled=True, format="%d"),
            "Artifacts (%)": st.column_config.TextColumn("Artifacts (%)", disabled=True, width="small"),
            "Total Events": st.column_config.NumberColumn("Total Events", disabled=True, format="%d"),
            "Duplicate Events": st.column_config.NumberColumn("Duplicate Events", disabled=True, format="%d"),
        },
        use_container_width=True,
        hide_index=True,
        key="participants_table",
        disabled=["Participant", "Saved", "Device", "Date/Time", "Total Beats", "Retained", "Duplicates",
                  "Artifacts (%)", "Duration (min)", "Events", "Total Events", "Duplicate Events",
                  "RR Range (ms)", "Mean RR (ms)"]
    )

    # Auto-save group assignments when changed
    groups_changed = False
    for idx, row in edited_df.iterrows():
        participant_id = row["Participant"]
        new_group = row["Group"]
        old_group = st.session_state.participant_groups.get(participant_id)
        if old_group != new_group:
            st.session_state.participant_groups[participant_id] = new_group
            groups_changed = True

    if groups_changed:
        save_participant_data()
        cached_load_participants.clear()
        show_toast("Group assignments saved", icon="success")

    # Show warning if any participant has duplicate RR intervals
    high_duplicates = [
        (row["Participant"], row["Duplicates"])
        for _, row in df_participants.iterrows()
        if row["Duplicates"] > 0
    ]
    if high_duplicates:
        st.warning(
            f"**Duplicate RR intervals detected!** "
            f"{len(high_duplicates)} participant(s) have duplicate RR intervals that were removed."
        )
        with st.expander("Show participants with duplicates"):
            for pid, dup_count in high_duplicates:
                st.text(f"- {pid}: {dup_count} duplicates removed")

    # Download button
    csv_participants = df_participants.to_csv(index=False)
    st.download_button(
        label="Download Participants CSV",
        data=csv_participants,
        file_name="participants_overview.csv",
        mime="text/csv",
        use_container_width=False,
    )
    st.info("**Tip:** Group assignments save automatically when you change them in the table above.")
