"""Data tab - Import settings and participant table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from rrational.cleaning.rr import CleaningConfig
from rrational.io import DEFAULT_ID_PATTERN, PREDEFINED_PATTERNS
from rrational.gui.help_text import CLEANING_THRESHOLDS_HELP, DATA_CORRECTION_WORKFLOW
from rrational.gui.shared import (
    cached_load_hrv_logger_preview,
    cached_load_vns_preview,
    cached_load_participants,
    cached_load_recording,
    cached_clean_rr_intervals,
    cached_quality_analysis,
    detect_time_gaps,
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
    tree_lines = [f"{root_path.name}/"]

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

            tree_lines.append(f"├── {subdir.name}/ ({len(all_files)} files) → {app_info['name']}")

            # Show first few files as preview
            for i, f in enumerate(sorted(all_files)[:3]):
                prefix = "│   ├──" if i < min(2, len(all_files) - 1) else "│   └──"
                tree_lines.append(f"{prefix} {f.name}")
            if len(all_files) > 3:
                tree_lines.append(f"│   └── ... and {len(all_files) - 3} more files")
        else:
            # Unknown folder - just count files
            all_files = list(subdir.glob("*.*"))
            data_files = [f for f in all_files if f.suffix.lower() in ('.csv', '.txt')]
            if data_files:
                tree_lines.append(f"├── {subdir.name}/ ({len(data_files)} data files) → Unknown format")
            else:
                tree_lines.append(f"├── {subdir.name}/")

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
            tree_lines.append(f"├── {f.name}")
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
           Review each participant individually in the Participants tab.

        ---

        **Key Terms:**
        - **RR Interval**: Time between consecutive heartbeats (in milliseconds)
        - **Canonical Event**: Standardized event name (e.g., `measurement_start`)
        - **Synonym**: Alternative label that maps to a canonical event
        - **Gap**: Period where data is missing (>15s between timestamps by default)
        """)

    # Data correction workflow help
    with st.expander("Data Correction Workflow & Best Practices", expanded=False):
        st.markdown(DATA_CORRECTION_WORKFLOW)

    # Import Settings section
    with st.expander("Import Settings", expanded=False):
        col_cfg1, col_cfg2, col_cfg3 = st.columns(3)

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
                # Convert percentage (0-100) back to decimal (0.0-1.0)
                sudden_pct = st.session_state.get("sudden_change_input_pct", 100) / 100.0
                st.session_state.cleaning_config = CleaningConfig(
                    rr_min_ms=st.session_state.rr_min_input,
                    rr_max_ms=st.session_state.rr_max_input,
                    sudden_change_pct=sudden_pct,
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
                "Sudden change threshold (%)",
                min_value=0,
                max_value=100,
                value=int(st.session_state.cleaning_config.sudden_change_pct * 100),
                step=5,
                key="sudden_change_input_pct",
                on_change=update_cleaning_config,
                help="100% = disabled. Use NeuroKit2 artifact correction instead."
            )
            with st.popover("About thresholds"):
                st.markdown(CLEANING_THRESHOLDS_HELP)

        with col_cfg3:
            st.markdown("**Device Settings**")

            # Initialize default device settings if not present
            if "default_device_settings" not in st.session_state:
                st.session_state.default_device_settings = {
                    "device": "Polar H10",
                    "sampling_rate": 1000,
                }

            # Device selector
            devices = ["Polar H10", "Polar H9", "Polar OH1", "Garmin HRM-Pro", "Movesense", "Other"]
            current_device = st.session_state.default_device_settings.get("device", "Polar H10")
            dev_idx = devices.index(current_device) if current_device in devices else 0

            def update_device():
                st.session_state.default_device_settings["device"] = st.session_state.device_select
                # Auto-set sampling rate based on device
                device_rates = {
                    "Polar H10": 1000, "Polar H9": 1000, "Polar OH1": 135,
                    "Garmin HRM-Pro": 1000, "Movesense": 512
                }
                if st.session_state.device_select in device_rates:
                    st.session_state.default_device_settings["sampling_rate"] = device_rates[st.session_state.device_select]

            st.selectbox(
                "HR Sensor Device",
                options=devices,
                index=dev_idx,
                key="device_select",
                on_change=update_device,
                help="The heart rate sensor used for recordings"
            )

            sampling_rate = st.session_state.default_device_settings.get("sampling_rate", 1000)
            st.text_input("Sampling Rate", value=f"{sampling_rate} Hz", disabled=True,
                         help="Based on device selection")

    # Data directory input - default to project data dir if available
    project_manager = st.session_state.get("project_manager")
    if project_manager:
        default_data_dir = str(project_manager.get_data_dir())
    elif st.session_state.data_dir:
        default_data_dir = st.session_state.data_dir
    else:
        default_data_dir = "data/raw"

    col1, col2 = st.columns([3, 1])
    with col1:
        data_dir_input = st.text_input(
            "Raw data directory path",
            value=default_data_dir,
            help="Path to your raw data folder (should contain subfolders like hrv_logger, vns, etc.)",
        )
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        analyze_clicked = st.button("Analyze Folder", type="secondary", width='stretch')

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

            # Create source selection - multiselect for loading multiple sources
            source_labels = {src['folder']: f"{src['folder']} → {src['name']} ({src['file_count']} files)"
                            for src in sources}

            # Default to all supported sources selected
            supported_defaults = [src['folder'] for src in sources if src['name'] != "Elite HRV"]

            selected_folders = st.multiselect(
                "Select data sources to load",
                options=[src['folder'] for src in sources],
                default=supported_defaults,
                format_func=lambda f: source_labels[f],
                key="source_selector",
                help="Select one or more data sources to load. You can combine HRV Logger and VNS data."
            )

            # Show info about selected sources
            if selected_folders:
                selected_sources = [src for src in sources if src['folder'] in selected_folders]
                total_files = sum(src['file_count'] for src in selected_sources)

                # Show summary metrics
                col_src1, col_src2, col_src3 = st.columns(3)
                with col_src1:
                    apps = ", ".join(set(src['name'] for src in selected_sources))
                    st.metric("Apps", apps)
                with col_src2:
                    st.metric("Total Files", total_files)
                with col_src3:
                    st.metric("Sources", len(selected_sources))

                # Check if any selected app is not supported
                unsupported = [src for src in selected_sources if src['name'] == "Elite HRV"]
                if unsupported:
                    st.warning(
                        "**Elite HRV** is detected but not yet supported. "
                        "It will be skipped during loading."
                    )

                # Show VNS-specific option if VNS source is selected
                has_vns = any(src['name'] == "VNS Analyse" for src in selected_sources)
                if has_vns:
                    def on_vns_corrected_change():
                        # Clear VNS cache when setting changes
                        cached_load_vns_preview.clear()

                    vns_use_corrected = st.checkbox(
                        "Use corrected RR values (VNS only)",
                        value=st.session_state.get("vns_use_corrected", False),
                        key="vns_corrected_checkbox_main",
                        on_change=on_vns_corrected_change,
                        help="VNS files contain both raw and corrected RR values. "
                             "Enable to use the corrected (artifact-cleaned) values."
                    )
                    st.session_state.vns_use_corrected = vns_use_corrected

                # Load button
                if st.button("Load Selected Sources", type="primary", width='stretch'):
                    with st.status("Loading recordings...", expanded=True) as status:
                        try:
                            all_summaries = []
                            config_dict = {
                                "rr_min_ms": st.session_state.cleaning_config.rr_min_ms,
                                "rr_max_ms": st.session_state.cleaning_config.rr_max_ms,
                                "sudden_change_pct": st.session_state.cleaning_config.sudden_change_pct,
                            }

                            for src in selected_sources:
                                if src['name'] == "Elite HRV":
                                    st.write(f"Skipping {src['folder']} (Elite HRV not yet supported)")
                                    continue

                                st.write(f"Loading from: {src['folder']} ({src['name']})")
                                load_path = Path(src["path"])

                                # Use the appropriate loader based on detected app
                                if src["name"] == "VNS Analyse":
                                    summaries = cached_load_vns_preview(
                                        str(load_path),
                                        pattern=id_pattern,
                                        config_dict=config_dict,
                                        gui_events_dict=st.session_state.all_events,
                                        use_corrected=st.session_state.get("vns_use_corrected", False),
                                    )
                                    app_name = "VNS Analyse"
                                else:
                                    # Default to HRV Logger format
                                    summaries = cached_load_hrv_logger_preview(
                                        str(load_path),
                                        pattern=id_pattern,
                                        config_dict=config_dict,
                                        gui_events_dict=st.session_state.all_events,
                                    )
                                    app_name = "HRV Logger"

                                # Ensure source_app is set (handles old cached data)
                                for s in summaries:
                                    if not hasattr(s, 'source_app') or s.source_app == "Unknown":
                                        object.__setattr__(s, 'source_app', app_name)

                                st.write(f"   Found {len(summaries)} participant(s)")
                                all_summaries.extend(summaries)

                            # Store all summaries
                            st.session_state.summaries = all_summaries
                            st.session_state.data_dir = str(data_path)

                            # Auto-assign to Default group if not assigned
                            for summary in all_summaries:
                                if summary.participant_id not in st.session_state.participant_groups:
                                    st.session_state.participant_groups[summary.participant_id] = "Default"

                            status.update(label=f"Loaded {len(all_summaries)} participant(s) total", state="complete")
                            show_toast(f"Loaded {len(all_summaries)} participant(s) from {len(selected_sources)} source(s)", icon="success")
                        except Exception as e:
                            status.update(label="Error loading data", state="error")
                            st.error(f"Error loading data: {e}")
                            import traceback
                            st.code(traceback.format_exc())
            else:
                st.info("Select at least one data source to load")
        else:
            st.warning("No supported data sources found. Check that your folder structure matches the expected format.")

    # Store id_pattern for use by other tabs
    st.session_state.id_pattern = id_pattern

    if st.session_state.summaries:
        st.markdown("---")
        _render_participants_table()

        # NOTE: Batch processing disabled - users should inspect participants individually
        # Uncomment below to re-enable if needed
        # st.markdown("---")
        # _render_batch_processing()


def _build_participants_csv() -> str:
    """Build CSV string from participants data for download."""
    loaded_participants = cached_load_participants()
    rows = []

    for summary in st.session_state.summaries:
        recording_dt_str = ""
        if summary.recording_datetime:
            recording_dt_str = summary.recording_datetime.strftime("%Y-%m-%d %H:%M")

        rr_count = getattr(summary, 'rr_file_count', 1)
        ev_count = getattr(summary, 'events_file_count', 1 if summary.events_detected > 0 else 0)
        files_str = f"{rr_count}RR/{ev_count}Ev"

        playlist_code = st.session_state.get("participant_playlists", {}).get(summary.participant_id, "")
        group_code = st.session_state.participant_groups.get(summary.participant_id, "Default")

        group_data = st.session_state.groups.get(group_code, {})
        group_label = group_data.get("label", "") if isinstance(group_data, dict) else ""
        group_display = group_label if group_label else group_code

        playlist_display = ""
        if playlist_code:
            playlist_data = st.session_state.get("playlist_groups", {}).get(playlist_code, {})
            playlist_label = playlist_data.get("label", "") if isinstance(playlist_data, dict) else ""
            playlist_display = playlist_label if playlist_label else playlist_code

        source_app = getattr(summary, 'source_app', 'Unknown')
        device_settings = st.session_state.get("default_device_settings", {})
        device_name = device_settings.get("device", "Unknown")

        rows.append({
            "Participant": summary.participant_id,
            "Group": group_display,
            "Playlist": playlist_display,
            "App": source_app,
            "Device": device_name,
            "Files": files_str,
            "Date/Time": recording_dt_str,
            "Total Beats": summary.total_beats,
            "Retained": summary.retained_beats,
            "Duplicates": summary.duplicate_rr_intervals,
            "Artifacts (%)": f"{summary.artifact_ratio * 100:.1f}",
            "Duration (min)": f"{summary.duration_s / 60:.1f}",
            "Events": summary.events_detected,
        })

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


@st.fragment
def _render_participants_data_editor(group_display_options, group_label_to_code,
                                      playlist_display_options, playlist_label_to_code):
    """Render the data editor as a fragment to allow consecutive edits.

    Using @st.fragment isolates this component - when changes are made,
    only this fragment reruns, not the whole page. This prevents the
    data_editor from resetting between consecutive edits.

    IMPORTANT: We store the dataframe in session state and only rebuild it
    when summaries change. This prevents the data_editor from resetting
    when we update session_state after detecting a change.
    """
    # Check if we need to rebuild the dataframe
    # Only rebuild when: summaries change, or dataframe doesn't exist yet
    summaries_key = tuple(s.participant_id for s in st.session_state.summaries) if st.session_state.summaries else ()
    need_rebuild = (
        "_participants_df" not in st.session_state or
        st.session_state.get("_participants_df_key") != summaries_key
    )

    if need_rebuild:
        # Build dataframe from current session state
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

            # Get playlist assignment
            playlist_code = st.session_state.get("participant_playlists", {}).get(summary.participant_id, "")
            group_code = st.session_state.participant_groups.get(summary.participant_id, "Default")

            # Get group label for display (show only label if defined, otherwise code)
            group_data = st.session_state.groups.get(group_code, {})
            group_label = group_data.get("label", "") if isinstance(group_data, dict) else ""
            group_display = group_label if group_label else group_code

            # Get playlist label for display (show only label if defined, otherwise code)
            playlist_display = ""
            if playlist_code:
                playlist_data = st.session_state.get("playlist_groups", {}).get(playlist_code, {})
                playlist_label = playlist_data.get("label", "") if isinstance(playlist_data, dict) else ""
                playlist_display = playlist_label if playlist_label else playlist_code

            # Get source app
            source_app = getattr(summary, 'source_app', 'Unknown')

            # Check if participant has CSV data (group or playlist assigned)
            has_group_assigned = group_code != "Default"
            has_playlist_assigned = bool(playlist_code)
            csv_status = ""
            if has_group_assigned and has_playlist_assigned:
                csv_status = "**"  # Both
            elif has_group_assigned:
                csv_status = "G"  # Group only
            elif has_playlist_assigned:
                csv_status = "P"  # Playlist only
            else:
                csv_status = "—"  # None

            participants_data.append({
                "Participant": summary.participant_id,
                "CSV": csv_status,
                "Quality": quality_badge,
                "Saved": "Y" if summary.participant_id in loaded_participants else "N",
                "App": source_app,
                "Device": device_name,
                "Files": files_str,
                "Date/Time": recording_dt_str,
                "Group": group_display,
                "Playlist": playlist_display,
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

        st.session_state._participants_df = pd.DataFrame(participants_data)
        st.session_state._participants_df_key = summaries_key

    df_participants = st.session_state._participants_df

    # Editable dataframe
    edited_df = st.data_editor(
        df_participants,
        column_config={
            "Participant": st.column_config.TextColumn("Participant", disabled=True, width="medium"),
            "CSV": st.column_config.TextColumn("CSV", disabled=True, width="small",
                help="CSV import status: **=Both Group & Playlist, G=Group only, P=Playlist only, —=None"),
            "Quality": st.column_config.TextColumn("Quality", disabled=True, width="small",
                help="Green=Good (<5% artifacts), Yellow=Moderate (5-15%), Red=Poor (>15%)"),
            "Saved": st.column_config.TextColumn("Saved", disabled=True, width="small"),
            "App": st.column_config.TextColumn("App", disabled=True, width="small",
                help="Recording app (HRV Logger, VNS Analyse, etc.)"),
            "Device": st.column_config.TextColumn("Device", disabled=True, width="small",
                help="Recording device used (e.g., Polar H10)"),
            "Files": st.column_config.TextColumn("Files", disabled=True, width="small",
                help="RR files / Events files. * indicates multiple files (merged)"),
            "Group": st.column_config.SelectboxColumn("Group", options=group_display_options,
                required=True, help="Select study group", width="small"),
            "Playlist": st.column_config.SelectboxColumn("Playlist", options=playlist_display_options,
                required=False, help="Select playlist/randomization", width="small"),
            "Total Beats": st.column_config.NumberColumn("Total Beats", disabled=True, format="%d"),
            "Retained": st.column_config.NumberColumn("Retained", disabled=True, format="%d"),
            "Artifacts (%)": st.column_config.TextColumn("Artifacts (%)", disabled=True, width="small"),
            "Total Events": st.column_config.NumberColumn("Total Events", disabled=True, format="%d"),
            "Duplicate Events": st.column_config.NumberColumn("Duplicate Events", disabled=True, format="%d"),
        },
        width='stretch',
        hide_index=True,
        key="participants_table",
        disabled=["Participant", "CSV", "Saved", "App", "Device", "Date/Time", "Total Beats", "Retained", "Duplicates",
                  "Artifacts (%)", "Duration (min)", "Events", "Total Events", "Duplicate Events",
                  "RR Range (ms)", "Mean RR (ms)"]
    )

    # Auto-save group and playlist assignments when changed
    groups_changed = False
    playlists_changed = False

    # Initialize participant_playlists if not exists
    if "participant_playlists" not in st.session_state:
        st.session_state.participant_playlists = {}

    for idx, row in edited_df.iterrows():
        participant_id = row["Participant"]

        # Check group change - convert label to code using mapping
        new_group_label = row["Group"]
        new_group_code = group_label_to_code.get(new_group_label, new_group_label)
        old_group_code = st.session_state.participant_groups.get(participant_id)
        if old_group_code != new_group_code:
            st.session_state.participant_groups[participant_id] = new_group_code
            groups_changed = True

        # Check playlist change - convert label to code using mapping
        new_playlist_label = row["Playlist"]
        new_playlist_code = playlist_label_to_code.get(new_playlist_label, new_playlist_label)
        old_playlist_code = st.session_state.participant_playlists.get(participant_id, "")
        if old_playlist_code != new_playlist_code:
            st.session_state.participant_playlists[participant_id] = new_playlist_code
            playlists_changed = True

    if groups_changed or playlists_changed:
        save_participant_data()
        msg = []
        if groups_changed:
            msg.append("Groups")
        if playlists_changed:
            msg.append("Playlists")
        show_toast(f"{' & '.join(msg)} saved", icon="success")


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

    # Build group display options (show only label, map label→code)
    group_display_options = []
    group_label_to_code = {}  # Map label back to code for saving
    for gid, gdata in st.session_state.groups.items():
        glabel = gdata.get("label", "") if isinstance(gdata, dict) else ""
        display = glabel if glabel else gid
        group_display_options.append(display)
        group_label_to_code[display] = gid

    # Build playlist display options (show only label, map label→code)
    playlist_display_options = [""]
    playlist_label_to_code = {"": ""}  # Map label back to code for saving
    for pid, pdata in st.session_state.get("playlist_groups", {}).items():
        plabel = pdata.get("label", "") if isinstance(pdata, dict) else ""
        display = plabel if plabel else pid
        playlist_display_options.append(display)
        playlist_label_to_code[display] = pid

    # Render the editable table as a fragment to prevent resets during consecutive edits
    _render_participants_data_editor(group_display_options, group_label_to_code,
                                      playlist_display_options, playlist_label_to_code)

    # Show warning if any participant has duplicate RR intervals (use session state directly)
    high_duplicates = [
        (s.participant_id, s.duplicate_rr_intervals)
        for s in st.session_state.summaries
        if s.duplicate_rr_intervals > 0
    ]
    if high_duplicates:
        st.warning(
            f"**Duplicate RR intervals detected!** "
            f"{len(high_duplicates)} participant(s) have duplicate RR intervals that were removed."
        )
        with st.expander("Show participants with duplicates"):
            for pid, dup_count in high_duplicates:
                st.text(f"- {pid}: {dup_count} duplicates removed")

    # Download and Import section
    col_dl, col_import = st.columns([1, 2])

    with col_dl:
        # Build CSV from session state
        csv_data = _build_participants_csv()
        st.download_button(
            label="Download Participants CSV",
            data=csv_data,
            file_name="participants_overview.csv",
            mime="text/csv",
            width='stretch',
        )

    with col_import:
        # CSV import for group/playlist matching
        # Track expander state in session state to prevent collapse on rerun
        if "csv_import_expanded" not in st.session_state:
            st.session_state.csv_import_expanded = False
        # Keep expanded if file is uploaded (check before expander renders)
        if st.session_state.get("group_playlist_csv_upload") is not None:
            st.session_state.csv_import_expanded = True
        with st.expander("Import Group/Playlist from CSV", expanded=st.session_state.csv_import_expanded):
            st.markdown("""
            Upload a CSV file to automatically assign groups and playlists.
            1. First define labels for your group/playlist values below
            2. Then upload your CSV and map the columns
            """)

            # Value → Label mappings
            st.markdown("**Step 1: Define Value Labels**")
            st.caption("Define what each value in your CSV means (e.g., group value '5' = 'MAR')")

            # Initialize label mappings in session state
            if "csv_group_labels" not in st.session_state:
                st.session_state.csv_group_labels = {}
            if "csv_playlist_labels" not in st.session_state:
                st.session_state.csv_playlist_labels = {}

            label_col1, label_col2 = st.columns(2)

            with label_col1:
                st.markdown("**Group Value Labels**")
                # Add new group label
                new_g_col1, new_g_col2, new_g_col3 = st.columns([2, 3, 1])
                with new_g_col1:
                    new_group_val = st.text_input("Value", key="new_group_val", placeholder="e.g., 5")
                with new_g_col2:
                    new_group_label = st.text_input("Label", key="new_group_label", placeholder="e.g., MAR")
                with new_g_col3:
                    st.write("")  # Spacer
                    if st.button("Add", key="add_group_label"):
                        if new_group_val and new_group_label:
                            st.session_state.csv_group_labels[new_group_val] = new_group_label
                            st.session_state.csv_import_expanded = True
                            st.rerun()

                # Show existing group labels
                if st.session_state.csv_group_labels:
                    for gval, glabel in list(st.session_state.csv_group_labels.items()):
                        gcol1, gcol2 = st.columns([4, 1])
                        with gcol1:
                            st.text(f"{gval} → {glabel}")
                        with gcol2:
                            if st.button("X", key=f"del_g_{gval}"):
                                del st.session_state.csv_group_labels[gval]
                                st.session_state.csv_import_expanded = True
                                st.rerun()
                else:
                    st.caption("No group labels defined yet")

            with label_col2:
                st.markdown("**Playlist Value Labels**")
                # Add new playlist label
                new_p_col1, new_p_col2, new_p_col3 = st.columns([2, 3, 1])
                with new_p_col1:
                    new_playlist_val = st.text_input("Value", key="new_playlist_val", placeholder="e.g., 1")
                with new_p_col2:
                    new_playlist_label = st.text_input("Label", key="new_playlist_label", placeholder="e.g., R1")
                with new_p_col3:
                    st.write("")  # Spacer
                    if st.button("Add", key="add_playlist_label"):
                        if new_playlist_val and new_playlist_label:
                            st.session_state.csv_playlist_labels[new_playlist_val] = new_playlist_label
                            st.session_state.csv_import_expanded = True
                            st.rerun()

                # Show existing playlist labels
                if st.session_state.csv_playlist_labels:
                    for pval, plabel in list(st.session_state.csv_playlist_labels.items()):
                        pcol1, pcol2 = st.columns([4, 1])
                        with pcol1:
                            st.text(f"{pval} → {plabel}")
                        with pcol2:
                            if st.button("X", key=f"del_p_{pval}"):
                                del st.session_state.csv_playlist_labels[pval]
                                st.session_state.csv_import_expanded = True
                                st.rerun()
                else:
                    st.caption("No playlist labels defined yet")

            st.markdown("---")

            # Column mapping settings
            st.markdown("**Step 2: Map CSV Columns**")
            col_map1, col_map2, col_map3 = st.columns(3)

            with col_map1:
                participant_col = st.text_input(
                    "Participant ID column",
                    value="code",
                    key="csv_col_participant",
                    help="Column name containing participant IDs"
                )
            with col_map2:
                group_col = st.text_input(
                    "Group column",
                    value="group",
                    key="csv_col_group",
                    help="Column name for group values"
                )
            with col_map3:
                playlist_col = st.text_input(
                    "Playlist column",
                    value="playlist",
                    key="csv_col_playlist",
                    help="Column name for playlist values"
                )

            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=['csv'],
                key="group_playlist_csv_upload",
                help="Upload your CSV file with participant assignments"
            )

            # Keep expander open when file is uploaded
            if uploaded_file is not None:
                st.session_state.csv_import_expanded = True

            if uploaded_file is not None:
                try:
                    import_df = pd.read_csv(uploaded_file)
                    csv_columns = list(import_df.columns)

                    st.success(f"Found {len(import_df)} rows, {len(csv_columns)} columns")

                    # Reorder columns: selected columns first, then others
                    priority_cols = []
                    for col in [participant_col, group_col, playlist_col]:
                        if col and col in csv_columns:
                            priority_cols.append(col)
                    other_cols = [c for c in csv_columns if c not in priority_cols]
                    reordered_cols = priority_cols + other_cols
                    preview_df = import_df[reordered_cols]

                    # Preview the data (scrollable with max height)
                    st.dataframe(preview_df, width='stretch', height=200)

                    # Validate participant column exists
                    if participant_col not in csv_columns:
                        st.error(f"Participant column '{participant_col}' not found in CSV. Available: {csv_columns}")
                    else:
                        # Check which mapped columns are present
                        has_group = group_col and group_col in csv_columns
                        has_playlist = playlist_col and playlist_col in csv_columns

                        # Show mapping status
                        st.markdown("**Detected mappings:**")
                        st.write(f"- Participant: `{participant_col}` *")
                        if group_col:
                            st.write(f"- Group: `{group_col}` {'*' if has_group else 'not found'}")
                        if playlist_col:
                            st.write(f"- Playlist: `{playlist_col}` {'*' if has_playlist else 'not found'}")

                        # Show defined labels
                        if st.session_state.csv_group_labels:
                            st.write(f"- Group labels defined: {len(st.session_state.csv_group_labels)}")
                        if st.session_state.csv_playlist_labels:
                            st.write(f"- Playlist labels defined: {len(st.session_state.csv_playlist_labels)}")

                        if not has_group and not has_playlist:
                            st.warning("No valid Group or Playlist column found. Check your column mapping above.")
                        else:
                            if st.button("Apply Assignments", type="primary", key="apply_csv_assignments"):
                                matched = 0
                                not_found = []
                                groups_created = set()
                                playlists_created = set()

                                # Get current participant IDs
                                current_pids = {s.participant_id for s in st.session_state.summaries}

                                # Get the value→label mappings from session state
                                group_labels = st.session_state.csv_group_labels
                                playlist_labels = st.session_state.csv_playlist_labels

                                # First pass: create groups and playlists with labels
                                unique_groups = set()
                                unique_playlists = set()

                                for _, row in import_df.iterrows():
                                    if has_group and pd.notna(row[group_col]):
                                        unique_groups.add(str(row[group_col]).strip())
                                    if has_playlist and pd.notna(row[playlist_col]):
                                        unique_playlists.add(str(row[playlist_col]).strip())

                                # Create/update groups with labels from mapping
                                for g_code in unique_groups:
                                    g_label = group_labels.get(g_code, g_code)  # Use code as label if not defined
                                    if g_code not in st.session_state.groups:
                                        st.session_state.groups[g_code] = {"events": [], "label": g_label}
                                        groups_created.add(g_code)
                                    else:
                                        # Update label if defined
                                        if g_code in group_labels:
                                            if isinstance(st.session_state.groups[g_code], dict):
                                                st.session_state.groups[g_code]["label"] = g_label
                                            else:
                                                events = st.session_state.groups[g_code]
                                                st.session_state.groups[g_code] = {"events": events, "label": g_label}

                                # Create/update playlist groups with labels from mapping
                                if "playlist_groups" not in st.session_state:
                                    st.session_state.playlist_groups = {}

                                for p_code in unique_playlists:
                                    if p_code:
                                        p_label = playlist_labels.get(p_code, p_code)  # Use code as label if not defined
                                        if p_code not in st.session_state.playlist_groups:
                                            st.session_state.playlist_groups[p_code] = {
                                                "label": p_label,
                                                "music_order": ["music_1", "music_2", "music_3"]
                                            }
                                            playlists_created.add(p_code)
                                        else:
                                            # Update label if defined
                                            if p_code in playlist_labels:
                                                st.session_state.playlist_groups[p_code]["label"] = p_label

                                # Second pass: assign participants
                                for _, row in import_df.iterrows():
                                    pid = str(row[participant_col]).strip()

                                    if pid in current_pids:
                                        matched += 1

                                        # Assign group if present
                                        if has_group and pd.notna(row[group_col]):
                                            group_val = str(row[group_col]).strip()
                                            st.session_state.participant_groups[pid] = group_val

                                        # Assign playlist if present
                                        if has_playlist and pd.notna(row[playlist_col]):
                                            playlist_val = str(row[playlist_col]).strip()
                                            if "participant_playlists" not in st.session_state:
                                                st.session_state.participant_playlists = {}
                                            st.session_state.participant_playlists[pid] = playlist_val
                                    else:
                                        not_found.append(pid)

                                # Save all changes
                                from rrational.gui.persistence import save_groups, save_playlist_groups
                                project_path = st.session_state.get("current_project")
                                save_groups(st.session_state.groups, project_path)
                                save_playlist_groups(st.session_state.playlist_groups, project_path)
                                save_participant_data()
                                cached_load_participants.clear()

                                # Show results
                                msg_parts = [f"Applied to {matched} participants"]
                                if groups_created:
                                    msg_parts.append(f"created {len(groups_created)} groups")
                                if playlists_created:
                                    msg_parts.append(f"created {len(playlists_created)} playlists")
                                show_toast(", ".join(msg_parts), icon="success")

                                if not_found:
                                    st.warning(f"Not found in loaded data: {', '.join(not_found[:10])}" +
                                              (f" and {len(not_found) - 10} more" if len(not_found) > 10 else ""))

                                st.rerun()

                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

    st.info("**Tip:** Group and Playlist assignments save automatically when you change them in the table above.")


def _render_batch_processing():
    """Render the Batch Processing section for bulk operations."""
    with st.expander("Batch Processing", expanded=False):
        st.markdown("""
        Apply operations to **multiple participants** at once. This saves time when you have
        predefined settings you want to apply consistently across your study.
        """)

        batch_col1, batch_col2 = st.columns(2)

        with batch_col1:
            st.markdown("##### Auto-Generate Music Events")
            st.caption("Generate music section events for all participants in a playlist group")

            # Select playlist group to process
            if "playlist_groups" in st.session_state and st.session_state.playlist_groups:
                batch_playlist = st.selectbox(
                    "Select Playlist Group",
                    options=list(st.session_state.playlist_groups.keys()),
                    key="batch_playlist_group",
                    help="Generate music events for all participants assigned to this playlist group"
                )

                batch_interval = st.number_input(
                    "Music interval (minutes)",
                    min_value=1, max_value=30, value=5, step=1,
                    key="batch_music_interval",
                    help="How often the music changes"
                )

                if st.button("Generate for All in Group", key="batch_generate_music"):
                    # Find participants in this playlist group
                    participants_in_group = [
                        pid for pid, pg in st.session_state.get("participant_playlists", {}).items()
                        if pg == batch_playlist
                    ]

                    if not participants_in_group:
                        st.warning(f"No participants assigned to playlist group '{batch_playlist}'")
                    else:
                        progress = st.progress(0)
                        status = st.empty()
                        generated_count = 0

                        for i, pid in enumerate(participants_in_group):
                            status.text(f"Processing {pid}...")
                            progress.progress((i + 1) / len(participants_in_group))

                            # Get participant's events and find boundaries
                            if pid in st.session_state.get("participant_events", {}):
                                stored = st.session_state.participant_events[pid]
                                all_events = stored.get('events', []) + stored.get('manual', [])

                                # Find boundary events
                                boundaries = {}
                                for evt in all_events:
                                    canonical = evt.canonical if hasattr(evt, 'canonical') else None
                                    if canonical in ['measurement_start', 'measurement_end', 'pause_start', 'pause_end']:
                                        if evt.first_timestamp:
                                            boundaries[canonical] = evt.first_timestamp

                                if 'measurement_start' in boundaries:
                                    # Generate music events
                                    from rrational.prep.summaries import EventStatus
                                    from datetime import timedelta

                                    if 'music_events' not in stored:
                                        stored['music_events'] = []
                                    stored['music_events'] = []  # Clear existing

                                    playlist_data = st.session_state.playlist_groups.get(batch_playlist, {})
                                    music_order = playlist_data.get("music_order", ["music_1", "music_2", "music_3"])

                                    # Pre-pause period
                                    start = boundaries['measurement_start']
                                    end = boundaries.get('pause_start') or boundaries.get('measurement_end')
                                    if start and end:
                                        current = start
                                        idx = 0
                                        while current < end:
                                            music_type = music_order[idx % len(music_order)]
                                            next_time = current + timedelta(minutes=batch_interval)
                                            if next_time > end:
                                                next_time = end

                                            stored['music_events'].append(EventStatus(
                                                raw_label=f"{music_type}_start",
                                                canonical=f"{music_type}_start",
                                                first_timestamp=current,
                                                last_timestamp=current
                                            ))
                                            stored['music_events'].append(EventStatus(
                                                raw_label=f"{music_type}_end",
                                                canonical=f"{music_type}_end",
                                                first_timestamp=next_time,
                                                last_timestamp=next_time
                                            ))
                                            current = next_time
                                            idx += 1

                                    # Post-pause period
                                    if 'pause_end' in boundaries and 'measurement_end' in boundaries:
                                        start = boundaries['pause_end']
                                        end = boundaries['measurement_end']
                                        current = start
                                        idx = 0
                                        while current < end:
                                            music_type = music_order[idx % len(music_order)]
                                            next_time = current + timedelta(minutes=batch_interval)
                                            if next_time > end:
                                                next_time = end

                                            stored['music_events'].append(EventStatus(
                                                raw_label=f"{music_type}_start",
                                                canonical=f"{music_type}_start",
                                                first_timestamp=current,
                                                last_timestamp=current
                                            ))
                                            stored['music_events'].append(EventStatus(
                                                raw_label=f"{music_type}_end",
                                                canonical=f"{music_type}_end",
                                                first_timestamp=next_time,
                                                last_timestamp=next_time
                                            ))
                                            current = next_time
                                            idx += 1

                                    generated_count += 1

                        progress.progress(1.0)
                        status.empty()
                        show_toast(f"Generated music events for {generated_count} participants", icon="success")
                        st.rerun()
            else:
                st.info("Create playlist groups in the Group Management tab first")

        with batch_col2:
            st.markdown("##### Auto-Create Quality Events")
            st.caption("Create gap and variability events for all participants")

            batch_gap_threshold = st.number_input(
                "Gap threshold (seconds)",
                min_value=1.0, max_value=60.0, value=15.0, step=1.0,
                key="batch_gap_threshold",
                help="Create gap events when time between measurements exceeds this threshold"
            )

            batch_cv_threshold = st.number_input(
                "Variability CV threshold (%)",
                min_value=5.0, max_value=50.0, value=20.0, step=1.0,
                key="batch_cv_threshold",
                help="Create variability events for segments exceeding this CV"
            )

            if st.button("Detect Quality Issues for All", key="batch_detect_quality"):
                progress = st.progress(0)
                status = st.empty()
                total_gaps = 0
                total_var = 0

                participant_ids = [s.participant_id for s in st.session_state.summaries]

                for i, pid in enumerate(participant_ids):
                    status.text(f"Processing {pid}...")
                    progress.progress((i + 1) / len(participant_ids))

                    try:
                        # Find the summary for this participant
                        summary = next((s for s in st.session_state.summaries if s.participant_id == pid), None)
                        if not summary:
                            continue

                        # Get RR paths from summary
                        if hasattr(summary, 'rr_paths') and summary.rr_paths:
                            # Load cached recording data
                            events_paths = getattr(summary, 'events_paths', []) or []
                            recording_data = cached_load_recording(
                                tuple(str(p) for p in summary.rr_paths),
                                tuple(str(p) for p in events_paths),
                                pid
                            )

                            config_dict = {
                                "rr_min_ms": st.session_state.cleaning_config.rr_min_ms,
                                "rr_max_ms": st.session_state.cleaning_config.rr_max_ms,
                                "sudden_change_pct": st.session_state.cleaning_config.sudden_change_pct
                            }
                            rr_with_timestamps, _ = cached_clean_rr_intervals(
                                tuple(recording_data['rr_intervals']),
                                config_dict
                            )

                            if rr_with_timestamps:
                                timestamps, rr_values = zip(*rr_with_timestamps)
                                timestamps_list = list(timestamps)
                                rr_list = list(rr_values)

                                # Initialize events
                                if pid not in st.session_state.get("participant_events", {}):
                                    if "participant_events" not in st.session_state:
                                        st.session_state.participant_events = {}
                                    st.session_state.participant_events[pid] = {
                                        'events': list(summary.events) if hasattr(summary, 'events') else [],
                                        'manual': []
                                    }

                                stored = st.session_state.participant_events[pid]

                                # Detect and create gap events
                                gap_result = detect_time_gaps(timestamps_list, rr_values=rr_list, gap_threshold_s=batch_gap_threshold)
                                from rrational.prep.summaries import EventStatus

                                for gap in gap_result.get("gaps", []):
                                    stored['manual'].append(EventStatus(
                                        raw_label="gap_start",
                                        canonical="gap_start",
                                        first_timestamp=gap["start_time"],
                                        last_timestamp=gap["start_time"]
                                    ))
                                    stored['manual'].append(EventStatus(
                                        raw_label="gap_end",
                                        canonical="gap_end",
                                        first_timestamp=gap["end_time"],
                                        last_timestamp=gap["end_time"]
                                    ))
                                    total_gaps += 1

                                # Detect variability
                                cp_result = cached_quality_analysis(tuple(rr_values), tuple(timestamps))
                                cv_threshold_decimal = batch_cv_threshold / 100.0

                                for seg in cp_result.get("segment_stats", []):
                                    if seg.get("cv", 0) > cv_threshold_decimal:
                                        if seg.get("start_time") and seg.get("end_time"):
                                            stored['manual'].append(EventStatus(
                                                raw_label="high_variability_start",
                                                canonical="high_variability_start",
                                                first_timestamp=seg["start_time"],
                                                last_timestamp=seg["start_time"]
                                            ))
                                            stored['manual'].append(EventStatus(
                                                raw_label="high_variability_end",
                                                canonical="high_variability_end",
                                                first_timestamp=seg["end_time"],
                                                last_timestamp=seg["end_time"]
                                            ))
                                            total_var += 1

                    except Exception as e:
                        st.warning(f"Error processing {pid}: {e}")

                progress.progress(1.0)
                status.empty()
                show_toast(f"Detected {total_gaps} gaps and {total_var} high variability segments", icon="success")
                st.rerun()
