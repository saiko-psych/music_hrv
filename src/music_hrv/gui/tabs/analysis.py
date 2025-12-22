"""Analysis tab - HRV analysis with NeuroKit2.

This module contains the render function for the Analysis tab.
Provides HRV metrics computation and visualization.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from music_hrv.gui.shared import (
    NEUROKIT_AVAILABLE,
    get_neurokit,
    get_matplotlib,
    get_participant_list,
    get_summary_dict,
    extract_section_rr_intervals,
    filter_exclusion_zones,
    detect_artifacts_fixpeaks,
    show_toast,
    cached_discover_recordings,
    cached_load_recording,
    cached_load_vns_recording,
)
from music_hrv.gui.help_text import ANALYSIS_HELP


def _get_exclusion_zones(participant_id: str) -> list[dict]:
    """Get exclusion zones for a participant from session state."""
    if 'participant_events' not in st.session_state:
        return []
    participant_data = st.session_state.participant_events.get(participant_id, {})
    return participant_data.get('exclusion_zones', [])


def _render_music_section_analysis():
    """Render the Music Section Analysis UI.

    Protocol-based analysis of 5-minute music sections with validation.
    """
    from music_hrv.analysis.music_sections import (
        ProtocolConfig,
        DurationMismatchStrategy,
        extract_music_sections,
        get_sections_by_music_type,
    )
    from music_hrv.gui.persistence import load_protocol, save_protocol

    st.markdown("""
    Analyze HRV metrics for each **5-minute music section** based on your protocol.
    This mode automatically extracts sections using measurement events and validates data quality.
    """)

    # Protocol Settings
    with st.expander("⚙️ Protocol Settings", expanded=False):
        protocol_data = load_protocol()

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            expected_duration = st.number_input(
                "Expected total duration (min)",
                min_value=30.0, max_value=180.0,
                value=float(protocol_data.get("expected_duration_min", 90.0)),
                step=5.0,
                key="protocol_expected_duration",
                help="Total expected duration of the measurement session"
            )
            section_length = st.number_input(
                "Section length (min)",
                min_value=1.0, max_value=15.0,
                value=float(protocol_data.get("section_length_min", 5.0)),
                step=1.0,
                key="protocol_section_length",
                help="Duration of each music section"
            )
            pre_pause_sections = st.number_input(
                "Pre-pause sections",
                min_value=1, max_value=20,
                value=int(protocol_data.get("pre_pause_sections", 9)),
                step=1,
                key="protocol_pre_pause",
                help="Number of music sections before the pause"
            )

        with col_p2:
            post_pause_sections = st.number_input(
                "Post-pause sections",
                min_value=1, max_value=20,
                value=int(protocol_data.get("post_pause_sections", 9)),
                step=1,
                key="protocol_post_pause",
                help="Number of music sections after the pause"
            )
            min_section_duration = st.number_input(
                "Minimum valid section duration (min)",
                min_value=1.0, max_value=10.0,
                value=float(protocol_data.get("min_section_duration_min", 4.0)),
                step=0.5,
                key="protocol_min_duration",
                help="Sections shorter than this are flagged as incomplete"
            )
            min_section_beats = st.number_input(
                "Minimum beats per section",
                min_value=50, max_value=500,
                value=int(protocol_data.get("min_section_beats", 100)),
                step=10,
                key="protocol_min_beats",
                help="Sections with fewer beats are flagged as incomplete"
            )

        # Duration mismatch handling
        mismatch_options = {
            "Flag only (include all, mark incomplete)": DurationMismatchStrategy.FLAG_ONLY,
            "Strict (exclude incomplete sections)": DurationMismatchStrategy.STRICT,
            "Proportional (scale sections to fit)": DurationMismatchStrategy.PROPORTIONAL,
        }
        current_strategy = protocol_data.get("mismatch_strategy", DurationMismatchStrategy.FLAG_ONLY)
        current_label = next(
            (k for k, v in mismatch_options.items() if v == current_strategy),
            "Flag only (include all, mark incomplete)"
        )
        mismatch_strategy = st.radio(
            "Duration mismatch handling",
            options=list(mismatch_options.keys()),
            index=list(mismatch_options.keys()).index(current_label),
            key="protocol_mismatch_strategy",
            horizontal=True,
            help="How to handle recordings that don't match expected duration"
        )

        if st.button("💾 Save Protocol Settings", key="save_protocol_btn"):
            new_protocol = {
                "expected_duration_min": expected_duration,
                "section_length_min": section_length,
                "pre_pause_sections": pre_pause_sections,
                "post_pause_sections": post_pause_sections,
                "min_section_duration_min": min_section_duration,
                "min_section_beats": min_section_beats,
                "mismatch_strategy": mismatch_options[mismatch_strategy],
            }
            save_protocol(new_protocol)
            st.success("Protocol settings saved!")

    # Build protocol config from current values
    protocol = ProtocolConfig(
        expected_duration_min=st.session_state.get("protocol_expected_duration", 90.0),
        section_length_min=st.session_state.get("protocol_section_length", 5.0),
        pre_pause_sections=st.session_state.get("protocol_pre_pause", 9),
        post_pause_sections=st.session_state.get("protocol_post_pause", 9),
        min_section_duration_min=st.session_state.get("protocol_min_duration", 4.0),
        min_section_beats=st.session_state.get("protocol_min_beats", 100),
    )

    st.markdown("---")

    # Participant/Playlist selection
    col_sel1, col_sel2 = st.columns(2)

    with col_sel1:
        participant_list = get_participant_list()
        selected_participant = st.selectbox(
            "Select Participant",
            options=participant_list,
            key="music_analysis_participant"
        )

    with col_sel2:
        # Get participant's playlist
        participant_playlist = st.session_state.get("participant_playlists", {}).get(selected_participant, "")
        playlist_groups = st.session_state.get("playlist_groups", {})

        if participant_playlist and participant_playlist in playlist_groups:
            playlist_data = playlist_groups[participant_playlist]
            music_order = playlist_data.get("music_order", ["music_1", "music_2", "music_3"])
            playlist_label = playlist_data.get("label", participant_playlist)
            st.info(f"**Playlist:** {playlist_label}")
            st.caption(f"Music order: {' → '.join(music_order)}")
        else:
            st.warning("No playlist assigned. Using default music order.")
            music_order = ["music_1", "music_2", "music_3"]

    # Artifact correction option
    apply_correction = st.checkbox(
        "Apply artifact correction (NeuroKit2 Kubios)",
        value=False,
        key="music_analysis_correction",
        help="Recommended for data with quality issues"
    )

    # Analyze button
    if st.button("🎵 Analyze Music Sections", key="analyze_music_btn", type="primary"):
        with st.status("Extracting music sections...", expanded=True) as status:
            try:
                st.write("📂 Loading recording data...")

                # Get participant's recording data
                summary = get_summary_dict().get(selected_participant)
                if not summary:
                    st.error(f"No data found for participant {selected_participant}")
                    return

                source_app = getattr(summary, 'source_app', 'HRV Logger')
                is_vns = (source_app == "VNS Analyse")

                # Load recording
                if is_vns and getattr(summary, 'vns_path', None):
                    recording_data = cached_load_vns_recording(
                        str(summary.vns_path),
                        selected_participant,
                        use_corrected=st.session_state.get("vns_use_corrected", False),
                    )
                else:
                    bundles = cached_discover_recordings(st.session_state.data_dir, st.session_state.id_pattern)
                    bundle = next((b for b in bundles if b.participant_id == selected_participant), None)
                    if not bundle:
                        st.error(f"No recording bundle found for {selected_participant}")
                        return
                    recording_data = cached_load_recording(
                        tuple(str(p) for p in bundle.rr_paths),
                        tuple(str(p) for p in bundle.events_paths),
                        selected_participant
                    )

                # Build RR intervals and events dict
                from music_hrv.io.hrv_logger import RRInterval
                rr_intervals = [
                    RRInterval(timestamp=ts, rr_ms=rr, elapsed_ms=elapsed)
                    for ts, rr, elapsed in recording_data['rr_intervals']
                ]

                # Build events dictionary (canonical -> timestamp)
                events_dict = {}
                stored_events = st.session_state.participant_events.get(selected_participant, {})
                all_events = stored_events.get('events', []) + stored_events.get('manual', [])

                for evt in all_events:
                    canonical = evt.canonical if hasattr(evt, 'canonical') else None
                    if canonical and evt.first_timestamp:
                        events_dict[canonical] = evt.first_timestamp

                st.write(f"📊 Found {len(rr_intervals)} RR intervals")
                st.write(f"📌 Events: {', '.join(events_dict.keys()) or 'None'}")

                # Extract music sections
                st.write("🎵 Extracting music sections...")
                mismatch_strategy_value = mismatch_options.get(
                    st.session_state.get("protocol_mismatch_strategy", "Flag only (include all, mark incomplete)"),
                    DurationMismatchStrategy.FLAG_ONLY
                )

                analysis = extract_music_sections(
                    rr_intervals=rr_intervals,
                    events=events_dict,
                    music_order=music_order,
                    protocol=protocol,
                    mismatch_strategy=mismatch_strategy_value,
                )

                # Show warnings
                if analysis.warnings:
                    for warning in analysis.warnings:
                        st.warning(f"⚠️ {warning}")

                st.write(f"✅ Extracted {len(analysis.sections)} sections "
                        f"({analysis.valid_sections} valid, {analysis.incomplete_sections} incomplete)")

                status.update(label="Section extraction complete", state="complete")

                # Display results
                st.markdown("---")
                st.subheader("📊 Music Section Analysis Results")

                # Duration overview
                col_dur1, col_dur2, col_dur3 = st.columns(3)
                with col_dur1:
                    st.metric(
                        "Expected Duration",
                        f"{protocol.expected_duration_min:.0f} min"
                    )
                with col_dur2:
                    st.metric(
                        "Actual Duration",
                        f"{analysis.actual_total_duration_s/60:.1f} min",
                        delta=f"{-analysis.duration_mismatch_s/60:.1f} min" if analysis.duration_mismatch_s > 60 else None,
                        delta_color="inverse"
                    )
                with col_dur3:
                    st.metric(
                        "Valid Sections",
                        f"{analysis.valid_sections}/{len(analysis.sections)}"
                    )

                # Section details table
                st.markdown("### Section Details")

                section_data = []
                for section in analysis.sections:
                    status_icon = "✅" if section.is_valid else "⚠️"
                    section_data.append({
                        "Status": status_icon,
                        "Section": section.label,
                        "Music": section.music_type,
                        "Phase": section.phase.replace("_", " ").title(),
                        "Duration (min)": f"{section.actual_duration_s/60:.1f}",
                        "Beats": section.beat_count,
                        "Duration %": f"{section.duration_ratio*100:.0f}%",
                        "Warnings": "; ".join(section.validation_warnings) if section.validation_warnings else "-",
                    })

                df_sections = pd.DataFrame(section_data)
                st.dataframe(df_sections, use_container_width=True, hide_index=True)

                # HRV Analysis for valid sections
                st.markdown("### HRV Metrics by Section")

                nk = get_neurokit()
                if nk is None:
                    st.error("NeuroKit2 not available for HRV computation")
                    return

                hrv_results = []
                for section in analysis.sections:
                    if not section.is_valid or section.beat_count < 50:
                        continue

                    rr_values = [rr.rr_ms for rr in section.rr_intervals]

                    # Apply artifact correction if requested
                    if apply_correction:
                        try:
                            peaks_corrected, info = nk.signal_fixpeaks(
                                {"ECG_R_Peaks": list(range(len(rr_values)))},
                                sampling_rate=1000,
                                iterative=True,
                                method="kubios"
                            )
                            # Reconstruct RR from corrected peaks
                            rr_values = [rr_values[i] for i in range(len(rr_values))
                                        if i not in info.get("artifacts", [])]
                        except Exception:
                            pass  # Use original if correction fails

                    try:
                        # Convert RR intervals to peaks for NeuroKit2
                        peaks = nk.intervals_to_peaks(rr_values, sampling_rate=1000)

                        # Compute HRV metrics using peaks
                        hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
                        hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)

                        hrv_results.append({
                            "Section": section.label,
                            "Music": section.music_type,
                            "Phase": section.phase.replace("_", " ").title(),
                            "Beats": section.beat_count,
                            "RMSSD": f"{hrv_time['HRV_RMSSD'].values[0]:.1f}",
                            "SDNN": f"{hrv_time['HRV_SDNN'].values[0]:.1f}",
                            "pNN50": f"{hrv_time['HRV_pNN50'].values[0]:.1f}",
                            "HF (ms²)": f"{hrv_freq['HRV_HF'].values[0]:.1f}",
                            "LF (ms²)": f"{hrv_freq['HRV_LF'].values[0]:.1f}",
                            "LF/HF": f"{hrv_freq['HRV_LFHF'].values[0]:.2f}",
                        })
                    except Exception as e:
                        st.warning(f"Could not compute HRV for {section.label}: {e}")

                if hrv_results:
                    df_hrv = pd.DataFrame(hrv_results)
                    st.dataframe(df_hrv, use_container_width=True, hide_index=True)

                    # Download button
                    csv_hrv = df_hrv.to_csv(index=False)
                    st.download_button(
                        "📥 Download HRV Results (CSV)",
                        data=csv_hrv,
                        file_name=f"music_sections_hrv_{selected_participant}.csv",
                        mime="text/csv"
                    )

                    # Summary by music type
                    st.markdown("### Summary by Music Type")
                    sections_by_type = get_sections_by_music_type(analysis, valid_only=True)

                    for music_type, sections in sections_by_type.items():
                        with st.expander(f"🎵 {music_type} ({len(sections)} sections)", expanded=False):
                            type_results = [r for r in hrv_results if r["Music"] == music_type]
                            if type_results:
                                df_type = pd.DataFrame(type_results)
                                st.dataframe(df_type, use_container_width=True, hide_index=True)

                                # Compute averages
                                try:
                                    avg_rmssd = sum(float(r["RMSSD"]) for r in type_results) / len(type_results)
                                    avg_sdnn = sum(float(r["SDNN"]) for r in type_results) / len(type_results)
                                    st.markdown(f"**Averages:** RMSSD={avg_rmssd:.1f} ms, SDNN={avg_sdnn:.1f} ms")
                                except (ValueError, ZeroDivisionError):
                                    pass

                else:
                    st.warning("No valid sections for HRV analysis")

            except Exception as e:
                status.update(label="Error during analysis", state="error")
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())


def render_analysis_tab():
    """Render the Analysis tab content.

    This tab contains:
    - Individual participant HRV analysis
    - Music section analysis (protocol-based)
    - Group-level HRV analysis
    """
    st.header("HRV Analysis")

    with st.expander("📖 Help - HRV Analysis & Scientific Best Practices", expanded=False):
        st.markdown(ANALYSIS_HELP)

    if not NEUROKIT_AVAILABLE:
        st.error("❌ NeuroKit2 is not installed. Please install it to use HRV analysis features.")
        st.code("uv add neurokit2")
        return

    if not st.session_state.summaries:
        st.info("📊 Load data from the 'Data & Groups' tab to perform analysis")
    else:
        st.markdown("Select a participant, choose multiple sections, and analyze HRV metrics for each section individually and combined.")

        # Initialize analysis results in session state
        if "analysis_results" not in st.session_state:
            st.session_state.analysis_results = {}

        # Selection mode
        analysis_mode = st.radio(
            "Analysis Mode",
            options=["Single Participant", "Music Section Analysis", "Group Analysis"],
            horizontal=True,
        )

        if analysis_mode == "Single Participant":
            _render_single_participant_analysis()

        elif analysis_mode == "Music Section Analysis":
            _render_music_section_analysis()

        else:  # Group Analysis
            _render_group_analysis()


def _render_single_participant_analysis():
    """Render single participant HRV analysis."""
    from music_hrv.cleaning.rr import clean_rr_intervals, RRInterval
    from music_hrv.io.hrv_logger import HRVLoggerRecording, EventMarker

    # Participant selection
    participant_list = get_participant_list()
    selected_participant = st.selectbox(
        "Select Participant",
        options=participant_list,
        key="analysis_participant"
    )

    # Section selection
    available_sections = list(st.session_state.sections.keys())
    if not available_sections:
        st.warning("⚠️ No sections defined. Please define sections in the Sections tab first.")
        return

    selected_sections = st.multiselect(
        "Select Sections to Analyze",
        options=available_sections,
        default=[available_sections[0]] if available_sections else [],
        key="analysis_sections_single"
    )

    # Artifact correction options
    with st.expander("🔧 Artifact Correction (signal_fixpeaks)", expanded=False):
        st.markdown("""
        Uses NeuroKit2's `signal_fixpeaks()` with the **Kubios algorithm** to detect and correct:
        - **Ectopic beats** (premature/delayed beats)
        - **Missed beats** (undetected R-peaks)
        - **Extra beats** (false positive detections)
        - **Long/short intervals** (physiologically implausible)
        """)
        apply_artifact_correction = st.checkbox(
            "Apply artifact correction before HRV analysis",
            value=False,
            key="apply_artifact_correction",
            help="Recommended for data with known quality issues"
        )

    if st.button("🔬 Analyze HRV", key="analyze_single_btn", type="primary"):
        if not selected_sections:
            st.error("Please select at least one section")
        else:
            # Use status context for multi-step analysis
            with st.status("Analyzing HRV for selected sections...", expanded=True) as status:
                try:
                    st.write("📂 Loading recording data...")
                    progress = st.progress(0)

                    # Check source type from summary
                    summary = get_summary_dict().get(selected_participant)
                    source_app = getattr(summary, 'source_app', 'HRV Logger') if summary else 'HRV Logger'
                    is_vns = (source_app == "VNS Analyse")

                    if is_vns and getattr(summary, 'vns_path', None):
                        # Load VNS recording
                        recording_data = cached_load_vns_recording(
                            str(summary.vns_path),
                            selected_participant,
                            use_corrected=st.session_state.get("vns_use_corrected", False),
                        )
                    else:
                        # Load HRV Logger recording
                        bundles = cached_discover_recordings(st.session_state.data_dir, st.session_state.id_pattern)
                        bundle = next(b for b in bundles if b.participant_id == selected_participant)
                        recording_data = cached_load_recording(
                            tuple(str(p) for p in bundle.rr_paths),
                            tuple(str(p) for p in bundle.events_paths),
                            selected_participant
                        )

                    # Reconstruct recording object from cached data
                    rr_intervals = [RRInterval(timestamp=ts, rr_ms=rr, elapsed_ms=elapsed)
                                    for ts, rr, elapsed in recording_data['rr_intervals']]
                    events = [EventMarker(label=label, timestamp=ts, offset_s=None)
                              for label, ts in recording_data['events']]
                    recording = HRVLoggerRecording(
                        participant_id=selected_participant,
                        rr_intervals=rr_intervals,
                        events=events
                    )
                    progress.progress(20)

                    # Store results for each section
                    section_results = {}
                    combined_rr = []

                    st.write(f"🔬 Analyzing {len(selected_sections)} section(s)...")

                    # Analyze each section individually
                    for idx, section_name in enumerate(selected_sections):
                        progress.progress(20 + int((idx / len(selected_sections)) * 60))
                        st.write(f"  • Processing section: {section_name}")

                        section_def = st.session_state.sections[section_name]
                        section_rr = extract_section_rr_intervals(
                            recording, section_def, st.session_state.normalizer
                        )

                        if section_rr:
                            # Apply exclusion zone filtering
                            exclusion_zones = _get_exclusion_zones(selected_participant)
                            if exclusion_zones:
                                section_rr, excl_stats = filter_exclusion_zones(section_rr, exclusion_zones)
                                if excl_stats["n_excluded"] > 0:
                                    st.write(f"    🚫 Excluded {excl_stats['n_excluded']} intervals ({excl_stats['excluded_duration_ms']/1000:.1f}s) from {excl_stats['zones_applied']} zone(s)")

                            # Clean RR intervals for this section
                            cleaned_section_rr, stats = clean_rr_intervals(
                                section_rr, st.session_state.cleaning_config
                            )

                            if cleaned_section_rr:
                                rr_ms = [rr.rr_ms for rr in cleaned_section_rr]

                                # Apply artifact correction if enabled
                                artifact_info = None
                                if apply_artifact_correction:
                                    st.write("    🔧 Applying artifact correction...")
                                    artifact_result = detect_artifacts_fixpeaks(rr_ms)
                                    if artifact_result["correction_applied"]:
                                        rr_ms = artifact_result["corrected_rr"]
                                        artifact_info = artifact_result
                                        st.write(f"    ✓ Corrected {artifact_result['total_artifacts']} artifacts")

                                combined_rr.extend(rr_ms)

                                # Calculate HRV metrics
                                nk = get_neurokit()
                                peaks = nk.intervals_to_peaks(rr_ms, sampling_rate=1000)
                                hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
                                hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)
                                hrv_results = pd.concat([hrv_time, hrv_freq], axis=1)

                                section_results[section_name] = {
                                    "hrv_results": hrv_results,
                                    "rr_intervals": rr_ms,
                                    "n_beats": len(rr_ms),
                                    "label": section_def.get("label", section_name),
                                    "artifact_info": artifact_info,
                                }
                        else:
                            st.write(f"  ⚠️ Could not find events for section '{section_name}'")

                    # Analyze combined sections if multiple selected
                    if len(selected_sections) > 1 and combined_rr:
                        progress.progress(80)
                        st.write("📊 Computing combined analysis...")
                        nk = get_neurokit()
                        peaks = nk.intervals_to_peaks(combined_rr, sampling_rate=1000)
                        hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
                        hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)
                        combined_hrv = pd.concat([hrv_time, hrv_freq], axis=1)
                        section_results["_combined"] = {
                            "hrv_results": combined_hrv,
                            "rr_intervals": combined_rr,
                            "n_beats": len(combined_rr),
                            "label": "Combined Sections",
                        }

                    # Store in session state
                    progress.progress(100)
                    st.session_state.analysis_results[selected_participant] = section_results

                    status.update(label=f"✅ Analysis complete for {len(section_results)} section(s)!", state="complete")
                    show_toast(f"Analysis complete for {len(section_results)} section(s)", icon="success")

                except Exception as e:
                    status.update(label="❌ Error during analysis", state="error")
                    st.error(f"Error during analysis: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # Display results if available
    if selected_participant in st.session_state.analysis_results:
        _display_single_participant_results(selected_participant)


def _display_single_participant_results(selected_participant: str):
    """Display HRV analysis results for a single participant."""
    st.markdown("---")
    st.subheader(f"📊 Results for {selected_participant}")

    section_results = st.session_state.analysis_results[selected_participant]

    for section_name, result_data in section_results.items():
        section_label = result_data["label"]
        hrv_results = result_data["hrv_results"]
        rr_intervals = result_data["rr_intervals"]
        n_beats = result_data["n_beats"]
        artifact_info = result_data.get("artifact_info")

        with st.expander(f"📈 {section_label} ({n_beats} beats)", expanded=True):
            # Show artifact correction info if applied
            if artifact_info:
                st.info(f"🔧 **Artifact Correction Applied**: {artifact_info['total_artifacts']} artifacts corrected "
                       f"({artifact_info['artifact_ratio']*100:.1f}% of beats)")
                art = artifact_info['artifacts']
                col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                with col_a1:
                    st.metric("Ectopic", art['ectopic'])
                with col_a2:
                    st.metric("Missed", art['missed'])
                with col_a3:
                    st.metric("Extra", art['extra'])
                with col_a4:
                    st.metric("Long/Short", art['longshort'])

            # Key metrics
            if not hrv_results.empty:
                metrics_to_show = {
                    "HRV_RMSSD": "RMSSD (ms)",
                    "HRV_SDNN": "SDNN (ms)",
                    "HRV_pNN50": "pNN50 (%)",
                    "HRV_HF": "HF Power",
                    "HRV_LF": "LF Power",
                    "HRV_LFHF": "LF/HF Ratio",
                }

                cols = st.columns(3)
                for idx, (col_name, display_name) in enumerate(metrics_to_show.items()):
                    if col_name in hrv_results.columns:
                        value = hrv_results[col_name].iloc[0]
                        with cols[idx % 3]:
                            st.metric(display_name, f"{value:.2f}")

                # Full results table
                st.markdown("**Full HRV Metrics:**")
                st.dataframe(hrv_results.T, use_container_width=True)

                # Download button for this section
                csv_hrv = hrv_results.to_csv(index=True)
                st.download_button(
                    label=f"📥 Download {section_label} Results",
                    data=csv_hrv,
                    file_name=f"hrv_{selected_participant}_{section_name}.csv",
                    mime="text/csv",
                    key=f"download_{selected_participant}_{section_name}",
                )

                # RR interval plot for this section
                st.markdown("**RR Interval Plot:**")
                plt = get_matplotlib()
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(rr_intervals, marker='o', markersize=2, linestyle='-', linewidth=0.5)
                ax.set_xlabel("Beat Index")
                ax.set_ylabel("RR Interval (ms)")
                ax.set_title(f"Tachogram - {section_label}")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)


def _render_group_analysis():
    """Render group-level HRV analysis."""
    from music_hrv.cleaning.rr import clean_rr_intervals, RRInterval
    from music_hrv.io.hrv_logger import HRVLoggerRecording, EventMarker

    # Group selection
    group_list = list(st.session_state.groups.keys())
    selected_group = st.selectbox(
        "Select Group",
        options=group_list,
        key="analysis_group"
    )

    # Section selection
    available_sections = list(st.session_state.sections.keys())
    if not available_sections:
        st.warning("⚠️ No sections defined. Please define sections in the Sections tab first.")
        return

    selected_sections = st.multiselect(
        "Select Sections to Analyze",
        options=available_sections,
        default=[available_sections[0]] if available_sections else [],
        key="analysis_sections_group"
    )

    if st.button("🔬 Analyze Group HRV", key="analyze_group_btn", type="primary"):
        if not selected_sections:
            st.error("Please select at least one section")
        else:
            # Get participants in selected group
            group_participants = [
                pid for pid, gname in st.session_state.participant_groups.items()
                if gname == selected_group
            ]

            if not group_participants:
                st.warning(f"No participants assigned to group '{selected_group}'")
            else:
                # Use status context for group analysis
                with st.status(f"Analyzing {len(group_participants)} participants...", expanded=True) as status:
                    bundles = cached_discover_recordings(st.session_state.data_dir, st.session_state.id_pattern)

                    # Results organized by section
                    results_by_section = {section: [] for section in selected_sections}
                    if len(selected_sections) > 1:
                        results_by_section["_combined"] = []

                    progress = st.progress(0)
                    total_steps = len(group_participants)

                    for idx, participant_id in enumerate(group_participants):
                        st.write(f"📊 Processing {participant_id} ({idx + 1}/{total_steps})")
                        progress.progress(int((idx / total_steps) * 100))
                        try:
                            bundle = next(b for b in bundles if b.participant_id == participant_id)
                            # Use CACHED loading
                            recording_data = cached_load_recording(
                                tuple(str(p) for p in bundle.rr_paths),
                                tuple(str(p) for p in bundle.events_paths),
                                participant_id
                            )
                            # Reconstruct recording object
                            rr_intervals = [RRInterval(timestamp=ts, rr_ms=rr, elapsed_ms=elapsed)
                                            for ts, rr, elapsed in recording_data['rr_intervals']]
                            events = [EventMarker(label=label, timestamp=ts, offset_s=None)
                                      for label, ts in recording_data['events']]
                            recording = HRVLoggerRecording(
                                participant_id=participant_id,
                                rr_intervals=rr_intervals,
                                events=events
                            )

                            combined_rr = []

                            # Analyze each section
                            for section_name in selected_sections:
                                section_def = st.session_state.sections[section_name]
                                section_rr = extract_section_rr_intervals(
                                    recording, section_def, st.session_state.normalizer
                                )

                                if section_rr:
                                    # Apply exclusion zone filtering
                                    exclusion_zones = _get_exclusion_zones(participant_id)
                                    if exclusion_zones:
                                        section_rr, _ = filter_exclusion_zones(section_rr, exclusion_zones)

                                    cleaned_rr, stats = clean_rr_intervals(
                                        section_rr, st.session_state.cleaning_config
                                    )

                                    if cleaned_rr:
                                        rr_ms = [rr.rr_ms for rr in cleaned_rr]
                                        combined_rr.extend(rr_ms)

                                        nk = get_neurokit()
                                        peaks = nk.intervals_to_peaks(rr_ms, sampling_rate=1000)
                                        hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
                                        hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)
                                        hrv_results = pd.concat([hrv_time, hrv_freq], axis=1)

                                        if not hrv_results.empty:
                                            result_row = {"participant_id": participant_id}
                                            for col in hrv_results.columns:
                                                result_row[col] = hrv_results[col].iloc[0]
                                            results_by_section[section_name].append(result_row)

                            # Combined analysis
                            if len(selected_sections) > 1 and combined_rr:
                                nk = get_neurokit()
                                peaks = nk.intervals_to_peaks(combined_rr, sampling_rate=1000)
                                hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
                                hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)
                                combined_hrv = pd.concat([hrv_time, hrv_freq], axis=1)

                                if not combined_hrv.empty:
                                    result_row = {"participant_id": participant_id}
                                    for col in combined_hrv.columns:
                                        result_row[col] = combined_hrv[col].iloc[0]
                                    results_by_section["_combined"].append(result_row)

                        except Exception as e:
                            st.write(f"  ⚠️ Could not analyze {participant_id}: {e}")

                    # Complete
                    progress.progress(100)
                    status.update(label="✅ Group analysis complete!", state="complete")
                    show_toast(f"Group analysis complete for {len(group_participants)} participants", icon="success")

                    # Display results by section
                    st.subheader(f"Group HRV Results - {selected_group}")

                    for section_name, results in results_by_section.items():
                        if results:
                            section_label = (
                                "Combined Sections"
                                if section_name == "_combined"
                                else st.session_state.sections[section_name].get("label", section_name)
                            )

                            with st.expander(f"📊 {section_label} ({len(results)} participants)", expanded=True):
                                df_results = pd.DataFrame(results)

                                # Summary statistics
                                st.markdown("**Summary Statistics:**")
                                st.dataframe(df_results.describe(), use_container_width=True)

                                # Individual results
                                st.markdown("**Individual Results:**")
                                st.dataframe(df_results, use_container_width=True)

                                # Download
                                csv_data = df_results.to_csv(index=False)
                                st.download_button(
                                    label=f"📥 Download {section_label} Results",
                                    data=csv_data,
                                    file_name=f"hrv_group_{selected_group}_{section_name}.csv",
                                    mime="text/csv",
                                    key=f"download_group_{section_name}",
                                )
