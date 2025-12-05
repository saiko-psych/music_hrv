"""Setup tab - Event mapping, Group management, and Sections.

Combines the Events, Groups, and Sections functionality into one
organized tab with nested sub-tabs.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from music_hrv.gui.persistence import (
    load_sections,
    load_playlist_groups,
    save_playlist_groups,
    save_music_labels,
)
from music_hrv.gui.shared import (
    auto_save_config,
    show_toast,
    update_normalizer,
    validate_regex_pattern,
)


def render_setup_tab():
    """Render the Setup tab with nested sub-tabs for Events, Groups, Playlists, Sections."""
    st.header("Setup")

    # Nested sub-tabs
    tab_events, tab_groups, tab_playlists, tab_sections = st.tabs(
        ["Events", "Groups", "Playlists", "Sections"]
    )

    with tab_events:
        _render_events_section()

    with tab_groups:
        _render_groups_section()

    with tab_playlists:
        _render_playlists_section()

    with tab_sections:
        _render_sections_section()


def _render_events_section():
    """Render the Events sub-section."""
    st.subheader("Event Mapping")

    with st.expander("Help - Event Mapping", expanded=False):
        st.markdown("""
        ### What are Events?

        Events are **markers in time** that define key moments in your HRV recording, such as:
        - `measurement_start` / `measurement_end` - Beginning and end of data collection
        - `rest_pre_start` / `rest_pre_end` - Pre-measurement rest period
        - `pause_start` / `pause_end` - Break between measurement blocks

        ### How Event Matching Works

        1. **Raw Label**: The label written in the HRV Logger (e.g., "Ruhe Pre Start")
        2. **Synonym Pattern**: A regex pattern to match variations (e.g., `ruhe[ _-]?pre[ _-]?start`)
        3. **Canonical Name**: The standardized internal name (e.g., `rest_pre_start`)

        The app automatically matches raw labels to canonical names using the synonym patterns you define.

        ### Tips for Synonyms

        - Use `[ _-]?` to match optional spaces, underscores, or hyphens
        - Use `.*` to match any characters (e.g., `start.*measurement`)
        - All matching is **case-insensitive** (lowercase automatically)
        """)

    st.info("All event matching is done in **lowercase** automatically to reduce the number of synonyms needed.")

    # Create new event
    with st.expander("Create New Event"):
        new_event_name = st.text_input("Event Name (canonical)", key="new_event_name_global")
        new_event_synonyms = st.text_area(
            "Synonyms (one per line, regex patterns supported)",
            key="new_event_synonyms_global",
            help="Enter regex patterns, one per line. All matching is lowercase. Example: ruhe[ _-]?pre[ _-]?start"
        )

        # Real-time validation of event name
        if new_event_name:
            if new_event_name in st.session_state.all_events:
                st.warning(f"Event '{new_event_name}' already exists")
            elif not new_event_name.replace("_", "").isalnum():
                st.warning("Event name should be alphanumeric with underscores")

        # Validate synonyms as regex patterns
        if new_event_synonyms:
            invalid_patterns = []
            for line in new_event_synonyms.split("\n"):
                if line.strip():
                    error = validate_regex_pattern(line.strip())
                    if error:
                        invalid_patterns.append(f"'{line.strip()}': {error}")
            if invalid_patterns:
                st.error("Invalid regex patterns:\n" + "\n".join(invalid_patterns))

        def create_event():
            """Callback to create new event."""
            if new_event_name and new_event_name not in st.session_state.all_events:
                synonyms_list = [s.strip().lower() for s in new_event_synonyms.split("\n") if s.strip()]
                st.session_state.all_events[new_event_name] = synonyms_list
                auto_save_config()
                update_normalizer()
                show_toast(f"Created event '{new_event_name}'", icon="success")
            elif new_event_name in st.session_state.all_events:
                show_toast(f"Event '{new_event_name}' already exists", icon="error")
            else:
                show_toast("Please enter an event name", icon="error")

        st.button("Create Event", key="create_event_btn_global", on_click=create_event, type="primary")

    st.markdown("---")

    # Show all events
    st.subheader("All Available Events")
    st.info(f"**{len(st.session_state.all_events)} event(s) defined**")

    if st.session_state.all_events:
        for event_name, synonyms in list(st.session_state.all_events.items()):
            with st.expander(f"Event: {event_name} ({len(synonyms)} synonym(s))", expanded=False):
                # Editable event name
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_event_name_edit = st.text_input(
                        "Event Name",
                        value=event_name,
                        key=f"edit_event_name_{event_name}"
                    )

                # Real-time validation
                name_valid = True
                if new_event_name_edit != event_name:
                    if new_event_name_edit in st.session_state.all_events:
                        st.warning(f"Event '{new_event_name_edit}' already exists")
                        name_valid = False
                    elif not new_event_name_edit.replace("_", "").isalnum():
                        st.warning("Event name should be alphanumeric with underscores")
                        name_valid = False

                with col2:
                    def rename_event(old_name, new_name):
                        """Callback to rename event."""
                        if new_name != old_name and new_name not in st.session_state.all_events:
                            st.session_state.all_events[new_name] = st.session_state.all_events.pop(old_name)
                            for group_data in st.session_state.groups.values():
                                if old_name in group_data.get("expected_events", {}):
                                    group_data["expected_events"][new_name] = group_data["expected_events"].pop(old_name)
                            auto_save_config()
                            update_normalizer()
                            show_toast(f"Renamed to '{new_name}'", icon="success")
                        elif new_name == old_name:
                            show_toast("Name unchanged", icon="info")
                        else:
                            show_toast(f"Event '{new_name}' already exists", icon="error")

                    st.button(
                        "Save Name",
                        key=f"save_event_name_{event_name}",
                        on_click=rename_event,
                        args=(event_name, new_event_name_edit),
                        disabled=not name_valid or new_event_name_edit == event_name,
                        use_container_width=True,
                    )

                # Show used in groups
                used_in_groups = [
                    gname for gname, gdata in st.session_state.groups.items()
                    if event_name in gdata.get("expected_events", {})
                ]
                if used_in_groups:
                    st.info(f"Used in groups: {', '.join(used_in_groups)}")
                else:
                    st.info("Not used in any groups yet")

                st.markdown("---")
                st.markdown("**Synonyms:**")

                # Display and manage synonyms
                if synonyms:
                    for syn_idx, synonym in enumerate(synonyms):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.text(synonym)
                        with col2:
                            def delete_synonym(evt_name, idx):
                                """Callback to delete synonym."""
                                syn_list = st.session_state.all_events[evt_name]
                                syn_list.pop(idx)
                                st.session_state.all_events[evt_name] = syn_list
                                for group_data in st.session_state.groups.values():
                                    if evt_name in group_data.get("expected_events", {}):
                                        group_data["expected_events"][evt_name] = syn_list.copy()
                                auto_save_config()
                                update_normalizer()
                                show_toast("Synonym deleted", icon="success")

                            st.button(
                                "X",
                                key=f"delete_syn_{event_name}_{syn_idx}",
                                on_click=delete_synonym,
                                args=(event_name, syn_idx),
                                help="Delete this synonym",
                            )
                else:
                    st.info("No synonyms defined")

                # Add new synonym
                st.markdown("**Add New Synonym:**")
                col1, col2 = st.columns([3, 1])
                with col1:
                    new_synonym = st.text_input(
                        "New synonym (regex pattern)",
                        key=f"new_syn_{event_name}",
                        placeholder="e.g., rest[ _-]?start"
                    )

                    if new_synonym:
                        error = validate_regex_pattern(new_synonym.strip())
                        if error:
                            st.error(f"Invalid regex: {error}")
                        elif new_synonym.strip().lower() in synonyms:
                            st.warning("This synonym already exists")

                with col2:
                    st.write("")
                    st.write("")

                    def add_synonym(evt_name, new_syn):
                        """Callback to add synonym."""
                        synonym_lower = new_syn.strip().lower()
                        syn_list = st.session_state.all_events[evt_name]
                        if synonym_lower and synonym_lower not in syn_list:
                            syn_list.append(synonym_lower)
                            st.session_state.all_events[evt_name] = syn_list
                            for group_data in st.session_state.groups.values():
                                if evt_name in group_data.get("expected_events", {}):
                                    group_data["expected_events"][evt_name] = syn_list.copy()
                            auto_save_config()
                            update_normalizer()
                            show_toast(f"Added '{synonym_lower}'", icon="success")
                        elif synonym_lower in syn_list:
                            show_toast("Synonym already exists", icon="warning")
                        else:
                            show_toast("Please enter a synonym", icon="error")

                    st.button(
                        "Add",
                        key=f"add_syn_btn_{event_name}",
                        on_click=add_synonym,
                        args=(event_name, new_synonym),
                        type="primary",
                        disabled=not new_synonym or validate_regex_pattern(new_synonym.strip()) is not None,
                    )

                # Delete event
                st.markdown("---")

                def delete_event(evt_name):
                    """Callback to delete event."""
                    del st.session_state.all_events[evt_name]
                    for group_data in st.session_state.groups.values():
                        if evt_name in group_data.get("expected_events", {}):
                            del group_data["expected_events"][evt_name]
                    auto_save_config()
                    update_normalizer()
                    show_toast(f"Deleted event '{evt_name}'", icon="success")

                st.button(
                    f"Delete Event '{event_name}'",
                    key=f"delete_event_{event_name}",
                    on_click=delete_event,
                    args=(event_name,),
                    type="secondary",
                )
    else:
        st.info("No events defined yet. Create events above.")


def _render_groups_section():
    """Render the Groups sub-section."""
    st.subheader("Group Management")

    with st.expander("Help - Groups & Playlists", expanded=False):
        st.markdown("""
        ### Study Groups

        Groups define **which events are expected** for each participant. For example:
        - **Control Group**: May only need `rest_pre`, `measurement`, `rest_post`
        - **Intervention Group**: May need additional events like `pause_start`, `pause_end`

        When you assign a participant to a group, the app will check if all expected events
        are present and warn you about missing ones.

        ### Playlist Groups (Music Randomization)

        If your study involves music interventions with different randomization orders:
        - **R1**: music_1 -> music_2 -> music_3
        - **R2**: music_1 -> music_3 -> music_2
        - etc.

        Assign participants to playlist groups. The playlist order is used when generating
        music section events in the **Participants** tab.
        """)

    st.markdown("Create groups, edit/rename/delete them, and assign events from the Event Mapping tab.")

    # Create new group
    with st.expander("Create New Group"):
        new_group_name = st.text_input("Group Name (internal ID)", key="new_group_name")
        new_group_label = st.text_input("Group Label (display name)", key="new_group_label")

        if new_group_name:
            if new_group_name in st.session_state.groups:
                st.warning(f"Group '{new_group_name}' already exists")
            elif not new_group_name.replace("_", "").replace("-", "").isalnum():
                st.warning("Group name should be alphanumeric with underscores/hyphens")

        def create_group():
            """Callback to create new group."""
            if new_group_name and new_group_name not in st.session_state.groups:
                st.session_state.groups[new_group_name] = {
                    "label": new_group_label or new_group_name,
                    "expected_events": {},
                    "selected_sections": []
                }
                auto_save_config()
                show_toast(f"Created group '{new_group_name}'", icon="success")
            elif new_group_name in st.session_state.groups:
                show_toast(f"Group '{new_group_name}' already exists", icon="error")
            else:
                show_toast("Please enter a group name", icon="error")

        st.button("Create Group", key="create_group_btn", on_click=create_group, type="primary")

    st.markdown("---")

    # Manage existing groups
    st.subheader("Existing Groups")

    for group_name, group_data in list(st.session_state.groups.items()):
        with st.expander(f"{group_name} - {group_data['label']}", expanded=(group_name == "Default")):

            st.markdown("**Edit Group:**")
            col1, col2 = st.columns(2)
            with col1:
                new_name = st.text_input(
                    "Group Name (ID)",
                    value=group_name,
                    key=f"edit_group_name_{group_name}"
                )
            with col2:
                new_label = st.text_input(
                    "Group Label",
                    value=group_data["label"],
                    key=f"edit_group_label_{group_name}"
                )

            def save_group_changes(old_name, new_name_val, new_label_val):
                """Callback to save group changes."""
                current_name = old_name
                if new_name_val != old_name:
                    st.session_state.groups[new_name_val] = st.session_state.groups.pop(old_name)
                    for pid, gname in st.session_state.participant_groups.items():
                        if gname == old_name:
                            st.session_state.participant_groups[pid] = new_name_val
                    current_name = new_name_val

                st.session_state.groups[current_name]["label"] = new_label_val
                auto_save_config()
                show_toast(f"Saved changes to '{current_name}'", icon="success")

            st.button(
                f"Save Changes to {group_name}",
                key=f"save_group_{group_name}",
                on_click=save_group_changes,
                args=(group_name, new_name, new_label),
                type="primary",
            )

            st.markdown("---")

            participant_count = sum(1 for g in st.session_state.participant_groups.values() if g == group_name)
            st.markdown(f"**Participants in this group:** {participant_count}")

            # Sections selection
            st.markdown("**Select Sections for Analysis:**")
            available_sections = list(st.session_state.sections.keys()) if hasattr(st.session_state, 'sections') else []
            if available_sections:
                def update_sections():
                    """Callback to update sections selection."""
                    st.session_state.groups[group_name]["selected_sections"] = st.session_state[f"sections_select_{group_name}"]
                    auto_save_config()
                    show_toast(f"Sections updated for {group_name}", icon="success")

                st.multiselect(
                    "Sections to use in analysis",
                    options=available_sections,
                    default=group_data.get("selected_sections", []),
                    key=f"sections_select_{group_name}",
                    help="Choose which sections to analyze for participants in this group (saves automatically)",
                    on_change=update_sections,
                )
            else:
                st.info("No sections defined yet. Create sections in the Sections tab first.")

            st.markdown("---")

            # Expected events
            st.markdown("**Select Expected Events:**")
            expected_events = group_data.get("expected_events", {})

            st.markdown("*Click events to add/remove from this group (saves automatically):*")

            num_cols = 3
            cols = st.columns(num_cols)

            available_event_names = list(st.session_state.all_events.keys())
            for idx, event_name in enumerate(available_event_names):
                col_idx = idx % num_cols
                with cols[col_idx]:
                    is_selected = event_name in expected_events

                    def toggle_event(grp_name, evt_name, currently_selected):
                        """Callback to toggle event selection."""
                        exp_events = st.session_state.groups[grp_name]["expected_events"]
                        if st.session_state[f"event_select_{grp_name}_{evt_name}"]:
                            if evt_name not in exp_events:
                                exp_events[evt_name] = st.session_state.all_events[evt_name].copy()
                                st.session_state.groups[grp_name]["expected_events"] = exp_events
                                auto_save_config()
                                show_toast(f"Added '{evt_name}' to {grp_name}", icon="success")
                        else:
                            if evt_name in exp_events:
                                del exp_events[evt_name]
                                st.session_state.groups[grp_name]["expected_events"] = exp_events
                                auto_save_config()
                                show_toast(f"Removed '{evt_name}' from {grp_name}", icon="info")

                    st.checkbox(
                        event_name,
                        value=is_selected,
                        key=f"event_select_{group_name}_{event_name}",
                        on_change=toggle_event,
                        args=(group_name, event_name, is_selected),
                    )

            st.markdown("---")

            # Show currently selected events
            if expected_events:
                st.markdown("**Currently Selected Events:**")
                events_list = []
                for event_name_item, synonyms in expected_events.items():
                    events_list.append({
                        "Event Name": event_name_item,
                        "Synonyms": ", ".join(synonyms[:3]) + ("..." if len(synonyms) > 3 else "") if synonyms else "No synonyms",
                    })

                df_group_events = pd.DataFrame(events_list)
                st.dataframe(df_group_events, use_container_width=True, hide_index=True)

                csv_group_events = df_group_events.to_csv(index=False)
                st.download_button(
                    label=f"Download Events for {group_name}",
                    data=csv_group_events,
                    file_name=f"group_events_{group_name}.csv",
                    mime="text/csv",
                    key=f"download_group_{group_name}"
                )
            else:
                st.info("No events selected for this group yet. Select events above.")

            # Delete group
            st.markdown("---")

            def delete_group(grp_name):
                """Callback to delete group."""
                for pid, gname in st.session_state.participant_groups.items():
                    if gname == grp_name:
                        st.session_state.participant_groups[pid] = "Default"
                del st.session_state.groups[grp_name]
                auto_save_config()
                show_toast(f"Deleted group '{grp_name}' and reassigned participants to Default", icon="success")

            st.button(
                f"Delete Group '{group_name}'",
                key=f"delete_group_{group_name}",
                on_click=delete_group,
                args=(group_name,),
                type="secondary",
            )

    st.markdown("---")
    st.info("**All changes save automatically** when you modify group settings or select events.")


def _render_playlists_section():
    """Render the Playlists sub-section for music randomization."""
    st.subheader("Playlist Groups (Music Randomization)")

    with st.expander("Help - Playlist Groups", expanded=False):
        st.markdown("""
        ### What are Playlist Groups?

        Playlist groups define the **music order** for randomization conditions in your study.
        Each participant can be assigned to a playlist group, which determines the order of
        music pieces they experience.

        ### Example

        If your study has 3 music pieces and 6 randomization conditions:
        - **playlist_01**: music_1 -> music_2 -> music_3
        - **playlist_02**: music_1 -> music_3 -> music_2
        - **playlist_03**: music_2 -> music_1 -> music_3
        - etc.

        ### How to Use

        1. Define playlist groups here with their music order
        2. Assign participants to playlist groups in the Data tab
        3. Use Music Section Analysis to analyze HRV by music type
        """)

    st.markdown("""
    Define music order for each randomization group. Participants assigned to a playlist group
    will have music events generated in the specified order.
    """)

    # Initialize playlist groups (already done at app startup, but ensure present)
    if "playlist_groups" not in st.session_state:
        loaded_playlist = load_playlist_groups()
        if loaded_playlist:
            st.session_state.playlist_groups = loaded_playlist
        else:
            st.session_state.playlist_groups = {}

    if "participant_playlists" not in st.session_state:
        st.session_state.participant_playlists = {}

    # Create new playlist group
    with st.expander("Create New Playlist Group"):
        new_playlist_name = st.text_input(
            "Playlist Group ID (e.g., playlist_06)",
            key="new_playlist_name"
        )
        new_playlist_label = st.text_input("Playlist Group Label", key="new_playlist_label")
        new_playlist_order = st.text_input(
            "Music Order (comma-separated, e.g., music_2, music_1, music_3)",
            key="new_playlist_order"
        )

        def create_playlist_group():
            if new_playlist_name and new_playlist_name not in st.session_state.playlist_groups:
                order_list = [m.strip() for m in new_playlist_order.split(",") if m.strip()]
                if not order_list:
                    order_list = ["music_1", "music_2", "music_3"]
                st.session_state.playlist_groups[new_playlist_name] = {
                    "label": new_playlist_label or new_playlist_name,
                    "music_order": order_list
                }
                save_playlist_groups(st.session_state.playlist_groups)
                show_toast(f"Created playlist group '{new_playlist_name}'", icon="success")
            elif new_playlist_name in st.session_state.playlist_groups:
                show_toast(f"Playlist group '{new_playlist_name}' already exists", icon="error")

        st.button("Create Playlist Group", key="create_playlist_btn", on_click=create_playlist_group)

    # Show existing playlist groups
    st.markdown("---")
    st.subheader("Existing Playlist Groups")

    if not st.session_state.playlist_groups:
        st.info("No playlist groups defined yet. Create one above.")
    else:
        for playlist_name, playlist_data in list(st.session_state.playlist_groups.items()):
            with st.expander(f"{playlist_name} - {playlist_data['label']}"):
                # Edit label
                new_label = st.text_input(
                    "Label",
                    value=playlist_data.get('label', playlist_name),
                    key=f"edit_playlist_label_{playlist_name}"
                )

                st.markdown(f"**Music Order:** {' -> '.join(playlist_data['music_order'])}")

                new_order = st.text_input(
                    "Edit Music Order (comma-separated)",
                    value=", ".join(playlist_data['music_order']),
                    key=f"edit_playlist_order_{playlist_name}"
                )

                col_pl1, col_pl2, col_pl3 = st.columns(3)
                with col_pl1:
                    def save_playlist_changes(pl_name, new_ord, new_lbl):
                        order_list = [m.strip() for m in new_ord.split(",") if m.strip()]
                        if order_list:
                            st.session_state.playlist_groups[pl_name]["music_order"] = order_list
                        st.session_state.playlist_groups[pl_name]["label"] = new_lbl
                        save_playlist_groups(st.session_state.playlist_groups)
                        show_toast(f"Updated '{pl_name}'", icon="success")

                    st.button(
                        "Save Changes",
                        key=f"save_playlist_{playlist_name}",
                        on_click=save_playlist_changes,
                        args=(playlist_name, new_order, new_label)
                    )

                with col_pl3:
                    def delete_playlist(pl_name):
                        del st.session_state.playlist_groups[pl_name]
                        for pid in list(st.session_state.participant_playlists.keys()):
                            if st.session_state.participant_playlists.get(pid) == pl_name:
                                del st.session_state.participant_playlists[pid]
                        # Also remove from participant_randomizations
                        for pid in list(st.session_state.get("participant_randomizations", {}).keys()):
                            if st.session_state.participant_randomizations.get(pid) == pl_name:
                                del st.session_state.participant_randomizations[pid]
                        save_playlist_groups(st.session_state.playlist_groups)
                        show_toast(f"Deleted playlist group '{pl_name}'", icon="success")

                    st.button(
                        "Delete",
                        key=f"delete_playlist_{playlist_name}",
                        on_click=delete_playlist,
                        args=(playlist_name,),
                        type="secondary"
                    )

                # Show participants assigned to this playlist
                participants_in_group = [
                    pid for pid, pl in st.session_state.get("participant_randomizations", {}).items()
                    if pl == playlist_name
                ]
                if participants_in_group:
                    st.markdown(f"**Participants:** {', '.join(participants_in_group)}")
                else:
                    st.caption("No participants assigned yet")

    # Music Item Labels Section
    st.markdown("---")
    st.subheader("Music Item Labels")
    st.markdown("""
    Define labels and descriptions for your music items (e.g., `music_1`, `music_2`).
    These labels will appear in exports and the codebook.
    """)

    # Collect all unique music items from playlist orders
    all_music_items = set()
    for pl_data in st.session_state.get("playlist_groups", {}).values():
        all_music_items.update(pl_data.get("music_order", []))
    all_music_items = sorted(all_music_items)

    if all_music_items:
        # Initialize music_labels if not present
        if "music_labels" not in st.session_state:
            st.session_state.music_labels = {}

        music_data = []
        for music_item in all_music_items:
            current_data = st.session_state.music_labels.get(music_item, {})
            music_data.append({
                "Code": music_item,
                "Label": current_data.get("label", music_item.replace("_", " ").title()),
                "Description": current_data.get("description", ""),
            })

        df_music = pd.DataFrame(music_data)

        edited_music = st.data_editor(
            df_music,
            use_container_width=True,
            hide_index=True,
            key="music_labels_table",
            column_config={
                "Code": st.column_config.TextColumn("Code", disabled=True, help="Internal identifier"),
                "Label": st.column_config.TextColumn("Label", help="Short display name"),
                "Description": st.column_config.TextColumn("Description", help="Full description (e.g., composer, piece name)", width="large"),
            }
        )

        def save_music_labels_callback():
            """Save music labels from the edited table."""
            for _, row in edited_music.iterrows():
                code = row["Code"]
                st.session_state.music_labels[code] = {
                    "label": row["Label"],
                    "description": row["Description"],
                }
            # Save to persistence (dedicated music_labels file)
            save_music_labels(st.session_state.music_labels)
            show_toast("Music labels saved", icon="success")

        st.button("Save Music Labels", key="save_music_labels_btn", on_click=save_music_labels_callback, type="primary")
    else:
        st.info("No music items defined yet. Add them to playlist groups above.")

    st.markdown("---")
    st.info("**All changes save automatically.** Playlist labels are used in the Data tab.")


def _render_sections_section():
    """Render the Sections sub-section."""
    st.subheader("Sections")

    with st.expander("Help - Sections", expanded=False):
        st.markdown("""
        ### What are Sections?

        Sections define **time ranges** between events for HRV analysis. Each section has:
        - **Code**: Internal identifier (e.g., `pre_pause`)
        - **Label**: Short display name (e.g., `Pre-Pause`)
        - **Start/End Events**: The events that mark the beginning and end
        - **Duration**: Expected duration in minutes
        - **Tolerance**: Acceptable deviation from expected duration

        ### Validation

        In the **Participants** tab, sections are validated:
        - ✅ Start and end events present
        - ✅ Duration within tolerance of expected

        ### Example

        | Code | Label | Start Event | End Event | Duration | Tolerance |
        |------|-------|-------------|-----------|----------|-----------|
        | pre_pause | Pre-Pause | measurement_start | pause_start | 90 min | 5 min |
        | rest_pre | Pre-Rest | rest_pre_start | rest_pre_end | 5 min | 1 min |
        """)

    st.markdown("Define time ranges (sections) between events for analysis.")

    # Initialize sections if not present
    if "sections" not in st.session_state:
        loaded_sections = load_sections()
        if not loaded_sections:
            # Default sections - end_events is a list (any of these events can end the section)
            st.session_state.sections = {
                "rest_pre": {"label": "Pre-Rest", "description": "Baseline rest period", "start_event": "rest_pre_start", "end_events": ["rest_pre_end"], "expected_duration_min": 5.0, "tolerance_min": 1.0},
                "pre_pause": {"label": "Pre-Pause", "description": "Music before pause", "start_event": "measurement_start", "end_events": ["pause_start"], "expected_duration_min": 90.0, "tolerance_min": 5.0},
                "post_pause": {"label": "Post-Pause", "description": "Music after pause", "start_event": "pause_end", "end_events": ["measurement_end"], "expected_duration_min": 90.0, "tolerance_min": 5.0},
                "rest_post": {"label": "Post-Rest", "description": "Post-measurement rest", "start_event": "rest_post_start", "end_events": ["rest_post_end"], "expected_duration_min": 5.0, "tolerance_min": 1.0},
            }
        else:
            # Migrate old format (end_event) to new format (end_events)
            for section_data in loaded_sections.values():
                if "end_event" in section_data and "end_events" not in section_data:
                    section_data["end_events"] = [section_data.pop("end_event")]
            st.session_state.sections = loaded_sections

    # Create new section
    with st.expander("Create New Section"):
        new_section_name = st.text_input("Section Code (internal ID)", key="new_section_name",
                                         help="e.g., music_01, rest_pre")
        new_section_label = st.text_input("Section Label (short name)", key="new_section_label",
                                          help="e.g., Music 1, Pre-Rest")
        new_section_desc = st.text_input("Description (detailed)", key="new_section_desc",
                                         help="e.g., Brandenburg Concerto No. 3 - Bach")

        available_events = list(st.session_state.all_events.keys())
        col1, col2 = st.columns(2)
        with col1:
            start_event = st.selectbox("Start Event", options=available_events, key="new_section_start")
        with col2:
            end_events = st.multiselect(
                "End Event(s)",
                options=available_events,
                default=[available_events[0]] if available_events else [],
                key="new_section_end",
                help="Select one or more events. Section ends when ANY of these events occurs."
            )

        col3, col4 = st.columns(2)
        with col3:
            expected_duration = st.number_input("Expected Duration (min)", min_value=0.0, max_value=300.0, value=5.0,
                                               key="new_section_duration", help="Expected section duration in minutes")
        with col4:
            tolerance = st.number_input("Tolerance (min)", min_value=0.0, max_value=60.0, value=1.0,
                                       key="new_section_tolerance", help="Acceptable deviation from expected duration")

        if new_section_name:
            if new_section_name in st.session_state.sections:
                st.warning(f"Section '{new_section_name}' already exists")
            elif not new_section_name.replace("_", "").isalnum():
                st.warning("Section name should be alphanumeric with underscores")

        def create_section():
            """Callback to create section."""
            if new_section_name and new_section_name not in st.session_state.sections:
                if not end_events:
                    show_toast("Please select at least one end event", icon="error")
                    return
                st.session_state.sections[new_section_name] = {
                    "label": new_section_label or new_section_name,
                    "description": new_section_desc or "",
                    "start_event": start_event,
                    "end_events": end_events,  # List of possible end events
                    "expected_duration_min": expected_duration,
                    "tolerance_min": tolerance,
                }
                auto_save_config()
                show_toast(f"Created section '{new_section_name}'", icon="success")
            elif new_section_name in st.session_state.sections:
                show_toast(f"Section '{new_section_name}' already exists", icon="error")
            else:
                show_toast("Please enter a section name", icon="error")

        st.button("Create Section", key="create_section_btn", on_click=create_section, type="primary")

    st.markdown("---")

    # Show all sections
    st.subheader("All Defined Sections")

    if st.session_state.sections:
        sections_list = []
        for section_name, section_data in st.session_state.sections.items():
            # Support both old (end_event) and new (end_events) format
            end_events = section_data.get("end_events", [])
            if not end_events and "end_event" in section_data:
                end_events = [section_data["end_event"]]
            sections_list.append({
                "Code": section_name,
                "Label": section_data.get("label", section_name),
                "Start Event": section_data.get("start_event", ""),
                "End Event(s)": ", ".join(end_events),  # Show as comma-separated
                "Duration (min)": section_data.get("expected_duration_min", 5.0),
                "Tolerance (min)": section_data.get("tolerance_min", 1.0),
            })

        df_sections = pd.DataFrame(sections_list)

        available_events = list(st.session_state.all_events.keys())
        edited_sections = st.data_editor(
            df_sections,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            key="sections_table",
            column_config={
                "Code": st.column_config.TextColumn("Code", help="Internal identifier", width="small"),
                "Label": st.column_config.TextColumn("Label", help="Short display name", width="medium"),
                "Start Event": st.column_config.SelectboxColumn("Start Event", options=available_events, required=True, width="medium"),
                "End Event(s)": st.column_config.TextColumn("End Event(s)", help="Comma-separated list of events (any can end section)", width="medium"),
                "Duration (min)": st.column_config.NumberColumn("Duration (min)", help="Expected duration in minutes", min_value=0.0, max_value=300.0, format="%.1f", width="small"),
                "Tolerance (min)": st.column_config.NumberColumn("Tolerance (min)", help="Acceptable deviation", min_value=0.0, max_value=60.0, format="%.1f", width="small"),
            }
        )

        def save_section_changes():
            """Callback to save section changes."""
            updated_sections = {}
            for _, row in edited_sections.iterrows():
                section_code = row["Code"]
                if section_code:  # Skip empty rows
                    # Parse comma-separated end events
                    end_events_str = row.get("End Event(s)", "")
                    end_events_list = [e.strip() for e in end_events_str.split(",") if e.strip()]
                    if not end_events_list:
                        end_events_list = ["measurement_end"]  # Fallback
                    updated_sections[section_code] = {
                        "label": row["Label"],
                        "description": "",  # Description removed from table view
                        "start_event": row["Start Event"],
                        "end_events": end_events_list,  # Store as list
                        "expected_duration_min": row.get("Duration (min)", 5.0),
                        "tolerance_min": row.get("Tolerance (min)", 1.0),
                    }

            st.session_state.sections = updated_sections
            auto_save_config()
            show_toast("Saved section changes", icon="success")

        st.button("Save Section Changes", key="save_sections_btn", on_click=save_section_changes, type="primary")

        csv_sections = df_sections.to_csv(index=False)
        st.download_button(
            label="Download Sections CSV",
            data=csv_sections,
            file_name="sections.csv",
            mime="text/csv",
            key="download_sections"
        )
    else:
        st.info("No sections defined yet. Create sections above.")

    # Codebook Export Section
    st.markdown("---")
    st.subheader("Codebook Export")
    st.markdown("Export all codes, labels, and descriptions for documentation.")

    def generate_codebook():
        """Generate a comprehensive codebook with all definitions."""
        codebook_data = []

        # Events
        for event_name, synonyms in st.session_state.get("all_events", {}).items():
            codebook_data.append({
                "Category": "Event",
                "Code": event_name,
                "Label": event_name.replace("_", " ").title(),
                "Description": f"Synonyms: {', '.join(synonyms[:3])}" if synonyms else "",
            })

        # Sections
        for section_code, section_data in st.session_state.get("sections", {}).items():
            codebook_data.append({
                "Category": "Section",
                "Code": section_code,
                "Label": section_data.get("label", section_code),
                "Description": section_data.get("description", ""),
            })

        # Groups
        for group_id, group_data in st.session_state.get("groups", {}).items():
            codebook_data.append({
                "Category": "Group",
                "Code": group_id,
                "Label": group_data.get("label", group_id),
                "Description": f"Expected events: {len(group_data.get('expected_events', {}))}",
            })

        # Playlist Groups
        for pl_id, pl_data in st.session_state.get("playlist_groups", {}).items():
            codebook_data.append({
                "Category": "Playlist",
                "Code": pl_id,
                "Label": pl_data.get("label", pl_id),
                "Description": f"Order: {' -> '.join(pl_data.get('music_order', []))}",
            })

        # Music Items
        for music_code, music_data in st.session_state.get("music_labels", {}).items():
            codebook_data.append({
                "Category": "Music",
                "Code": music_code,
                "Label": music_data.get("label", music_code),
                "Description": music_data.get("description", ""),
            })

        # Device Settings
        device_settings = st.session_state.get("default_device_settings", {})
        if device_settings:
            codebook_data.append({
                "Category": "Device",
                "Code": "recording_app",
                "Label": "Recording App",
                "Description": device_settings.get("recording_app", "HRV Logger"),
            })
            codebook_data.append({
                "Category": "Device",
                "Code": "device",
                "Label": "HR Sensor",
                "Description": device_settings.get("device", "Polar H10"),
            })
            codebook_data.append({
                "Category": "Device",
                "Code": "sampling_rate",
                "Label": "Sampling Rate",
                "Description": f"{device_settings.get('sampling_rate', 1000)} Hz",
            })

        return pd.DataFrame(codebook_data)

    if st.button("Generate Codebook", key="generate_codebook_btn"):
        df_codebook = generate_codebook()
        st.dataframe(df_codebook, use_container_width=True, hide_index=True)

        csv_codebook = df_codebook.to_csv(index=False)
        st.download_button(
            label="Download Codebook CSV",
            data=csv_codebook,
            file_name="codebook.csv",
            mime="text/csv",
            key="download_codebook"
        )
