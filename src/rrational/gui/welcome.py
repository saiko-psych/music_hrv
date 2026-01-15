"""Welcome screen and project selection UI for RRational.

This module provides the welcome screen shown when the app starts
without a project selected, allowing users to:
- Open recent projects
- Create a new project
- Open an existing project folder
- Continue without a project (temporary workspace)
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from rrational.gui.project import (
    ProjectManager,
    add_recent_project,
    get_global_config_summary,
    get_recent_projects,
    has_global_config,
    migrate_global_config_to_project,
    remove_recent_project,
)


def _open_folder_dialog(title: str = "Select Folder", initial_dir: str | None = None) -> str | None:
    """Open a native folder picker dialog using subprocess.

    Uses subprocess to avoid blocking Streamlit's event loop.

    Args:
        title: Dialog title
        initial_dir: Initial directory to show

    Returns:
        Selected folder path or None if cancelled
    """
    import subprocess
    import sys

    # Set initial directory - use home folder as default (works on all OS)
    if initial_dir is None or not Path(initial_dir).exists():
        initial_dir = str(Path.home())

    # Python code to run in subprocess
    script = f'''
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
root.wm_attributes("-topmost", True)
root.focus_force()
root.update()

folder = filedialog.askdirectory(
    parent=root,
    title="{title}",
    initialdir=r"{initial_dir}",
)

root.destroy()

if folder:
    print(folder)
'''

    try:
        # Run in subprocess - this won't block Streamlit
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout for user to select
        )

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        return None

    except subprocess.TimeoutExpired:
        st.warning("Folder selection timed out")
        return None
    except Exception as e:
        st.error(f"Could not open folder dialog: {e}")
        return None


def _trigger_folder_dialog(session_key: str, title: str, initial_dir: str | None = None) -> bool:
    """Trigger a folder dialog and store result in session state.

    Args:
        session_key: Session state key to store the result
        title: Dialog title
        initial_dir: Initial directory

    Returns:
        True if a folder was selected, False otherwise
    """
    selected = _open_folder_dialog(title=title, initial_dir=initial_dir)
    if selected:
        st.session_state[session_key] = selected
        return True
    return False


def render_welcome_screen() -> str | None:
    """Render the welcome/project selection screen.

    Returns:
        - Project path (str) if a project was selected
        - Empty string ("") if user chose temporary workspace
        - None if still showing welcome screen (no selection yet)
    """
    # Initialize wizard state
    if "welcome_mode" not in st.session_state:
        st.session_state.welcome_mode = "main"  # main, create, open

    # Header
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">RRational</h1>
            <p style="color: #666; font-size: 1.1rem;">HRV Analysis Toolkit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Route to appropriate view
    if st.session_state.welcome_mode == "create":
        return _render_create_project_wizard()
    elif st.session_state.welcome_mode == "open":
        return _render_open_project()
    else:
        return _render_main_welcome()


def _render_main_welcome() -> str | None:
    """Render the main welcome screen with options."""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Recent Projects
        recent = get_recent_projects()
        if recent:
            st.subheader("Recent Projects")
            for i, project in enumerate(recent[:5]):
                project_path = project.get("path", "")
                project_name = project.get("name", "Unknown")

                # Check if project still exists
                exists = Path(project_path).exists() if project_path else False

                col_name, col_action = st.columns([4, 1])
                with col_name:
                    if exists:
                        if st.button(
                            f"**{project_name}**",
                            key=f"recent_{i}",
                            use_container_width=True,
                        ):
                            add_recent_project(Path(project_path), project_name)
                            return project_path
                        st.caption(f"  {project_path}")
                    else:
                        st.markdown(f"~~{project_name}~~ (not found)")
                        st.caption(f"  {project_path}")

                with col_action:
                    if not exists:
                        if st.button("Remove", key=f"remove_{i}"):
                            remove_recent_project(Path(project_path))
                            st.rerun()

            st.markdown("---")

        # Main action buttons
        st.subheader("Get Started")

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button(
                "Create New Project",
                key="btn_create",
                use_container_width=True,
                type="primary",
            ):
                st.session_state.welcome_mode = "create"
                st.rerun()

        with col_b:
            if st.button(
                "Open Existing Project",
                key="btn_open",
                use_container_width=True,
            ):
                st.session_state.welcome_mode = "open"
                st.rerun()

        st.markdown("")

        # Temporary workspace option
        if st.button(
            "Continue Without Project",
            key="btn_temp",
            use_container_width=True,
            help="Use a temporary workspace. Settings will be saved globally.",
        ):
            return ""  # Empty string signals temporary workspace

        st.caption(
            "Temporary workspace stores settings in your user folder. "
            "For better organization, create a project."
        )

    return None


def _render_create_project_wizard() -> str | None:
    """Render the project creation wizard."""
    st.subheader("Create New Project")

    # Back button
    if st.button("< Back", key="back_create"):
        st.session_state.welcome_mode = "main"
        for key in ["new_project_path", "new_project_name", "new_project_description",
                    "new_project_author", "create_parent_path"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.markdown("---")

    # Step 1: Project location
    st.markdown("**Step 1: Project Location**")

    # Parent folder selection with Browse button
    col_path, col_browse = st.columns([4, 1])

    # Initialize default path if not set (use home folder - works on all OS)
    if "create_parent_path" not in st.session_state:
        st.session_state.create_parent_path = str(Path.home())

    with col_path:
        # Use session state key directly for the text input
        parent_path = st.text_input(
            "Parent Folder",
            key="create_parent_path",  # Use same key as session state
            help="The folder where your project folder will be created",
        )

    with col_browse:
        st.markdown("")  # Spacing to align with text input
        st.markdown("")
        st.button(
            "Browse...",
            key="browse_create",
            on_click=_trigger_folder_dialog,
            args=("create_parent_path", "Select Parent Folder for Project", st.session_state.get("create_parent_path")),
        )

    # Get the current parent path from session state
    parent_path = st.session_state.get("create_parent_path", "")

    # New folder name input
    new_folder_name = st.text_input(
        "Project Folder Name",
        value=st.session_state.get("new_folder_name", ""),
        placeholder="MyHRVStudy",
        help="Name for the new project folder",
    )
    if new_folder_name:
        st.session_state.new_folder_name = new_folder_name

    # Build and show full project path
    if parent_path and new_folder_name:
        project_folder = str(Path(parent_path) / new_folder_name)
        st.session_state.new_project_path = project_folder
        st.info(f"Project will be created at: `{project_folder}`")
    else:
        project_folder = ""

    st.markdown("---")

    # Step 2: Project details
    st.markdown("**Step 2: Project Details**")

    project_name = st.text_input(
        "Project Name",
        value=st.session_state.get("new_project_name", ""),
        placeholder="My HRV Study",
        help="A descriptive name for your project",
    )
    if project_name:
        st.session_state.new_project_name = project_name

    project_description = st.text_area(
        "Description (optional)",
        value=st.session_state.get("new_project_description", ""),
        placeholder="Brief description of your study...",
        height=100,
    )
    if project_description:
        st.session_state.new_project_description = project_description

    project_author = st.text_input(
        "Author (optional)",
        value=st.session_state.get("new_project_author", ""),
        placeholder="Your name",
    )
    if project_author:
        st.session_state.new_project_author = project_author

    st.markdown("---")

    # Step 3: Data Sources
    st.markdown("**Step 3: Data Sources**")
    st.caption("Select which apps/devices you'll import data from. Folders will be created in `data/raw/`.")

    # Available data sources (can be extended in the future)
    data_sources = {
        "hrv_logger": "HRV Logger (iOS/Android app - CSV files)",
        "vns": "VNS Analyse (Windows software - TXT files)",
    }

    # Initialize data sources selection
    if "new_project_data_sources" not in st.session_state:
        st.session_state.new_project_data_sources = ["hrv_logger"]  # Default selection

    selected_sources = []
    for source_id, source_label in data_sources.items():
        default_checked = source_id in st.session_state.get("new_project_data_sources", ["hrv_logger"])
        if st.checkbox(source_label, value=default_checked, key=f"source_{source_id}"):
            selected_sources.append(source_id)

    st.session_state.new_project_data_sources = selected_sources

    if not selected_sources:
        st.warning("Select at least one data source")

    st.markdown("---")

    # Step 4: Migration option
    st.markdown("**Step 4: Import Settings**")

    import_config = False
    if has_global_config():
        import_config = st.checkbox(
            "Import existing settings from previous sessions",
            value=True,
            help="Copy your groups, events, sections, and other settings into the new project",
        )

        if import_config:
            with st.expander("Preview settings to import"):
                summary = get_global_config_summary()
                for filename, info in summary.items():
                    item_count = info.get("item_count", 0)
                    if item_count > 0:
                        st.write(f"- **{filename}**: {item_count} items")
    else:
        st.info("No existing settings found. Project will start with defaults.")

    # Validation
    can_create = bool(project_folder and project_name and selected_sources)

    if project_folder:
        path = Path(project_folder)
        if path.exists():
            project_file = path / ProjectManager.PROJECT_FILE
            if project_file.exists():
                st.warning("A project already exists at this location. Choose a different folder.")
                can_create = False
            elif any(path.iterdir()):
                st.warning("This folder is not empty. Consider using a new folder name.")

    st.markdown("---")

    # Create button
    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button(
            "Create Project",
            key="btn_do_create",
            type="primary",
            disabled=not can_create,
            use_container_width=True,
        ):
            try:
                # Create the project with selected data sources
                pm = ProjectManager.create_project(
                    path=Path(project_folder),
                    name=project_name,
                    description=project_description,
                    author=project_author,
                    data_sources=selected_sources,
                )

                # Migrate config if requested
                if import_config:
                    result = migrate_global_config_to_project(pm)
                    if result.get("migrated"):
                        st.success(f"Imported: {', '.join(result['migrated'])}")
                    if result.get("errors"):
                        st.warning(f"Some files had errors: {', '.join(result['errors'])}")

                # Add to recent projects
                add_recent_project(Path(project_folder), project_name)

                # Clean up wizard state
                for key in ["new_project_path", "new_project_name",
                           "new_project_description", "new_project_author",
                           "new_folder_name", "create_parent_path",
                           "new_project_data_sources"]:
                    if key in st.session_state:
                        del st.session_state[key]
                # Clean up source checkboxes
                for source_id in data_sources.keys():
                    key = f"source_{source_id}"
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.welcome_mode = "main"

                st.success(f"Project '{project_name}' created successfully!")
                return str(Path(project_folder).resolve())

            except FileExistsError as e:
                st.error(f"Project already exists: {e}")
            except Exception as e:
                st.error(f"Failed to create project: {e}")

    return None


def _render_open_project() -> str | None:
    """Render the open project dialog."""
    st.subheader("Open Existing Project")

    # Back button
    if st.button("< Back", key="back_open"):
        st.session_state.welcome_mode = "main"
        if "open_project_path" in st.session_state:
            del st.session_state["open_project_path"]
        st.rerun()

    st.markdown("---")

    st.markdown(
        "Select your project folder "
        "(the folder containing `project.rrational`)."
    )

    # Project path selection with Browse button
    col_path, col_browse = st.columns([4, 1])

    # Initialize if not set
    if "open_project_path" not in st.session_state:
        st.session_state.open_project_path = ""

    with col_path:
        # Use session state key directly for the text input
        project_path = st.text_input(
            "Project Folder",
            key="open_project_path",  # Use same key as session state
            placeholder="Select or enter project folder path",
        )

    with col_browse:
        st.markdown("")  # Spacing to align with text input
        st.markdown("")
        st.button(
            "Browse...",
            key="browse_open",
            on_click=_trigger_folder_dialog,
            args=("open_project_path", "Select Project Folder", st.session_state.get("open_project_path") or str(Path.home())),
        )

    # Get current path from session state
    project_path = st.session_state.get("open_project_path", "")

    st.markdown("---")

    # Validation and info display
    can_open = False
    if project_path:
        path = Path(project_path)
        is_valid, issues = ProjectManager.is_valid_project(path)

        if is_valid:
            st.success("Valid project found!")
            can_open = True

            # Show project info
            try:
                pm = ProjectManager.open_project(path)
                if pm.metadata:
                    st.write(f"**Name:** {pm.metadata.name}")
                    if pm.metadata.description:
                        st.write(f"**Description:** {pm.metadata.description}")
                    if pm.metadata.author:
                        st.write(f"**Author:** {pm.metadata.author}")
            except Exception:
                pass

        elif path.exists():
            # Folder exists but not a valid project
            st.warning("This folder is not a valid RRational project.")
            if issues:
                with st.expander("Issues found"):
                    for issue in issues:
                        st.write(f"- {issue}")

            # Offer to create project here
            if st.checkbox("Create a new project in this folder?"):
                st.session_state.welcome_mode = "create"
                st.session_state.create_parent_path = str(path.parent)
                st.session_state.new_folder_name = path.name
                st.rerun()
        else:
            st.info("Click 'Browse...' to select a project folder")

    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button(
            "Open Project",
            key="btn_do_open",
            type="primary",
            disabled=not can_open,
            use_container_width=True,
        ):
            path = Path(project_path)
            try:
                pm = ProjectManager.open_project(path)
                add_recent_project(path, pm.metadata.name if pm.metadata else path.name)

                # Clean up
                if "open_project_path" in st.session_state:
                    del st.session_state["open_project_path"]
                st.session_state.welcome_mode = "main"

                return str(path.resolve())

            except Exception as e:
                st.error(f"Failed to open project: {e}")

    return None


__all__ = ["render_welcome_screen"]
