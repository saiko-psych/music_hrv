"""Project management for RRational HRV analysis application.

This module provides project-based organization where each study/project
is self-contained with its own data, configuration, and analysis results.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectMetadata:
    """Metadata stored in project.rrational file."""

    name: str
    description: str = ""
    created_at: str = ""
    modified_at: str = ""
    rrational_version: str = ""
    author: str = ""
    notes: str = ""
    data_sources: list[dict[str, Any]] = field(default_factory=list)


class ProjectManager:
    """Manages RRational project files and structure.

    A project has the following structure:
        MyStudy/
        ├── project.rrational      # Project metadata (YAML)
        ├── data/
        │   ├── raw/               # Raw HRV data files
        │   │   ├── hrv_logger/    # HRV Logger CSV files
        │   │   └── vns/           # VNS Analyse TXT files
        │   └── processed/         # Exported .rrational files + participant events
        ├── config/                # Project-specific configuration
        │   ├── groups.yml
        │   ├── events.yml
        │   ├── sections.yml
        │   ├── participants.yml
        │   ├── playlist_groups.yml
        │   ├── music_labels.yml
        │   └── protocol.yml
        └── analysis/              # Analysis results (future)
    """

    PROJECT_FILE = "project.rrational"
    REQUIRED_DIRS = ["data/raw", "data/processed", "config", "analysis"]
    CONFIG_FILES = [
        "groups.yml",
        "events.yml",
        "sections.yml",
        "participants.yml",
        "playlist_groups.yml",
        "music_labels.yml",
        "protocol.yml",
    ]

    def __init__(self, project_path: Path):
        """Initialize ProjectManager with a project path.

        Args:
            project_path: Path to the project directory
        """
        self.project_path = Path(project_path).resolve()
        self.metadata: ProjectMetadata | None = None

    # Supported data sources (ID -> display name)
    DATA_SOURCES = {
        "hrv_logger": "HRV Logger",
        "vns": "VNS Analyse",
    }

    @classmethod
    def create_project(
        cls,
        path: Path,
        name: str,
        description: str = "",
        author: str = "",
        notes: str = "",
        data_sources: list[str] | None = None,
    ) -> "ProjectManager":
        """Create a new project with folder structure.

        Args:
            path: Directory where project will be created
            name: Project name
            description: Project description
            author: Author name
            notes: Additional notes
            data_sources: List of data source IDs (e.g., ["hrv_logger", "vns"])

        Returns:
            ProjectManager instance for the new project

        Raises:
            FileExistsError: If project.rrational already exists
            OSError: If directory creation fails
        """
        path = Path(path).resolve()
        project_file = path / cls.PROJECT_FILE

        if project_file.exists():
            raise FileExistsError(f"Project already exists at {path}")

        # Default to hrv_logger if no sources specified
        if data_sources is None:
            data_sources = ["hrv_logger"]

        # Create directory structure
        path.mkdir(parents=True, exist_ok=True)
        for dir_name in cls.REQUIRED_DIRS:
            (path / dir_name).mkdir(parents=True, exist_ok=True)

        # Create data source subfolders in data/raw
        for source_id in data_sources:
            source_dir = path / "data" / "raw" / source_id
            source_dir.mkdir(parents=True, exist_ok=True)

        # Create project metadata with data sources info
        now = datetime.now().isoformat()
        data_sources_info = [
            {"id": src, "name": cls.DATA_SOURCES.get(src, src)}
            for src in data_sources
        ]
        metadata = ProjectMetadata(
            name=name,
            description=description,
            created_at=now,
            modified_at=now,
            rrational_version=_get_version(),
            author=author,
            notes=notes,
            data_sources=data_sources_info,
        )

        # Create project manager and save metadata
        pm = cls(path)
        pm.metadata = metadata
        pm.save_metadata()

        return pm

    @classmethod
    def open_project(cls, path: Path) -> "ProjectManager":
        """Open an existing project.

        Args:
            path: Path to project directory

        Returns:
            ProjectManager instance

        Raises:
            FileNotFoundError: If project.rrational doesn't exist
            ValueError: If project file is invalid
        """
        path = Path(path).resolve()
        project_file = path / cls.PROJECT_FILE

        if not project_file.exists():
            raise FileNotFoundError(f"No project found at {path}")

        pm = cls(path)
        pm._load_metadata()

        return pm

    @classmethod
    def is_valid_project(cls, path: Path) -> tuple[bool, list[str]]:
        """Validate if a path contains a valid project.

        Args:
            path: Path to check

        Returns:
            Tuple of (is_valid, list of issues)
        """
        path = Path(path).resolve()
        issues = []

        # Check project file exists
        project_file = path / cls.PROJECT_FILE
        if not project_file.exists():
            issues.append(f"Missing {cls.PROJECT_FILE}")

        # Check required directories
        for dir_name in cls.REQUIRED_DIRS:
            dir_path = path / dir_name
            if not dir_path.exists():
                issues.append(f"Missing directory: {dir_name}")

        # Check config directory
        config_dir = path / "config"
        if config_dir.exists():
            # Config files are optional but should be valid YAML if they exist
            for config_file in cls.CONFIG_FILES:
                config_path = config_dir / config_file
                if config_path.exists():
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            yaml.safe_load(f)
                    except yaml.YAMLError as e:
                        issues.append(f"Invalid YAML in {config_file}: {e}")

        return len(issues) == 0, issues

    def validate(self) -> tuple[bool, list[str]]:
        """Validate this project's structure.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        return self.is_valid_project(self.project_path)

    def repair_structure(self) -> list[str]:
        """Create missing directories in project structure.

        Returns:
            List of directories that were created
        """
        created = []

        for dir_name in self.REQUIRED_DIRS:
            dir_path = self.project_path / dir_name
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                created.append(dir_name)

        return created

    def get_data_dir(self) -> Path:
        """Get the data/raw directory path."""
        return self.project_path / "data" / "raw"

    def get_processed_dir(self) -> Path:
        """Get the processed directory path (inside data folder)."""
        processed = self.project_path / "data" / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        return processed

    def get_config_dir(self) -> Path:
        """Get the config directory path."""
        return self.project_path / "config"

    def get_config_path(self, config_name: str) -> Path:
        """Get path to a config file.

        Args:
            config_name: Name of config file (e.g., 'groups.yml')

        Returns:
            Path to the config file
        """
        return self.project_path / "config" / config_name

    def save_metadata(self) -> None:
        """Save project metadata to project.rrational."""
        if self.metadata is None:
            return

        self.metadata.modified_at = datetime.now().isoformat()

        data = {
            "rrational_project_version": "1.0",
            "metadata": {
                "name": self.metadata.name,
                "description": self.metadata.description,
                "created_at": self.metadata.created_at,
                "modified_at": self.metadata.modified_at,
                "author": self.metadata.author,
                "notes": self.metadata.notes,
            },
            "data_sources": self.metadata.data_sources,
            "software": {
                "rrational_version": self.metadata.rrational_version,
                "created_with": self.metadata.rrational_version,
            },
        }

        project_file = self.project_path / self.PROJECT_FILE
        with open(project_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)

    def _load_metadata(self) -> None:
        """Load project metadata from project.rrational."""
        project_file = self.project_path / self.PROJECT_FILE

        with open(project_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        meta = data.get("metadata", {})
        software = data.get("software", {})

        self.metadata = ProjectMetadata(
            name=meta.get("name", self.project_path.name),
            description=meta.get("description", ""),
            created_at=meta.get("created_at", ""),
            modified_at=meta.get("modified_at", ""),
            rrational_version=software.get("rrational_version", ""),
            author=meta.get("author", ""),
            notes=meta.get("notes", ""),
            data_sources=data.get("data_sources", []),
        )

    def update_data_sources(self, sources: list[dict[str, Any]]) -> None:
        """Update the data sources list in metadata.

        Args:
            sources: List of data source dictionaries with type, folder, count
        """
        if self.metadata:
            self.metadata.data_sources = sources
            self.save_metadata()


def _get_version() -> str:
    """Get the current RRational version."""
    try:
        from importlib.metadata import version
        return version("rrational")
    except Exception:
        return "0.7.1"  # Fallback


# --- Recent Projects Management ---

def get_recent_projects() -> list[dict[str, Any]]:
    """Get list of recently opened projects from settings.

    Returns:
        List of dicts with 'path', 'name', 'last_opened' keys
    """
    from rrational.gui.persistence import load_settings

    settings = load_settings()
    return settings.get("recent_projects", [])


def add_recent_project(path: Path, name: str) -> None:
    """Add a project to the recent projects list.

    Args:
        path: Project directory path
        name: Project name
    """
    from rrational.gui.persistence import load_settings, save_settings

    settings = load_settings()
    recent = settings.get("recent_projects", [])
    max_recent = settings.get("max_recent_projects", 10)

    path_str = str(Path(path).resolve())

    # Remove existing entry for this path
    recent = [p for p in recent if p.get("path") != path_str]

    # Add new entry at the beginning
    recent.insert(0, {
        "path": path_str,
        "name": name,
        "last_opened": datetime.now().isoformat(),
    })

    # Trim to max
    recent = recent[:max_recent]

    settings["recent_projects"] = recent
    save_settings(settings)


def remove_recent_project(path: Path) -> None:
    """Remove a project from the recent projects list.

    Args:
        path: Project directory path to remove
    """
    from rrational.gui.persistence import load_settings, save_settings

    settings = load_settings()
    recent = settings.get("recent_projects", [])

    path_str = str(Path(path).resolve())
    recent = [p for p in recent if p.get("path") != path_str]

    settings["recent_projects"] = recent
    save_settings(settings)


# --- Migration ---

def has_global_config() -> bool:
    """Check if global config exists with user data.

    Returns:
        True if ~/.rrational/ has config files
    """
    from rrational.gui.persistence import CONFIG_DIR

    if not CONFIG_DIR.exists():
        return False

    # Check if any config file exists and is non-empty
    for filename in ProjectManager.CONFIG_FILES:
        config_file = CONFIG_DIR / filename
        if config_file.exists() and config_file.stat().st_size > 0:
            return True

    return False


def migrate_global_config_to_project(project: ProjectManager) -> dict[str, list[str]]:
    """Migrate ~/.rrational/ config to a project.

    Args:
        project: ProjectManager instance to migrate config to

    Returns:
        Dict with 'migrated', 'skipped', 'errors' lists
    """
    from rrational.gui.persistence import CONFIG_DIR

    result: dict[str, list[str]] = {"migrated": [], "skipped": [], "errors": []}

    if not CONFIG_DIR.exists():
        return result

    config_dir = project.get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)

    for filename in ProjectManager.CONFIG_FILES:
        global_file = CONFIG_DIR / filename
        project_file = config_dir / filename

        if not global_file.exists():
            continue

        try:
            should_migrate = False

            if not project_file.exists():
                should_migrate = True
            else:
                # Migrate if global file has more data
                global_size = global_file.stat().st_size
                project_size = project_file.stat().st_size
                if global_size > project_size:
                    should_migrate = True

            if should_migrate:
                shutil.copy2(global_file, project_file)
                result["migrated"].append(filename)
            else:
                result["skipped"].append(filename)

        except Exception as e:
            result["errors"].append(f"{filename}: {str(e)}")

    return result


def get_global_config_summary() -> dict[str, Any]:
    """Get summary of global config for migration preview.

    Returns:
        Dict with config file info (exists, size, item_count)
    """
    from rrational.gui.persistence import CONFIG_DIR

    summary: dict[str, Any] = {}

    if not CONFIG_DIR.exists():
        return summary

    for filename in ProjectManager.CONFIG_FILES:
        config_file = CONFIG_DIR / filename
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                summary[filename] = {
                    "exists": True,
                    "size": config_file.stat().st_size,
                    "item_count": len(data) if isinstance(data, dict) else 0,
                }
            except Exception:
                summary[filename] = {"exists": True, "size": 0, "item_count": 0}

    return summary


__all__ = [
    "ProjectMetadata",
    "ProjectManager",
    "get_recent_projects",
    "add_recent_project",
    "remove_recent_project",
    "has_global_config",
    "migrate_global_config_to_project",
    "get_global_config_summary",
]
