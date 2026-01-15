"""RRational Export Format (.rrational) - Ready for Analysis files.

This module handles the .rrational file format for exporting inspected
participant data with full audit trail for scientific reproducibility.

File format: YAML with structured metadata, RR data, processing state,
and audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml


# Version of the .rrational file format
RRATIONAL_VERSION = "1.0"


@dataclass
class RRIntervalExport:
    """Single RR interval with metadata."""
    timestamp: str  # ISO format
    rr_ms: int
    elapsed_ms: int | None = None
    original_idx: int | None = None  # Index in original full recording


@dataclass
class ExclusionZone:
    """Time range excluded from analysis."""
    start: str  # ISO format
    end: str  # ISO format
    reason: str = ""
    exclude_from_duration: bool = True


@dataclass
class ManualArtifact:
    """User-marked artifact."""
    original_idx: int
    timestamp: str  # ISO format
    rr_value: float
    source: str = "manual_click"
    marked_at: str | None = None  # When user marked it


@dataclass
class ArtifactDetection:
    """Artifact detection results."""
    method: str  # "threshold", "lipponen2019", "lipponen2019_segmented"
    threshold_pct: float | None = None
    segment_beats: int | None = None
    gap_handling: str = "include"  # "include", "exclude", "boundary"
    total: int = 0
    by_type: dict = field(default_factory=dict)  # {"ectopic": N, "missed": N, ...}
    indices: list[int] = field(default_factory=list)


@dataclass
class CorrectedInterval:
    """Corrected NN interval."""
    timestamp: str  # ISO format
    nn_ms: int
    was_corrected: bool = False
    original_rr_ms: int | None = None


@dataclass
class ProcessingStep:
    """Single step in the audit trail."""
    step: int
    action: str
    timestamp: str  # ISO format
    details: str


@dataclass
class SegmentDefinition:
    """Defines the segment being exported."""
    type: str  # "section", "manual_range", "full_recording"
    section_name: str | None = None
    section_definition: dict | None = None  # {start_event, end_events, label}
    time_range: dict | None = None  # {start, end, label}


@dataclass
class QualityMetrics:
    """Quality assessment of the exported data."""
    artifact_rate_raw: float = 0.0
    artifact_rate_final: float = 0.0
    beats_after_cleaning: int = 0
    beats_after_exclusion_zones: int = 0
    data_retention_pct: float = 100.0
    recording_duration_sec: float = 0.0
    quality_grade: str = "unknown"  # "excellent", "good", "moderate", "poor"
    quigley_recommendation: str = ""


@dataclass
class RRationalExport:
    """Complete .rrational export file structure."""

    # Metadata
    participant_id: str
    export_timestamp: str  # ISO format
    exported_by: str = "RRational"
    source_app: str = "HRV Logger"
    source_file_paths: list[str] = field(default_factory=list)
    recording_datetime: str | None = None

    # Segment definition
    segment: SegmentDefinition | None = None

    # Raw data
    n_beats: int = 0
    rr_intervals: list[RRIntervalExport] = field(default_factory=list)

    # Processing state
    cleaning_config: dict = field(default_factory=dict)
    exclusion_zones: list[ExclusionZone] = field(default_factory=list)
    artifact_detection: ArtifactDetection | None = None
    manual_artifacts: list[ManualArtifact] = field(default_factory=list)
    excluded_detected_indices: list[int] = field(default_factory=list)
    final_artifact_indices: list[int] = field(default_factory=list)

    # Corrected data (optional)
    include_corrected: bool = False
    corrected_intervals: list[CorrectedInterval] = field(default_factory=list)
    correction_method: str | None = None
    n_corrected: int = 0

    # Quality metrics
    quality: QualityMetrics = field(default_factory=QualityMetrics)

    # Audit trail
    processing_steps: list[ProcessingStep] = field(default_factory=list)
    software_versions: dict = field(default_factory=dict)


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses to dicts for YAML serialization."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, list):
        return [_dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


def save_rrational(export_data: RRationalExport, filepath: Path | str) -> None:
    """Save export data to a .rrational file.

    Args:
        export_data: The RRationalExport object to save
        filepath: Path to save the file (should end in .rrational)
    """
    filepath = Path(filepath)

    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Build the YAML structure
    data = {
        "rrational_version": RRATIONAL_VERSION,
        "file_type": "ready_for_analysis",

        "metadata": {
            "participant_id": export_data.participant_id,
            "export_timestamp": export_data.export_timestamp,
            "exported_by": export_data.exported_by,
            "source_app": export_data.source_app,
            "source_file_paths": export_data.source_file_paths,
            "recording_datetime": export_data.recording_datetime,
        },

        "segment": _dataclass_to_dict(export_data.segment) if export_data.segment else None,

        "raw_data": {
            "n_beats": export_data.n_beats,
            "rr_intervals": _dataclass_to_dict(export_data.rr_intervals),
        },

        "processing": {
            "cleaning_config": export_data.cleaning_config,
            "exclusion_zones": _dataclass_to_dict(export_data.exclusion_zones),
            "artifact_detection": _dataclass_to_dict(export_data.artifact_detection) if export_data.artifact_detection else None,
            "manual_artifacts": _dataclass_to_dict(export_data.manual_artifacts),
            "excluded_detected_indices": export_data.excluded_detected_indices,
            "final_artifact_indices": export_data.final_artifact_indices,
        },

        "quality": _dataclass_to_dict(export_data.quality),

        "audit": {
            "processing_steps": _dataclass_to_dict(export_data.processing_steps),
            "software_versions": export_data.software_versions,
        },
    }

    # Add corrected data if included
    if export_data.include_corrected and export_data.corrected_intervals:
        data["corrected_data"] = {
            "correction_method": export_data.correction_method,
            "n_corrected": export_data.n_corrected,
            "nn_intervals": _dataclass_to_dict(export_data.corrected_intervals),
        }

    # Write YAML file
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_rrational(filepath: Path | str) -> RRationalExport:
    """Load a .rrational file.

    Args:
        filepath: Path to the .rrational file

    Returns:
        RRationalExport object with the loaded data
    """
    filepath = Path(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Validate version
    version = data.get("rrational_version", "0.0")
    if version != RRATIONAL_VERSION:
        # Future: handle version migration
        pass

    metadata = data.get("metadata", {})
    segment_data = data.get("segment")
    raw_data = data.get("raw_data", {})
    processing = data.get("processing", {})
    quality_data = data.get("quality", {})
    audit = data.get("audit", {})
    corrected_data = data.get("corrected_data")

    # Build segment definition
    segment = None
    if segment_data:
        segment = SegmentDefinition(
            type=segment_data.get("type", "full_recording"),
            section_name=segment_data.get("section_name"),
            section_definition=segment_data.get("section_definition"),
            time_range=segment_data.get("time_range"),
        )

    # Build RR intervals
    rr_intervals = []
    for rr in raw_data.get("rr_intervals", []):
        rr_intervals.append(RRIntervalExport(
            timestamp=rr.get("timestamp", ""),
            rr_ms=rr.get("rr_ms", 0),
            elapsed_ms=rr.get("elapsed_ms"),
            original_idx=rr.get("original_idx"),
        ))

    # Build exclusion zones
    exclusion_zones = []
    for zone in processing.get("exclusion_zones", []):
        exclusion_zones.append(ExclusionZone(
            start=zone.get("start", ""),
            end=zone.get("end", ""),
            reason=zone.get("reason", ""),
            exclude_from_duration=zone.get("exclude_from_duration", True),
        ))

    # Build artifact detection
    artifact_detection = None
    ad = processing.get("artifact_detection")
    if ad:
        artifact_detection = ArtifactDetection(
            method=ad.get("method", "threshold"),
            threshold_pct=ad.get("threshold_pct"),
            segment_beats=ad.get("segment_beats"),
            gap_handling=ad.get("gap_handling", "include"),
            total=ad.get("total", 0),
            by_type=ad.get("by_type", {}),
            indices=ad.get("indices", []),
        )

    # Build manual artifacts
    manual_artifacts = []
    for ma in processing.get("manual_artifacts", []):
        manual_artifacts.append(ManualArtifact(
            original_idx=ma.get("original_idx", 0),
            timestamp=ma.get("timestamp", ""),
            rr_value=ma.get("rr_value", 0),
            source=ma.get("source", "manual_click"),
            marked_at=ma.get("marked_at"),
        ))

    # Build quality metrics
    quality = QualityMetrics(
        artifact_rate_raw=quality_data.get("artifact_rate_raw", 0),
        artifact_rate_final=quality_data.get("artifact_rate_final", 0),
        beats_after_cleaning=quality_data.get("beats_after_cleaning", 0),
        beats_after_exclusion_zones=quality_data.get("beats_after_exclusion_zones", 0),
        data_retention_pct=quality_data.get("data_retention_pct", 100),
        recording_duration_sec=quality_data.get("recording_duration_sec", 0),
        quality_grade=quality_data.get("quality_grade", "unknown"),
        quigley_recommendation=quality_data.get("quigley_recommendation", ""),
    )

    # Build processing steps
    processing_steps = []
    for step in audit.get("processing_steps", []):
        processing_steps.append(ProcessingStep(
            step=step.get("step", 0),
            action=step.get("action", ""),
            timestamp=step.get("timestamp", ""),
            details=step.get("details", ""),
        ))

    # Build corrected intervals if present
    corrected_intervals = []
    include_corrected = False
    correction_method = None
    n_corrected = 0
    if corrected_data:
        include_corrected = True
        correction_method = corrected_data.get("correction_method")
        n_corrected = corrected_data.get("n_corrected", 0)
        for ci in corrected_data.get("nn_intervals", []):
            corrected_intervals.append(CorrectedInterval(
                timestamp=ci.get("timestamp", ""),
                nn_ms=ci.get("nn_ms", 0),
                was_corrected=ci.get("was_corrected", False),
                original_rr_ms=ci.get("original_rr_ms"),
            ))

    return RRationalExport(
        participant_id=metadata.get("participant_id", ""),
        export_timestamp=metadata.get("export_timestamp", ""),
        exported_by=metadata.get("exported_by", "RRational"),
        source_app=metadata.get("source_app", "HRV Logger"),
        source_file_paths=metadata.get("source_file_paths", []),
        recording_datetime=metadata.get("recording_datetime"),
        segment=segment,
        n_beats=raw_data.get("n_beats", 0),
        rr_intervals=rr_intervals,
        cleaning_config=processing.get("cleaning_config", {}),
        exclusion_zones=exclusion_zones,
        artifact_detection=artifact_detection,
        manual_artifacts=manual_artifacts,
        excluded_detected_indices=processing.get("excluded_detected_indices", []),
        final_artifact_indices=processing.get("final_artifact_indices", []),
        include_corrected=include_corrected,
        corrected_intervals=corrected_intervals,
        correction_method=correction_method,
        n_corrected=n_corrected,
        quality=quality,
        processing_steps=processing_steps,
        software_versions=audit.get("software_versions", {}),
    )


def validate_rrational(data: RRationalExport) -> tuple[bool, list[str]]:
    """Validate a RRationalExport object.

    Args:
        data: The export object to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    if not data.participant_id:
        errors.append("Missing participant_id")

    if not data.rr_intervals:
        errors.append("No RR intervals in export")

    if data.n_beats != len(data.rr_intervals):
        errors.append(f"n_beats ({data.n_beats}) doesn't match interval count ({len(data.rr_intervals)})")

    # Check RR interval validity
    for i, rr in enumerate(data.rr_intervals):
        if rr.rr_ms < 200 or rr.rr_ms > 3000:
            errors.append(f"RR interval {i} has unusual value: {rr.rr_ms}ms")

    # Check artifact indices are valid
    max_idx = len(data.rr_intervals) - 1
    for idx in data.final_artifact_indices:
        if idx < 0 or idx > max_idx:
            errors.append(f"Invalid artifact index: {idx}")

    return len(errors) == 0, errors


def get_quality_grade(artifact_rate: float) -> str:
    """Determine quality grade based on artifact rate.

    Based on Quigley et al. (2024) guidelines.
    """
    if artifact_rate <= 0.02:  # <= 2%
        return "excellent"
    elif artifact_rate <= 0.05:  # <= 5%
        return "good"
    elif artifact_rate <= 0.10:  # <= 10%
        return "moderate"
    else:
        return "poor"


def get_quigley_recommendation(artifact_rate: float, beat_count: int) -> str:
    """Get analysis recommendation based on Quigley et al. (2024) guidelines."""
    if artifact_rate > 0.10:
        return "exclude_or_manual_review"
    elif artifact_rate > 0.05:
        return "suitable_with_correction"
    elif beat_count < 100:
        return "insufficient_beats_time_domain"
    elif beat_count < 300:
        return "time_domain_only"
    else:
        return "suitable_for_all_metrics"


def find_rrational_files(
    participant_id: str,
    data_dir: Path | str | None = None,
    project_path: Path | None = None,
) -> list[Path]:
    """Find all .rrational files for a participant.

    Args:
        participant_id: The participant to find files for
        data_dir: Optional data directory (checks ../processed/)
        project_path: Project path (takes priority if provided)

    Returns:
        List of paths to .rrational files
    """
    files = []

    # Check project processed folder (highest priority)
    if project_path:
        project_path = Path(project_path)
        processed_dir = project_path / "processed"
        if processed_dir.exists():
            pattern = f"{participant_id}*.rrational"
            files.extend(processed_dir.glob(pattern))

    # Check processed folder relative to data_dir
    if data_dir:
        data_dir = Path(data_dir)
        processed_dir = data_dir.parent / "processed"
        if processed_dir.exists():
            pattern = f"{participant_id}*.rrational"
            files.extend(processed_dir.glob(pattern))

    # Check user config directory
    config_dir = Path.home() / ".rrational" / "exports"
    if config_dir.exists():
        pattern = f"{participant_id}*.rrational"
        files.extend(config_dir.glob(pattern))

    # Deduplicate (same file might be found via different paths)
    unique_files = list({f.resolve(): f for f in files}.values())
    return sorted(unique_files, key=lambda p: p.stat().st_mtime, reverse=True)


def build_export_filename(participant_id: str, segment_name: str | None = None) -> str:
    """Build a filename for the export.

    Args:
        participant_id: Participant ID
        segment_name: Optional segment name (e.g., "rest_pre")

    Returns:
        Filename like "0123ABCD_rest_pre.rrational"
    """
    if segment_name:
        return f"{participant_id}_{segment_name}.rrational"
    return f"{participant_id}_full.rrational"
