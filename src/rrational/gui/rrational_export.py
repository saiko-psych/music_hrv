"""RRational Export Format (.rrational) - Ready for Analysis files.

This module handles the .rrational file format for exporting inspected
participant data with full audit trail for scientific reproducibility.

Version History:
- v1.0: Original format with raw RR data per segment
- v2.0: Section-based format with only corrected NN intervals (no raw RR)

File format: YAML with structured metadata, validated sections, NN intervals,
and audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml


# Version of the .rrational file format
RRATIONAL_VERSION_V1 = "1.0"
RRATIONAL_VERSION_V2 = "2.0"
RRATIONAL_VERSION = RRATIONAL_VERSION_V2  # Current default


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


# =============================================================================
# V2.0 DATA STRUCTURES
# =============================================================================


@dataclass
class EventChoiceV2:
    """Which event was chosen for a section boundary (v2.0)."""
    label: str
    timestamp: str  # ISO format
    beat_idx: int
    raw_label: str | None = None


@dataclass
class SectionValidationV2:
    """Section validation state (v2.0)."""
    validated_at: str  # ISO format
    start_event: EventChoiceV2
    end_event: EventChoiceV2
    total_duration_s: float
    total_beat_count: int


@dataclass
class ExclusionZoneV2:
    """Time range excluded from analysis (v2.0)."""
    id: str
    start_timestamp: str  # ISO format
    end_timestamp: str  # ISO format
    start_beat_idx: int
    end_beat_idx: int
    reason: str = ""
    created_at: str | None = None


@dataclass
class RecordingGapV2:
    """Gap in recording from multi-file sources like VNS (v2.0)."""
    gap_id: str
    after_file: str
    before_file: str
    gap_start: str  # ISO format
    gap_end: str  # ISO format
    gap_duration_s: float


@dataclass
class ArtifactDetectionV2:
    """Artifact detection results for a section (v2.0)."""
    method: str  # "lipponen2019", "threshold", etc.
    threshold_pct: float | None = None
    run_at: str | None = None  # ISO format
    detected_count: int = 0
    by_type: dict = field(default_factory=dict)  # {"ectopic": N, "missed": N, ...}
    artifact_rate_detected: float = 0.0


@dataclass
class ManualArtifactsV2:
    """Manual artifact markings for a section (v2.0)."""
    added_indices: list[int] = field(default_factory=list)
    removed_indices: list[int] = field(default_factory=list)
    last_modified: str | None = None  # ISO format


@dataclass
class FinalArtifactsV2:
    """Combined artifact summary for a section (v2.0)."""
    indices: list[int] = field(default_factory=list)
    count: int = 0
    rate: float = 0.0


@dataclass
class QualityV2:
    """Quality assessment for a section (v2.0)."""
    grade: str = "unknown"  # "excellent", "good", "moderate", "poor"
    recommendation: str = ""
    usable_beats: int = 0
    usable_duration_s: float = 0.0
    meets_time_domain_min: bool = False  # >= 100 beats
    meets_freq_domain_min: bool = False  # >= 300 beats AND >= 2 min


@dataclass
class NNCorrectionV2:
    """NN correction metadata for a section (v2.0)."""
    method: str = "none"  # "kubios", "none"
    corrected_at: str | None = None  # ISO format
    intervals_corrected: int = 0


@dataclass
class AnalysisSegmentV2:
    """A segment of data for analysis (v2.0).

    Sections are split into segments by exclusion zones and gaps.
    """
    segment_id: str
    type: str  # "data", "exclusion", "gap"
    start_timestamp: str  # ISO format
    end_timestamp: str  # ISO format
    duration_s: float = 0.0
    nn_count: int = 0
    nn_start_idx: int | None = None  # Index in nn_intervals.data
    nn_end_idx: int | None = None
    reason: str | None = None  # For exclusion/gap types


@dataclass
class NNIntervalsDataV2:
    """NN intervals data for a section (v2.0)."""
    # Compact format: [[timestamp_ms_from_section_start, nn_ms, was_corrected], ...]
    data: list[list] = field(default_factory=list)
    # Correction details: [{"nn_idx": int, "original_rr_ms": int, "corrected_nn_ms": int}, ...]
    corrections: list[dict] = field(default_factory=list)


@dataclass
class SectionDefinitionV2:
    """Section definition (v2.0)."""
    start_event: str
    end_event: str
    label: str


@dataclass
class SectionExportV2:
    """Complete export data for one section (v2.0)."""
    definition: SectionDefinitionV2
    validation: SectionValidationV2
    exclusion_zones: list[ExclusionZoneV2] = field(default_factory=list)
    gaps: list[RecordingGapV2] = field(default_factory=list)
    artifact_detection: ArtifactDetectionV2 | None = None
    manual_artifacts: ManualArtifactsV2 = field(default_factory=ManualArtifactsV2)
    final_artifacts: FinalArtifactsV2 = field(default_factory=FinalArtifactsV2)
    quality: QualityV2 = field(default_factory=QualityV2)
    nn_correction: NNCorrectionV2 = field(default_factory=NNCorrectionV2)
    analysis_segments: list[AnalysisSegmentV2] = field(default_factory=list)
    nn_intervals: NNIntervalsDataV2 = field(default_factory=NNIntervalsDataV2)


@dataclass
class MetadataV2:
    """Export metadata (v2.0)."""
    participant_id: str
    created_at: str  # ISO format
    last_modified: str  # ISO format
    source_app: str = "HRV Logger"
    source_files: list[dict] = field(default_factory=list)  # [{path, type, hash}, ...]
    recording_info: dict = field(default_factory=dict)  # {start, end, total_beats, total_duration_s}
    software_versions: dict = field(default_factory=dict)


@dataclass
class AuditEntryV2:
    """Single audit trail entry (v2.0)."""
    step: int
    action: str
    timestamp: str  # ISO format
    details: str
    section: str | None = None
    extra: dict | None = None


@dataclass
class RRationalExportV2:
    """Complete .rrational v2.0 export file structure.

    Key differences from v1.0:
    - No raw RR data (only corrected NN intervals)
    - Section-based structure (each section has all its data)
    - Analysis segments for handling exclusion zones and gaps
    - Incremental update support
    """
    metadata: MetadataV2
    sections: dict[str, SectionExportV2] = field(default_factory=dict)
    exclusion_zones_summary: list[dict] = field(default_factory=list)
    recording_gaps: list[RecordingGapV2] = field(default_factory=list)
    audit_trail: list[AuditEntryV2] = field(default_factory=list)


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


# =============================================================================
# V2.0 SAVE/LOAD FUNCTIONS
# =============================================================================


def _section_export_to_dict(section: SectionExportV2) -> dict:
    """Convert SectionExportV2 to dict for YAML serialization."""
    return {
        "definition": {
            "start_event": section.definition.start_event,
            "end_event": section.definition.end_event,
            "label": section.definition.label,
        },
        "validation": {
            "validated_at": section.validation.validated_at,
            "start_event": {
                "label": section.validation.start_event.label,
                "timestamp": section.validation.start_event.timestamp,
                "beat_idx": section.validation.start_event.beat_idx,
                "raw_label": section.validation.start_event.raw_label,
            },
            "end_event": {
                "label": section.validation.end_event.label,
                "timestamp": section.validation.end_event.timestamp,
                "beat_idx": section.validation.end_event.beat_idx,
                "raw_label": section.validation.end_event.raw_label,
            },
            "total_duration_s": section.validation.total_duration_s,
            "total_beat_count": section.validation.total_beat_count,
        },
        "exclusion_zones": [
            {
                "id": z.id,
                "start_timestamp": z.start_timestamp,
                "end_timestamp": z.end_timestamp,
                "start_beat_idx": z.start_beat_idx,
                "end_beat_idx": z.end_beat_idx,
                "reason": z.reason,
                "created_at": z.created_at,
            }
            for z in section.exclusion_zones
        ],
        "gaps": [
            {
                "gap_id": g.gap_id,
                "after_file": g.after_file,
                "before_file": g.before_file,
                "gap_start": g.gap_start,
                "gap_end": g.gap_end,
                "gap_duration_s": g.gap_duration_s,
            }
            for g in section.gaps
        ],
        "artifact_detection": {
            "method": section.artifact_detection.method,
            "threshold_pct": section.artifact_detection.threshold_pct,
            "run_at": section.artifact_detection.run_at,
            "detected_count": section.artifact_detection.detected_count,
            "by_type": section.artifact_detection.by_type,
            "artifact_rate_detected": section.artifact_detection.artifact_rate_detected,
        } if section.artifact_detection else None,
        "manual_artifacts": {
            "added_indices": section.manual_artifacts.added_indices,
            "removed_indices": section.manual_artifacts.removed_indices,
            "last_modified": section.manual_artifacts.last_modified,
        },
        "final_artifacts": {
            "indices": section.final_artifacts.indices,
            "count": section.final_artifacts.count,
            "rate": section.final_artifacts.rate,
        },
        "quality": {
            "grade": section.quality.grade,
            "recommendation": section.quality.recommendation,
            "usable_beats": section.quality.usable_beats,
            "usable_duration_s": section.quality.usable_duration_s,
            "meets_time_domain_min": section.quality.meets_time_domain_min,
            "meets_freq_domain_min": section.quality.meets_freq_domain_min,
        },
        "nn_correction": {
            "method": section.nn_correction.method,
            "corrected_at": section.nn_correction.corrected_at,
            "intervals_corrected": section.nn_correction.intervals_corrected,
        },
        "analysis_segments": [
            {
                "segment_id": seg.segment_id,
                "type": seg.type,
                "start_timestamp": seg.start_timestamp,
                "end_timestamp": seg.end_timestamp,
                "duration_s": seg.duration_s,
                "nn_count": seg.nn_count,
                "nn_start_idx": seg.nn_start_idx,
                "nn_end_idx": seg.nn_end_idx,
                "reason": seg.reason,
            }
            for seg in section.analysis_segments
        ],
        "nn_intervals": {
            "data": section.nn_intervals.data,
            "corrections": section.nn_intervals.corrections,
        },
    }


def _dict_to_section_export(data: dict) -> SectionExportV2:
    """Convert dict to SectionExportV2."""
    definition = SectionDefinitionV2(
        start_event=data["definition"]["start_event"],
        end_event=data["definition"]["end_event"],
        label=data["definition"]["label"],
    )

    validation_data = data["validation"]
    validation = SectionValidationV2(
        validated_at=validation_data["validated_at"],
        start_event=EventChoiceV2(
            label=validation_data["start_event"]["label"],
            timestamp=validation_data["start_event"]["timestamp"],
            beat_idx=validation_data["start_event"]["beat_idx"],
            raw_label=validation_data["start_event"].get("raw_label"),
        ),
        end_event=EventChoiceV2(
            label=validation_data["end_event"]["label"],
            timestamp=validation_data["end_event"]["timestamp"],
            beat_idx=validation_data["end_event"]["beat_idx"],
            raw_label=validation_data["end_event"].get("raw_label"),
        ),
        total_duration_s=validation_data["total_duration_s"],
        total_beat_count=validation_data["total_beat_count"],
    )

    exclusion_zones = [
        ExclusionZoneV2(
            id=z["id"],
            start_timestamp=z["start_timestamp"],
            end_timestamp=z["end_timestamp"],
            start_beat_idx=z["start_beat_idx"],
            end_beat_idx=z["end_beat_idx"],
            reason=z.get("reason", ""),
            created_at=z.get("created_at"),
        )
        for z in data.get("exclusion_zones", [])
    ]

    gaps = [
        RecordingGapV2(
            gap_id=g["gap_id"],
            after_file=g["after_file"],
            before_file=g["before_file"],
            gap_start=g["gap_start"],
            gap_end=g["gap_end"],
            gap_duration_s=g["gap_duration_s"],
        )
        for g in data.get("gaps", [])
    ]

    artifact_detection = None
    ad = data.get("artifact_detection")
    if ad:
        artifact_detection = ArtifactDetectionV2(
            method=ad["method"],
            threshold_pct=ad.get("threshold_pct"),
            run_at=ad.get("run_at"),
            detected_count=ad.get("detected_count", 0),
            by_type=ad.get("by_type", {}),
            artifact_rate_detected=ad.get("artifact_rate_detected", 0.0),
        )

    ma = data.get("manual_artifacts", {})
    manual_artifacts = ManualArtifactsV2(
        added_indices=ma.get("added_indices", []),
        removed_indices=ma.get("removed_indices", []),
        last_modified=ma.get("last_modified"),
    )

    fa = data.get("final_artifacts", {})
    final_artifacts = FinalArtifactsV2(
        indices=fa.get("indices", []),
        count=fa.get("count", 0),
        rate=fa.get("rate", 0.0),
    )

    q = data.get("quality", {})
    quality = QualityV2(
        grade=q.get("grade", "unknown"),
        recommendation=q.get("recommendation", ""),
        usable_beats=q.get("usable_beats", 0),
        usable_duration_s=q.get("usable_duration_s", 0.0),
        meets_time_domain_min=q.get("meets_time_domain_min", False),
        meets_freq_domain_min=q.get("meets_freq_domain_min", False),
    )

    nc = data.get("nn_correction", {})
    nn_correction = NNCorrectionV2(
        method=nc.get("method", "none"),
        corrected_at=nc.get("corrected_at"),
        intervals_corrected=nc.get("intervals_corrected", 0),
    )

    analysis_segments = [
        AnalysisSegmentV2(
            segment_id=seg["segment_id"],
            type=seg["type"],
            start_timestamp=seg["start_timestamp"],
            end_timestamp=seg["end_timestamp"],
            duration_s=seg.get("duration_s", 0.0),
            nn_count=seg.get("nn_count", 0),
            nn_start_idx=seg.get("nn_start_idx"),
            nn_end_idx=seg.get("nn_end_idx"),
            reason=seg.get("reason"),
        )
        for seg in data.get("analysis_segments", [])
    ]

    nn = data.get("nn_intervals", {})
    nn_intervals = NNIntervalsDataV2(
        data=nn.get("data", []),
        corrections=nn.get("corrections", []),
    )

    return SectionExportV2(
        definition=definition,
        validation=validation,
        exclusion_zones=exclusion_zones,
        gaps=gaps,
        artifact_detection=artifact_detection,
        manual_artifacts=manual_artifacts,
        final_artifacts=final_artifacts,
        quality=quality,
        nn_correction=nn_correction,
        analysis_segments=analysis_segments,
        nn_intervals=nn_intervals,
    )


def save_rrational_v2(
    export_data: RRationalExportV2,
    filepath: Path | str,
    incremental: bool = False,
) -> None:
    """Save export data to a .rrational v2.0 file.

    Args:
        export_data: The RRationalExportV2 object to save
        filepath: Path to save the file (should end in .rrational)
        incremental: If True and file exists, only update changed sections
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    existing_data = None
    if incremental and filepath.exists():
        try:
            existing_data = load_rrational_v2(filepath)
        except Exception:
            existing_data = None

    # Build sections dict
    sections_dict = {}
    for section_name, section in export_data.sections.items():
        sections_dict[section_name] = _section_export_to_dict(section)

    # If incremental, merge with existing
    if existing_data and incremental:
        for section_name, section in existing_data.sections.items():
            if section_name not in sections_dict:
                sections_dict[section_name] = _section_export_to_dict(section)

    # Build the YAML structure
    data = {
        "rrational_version": RRATIONAL_VERSION_V2,
        "file_type": "analysis_ready",

        "metadata": {
            "participant_id": export_data.metadata.participant_id,
            "created_at": export_data.metadata.created_at,
            "last_modified": export_data.metadata.last_modified,
            "source_app": export_data.metadata.source_app,
            "source_files": export_data.metadata.source_files,
            "recording_info": export_data.metadata.recording_info,
            "software_versions": export_data.metadata.software_versions,
        },

        "sections": sections_dict,

        "exclusion_zones_summary": export_data.exclusion_zones_summary,

        "recording_gaps": [
            {
                "gap_id": g.gap_id,
                "after_file": g.after_file,
                "before_file": g.before_file,
                "gap_start": g.gap_start,
                "gap_end": g.gap_end,
                "gap_duration_s": g.gap_duration_s,
            }
            for g in export_data.recording_gaps
        ],

        "audit_trail": [
            {
                "step": entry.step,
                "action": entry.action,
                "timestamp": entry.timestamp,
                "details": entry.details,
                "section": entry.section,
                "extra": entry.extra,
            }
            for entry in export_data.audit_trail
        ],
    }

    # Write YAML file
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def load_rrational_v2(filepath: Path | str) -> RRationalExportV2:
    """Load a .rrational v2.0 file.

    Args:
        filepath: Path to the .rrational file

    Returns:
        RRationalExportV2 object with the loaded data

    Raises:
        ValueError: If file is v1.0 format (use load_rrational for v1.0)
    """
    filepath = Path(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    version = data.get("rrational_version", "1.0")
    if version == RRATIONAL_VERSION_V1:
        raise ValueError(
            f"File is v1.0 format. Use load_rrational() or migrate_v1_to_v2() instead."
        )

    meta = data.get("metadata", {})
    metadata = MetadataV2(
        participant_id=meta.get("participant_id", ""),
        created_at=meta.get("created_at", ""),
        last_modified=meta.get("last_modified", ""),
        source_app=meta.get("source_app", "HRV Logger"),
        source_files=meta.get("source_files", []),
        recording_info=meta.get("recording_info", {}),
        software_versions=meta.get("software_versions", {}),
    )

    sections = {}
    for section_name, section_data in data.get("sections", {}).items():
        sections[section_name] = _dict_to_section_export(section_data)

    recording_gaps = [
        RecordingGapV2(
            gap_id=g["gap_id"],
            after_file=g["after_file"],
            before_file=g["before_file"],
            gap_start=g["gap_start"],
            gap_end=g["gap_end"],
            gap_duration_s=g["gap_duration_s"],
        )
        for g in data.get("recording_gaps", [])
    ]

    audit_trail = [
        AuditEntryV2(
            step=entry["step"],
            action=entry["action"],
            timestamp=entry["timestamp"],
            details=entry["details"],
            section=entry.get("section"),
            extra=entry.get("extra"),
        )
        for entry in data.get("audit_trail", [])
    ]

    return RRationalExportV2(
        metadata=metadata,
        sections=sections,
        exclusion_zones_summary=data.get("exclusion_zones_summary", []),
        recording_gaps=recording_gaps,
        audit_trail=audit_trail,
    )


def get_rrational_version(filepath: Path | str) -> str:
    """Get the version of a .rrational file without fully loading it.

    Args:
        filepath: Path to the .rrational file

    Returns:
        Version string (e.g., "1.0", "2.0")
    """
    filepath = Path(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        # Only read first few lines to get version
        for line in f:
            if line.startswith("rrational_version:"):
                return line.split(":")[1].strip().strip('"').strip("'")
            # Stop after first 10 lines to avoid reading whole file
            if f.tell() > 500:
                break

    return "1.0"  # Default to v1.0 if not found


def load_rrational_any_version(filepath: Path | str) -> RRationalExportV2 | RRationalExport:
    """Load a .rrational file of any version.

    Args:
        filepath: Path to the .rrational file

    Returns:
        RRationalExportV2 for v2.0 files, RRationalExport for v1.0 files
    """
    version = get_rrational_version(filepath)

    if version == RRATIONAL_VERSION_V2:
        return load_rrational_v2(filepath)
    else:
        return load_rrational(filepath)


def build_rrational_v2(
    participant_id: str,
    sections_to_export: list[str],
    data_dir: str | None = None,
    project_path: Path | None = None,
) -> tuple[RRationalExportV2, list[str]]:
    """Build a v2.0 export by consolidating data from intermediate files.

    This function loads data from:
    - {participant_id}_events.yml (events, exclusion zones)
    - {participant_id}_artifacts.yml (artifact detection per section)
    - {participant_id}_section_validations.yml (validated sections)
    - {participant_id}_nn_intervals.yml (corrected NN intervals)

    Args:
        participant_id: The participant ID
        sections_to_export: List of section names to include
        data_dir: Optional data directory
        project_path: Project path (takes priority if provided)

    Returns:
        Tuple of (RRationalExportV2, list of warnings/errors)
    """
    from datetime import datetime as dt
    from rrational.gui.persistence import (
        load_participant_events,
        load_artifact_corrections,
        load_section_validations,
        load_nn_intervals,
        get_processed_dir,
    )

    warnings = []
    now = dt.now().isoformat()

    # Load all intermediate data
    events_data = load_participant_events(participant_id, data_dir, project_path=project_path)
    artifacts_data = load_artifact_corrections(participant_id, data_dir=data_dir, project_path=project_path)
    validations_data = load_section_validations(participant_id, data_dir=data_dir, project_path=project_path)
    nn_data = load_nn_intervals(participant_id, data_dir=data_dir, project_path=project_path)

    if not validations_data:
        warnings.append(f"No section validations found for {participant_id}")

    # Build metadata
    metadata = MetadataV2(
        participant_id=participant_id,
        created_at=now,
        last_modified=now,
        source_app="Unknown",  # Will be updated if we can determine it
        software_versions={"rrational": "0.7.7"},  # TODO: Get from package
    )

    # Get exclusion zones from events data
    exclusion_zones_raw = events_data.get("exclusion_zones", []) if events_data else []

    # Build sections
    sections = {}
    exclusion_zones_summary = []
    audit_trail = []
    audit_step = 1

    validations = validations_data.get("sections", {}) if validations_data else {}
    artifacts = artifacts_data.get("sections", {}) if artifacts_data else {}
    nn_sections = nn_data.get("sections", {}) if nn_data else {}

    for section_name in sections_to_export:
        # Get validation data for this section
        section_validation = validations.get(section_name, {})

        if not section_validation:
            warnings.append(f"Section '{section_name}' has no validation data - skipping")
            continue

        if not section_validation.get("is_valid", False):
            warnings.append(f"Section '{section_name}' is not validated - skipping")
            continue

        # Build definition
        definition = SectionDefinitionV2(
            start_event=section_validation.get("start_event", {}).get("canonical", ""),
            end_event=section_validation.get("end_event", {}).get("canonical", ""),
            label=section_name,
        )

        # Build validation
        start_evt = section_validation.get("start_event", {})
        end_evt = section_validation.get("end_event", {})

        validation = SectionValidationV2(
            validated_at=validations_data.get("saved_at", now) if validations_data else now,
            start_event=EventChoiceV2(
                label=start_evt.get("canonical", ""),
                timestamp=start_evt.get("timestamp", ""),
                beat_idx=start_evt.get("beat_idx", 0),
                raw_label=start_evt.get("raw_label"),
            ),
            end_event=EventChoiceV2(
                label=end_evt.get("canonical", ""),
                timestamp=end_evt.get("timestamp", ""),
                beat_idx=end_evt.get("beat_idx", 0),
                raw_label=end_evt.get("raw_label"),
            ),
            total_duration_s=section_validation.get("duration_s", 0.0),
            total_beat_count=section_validation.get("beat_count", 0),
        )

        # Find exclusion zones that affect this section
        section_exclusions = []
        section_start_ts = start_evt.get("timestamp", "")
        section_end_ts = end_evt.get("timestamp", "")

        for i, zone in enumerate(exclusion_zones_raw):
            zone_start = zone.get("start", "")
            zone_end = zone.get("end", "")

            # Check if zone overlaps with section (simple string comparison for ISO timestamps)
            if zone_start and zone_end and section_start_ts and section_end_ts:
                if zone_end > section_start_ts and zone_start < section_end_ts:
                    excl = ExclusionZoneV2(
                        id=f"excl_{i+1}",
                        start_timestamp=zone_start,
                        end_timestamp=zone_end,
                        start_beat_idx=zone.get("start_beat_idx", 0),
                        end_beat_idx=zone.get("end_beat_idx", 0),
                        reason=zone.get("reason", ""),
                        created_at=zone.get("created_at"),
                    )
                    section_exclusions.append(excl)

                    # Add to summary if not already there
                    summary_entry = {
                        "id": excl.id,
                        "timestamp_range": f"{zone_start} - {zone_end}",
                        "reason": zone.get("reason", ""),
                        "affects_sections": [section_name],
                    }
                    # Check if already in summary
                    existing = next((s for s in exclusion_zones_summary if s["id"] == excl.id), None)
                    if existing:
                        if section_name not in existing["affects_sections"]:
                            existing["affects_sections"].append(section_name)
                    else:
                        exclusion_zones_summary.append(summary_entry)

        # Get artifact detection for this section
        section_artifacts = artifacts.get(section_name, {})
        artifact_detection = None
        manual_artifacts = ManualArtifactsV2()
        final_artifacts = FinalArtifactsV2()

        if section_artifacts:
            # Algorithm detection
            algo = section_artifacts.get("algorithm", {})
            if algo:
                artifact_detection = ArtifactDetectionV2(
                    method=algo.get("method", "unknown"),
                    threshold_pct=algo.get("threshold_pct"),
                    run_at=algo.get("run_at"),
                    detected_count=algo.get("detected_count", len(algo.get("detected_indices", []))),
                    by_type=algo.get("by_type", {}),
                    artifact_rate_detected=algo.get("artifact_rate", 0.0),
                )

            # Manual artifacts
            manual = section_artifacts.get("manual", {})
            manual_artifacts = ManualArtifactsV2(
                added_indices=manual.get("added_indices", []),
                removed_indices=manual.get("removed_indices", []),
                last_modified=manual.get("last_modified"),
            )

            # Final artifacts
            final_indices = section_artifacts.get("final_artifact_indices", [])
            final_rate = section_artifacts.get("final_artifact_rate", 0.0)
            final_artifacts = FinalArtifactsV2(
                indices=final_indices,
                count=len(final_indices),
                rate=final_rate,
            )

        # Get NN intervals for this section
        section_nn = nn_sections.get(section_name, {})
        nn_correction = NNCorrectionV2()
        nn_intervals = NNIntervalsDataV2()

        if section_nn:
            nn_correction = NNCorrectionV2(
                method=section_nn.get("correction_method", "none"),
                corrected_at=section_nn.get("corrected_at"),
                intervals_corrected=section_nn.get("intervals_corrected", 0),
            )
            nn_intervals = NNIntervalsDataV2(
                data=section_nn.get("intervals", []),
                corrections=section_nn.get("corrections", []),
            )
        else:
            warnings.append(f"Section '{section_name}' has no NN intervals - analysis may use raw data")

        # Calculate quality
        usable_beats = len(nn_intervals.data) if nn_intervals.data else validation.total_beat_count
        usable_duration = validation.total_duration_s
        # Subtract exclusion zone durations
        for excl in section_exclusions:
            try:
                excl_start = dt.fromisoformat(excl.start_timestamp)
                excl_end = dt.fromisoformat(excl.end_timestamp)
                usable_duration -= (excl_end - excl_start).total_seconds()
            except (ValueError, TypeError):
                pass

        quality = QualityV2(
            grade=get_quality_grade(final_artifacts.rate),
            recommendation=get_quigley_recommendation(final_artifacts.rate, usable_beats),
            usable_beats=usable_beats,
            usable_duration_s=usable_duration,
            meets_time_domain_min=usable_beats >= 100,
            meets_freq_domain_min=usable_beats >= 300 and usable_duration >= 120,
        )

        # Calculate analysis segments
        analysis_segments = []
        if section_start_ts and section_end_ts:
            try:
                start_dt = dt.fromisoformat(section_start_ts)
                end_dt = dt.fromisoformat(section_end_ts)
                analysis_segments = calculate_analysis_segments(
                    start_dt, end_dt, section_exclusions, [], section_name
                )

                # Update nn_count and indices for data segments
                nn_idx = 0
                for seg in analysis_segments:
                    if seg.type == "data" and nn_intervals.data:
                        # Estimate NN count based on duration proportion
                        total_data_duration = sum(s.duration_s for s in analysis_segments if s.type == "data")
                        if total_data_duration > 0:
                            seg.nn_count = int(len(nn_intervals.data) * seg.duration_s / total_data_duration)
                            seg.nn_start_idx = nn_idx
                            seg.nn_end_idx = nn_idx + seg.nn_count - 1
                            nn_idx += seg.nn_count
            except (ValueError, TypeError):
                pass

        # Build section export
        section_export = SectionExportV2(
            definition=definition,
            validation=validation,
            exclusion_zones=section_exclusions,
            gaps=[],  # TODO: Load from VNS recording info if available
            artifact_detection=artifact_detection,
            manual_artifacts=manual_artifacts,
            final_artifacts=final_artifacts,
            quality=quality,
            nn_correction=nn_correction,
            analysis_segments=analysis_segments,
            nn_intervals=nn_intervals,
        )

        sections[section_name] = section_export

        # Add audit entry
        audit_trail.append(AuditEntryV2(
            step=audit_step,
            action="section_exported",
            timestamp=now,
            details=f"Exported {section_name}: {usable_beats} NN intervals, {len(analysis_segments)} segments",
            section=section_name,
        ))
        audit_step += 1

    # Create export
    export = RRationalExportV2(
        metadata=metadata,
        sections=sections,
        exclusion_zones_summary=exclusion_zones_summary,
        recording_gaps=[],
        audit_trail=audit_trail,
    )

    return export, warnings


def calculate_analysis_segments(
    section_start_ts: datetime,
    section_end_ts: datetime,
    exclusion_zones: list[ExclusionZoneV2],
    gaps: list[RecordingGapV2],
    section_name: str,
) -> list[AnalysisSegmentV2]:
    """Calculate analysis segments for a section.

    Splits the section into data segments separated by exclusion zones and gaps.

    Args:
        section_start_ts: Section start timestamp
        section_end_ts: Section end timestamp
        exclusion_zones: List of exclusion zones within the section
        gaps: List of recording gaps within the section
        section_name: Name of the section (for segment IDs)

    Returns:
        List of AnalysisSegmentV2 objects representing the segments
    """
    from datetime import datetime as dt

    # Collect all "breaks" (exclusions and gaps) sorted by start time
    breaks = []

    for zone in exclusion_zones:
        zone_start = dt.fromisoformat(zone.start_timestamp) if isinstance(zone.start_timestamp, str) else zone.start_timestamp
        zone_end = dt.fromisoformat(zone.end_timestamp) if isinstance(zone.end_timestamp, str) else zone.end_timestamp
        breaks.append({
            "type": "exclusion",
            "start": zone_start,
            "end": zone_end,
            "id": zone.id,
            "reason": zone.reason,
        })

    for gap in gaps:
        gap_start = dt.fromisoformat(gap.gap_start) if isinstance(gap.gap_start, str) else gap.gap_start
        gap_end = dt.fromisoformat(gap.gap_end) if isinstance(gap.gap_end, str) else gap.gap_end
        breaks.append({
            "type": "gap",
            "start": gap_start,
            "end": gap_end,
            "id": gap.gap_id,
            "reason": f"Recording gap ({gap.gap_duration_s:.1f}s)",
        })

    # Sort breaks by start time
    breaks.sort(key=lambda b: b["start"])

    # Build segments
    segments = []
    current_start = section_start_ts
    data_segment_num = 1

    for brk in breaks:
        # Skip breaks outside section
        if brk["end"] <= section_start_ts or brk["start"] >= section_end_ts:
            continue

        # Data segment before this break
        if brk["start"] > current_start:
            duration = (brk["start"] - current_start).total_seconds()
            segments.append(AnalysisSegmentV2(
                segment_id=f"{section_name}_seg{data_segment_num}",
                type="data",
                start_timestamp=current_start.isoformat(),
                end_timestamp=brk["start"].isoformat(),
                duration_s=duration,
            ))
            data_segment_num += 1

        # The break itself
        segments.append(AnalysisSegmentV2(
            segment_id=f"{section_name}_{brk['type']}{len([s for s in segments if s.type == brk['type']]) + 1}",
            type=brk["type"],
            start_timestamp=brk["start"].isoformat(),
            end_timestamp=brk["end"].isoformat(),
            duration_s=(brk["end"] - brk["start"]).total_seconds(),
            reason=brk["reason"],
        ))

        current_start = brk["end"]

    # Final data segment after last break
    if current_start < section_end_ts:
        duration = (section_end_ts - current_start).total_seconds()
        segments.append(AnalysisSegmentV2(
            segment_id=f"{section_name}_seg{data_segment_num}",
            type="data",
            start_timestamp=current_start.isoformat(),
            end_timestamp=section_end_ts.isoformat(),
            duration_s=duration,
        ))

    # If no breaks, entire section is one data segment
    if not segments:
        duration = (section_end_ts - section_start_ts).total_seconds()
        segments.append(AnalysisSegmentV2(
            segment_id=f"{section_name}_seg1",
            type="data",
            start_timestamp=section_start_ts.isoformat(),
            end_timestamp=section_end_ts.isoformat(),
            duration_s=duration,
        ))

    return segments
