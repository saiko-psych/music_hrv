"""Generate simulated HRV data for testing and demos.

Creates 18 participants across 3 groups:
- Default group (6): HRV Logger format - mostly GOOD data
- Intervention group (6): HRV Logger format - mixed quality
- Music_Study group (6): VNS Analyse format - mostly GOOD data

Usage:
    uv run python scripts/generate_demo_data.py
"""

from __future__ import annotations

import csv
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "demo"


def make_id(num: int, suffix: str) -> str:
    """Generate participant ID like 0001DEMO."""
    return f"{num:04d}{suffix.upper()}"


# ============================================================================
# HRV Signal Generation
# ============================================================================

def generate_rr_intervals(
    duration_min: float,
    base_hr: float = 70,
    hrv_amplitude: float = 60,
    artifact_rate: float = 0.0,
) -> list[int]:
    """Generate realistic RR intervals with natural HRV patterns.

    Args:
        duration_min: Recording duration in minutes
        base_hr: Base heart rate in BPM (60-100 typical)
        hrv_amplitude: HRV variation in ms (40-80 for realistic range)
        artifact_rate: Fraction of beats to corrupt (0.0-0.15)

    Returns:
        List of RR intervals in milliseconds
    """
    # Base RR from heart rate
    base_rr = 60000 / base_hr  # ~857ms for 70 BPM

    rr_intervals = []
    current_time_ms = 0
    duration_ms = duration_min * 60 * 1000

    while current_time_ms < duration_ms:
        # Generate natural HRV pattern (respiratory sinus arrhythmia)
        phase = current_time_ms / 1000

        # Very slow trend (simulates activity changes) - 0.002 Hz
        trend = hrv_amplitude * 0.6 * math.sin(2 * math.pi * 0.002 * phase)

        # Slow oscillation (0.1 Hz - ~6 breaths/min during rest)
        slow_wave = hrv_amplitude * 0.4 * math.sin(2 * math.pi * 0.1 * phase)

        # Faster component (0.25 Hz - ~15 breaths/min)
        fast_wave = hrv_amplitude * 0.25 * math.sin(2 * math.pi * 0.25 * phase)

        # Random component
        random_var = random.gauss(0, hrv_amplitude * 0.3)

        rr = int(base_rr + trend + slow_wave + fast_wave + random_var)

        # Add artifacts if specified (mild artifacts that won't hit clamp limits)
        if artifact_rate > 0 and random.random() < artifact_rate:
            artifact_type = random.choice(["ectopic", "missed", "noise"])
            if artifact_type == "ectopic":
                rr = int(rr * random.uniform(0.70, 0.85))  # Mild ectopic
            elif artifact_type == "missed":
                rr = int(rr * random.uniform(1.6, 1.9))  # Mild missed beat
            else:
                # Noise within reasonable bounds
                rr = int(rr + random.gauss(0, 100))

        # Keep within physiological bounds (tighter to avoid flagging)
        rr = max(450, min(1200, rr))

        rr_intervals.append(rr)
        current_time_ms += rr

    return rr_intervals


# ============================================================================
# HRV Logger Format Writers
# ============================================================================

def write_hrv_logger_rr(
    path: Path,
    rr_intervals: list[int],
    start_time: datetime,
) -> None:
    """Write HRV Logger RR CSV file.

    Format: date, rr, since start
    Example: 2025-03-11 09:57:38 +0000, 655, 1326
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        # Header matches real format exactly (no newline at end - real format has none)
        f.write("date, rr, since start")

        elapsed_ms = 0
        for rr in rr_intervals:
            timestamp = start_time + timedelta(milliseconds=elapsed_ms)
            date_str = timestamp.strftime("%Y-%m-%d %H:%M:%S +0000")
            f.write(f"\n{date_str}, {rr}, {elapsed_ms}")
            elapsed_ms += rr


def write_hrv_logger_events(
    path: Path,
    events: list[tuple[str, datetime, float]],
) -> None:
    """Write HRV Logger Events CSV file.

    Format: date, timestamp, annotation, manual
    Example: 2025-03-11 09:58:28 +0000, 60.651170, Start Ruhe, 1
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        # Header matches real format exactly
        f.write("date, timestamp, annotation, manual")

        for label, timestamp, offset_s in events:
            date_str = timestamp.strftime("%Y-%m-%d %H:%M:%S +0000")
            # timestamp is offset in seconds with decimals
            f.write(f"\n{date_str}, {offset_s:.6f}, {label}, 1")


# ============================================================================
# VNS Analyse Format Writer
# ============================================================================

def write_vns_file(
    path: Path,
    rr_intervals: list[int],
    events: list[tuple[str, datetime, float]],
) -> None:
    """Write VNS Analyse TXT file matching real format.

    Real format has:
    - Korrektur header
    - Main parameters sections (Rohwerte and Korrigierte Werte)
    - RR-Intervalle sections (raw has occasional outliers, corrected smooths them)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build event lookup by elapsed time (ms)
    event_by_offset = {}
    for label, _, offset_s in events:
        event_by_offset[int(offset_s * 1000)] = label

    # Create corrected version (smooth any outliers)
    # In real VNS data, corrected values replace outliers with interpolated values
    corrected_rr = []
    for i, rr in enumerate(rr_intervals):
        # Check if this value looks like an outlier compared to neighbors
        if i > 0 and i < len(rr_intervals) - 1:
            prev_rr = rr_intervals[i - 1]
            next_rr = rr_intervals[i + 1]
            neighbor_avg = (prev_rr + next_rr) / 2

            # If >40% different from neighbors, it might be an artifact - smooth it
            if abs(rr - neighbor_avg) / neighbor_avg > 0.4:
                corrected_rr.append(int(neighbor_avg))
            else:
                corrected_rr.append(rr)
        else:
            corrected_rr.append(rr)

    # Calculate HRV parameters from corrected data
    mean_rr = sum(corrected_rr) / len(corrected_rr)
    hr_mean = 60000 / mean_rr

    # SDNN
    variance = sum((rr - mean_rr) ** 2 for rr in corrected_rr) / len(corrected_rr)
    sdnn = variance ** 0.5

    # RMSSD
    diffs = [(corrected_rr[i+1] - corrected_rr[i]) ** 2 for i in range(len(corrected_rr) - 1)]
    rmssd = (sum(diffs) / len(diffs)) ** 0.5 if diffs else 0

    with open(path, "w", encoding="utf-8") as f:
        # Header matching real format
        f.write("Korrektur\tAktiv\n")
        f.write("\n")

        # Main parameters - Rohwerte
        f.write("Hauptparameter der VNS Analyse - Rohwerte (Nicht aktiv)\n")
        f.write("mainParameterSI\tmainParameterRMSSD\tmainParameterHRMean\tmainParameterAlpha1\tmainParameterSDNN\n")
        f.write(f"10.00\t{rmssd:.2f}\t{hr_mean:.2f}\t1.00\t{sdnn:.2f}\t\n")
        f.write("\n")

        # Main parameters - Korrigierte Werte
        f.write("Hauptparameter der VNS Analyse - Korrigierte Werte (Aktiv)\n")
        f.write("mainParameterSI\tmainParameterRMSSD\tmainParameterHRMean\tmainParameterAlpha1\tmainParameterSDNN\n")
        f.write(f"10.00\t{rmssd:.2f}\t{hr_mean:.2f}\t1.00\t{sdnn:.2f}\t\n")
        f.write("\n")

        # RR intervals - Rohwerte (raw data with possible outliers)
        f.write("RR-Intervalle - Rohwerte (Nicht aktiv)\n")

        elapsed_ms = 0
        for rr in rr_intervals:
            rr_seconds = rr / 1000
            line = f"{rr_seconds:.3f}\t"

            # Check for event at this position (within 500ms tolerance)
            for event_offset in list(event_by_offset.keys()):
                if abs(elapsed_ms - event_offset) < 500:
                    line += f"Notiz: {event_by_offset.pop(event_offset)}"
                    break

            f.write(line + "\n")
            elapsed_ms += rr

        f.write("\n")

        # RR intervals - Korrigierte Werte (smoothed/corrected data)
        f.write("RR-Intervalle - Korrigierte Werte (Aktiv)\n")
        for rr in corrected_rr:
            rr_seconds = rr / 1000
            f.write(f"{rr_seconds:.3f}\t\n")


# ============================================================================
# Participant Profiles - MOSTLY GOOD DATA!
# ============================================================================

# Standard events for full protocol (~105 min)
def make_standard_events(start_time: datetime) -> list[tuple[str, datetime, float]]:
    """Create standard boundary events."""
    events = [
        ("Start Ruhe", 0),          # Rest pre start
        ("Ruhe Ende", 5),           # Rest pre end (5 min rest)
        ("Messung Start", 5.5),     # Measurement start
        ("Pause Start", 50),        # Pause start (~45 min music)
        ("Pause Ende", 55),         # Pause end (5 min pause)
        ("Ruhe Start", 100),        # Rest post start (~45 min music)
        ("Ruhe Ende", 105),         # Rest post end (5 min rest)
        ("Messung Ende", 105.5),    # Measurement end
    ]
    return [(label, start_time + timedelta(minutes=t), t * 60) for label, t in events]


def make_short_events(start_time: datetime) -> list[tuple[str, datetime, float]]:
    """Create events for shorter protocol (~45 min)."""
    events = [
        ("Start Ruhe", 0),
        ("Ruhe Ende", 5),
        ("Messung Start", 5.5),
        ("Ruhe Start", 40),
        ("Ruhe Ende", 45),
    ]
    return [(label, start_time + timedelta(minutes=t), t * 60) for label, t in events]


# Profile definitions - emphasis on GOOD data with realistic variation
# Artifact rates: excellent=0, good=0.001, fair=0.01, poor=0.03
PROFILES = {
    # Default group - HRV Logger format, mostly excellent data
    "default": [
        # 4 PERFECT participants (NO artifacts for excellent)
        {"id": make_id(1, "CTRL"), "quality": "excellent", "duration": 106, "hr": 62, "artifact": 0.0},
        {"id": make_id(2, "CTRL"), "quality": "excellent", "duration": 105, "hr": 75, "artifact": 0.0},
        {"id": make_id(3, "CTRL"), "quality": "excellent", "duration": 107, "hr": 58, "artifact": 0.0},
        {"id": make_id(4, "CTRL"), "quality": "good", "duration": 104, "hr": 82, "artifact": 0.001},
        # 2 with minor issues
        {"id": make_id(5, "CTRL"), "quality": "fair", "duration": 108, "hr": 70, "artifact": 0.01},
        {"id": make_id(6, "CTRL"), "quality": "fair", "duration": 102, "hr": 88, "artifact": 0.015, "missing_events": ["Pause Start"]},
    ],

    # Intervention group - HRV Logger format, mixed but mostly good
    "intervention": [
        # 3 EXCELLENT (NO artifacts)
        {"id": make_id(1, "EXPR"), "quality": "excellent", "duration": 106, "hr": 64, "artifact": 0.0},
        {"id": make_id(2, "EXPR"), "quality": "excellent", "duration": 105, "hr": 78, "artifact": 0.0},
        {"id": make_id(3, "EXPR"), "quality": "good", "duration": 104, "hr": 68, "artifact": 0.001},
        # 2 GOOD
        {"id": make_id(4, "EXPR"), "quality": "good", "duration": 107, "hr": 72, "artifact": 0.002},
        {"id": make_id(5, "EXPR"), "quality": "fair", "duration": 103, "hr": 85, "artifact": 0.01},
        # 1 with issues (for testing)
        {"id": make_id(6, "EXPR"), "quality": "poor", "duration": 95, "hr": 92, "artifact": 0.03,
         "missing_events": ["Pause Ende", "Ruhe Start"]},
    ],

    # Music Study group - VNS format, mostly excellent (HR range like real VNS data ~70-90)
    "music_study": [
        # 4 EXCELLENT VNS recordings (NO artifacts)
        {"id": make_id(1, "VNST"), "quality": "excellent", "duration": 106, "hr": 72, "artifact": 0.0},
        {"id": make_id(2, "VNST"), "quality": "excellent", "duration": 105, "hr": 80, "artifact": 0.0},
        {"id": make_id(3, "VNST"), "quality": "excellent", "duration": 107, "hr": 68, "artifact": 0.0},
        {"id": make_id(4, "VNST"), "quality": "good", "duration": 104, "hr": 85, "artifact": 0.001},
        # 2 with minor issues
        {"id": make_id(5, "VNST"), "quality": "fair", "duration": 108, "hr": 76, "artifact": 0.01},
        {"id": make_id(6, "VNST"), "quality": "fair", "duration": 45, "hr": 90, "artifact": 0.015, "short": True},
    ],
}


# ============================================================================
# Main Generation
# ============================================================================

def generate_all_data():
    """Generate all demo data files."""

    print("Generating demo data...")
    print(f"Output directory: {OUTPUT_DIR}")

    # Create output directories
    hrv_logger_dir = OUTPUT_DIR / "hrv_logger"
    vns_dir = OUTPUT_DIR / "vns_analyse"
    hrv_logger_dir.mkdir(parents=True, exist_ok=True)
    vns_dir.mkdir(parents=True, exist_ok=True)

    # Base date for recordings
    base_date = datetime(2025, 3, 15, 9, 0, 0)

    generated = []

    for group_name, profiles in PROFILES.items():
        print(f"\n{group_name.upper()} group:")

        for i, profile in enumerate(profiles):
            pid = profile["id"]
            quality = profile["quality"]

            # Vary start times
            start_time = base_date + timedelta(days=i, hours=random.randint(0, 2), minutes=random.randint(0, 30))

            # Generate RR intervals
            rr_intervals = generate_rr_intervals(
                duration_min=profile["duration"],
                base_hr=profile["hr"],
                hrv_amplitude=random.uniform(40, 55),
                artifact_rate=profile["artifact"],
            )

            # Generate events
            if profile.get("short"):
                events = make_short_events(start_time)
            else:
                events = make_standard_events(start_time)

            # Remove missing events if specified
            missing = profile.get("missing_events", [])
            events = [(l, t, o) for l, t, o in events if l not in missing]

            # Write files
            if group_name == "music_study":
                # VNS format with proper filename (note: double space after "VNS -" to match real format)
                hours = profile["duration"] // 60
                mins = profile["duration"] % 60
                filename = f"VNS -  {pid}, {pid} (0) - {start_time.strftime('%d.%m.%Y %H.%M')} Langzeit, {hours}h {mins}min KORRIGIERT.txt"
                filepath = vns_dir / filename
                write_vns_file(filepath, rr_intervals, events)
                print(f"  [{quality:9}] {pid} -> VNS ({len(rr_intervals)} RR)")
            else:
                # HRV Logger format
                date_str = start_time.strftime("%Y-%m-%d")
                rr_path = hrv_logger_dir / f"{date_str}_RR_{pid}.csv"
                events_path = hrv_logger_dir / f"{date_str}_Events_{pid}.csv"
                write_hrv_logger_rr(rr_path, rr_intervals, start_time)
                write_hrv_logger_events(events_path, events)
                print(f"  [{quality:9}] {pid} -> HRV Logger ({len(rr_intervals)} RR, {len(events)} events)")

            generated.append({
                "id": pid,
                "group": group_name,
                "quality": quality,
                "format": "vns" if group_name == "music_study" else "hrv_logger",
            })

    return generated


def generate_csv_assignments(generated: list[dict]):
    """Generate CSV for group/playlist assignments (partially complete for testing)."""

    csv_path = OUTPUT_DIR / "assignments.csv"
    playlists = ["R1", "R2", "R3", "R4", "R5", "R6"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["code", "group", "playlist"])

        for i, item in enumerate(generated):
            pid = item["id"]
            group = item["group"]
            playlist = playlists[i % 6]

            # Make some incomplete for testing CSV import
            if i < 2:
                # First 2: missing from CSV entirely
                continue
            elif i < 4:
                # Next 2: group only
                writer.writerow([pid, group, ""])
            elif i < 6:
                # Next 2: playlist only
                writer.writerow([pid, "", playlist])
            else:
                # Rest: complete
                writer.writerow([pid, group, playlist])

    print(f"\nGenerated assignments CSV: {csv_path}")
    print(f"  - 2 participants missing entirely (for manual assignment testing)")
    print(f"  - 2 with group only")
    print(f"  - 2 with playlist only")
    print(f"  - {len(generated) - 6} complete")


def main():
    """Main entry point."""
    random.seed(42)  # Reproducible

    generated = generate_all_data()
    generate_csv_assignments(generated)

    print(f"\n{'='*60}")
    print(f"Generated {len(generated)} participants")
    print(f"{'='*60}")

    # Quality summary
    print("\nQuality distribution:")
    for q in ["excellent", "good", "fair", "poor"]:
        count = sum(1 for g in generated if g["quality"] == q)
        if count > 0:
            print(f"  {q}: {count}")


if __name__ == "__main__":
    main()
