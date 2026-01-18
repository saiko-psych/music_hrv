"""Analysis tab - HRV analysis with NeuroKit2.

This module contains the render function for the Analysis tab.
Provides HRV metrics computation and visualization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

# Lazy import for plotly (saves ~0.12s on startup)
_go = None
_make_subplots = None
PLOTLY_AVAILABLE = True


def get_plotly_analysis():
    """Lazily import plotly for analysis tab."""
    global _go, _make_subplots, PLOTLY_AVAILABLE
    if _go is None:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            _go = go
            _make_subplots = make_subplots
        except ImportError:
            PLOTLY_AVAILABLE = False
            _go = None
            _make_subplots = None
    return _go, _make_subplots


from rrational.gui.shared import (  # noqa: E402
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
from rrational.gui.help_text import ANALYSIS_HELP  # noqa: E402
from rrational.gui.rrational_export import (  # noqa: E402
    find_rrational_files,
    load_rrational,
    load_rrational_v2,
    get_rrational_version,
    RRATIONAL_VERSION_V2,
)


# =============================================================================
# GROUP ANALYSIS DATA STRUCTURES
# =============================================================================


@dataclass
class ParticipantSectionResult:
    """Result of HRV analysis for one participant-section combination."""
    participant_id: str
    group: str
    section_name: str
    n_beats: int
    duration_s: float
    quality_grade: str
    artifact_rate: float
    hrv_metrics: dict  # All HRV metrics
    hrv_std: dict | None  # SD if overlapping windows used
    n_windows: int  # Number of windows (1 if no overlapping)


# =============================================================================
# HRV METRIC DEFINITIONS AND PRESETS
# =============================================================================

# All available HRV metrics organized by category
HRV_METRICS_CATALOG = {
    "time_basic": {
        "RMSSD": {"label": "RMSSD", "unit": "ms", "description": "Root mean square of successive differences"},
        "SDNN": {"label": "SDNN", "unit": "ms", "description": "Standard deviation of NN intervals"},
        "pNN50": {"label": "pNN50", "unit": "%", "description": "Percentage of successive intervals differing by >50ms"},
        "MeanNN": {"label": "Mean NN", "unit": "ms", "description": "Mean of NN intervals"},
        "MeanHR": {"label": "Mean HR", "unit": "bpm", "description": "Mean heart rate"},
    },
    "time_extended": {
        "SDSD": {"label": "SDSD", "unit": "ms", "description": "SD of successive differences"},
        "pNN20": {"label": "pNN20", "unit": "%", "description": "Percentage of successive intervals differing by >20ms"},
        "MedianNN": {"label": "Median NN", "unit": "ms", "description": "Median of NN intervals"},
        "CVNN": {"label": "CVNN", "unit": "", "description": "Coefficient of variation (SDNN/MeanNN)"},
        "CVSD": {"label": "CVSD", "unit": "", "description": "Coefficient of variation of successive differences"},
        "MadNN": {"label": "MadNN", "unit": "ms", "description": "Median absolute deviation of NN intervals"},
        "MCVNN": {"label": "MCVNN", "unit": "", "description": "Median-based CV (MadNN/MedianNN)"},
        "IQRNN": {"label": "IQRNN", "unit": "ms", "description": "Interquartile range of NN intervals"},
        "HTI": {"label": "HTI", "unit": "", "description": "HRV Triangular Index"},
        "TINN": {"label": "TINN", "unit": "ms", "description": "Triangular interpolation of NN histogram"},
    },
    "frequency": {
        "VLF": {"label": "VLF", "unit": "ms²", "description": "Very low frequency power (0.0033-0.04 Hz)"},
        "LF": {"label": "LF", "unit": "ms²", "description": "Low frequency power (0.04-0.15 Hz)"},
        "HF": {"label": "HF", "unit": "ms²", "description": "High frequency power (0.15-0.4 Hz)"},
        "LF_HF": {"label": "LF/HF", "unit": "", "description": "LF to HF ratio"},
        "LFn": {"label": "LF norm", "unit": "n.u.", "description": "Normalized LF power"},
        "HFn": {"label": "HF norm", "unit": "n.u.", "description": "Normalized HF power"},
        "TP": {"label": "Total Power", "unit": "ms²", "description": "Total spectral power"},
    },
    "nonlinear": {
        "SD1": {"label": "SD1", "unit": "ms", "description": "Poincaré plot SD perpendicular to identity line"},
        "SD2": {"label": "SD2", "unit": "ms", "description": "Poincaré plot SD along identity line"},
        "SD1SD2": {"label": "SD1/SD2", "unit": "", "description": "Ratio of SD1 to SD2"},
        "ApEn": {"label": "ApEn", "unit": "", "description": "Approximate entropy"},
        "SampEn": {"label": "SampEn", "unit": "", "description": "Sample entropy"},
        "DFA_alpha1": {"label": "DFA α1", "unit": "", "description": "Detrended fluctuation analysis short-term"},
        "DFA_alpha2": {"label": "DFA α2", "unit": "", "description": "Detrended fluctuation analysis long-term"},
    },
}

# Metric presets
HRV_METRIC_PRESETS = {
    "Basic": {
        "description": "Essential time-domain metrics for quick analysis",
        "metrics": ["RMSSD", "SDNN", "pNN50", "MeanNN", "MeanHR"],
    },
    "Time + Frequency": {
        "description": "Time-domain and frequency-domain metrics",
        "metrics": ["RMSSD", "SDNN", "pNN50", "MeanNN", "MeanHR", "LF", "HF", "LF_HF", "VLF", "TP"],
    },
    "Full (with nonlinear)": {
        "description": "All available metrics including nonlinear analysis",
        "metrics": list({m for cat in HRV_METRICS_CATALOG.values() for m in cat.keys()}),
    },
    "Poincaré Focus": {
        "description": "Metrics related to Poincaré plot analysis",
        "metrics": ["RMSSD", "SDNN", "SD1", "SD2", "SD1SD2", "MeanNN", "MeanHR"],
    },
    "Custom": {
        "description": "Select metrics manually",
        "metrics": [],  # User selects
    },
}

# Flatten all metrics for easy lookup
ALL_HRV_METRICS = {m: info for cat in HRV_METRICS_CATALOG.values() for m, info in cat.items()}


def get_metric_info(metric_name: str) -> dict:
    """Get info for a metric by name."""
    return ALL_HRV_METRICS.get(metric_name, {"label": metric_name, "unit": "", "description": ""})


# =============================================================================
# OVERLAPPING WINDOW ANALYSIS HELPERS
# =============================================================================


def generate_overlapping_windows_time(
    rr_intervals: list,
    window_duration_ms: float,
    step_size_ms: float,
) -> list[tuple[int, float, list]]:
    """Generate overlapping windows from a list of RR intervals (time-based).

    Args:
        rr_intervals: List of RR interval values in ms (or RRInterval objects with .rr_ms)
        window_duration_ms: Duration of each window in milliseconds
        step_size_ms: Step size between window starts in milliseconds

    Returns:
        List of tuples: (window_idx, window_start_ms, window_rr_list)
        Each window_rr_list contains the RR values that fall within that window.
    """
    if not rr_intervals:
        return []

    # Handle both raw values and RRInterval objects
    if hasattr(rr_intervals[0], 'rr_ms'):
        rr_values = [rr.rr_ms for rr in rr_intervals]
    else:
        rr_values = list(rr_intervals)

    # Calculate cumulative time (elapsed time at start of each beat)
    cumulative_time = [0.0]
    for rr in rr_values[:-1]:
        cumulative_time.append(cumulative_time[-1] + rr)

    total_duration_ms = cumulative_time[-1] + rr_values[-1]

    # Generate windows
    windows = []
    window_idx = 0
    window_start = 0.0

    while window_start + window_duration_ms <= total_duration_ms + step_size_ms / 2:
        window_end = window_start + window_duration_ms

        # Find RR intervals within this window
        window_rr = []
        for i, (elapsed, rr) in enumerate(zip(cumulative_time, rr_values)):
            # Include beat if it starts within the window
            if window_start <= elapsed < window_end:
                window_rr.append(rr)

        if window_rr:  # Only include non-empty windows
            windows.append((window_idx, window_start, window_rr))
            window_idx += 1

        window_start += step_size_ms

        # Safety: stop if we've generated too many windows
        if window_idx > 100:
            break

    return windows


def generate_overlapping_windows_beats(
    rr_intervals: list,
    window_beats: int,
    step_beats: int,
) -> list[tuple[int, int, list]]:
    """Generate overlapping windows from a list of RR intervals (beat-based).

    Args:
        rr_intervals: List of RR interval values in ms (or RRInterval objects with .rr_ms)
        window_beats: Number of beats in each window
        step_beats: Number of beats to step between window starts

    Returns:
        List of tuples: (window_idx, start_beat_idx, window_rr_list)
        Each window_rr_list contains the RR values for that window.
    """
    if not rr_intervals:
        return []

    # Handle both raw values and RRInterval objects
    if hasattr(rr_intervals[0], 'rr_ms'):
        rr_values = [rr.rr_ms for rr in rr_intervals]
    else:
        rr_values = list(rr_intervals)

    total_beats = len(rr_values)

    # Generate windows
    windows = []
    window_idx = 0
    start_beat = 0

    while start_beat + window_beats <= total_beats:
        end_beat = start_beat + window_beats
        window_rr = rr_values[start_beat:end_beat]

        windows.append((window_idx, start_beat, window_rr))
        window_idx += 1

        start_beat += step_beats

        # Safety: stop if we've generated too many windows
        if window_idx > 100:
            break

    return windows


# Backwards compatibility alias
def generate_overlapping_windows(
    rr_intervals: list,
    window_duration_ms: float,
    step_size_ms: float,
) -> list[tuple[int, float, list]]:
    """Backwards-compatible alias for generate_overlapping_windows_time."""
    return generate_overlapping_windows_time(rr_intervals, window_duration_ms, step_size_ms)


def aggregate_hrv_results(window_results: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate HRV results from multiple overlapping windows.

    Args:
        window_results: List of DataFrames, each containing HRV metrics for one window

    Returns:
        Tuple of (mean_results, std_results) DataFrames
    """
    if not window_results:
        return pd.DataFrame(), pd.DataFrame()

    # Concatenate all results
    all_results = pd.concat(window_results, ignore_index=True)

    # Calculate mean and std for each metric
    mean_results = all_results.mean().to_frame().T
    std_results = all_results.std().to_frame().T

    return mean_results, std_results


# =============================================================================
# PROFESSIONAL HRV VISUALIZATION FUNCTIONS
# =============================================================================


def _format_power(value: float, unit: str = "ms²") -> str:
    """Format power values smartly - show decimals for small values.

    Avoids showing "0 ms²" when actual value is e.g. 0.3 ms².
    """
    if value >= 10:
        return f"{value:.0f} {unit}"
    elif value >= 1:
        return f"{value:.1f} {unit}"
    elif value >= 0.1:
        return f"{value:.2f} {unit}"
    else:
        return f"{value:.3f} {unit}"


# Educational resources for HRV visualizations
VISUALIZATION_RESOURCES = {
    "tachogram": {
        "title": "Tachogram (RR Interval Plot)",
        "description": """
The tachogram displays beat-to-beat RR intervals over time. It's the primary visualization
for inspecting raw HRV data and identifying artifacts, trends, and patterns.

**What to look for:**
- **Stable baseline**: Healthy HRV shows regular oscillation around the mean
- **Sudden spikes/drops**: May indicate artifacts, ectopic beats, or missed detections
- **Trends**: Gradual changes may reflect autonomic shifts (e.g., relaxation, stress)
- **±1 SD band**: ~68% of intervals should fall within this range
- **±2 SD band**: ~95% of intervals should fall within this range
        """,
        "references": [
            ("Task Force (1996) - HRV Standards", "https://doi.org/10.1161/01.CIR.93.5.1043"),
            ("Shaffer & Ginsberg (2017) - HRV Overview", "https://doi.org/10.3389/fpubh.2017.00258"),
        ]
    },
    "poincare": {
        "title": "Poincaré Plot (Return Map)",
        "description": """
The Poincaré plot shows each RR interval against the next one (RR[n] vs RR[n+1]).
It visualizes short-term and long-term variability in a single view.

**Key measures:**
- **SD1** (perpendicular to identity line): Short-term, beat-to-beat variability
  - Reflects parasympathetic (vagal) activity
  - Related to RMSSD
- **SD2** (along identity line): Long-term variability
  - Reflects overall HRV including sympathetic influences
  - Related to SDNN
- **SD1/SD2 ratio**: Shape of the ellipse
  - Low ratio (<0.5): Reduced short-term variability
  - Normal ratio (0.5-1.0): Balanced variability
        """,
        "references": [
            ("Brennan et al. (2001) - Poincaré Plot Analysis", "https://doi.org/10.1109/10.959330"),
            ("Guzik et al. (2007) - Poincaré Plot Asymmetry", "https://doi.org/10.1088/0967-3334/28/3/N01"),
        ]
    },
    "frequency": {
        "title": "Power Spectral Density (Frequency Domain)",
        "description": """
Frequency domain analysis decomposes HRV into oscillatory components using spectral analysis.
Different frequency bands reflect different physiological mechanisms.

**Frequency bands:**
- **VLF (0.0033-0.04 Hz)**: Very Low Frequency
  - Thermoregulation, hormonal fluctuations
  - Requires long recordings (>5 min) for reliable estimation
- **LF (0.04-0.15 Hz)**: Low Frequency
  - Mixed sympathetic and parasympathetic activity
  - Baroreflex activity, blood pressure regulation
- **HF (0.15-0.4 Hz)**: High Frequency
  - Primarily parasympathetic (vagal) activity
  - Respiratory sinus arrhythmia

**LF/HF Ratio interpretation:**
- <1.0: Parasympathetic dominant
- 1.0-2.0: Balanced autonomic activity
- >2.0: Sympathetic dominant
        """,
        "references": [
            ("Task Force (1996) - Frequency Bands", "https://doi.org/10.1161/01.CIR.93.5.1043"),
            ("Laborde et al. (2017) - HRV and Cardiac Vagal Tone", "https://doi.org/10.3389/fpsyg.2017.00213"),
        ]
    },
    "hr_distribution": {
        "title": "Heart Rate Distribution",
        "description": """
The heart rate distribution histogram shows the frequency of different heart rate values
during the recording period.

**What to look for:**
- **Normal distribution**: Most healthy recordings show approximately normal distribution
- **Skewness**: May indicate periods of sustained high or low HR
- **Multiple peaks**: Could indicate distinct activity states or artifacts
- **Width (SD)**: Reflects overall HR variability

**Normal resting HR ranges:**
- Adults: 60-100 BPM (athletes may have lower)
- Well-trained athletes: 40-60 BPM
        """,
        "references": [
            ("Nunan et al. (2010) - Normal HR Values", "https://doi.org/10.1097/HJR.0b013e32833e4598"),
        ]
    },
}

# Color scheme for professional plots
PLOT_COLORS = {
    "primary": "#2E86AB",      # Blue - main data
    "secondary": "#A23B72",    # Magenta - secondary data
    "accent": "#F18F01",       # Orange - highlights
    "success": "#C73E1D",      # Red - alerts/artifacts
    "neutral": "#6C757D",      # Gray - grid/reference
    "background": "#FAFAFA",   # Light gray background
    "lf_band": "rgba(255, 193, 7, 0.3)",   # Yellow - LF band
    "hf_band": "rgba(46, 134, 171, 0.3)",  # Blue - HF band
    "vlf_band": "rgba(108, 117, 125, 0.2)",  # Gray - VLF band
}

# Reference values for HRV interpretation (Shaffer & Ginsberg, 2017; Nunan et al., 2010)
# These are population means for healthy adults at rest (5-minute recordings)
HRV_REFERENCE_VALUES = {
    "RMSSD": {
        "low": 20,      # Below this suggests reduced vagal tone
        "normal": 42,   # Population mean ~42 ms
        "high": 70,     # Above this indicates high vagal activity
        "unit": "ms",
        "interpretation": {
            "low": "Reduced parasympathetic activity",
            "normal": "Normal vagal tone",
            "high": "High parasympathetic activity",
        }
    },
    "SDNN": {
        "low": 50,      # Below this may indicate health risk
        "normal": 141,  # Population mean ~141 ms (24h), ~50 ms (5-min)
        "high": 200,
        "unit": "ms",
        "interpretation": {
            "low": "Reduced overall HRV",
            "normal": "Normal overall variability",
            "high": "High overall variability",
        }
    },
    "pNN50": {
        "low": 3,
        "normal": 20,
        "high": 40,
        "unit": "%",
    },
    "LF_HF": {
        "low": 0.5,     # Parasympathetic dominant
        "normal": 1.5,  # Balanced
        "high": 3.0,    # Sympathetic dominant
        "unit": "",
    },
}

# Minimum data requirements per Quigley et al. (2024)
MIN_BEATS_TIME_DOMAIN = 100
MIN_BEATS_FREQUENCY_DOMAIN = 300
MIN_DURATION_FREQUENCY_DOMAIN_SEC = 120  # 2 minutes minimum, 5 minutes recommended


def get_theme_colors():
    """Get colors for chart rendering.

    Always returns light theme colors to match config.toml base theme.
    Dark mode is handled by JavaScript updatePlotlyTheme() function
    which updates charts dynamically when user switches themes.
    """
    # Always use light theme for initial render (matches config.toml)
    # JavaScript handles dark mode switching dynamically
    return {
        'bg': '#FFFFFF',
        'text': '#31333F',
        'grid': 'rgba(0,0,0,0.1)',
    }


def create_professional_tachogram(rr_intervals: list, section_label: str,
                                   artifact_indices: list = None):
    """Create a professional tachogram with clean layout.

    Features:
    - RR intervals as connected scatter plot
    - Mean line with ±1 SD and ±2 SD bands
    - Artifact markers if provided
    - Professional styling with legend below plot

    Returns:
        Tuple of (figure, stats_dict) for external display of statistics
    """
    go, _ = get_plotly_analysis()
    if go is None:
        return None, {}

    rr = np.array(rr_intervals)
    n_beats = len(rr)

    # Calculate statistics
    mean_rr = np.mean(rr)
    std_rr = np.std(rr)
    min_rr = np.min(rr)
    max_rr = np.max(rr)
    mean_hr = 60000 / mean_rr

    # Stats for external display
    stats = {
        "N beats": n_beats,
        "Mean RR": f"{mean_rr:.1f} ms",
        "SD": f"{std_rr:.1f} ms",
        "Mean HR": f"{mean_hr:.1f} bpm",
        "Range": f"{min_rr:.0f}–{max_rr:.0f} ms",
    }

    # Create figure
    fig = go.Figure()

    # Add ±2 SD band (lighter)
    fig.add_trace(go.Scatter(
        x=list(range(n_beats)) + list(range(n_beats-1, -1, -1)),
        y=[mean_rr + 2*std_rr] * n_beats + [mean_rr - 2*std_rr] * n_beats,
        fill='toself',
        fillcolor='rgba(46, 134, 171, 0.1)',
        line=dict(width=0),
        name='±2 SD',
        hoverinfo='skip',
        showlegend=True
    ))

    # Add ±1 SD band (darker)
    fig.add_trace(go.Scatter(
        x=list(range(n_beats)) + list(range(n_beats-1, -1, -1)),
        y=[mean_rr + std_rr] * n_beats + [mean_rr - std_rr] * n_beats,
        fill='toself',
        fillcolor='rgba(46, 134, 171, 0.2)',
        line=dict(width=0),
        name='±1 SD',
        hoverinfo='skip',
        showlegend=True
    ))

    # Add mean line
    fig.add_trace(go.Scatter(
        x=[0, n_beats-1],
        y=[mean_rr, mean_rr],
        mode='lines',
        line=dict(color=PLOT_COLORS["accent"], width=2, dash='dash'),
        name=f'Mean ({mean_rr:.0f} ms)',
        hoverinfo='name'
    ))

    # Add RR intervals
    fig.add_trace(go.Scattergl(
        x=list(range(n_beats)),
        y=rr.tolist(),
        mode='lines+markers',
        marker=dict(size=3, color=PLOT_COLORS["primary"]),
        line=dict(width=1, color=PLOT_COLORS["primary"]),
        name='RR Intervals',
        hovertemplate='Beat %{x}<br>RR: %{y:.0f} ms<br>HR: %{customdata:.1f} bpm<extra></extra>',
        customdata=60000/rr
    ))

    # Add artifact markers if provided
    if artifact_indices and len(artifact_indices) > 0:
        artifact_rr = [rr[i] for i in artifact_indices if i < len(rr)]
        fig.add_trace(go.Scatter(
            x=artifact_indices,
            y=artifact_rr,
            mode='markers',
            marker=dict(size=10, color=PLOT_COLORS["success"], symbol='x', line=dict(width=2)),
            name=f'Artifacts ({len(artifact_indices)})',
            hovertemplate='Artifact at beat %{x}<br>RR: %{y:.0f} ms<extra></extra>'
        ))
        stats["Artifacts"] = len(artifact_indices)

    # Update layout - legend below plot
    theme = get_theme_colors()
    fig.update_layout(
        title=dict(
            text=f"<b>Tachogram</b> — {section_label}",
            font=dict(size=16, color=theme['text'])
        ),
        xaxis=dict(
            title=dict(text="Beat Number", font=dict(color=theme['text'])),
            showgrid=True,
            gridcolor=theme['grid'],
            zeroline=False,
            tickfont=dict(color=theme['text'])
        ),
        yaxis=dict(
            title=dict(text="RR Interval (ms)", font=dict(color=theme['text'])),
            showgrid=True,
            gridcolor=theme['grid'],
            zeroline=False,
            tickfont=dict(color=theme['text'])
        ),
        height=400,
        margin=dict(l=60, r=20, t=50, b=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color=theme['text'])
        ),
        hovermode='x unified',
        plot_bgcolor=theme['bg'],
        paper_bgcolor=theme['bg'],
        font=dict(color=theme['text'])
    )

    return fig, stats


def create_poincare_plot(rr_intervals: list, section_label: str):
    """Create a Poincaré plot (RR[n] vs RR[n+1]) with SD1/SD2 ellipse.

    The Poincaré plot visualizes short-term (SD1) and long-term (SD2) HRV.
    - SD1: Perpendicular to identity line - short-term variability
    - SD2: Along identity line - long-term variability

    Returns:
        Tuple of (figure, stats_dict) for external display of statistics
    """
    go, _ = get_plotly_analysis()
    if go is None:
        return None, {}

    rr = np.array(rr_intervals)
    rr_n = rr[:-1]   # RR[n]
    rr_n1 = rr[1:]   # RR[n+1]

    # Calculate SD1 and SD2
    diff_rr = rr_n1 - rr_n
    sum_rr = rr_n1 + rr_n

    sd1 = np.std(diff_rr) / np.sqrt(2)
    sd2 = np.std(sum_rr) / np.sqrt(2)
    sd_ratio = sd1/sd2 if sd2 > 0 else 0

    # Stats for external display
    stats = {
        "SD1 (short-term)": f"{sd1:.1f} ms",
        "SD2 (long-term)": f"{sd2:.1f} ms",
        "SD1/SD2": f"{sd_ratio:.2f}",
        "N pairs": len(rr_n),
    }

    # Center of ellipse
    center_x = np.mean(rr_n)
    center_y = np.mean(rr_n1)

    # Create ellipse points (rotated 45 degrees)
    theta = np.linspace(0, 2*np.pi, 100)
    a = sd2  # Semi-major axis (along identity line)
    b = sd1  # Semi-minor axis (perpendicular to identity line)

    cos_45 = np.cos(np.pi/4)
    sin_45 = np.sin(np.pi/4)

    ellipse_x = center_x + a * np.cos(theta) * cos_45 - b * np.sin(theta) * sin_45
    ellipse_y = center_y + a * np.cos(theta) * sin_45 + b * np.sin(theta) * cos_45

    # Create figure
    fig = go.Figure()

    # Add identity line
    min_val = min(np.min(rr_n), np.min(rr_n1)) - 50
    max_val = max(np.max(rr_n), np.max(rr_n1)) + 50
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color=PLOT_COLORS["neutral"], width=1, dash='dash'),
        name='Identity Line',
        hoverinfo='skip'
    ))

    # Add SD1/SD2 ellipse
    fig.add_trace(go.Scatter(
        x=ellipse_x.tolist(),
        y=ellipse_y.tolist(),
        mode='lines',
        fill='toself',
        fillcolor='rgba(46, 134, 171, 0.2)',
        line=dict(color=PLOT_COLORS["primary"], width=2),
        name='SD1/SD2 Ellipse',
        hoverinfo='name'
    ))

    # Add scatter points
    fig.add_trace(go.Scattergl(
        x=rr_n.tolist(),
        y=rr_n1.tolist(),
        mode='markers',
        marker=dict(
            size=5,
            color=PLOT_COLORS["primary"],
            opacity=0.6
        ),
        name='RR pairs',
        hovertemplate='RR[n]: %{x:.0f} ms<br>RR[n+1]: %{y:.0f} ms<extra></extra>'
    ))

    # Add center point
    fig.add_trace(go.Scatter(
        x=[center_x],
        y=[center_y],
        mode='markers',
        marker=dict(size=12, color=PLOT_COLORS["accent"], symbol='cross'),
        name='Center',
        hovertemplate=f'Center<br>RR[n]: {center_x:.0f} ms<br>RR[n+1]: {center_y:.0f} ms<extra></extra>'
    ))

    # Add SD1 line (perpendicular to identity - short-term variability)
    sd1_x = [center_x - sd1 * sin_45, center_x + sd1 * sin_45]
    sd1_y = [center_y + sd1 * cos_45, center_y - sd1 * cos_45]
    fig.add_trace(go.Scatter(
        x=sd1_x,
        y=sd1_y,
        mode='lines',
        line=dict(color='#e74c3c', width=2),
        name=f'SD1 = {sd1:.1f} ms',
        hoverinfo='name'
    ))

    # Add SD2 line (along identity - long-term variability)
    sd2_x = [center_x - sd2 * cos_45, center_x + sd2 * cos_45]
    sd2_y = [center_y - sd2 * sin_45, center_y + sd2 * sin_45]
    fig.add_trace(go.Scatter(
        x=sd2_x,
        y=sd2_y,
        mode='lines',
        line=dict(color='#3498db', width=2),
        name=f'SD2 = {sd2:.1f} ms',
        hoverinfo='name'
    ))

    # Update layout - legend below plot
    theme = get_theme_colors()
    fig.update_layout(
        title=dict(
            text=f"<b>Poincaré Plot</b> — {section_label}",
            font=dict(size=16, color=theme['text'])
        ),
        xaxis=dict(
            title=dict(text="RR[n] (ms)", font=dict(color=theme['text'])),
            showgrid=True,
            gridcolor=theme['grid'],
            scaleanchor="y",
            scaleratio=1,
            tickfont=dict(color=theme['text'])
        ),
        yaxis=dict(
            title=dict(text="RR[n+1] (ms)", font=dict(color=theme['text'])),
            showgrid=True,
            gridcolor=theme['grid'],
            tickfont=dict(color=theme['text'])
        ),
        height=500,
        margin=dict(l=60, r=20, t=50, b=100),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(color=theme['text'], size=11)
        ),
        plot_bgcolor=theme['bg'],
        paper_bgcolor=theme['bg'],
        font=dict(color=theme['text'])
    )

    return fig, stats


def create_frequency_domain_plot(rr_intervals: list, section_label: str,
                                  sampling_rate: int = 4):
    """Create a power spectral density plot with frequency bands highlighted.

    Frequency bands (standard):
    - VLF: 0.0033-0.04 Hz (very low frequency)
    - LF: 0.04-0.15 Hz (low frequency - sympathetic + parasympathetic)
    - HF: 0.15-0.4 Hz (high frequency - parasympathetic/vagal)

    Returns:
        Tuple of (figure, stats_dict) for external display, or (None, None) on error
    """
    go, _ = get_plotly_analysis()
    if go is None:
        return None, None

    nk = get_neurokit()
    if nk is None:
        return None, None

    try:
        rr = np.array(rr_intervals)

        # Interpolate RR intervals to uniform time series
        time_rr = np.cumsum(rr) / 1000.0
        time_rr = time_rr - time_rr[0]

        duration = time_rr[-1]
        time_uniform = np.arange(0, duration, 1/sampling_rate)

        rr_interp = np.interp(time_uniform, time_rr, rr)
        rr_detrend = rr_interp - np.mean(rr_interp)

        # Compute PSD using Welch's method
        from scipy import signal
        nperseg = min(256, len(rr_detrend) // 2)
        if nperseg < 16:
            nperseg = len(rr_detrend)

        freqs, psd = signal.welch(rr_detrend, fs=sampling_rate, nperseg=nperseg)

        # Filter to relevant frequency range
        mask = freqs <= 0.5
        freqs = freqs[mask]
        psd = psd[mask]

        # Calculate band powers
        vlf_mask = (freqs >= 0.0033) & (freqs < 0.04)
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.4)

        vlf_power = np.trapz(psd[vlf_mask], freqs[vlf_mask]) if np.any(vlf_mask) else 0
        lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
        hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
        total_power = vlf_power + lf_power + hf_power

        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
        lf_pct = 100*lf_power/total_power if total_power > 0 else 0
        hf_pct = 100*hf_power/total_power if total_power > 0 else 0

        # Stats for external display (tuples for values with percentage delta)
        stats = {
            "VLF Power": _format_power(vlf_power),
            "LF Power": (_format_power(lf_power), f"{lf_pct:.0f}%"),
            "HF Power": (_format_power(hf_power), f"{hf_pct:.0f}%"),
            "LF/HF Ratio": f"{lf_hf_ratio:.2f}",
            "Total Power": _format_power(total_power),
        }

        # Create figure
        fig = go.Figure()

        max_psd = np.max(psd) * 1.1

        # VLF band with label
        fig.add_trace(go.Scatter(
            x=[0.0033, 0.04, 0.04, 0.0033],
            y=[0, 0, max_psd, max_psd],
            fill='toself',
            fillcolor=PLOT_COLORS["vlf_band"],
            line=dict(width=0),
            name=f'VLF ({_format_power(vlf_power)})',
            hoverinfo='name'
        ))

        # LF band with label
        fig.add_trace(go.Scatter(
            x=[0.04, 0.15, 0.15, 0.04],
            y=[0, 0, max_psd, max_psd],
            fill='toself',
            fillcolor=PLOT_COLORS["lf_band"],
            line=dict(width=0),
            name=f'LF ({_format_power(lf_power)}, {lf_pct:.0f}%)',
            hoverinfo='name'
        ))

        # HF band with label
        fig.add_trace(go.Scatter(
            x=[0.15, 0.4, 0.4, 0.15],
            y=[0, 0, max_psd, max_psd],
            fill='toself',
            fillcolor=PLOT_COLORS["hf_band"],
            line=dict(width=0),
            name=f'HF ({_format_power(hf_power)}, {hf_pct:.0f}%)',
            hoverinfo='name'
        ))

        # Add PSD line
        fig.add_trace(go.Scatter(
            x=freqs.tolist(),
            y=psd.tolist(),
            mode='lines',
            line=dict(color=PLOT_COLORS["primary"], width=2.5),
            name='PSD',
            hovertemplate='Freq: %{x:.3f} Hz<br>Power: %{y:.1f} ms²/Hz<extra></extra>'
        ))

        # Add band boundary lines and labels
        for freq, label in [(0.04, 'VLF|LF'), (0.15, 'LF|HF'), (0.4, 'HF')]:
            fig.add_vline(
                x=freq, line=dict(color='gray', width=1, dash='dot'),
                annotation_text=f'{freq}',
                annotation_position='top',
                annotation=dict(font_size=9, font_color='gray')
            )

        # Update layout - legend below plot
        theme = get_theme_colors()
        fig.update_layout(
            title=dict(
                text=f"<b>Power Spectral Density</b> — {section_label}",
                font=dict(size=16, color=theme['text'])
            ),
            xaxis=dict(
                title=dict(text="Frequency (Hz)", font=dict(color=theme['text'])),
                showgrid=True,
                gridcolor=theme['grid'],
                range=[0, 0.5],
                tickvals=[0, 0.04, 0.15, 0.4, 0.5],
                ticktext=['0', '0.04', '0.15', '0.4', '0.5'],
                tickfont=dict(color=theme['text'])
            ),
            yaxis=dict(
                title=dict(text="Power (ms²/Hz)", font=dict(color=theme['text'])),
                showgrid=True,
                gridcolor=theme['grid'],
                rangemode='tozero',
                tickfont=dict(color=theme['text'])
            ),
            height=420,
            margin=dict(l=60, r=20, t=50, b=90),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="center",
                x=0.5,
                font=dict(color=theme['text'])
            ),
            plot_bgcolor=theme['bg'],
            paper_bgcolor=theme['bg'],
            font=dict(color=theme['text'])
        )

        return fig, stats

    except Exception as e:
        st.warning(f"Could not create frequency plot: {e}")
        return None, None


def create_hr_distribution_plot(rr_intervals: list, section_label: str):
    """Create a heart rate distribution histogram with density curve.

    Returns:
        Tuple of (figure, stats_dict) for external display of statistics
    """
    go, make_subplots = get_plotly_analysis()
    if go is None:
        return None, {}

    rr = np.array(rr_intervals)
    hr = 60000 / rr  # Convert to beats per minute

    # Calculate statistics
    mean_hr = np.mean(hr)
    std_hr = np.std(hr)
    min_hr = np.min(hr)
    max_hr = np.max(hr)

    # Stats for external display
    stats = {
        "Mean HR": f"{mean_hr:.1f} bpm",
        "SD": f"{std_hr:.1f} bpm",
        "Min": f"{min_hr:.0f} bpm",
        "Max": f"{max_hr:.0f} bpm",
        "Range": f"{max_hr-min_hr:.0f} bpm",
    }

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=hr.tolist(),
            nbinsx=30,
            name='Distribution',
            marker_color=PLOT_COLORS["primary"],
            opacity=0.7,
            hovertemplate='HR: %{x:.0f} bpm<br>Count: %{y}<extra></extra>'
        ),
        secondary_y=False
    )

    # Add KDE (kernel density estimate) curve
    try:
        from scipy import stats as sp_stats
        kde = sp_stats.gaussian_kde(hr)
        x_kde = np.linspace(min_hr - 5, max_hr + 5, 200)
        y_kde = kde(x_kde)
        y_kde_scaled = y_kde * len(hr) * (max_hr - min_hr) / 30

        fig.add_trace(
            go.Scatter(
                x=x_kde.tolist(),
                y=y_kde_scaled.tolist(),
                mode='lines',
                name='Density',
                line=dict(color=PLOT_COLORS["secondary"], width=3),
                hoverinfo='skip'
            ),
            secondary_y=False
        )
    except ImportError:
        pass

    # Add mean line
    fig.add_vline(
        x=mean_hr,
        line=dict(color=PLOT_COLORS["accent"], width=2, dash='dash'),
        annotation_text=f"Mean: {mean_hr:.1f}",
        annotation_position="top"
    )

    # Update layout - legend below plot
    theme = get_theme_colors()
    fig.update_layout(
        title=dict(
            text=f"<b>Heart Rate Distribution</b> — {section_label}",
            font=dict(size=16, color=theme['text'])
        ),
        xaxis=dict(
            title=dict(text="Heart Rate (bpm)", font=dict(color=theme['text'])),
            showgrid=True,
            gridcolor=theme['grid'],
            tickfont=dict(color=theme['text'])
        ),
        yaxis=dict(
            title=dict(text="Count", font=dict(color=theme['text'])),
            showgrid=True,
            gridcolor=theme['grid'],
            tickfont=dict(color=theme['text'])
        ),
        height=350,
        margin=dict(l=60, r=20, t=50, b=90),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(color=theme['text'])
        ),
        plot_bgcolor=theme['bg'],
        paper_bgcolor=theme['bg'],
        font=dict(color=theme['text']),
        bargap=0.05
    )

    return fig, stats


def display_hrv_metrics_professional(hrv_results: pd.DataFrame, n_beats: int,
                                      artifact_info: dict = None,
                                      recording_duration_sec: float = None) -> None:
    """Display HRV metrics using pure Streamlit native components.

    Clean, professional design that works in both light and dark modes.
    Uses only native Streamlit components - no custom HTML/CSS.
    """

    # Extract key metrics
    rmssd = hrv_results.get('HRV_RMSSD', [0]).iloc[0] if 'HRV_RMSSD' in hrv_results.columns else 0
    sdnn = hrv_results.get('HRV_SDNN', [0]).iloc[0] if 'HRV_SDNN' in hrv_results.columns else 0
    pnn50 = hrv_results.get('HRV_pNN50', [0]).iloc[0] if 'HRV_pNN50' in hrv_results.columns else 0
    lf_hf = hrv_results.get('HRV_LFHF', [0]).iloc[0] if 'HRV_LFHF' in hrv_results.columns else 0
    hf = hrv_results.get('HRV_HF', [0]).iloc[0] if 'HRV_HF' in hrv_results.columns else 0
    lf = hrv_results.get('HRV_LF', [0]).iloc[0] if 'HRV_LF' in hrv_results.columns else 0
    mean_hr = hrv_results.get('HRV_MeanNN', [0]).iloc[0] if 'HRV_MeanNN' in hrv_results.columns else 0
    if mean_hr > 0:
        mean_hr_bpm = 60000 / mean_hr  # Convert ms to BPM
    else:
        mean_hr_bpm = 0

    # Calculate total power for percentages
    total_power = lf + hf if (lf + hf) > 0 else 1
    lf_pct = (lf / total_power) * 100
    hf_pct = (hf / total_power) * 100

    # Duration display
    duration_min = recording_duration_sec / 60 if recording_duration_sec else 0

    # Data quality assessment
    quality_issues = []
    if n_beats < MIN_BEATS_TIME_DOMAIN:
        quality_issues.append(f"Low beat count: {n_beats} (min: {MIN_BEATS_TIME_DOMAIN})")
    if n_beats < MIN_BEATS_FREQUENCY_DOMAIN:
        quality_issues.append(f"Insufficient for frequency domain: {n_beats}/{MIN_BEATS_FREQUENCY_DOMAIN} beats")
    if recording_duration_sec and recording_duration_sec < MIN_DURATION_FREQUENCY_DOMAIN_SEC:
        quality_issues.append(f"Short recording: {duration_min:.1f} min (recommended: ≥5 min)")

    # === RECORDING SUMMARY ===
    st.markdown("##### Recording Summary")

    # Summary metrics row
    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    with sum_col1:
        st.metric("Total Beats", f"{n_beats:,}")
    with sum_col2:
        st.metric("Duration", f"{duration_min:.1f} min")
    with sum_col3:
        st.metric("Mean HR", f"{mean_hr_bpm:.0f} BPM")
    with sum_col4:
        if artifact_info:
            artifact_pct = artifact_info.get('artifact_ratio', 0) * 100
            st.metric("Artifacts", f"{artifact_pct:.1f}%")
        else:
            # Data quality indicator
            if not quality_issues:
                st.metric("Data Quality", "Good", delta="*", delta_color="normal")
            elif len(quality_issues) == 1:
                st.metric("Data Quality", "Fair", delta="!", delta_color="off")
            else:
                st.metric("Data Quality", "Limited", delta="(!)", delta_color="inverse")

    # Show quality warnings if any
    if quality_issues:
        with st.expander("Data Quality Notes", expanded=False):
            for issue in quality_issues:
                st.warning(issue)

    st.divider()

    # === TIME DOMAIN METRICS ===
    st.markdown("##### Time Domain")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        rmssd_ref = HRV_REFERENCE_VALUES["RMSSD"]
        # Interpretation
        if rmssd >= rmssd_ref["high"]:
            delta_text = "High"
            delta_color = "normal"
        elif rmssd >= rmssd_ref["low"]:
            delta_text = "Normal"
            delta_color = "off"
        else:
            delta_text = "Low"
            delta_color = "inverse"
        st.metric(
            "RMSSD",
            f"{rmssd:.1f} ms",
            delta=delta_text,
            delta_color=delta_color,
            help=f"Parasympathetic indicator. Reference: {rmssd_ref['low']}–{rmssd_ref['high']} ms"
        )

    with col2:
        sdnn_ref = HRV_REFERENCE_VALUES["SDNN"]
        if sdnn >= sdnn_ref["low"]:
            delta_text = "Normal"
            delta_color = "off"
        else:
            delta_text = "Low"
            delta_color = "inverse"
        st.metric(
            "SDNN",
            f"{sdnn:.1f} ms",
            delta=delta_text,
            delta_color=delta_color,
            help=f"Overall HRV. Reference: ≥{sdnn_ref['low']} ms"
        )

    with col3:
        pnn_ref = HRV_REFERENCE_VALUES["pNN50"]
        if pnn50 >= pnn_ref["high"]:
            delta_text = "High"
            delta_color = "normal"
        elif pnn50 >= pnn_ref["low"]:
            delta_text = "Normal"
            delta_color = "off"
        else:
            delta_text = "Low"
            delta_color = "inverse"
        st.metric(
            "pNN50",
            f"{pnn50:.1f}%",
            delta=delta_text,
            delta_color=delta_color,
            help=f"% of RR differences >50ms. Reference: {pnn_ref['low']}–{pnn_ref['high']}%"
        )

    with col4:
        # Heart rate interpretation
        if 60 <= mean_hr_bpm <= 100:
            delta_text = "Normal"
            delta_color = "off"
        elif mean_hr_bpm < 60:
            delta_text = "Bradycardia"
            delta_color = "off"
        else:
            delta_text = "Elevated"
            delta_color = "off"
        st.metric(
            "Mean HR",
            f"{mean_hr_bpm:.0f} BPM",
            delta=delta_text,
            delta_color=delta_color,
            help="Average heart rate. Normal resting: 60–100 BPM"
        )

    st.divider()

    # === FREQUENCY DOMAIN METRICS ===
    st.markdown("##### Frequency Domain")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "LF Power",
            _format_power(lf),
            delta=f"{lf_pct:.0f}%",
            delta_color="off",
            help="Low Frequency (0.04–0.15 Hz). Mixed sympathetic/parasympathetic."
        )

    with col2:
        st.metric(
            "HF Power",
            _format_power(hf),
            delta=f"{hf_pct:.0f}%",
            delta_color="off",
            help="High Frequency (0.15–0.4 Hz). Parasympathetic/vagal activity."
        )

    with col3:
        lf_hf_ref = HRV_REFERENCE_VALUES["LF_HF"]
        if lf_hf < lf_hf_ref["low"]:
            lfhf_delta = "PNS dominant"
        elif lf_hf < lf_hf_ref["high"]:
            lfhf_delta = "Balanced"
        else:
            lfhf_delta = "SNS dominant"
        st.metric(
            "LF/HF Ratio",
            f"{lf_hf:.2f}",
            delta=lfhf_delta,
            delta_color="off",
            help="Sympathovagal balance. <0.5: PNS dominant, 0.5–3.0: Balanced, >3.0: SNS dominant"
        )

    # Autonomic balance indicator using progress bar
    st.caption("Autonomic Balance")
    balance_col1, balance_col2, balance_col3 = st.columns([1, 3, 1])
    with balance_col1:
        st.caption("PNS")
    with balance_col2:
        # Use HF percentage as indicator (higher = more parasympathetic)
        st.progress(min(1.0, hf_pct / 100))
    with balance_col3:
        st.caption("SNS")


def create_hrv_metrics_card(hrv_results: pd.DataFrame, n_beats: int,
                            artifact_info: dict = None,
                            recording_duration_sec: float = None) -> str:
    """Legacy function - returns empty string. Use display_hrv_metrics_professional() instead."""
    return ""


def display_visualization_info(viz_type: str) -> None:
    """Display educational information about a visualization type.

    Args:
        viz_type: One of 'tachogram', 'poincare', 'frequency', 'hr_distribution'
    """
    if viz_type not in VISUALIZATION_RESOURCES:
        return

    info = VISUALIZATION_RESOURCES[viz_type]

    with st.expander(f"About: {info['title']}", expanded=False):
        st.markdown(info["description"])

        if info.get("references"):
            st.markdown("**References:**")
            for title, url in info["references"]:
                st.markdown(f"- [{title}]({url})")


class AnalysisDocumentation:
    """Generates documentation for HRV analysis procedures.

    This class captures all analysis parameters and generates a markdown
    report that can be exported for reproducibility and publication.
    """

    def __init__(self, participant_id: str):
        self.participant_id = participant_id
        self.timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        self.sections_analyzed = []
        self.cleaning_config = {}
        self.artifact_correction = False
        self.artifact_results = {}
        self.exclusion_zones = []
        self.hrv_results = {}
        self.data_source = ""
        self.total_beats_raw = 0
        self.total_beats_analyzed = 0
        self.recording_duration_sec = 0

    def set_data_source(self, source: str, raw_beats: int, duration_sec: float):
        """Set data source information."""
        self.data_source = source
        self.total_beats_raw = raw_beats
        self.recording_duration_sec = duration_sec

    def set_cleaning_config(self, config):
        """Set cleaning configuration used."""
        if config is None:
            self.cleaning_config = {}
        elif isinstance(config, dict):
            self.cleaning_config = config.copy()
        elif hasattr(config, '__dict__'):
            # Handle dataclass or object with attributes
            self.cleaning_config = {
                "rr_min_ms": getattr(config, 'rr_min_ms', 200),
                "rr_max_ms": getattr(config, 'rr_max_ms', 2000),
                "sudden_change_pct": getattr(config, 'sudden_change_pct', 100),
            }
        else:
            self.cleaning_config = {}

    def set_artifact_correction(self, enabled: bool, results: dict = None):
        """Set artifact correction settings."""
        self.artifact_correction = enabled
        if results:
            self.artifact_results = results.copy()

    def add_section(self, name: str, label: str, start_event: str, end_events: list,
                    beats_extracted: int, beats_after_cleaning: int):
        """Add a section to the documentation."""
        self.sections_analyzed.append({
            "name": name,
            "label": label,
            "start_event": start_event,
            "end_events": end_events,
            "beats_extracted": beats_extracted,
            "beats_after_cleaning": beats_after_cleaning,
        })
        self.total_beats_analyzed += beats_after_cleaning

    def add_exclusion_zones(self, zones: list):
        """Add exclusion zones used."""
        self.exclusion_zones = zones.copy() if zones else []

    def add_hrv_results(self, section_name: str, results: pd.DataFrame):
        """Add HRV results for a section."""
        if not results.empty:
            self.hrv_results[section_name] = results.to_dict('records')[0]

    def generate_markdown(self) -> str:
        """Generate a complete markdown documentation report."""
        lines = []

        # Header
        lines.append("# HRV Analysis Report")
        lines.append("")
        lines.append(f"**Participant:** {self.participant_id}")
        lines.append(f"**Generated:** {self.timestamp}")
        lines.append("**Software:** Music HRV Toolkit v0.6.8")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Data Source
        lines.append("## 1. Data Source")
        lines.append("")
        lines.append(f"- **Source Application:** {self.data_source}")
        lines.append(f"- **Total Raw Beats:** {self.total_beats_raw:,}")
        lines.append(f"- **Recording Duration:** {self.recording_duration_sec/60:.1f} minutes")
        lines.append("")

        # Data Preparation
        lines.append("## 2. Data Preparation")
        lines.append("")
        lines.append("### 2.1 Cleaning Thresholds")
        lines.append("")
        if self.cleaning_config:
            lines.append("| Parameter | Value |")
            lines.append("|-----------|-------|")
            lines.append(f"| Minimum RR | {self.cleaning_config.get('rr_min_ms', 200)} ms |")
            lines.append(f"| Maximum RR | {self.cleaning_config.get('rr_max_ms', 2000)} ms |")
            lines.append(f"| Sudden Change Threshold | {self.cleaning_config.get('sudden_change_pct', 100)}% |")
        else:
            lines.append("*Default cleaning thresholds applied (200-2000 ms)*")
        lines.append("")

        # Exclusion Zones
        if self.exclusion_zones:
            lines.append("### 2.2 Exclusion Zones")
            lines.append("")
            lines.append("| Start | End | Reason |")
            lines.append("|-------|-----|--------|")
            for zone in self.exclusion_zones:
                start = zone.get('start', 'N/A')
                end = zone.get('end', 'N/A')
                reason = zone.get('reason', 'Not specified')
                lines.append(f"| {start} | {end} | {reason} |")
            lines.append("")

        # Artifact Correction
        lines.append("### 2.3 Artifact Correction")
        lines.append("")
        if self.artifact_correction:
            lines.append("- **Method:** NeuroKit2 Kubios Algorithm")
            lines.append("- **Status:** Applied")
            if self.artifact_results:
                lines.append(f"- **Artifacts Detected:** {self.artifact_results.get('total_artifacts', 'N/A')}")
                lines.append(f"- **Artifact Rate:** {self.artifact_results.get('artifact_ratio', 0)*100:.1f}%")
                if 'artifact_types' in self.artifact_results:
                    lines.append("- **Artifact Types:**")
                    for atype, count in self.artifact_results['artifact_types'].items():
                        lines.append(f"  - {atype}: {count}")
        else:
            lines.append("- **Status:** Not applied (raw RR intervals used)")
        lines.append("")

        # Sections Analyzed
        lines.append("## 3. Sections Analyzed")
        lines.append("")
        if self.sections_analyzed:
            for section in self.sections_analyzed:
                lines.append(f"### {section['label']}")
                lines.append("")
                lines.append(f"- **Section Name:** {section['name']}")
                lines.append(f"- **Start Event:** `{section['start_event']}`")
                lines.append(f"- **End Event(s):** `{', '.join(section['end_events'])}`")
                lines.append(f"- **Beats Extracted:** {section['beats_extracted']:,}")
                lines.append(f"- **Beats After Cleaning:** {section['beats_after_cleaning']:,}")
                lines.append(f"- **Data Retention:** {100*section['beats_after_cleaning']/max(section['beats_extracted'],1):.1f}%")
                lines.append("")
        else:
            lines.append("*No sections analyzed*")
            lines.append("")

        # HRV Results
        lines.append("## 4. HRV Results")
        lines.append("")
        if self.hrv_results:
            for section_name, results in self.hrv_results.items():
                label = section_name if section_name != "_combined" else "Combined Sections"
                lines.append(f"### {label}")
                lines.append("")
                lines.append("#### Time Domain")
                lines.append("")
                lines.append("| Metric | Value | Unit |")
                lines.append("|--------|-------|------|")
                if 'HRV_RMSSD' in results:
                    lines.append(f"| RMSSD | {results['HRV_RMSSD']:.2f} | ms |")
                if 'HRV_SDNN' in results:
                    lines.append(f"| SDNN | {results['HRV_SDNN']:.2f} | ms |")
                if 'HRV_pNN50' in results:
                    lines.append(f"| pNN50 | {results['HRV_pNN50']:.2f} | % |")
                if 'HRV_MeanNN' in results:
                    mean_hr = 60000 / results['HRV_MeanNN'] if results['HRV_MeanNN'] > 0 else 0
                    lines.append(f"| Mean NN | {results['HRV_MeanNN']:.2f} | ms |")
                    lines.append(f"| Mean HR | {mean_hr:.1f} | BPM |")
                lines.append("")

                lines.append("#### Frequency Domain")
                lines.append("")
                lines.append("| Metric | Value | Unit |")
                lines.append("|--------|-------|------|")
                if 'HRV_LF' in results:
                    lines.append(f"| LF Power | {results['HRV_LF']:.2f} | ms² |")
                if 'HRV_HF' in results:
                    lines.append(f"| HF Power | {results['HRV_HF']:.2f} | ms² |")
                if 'HRV_LFHF' in results:
                    lines.append(f"| LF/HF Ratio | {results['HRV_LFHF']:.2f} | - |")
                lines.append("")
        else:
            lines.append("*No HRV results available*")
            lines.append("")

        # Methods Summary
        lines.append("## 5. Methods Summary")
        lines.append("")
        lines.append("### For Publication")
        lines.append("")
        artifact_text = "with Kubios artifact correction (NeuroKit2)" if self.artifact_correction else "without artifact correction"
        sections_text = ", ".join([s['label'] for s in self.sections_analyzed]) if self.sections_analyzed else "all data"

        lines.append("> HRV analysis was performed using Music HRV Toolkit (v0.6.8). ")
        lines.append(f"> RR intervals were extracted from {self.data_source} recordings ")
        lines.append(f"> and cleaned using threshold filtering (RR: {self.cleaning_config.get('rr_min_ms', 200)}-{self.cleaning_config.get('rr_max_ms', 2000)} ms). ")
        if self.exclusion_zones:
            lines.append(f"> {len(self.exclusion_zones)} exclusion zone(s) were applied to remove artifacts. ")
        lines.append(f"> Time-domain and frequency-domain HRV metrics were computed {artifact_text} ")
        lines.append("> using NeuroKit2 (Makowski et al., 2021). ")
        lines.append(f"> Analysis was performed on the following section(s): {sections_text}.")
        lines.append("")

        # References
        lines.append("## 6. References")
        lines.append("")
        lines.append("- Makowski, D., et al. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. *Behavior Research Methods*. https://doi.org/10.3758/s13428-020-01516-y")
        lines.append("- Task Force of ESC and NASPE (1996). Heart rate variability: Standards of measurement. *Circulation*, 93(5), 1043-1065.")
        lines.append("- Quigley, K. S., et al. (2024). Publication guidelines for heart rate variability studies. *Psychophysiology*, 61(9), e14604.")
        lines.append("")

        lines.append("---")
        lines.append("*Report generated by Music HRV Toolkit*")

        return "\n".join(lines)


def display_documentation_panel(doc: AnalysisDocumentation) -> None:
    """Display the analysis documentation panel with preview and export."""
    st.markdown("---")
    st.subheader("Analysis Documentation")

    with st.expander("Preview & Export Analysis Report", expanded=False):
        st.markdown("""
        This report documents all analysis parameters for reproducibility.
        Export as Markdown (.md) to include in your research documentation.
        """)

        # Generate markdown
        md_content = doc.generate_markdown()

        # Preview tabs
        preview_tab, raw_tab = st.tabs(["Preview", "Raw Markdown"])

        with preview_tab:
            st.markdown(md_content)

        with raw_tab:
            st.code(md_content, language="markdown")

        # Download button
        st.download_button(
            label="Download Report (.md)",
            data=md_content,
            file_name=f"hrv_analysis_{doc.participant_id}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            key=f"download_doc_{doc.participant_id}",
        )


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
    from rrational.analysis.music_sections import (
        ProtocolConfig,
        DurationMismatchStrategy,
        extract_music_sections,
        get_sections_by_music_type,
    )
    from rrational.gui.persistence import load_protocol, save_protocol

    st.markdown("""
    Analyze HRV metrics for each **5-minute music section** based on your protocol.
    This mode automatically extracts sections using measurement events and validates data quality.
    """)

    # Protocol Settings
    with st.expander("Protocol Settings", expanded=False):
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

        if st.button("Save Protocol Settings", key="save_protocol_btn"):
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
    if st.button("Analyze Music Sections", key="analyze_music_btn", type="primary"):
        with st.status("Extracting music sections...", expanded=True) as status:
            try:
                st.write("Loading recording data...")

                # Get participant's recording data
                summary = get_summary_dict().get(selected_participant)
                if not summary:
                    st.error(f"No data found for participant {selected_participant}")
                    return

                source_app = getattr(summary, 'source_app', 'HRV Logger')
                is_vns = (source_app == "VNS Analyse")

                # Load recording
                if is_vns:
                    vns_paths = getattr(summary, 'vns_paths', None)
                    if vns_paths:
                        recording_data = cached_load_vns_recording(
                            tuple(str(p) for p in vns_paths),
                            selected_participant,
                            use_corrected=st.session_state.get("vns_use_corrected", False),
                        )
                    elif getattr(summary, 'vns_path', None):
                        # Fallback: single path (old cached summary)
                        recording_data = cached_load_vns_recording(
                            (str(summary.vns_path),),
                            selected_participant,
                            use_corrected=st.session_state.get("vns_use_corrected", False),
                        )
                    else:
                        # Re-discover VNS recordings
                        from rrational.io.vns_analyse import discover_vns_recordings
                        from pathlib import Path
                        vns_bundles = discover_vns_recordings(
                            Path(st.session_state.data_dir),
                            pattern=st.session_state.id_pattern
                        )
                        vns_bundle = next((b for b in vns_bundles if b.participant_id == selected_participant), None)
                        if not vns_bundle:
                            st.error(f"No VNS recording found for {selected_participant}")
                            return
                        recording_data = cached_load_vns_recording(
                            tuple(str(p) for p in vns_bundle.file_paths),
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
                from rrational.io.hrv_logger import RRInterval
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

                st.write(f"Found {len(rr_intervals)} RR intervals")
                st.write(f"Events: {', '.join(events_dict.keys()) or 'None'}")

                # Extract music sections
                st.write("Extracting music sections...")
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
                        st.warning(f"{warning}")

                st.write(f"Extracted {len(analysis.sections)} sections "
                        f"({analysis.valid_sections} valid, {analysis.incomplete_sections} incomplete)")

                status.update(label="Section extraction complete", state="complete")

                # Display results
                st.markdown("---")
                st.subheader("Music Section Analysis Results")

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
                    status_icon = "[OK]" if section.is_valid else "(!)"
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
                st.dataframe(df_sections, width='stretch', hide_index=True)

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
                            import numpy as np
                            # Convert RR intervals to peak indices for signal_fixpeaks
                            rr_array = np.array(rr_values, dtype=float)
                            peak_indices = np.cumsum(rr_array).astype(int)
                            peak_indices = np.insert(peak_indices, 0, 0)

                            # Call signal_fixpeaks with correct format
                            info, corrected_peaks = nk.signal_fixpeaks(
                                peak_indices,
                                sampling_rate=1000,
                                iterative=True,
                                method="Kubios",
                                show=False,
                            )
                            # Use corrected RR intervals from NeuroKit2
                            rr_values = list(np.diff(corrected_peaks))
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
                    st.dataframe(df_hrv, width='stretch', hide_index=True)

                    # Download button
                    csv_hrv = df_hrv.to_csv(index=False)
                    st.download_button(
                        "Download HRV Results (CSV)",
                        data=csv_hrv,
                        file_name=f"music_sections_hrv_{selected_participant}.csv",
                        mime="text/csv"
                    )

                    # Summary by music type
                    st.markdown("### Summary by Music Type")
                    sections_by_type = get_sections_by_music_type(analysis, valid_only=True)

                    for music_type, sections in sections_by_type.items():
                        with st.expander(f"{music_type} ({len(sections)} sections)", expanded=False):
                            type_results = [r for r in hrv_results if r["Music"] == music_type]
                            if type_results:
                                df_type = pd.DataFrame(type_results)
                                st.dataframe(df_type, width='stretch', hide_index=True)

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

    with st.expander("Help - HRV Analysis & Scientific Best Practices", expanded=False):
        st.markdown(ANALYSIS_HELP)

    if not NEUROKIT_AVAILABLE:
        st.error("NeuroKit2 is not installed. Please install it to use HRV analysis features.")
        st.code("uv add neurokit2")
        return

    if not st.session_state.summaries:
        st.info("Load data from the 'Data & Groups' tab to perform analysis")
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
    from rrational.cleaning.rr import clean_rr_intervals, RRInterval
    from rrational.io.hrv_logger import HRVLoggerRecording, EventMarker

    # Participant selection
    participant_list = get_participant_list()
    selected_participant = st.selectbox(
        "Select Participant",
        options=participant_list,
        key="analysis_participant"
    )

    # Check for .rrational ready files
    ready_files = []
    use_ready_file = False
    selected_ready_file = None
    ready_file_version = "1.0"
    selected_v2_sections = []  # For v2.0 section selection

    if selected_participant:
        data_dir = st.session_state.get("data_dir")
        ready_files = find_rrational_files(selected_participant, data_dir)

    if ready_files:
        with st.expander(f"Ready Files ({len(ready_files)} found)", expanded=True):  # Expanded by default
            st.info(
                "Ready files contain pre-inspected data with artifact detection "
                "and corrected NN intervals. Using a ready file provides the highest "
                "data quality for analysis."
            )

            data_source = st.radio(
                "Data source",
                options=["ready", "raw"],  # Ready file is default when available
                format_func=lambda x: "Use ready file (.rrational)" if x == "ready" else "Use raw data (extract from recording)",
                key="analysis_data_source",
                horizontal=True,
            )
            use_ready_file = (data_source == "ready")

            if use_ready_file:
                # Format file options for display
                file_options = []
                for f in ready_files:
                    # Extract segment name from filename
                    name = f.stem  # e.g., "0123ABCD_rest_pre" or "VP01"
                    segment = name.replace(f"{selected_participant}_", "") or name
                    # Get version
                    try:
                        version = get_rrational_version(f)
                    except Exception:
                        version = "1.0"
                    file_options.append((f, segment, version, f.stat().st_mtime))

                selected_file_idx = st.selectbox(
                    "Select ready file",
                    options=range(len(file_options)),
                    format_func=lambda i: f"{file_options[i][1]} (v{file_options[i][2]}) - {file_options[i][0].name}",
                    key="analysis_ready_file_select",
                )
                selected_ready_file = file_options[selected_file_idx][0]
                ready_file_version = file_options[selected_file_idx][2]

                # Show file info based on version
                try:
                    if ready_file_version == RRATIONAL_VERSION_V2:
                        # V2.0 file - load and show sections
                        ready_data_v2 = load_rrational_v2(selected_ready_file)
                        st.success(f"**v2.0 Export** - {len(ready_data_v2.sections)} section(s) available")

                        # Show available sections with quality info
                        section_info = []
                        for sec_name, sec_data in ready_data_v2.sections.items():
                            nn_count = len(sec_data.nn_intervals.data)
                            quality = sec_data.quality.grade
                            artifact_rate = sec_data.final_artifacts.rate * 100
                            section_info.append({
                                "Section": sec_name,
                                "NN Intervals": nn_count,
                                "Quality": quality.capitalize(),
                                "Artifact %": f"{artifact_rate:.1f}%",
                            })

                        if section_info:
                            st.dataframe(
                                pd.DataFrame(section_info),
                                use_container_width=True,
                                hide_index=True,
                            )

                        # Let user select which sections to analyze
                        available_sections = list(ready_data_v2.sections.keys())
                        selected_v2_sections = st.multiselect(
                            "Select section(s) to analyze",
                            options=available_sections,
                            default=available_sections,  # Select all sections by default
                            key="analysis_v2_sections",
                        )

                        # Store the loaded data for later use
                        st.session_state._analysis_ready_v2_data = ready_data_v2

                        if ready_data_v2.audit_trail:
                            with st.expander("Audit Trail"):
                                for entry in ready_data_v2.audit_trail[-5:]:  # Last 5 entries
                                    st.write(f"**{entry.action}**: {entry.details}")

                        # Overlapping window options for v2.0 ready files
                        with st.expander("Overlapping Window Analysis", expanded=True):
                            st.markdown("""
                            Split each section into **overlapping windows** for more reliable HRV metrics.
                            Results are averaged across all windows within each section.
                            """)
                            use_overlapping_windows = st.checkbox(
                                "Enable overlapping window analysis",
                                value=True,
                                key="use_overlapping_windows_v2",
                                help="Analyze each section using multiple overlapping windows"
                            )

                            if use_overlapping_windows:
                                window_mode = st.radio(
                                    "Window mode",
                                    options=["beats", "time"],
                                    format_func=lambda x: "Beat-based (number of beats)" if x == "beats" else "Time-based (minutes)",
                                    horizontal=True,
                                    key="overlap_window_mode_v2",
                                )

                                if window_mode == "time":
                                    st.caption("**Recommended:** 5-minute windows with 50% overlap")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        window_duration_min = st.slider(
                                            "Window duration (minutes)", 1, 10, 5, key="overlap_window_duration_v2"
                                        )
                                    with col2:
                                        overlap_percent = st.slider(
                                            "Overlap (%)", 0, 75, 50, step=25, key="overlap_percent_v2"
                                        )
                                    step_size_min = window_duration_min * (1 - overlap_percent / 100)
                                    st.caption(f"Step size: {step_size_min:.1f} minutes")
                                    window_beats = None
                                    step_beats = None
                                else:
                                    st.caption("**Default:** 150 beats with 75% overlap")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        window_beats = st.slider(
                                            "Window size (beats)", 100, 1000, 150, step=50, key="overlap_window_beats_v2"
                                        )
                                    with col2:
                                        overlap_beats_percent = st.slider(
                                            "Overlap (%)", 0, 75, 75, step=25, key="overlap_beats_percent_v2"
                                        )
                                    step_beats = int(window_beats * (1 - overlap_beats_percent / 100))
                                    st.caption(f"Step size: {step_beats} beats")
                                    window_duration_min = None
                                    overlap_percent = None
                            else:
                                window_mode = "beats"
                                window_beats = 150
                                step_beats = 37
                                window_duration_min = None
                                overlap_percent = None
                    else:
                        # V1.0 file - original behavior
                        ready_data = load_rrational(selected_ready_file)
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Beats", ready_data.n_beats)
                        with col2:
                            artifact_rate = ready_data.quality.artifact_rate_final * 100
                            st.metric("Artifact Rate", f"{artifact_rate:.1f}%")
                        with col3:
                            st.metric("Quality", ready_data.quality.quality_grade.capitalize())

                        if ready_data.processing_steps:
                            with st.expander("Audit Trail"):
                                for step in ready_data.processing_steps:
                                    st.write(f"**{step.action}**: {step.details}")

                        # Overlapping window options for v1.0 ready files
                        with st.expander("Overlapping Window Analysis", expanded=True):
                            st.markdown("""
                            Split each section into **overlapping windows** for more reliable HRV metrics.
                            Results are averaged across all windows within each section.
                            """)
                            use_overlapping_windows = st.checkbox(
                                "Enable overlapping window analysis",
                                value=True,
                                key="use_overlapping_windows_v1",
                                help="Analyze each section using multiple overlapping windows"
                            )

                            if use_overlapping_windows:
                                window_mode = st.radio(
                                    "Window mode",
                                    options=["beats", "time"],
                                    format_func=lambda x: "Beat-based (number of beats)" if x == "beats" else "Time-based (minutes)",
                                    horizontal=True,
                                    key="overlap_window_mode_v1",
                                )

                                if window_mode == "time":
                                    st.caption("**Recommended:** 5-minute windows with 50% overlap")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        window_duration_min = st.slider(
                                            "Window duration (minutes)", 1, 10, 5, key="overlap_window_duration_v1"
                                        )
                                    with col2:
                                        overlap_percent = st.slider(
                                            "Overlap (%)", 0, 75, 50, step=25, key="overlap_percent_v1"
                                        )
                                    step_size_min = window_duration_min * (1 - overlap_percent / 100)
                                    st.caption(f"Step size: {step_size_min:.1f} minutes")
                                    window_beats = None
                                    step_beats = None
                                else:
                                    st.caption("**Default:** 150 beats with 75% overlap")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        window_beats = st.slider(
                                            "Window size (beats)", 100, 1000, 150, step=50, key="overlap_window_beats_v1"
                                        )
                                    with col2:
                                        overlap_beats_percent = st.slider(
                                            "Overlap (%)", 0, 75, 75, step=25, key="overlap_beats_percent_v1"
                                        )
                                    step_beats = int(window_beats * (1 - overlap_beats_percent / 100))
                                    st.caption(f"Step size: {step_beats} beats")
                                    window_duration_min = None
                                    overlap_percent = None
                            else:
                                window_mode = "beats"
                                window_beats = 150
                                step_beats = 37
                                window_duration_min = None
                                overlap_percent = None
                except Exception as e:
                    st.error(f"Error loading ready file: {e}")
                    use_ready_file = False

    # Section selection (only when NOT using ready file)
    selected_sections = []
    apply_artifact_correction = False

    if not use_ready_file:
        # Show warning about using raw data
        if ready_files:
            st.warning(
                "You have ready files available but are using raw data. "
                "For best results, use the ready file with pre-inspected artifact correction. "
                "Raw data analysis may include artifacts that affect HRV metrics."
            )
        else:
            st.info(
                "No ready file found. To create one, go to the Participants tab and "
                "use the 'Export for Analysis' button after reviewing artifacts."
            )

        available_sections = list(st.session_state.sections.keys())
        if not available_sections:
            st.warning("No sections defined. Please define sections in the Sections tab first.")
            return

        selected_sections = st.multiselect(
            "Select Sections to Analyze",
            options=available_sections,
            default=[available_sections[0]] if available_sections else [],
            key="analysis_sections_single"
        )

        # Artifact correction options
        with st.expander("Artifact Correction (signal_fixpeaks)", expanded=False):
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

    # Overlapping window analysis options (available for BOTH raw data and ready files)
    with st.expander("Overlapping Window Analysis", expanded=True):
        st.markdown("""
        Split each section into **overlapping windows** for more reliable HRV metrics.
        Results are averaged across all windows within each section.
        """)
        use_overlapping_windows = st.checkbox(
            "Enable overlapping window analysis",
            value=True,
            key="use_overlapping_windows",
            help="Analyze each section using multiple overlapping windows"
        )

        if use_overlapping_windows:
            # Mode selection: time-based or beat-based
            window_mode = st.radio(
                "Window mode",
                options=["beats", "time"],
                format_func=lambda x: "Beat-based (number of beats)" if x == "beats" else "Time-based (minutes)",
                horizontal=True,
                key="overlap_window_mode",
                help="Choose whether to define windows by duration (time) or by number of beats"
            )

            if window_mode == "time":
                st.caption("**Recommended:** 5-minute windows with 50% overlap for frequency-domain metrics")
                col1, col2 = st.columns(2)
                with col1:
                    window_duration_min = st.slider(
                        "Window duration (minutes)",
                        min_value=1,
                        max_value=10,
                        value=5,
                        step=1,
                        key="overlap_window_duration",
                        help="Duration of each analysis window"
                    )
                with col2:
                    overlap_percent = st.slider(
                        "Overlap (%)",
                        min_value=0,
                        max_value=75,
                        value=50,
                        step=25,
                        key="overlap_percent",
                        help="Percentage overlap between consecutive windows"
                    )

                # Show calculated step size
                step_size_min = window_duration_min * (1 - overlap_percent / 100)
                st.caption(f"Step size: {step_size_min:.1f} minutes between window starts")
                # Set beat variables to None for time mode
                window_beats = None
                step_beats = None
            else:
                st.caption("**Default:** 150 beats with 75% overlap (37-beat step)")
                col1, col2 = st.columns(2)
                with col1:
                    window_beats = st.slider(
                        "Window size (beats)",
                        min_value=100,
                        max_value=1000,
                        value=150,
                        step=50,
                        key="overlap_window_beats",
                        help="Number of beats in each analysis window"
                    )
                with col2:
                    overlap_beats_percent = st.slider(
                        "Overlap (%)",
                        min_value=0,
                        max_value=75,
                        value=75,
                        step=25,
                        key="overlap_beats_percent",
                        help="Percentage overlap between consecutive windows"
                    )

                # Calculate step size in beats
                step_beats = int(window_beats * (1 - overlap_beats_percent / 100))
                st.caption(f"Step size: {step_beats} beats between window starts")
                # Set time variables to None for beat mode
                window_duration_min = None
                overlap_percent = None
        else:
            window_mode = "beats"
            window_beats = 150
            step_beats = 37
            window_duration_min = None
            overlap_percent = None

    if st.button("Analyze HRV", key="analyze_single_btn", type="primary"):
        # Validate inputs
        if use_ready_file:
            if not selected_ready_file:
                st.error("Please select a ready file")
                return
            # For v2.0 files, need section selection
            if ready_file_version == RRATIONAL_VERSION_V2 and not selected_v2_sections:
                st.error("Please select at least one section from the v2.0 file")
                return
        else:
            if not selected_sections:
                st.error("Please select at least one section")
                return

        # ===== READY FILE ANALYSIS PATH =====
        if use_ready_file and selected_ready_file:
            # Check file version and use appropriate loading path
            if ready_file_version == RRATIONAL_VERSION_V2:
                # ===== V2.0 ANALYSIS PATH =====
                status_msg = "Analyzing HRV from v2.0 export..."
                with st.status(status_msg, expanded=True) as status:
                    try:
                        st.write(f"Loading v2.0 export: {selected_ready_file.name}")
                        progress = st.progress(0)

                        # Get cached v2.0 data or reload
                        ready_data_v2 = st.session_state.get("_analysis_ready_v2_data")
                        if ready_data_v2 is None:
                            ready_data_v2 = load_rrational_v2(selected_ready_file)
                        progress.progress(10)

                        section_results = {}
                        total_sections = len(selected_v2_sections)
                        nk = get_neurokit()

                        for idx, sec_name in enumerate(selected_v2_sections):
                            st.write(f"Analyzing section: {sec_name}")
                            sec_data = ready_data_v2.sections[sec_name]

                            # Extract NN intervals from v2.0 format
                            # Data format: [[timestamp_ms, nn_ms, was_corrected], ...]
                            nn_intervals_ms = [item[1] for item in sec_data.nn_intervals.data]

                            if not nn_intervals_ms:
                                st.warning(f"Section {sec_name} has no NN intervals, skipping")
                                continue

                            # Quality check
                            quality_grade = sec_data.quality.grade
                            meets_time = sec_data.quality.meets_time_domain_min
                            meets_freq = sec_data.quality.meets_freq_domain_min

                            if quality_grade == "poor":
                                st.warning(f"Section {sec_name} has poor quality - results may be unreliable")

                            # Calculate HRV metrics - with optional overlapping windows
                            st.write(f"  Computing HRV for {len(nn_intervals_ms)} NN intervals...")

                            if use_overlapping_windows:
                                # Generate overlapping windows based on mode
                                if window_mode == "beats":
                                    windows = generate_overlapping_windows_beats(
                                        nn_intervals_ms, window_beats, step_beats
                                    )
                                    window_info_str = f"{window_beats} beats, {step_beats}-beat step"
                                else:
                                    window_duration_ms = window_duration_min * 60 * 1000
                                    step_size_ms = window_duration_ms * (1 - overlap_percent / 100)
                                    windows = generate_overlapping_windows_time(
                                        nn_intervals_ms, window_duration_ms, step_size_ms
                                    )
                                    window_info_str = f"{window_duration_min}min, {overlap_percent}% overlap"

                                if len(windows) >= 1:
                                    st.write(f"    Analyzing {len(windows)} overlapping windows ({window_info_str})...")

                                    window_hrv_results = []
                                    window_details = []

                                    for win_idx, win_start, win_rr in windows:
                                        if len(win_rr) < 30:
                                            continue

                                        try:
                                            win_peaks = nk.intervals_to_peaks(win_rr, sampling_rate=1000)
                                            win_hrv_time = nk.hrv_time(win_peaks, sampling_rate=1000, show=False)
                                            if meets_freq:
                                                win_hrv_freq = nk.hrv_frequency(win_peaks, sampling_rate=1000, show=False)
                                                win_hrv = pd.concat([win_hrv_time, win_hrv_freq], axis=1)
                                            else:
                                                win_hrv = win_hrv_time

                                            window_hrv_results.append(win_hrv)
                                            detail = {
                                                "window_idx": win_idx,
                                                "n_beats": len(win_rr),
                                                "duration_s": sum(win_rr) / 1000,
                                                "hrv_results": win_hrv,
                                            }
                                            if window_mode == "beats":
                                                detail["start_beat"] = win_start
                                            else:
                                                detail["start_ms"] = win_start
                                            window_details.append(detail)
                                        except Exception as e:
                                            st.write(f"      Window {win_idx + 1} failed: {e}")
                                            continue

                                    if window_hrv_results:
                                        hrv_results, hrv_std = aggregate_hrv_results(window_hrv_results)
                                        st.write(f"    ✓ Aggregated results from {len(window_hrv_results)} valid windows")

                                        section_results[sec_name] = {
                                            "hrv_results": hrv_results,
                                            "hrv_std": hrv_std,
                                            "rr_intervals": nn_intervals_ms,
                                            "n_beats": len(nn_intervals_ms),
                                            "label": sec_data.definition.label or sec_name,
                                            "artifact_info": {
                                                "total_artifacts": sec_data.final_artifacts.count,
                                                "artifact_rate": sec_data.final_artifacts.rate,
                                                "method": sec_data.artifact_detection.method if sec_data.artifact_detection else "manual",
                                            },
                                            "ready_file": str(selected_ready_file),
                                            "quality_grade": quality_grade,
                                            "quality": {
                                                "meets_time_domain": meets_time,
                                                "meets_freq_domain": meets_freq,
                                                "usable_beats": sec_data.quality.usable_beats,
                                                "usable_duration_s": sec_data.quality.usable_duration_s,
                                            },
                                            "overlapping_analysis": True,
                                            "n_windows": len(window_hrv_results),
                                            "window_mode": window_mode,
                                            "window_duration_min": window_duration_min,
                                            "overlap_percent": overlap_percent,
                                            "window_beats": window_beats,
                                            "step_beats": step_beats,
                                            "window_details": window_details,
                                            "version": "2.0",
                                        }
                                        # Update progress and continue to next section
                                        progress.progress(10 + int(80 * (idx + 1) / total_sections))
                                        continue
                                    else:
                                        st.warning(f"    No valid windows for section '{sec_name}', falling back to single analysis")
                                else:
                                    st.warning("    Section too short for overlapping windows, using single analysis")

                            # Standard single analysis (fallback or when overlapping disabled)
                            peaks = nk.intervals_to_peaks(nn_intervals_ms, sampling_rate=1000)
                            hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)

                            # Only compute frequency metrics if enough data
                            if meets_freq:
                                hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)
                                hrv_results = pd.concat([hrv_time, hrv_freq], axis=1)
                            else:
                                st.info(f"  Section {sec_name}: frequency domain skipped (insufficient data)")
                                hrv_results = hrv_time

                            # Store results
                            section_results[sec_name] = {
                                "hrv_results": hrv_results,
                                "rr_intervals": nn_intervals_ms,
                                "n_beats": len(nn_intervals_ms),
                                "label": sec_data.definition.label or sec_name,
                                "artifact_info": {
                                    "total_artifacts": sec_data.final_artifacts.count,
                                    "artifact_rate": sec_data.final_artifacts.rate,
                                    "method": sec_data.artifact_detection.method if sec_data.artifact_detection else "manual",
                                },
                                "ready_file": str(selected_ready_file),
                                "quality_grade": quality_grade,
                                "quality": {
                                    "meets_time_domain": meets_time,
                                    "meets_freq_domain": meets_freq,
                                    "usable_beats": sec_data.quality.usable_beats,
                                    "usable_duration_s": sec_data.quality.usable_duration_s,
                                },
                                "analysis_segments": [
                                    {
                                        "id": seg.segment_id,
                                        "type": seg.type,
                                        "nn_count": seg.nn_count,
                                        "duration_s": seg.duration_s,
                                    }
                                    for seg in sec_data.analysis_segments
                                ],
                                "version": "2.0",
                            }

                            # Update progress
                            progress.progress(10 + int(80 * (idx + 1) / total_sections))

                        if not section_results:
                            st.error("No sections could be analyzed")
                            status.update(label="Analysis failed", state="error")
                            return

                        progress.progress(100)
                        st.session_state.analysis_results[selected_participant] = section_results
                        status.update(label=f"Analysis complete! ({len(section_results)} sections)", state="complete")
                        show_toast(f"v2.0 analysis complete: {len(section_results)} section(s)", icon="success")

                    except Exception as e:
                        status.update(label="Error during v2.0 analysis", state="error")
                        st.error(f"Error analyzing v2.0 file: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            else:
                # ===== V1.0 ANALYSIS PATH (original behavior) =====
                status_msg = "Analyzing HRV from ready file..."
                with st.status(status_msg, expanded=True) as status:
                    try:
                        st.write(f"Loading ready file: {selected_ready_file.name}")
                        progress = st.progress(0)

                        # Load ready file
                        ready_data = load_rrational(selected_ready_file)
                        progress.progress(20)

                        # Get clean RR intervals (exclude artifact indices)
                        artifact_indices = set(ready_data.final_artifact_indices)
                        clean_rr_ms = []
                        for i, rr in enumerate(ready_data.rr_intervals):
                            if i not in artifact_indices:
                                clean_rr_ms.append(rr.rr_ms)

                        if not clean_rr_ms:
                            st.error("No clean RR intervals after removing artifacts")
                            status.update(label="Analysis failed - no clean data", state="error")
                            return

                        st.write(f"Using {len(clean_rr_ms)} clean beats ({len(artifact_indices)} artifacts removed)")
                        progress.progress(40)

                        # Get segment name
                        segment_name = "ready_file"
                        if ready_data.segment:
                            if ready_data.segment.section_name:
                                segment_name = ready_data.segment.section_name
                            elif ready_data.segment.time_range:
                                segment_name = ready_data.segment.time_range.get("label", "custom_range")

                        # Calculate HRV metrics - with optional overlapping windows
                        st.write("Computing HRV metrics...")
                        nk = get_neurokit()

                        if use_overlapping_windows:
                            # Generate overlapping windows based on mode
                            if window_mode == "beats":
                                windows = generate_overlapping_windows_beats(
                                    clean_rr_ms, window_beats, step_beats
                                )
                                window_info_str = f"{window_beats} beats, {step_beats}-beat step"
                            else:
                                window_duration_ms = window_duration_min * 60 * 1000
                                step_size_ms = window_duration_ms * (1 - overlap_percent / 100)
                                windows = generate_overlapping_windows_time(
                                    clean_rr_ms, window_duration_ms, step_size_ms
                                )
                                window_info_str = f"{window_duration_min}min, {overlap_percent}% overlap"

                            if len(windows) >= 1:
                                st.write(f"  Analyzing {len(windows)} overlapping windows ({window_info_str})...")

                                window_hrv_results = []
                                window_details = []

                                for win_idx, win_start, win_rr in windows:
                                    if len(win_rr) < 30:
                                        continue

                                    try:
                                        win_peaks = nk.intervals_to_peaks(win_rr, sampling_rate=1000)
                                        win_hrv_time = nk.hrv_time(win_peaks, sampling_rate=1000, show=False)
                                        win_hrv_freq = nk.hrv_frequency(win_peaks, sampling_rate=1000, show=False)
                                        win_hrv = pd.concat([win_hrv_time, win_hrv_freq], axis=1)

                                        window_hrv_results.append(win_hrv)
                                        detail = {
                                            "window_idx": win_idx,
                                            "n_beats": len(win_rr),
                                            "duration_s": sum(win_rr) / 1000,
                                            "hrv_results": win_hrv,
                                        }
                                        if window_mode == "beats":
                                            detail["start_beat"] = win_start
                                        else:
                                            detail["start_ms"] = win_start
                                        window_details.append(detail)
                                    except Exception as e:
                                        st.write(f"    Window {win_idx + 1} failed: {e}")
                                        continue

                                if window_hrv_results:
                                    hrv_results, hrv_std = aggregate_hrv_results(window_hrv_results)
                                    st.write(f"  ✓ Aggregated results from {len(window_hrv_results)} valid windows")
                                    progress.progress(80)

                                    section_results = {
                                        segment_name: {
                                            "hrv_results": hrv_results,
                                            "hrv_std": hrv_std,
                                            "rr_intervals": clean_rr_ms,
                                            "n_beats": len(clean_rr_ms),
                                            "label": segment_name,
                                            "artifact_info": {
                                                "total_artifacts": len(artifact_indices),
                                                "artifact_rate": ready_data.quality.artifact_rate_final,
                                                "method": ready_data.artifact_detection.method if ready_data.artifact_detection else "manual",
                                            },
                                            "ready_file": str(selected_ready_file),
                                            "quality_grade": ready_data.quality.quality_grade,
                                            "audit_trail": ready_data.processing_steps,
                                            "overlapping_analysis": True,
                                            "n_windows": len(window_hrv_results),
                                            "window_mode": window_mode,
                                            "window_duration_min": window_duration_min,
                                            "overlap_percent": overlap_percent,
                                            "window_beats": window_beats,
                                            "step_beats": step_beats,
                                            "window_details": window_details,
                                            "version": "1.0",
                                        }
                                    }

                                    progress.progress(100)
                                    st.session_state.analysis_results[selected_participant] = section_results
                                    status.update(label="Analysis complete from ready file!", state="complete")
                                    show_toast("Ready file analysis complete (overlapping windows)", icon="success")
                                    return  # Exit early, skip standard analysis
                                else:
                                    st.warning("  No valid windows, falling back to single analysis")
                            else:
                                st.warning("  Data too short for overlapping windows, using single analysis")

                        # Standard single analysis (fallback or when overlapping disabled)
                        peaks = nk.intervals_to_peaks(clean_rr_ms, sampling_rate=1000)
                        hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
                        hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)
                        hrv_results = pd.concat([hrv_time, hrv_freq], axis=1)
                        progress.progress(80)

                        # Store results
                        section_results = {
                            segment_name: {
                                "hrv_results": hrv_results,
                                "rr_intervals": clean_rr_ms,
                                "n_beats": len(clean_rr_ms),
                                "label": segment_name,
                                "artifact_info": {
                                    "total_artifacts": len(artifact_indices),
                                    "artifact_rate": ready_data.quality.artifact_rate_final,
                                    "method": ready_data.artifact_detection.method if ready_data.artifact_detection else "manual",
                                },
                                "ready_file": str(selected_ready_file),
                                "quality_grade": ready_data.quality.quality_grade,
                                "audit_trail": ready_data.processing_steps,
                                "version": "1.0",
                            }
                        }

                        progress.progress(100)
                        st.session_state.analysis_results[selected_participant] = section_results
                        status.update(label="Analysis complete from ready file!", state="complete")
                        show_toast("Ready file analysis complete", icon="success")

                    except Exception as e:
                        status.update(label="Error during analysis", state="error")
                        st.error(f"Error analyzing ready file: {e}")
                        import traceback
                        st.code(traceback.format_exc())

        # ===== SECTION-BASED ANALYSIS PATH =====
        else:
            # Use status context for multi-step analysis
            with st.status("Analyzing HRV for selected sections...", expanded=True) as status:
                try:
                    st.write("Loading recording data...")
                    progress = st.progress(0)

                    # Check source type from summary
                    summary = get_summary_dict().get(selected_participant)
                    source_app = getattr(summary, 'source_app', 'HRV Logger') if summary else 'HRV Logger'
                    is_vns = (source_app == "VNS Analyse")

                    if is_vns:
                        vns_paths = getattr(summary, 'vns_paths', None)
                        if vns_paths:
                            recording_data = cached_load_vns_recording(
                                tuple(str(p) for p in vns_paths),
                                selected_participant,
                                use_corrected=st.session_state.get("vns_use_corrected", False),
                            )
                        elif getattr(summary, 'vns_path', None):
                            # Fallback: single path (old cached summary)
                            recording_data = cached_load_vns_recording(
                                (str(summary.vns_path),),
                                selected_participant,
                                use_corrected=st.session_state.get("vns_use_corrected", False),
                            )
                        else:
                            # Re-discover VNS recordings
                            from rrational.io.vns_analyse import discover_vns_recordings
                            from pathlib import Path
                            vns_bundles = discover_vns_recordings(
                                Path(st.session_state.data_dir),
                                pattern=st.session_state.id_pattern
                            )
                            vns_bundle = next((b for b in vns_bundles if b.participant_id == selected_participant), None)
                            if not vns_bundle:
                                st.error(f"No VNS recording found for {selected_participant}")
                                return
                            recording_data = cached_load_vns_recording(
                                tuple(str(p) for p in vns_bundle.file_paths),
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

                    # Load stored/saved events from YAML - REQUIRED for analysis
                    # User must review and save events in Participants tab first
                    if selected_participant not in st.session_state.participant_events:
                        from rrational.gui.persistence import load_participant_events
                        from rrational.prep.summaries import EventStatus
                        from datetime import datetime as dt

                        saved = load_participant_events(selected_participant, st.session_state.data_dir)
                        if saved:
                            # Convert dicts to EventStatus objects (same as app.py)
                            def dict_to_event(d):
                                ts = d.get("first_timestamp")
                                if ts and isinstance(ts, str):
                                    ts = dt.fromisoformat(ts)
                                last_ts = d.get("last_timestamp")
                                if last_ts and isinstance(last_ts, str):
                                    last_ts = dt.fromisoformat(last_ts)
                                return EventStatus(
                                    raw_label=d.get("raw_label", ""),
                                    canonical=d.get("canonical"),
                                    first_timestamp=ts,
                                    last_timestamp=last_ts,
                                )

                            st.session_state.participant_events[selected_participant] = {
                                'events': [dict_to_event(e) for e in saved.get('events', [])],
                                'manual': [dict_to_event(e) for e in saved.get('manual', [])],
                                'music_events': [dict_to_event(e) for e in saved.get('music_events', [])],
                                'exclusion_zones': saved.get('exclusion_zones', []),
                            }

                    stored_events = st.session_state.participant_events.get(selected_participant, {})
                    all_stored = stored_events.get('events', []) + stored_events.get('manual', [])

                    if not all_stored:
                        # No saved events - STOP and warn user
                        st.error(f"No saved events found for {selected_participant}!")
                        st.warning(
                            "**Please review and save the participant's data first:**\n"
                            "1. Go to the **Participants** tab\n"
                            "2. Select this participant\n"
                            "3. Review and edit events as needed\n"
                            "4. Click **Save Events** to save your changes\n\n"
                            "Analysis requires processed/saved events to ensure data quality."
                        )
                        status.update(label="Analysis stopped - no saved events", state="error")
                        return

                    # Use saved/processed events (with canonical labels)
                    st.write(f"Using {len(all_stored)} saved events for {selected_participant}")
                    events = []
                    for evt in all_stored:
                        # Handle both dict (from YAML) and object formats
                        if isinstance(evt, dict):
                            ts = evt.get('first_timestamp')
                            label = evt.get('canonical') or evt.get('raw_label', 'unknown')
                        else:
                            ts = getattr(evt, 'first_timestamp', None)
                            label = getattr(evt, 'canonical', None) or getattr(evt, 'raw_label', 'unknown')

                        if ts:
                            # Convert string timestamps from YAML to datetime
                            if isinstance(ts, str):
                                from datetime import datetime
                                ts = datetime.fromisoformat(ts)
                            events.append(EventMarker(label=label, timestamp=ts, offset_s=None))

                    # Debug: show event labels
                    with st.expander("Debug: Event labels", expanded=False):
                        for evt in events:
                            st.write(f"  - '{evt.label}' at {evt.timestamp}")

                    recording = HRVLoggerRecording(
                        participant_id=selected_participant,
                        rr_intervals=rr_intervals,
                        events=events
                    )
                    progress.progress(20)

                    # Store results for each section
                    section_results = {}
                    combined_rr = []

                    st.write(f"Analyzing {len(selected_sections)} section(s)...")

                    # Analyze each section individually
                    for idx, section_name in enumerate(selected_sections):
                        progress.progress(20 + int((idx / len(selected_sections)) * 60))
                        st.write(f"  • Processing section: {section_name}")

                        section_def = st.session_state.sections[section_name]
                        start_evt = section_def.get("start_event")
                        end_evts = section_def.get("end_events", []) or [section_def.get("end_event")]
                        st.write(f"    Looking for: start='{start_evt}', end={end_evts}")

                        section_rr = extract_section_rr_intervals(
                            recording, section_def, st.session_state.normalizer,
                            saved_events=all_stored  # Use saved/edited events, not raw file events
                        )

                        if section_rr:
                            # Apply exclusion zone filtering
                            exclusion_zones = _get_exclusion_zones(selected_participant)
                            if exclusion_zones:
                                section_rr, excl_stats = filter_exclusion_zones(section_rr, exclusion_zones)
                                if excl_stats["n_excluded"] > 0:
                                    st.write(f"    Excluded {excl_stats['n_excluded']} intervals ({excl_stats['excluded_duration_ms']/1000:.1f}s) from {excl_stats['zones_applied']} zone(s)")

                            # Clean RR intervals for this section
                            cleaned_section_rr, stats = clean_rr_intervals(
                                section_rr, st.session_state.cleaning_config
                            )

                            if cleaned_section_rr:
                                rr_ms = [rr.rr_ms for rr in cleaned_section_rr]

                                # Apply artifact correction if enabled
                                artifact_info = None
                                if apply_artifact_correction:
                                    st.write("    Applying artifact correction...")
                                    artifact_result = detect_artifacts_fixpeaks(rr_ms)
                                    if artifact_result["correction_applied"]:
                                        rr_ms = artifact_result["corrected_rr"]
                                        artifact_info = artifact_result
                                        st.write(f"    * Corrected {artifact_result['total_artifacts']} artifacts")

                                combined_rr.extend(rr_ms)

                                # Calculate HRV metrics
                                nk = get_neurokit()

                                # Check if overlapping window analysis is enabled
                                if use_overlapping_windows:
                                    # Generate overlapping windows based on mode
                                    if window_mode == "beats":
                                        windows = generate_overlapping_windows_beats(
                                            rr_ms, window_beats, step_beats
                                        )
                                        window_info_str = f"{window_beats} beats, {step_beats}-beat step"
                                    else:
                                        window_duration_ms = window_duration_min * 60 * 1000
                                        step_size_ms = window_duration_ms * (1 - overlap_percent / 100)
                                        windows = generate_overlapping_windows_time(
                                            rr_ms, window_duration_ms, step_size_ms
                                        )
                                        window_info_str = f"{window_duration_min}min, {overlap_percent}% overlap"

                                    if len(windows) >= 1:
                                        st.write(f"    Analyzing {len(windows)} overlapping windows ({window_info_str})...")

                                        window_hrv_results = []
                                        window_details = []

                                        for win_idx, win_start, win_rr in windows:
                                            if len(win_rr) < 30:  # Skip windows with too few beats
                                                continue

                                            try:
                                                win_peaks = nk.intervals_to_peaks(win_rr, sampling_rate=1000)
                                                win_hrv_time = nk.hrv_time(win_peaks, sampling_rate=1000, show=False)
                                                win_hrv_freq = nk.hrv_frequency(win_peaks, sampling_rate=1000, show=False)
                                                win_hrv = pd.concat([win_hrv_time, win_hrv_freq], axis=1)

                                                window_hrv_results.append(win_hrv)
                                                detail = {
                                                    "window_idx": win_idx,
                                                    "n_beats": len(win_rr),
                                                    "duration_s": sum(win_rr) / 1000,
                                                    "hrv_results": win_hrv,
                                                }
                                                if window_mode == "beats":
                                                    detail["start_beat"] = win_start
                                                else:
                                                    detail["start_ms"] = win_start
                                                window_details.append(detail)
                                            except Exception as e:
                                                st.write(f"      Window {win_idx + 1} failed: {e}")
                                                continue

                                        if window_hrv_results:
                                            # Aggregate results across windows
                                            hrv_results, hrv_std = aggregate_hrv_results(window_hrv_results)
                                            st.write(f"    ✓ Aggregated results from {len(window_hrv_results)} valid windows")

                                            section_results[section_name] = {
                                                "hrv_results": hrv_results,
                                                "hrv_std": hrv_std,
                                                "rr_intervals": rr_ms,
                                                "n_beats": len(rr_ms),
                                                "label": section_def.get("label", section_name),
                                                "artifact_info": artifact_info,
                                                "overlapping_analysis": True,
                                                "n_windows": len(window_hrv_results),
                                                "window_mode": window_mode,
                                                "window_duration_min": window_duration_min,
                                                "overlap_percent": overlap_percent,
                                                "window_beats": window_beats,
                                                "step_beats": step_beats,
                                                "window_details": window_details,
                                            }
                                        else:
                                            st.warning(f"    No valid windows for section '{section_name}'")
                                    else:
                                        st.warning("    Section too short for overlapping windows, using single analysis")
                                        # Fall back to single analysis
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
                                    # Standard single-window analysis
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
                            st.write(f"  Could not find events for section '{section_name}'")

                    # Analyze combined sections if multiple selected
                    if len(selected_sections) > 1 and combined_rr:
                        progress.progress(80)
                        st.write("Computing combined analysis...")
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

                    status.update(label=f"Analysis complete for {len(section_results)} section(s)!", state="complete")
                    show_toast(f"Analysis complete for {len(section_results)} section(s)", icon="success")

                except Exception as e:
                    status.update(label="Error during analysis", state="error")
                    st.error(f"Error during analysis: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # Display results if available
    if selected_participant in st.session_state.analysis_results:
        _display_single_participant_results(selected_participant)


def _display_stats_row(stats: dict, key_prefix: str = ""):
    """Display statistics as a row of metrics below a plot.

    Stats can be either:
    - Simple string values: {"Label": "123 ms"}
    - Tuple with delta: {"Label": ("123 ms", "45%")} for value with delta
    """
    if not stats:
        return
    n_cols = min(len(stats), 5)  # Max 5 columns
    cols = st.columns(n_cols)
    for i, (label, value) in enumerate(stats.items()):
        with cols[i % n_cols]:
            if isinstance(value, tuple) and len(value) == 2:
                # Value with delta (e.g., ("651 ms²", "38%"))
                st.metric(label, value[0], delta=value[1], delta_color="off")
            else:
                st.metric(label, value)


def _display_single_participant_results(selected_participant: str):
    """Display HRV analysis results for a single participant with professional visualizations."""
    st.markdown("---")
    st.subheader(f"Results for {selected_participant}")

    section_results = st.session_state.analysis_results[selected_participant]

    # Create documentation object if we have results
    if section_results:
        doc = AnalysisDocumentation(selected_participant)

        # Try to get data source info
        summary = get_summary_dict().get(selected_participant)
        source_app = getattr(summary, 'source_app', 'HRV Logger') if summary else 'HRV Logger'
        total_raw_beats = sum(r.get("n_beats", 0) for r in section_results.values())
        total_duration = sum(sum(r.get("rr_intervals", [])) / 1000.0 for r in section_results.values())

        doc.set_data_source(source_app, total_raw_beats, total_duration)
        doc.set_cleaning_config(st.session_state.get("cleaning_config", {}))

        # Check if artifact correction was applied
        artifact_correction_applied = any(
            r.get("artifact_info") is not None for r in section_results.values()
        )
        if artifact_correction_applied:
            first_artifact = next(
                (r.get("artifact_info") for r in section_results.values() if r.get("artifact_info")),
                None
            )
            doc.set_artifact_correction(True, first_artifact)
        else:
            doc.set_artifact_correction(False)

        # Add exclusion zones
        exclusion_zones = _get_exclusion_zones(selected_participant)
        doc.add_exclusion_zones(exclusion_zones)

    for section_name, result_data in section_results.items():
        section_label = result_data["label"]
        hrv_results = result_data["hrv_results"]
        rr_intervals = result_data["rr_intervals"]
        n_beats = result_data["n_beats"]
        artifact_info = result_data.get("artifact_info")

        # Calculate recording duration from RR intervals (sum of intervals)
        recording_duration_sec = sum(rr_intervals) / 1000.0 if rr_intervals else 0

        # Add to documentation
        if section_results:
            section_def = st.session_state.sections.get(section_name, {})
            doc.add_section(
                name=section_name,
                label=section_label,
                start_event=section_def.get("start_event", "N/A"),
                end_events=section_def.get("end_events", []) or [section_def.get("end_event", "N/A")],
                beats_extracted=n_beats,
                beats_after_cleaning=n_beats,
            )
            doc.add_hrv_results(section_name, hrv_results)

        with st.expander(f"{section_label} ({n_beats} beats, {recording_duration_sec/60:.1f} min)", expanded=True):
            # Show overlapping window analysis info if used
            if result_data.get("overlapping_analysis"):
                n_windows = result_data.get("n_windows", 0)
                window_mode = result_data.get("window_mode", "time")
                hrv_std = result_data.get("hrv_std")

                if window_mode == "beats":
                    window_beats = result_data.get("window_beats", 300)
                    step_beats = result_data.get("step_beats", 150)
                    window_info = f"{window_beats} beats, {step_beats}-beat step"
                else:
                    window_duration = result_data.get("window_duration_min", 5)
                    overlap_pct = result_data.get("overlap_percent", 50)
                    window_info = f"{window_duration}min, {overlap_pct}% overlap"

                # Get segment info from analysis_segments if available
                analysis_segments = result_data.get("analysis_segments", [])
                gap_segments = len([s for s in analysis_segments if s.get("type") == "usable"])
                exclusion_segments = len([s for s in analysis_segments if s.get("type") == "exclusion"])

                st.info(f"**Overlapping Window Analysis:** {n_windows} windows analyzed ({window_info})")
                if analysis_segments:
                    st.caption(f"Based on {gap_segments} usable segment(s)" +
                              (f", {exclusion_segments} exclusion zone(s)" if exclusion_segments else ""))

                # Show std values if available
                if hrv_std is not None and not hrv_std.empty:
                    with st.expander("Window Variability (Std Dev)", expanded=False):
                        st.caption("Standard deviation across overlapping windows:")
                        # Show key metrics with std
                        key_metrics = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_LF", "HRV_HF"]
                        std_display = {}
                        for m in key_metrics:
                            if m in hrv_std.columns:
                                std_display[m.replace("HRV_", "")] = f"±{hrv_std[m].values[0]:.2f}"
                        if std_display:
                            cols = st.columns(len(std_display))
                            for i, (name, val) in enumerate(std_display.items()):
                                cols[i].metric(name, val)

            # Show ready file info if this came from a .rrational file
            ready_file_path = result_data.get("ready_file")
            quality_grade = result_data.get("quality_grade")
            audit_trail = result_data.get("audit_trail")

            if ready_file_path:
                st.caption(f"Source: {ready_file_path}")
                if quality_grade:
                    grade_colors = {
                        "excellent": "green", "good": "blue",
                        "moderate": "orange", "poor": "red"
                    }
                    st.markdown(f"Quality grade: **:{grade_colors.get(quality_grade, 'gray')}[{quality_grade.upper()}]**")

                if audit_trail:
                    with st.expander("Audit Trail", expanded=False):
                        for step in audit_trail:
                            st.write(f"**{step.action}**: {step.details}")

            # Display HRV metrics using professional layout
            if not hrv_results.empty:
                display_hrv_metrics_professional(
                    hrv_results, n_beats, artifact_info,
                    recording_duration_sec=recording_duration_sec
                )

            # Visualization tabs for professional plots
            if PLOTLY_AVAILABLE and len(rr_intervals) > 10:
                plot_tabs = st.tabs(["Tachogram", "Poincaré", "Frequency", "HR Distribution", "Data"])

                with plot_tabs[0]:
                    # Educational info
                    display_visualization_info("tachogram")
                    # Professional Tachogram
                    artifact_indices = None
                    if artifact_info and 'artifact_indices' in artifact_info:
                        artifact_indices = artifact_info['artifact_indices']
                    fig_tach, tach_stats = create_professional_tachogram(rr_intervals, section_label, artifact_indices)
                    st.plotly_chart(fig_tach, use_container_width=True)
                    _display_stats_row(tach_stats, f"tach_{section_name}")

                with plot_tabs[1]:
                    # Educational info
                    display_visualization_info("poincare")
                    # Poincaré Plot
                    if len(rr_intervals) > 20:
                        fig_poincare, poincare_stats = create_poincare_plot(rr_intervals, section_label)
                        st.plotly_chart(fig_poincare, use_container_width=True)
                        _display_stats_row(poincare_stats, f"poincare_{section_name}")
                    else:
                        st.warning("Not enough data points for Poincaré plot (need >20 beats)")

                with plot_tabs[2]:
                    # Educational info
                    display_visualization_info("frequency")
                    # Frequency Domain Plot
                    if len(rr_intervals) > 100:
                        fig_freq, freq_stats = create_frequency_domain_plot(rr_intervals, section_label)
                        if fig_freq:
                            st.plotly_chart(fig_freq, use_container_width=True)
                            _display_stats_row(freq_stats, f"freq_{section_name}")
                    else:
                        st.warning("Not enough data for reliable frequency analysis (need >100 beats, ideally >300)")

                with plot_tabs[3]:
                    # Educational info
                    display_visualization_info("hr_distribution")
                    # Heart Rate Distribution
                    fig_hr, hr_stats = create_hr_distribution_plot(rr_intervals, section_label)
                    st.plotly_chart(fig_hr, use_container_width=True)
                    _display_stats_row(hr_stats, f"hr_{section_name}")

                with plot_tabs[4]:
                    # Full results table and download
                    if not hrv_results.empty:
                        st.markdown("**Complete HRV Metrics:**")
                        st.dataframe(hrv_results.T, use_container_width=True)

                        # Download buttons
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            csv_hrv = hrv_results.to_csv(index=True)
                            st.download_button(
                                label="Download HRV Results (CSV)",
                                data=csv_hrv,
                                file_name=f"hrv_{selected_participant}_{section_name}.csv",
                                mime="text/csv",
                                key=f"download_hrv_{selected_participant}_{section_name}",
                            )
                        with col_dl2:
                            # Download RR intervals
                            rr_df = pd.DataFrame({"beat_index": range(len(rr_intervals)), "rr_ms": rr_intervals})
                            csv_rr = rr_df.to_csv(index=False)
                            st.download_button(
                                label="Download RR Intervals (CSV)",
                                data=csv_rr,
                                file_name=f"rr_{selected_participant}_{section_name}.csv",
                                mime="text/csv",
                                key=f"download_rr_{selected_participant}_{section_name}",
                            )

            else:
                # Fallback to matplotlib if Plotly not available
                if not hrv_results.empty:
                    st.markdown("**Key Metrics:**")
                    cols = st.columns(3)
                    metrics = [("HRV_RMSSD", "RMSSD"), ("HRV_SDNN", "SDNN"), ("HRV_pNN50", "pNN50")]
                    for i, (col_name, label) in enumerate(metrics):
                        if col_name in hrv_results.columns:
                            with cols[i]:
                                st.metric(label, f"{hrv_results[col_name].iloc[0]:.2f}")

                    st.dataframe(hrv_results.T, use_container_width=True)

                # Simple matplotlib plot
                st.markdown("**Tachogram:**")
                plt = get_matplotlib()
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(rr_intervals, marker='o', markersize=2, linestyle='-', linewidth=0.5)
                ax.set_xlabel("Beat Index")
                ax.set_ylabel("RR Interval (ms)")
                ax.set_title(f"Tachogram - {section_label}")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

    # Display documentation panel at the end
    if section_results:
        display_documentation_panel(doc)


# =============================================================================
# GROUP ANALYSIS HELPER FUNCTIONS
# =============================================================================


def _collect_group_participants(selected_groups: list[str]) -> dict[str, list[str]]:
    """Collect participants for each selected group.

    Args:
        selected_groups: List of group names to collect

    Returns:
        Dict mapping group name to list of participant IDs
    """
    result = {}
    for group in selected_groups:
        participants = [
            pid for pid, gname in st.session_state.participant_groups.items()
            if gname == group
        ]
        result[group] = participants
    return result


def _find_rrational_v2_file(
    participant_id: str,
    project_path: str | None = None,
    data_dir: str | None = None,
) -> str | None:
    """Find the .rrational v2 file for a participant.

    Args:
        participant_id: The participant ID
        project_path: Path to the project folder
        data_dir: Alternative data directory

    Returns:
        Path to the .rrational v2 file, or None if not found
    """
    from pathlib import Path

    files = find_rrational_files(
        participant_id,
        data_dir=data_dir,
        project_path=Path(project_path) if project_path else None,
    )

    # Find v2 file
    for f in files:
        try:
            version = get_rrational_version(f)
            if version == RRATIONAL_VERSION_V2:
                return str(f)
        except Exception:
            continue

    return None


def _load_nn_from_rrational_v2(
    file_path: str,
    section_name: str,
) -> tuple[list[float] | None, dict]:
    """Load NN intervals from a v2 .rrational file for a specific section.

    Args:
        file_path: Path to the .rrational v2 file
        section_name: Name of the section to load

    Returns:
        Tuple of (nn_ms_list, info_dict) where:
        - nn_ms_list: List of NN interval values in ms, or None if not found
        - info_dict: Contains quality_grade, artifact_rate, n_beats, duration_s
    """
    try:
        export_data = load_rrational_v2(file_path)

        if section_name not in export_data.sections:
            return None, {"error": f"Section '{section_name}' not found in file"}

        section = export_data.sections[section_name]
        nn_data = section.nn_intervals.data

        if not nn_data:
            return None, {"error": "No NN intervals in section"}

        # Extract NN values (format: [[timestamp_ms, nn_ms, was_corrected], ...])
        nn_ms_list = [entry[1] for entry in nn_data]

        # Get quality info
        quality = section.quality
        info = {
            "quality_grade": quality.grade,
            "artifact_rate": (
                section.artifact_detection.artifact_rate
                if section.artifact_detection else 0.0
            ),
            "n_beats": len(nn_ms_list),
            "duration_s": quality.usable_duration_s,
            "meets_time_domain": quality.meets_time_domain_min,
            "meets_freq_domain": quality.meets_freq_domain_min,
        }

        return nn_ms_list, info

    except Exception as e:
        return None, {"error": str(e)}


def _calculate_hrv_metrics(
    nn_ms_list: list[float],
    use_windows: bool = True,
    window_beats: int = 150,
    overlap_pct: float = 75.0,
    selected_metrics: list[str] | None = None,
) -> tuple[dict, dict | None, int]:
    """Calculate HRV metrics from NN intervals.

    Args:
        nn_ms_list: List of NN interval values in ms
        use_windows: Whether to use overlapping windows
        window_beats: Number of beats per window
        overlap_pct: Overlap percentage (0-100)
        selected_metrics: List of metric names to calculate (None = all basic metrics)

    Returns:
        Tuple of (metrics_dict, std_dict, n_windows)
        - metrics_dict: Mean HRV metrics
        - std_dict: SD of metrics (None if no windows)
        - n_windows: Number of windows used
    """
    nk = get_neurokit()

    # Default to basic metrics if not specified
    if selected_metrics is None:
        selected_metrics = HRV_METRIC_PRESETS["Basic"]["metrics"]

    # Determine which analysis types we need
    time_basic = set(HRV_METRICS_CATALOG["time_basic"].keys())
    time_extended = set(HRV_METRICS_CATALOG["time_extended"].keys())
    frequency = set(HRV_METRICS_CATALOG["frequency"].keys())
    nonlinear = set(HRV_METRICS_CATALOG["nonlinear"].keys())

    selected_set = set(selected_metrics)
    need_time = bool(selected_set & (time_basic | time_extended))
    need_freq = bool(selected_set & frequency)
    need_nonlinear = bool(selected_set & nonlinear)

    def compute_hrv(rr_list: list[float]) -> dict:
        """Compute HRV for a single window."""
        result = {}
        peaks = nk.intervals_to_peaks(rr_list, sampling_rate=1000)

        # Time domain metrics
        if need_time:
            try:
                hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)

                # Basic time metrics
                if "RMSSD" in selected_set:
                    result["RMSSD"] = hrv_time.get("HRV_RMSSD", [None])[0]
                if "SDNN" in selected_set:
                    result["SDNN"] = hrv_time.get("HRV_SDNN", [None])[0]
                if "pNN50" in selected_set:
                    result["pNN50"] = hrv_time.get("HRV_pNN50", [None])[0]
                if "MeanNN" in selected_set:
                    result["MeanNN"] = hrv_time.get("HRV_MeanNN", [None])[0]
                if "MeanHR" in selected_set:
                    mean_nn = hrv_time.get("HRV_MeanNN", [None])[0]
                    result["MeanHR"] = 60000 / mean_nn if mean_nn and mean_nn > 0 else None

                # Extended time metrics
                if "SDSD" in selected_set:
                    result["SDSD"] = hrv_time.get("HRV_SDSD", [None])[0]
                if "pNN20" in selected_set:
                    result["pNN20"] = hrv_time.get("HRV_pNN20", [None])[0]
                if "MedianNN" in selected_set:
                    result["MedianNN"] = hrv_time.get("HRV_MedianNN", [None])[0]
                if "CVNN" in selected_set:
                    result["CVNN"] = hrv_time.get("HRV_CVNN", [None])[0]
                if "CVSD" in selected_set:
                    result["CVSD"] = hrv_time.get("HRV_CVSD", [None])[0]
                if "MadNN" in selected_set:
                    result["MadNN"] = hrv_time.get("HRV_MadNN", [None])[0]
                if "MCVNN" in selected_set:
                    result["MCVNN"] = hrv_time.get("HRV_MCVNN", [None])[0]
                if "IQRNN" in selected_set:
                    result["IQRNN"] = hrv_time.get("HRV_IQRNN", [None])[0]
                if "HTI" in selected_set:
                    result["HTI"] = hrv_time.get("HRV_HTI", [None])[0]
                if "TINN" in selected_set:
                    result["TINN"] = hrv_time.get("HRV_TINN", [None])[0]
            except Exception:
                # Set all requested time metrics to None on error
                for m in selected_set & (time_basic | time_extended):
                    result[m] = None

        # Frequency domain metrics
        if need_freq and len(rr_list) >= MIN_BEATS_FREQUENCY_DOMAIN:
            try:
                hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False)

                if "VLF" in selected_set:
                    result["VLF"] = hrv_freq.get("HRV_VLF", [None])[0]
                if "LF" in selected_set:
                    result["LF"] = hrv_freq.get("HRV_LF", [None])[0]
                if "HF" in selected_set:
                    result["HF"] = hrv_freq.get("HRV_HF", [None])[0]
                if "LF_HF" in selected_set:
                    result["LF_HF"] = hrv_freq.get("HRV_LFHF", [None])[0]
                if "LFn" in selected_set:
                    result["LFn"] = hrv_freq.get("HRV_LFn", [None])[0]
                if "HFn" in selected_set:
                    result["HFn"] = hrv_freq.get("HRV_HFn", [None])[0]
                if "TP" in selected_set:
                    # Total power = VLF + LF + HF
                    vlf = hrv_freq.get("HRV_VLF", [0])[0] or 0
                    lf = hrv_freq.get("HRV_LF", [0])[0] or 0
                    hf = hrv_freq.get("HRV_HF", [0])[0] or 0
                    result["TP"] = vlf + lf + hf if any([vlf, lf, hf]) else None
            except Exception:
                for m in selected_set & frequency:
                    result[m] = None
        elif need_freq:
            # Not enough beats for frequency analysis
            for m in selected_set & frequency:
                result[m] = None

        # Nonlinear metrics
        if need_nonlinear:
            try:
                # Poincaré metrics (SD1, SD2)
                if selected_set & {"SD1", "SD2", "SD1SD2"}:
                    hrv_nl = nk.hrv_nonlinear(peaks, sampling_rate=1000, show=False)
                    if "SD1" in selected_set:
                        result["SD1"] = hrv_nl.get("HRV_SD1", [None])[0]
                    if "SD2" in selected_set:
                        result["SD2"] = hrv_nl.get("HRV_SD2", [None])[0]
                    if "SD1SD2" in selected_set:
                        result["SD1SD2"] = hrv_nl.get("HRV_SD1SD2", [None])[0]

                    # Entropy metrics
                    if "ApEn" in selected_set:
                        result["ApEn"] = hrv_nl.get("HRV_ApEn", [None])[0]
                    if "SampEn" in selected_set:
                        result["SampEn"] = hrv_nl.get("HRV_SampEn", [None])[0]

                    # DFA metrics
                    if "DFA_alpha1" in selected_set:
                        result["DFA_alpha1"] = hrv_nl.get("HRV_DFA_alpha1", [None])[0]
                    if "DFA_alpha2" in selected_set:
                        result["DFA_alpha2"] = hrv_nl.get("HRV_DFA_alpha2", [None])[0]
                else:
                    # Only entropy or DFA requested
                    hrv_nl = nk.hrv_nonlinear(peaks, sampling_rate=1000, show=False)
                    if "ApEn" in selected_set:
                        result["ApEn"] = hrv_nl.get("HRV_ApEn", [None])[0]
                    if "SampEn" in selected_set:
                        result["SampEn"] = hrv_nl.get("HRV_SampEn", [None])[0]
                    if "DFA_alpha1" in selected_set:
                        result["DFA_alpha1"] = hrv_nl.get("HRV_DFA_alpha1", [None])[0]
                    if "DFA_alpha2" in selected_set:
                        result["DFA_alpha2"] = hrv_nl.get("HRV_DFA_alpha2", [None])[0]
            except Exception:
                for m in selected_set & nonlinear:
                    result[m] = None

        return result

    if not use_windows or len(nn_ms_list) < window_beats:
        # Single analysis on full data
        metrics = compute_hrv(nn_ms_list)
        return metrics, None, 1

    # Overlapping windows analysis
    step_beats = int(window_beats * (1 - overlap_pct / 100))
    if step_beats < 1:
        step_beats = 1

    windows = generate_overlapping_windows_beats(nn_ms_list, window_beats, step_beats)

    if not windows:
        # Fallback to single analysis
        metrics = compute_hrv(nn_ms_list)
        return metrics, None, 1

    # Calculate HRV for each window
    window_results = []
    for _, _, window_rr in windows:
        try:
            window_hrv = compute_hrv(window_rr)
            window_results.append(window_hrv)
        except Exception:
            continue

    if not window_results:
        metrics = compute_hrv(nn_ms_list)
        return metrics, None, 1

    # Aggregate results
    metrics_df = pd.DataFrame(window_results)

    mean_metrics = {}
    std_metrics = {}
    for col in metrics_df.columns:
        values = metrics_df[col].dropna()
        if len(values) > 0:
            mean_metrics[col] = float(values.mean())
            std_metrics[col] = float(values.std()) if len(values) > 1 else 0.0
        else:
            mean_metrics[col] = None
            std_metrics[col] = None

    return mean_metrics, std_metrics, len(window_results)


# =============================================================================
# GROUP ANALYSIS RESULT AGGREGATION FUNCTIONS
# =============================================================================


def _results_to_long_df(results: list[ParticipantSectionResult]) -> pd.DataFrame:
    """Convert analysis results to long-format DataFrame.

    Args:
        results: List of ParticipantSectionResult objects

    Returns:
        DataFrame with one row per participant-section combination
    """
    rows = []
    for r in results:
        row = {
            "participant_id": r.participant_id,
            "group": r.group,
            "section": r.section_name,
            "n_beats": r.n_beats,
            "duration_s": r.duration_s,
            "quality": r.quality_grade,
            "artifact_rate": r.artifact_rate,
            "n_windows": r.n_windows,
        }
        # Add HRV metrics
        for key, value in r.hrv_metrics.items():
            row[key.lower()] = value
        # Add SD columns if available
        if r.hrv_std:
            for key, value in r.hrv_std.items():
                row[f"{key.lower()}_sd"] = value
        rows.append(row)

    return pd.DataFrame(rows)


def _results_to_wide_df(results: list[ParticipantSectionResult]) -> pd.DataFrame:
    """Convert analysis results to wide-format DataFrame.

    Args:
        results: List of ParticipantSectionResult objects

    Returns:
        DataFrame with one row per participant, columns like section_metric
    """
    # Group by participant
    participants = {}
    for r in results:
        if r.participant_id not in participants:
            participants[r.participant_id] = {"participant_id": r.participant_id, "group": r.group}

        prefix = r.section_name.replace(" ", "_").lower()

        # Add metrics with section prefix
        for key, value in r.hrv_metrics.items():
            col_name = f"{prefix}_{key.lower()}"
            participants[r.participant_id][col_name] = value

        # Add quality info
        participants[r.participant_id][f"{prefix}_n_beats"] = r.n_beats
        participants[r.participant_id][f"{prefix}_quality"] = r.quality_grade

    return pd.DataFrame(list(participants.values()))


def _calculate_group_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive statistics per group and section.

    Args:
        long_df: Long-format DataFrame from _results_to_long_df

    Returns:
        DataFrame with columns: group, section, metric, n, mean, sd, min, max
    """
    # Dynamically find all HRV metric columns (lowercase in dataframe)
    exclude_cols = {"participant_id", "group", "section", "n_beats", "duration_s",
                    "quality", "artifact_rate", "n_windows"}
    # Also exclude _sd columns (standard deviation columns)
    metrics = [col for col in long_df.columns
               if col not in exclude_cols and not col.endswith("_sd")]

    rows = []

    for (group, section), group_df in long_df.groupby(["group", "section"]):
        for metric in metrics:
            if metric not in group_df.columns:
                continue
            values = group_df[metric].dropna()
            if len(values) == 0:
                continue

            rows.append({
                "group": group,
                "section": section,
                "metric": metric.upper(),
                "n": len(values),
                "mean": round(values.mean(), 2),
                "sd": round(values.std(), 2) if len(values) > 1 else 0.0,
                "min": round(values.min(), 2),
                "max": round(values.max(), 2),
            })

    return pd.DataFrame(rows)


# =============================================================================
# GROUP ANALYSIS VISUALIZATION FUNCTIONS
# =============================================================================


def _create_group_bar_chart(
    stats_df: pd.DataFrame,
    metric: str,
    sections: list[str] | None = None,
):
    """Create a grouped bar chart for HRV metrics.

    Args:
        stats_df: DataFrame from _calculate_group_stats
        metric: Metric to plot (e.g., "RMSSD", "SDNN")
        sections: Optional list of sections to include

    Returns:
        Plotly Figure object
    """
    go, _ = get_plotly_analysis()
    if go is None:
        return None

    # Filter to specific metric
    df = stats_df[stats_df["metric"] == metric.upper()].copy()
    if df.empty:
        return None

    # Filter sections if specified
    if sections:
        df = df[df["section"].isin(sections)]

    # Get unique groups and sections
    groups = df["group"].unique().tolist()
    section_list = df["section"].unique().tolist()

    # Colors for sections
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6C757D", "#28A745"]

    fig = go.Figure()

    for i, section in enumerate(section_list):
        section_df = df[df["section"] == section]

        # Align with groups (may have missing data)
        means = []
        sds = []
        for group in groups:
            group_row = section_df[section_df["group"] == group]
            if not group_row.empty:
                means.append(group_row["mean"].values[0])
                sds.append(group_row["sd"].values[0])
            else:
                means.append(None)
                sds.append(None)

        fig.add_trace(
            go.Bar(
                name=section,
                x=groups,
                y=means,
                error_y=dict(type="data", array=sds, visible=True),
                marker_color=colors[i % len(colors)],
            )
        )

    # Get theme colors
    theme = get_theme_colors()

    fig.update_layout(
        title=dict(
            text=f"{metric.upper()} by Group and Section",
            font=dict(size=16),
        ),
        xaxis_title="Group",
        yaxis_title=f"{metric.upper()} (ms)" if metric.upper() not in ["LF_HF", "PNN50"] else metric.upper(),
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        plot_bgcolor=theme["bg"],
        paper_bgcolor=theme["bg"],
        font=dict(color=theme["text"]),
        margin=dict(l=60, r=20, t=80, b=60),
    )

    fig.update_xaxes(gridcolor=theme["grid"], showline=True, linewidth=1, linecolor=theme["grid"])
    fig.update_yaxes(gridcolor=theme["grid"], showline=True, linewidth=1, linecolor=theme["grid"])

    return fig


def _create_box_violin_plot(
    long_df: pd.DataFrame,
    metric: str,
    plot_type: str = "box",
    group_by: str = "group",
    color_by: str = "section",
):
    """Create a box plot or violin plot for HRV metrics.

    Args:
        long_df: Long-format DataFrame from _results_to_long_df
        metric: Metric column name (lowercase)
        plot_type: "box" or "violin"
        group_by: Column to use for x-axis grouping
        color_by: Column to use for color grouping

    Returns:
        Plotly Figure object
    """
    go, _ = get_plotly_analysis()
    if go is None:
        return None

    metric_lower = metric.lower()
    if metric_lower not in long_df.columns:
        return None

    # Filter out NaN values
    df = long_df[long_df[metric_lower].notna()].copy()
    if df.empty:
        return None

    theme = get_theme_colors()
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6C757D", "#28A745", "#17A2B8", "#FFC107"]

    fig = go.Figure()

    color_categories = df[color_by].unique().tolist()

    for i, color_cat in enumerate(color_categories):
        subset = df[df[color_by] == color_cat]

        if plot_type == "violin":
            fig.add_trace(
                go.Violin(
                    x=subset[group_by],
                    y=subset[metric_lower],
                    name=str(color_cat),
                    legendgroup=str(color_cat),
                    scalegroup=str(color_cat),
                    line_color=colors[i % len(colors)],
                    fillcolor=colors[i % len(colors)],
                    opacity=0.6,
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    pointpos=0,
                    jitter=0.05,
                )
            )
        else:  # box plot
            fig.add_trace(
                go.Box(
                    x=subset[group_by],
                    y=subset[metric_lower],
                    name=str(color_cat),
                    legendgroup=str(color_cat),
                    marker_color=colors[i % len(colors)],
                    line_color=colors[i % len(colors)],
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                )
            )

    # Get metric info for labels
    metric_info = get_metric_info(metric.upper())
    unit_str = f" ({metric_info['unit']})" if metric_info.get("unit") else ""

    fig.update_layout(
        title=dict(
            text=f"{metric.upper()} Distribution by {group_by.title()}",
            font=dict(size=16),
        ),
        xaxis_title=group_by.title(),
        yaxis_title=f"{metric.upper()}{unit_str}",
        boxmode="group" if plot_type == "box" else None,
        violinmode="group" if plot_type == "violin" else None,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        plot_bgcolor=theme["bg"],
        paper_bgcolor=theme["bg"],
        font=dict(color=theme["text"]),
        margin=dict(l=60, r=20, t=80, b=60),
    )

    fig.update_xaxes(gridcolor=theme["grid"], showline=True, linewidth=1, linecolor=theme["grid"])
    fig.update_yaxes(gridcolor=theme["grid"], showline=True, linewidth=1, linecolor=theme["grid"])

    return fig


def _create_sd1_sd2_scatter(
    long_df: pd.DataFrame,
    color_by: str = "group",
):
    """Create SD1 vs SD2 scatter plot (Poincaré-derived measures).

    Args:
        long_df: Long-format DataFrame from _results_to_long_df
        color_by: Column to use for color grouping ("group" or "section")

    Returns:
        Plotly Figure object
    """
    go, _ = get_plotly_analysis()
    if go is None:
        return None

    # Check if SD1 and SD2 are available
    if "sd1" not in long_df.columns or "sd2" not in long_df.columns:
        return None

    # Filter out NaN values
    df = long_df[long_df["sd1"].notna() & long_df["sd2"].notna()].copy()
    if df.empty:
        return None

    theme = get_theme_colors()
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6C757D", "#28A745", "#17A2B8", "#FFC107"]

    fig = go.Figure()

    categories = df[color_by].unique().tolist()

    for i, cat in enumerate(categories):
        subset = df[df[color_by] == cat]

        fig.add_trace(
            go.Scatter(
                x=subset["sd2"],
                y=subset["sd1"],
                mode="markers",
                name=str(cat),
                marker=dict(
                    size=10,
                    color=colors[i % len(colors)],
                    opacity=0.7,
                    line=dict(width=1, color="white"),
                ),
                text=subset["participant_id"],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "SD1: %{y:.2f} ms<br>"
                    "SD2: %{x:.2f} ms<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Add reference line (SD1 = SD2 means circular Poincaré)
    max_val = max(df["sd1"].max(), df["sd2"].max()) * 1.1
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="SD1 = SD2",
            line=dict(color="gray", dash="dash", width=1),
            showlegend=True,
        )
    )

    fig.update_layout(
        title=dict(
            text="Poincaré Plot Measures: SD1 vs SD2",
            font=dict(size=16),
        ),
        xaxis_title="SD2 (ms) - Long-term variability",
        yaxis_title="SD1 (ms) - Short-term variability",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        plot_bgcolor=theme["bg"],
        paper_bgcolor=theme["bg"],
        font=dict(color=theme["text"]),
        margin=dict(l=60, r=20, t=80, b=60),
    )

    fig.update_xaxes(gridcolor=theme["grid"], showline=True, linewidth=1, linecolor=theme["grid"])
    fig.update_yaxes(gridcolor=theme["grid"], showline=True, linewidth=1, linecolor=theme["grid"], scaleanchor="x")

    return fig


def _create_raincloud_plot(
    long_df: pd.DataFrame,
    metric: str,
    group_by: str = "group",
    color_by: str = "section",
):
    """Create a raincloud plot (half-violin + box + strip).

    Raincloud plots combine:
    - Half-violin showing distribution
    - Box plot showing quartiles
    - Individual points (strip/jitter)

    Args:
        long_df: Long-format DataFrame from _results_to_long_df
        metric: Metric column name (lowercase)
        group_by: Column for x-axis grouping
        color_by: Column for color grouping

    Returns:
        Plotly Figure object
    """
    go, _ = get_plotly_analysis()
    if go is None:
        return None

    metric_lower = metric.lower()
    if metric_lower not in long_df.columns:
        return None

    # Filter out NaN values
    df = long_df[long_df[metric_lower].notna()].copy()
    if df.empty:
        return None

    theme = get_theme_colors()
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6C757D", "#28A745", "#17A2B8", "#FFC107"]

    fig = go.Figure()

    color_categories = df[color_by].unique().tolist()

    for i, color_cat in enumerate(color_categories):
        subset = df[df[color_by] == color_cat]
        color = colors[i % len(colors)]

        # Half violin (positive side only)
        fig.add_trace(
            go.Violin(
                x=subset[group_by],
                y=subset[metric_lower],
                name=str(color_cat),
                legendgroup=str(color_cat),
                side="positive",
                line_color=color,
                fillcolor=color,
                opacity=0.5,
                meanline_visible=False,
                points=False,
                width=0.8,
            )
        )

        # Box plot (narrow, on left side)
        fig.add_trace(
            go.Box(
                x=subset[group_by],
                y=subset[metric_lower],
                name=str(color_cat),
                legendgroup=str(color_cat),
                marker_color=color,
                line_color=color,
                boxpoints=False,
                width=0.15,
                showlegend=False,
            )
        )

        # Individual points (strip with jitter)
        fig.add_trace(
            go.Scatter(
                x=[f"{g}" for g in subset[group_by]],
                y=subset[metric_lower],
                mode="markers",
                name=str(color_cat),
                legendgroup=str(color_cat),
                marker=dict(
                    size=5,
                    color=color,
                    opacity=0.6,
                ),
                showlegend=False,
                # Add jitter
                hovertemplate=(
                    f"<b>{color_cat}</b><br>"
                    f"{metric.upper()}: %{{y:.2f}}<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Get metric info for labels
    metric_info = get_metric_info(metric.upper())
    unit_str = f" ({metric_info['unit']})" if metric_info.get("unit") else ""

    fig.update_layout(
        title=dict(
            text=f"{metric.upper()} Raincloud Plot by {group_by.title()}",
            font=dict(size=16),
        ),
        xaxis_title=group_by.title(),
        yaxis_title=f"{metric.upper()}{unit_str}",
        violinmode="group",
        boxmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        plot_bgcolor=theme["bg"],
        paper_bgcolor=theme["bg"],
        font=dict(color=theme["text"]),
        margin=dict(l=60, r=20, t=80, b=60),
    )

    fig.update_xaxes(gridcolor=theme["grid"], showline=True, linewidth=1, linecolor=theme["grid"])
    fig.update_yaxes(gridcolor=theme["grid"], showline=True, linewidth=1, linecolor=theme["grid"])

    return fig


# =============================================================================
# GROUP ANALYSIS MAIN PIPELINE
# =============================================================================


def _run_group_analysis(config: dict) -> tuple[list[ParticipantSectionResult], dict, dict]:
    """Run HRV analysis for multiple groups.

    Args:
        config: Configuration dict with keys:
            - selected_groups: List of group names
            - sections_per_group: Dict mapping group -> list of section names
            - use_overlapping_windows: bool
            - window_beats: int
            - overlap_percent: float
            - completeness_filter: bool
            - selected_metrics: List of metric names to calculate

    Returns:
        Tuple of (results, missing, excluded)
        - results: List of ParticipantSectionResult
        - missing: Dict of {participant_id: {section: reason}}
        - excluded: Dict of {participant_id: reason}
    """
    results = []
    missing = {}
    excluded = {}

    project_path = st.session_state.get("project_path")
    data_dir = st.session_state.get("data_dir")
    selected_metrics = config.get("selected_metrics")

    group_participants = _collect_group_participants(config["selected_groups"])

    for group in config["selected_groups"]:
        sections = config["sections_per_group"].get(group, [])
        participants = group_participants.get(group, [])

        for pid in participants:
            # Find .rrational v2 file
            rrational_path = _find_rrational_v2_file(pid, project_path=project_path, data_dir=data_dir)

            if not rrational_path:
                missing[pid] = {"_all": "No .rrational v2 file found"}
                continue

            # Check which sections are available
            available = []
            for section in sections:
                nn_data, info = _load_nn_from_rrational_v2(rrational_path, section)
                if nn_data and len(nn_data) >= MIN_BEATS_TIME_DOMAIN:
                    available.append((section, nn_data, info))
                else:
                    missing.setdefault(pid, {})[section] = info.get("error", "Insufficient data")

            # Completeness filter
            if config["completeness_filter"] and len(available) < len(sections):
                excluded[pid] = f"Missing {len(sections) - len(available)} of {len(sections)} sections"
                continue

            # Calculate HRV for each available section
            for section, nn_data, info in available:
                metrics, std, n_win = _calculate_hrv_metrics(
                    nn_data,
                    config["use_overlapping_windows"],
                    config["window_beats"],
                    config["overlap_percent"],
                    selected_metrics=selected_metrics,
                )

                results.append(ParticipantSectionResult(
                    participant_id=pid,
                    group=group,
                    section_name=section,
                    n_beats=info.get("n_beats", len(nn_data)),
                    duration_s=info.get("duration_s", sum(nn_data) / 1000),
                    quality_grade=info.get("quality_grade", "unknown"),
                    artifact_rate=info.get("artifact_rate", 0.0),
                    hrv_metrics=metrics,
                    hrv_std=std,
                    n_windows=n_win,
                ))

    return results, missing, excluded


def _render_group_analysis():
    """Render group-level HRV analysis with multi-group support."""
    # Help expander
    with st.expander("**Group Analysis Best Practices**", expanded=False):
        st.markdown("""
**Overview:**
Group analysis allows you to compare HRV metrics across multiple groups and sections.
This feature uses `.rrational` v2 files which contain pre-processed NN intervals.

**Requirements:**
- Participants must have `.rrational` v2 files exported (from the Participants tab)
- At least 100 beats per section for time-domain metrics
- At least 300 beats per section for frequency-domain metrics

**Overlapping Windows:**
Using overlapping windows improves the reliability of HRV estimates by providing:
- Multiple measurements per participant per section
- Standard deviation as a measure of within-participant variability
- Reduced impact of transient artifacts

**Recommended settings:** 150 beats/window, 75% overlap
        """)

    # Check prerequisites
    available_sections = list(st.session_state.sections.keys())
    if not available_sections:
        st.warning("No sections defined. Please define sections in the Sections tab first.")
        return

    group_list = list(st.session_state.groups.keys())
    if not group_list:
        st.warning("No groups defined. Please define groups in the Participants tab first.")
        return

    # -------------------------------------------------------------------------
    # Step 1: Select Groups
    # -------------------------------------------------------------------------
    st.markdown("### Step 1: Select Groups")

    # Count participants per group
    group_counts = {}
    for group in group_list:
        count = sum(1 for g in st.session_state.participant_groups.values() if g == group)
        group_counts[group] = count

    # Multi-select with counts
    group_options = [f"{g} ({group_counts[g]} participants)" for g in group_list]
    group_map = {f"{g} ({group_counts[g]} participants)": g for g in group_list}

    selected_group_labels = st.multiselect(
        "Select groups to analyze",
        options=group_options,
        default=group_options,  # All groups selected by default
        key="group_analysis_groups",
    )
    selected_groups = [group_map[label] for label in selected_group_labels]

    if not selected_groups:
        st.info("Please select at least one group to analyze.")
        return

    # -------------------------------------------------------------------------
    # Step 2: Configure Sections per Group
    # -------------------------------------------------------------------------
    st.markdown("### Step 2: Configure Sections")

    sections_per_group = {}
    for group in selected_groups:
        with st.expander(f"**{group}** - Select sections", expanded=True):
            sections_per_group[group] = st.multiselect(
                f"Sections for {group}",
                options=available_sections,
                default=available_sections,  # All sections by default
                key=f"group_analysis_sections_{group}",
                label_visibility="collapsed",
            )

    # Check if any sections selected
    total_sections = sum(len(s) for s in sections_per_group.values())
    if total_sections == 0:
        st.info("Please select at least one section for at least one group.")
        return

    # -------------------------------------------------------------------------
    # Step 3: Analysis Options
    # -------------------------------------------------------------------------
    st.markdown("### Step 3: Analysis Options")

    # Metric preset selection
    st.markdown("**HRV Metrics**")
    preset_names = list(HRV_METRIC_PRESETS.keys())
    preset_col1, preset_col2 = st.columns([1, 2])

    with preset_col1:
        selected_preset = st.selectbox(
            "Metric preset",
            options=preset_names,
            index=1,  # Default to "Time + Frequency"
            key="group_analysis_metric_preset",
            help="Choose a preset or select 'Custom' to pick individual metrics",
        )

    # Show preset description
    with preset_col2:
        preset_info = HRV_METRIC_PRESETS[selected_preset]
        st.caption(preset_info["description"])

    # Custom metric selection
    if selected_preset == "Custom":
        # Organize by category for easier selection
        st.markdown("**Select metrics:**")
        metric_cols = st.columns(4)

        selected_metrics = []
        categories = [
            ("Time (Basic)", "time_basic"),
            ("Time (Extended)", "time_extended"),
            ("Frequency", "frequency"),
            ("Nonlinear", "nonlinear"),
        ]

        for i, (cat_label, cat_key) in enumerate(categories):
            with metric_cols[i]:
                st.markdown(f"*{cat_label}*")
                for metric_name in HRV_METRICS_CATALOG[cat_key].keys():
                    metric_info = HRV_METRICS_CATALOG[cat_key][metric_name]
                    if st.checkbox(
                        metric_info["label"],
                        value=metric_name in ["RMSSD", "SDNN", "MeanHR"],  # Default selection
                        key=f"group_metric_{metric_name}",
                        help=metric_info["description"],
                    ):
                        selected_metrics.append(metric_name)
    else:
        selected_metrics = HRV_METRIC_PRESETS[selected_preset]["metrics"]

    # Show selected metrics summary
    if selected_metrics:
        st.caption(f"**Selected:** {', '.join(selected_metrics[:8])}{'...' if len(selected_metrics) > 8 else ''} ({len(selected_metrics)} metrics)")
    else:
        st.warning("Please select at least one metric.")
        return

    st.markdown("**Analysis Settings**")
    col1, col2 = st.columns(2)
    with col1:
        use_overlapping = st.checkbox(
            "Use overlapping windows",
            value=True,
            key="group_analysis_overlapping",
            help="Calculate HRV using overlapping windows for more reliable estimates",
        )
    with col2:
        completeness_filter = st.checkbox(
            "Only complete participants",
            value=False,
            key="group_analysis_completeness",
            help="Exclude participants missing any selected sections",
        )

    if use_overlapping:
        col1, col2 = st.columns(2)
        with col1:
            window_beats = st.number_input(
                "Window size (beats)",
                min_value=50,
                max_value=500,
                value=150,
                step=10,
                key="group_analysis_window_beats",
            )
        with col2:
            overlap_pct = st.slider(
                "Overlap (%)",
                min_value=0,
                max_value=90,
                value=75,
                step=5,
                key="group_analysis_overlap",
            )
    else:
        window_beats = 150
        overlap_pct = 75

    # -------------------------------------------------------------------------
    # Run Analysis Button
    # -------------------------------------------------------------------------
    st.divider()

    if st.button("**Analyze Groups**", key="run_group_analysis_btn", type="primary", use_container_width=True):
        # Build configuration
        config = {
            "selected_groups": selected_groups,
            "sections_per_group": sections_per_group,
            "use_overlapping_windows": use_overlapping,
            "window_beats": window_beats,
            "overlap_percent": overlap_pct,
            "completeness_filter": completeness_filter,
            "selected_metrics": selected_metrics,
        }

        # Count total participants
        total_participants = sum(
            group_counts[g] for g in selected_groups
        )

        with st.status(f"Analyzing {total_participants} participants across {len(selected_groups)} groups...", expanded=True) as status:
            st.write("Starting analysis...")

            # Run the analysis
            results, missing, excluded = _run_group_analysis(config)

            status.update(label="Analysis complete!", state="complete")

        # Store results in session state for persistence
        st.session_state.group_analysis_results = {
            "results": results,
            "missing": missing,
            "excluded": excluded,
            "config": config,
        }

        show_toast(f"Analysis complete: {len(results)} participant-section results", icon="success")

    # -------------------------------------------------------------------------
    # Display Results
    # -------------------------------------------------------------------------
    if "group_analysis_results" not in st.session_state:
        return

    stored = st.session_state.group_analysis_results
    results = stored["results"]
    missing = stored["missing"]
    excluded = stored["excluded"]
    config = stored["config"]

    if not results:
        st.warning("No results available. Check the missing sections report below.")
        # Show missing info
        if missing or excluded:
            with st.expander("**Missing / Excluded Participants**", expanded=True):
                if excluded:
                    st.markdown("**Excluded (completeness filter):**")
                    for pid, reason in excluded.items():
                        st.write(f"- `{pid}`: {reason}")
                if missing:
                    st.markdown("**Missing data:**")
                    for pid, sections in missing.items():
                        section_list = ", ".join(f"{s}: {r}" for s, r in sections.items())
                        st.write(f"- `{pid}`: {section_list}")
        return

    # Summary
    st.markdown("---")
    n_participants = len(set(r.participant_id for r in results))
    n_sections = len(set(r.section_name for r in results))
    n_groups = len(set(r.group for r in results))

    st.markdown(f"""
### Results Summary
- **{n_participants}** participants analyzed
- **{n_groups}** groups
- **{n_sections}** sections
- **{len(results)}** total participant-section combinations
    """)

    # Missing sections (collapsible)
    if missing or excluded:
        with st.expander(f"**Missing / Excluded ({len(missing) + len(excluded)} participants)**", expanded=False):
            if excluded:
                st.markdown("**Excluded (completeness filter):**")
                for pid, reason in excluded.items():
                    st.write(f"- `{pid}`: {reason}")
            if missing:
                st.markdown("**Missing data:**")
                for pid, sections in missing.items():
                    section_list = ", ".join(f"{s}: {r}" for s, r in sections.items())
                    st.write(f"- `{pid}`: {section_list}")

    # Convert to DataFrames
    long_df = _results_to_long_df(results)
    wide_df = _results_to_wide_df(results)
    stats_df = _calculate_group_stats(long_df)

    # Tabs for different views
    tab_data, tab_stats, tab_chart = st.tabs(["**Data**", "**Statistics**", "**Chart**"])

    with tab_data:
        # Format toggle
        format_choice = st.radio(
            "Data format",
            options=["Long (one row per section)", "Wide (one row per participant)"],
            horizontal=True,
            key="group_analysis_format",
        )

        if "Long" in format_choice:
            st.dataframe(long_df, use_container_width=True, height=400)
            csv_data = long_df.to_csv(index=False)
            filename = "hrv_group_results_long.csv"
        else:
            st.dataframe(wide_df, use_container_width=True, height=400)
            csv_data = wide_df.to_csv(index=False)
            filename = "hrv_group_results_wide.csv"

        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            key="download_group_data",
        )

    with tab_stats:
        st.markdown("**Descriptive Statistics by Group and Section**")
        st.dataframe(stats_df, use_container_width=True, height=400)

        stats_csv = stats_df.to_csv(index=False)
        st.download_button(
            label="Download Statistics CSV",
            data=stats_csv,
            file_name="hrv_group_statistics.csv",
            mime="text/csv",
            key="download_group_stats",
        )

    with tab_chart:
        # Visualization type selector
        viz_type = st.radio(
            "Visualization type",
            options=["Bar Chart", "Box Plot", "Violin Plot", "Raincloud Plot", "SD1/SD2 Scatter"],
            horizontal=True,
            key="group_analysis_viz_type",
        )

        # Build available metrics from actual data columns
        available_metrics = []
        for col in long_df.columns:
            col_upper = col.upper()
            if col_upper in ALL_HRV_METRICS and long_df[col].notna().any():
                available_metrics.append(col_upper)

        # Ensure basic metrics are first
        priority_order = ["RMSSD", "SDNN", "PNN50", "MEANNN", "MEANHR", "LF", "HF", "LF_HF", "SD1", "SD2"]
        available_metrics = sorted(
            available_metrics,
            key=lambda x: priority_order.index(x) if x in priority_order else 100
        )

        if viz_type == "SD1/SD2 Scatter":
            # Special handling for SD1/SD2 scatter
            if "sd1" in long_df.columns and "sd2" in long_df.columns:
                color_by = st.radio(
                    "Color by",
                    options=["group", "section"],
                    horizontal=True,
                    key="group_analysis_scatter_color",
                )

                fig = _create_sd1_sd2_scatter(long_df, color_by=color_by)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No SD1/SD2 data available. Select 'Poincaré Focus' or 'Full' preset to include these metrics.")
            else:
                st.info("SD1 and SD2 metrics not available. Run analysis with 'Poincaré Focus' or 'Full (with nonlinear)' preset.")
        else:
            # Metric selector for other charts
            if not available_metrics:
                st.warning("No metrics available in the results.")
            else:
                col1, col2 = st.columns([2, 1])
                with col1:
                    selected_chart_metric = st.selectbox(
                        "Select metric to visualize",
                        options=available_metrics,
                        key="group_analysis_chart_metric",
                    )

                # Section filter
                all_sections = sorted(long_df["section"].unique().tolist())
                chart_sections = st.multiselect(
                    "Sections to include",
                    options=all_sections,
                    default=all_sections,
                    key="group_analysis_chart_sections",
                )

                if not chart_sections:
                    st.info("Select at least one section to display the chart.")
                else:
                    # Filter data for selected sections
                    filtered_df = long_df[long_df["section"].isin(chart_sections)]

                    if viz_type == "Bar Chart":
                        fig = _create_group_bar_chart(stats_df, selected_chart_metric, chart_sections)
                    elif viz_type == "Box Plot":
                        fig = _create_box_violin_plot(
                            filtered_df, selected_chart_metric,
                            plot_type="box", group_by="group", color_by="section"
                        )
                    elif viz_type == "Violin Plot":
                        fig = _create_box_violin_plot(
                            filtered_df, selected_chart_metric,
                            plot_type="violin", group_by="group", color_by="section"
                        )
                    elif viz_type == "Raincloud Plot":
                        fig = _create_raincloud_plot(
                            filtered_df, selected_chart_metric,
                            group_by="group", color_by="section"
                        )
                    else:
                        fig = None

                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"No data available for {selected_chart_metric} in selected sections.")
