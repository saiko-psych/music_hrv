"""Tab modules for the Music HRV GUI."""

__all__ = [
    "render_data_tab",
    "render_participant_tab",
    "render_setup_tab",
    "render_analysis_tab",
]


def __getattr__(name: str):
    """Lazy import tab modules to avoid loading everything on package import."""
    if name == "render_data_tab":
        from rrational.gui.tabs.data import render_data_tab
        return render_data_tab
    elif name == "render_participant_tab":
        from rrational.gui.tabs.participant import render_participant_tab
        return render_participant_tab
    elif name == "render_setup_tab":
        from rrational.gui.tabs.setup import render_setup_tab
        return render_setup_tab
    elif name == "render_analysis_tab":
        from rrational.gui.tabs.analysis import render_analysis_tab
        return render_analysis_tab
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
