"""Streamlit-based GUI exposing the pipeline to non-programmers."""

__all__: list[str] = ["main"]


def __getattr__(name: str):
    """Lazy import to avoid loading app.py on package import."""
    if name == "main":
        from rrational.gui.app import main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
