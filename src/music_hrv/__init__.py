"""Top-level package for the music HRV analysis toolkit."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("music-hrv")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"


__all__ = ["__version__"]
