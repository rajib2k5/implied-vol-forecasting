"""ivf: forecast implied volatility (VIX first) and trade the forecasts.

This package does all of the computation (ADR 0001 D1). The React app in
app/ only draws what tools/export_app_data.py writes.
"""

from importlib.metadata import PackageNotFoundError, version

# Read the version from the installed package metadata, which uv builds from
# pyproject.toml. One source of truth: bumping pyproject.toml is enough.
try:
    __version__ = version("implied-vol-forecasting")
except PackageNotFoundError:  # running from a plain checkout that was never installed
    __version__ = "0.0.0+uninstalled"

__all__ = ["__version__"]
