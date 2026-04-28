"""Shared matplotlib styling for figures."""

from __future__ import annotations

import matplotlib as mpl


def use_default_style() -> None:
    """Apply a consistent style to all FresnelAnts figures."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.family": "DejaVu Sans",
            "image.cmap": "viridis",
        }
    )


# Apply on import so all viz functions see consistent rcParams.
use_default_style()
