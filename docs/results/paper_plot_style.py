"""Shared matplotlib style for the learning-curve figures in docs/03-experiments.md."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tciep_mpl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT_SIZE = 9
matplotlib.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 0.5,
        "axes.titleweight": "normal",
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "lines.markersize": 3,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

# Okabe-Ito, colour-blind safe and distinguishable in greyscale by lightness.
BASE = "#6f6f6f"  # prefix_base / prompt-free reference
FULL = "#0072B2"  # topo_prompt_full / coord_gen_full
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
UNIVERSE_COLORS = {"train": "#0072B2", "val": "#E69F00", "test": "#D55E00"}


def save(fig: plt.Figure, root, name: str) -> None:
    """Write png/svg/pdf next to the script that built the figure."""
    for ext, dpi in (("png", 220), ("svg", 300), ("pdf", 300)):
        fig.savefig(root / f"{name}.{ext}", dpi=dpi, facecolor="white")
    plt.close(fig)
