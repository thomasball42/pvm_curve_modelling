from pathlib import Path

import matplotlib

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
VECTOR_FIG_DIR = RESULTS_DIR / "vector_figs"
# main-text figures are collected here, alongside results/
MAIN_FIG_DIR = RESULTS_DIR.parent / "main_figs"

SANS_SERIF_STACK = [
    "Helvetica",
    "Helvetica Neue",
    "Arial",
    "Nimbus Sans",
    "Liberation Sans",
    "FreeSans",
    "DejaVu Sans",
]

FONT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": SANS_SERIF_STACK,
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    # keep text as editable TrueType in the vector outputs rather than Type 3
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # keep SVG text as text rather than outlined paths
    "svg.fonttype": "none",
}


def use_sans_serif_font():
    """Apply the Helvetica-first font stack to every figure in this session."""
    matplotlib.rcParams.update(FONT_RC)


# applied on import so all plot scripts pick it up, not just main-text figures
use_sans_serif_font()

MM_PER_INCH = 25.4
# Nature Communications: main-text figures must be no wider than 180 mm, but the
# journal accepts up to 210 mm (full page width) at submission.
MAIN_FIG_MAX_WIDTH_MM = 180
MAIN_FIG_MAX_WIDTH_IN = MAIN_FIG_MAX_WIDTH_MM / MM_PER_INCH

# ---------------------------------------------------------------------------
# Shared typography for main-text figures. Change sizes here only.
# ---------------------------------------------------------------------------
TICK_LABEL_SIZE = 7
AXIS_LABEL_SIZE = 7
TITLE_SIZE = 7
LEGEND_SIZE = 7
PANEL_LABEL_SIZE = 7

MAIN_FIG_RC = {
    **FONT_RC,
    "font.size": AXIS_LABEL_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "legend.title_fontsize": LEGEND_SIZE,
    "figure.labelsize": AXIS_LABEL_SIZE,
    "figure.titlesize": TITLE_SIZE,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "savefig.dpi": 300,
}


def use_main_fig_style():
    """Apply the shared main-figure typography. Call before creating figures."""
    matplotlib.rcParams.update(MAIN_FIG_RC)


def fit_main_fig_width(fig, max_width_mm=MAIN_FIG_MAX_WIDTH_MM, retighten=True):
    """Shrink fig (preserving aspect ratio) so it is at most max_width_mm wide."""
    max_width_in = max_width_mm / MM_PER_INCH
    width_in, height_in = fig.get_size_inches()
    if width_in <= max_width_in:
        return fig
    scale = max_width_in / width_in
    fig.set_size_inches(max_width_in, height_in * scale)
    if retighten:
        try:
            fig.tight_layout()
        except Exception:
            pass
    return fig


def save_figure(fig, name, subdir=None, dpi=300, main_fig=False, svg=False, **kwargs):
    if main_fig:
        fit_main_fig_width(fig)
    stem = Path(name).stem
    png_dir = FIG_DIR / subdir if subdir else FIG_DIR
    pdf_dir = VECTOR_FIG_DIR / subdir if subdir else VECTOR_FIG_DIR
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_path = png_dir / f"{stem}.png"
    pdf_path = pdf_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, **kwargs)
    fig.savefig(pdf_path, **kwargs)
    if svg:
        fig.savefig(pdf_dir / f"{stem}.svg", **kwargs)
    if main_fig:
        MAIN_FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(MAIN_FIG_DIR / f"{stem}.png", dpi=dpi, **kwargs)
        fig.savefig(MAIN_FIG_DIR / f"{stem}.pdf", **kwargs)
        if svg:
            fig.savefig(MAIN_FIG_DIR / f"{stem}.svg", **kwargs)
    return png_path, pdf_path
