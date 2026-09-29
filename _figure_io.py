from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
VECTOR_FIG_DIR = RESULTS_DIR / "vector_figs"


def save_figure(fig, name, subdir=None, dpi=300, **kwargs):
    stem = Path(name).stem
    png_dir = FIG_DIR / subdir if subdir else FIG_DIR
    pdf_dir = VECTOR_FIG_DIR / subdir if subdir else VECTOR_FIG_DIR
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    png_path = png_dir / f"{stem}.png"
    pdf_path = pdf_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, **kwargs)
    fig.savefig(pdf_path, **kwargs)
    return png_path, pdf_path
