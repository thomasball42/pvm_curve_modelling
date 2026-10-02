# -*- coding: utf-8 -*-
"""
Overlay helpers for supplementary-figure compliance.

The journal requires that any figure presenting data as a range (box, violin,
min-max band, mean +/- SD band) also shows the individual underlying data
points, states a precise sample size as "n=X", and defines what the range is.

Nothing here mutates an existing figure's data or geometry.  The intended
pattern is "save, overlay, save again":

    save_figure(fig, "my_figure", subdir=SUBDIR)                  # original
    _, n_tot, n_sh = _overlay.jittered_dots(ax, 0, vals, width=0.5, glyph=0)
    _overlay.annotate_n(ax, 0, n_tot, n_sh)
    save_figure(fig, "my_figure" + _overlay.SUFFIX, subdir=SUBDIR)

so the "_wdots" output is provably the same figure plus overlay artists.

Two rules that are easy to get wrong:

1. Z-order.  violinplot bodies and fill_between are zorder 1, and ax.scatter
   ALSO defaults to zorder 1.  Axes.draw sorts stably, so a scatter added after
   a violin ties and draws ON TOP of it.  An explicit zorder is therefore
   required, not optional:
     * beneath glyphs whose fill carries information (violin density, box
       quartiles)                            -> zorder 0.5 (the default here)
     * above a glyph that is merely the envelope of the very curves being
       overlaid (min-max band, +/-SD band)   -> pass zorder=1.5
   patch_artist boxes are zorder 2 and live in ax.patches, not ax.collections.

2. Never re-run fig.tight_layout() in a "_wdots" block.  Every annotation here
   is placed inside the axes (axes-fraction or data coordinates) precisely so
   the overlaid figure keeps the original's exact geometry.

The VALUE axis is never jittered -- only the category axis.

@author: Thomas Ball
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# --- output naming -------------------------------------------------------
SUFFIX = "_wdots"

# --- reproducible subsampling -------------------------------------------
RNG_SEED = 20260101            # FIXED.  Do not change once figures are submitted.
MAX_POINTS_PER_GLYPH = 500     # cap per categorical glyph
MAX_CURVES_PER_BAND = 150      # cap per band panel

# --- dot style -----------------------------------------------------------
DOT_SIZE = 2.0                 # points^2
DOT_COLOUR = "#2b2b2b"
DOT_ALPHA = 0.35
DOT_ZORDER = 0.5
JITTER_FRAC = 0.60             # jitter band = JITTER_FRAC * glyph width

# --- faint-curve style ---------------------------------------------------
CURVE_COLOUR = "#404040"
CURVE_ALPHA = 0.18
CURVE_LW = 0.4
CURVE_ZORDER = 0.5

# --- n= annotation style -------------------------------------------------
N_LABEL_SIZE = 7               # matches the existing label in figure_gamma_ci.py
N_LABEL_COLOUR = "#555555"
N_LABEL_Y = 0.02               # axes fraction

# --- rasterization -------------------------------------------------------
RASTER_THRESHOLD = 2000        # marks above which a dot overlay is rasterized
RASTER_VERTEX_THRESHOLD = 40000
RASTER_DPI = 600


def make_rng(seed=RNG_SEED, glyph=0, rng=None):
    """Deterministic, order-independent RNG for one glyph.

    Seeding on (seed, glyph) rather than advancing one shared stream means the
    draw for glyph k does not depend on how many glyphs were drawn before it,
    and two glyphs that happen to have identical n (e.g. models A and B in
    figure_gamma_dist.py, both n=82) still get different jitter patterns.
    """
    if rng is not None:
        return rng
    return np.random.default_rng([int(seed), int(glyph)])


def subsample(values, cap=MAX_POINTS_PER_GLYPH, seed=RNG_SEED, glyph=0, rng=None):
    """Return (sample, n_total, n_shown) for one glyph.

    `values` is flattened and stripped of non-finite entries, so n_total is
    exactly the count the figure's own statistics rest on -- pass the array the
    script has ALREADY filtered, not the raw column.
    """
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    n_total = int(v.size)
    if cap is None or n_total <= cap:
        return v, n_total, n_total
    r = make_rng(seed, glyph, rng)
    idx = np.sort(r.choice(n_total, size=int(cap), replace=False))
    return v[idx], n_total, int(cap)


def _jitter_x(position, width, n, log_x, jitter_frac, rng):
    """Random horizontal offsets within a glyph, for legibility only."""
    half = 0.5 * jitter_frac * float(width)
    if n == 0 or half <= 0:
        return np.full(n, float(position))
    u = rng.uniform(-1.0, 1.0, size=n)
    if not log_x:
        return position + u * half
    if position <= 0:
        raise ValueError("log_x jitter requires a strictly positive position")
    # On a log axis `widths` is still given in DATA units, i.e. as a fraction
    # of `position` (see the comment in figure_redlist_comparison_violin.py).
    # Multiplicative jitter therefore keeps the band a constant width on screen.
    return position * np.exp(u * (half / position))


def jittered_dots(ax, position, values, width,
                  log_x=False, cap=MAX_POINTS_PER_GLYPH,
                  seed=RNG_SEED, glyph=0, rng=None,
                  jitter_frac=JITTER_FRAC,
                  s=DOT_SIZE, color=DOT_COLOUR, alpha=DOT_ALPHA,
                  zorder=DOT_ZORDER, rasterized=None,
                  freeze_limits=True, **kwargs):
    """Overlay `values` as a jittered dot column on one categorical glyph.

    position : glyph centre in DATA units -- the value passed to
               boxplot/violinplot `positions`.
    width    : glyph FULL width in DATA units -- the value passed to `widths`.
    log_x    : True when the category axis is log-scaled; jitter then becomes
               multiplicative.
    values   : must already be in the figure's own y units.  If the figure
               plots log10(x) (figure_gamma_ci.py does), pass log10 of the
               values, not the values.
    glyph    : integer index of this glyph, for distinct reproducible jitter.

    Returns (PathCollection | None, n_total, n_shown).
    """
    sample, n_total, n_shown = subsample(values, cap=cap, seed=seed,
                                         glyph=glyph, rng=rng)
    if n_shown == 0:
        return None, n_total, 0
    jrng = make_rng(seed, glyph + 10_000, rng)       # separate substream
    x = _jitter_x(position, width, n_shown, log_x, jitter_frac, jrng)
    if rasterized is None:
        rasterized = n_shown > RASTER_THRESHOLD
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    sc = ax.scatter(x, sample, s=s, color=color, alpha=alpha,
                    marker="o", linewidths=0, edgecolors="none",
                    zorder=zorder, rasterized=rasterized,
                    label="_nolegend_", **kwargs)
    if freeze_limits:
        # an overlay must never rescale the axes, or the "_wdots" figure would
        # no longer be directly comparable with the original
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    return sc, n_total, n_shown


def faint_curves(ax, x, curves,
                 cap=MAX_CURVES_PER_BAND, seed=RNG_SEED, glyph=0, rng=None,
                 color=CURVE_COLOUR, alpha=CURVE_ALPHA, lw=CURVE_LW,
                 zorder=CURVE_ZORDER, linestyle="-",
                 rasterized=None, label=None, **kwargs):
    """Overlay the individual replicate curves that a band summarises.

    x      : shape (n_x,) shared x grid.
    curves : shape (n_runs, n_x), one row per run, in the figure's own y units
             (pass 1 - P if the figure plots 1 - P).

    Drawn as ONE LineCollection added with autolim=False, so the band's axis
    limits are untouched -- essential, because individual curves can run far
    outside a mean +/- SD band.  Curves outside the limits are clipped; use
    count_clipped_curves() to report how many, per the honesty requirement.

    Returns (LineCollection | None, n_total, n_shown).
    """
    arr = np.atleast_2d(np.asarray(curves, dtype=float))
    xx = np.asarray(x, dtype=float).ravel()
    if arr.shape[1] != xx.size:
        raise ValueError(f"curves has {arr.shape[1]} columns, x has {xx.size}")
    arr = arr[np.isfinite(arr).any(axis=1)]
    n_total = int(arr.shape[0])
    if n_total == 0:
        return None, 0, 0
    if cap is not None and n_total > cap:
        r = make_rng(seed, glyph, rng)
        arr = arr[np.sort(r.choice(n_total, size=int(cap), replace=False))]
    n_shown = int(arr.shape[0])
    if rasterized is None:
        rasterized = n_shown * xx.size > RASTER_VERTEX_THRESHOLD
    segs = [np.column_stack((xx, row)) for row in arr]
    lc = LineCollection(segs, colors=color, linewidths=lw, alpha=alpha,
                        linestyles=linestyle, zorder=zorder,
                        rasterized=rasterized, **kwargs)
    lc.set_label(label if label else "_nolegend_")
    ax.add_collection(lc, autolim=False)
    return lc, n_total, n_shown


def n_text(n_total, n_shown=None, prefix="", sep=" "):
    """'n=1,615 (500 shown)' -- the 'shown' clause appears only if subsampled."""
    txt = f"{prefix}n={n_total:,}"
    if n_shown is not None and n_shown < n_total:
        txt += f"{sep}({n_shown:,} shown)"
    return txt


def annotate_n(ax, position, n_total, n_shown=None,
               y=N_LABEL_Y, fontsize=N_LABEL_SIZE, color=N_LABEL_COLOUR,
               ha="center", va="bottom", rotation=0, prefix="", sep=" ",
               **kwargs):
    """Write "n=X (Y shown)" under a glyph: x in DATA units, y in AXES fraction.

    Uses ax.get_xaxis_transform(), the idiom already used in figure_gamma_ci.py.
    Because y is an axes fraction the label stays INSIDE the existing axes bbox,
    so tight_layout() need not be re-run and the "_wdots" figure keeps the
    original's exact geometry.  Works unchanged on a log-scaled category axis.
    """
    return ax.text(position, y, n_text(n_total, n_shown, prefix, sep),
                   transform=ax.get_xaxis_transform(),
                   ha=ha, va=va, fontsize=fontsize, color=color,
                   rotation=rotation, **kwargs)


def annotate_n_corner(ax, n_total, n_shown=None, loc=(0.03, 0.97),
                      fontsize=N_LABEL_SIZE, color=N_LABEL_COLOUR,
                      ha="left", va="top", prefix="", sep=" ", **kwargs):
    """Same string in an axes corner -- for band panels with no category axis."""
    return ax.text(loc[0], loc[1], n_text(n_total, n_shown, prefix, sep),
                   transform=ax.transAxes, ha=ha, va=va,
                   fontsize=fontsize, color=color, **kwargs)


def count_clipped(ax, values, axis="y"):
    """How many of `values` fall outside the current axis limits, i.e. are
    clipped out of the overlay.  Report this in the caption."""
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    return int(((v < min(lo, hi)) | (v > max(lo, hi))).sum())


def count_clipped_curves(ax, curves):
    """How many whole curves leave the panel at any x.  Report in the caption."""
    lo, hi = ax.get_ylim()
    arr = np.atleast_2d(np.asarray(curves, dtype=float))
    out = (arr < min(lo, hi)) | (arr > max(lo, hi))
    return int(np.nansum(out.any(axis=1)))


def use_raster_dpi(dpi=RASTER_DPI):
    """MUST be called before a "_wdots" save if any overlay is rasterized.

    _figure_io.save_figure() passes `dpi` only to the PNG savefig; the PDF
    savefig gets NO dpi, so rasterized artists land in the PDF at
    rcParams["savefig.dpi"], which is the matplotlib default "figure"
    (== fig.dpi == 100) unless _figure_io.use_main_fig_style() has been called
    -- and none of the target scripts call it.  100 dpi raster in a vector PDF
    would be rejected.  Calling this AFTER the original save leaves the
    original file untouched.
    """
    plt.rcParams["savefig.dpi"] = dpi
