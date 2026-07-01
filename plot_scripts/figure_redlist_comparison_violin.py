import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

sys.path.append(str(Path(__file__).resolve().parents[1]))
from other_scripts import process_mammal_data
import _curve_fit


model_names = ["model_A", "model_B", "model_C", "model_D"]
figs_dir = Path("..", "figs", "figs_redlist_comparison")

quants = [0, 0.25, 0.75, 1]
COLOURS = {"CR": "#17DCE3", "EN": "#491EA7", "VU": "#D2E317"}

redlist_criteria = {
    "CR": {"name": "Critically Endangered", "time": 10, "num_gens": 3, "p_ext": 0.5, "mature_individs": 50},
    "EN": {"name": "Endangered", "time": 20, "num_gens": 5, "p_ext": 0.2, "mature_individs": 250},
    "VU": {"name": "Vulnerable", "time": 100, "num_gens": 0, "p_ext": 0.1, "mature_individs": 1000},
}

mammal_df = process_mammal_data.return_extended_mammal_data(
    redlist_criteria=redlist_criteria, save_file=True
)

# Load fit data
all_data_fits = {}
for model_name in model_names:
    path = Path("..", "results", "data_fits", "data_fits_main", f"data_fits_{model_name}.csv")
    df = pd.read_csv(path, index_col=0).dropna(subset=["param_a", "param_b", "param_alpha"])
    df = df[df.MAX_Y > 0.999]
    all_data_fits[model_name] = df.copy()

# Red List mammal quantiles
cr_quants = {}
for tc in redlist_criteria:
    mammal_ext_risks_100 = mammal_df[f"{tc}_ext_risk_eq100"]
    cr_quants[tc] = mammal_ext_risks_100.quantile(quants).to_dict()

criteria_order = ["CR", "EN", "VU"]
thresholds = {tc: redlist_criteria[tc]["mature_individs"] for tc in criteria_order}

# ------------------------------------------------------------------
# Build pooled model values at each criterion threshold
# ------------------------------------------------------------------
pooled_vals_by_tc = {tc: [] for tc in criteria_order}

for tc in criteria_order:
    n_thresh = thresholds[tc]
    vals_all_models = []

    for model_name in model_names:
        data_fits = all_data_fits[model_name]
        vals = 1 - _curve_fit.mod_gompertz(
            n_thresh,
            data_fits["param_a"].to_numpy(),
            data_fits["param_b"].to_numpy(),
            data_fits["param_alpha"].to_numpy(),
        )
        vals_all_models.append(vals)

    vals_all_models = np.concatenate(vals_all_models)
    vals_all_models = vals_all_models[np.isfinite(vals_all_models)]
    pooled_vals_by_tc[tc] = vals_all_models

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

# Two mini-columns in log2 space:
# violin column (left), redlist points column (right)
VIOLIN_SHIFT = -0.14
REDLIST_SHIFT = +0.14

violin_positions = [thresholds[tc] * (2 ** VIOLIN_SHIFT) for tc in criteria_order]
violin_data = [pooled_vals_by_tc[tc] for tc in criteria_order]

# Widths are in x-data units even on log axis, so scale with x
violin_widths = [p * 0.20 for p in violin_positions]

vp = ax.violinplot(
    violin_data,
    positions=violin_positions,
    widths=violin_widths,
    showmeans=False,
    showmedians=False,  
    showextrema=False,  
    bw_method=0.2,      # lower = more detail, higher = smoother
)

# Style violins
for body in vp["bodies"]:
    body.set_facecolor("#7f7f7f")
    body.set_edgecolor("#555555")
    body.set_alpha(0.35)
    body.set_linewidth(1.0)

# Overlay pooled median + max on each violin
for i, tc in enumerate(criteria_order):
    x = violin_positions[i]
    vals = pooled_vals_by_tc[tc]
    med = np.nanmedian(vals)
    mx = np.nanmax(vals)

    # median marker
    ax.scatter(
        x, med, marker="o", s=42,
        color="black", edgecolors="white", linewidths=0.5, zorder=7
    )
    # max marker
    ax.scatter(
        x, mx, marker="^", s=52,
        color="black", edgecolors="white", linewidths=0.5, zorder=7
    )

# Red List points (right mini-column)
QUANT_MARKERS = {
    0:    ("v", "Red List (gen-length) min/max"),
    0.25: ("o", "Red List (gen-length) 25th-75th%"),
    0.75: ("o", None),
    1:    ("v", None),
}

for tc in criteria_order:
    crit = redlist_criteria[tc]
    col = COLOURS[tc]
    xr = thresholds[tc] * (2 ** REDLIST_SHIFT)

    # Red List threshold diamond
    ax.scatter(
        xr, crit["p_ext"],
        marker="D", s=130, color=col,
        edgecolors="black", linewidths=2, zorder=6
    )

    # Red List mammal quantile markers
    for q in quants:
        p_val = cr_quants[tc][q]
        marker, _ = QUANT_MARKERS[q]
        ax.scatter(
            xr, p_val,
            marker=marker, s=80, color=col,
            edgecolors="white", linewidths=0.6, zorder=5
        )

# Axes/ticks
tick_vals = [thresholds[tc] for tc in criteria_order]
tick_labels = [f"{tc}\n({thresholds[tc]:,})" for tc in criteria_order]

ax.set_xscale("log", base=2)
ax.set_xlim(min(tick_vals) / 2.0, max(tick_vals) * 2.0)
ax.set_xticks(tick_vals)
ax.set_xticklabels(tick_labels)

ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("Red List criterion (mature individuals threshold)", fontsize=11)
ax.set_ylabel("Extinction probability P(E)", fontsize=11)

# ------------------------------------------------------------------
# Legend
# ------------------------------------------------------------------
dist_legend = [
    Patch(facecolor="#7f7f7f", edgecolor="#555555", alpha=0.35,
          label="Model fits (A-D) violin density"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="black",
           markeredgecolor="white", markeredgewidth=0.5, markersize=7,
           label="Models A-D median"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="black",
           markeredgecolor="white", markeredgewidth=0.5, markersize=7,
           label="Models A-D max"),
]

colour_legend = [
    Patch(facecolor=COLOURS[tc], alpha=0.85, label=redlist_criteria[tc]["name"])
    for tc in criteria_order
]

redlist_legend = [
    Line2D([0], [0], marker="D", color="w", markerfacecolor="grey",
           markeredgecolor="black", markeredgewidth=0.8, markersize=7,
           label="Red List (x-years threshold)")
] + [
    Line2D([0], [0], marker=marker, color="w", markerfacecolor="grey",
           markeredgecolor="white", markeredgewidth=0.6, markersize=9,
           label=label)
    for q, (marker, label) in QUANT_MARKERS.items()
    if label is not None
]

all_handles = dist_legend + [Patch(alpha=0)] + colour_legend + redlist_legend

ax.legend(
    handles=all_handles,
    labels=[h.get_label() for h in all_handles],
    fontsize=9,
    framealpha=0.9,
    loc="upper right",
)

fig.tight_layout()
figs_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(figs_dir / "redlist_comparison_violin_vs_redlist.png", dpi=300, bbox_inches="tight")
plt.show()