import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
COLOURS = {
    'CR': "#17DCE3",
    'EN': "#491EA7",
    'VU': "#D2E317",
}

MODEL_COLOURS = {
    "model_A": "#E07B39",
    "model_B": "#3A86FF",
    "model_C": "#FF006E",
    "model_D": "#8338EC",
}

redlist_criteria = {
    'CR': {
        "name": 'Critically Endangered',
        "time": 10,
        "num_gens": 3,
        "p_ext": 0.5,
        "mature_individs": 50
    },
    'EN': {
        "name": 'Endangered',
        "time": 20,
        "num_gens": 5,
        "p_ext": 0.2,
        "mature_individs": 250
    },
    'VU': {
        "name": 'Vulnerable',
        "time": 100,
        "num_gens": 0,
        "p_ext": 0.1,
        "mature_individs": 1000
    },
}

mammal_df = process_mammal_data.return_extended_mammal_data(
    redlist_criteria=redlist_criteria, save_file=True
)

all_data_fits = {}
for model_name in model_names:

    path = Path("..", "results", "data_fits", "data_fits_main",
                f"data_fits_{model_name}.csv")
    df = pd.read_csv(path, index_col=0).dropna(
        subset=["param_a", "param_b", "param_alpha"]
    )
    df = df[df.MAX_Y > 0.999]
    all_data_fits[model_name] = df.copy()

cr_quants = {}
for tc in redlist_criteria:
    mammal_ext_risks_100 = mammal_df[f'{tc}_ext_risk_eq100']
    cr_quants[tc] = mammal_ext_risks_100.quantile(quants).to_dict()

N_range = np.logspace(0, 4, 400)

QUANT_MARKERS = {
    0:    ("v", "Red List (gen-length) min/max"),
    0.25: ("o", "Red List (gen-length) 25th-75th%"),
    0.75: ("o", None),
    1:    ("v", None),
}

curve_median = {m: {} for m in model_names}

curve_lo_combined  = {}
curve_hi_combined  = {}
curve_q25_combined = {}
curve_q75_combined = {}

curve_median = {}
all_curves_combined = []

for model_name in model_names:
    data_fits = all_data_fits[model_name]
    model_curves = np.array([
        [1 - _curve_fit.mod_gompertz(n, row["param_a"], row["param_b"], row["param_alpha"])
         for n in N_range]
        for _, row in data_fits.iterrows()
    ])
    curve_median[model_name] = np.nanmedian(model_curves, axis=0)
    all_curves_combined.append(model_curves)

all_curves_combined = np.vstack(all_curves_combined)
curve_lo_combined  = np.nanpercentile(all_curves_combined, 10, axis=0)
curve_hi_combined  = np.nanpercentile(all_curves_combined, 90, axis=0)
curve_q25_combined = np.nanpercentile(all_curves_combined, 25, axis=0)
curve_q75_combined = np.nanpercentile(all_curves_combined, 75, axis=0)


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.fill_between(N_range, curve_lo_combined, curve_hi_combined, color="grey", alpha=0.15, linewidth=0)
ax.fill_between(N_range, curve_q25_combined, curve_q75_combined, color="grey", alpha=0.25, linewidth=0)

for model_name in model_names:
    ax.plot(N_range, curve_median[model_name],
            color=MODEL_COLOURS[model_name], linewidth=1.8, linestyle="--", alpha=0.95)

for tc, crit in redlist_criteria.items():
    col = COLOURS[tc]
    n_thresh = crit["mature_individs"]

    # red List threshold diamond
    ax.scatter(
        n_thresh, crit["p_ext"],
        marker="D", s=130, color=col,
        edgecolors="black", linewidths=2,
        zorder=6,
    )

    # red List mammal quantile markers
    for q in quants:
        p_val = cr_quants[tc][q]
        marker, _ = QUANT_MARKERS[q]
        ax.scatter(
            n_thresh, p_val,
            color=col, edgecolors="white", linewidths=0.6,
            marker=marker, s=80, zorder=5,
        )

ax.set_xscale("log", base=2)
ax.set_xlim(1, 1.2e4)
ax.set_ylim(-0.02, 1.02)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{int(x):,}"
))
ax.set_xlabel("Mature individuals (N)", fontsize=11)
ax.set_ylabel("Extinction probability P(E)", fontsize=11)

# ------------------------------------------------------------------
# legend (s..)
# ------------------------------------------------------------------
model_legend = [
    Line2D([0], [0], color=MODEL_COLOURS[m], linewidth=1.8, linestyle="--",
           label=f"{m.replace('_', '').replace('model', 'Model')} median")
    for m in model_names
]

band_legend = [
    Patch(facecolor="grey", alpha=0.25, label="All models 25–75th%"),
    Patch(facecolor="grey", alpha=0.15, label="All models 10–90th%"),
]

colour_legend = [
    Patch(facecolor=COLOURS[tc], alpha=0.85, label=crit["name"])
    for tc, crit in redlist_criteria.items()
]
quant_legend = [
    Line2D([0], [0], marker="D", color="w", markerfacecolor="grey",
           markeredgecolor="black", markeredgewidth=0.8, markersize=7,
           label="Red List (x-Years)"),
] + [
    Line2D([0], [0], marker=marker, color="w", markerfacecolor="grey",
           markeredgecolor="white", markeredgewidth=0.6, markersize=9,
           label=label)
    for q, (marker, label) in QUANT_MARKERS.items()
    if label is not None
]

all_handles = model_legend \
            + band_legend \
            + [Patch(alpha=0)] \
            + colour_legend \
            + quant_legend

ax.legend(
    handles=all_handles,
    labels=[h.get_label() for h in all_handles],
    fontsize=9, framealpha=0.9, loc="upper right",
)

fig.tight_layout()
figs_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(figs_dir / "redlist_comparison.png", dpi=300, bbox_inches="tight")
plt.show()