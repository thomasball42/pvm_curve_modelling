import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import _analysis_utils
import _curve_fit

data_path = Path("..\\results\\simulation_results\\results_griffon_vulture")
figs_dir = Path("..", "figs", "figs_griffon_vulture")

list_of_files = []
for path, subdirs, files in os.walk(data_path):
    for name in files:
        list_of_files.append(os.path.join(path, name))

fig, axs = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

for __, upsil_filt in enumerate(["upsil0.0", "upsil0.1", "upsil0.2"]):

    ax = axs.flatten()[__]

    filter = ["theta35112", "qrev1", upsil_filt]
    exclude = [
                # "qsd0.2"
                ]

    sub_files = [f for f in list_of_files if all(filt.lower() in f.lower() for filt in filter) and not any(excl.lower() in f.lower() for excl in exclude)]

    print(f"{len(sub_files)} files found with filter '{filter}' and excluding '{exclude}'.")

    def extract_qsd_from_filename(filepath):
        match = re.search(r'QSD([\d.]+)', filepath, re.IGNORECASE)
        return float(match.group(1)) if match else None

    qsd_groups = {}
    for file in sub_files:
        qsd = extract_qsd_from_filename(file)
        if qsd is not None:
            qsd_groups.setdefault(qsd, []).append(file)

    unique_qsds = sorted(qsd_groups.keys())

    for q, qsd_val in enumerate(unique_qsds):

        files_for_qsd = qsd_groups[qsd_val]
        all_runs = []

        for file in files_for_qsd:
            dat = pd.read_csv(file)

            if np.max(dat.P) < 1:
                continue

            all_runs.append(dat.set_index("K")["P"])

        if len(all_runs) == 0:
            print(f"QSD={qsd_val}: no runs reached max P, skipping.")
            continue
        
        theta = dat.ALLEE_theta.iloc[0]
        upsil = dat.ALLEE_upsil.iloc[0]
        qrev = dat.QREV.iloc[0]

        runs_df = pd.concat(all_runs, axis=1)
        x = runs_df.index.values
        max_y = 1 - runs_df.min(axis=1).values 
        min_y = 1 - runs_df.max(axis=1).values 
        mean_y = 1 - runs_df.mean(axis=1).values
        sem_y = runs_df.sem(axis=1).values

        color = plt.cm.viridis(q / max(len(unique_qsds) - 1, 1))

        

        fit_result = _analysis_utils.fit_gompertz_curve(x, mean_y, 
                                                    alpha_space=np.arange(0, 1, 0.05), 
                                                    ylim=(0.05, 0.95),
                                                    iteration_depth= 4
                                                    )

        # label = f"$\\sigma$={qsd_val}, $\\theta={theta}$, $\\upsilon={upsil}$, $q_{{rev}}={qrev}$, R2={_analysis_utils.format_R2_str(fit_result['R2'])}"
        label = f"$\\sigma$={qsd_val}, $\\theta={theta}$, R2={_analysis_utils.format_R2_str(fit_result['R2'])}"

        ax.fill_between(x, min_y, max_y, color=color, alpha=0.4, label=label)
        ax.plot(x, mean_y, color=color, linewidth=1.5)

        _analysis_utils.ax_log2_scale(ax)
        
        ax.set_xlabel("K")
        ax.set_ylabel("P(E)")
        ax.legend()
    ax.set_title(f"$\\upsilon$={upsil}")
    
fig.tight_layout()
figs_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(figs_dir / "griffon_vulture_P_curves.png", dpi=300)
plt.show()