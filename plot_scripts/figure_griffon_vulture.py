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
figs_dir = Path("..", "figs", "figs_atlantic_puffin")

list_of_files = []
for path, subdirs, files in os.walk(data_path):
    for name in files:
        list_of_files.append(os.path.join(path, name))

filter = ["theta1", "qrev1"]
exclude = [
            "qsd0.2"
            ]

list_of_files = [f for f in list_of_files if all(filt.lower() in f.lower() for filt in filter) and not any(excl.lower() in f.lower() for excl in exclude)]

print(f"{len(list_of_files)} files found with filter '{filter}' and excluding '{exclude}'.")

def extract_qsd_from_filename(filepath):
    match = re.search(r'QSD([\d.]+)', filepath, re.IGNORECASE)
    return float(match.group(1)) if match else None

qsd_groups = {}
for file in list_of_files:
    qsd = extract_qsd_from_filename(file)
    if qsd is not None:
        qsd_groups.setdefault(qsd, []).append(file)

fig, ax = plt.subplots()

unique_qsds = sorted(qsd_groups.keys())

for q, qsd_val in enumerate(unique_qsds):

    files_for_qsd = qsd_groups[qsd_val]
    all_runs = []

    for file in files_for_qsd:
        dat = pd.read_csv(file)

        # if np.max(dat.P) < 1:
        #     continue

        all_runs.append(dat.set_index("K")["P"])

    if len(all_runs) == 0:
        print(f"QSD={qsd_val}: no runs reached max P, skipping.")
        continue

    runs_df = pd.concat(all_runs, axis=1)
    x = runs_df.index.values
    max_y = 1 - runs_df.min(axis=1).values 
    min_y = 1 - runs_df.max(axis=1).values 
    mean_y = 1 - runs_df.mean(axis=1).values

    color = plt.cm.viridis(q / max(len(unique_qsds) - 1, 1))


    ax.fill_between(x, min_y, max_y, color=color, alpha=0.4, label=f"QSD={qsd_val}")
    ax.plot(x, mean_y, color=color, linewidth=1.5)

    _analysis_utils.ax_log2_scale(ax)
    
ax.set_xlabel("K")
ax.set_ylabel("P(E)")
ax.legend()
plt.tight_layout()
plt.show()