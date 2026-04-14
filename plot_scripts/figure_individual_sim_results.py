import matplotlib
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import _analysis_utils
import _curve_fit

data_path = Path("..\\results\\individual_simulation_results\\individual_simulation_examples")
figs_dir = Path("..", "figs", "individual_simulation_examples")
data_fits_path = Path("..", "results", "data_fits", "data_fits_individual_sims")

list_of_files = []
for path, subdirs, files in os.walk(data_path):
    for name in files:
        list_of_files.append(os.path.join(path, name)) 

data_fits_path = Path(data_fits_path) / f"data_fits_individual_sims.csv"

fig, ax = plt.subplots(figsize=(6, 5))

for f, file in enumerate(list_of_files):

    mort = [_ for _ in file.split("_") if "mort" in _][0].split("mort")[1]

    dat = pd.read_csv(file)

    x = dat.K
    y = dat.P

    max_y = np.max(y)

    color = (f+1) / len(list_of_files)

    fit_result = _analysis_utils.fit_gompertz_curve(x, y, 
                                                alpha_space=np.arange(0, 1, 0.05), 
                                                ylim=(0.05, 0.95),
                                                iteration_depth= 4,
                                                )
    
    fit_params = fit_result["params"]
    R2 = fit_result["R2"]

    _analysis_utils.plot_curve_fit( ax, x, y, _curve_fit.mod_gompertz, 
                                    fit_params, R2, 
                                    "Individual-based simulation",
                                    label = f"Mortality: {mort} (R$^2$: {_analysis_utils.format_R2_str(R2)})",
                                    color = color,colormap = "viridis",
                                    skip_nans = True
                                    )

    yerr = dat.P_SEM
    ax.errorbar(x, 1-y, yerr=yerr, fmt="o",
                markersize=0,
                color=plt.cm.viridis(color))

ax.legend(loc="upper right")
ax.set_xlabel("Resource availability (~K)")
ax.set_ylabel("P(extinction) (N = 1000)")

fig.tight_layout()

figs_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(figs_dir / f"individual_simulation_example.png", dpi=300)

plt.show()