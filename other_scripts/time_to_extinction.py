import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import _analysis_utils


### time to extinction vs K
data_path = Path("..\\results\\simulation_results\\results_mean_tte")

list_of_files = []
for path, subdirs, files in os.walk(data_path):
    for name in files:
        list_of_files.append(os.path.join(path, name)) 

files = [file for file in list_of_files if ".csv" in file and "LogGrowth" in file]

filter = ["YT100"]

fig, ax = plt.subplots()

for file in files:

    dat = pd.read_csv(file)
    run_params = _analysis_utils.extract_run_parameters(dat)

    K = dat.K
    mean_tte = dat.mean_TTE
    mean_tte_sem = dat.mean_TTE_SEM
    Peq1 = dat.P == 1 # persistence = 1
    
    ext_component = ((dat.P[~Peq1]) * dat.mean_TTE[~Peq1])
    sur_component = dat.P[Peq1] * dat.YEAR_THRESHOLD[Peq1]


    expected_tte = np.array(ext_component.to_list() + sur_component.to_list())

    #using rule of 3 (i.e. no observed events)
    expected_tte_sem_sur = np.full_like(dat.YEAR_THRESHOLD[Peq1], 100 * ( 1 - (3/dat[Peq1].N) ))
    expected_tte_sem_ext = dat.P[~Peq1] * dat.mean_TTE_SEM[~Peq1] 

    x = K
    y = expected_tte
    e = np.array(expected_tte_sem_ext.to_list() + expected_tte_sem_sur.tolist())

    ax.errorbar(x, y, yerr=e, fmt="", label=run_params['runName'],
                 linewidth = 0, elinewidth=3)

# ax.set_xlim(-1, dat[dat.P < 1].K.max()+1)
ax.set_xlabel("K")
ax.set_ylabel("Expected Time to Extinction")
ax.legend()
plt.show()