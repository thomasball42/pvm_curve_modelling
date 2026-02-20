import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import _analysis_utils

data_path = Path("..\\results\\simulation_results")

list_of_files = []
for path, subdirs, files in os.walk(data_path):
    for name in files:
        list_of_files.append(os.path.join(path, name)) 
