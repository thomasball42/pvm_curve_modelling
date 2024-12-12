OVERVIEW

Full simulation and analysis code used to generate and analyse the data described in the accompanying manuscript "A general relationship between population size and carrying capacity".

SYSTEM REQUIREMENTS

This code was developed and run on Windows 11 and Linux 6.8.0-45. All code was written using Python 3.12. In addition to the Python Standard Library (https://docs.python.org/3/library/index.html) the code is relient on the following packages, which can be installed using pip:

numpy
scipy
matplotlib
pandas
tqdm

INSTRUCTIONS

Simulations can be run using 'run_scenarios.py', with simulation parameters being found at the top of that file. The variable 'RESULTS_PATH' dictates the directory in which the simulation results will be saved.

'MULTIPROCESSING_ENABLED' allows simultaneous simulations to be carried out on machines with plentiful resources, 'NUM_WORKERS' governs the number of CPU cores utilised.

Setting 'OVERWRITE_EXISTING_FILES' to True will cause any existing results to be overwritten, otherwise, simulations for which results already exist will be skipped.

Running the simulations using the unmodified files will produce the quantitative results described in the manuscript.


