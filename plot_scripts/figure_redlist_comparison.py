import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from other_scripts import process_mammal_data

redlist_criteria = {
                            'CR': {
                                    "name" : 'Critically Endangered',        
                                    "time" : 10,
                                    "num_gens" : 3, 
                                    "p_ext": 0.5,
                                    "mature_individs" : 50
                                    },
                            'EN': {
                                    "name" : 'Endangered', 
                                    "time" : 20,
                                    "num_gens" : 5,
                                    "p_ext": 0.2,
                                    "mature_individs" : 250
                                    },
                            'VU': {
                                    "name" : 'Vulnerable',
                                    "time" : 100,
                                    "num_gens" : 0, # no gen requirement for vulnerable
                                    "p_ext": 0.1,
                                    "mature_individs" : 1000
                                    },

                            'NT': {"name" : 'Near Threatened'},
                            'LC': {"name" : 'Least Concern'},     

                        }

mammal_df = process_mammal_data.return_extended_mammal_data(redlist_criteria=redlist_criteria,
                                                            save_file=True)
fig, ax = plt.subplots()

for tc in ['CR', 'EN', 'VU']:
    
    mature_indidivuals = redlist_criteria[tc]['mature_individs']

    ext_risk_100yr_eq = process_mammal_data.est_ext_risk_100(redlist_criteria[tc]['p_ext'], redlist_criteria[tc]['time'])

    mammal_ext_risks_100 = mammal_df[f'{tc}_ext_risk_eq100']

    ax.scatter(mature_indidivuals, ext_risk_100yr_eq, label=redlist_criteria[tc]['name'])
    ax.boxplot(mammal_ext_risks_100, positions=[mature_indidivuals], widths=100, )

ax.set_xlim(left=-50)
ax.set_xlabel('Number of mature individuals')
ax.set_ylabel('Extinction risk 100-year equivalent')

plt.show()

