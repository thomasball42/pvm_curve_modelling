import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def est_ext_risk_100(known_ext_risk, known_ext_risk_time):
        """Assume constant annual extinction risk"""
        ext_risk_100 = 1 - (1 - known_ext_risk) ** (100 / known_ext_risk_time)    
        return ext_risk_100

def est_exp_ext_time(known_ext_risk, known_ext_risk_time):
        """Assume discrete time constant annual extinction risk"""
        exp_ext_time = 1 / (1 - (1 - known_ext_risk) ** (1 / known_ext_risk_time))
        return exp_ext_time

def return_extended_mammal_data(redlist_criteria=None,
                                save_file=False):
    
    assert redlist_criteria is not None, "Please provide redlist criteria to calculate extinction risk thresholds"
    
    mammals_file = Path("manuscript_inputs", "pacifici2013_generation_length_mammals.csv")
    mammals_df = pd.read_csv(mammals_file)

    def gen_sp_thresholds(sp_gen_length_days, num_gens_years, time_years, p_ext):
        sp_genXnum_gens = (sp_gen_length_days / 365) * num_gens_years 
        
        # choose the longer of X gen lengths or time thresh:
        threshhold_time = max(sp_genXnum_gens, time_years)

        # 100 year ext risk equivalent:
        p_ext_100 = est_ext_risk_100(p_ext, threshhold_time)
        exp_tte = est_exp_ext_time(p_ext, threshhold_time)

        return p_ext_100, exp_tte

    for t in ['CR', 'EN', 'VU']:

        mammals_df[[f'{t}_ext_risk_eq100', f'{t}_exp_tte']] = mammals_df.apply(
            lambda x: gen_sp_thresholds(x['GenerationLength_d'], 
                                        redlist_criteria[t]['num_gens'], 
                                        redlist_criteria[t]['time'], 
                                        redlist_criteria[t]['p_ext']), 
                                        axis=1, result_type='expand')

    if save_file:
        mammals_df.to_csv(Path("manuscript_inputs", "pacifici2013_generation_length_mammals_EXTENDED.csv"), index=False)
    
    return mammals_df

if __name__ == "__main__":
    redlist_criteria = {
                                'CR': {
                                        "name" : 'Critically Endangered',        
                                        "time" : 10,
                                        "num_gens" : 3, 
                                        "p_ext": 0.5
                                        },
                                'EN': {
                                        "name" : 'Endangered', 
                                        "time" : 20,
                                        "num_gens" : 5,
                                        "p_ext": 0.2
                                        },
                                'VU': {
                                        "name" : 'Vulnerable',
                                        "time" : 100,
                                        "num_gens" : 0, # no gen requirement for vulnerable
                                        "p_ext": 0.1
                                        },

                                'NT': {"name" : 'Near Threatened'},
                                'LC': {"name" : 'Least Concern'},     

                            }
    mammals_df = return_extended_mammal_data(redlist_criteria=redlist_criteria, save_file=True)    
   