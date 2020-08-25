import numpy as np


def model_output(model, time_t, state_monitor, input_all):
    airspeed = input_all[0]
    altitude = input_all[11]
    roll = input_all[3]
    finish_gcas = str(input_all[-1])
    gcas_primitive = (altitude > 3400 and altitude < 4900 and np.abs(roll) < 30.0)
    if (gcas_primitive) or (finish_gcas not in ['Waiting', 'Finished']):
        select =  ["gcas"]
    elif airspeed < 750.0: # Fixed Altitude
        select = ["altitude"]
    else:
        select = ["airspeed"] # Fixed Airspeed
    return select
