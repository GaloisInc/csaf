import os
import toml
import numpy as np

parameters = {}

def main(time=0.0, state=None, input=[0]*4, update=False, output=False):
    global parameters

    if len(parameters.keys()) == 0:
        this_path = os.path.dirname(os.path.realpath(__file__))
        info_file = os.path.join(this_path, "autopilot.toml")
        with open(info_file, 'r') as ifp:
            info = toml.load(ifp)
        parameters = info["parameters"]

    airspeed = input[0]
    altitude = input[11]
    roll = input[3]
    pitch = input[4]
    finish_gcas = str(input[-1])
    if output:
        gcas_primitive = (altitude > 3400 and altitude < 4900 and np.abs(roll) < 30.0)
        if (gcas_primitive) or (finish_gcas not in ['Waiting', 'Finished']):
            select =  ["gcas"]
        elif airspeed < 750.0: # Fixed Altitude
            select = ["altitude"]
        else:
            select = ["airspeed"] # Fixed Airspeed
        return select
