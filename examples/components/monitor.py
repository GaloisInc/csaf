import os
import toml
import numpy as np

parameters = {}
finish_gcas = None

def main(time=0.0, state=None, input=[0]*4, update=False, output=False):
    global parameters
    global finish_gcas

    if len(parameters.keys()) == 0:
        this_path = os.path.dirname(os.path.realpath(__file__))
        info_file = os.path.join(this_path, "autopilot.toml")
        with open(info_file, 'r') as ifp:
            info = toml.load(ifp)
        parameters = info["parameters"]

    altitude = input[11]
    roll = input[6]
    if output:
        if finish_gcas is not None:
            if time > (finish_gcas +  10.0):
                finish_gcas = None
            else:
                return [0]

        if (altitude > 3400 and altitude < 4900 and np.abs(roll) < 30.0):
            select =  [0]
            finish_gcas = time
        elif altitude < 5000: # Fixed Altitude
            select = [1]
        else:
            select = [2] # Fixed Airspeed
        return select
