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

    sidx = int(input[-1])
    assert len(input) == 13
    #sidx = 0
    print(4*sidx, 4*sidx+4)
    if output:
        return input[4*sidx:4*sidx+4]

