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

    mapper = ["gcas", "altitude", "airspeed"]
    controller = input[-1]
    if controller == 0.0:
        sidx = int(0.0)
    else:
        sidx = mapper.index(controller)
    assert len(input) == 13
    if output:
        return input[4*sidx:4*sidx+4]

