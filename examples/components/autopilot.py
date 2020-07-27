import os
import toml
import json

import numpy as np
from csaf import message

def main(time=0.0, state=None, input=[0]*4, update=False, output=False):

    msg_writer = message.Message()
    epoch = 0
    fs = 100

    this_path = os.path.dirname(os.path.realpath(__file__))
    info_file = os.path.join(this_path, "f16plant.toml")
    with open(info_file, 'r') as ifp:
        info = toml.load(ifp)

    parameters = info["parameters"]

    if output:
        return [0.0] * 4
    else:
        return




if __name__ == '__main__':
    import fire
    fire.Fire(main)