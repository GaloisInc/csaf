import os
import toml

import numpy as np

parameters = {}


def _ss_inv_pend_lqr():
    """
    Inverted Pendulum System -- LQR Gain Matrix
    Taken from
    http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlDigital#4
    Run at a sampling period Ts = 1/100 s
    :return: K gain matrix
    """
    return np.array([[-1.0000],   [-1.6567],   [18.6854],    [3.4594]]).T


def main(time=0.0, state=None, input=None, update=False, output=False):
    global parameters
    if len(parameters.keys()) == 0:
        this_path = os.path.dirname(os.path.realpath(__file__))
        info_file = os.path.join(this_path, "ipcontroller.toml")
        with open(info_file, 'r') as ifp:
            info = toml.load(ifp)
        parameters = info["parameters"]

    if output:
        return list(input[-1] - (_ss_inv_pend_lqr() @ np.array(input)[:-1, np.newaxis]).flatten())
    else:
        return []
