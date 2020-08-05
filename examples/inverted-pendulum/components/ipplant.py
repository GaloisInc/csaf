import os
import toml

import numpy as np


parameters = {}

def _ss_inv_pend_cont():
    """
    Inverted Pendulum System -- Linear State Space Representation
    Taken from
    http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlDigital#4
    :return: A, B, C and D matrices
    """
    global parameters
    mm = parameters["mm"]        # Mass of the cart
    m = parameters["m"]         # Mass of the pendulum
    b = parameters["b"]        # coefficient of friction on cart
    ii = parameters["ii"]       # Mass Moment of Inertia on Pendulum
    g = parameters["g"]         # Gravitational acceleration
    length = parameters["length"]         # length to pendulum COM
    p = ii * (mm + m) + mm * m * length ** 2     # Denominator

    # Continuous time system
    a = np.array([[0, 1, 0, 0],
                  [0, -(ii+m*length**2)*b/p,  (m**2*g*length**2)/p,   0],
                  [0, 0, 0, 1],
                  [0, -(m*length*b)/p,       m*g*length*(mm+m)/p,  0]])

    b = np.array([[0],
                  [(ii+m*length**2)/p],
                  [0],
                  [m*length/p]])

    c = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])

    d = np.array([[0], [0]])
    return a, b, c, d


def main(time=0.0, state=None, input=None, update=False, output=False):
    global parameters
    if len(parameters.keys()) == 0:
        this_path = os.path.dirname(os.path.realpath(__file__))
        info_file = os.path.join(this_path, "ipplant.toml")
        with open(info_file, 'r') as ifp:
            info = toml.load(ifp)
        parameters = info["parameters"]

    if update:
        a, b, _, _ = _ss_inv_pend_cont()
        return list((a @ np.array(state)[:, np.newaxis] + b @ np.array(input)[:, np.newaxis]).flatten())
    else:
        return []


