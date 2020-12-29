import numpy as np


def model_output(model, time_t, state_air, input_forces):
    return []


def model_state_update(model, time_t, state_air, input_rate):
    input_rate = input_rate[model.idx]
    xdot = np.zeros((3,))
    xdot[0] = model.v * np.cos(state_air[-1])
    xdot[1] = model.v * np.sin(state_air[-1])
    xdot[2] = input_rate
    return xdot
