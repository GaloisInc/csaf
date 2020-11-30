import numpy as np


def model_output(model, time_t, state_sat, input_forces):
    return []


def model_state_update(model, time_t, state_sat, input_forces):
    xdot = np.zeros((4, ))
    xdot[0] = state_sat[2]
    xdot[1] = state_sat[3]
    xdot[2] = 3 * model.n**2 * model.state[0] + \
        2 * model.n * model.state[3] + 1 / model.mc * input_forces[0]
    xdot[3] = -2 * model.n * model.state[2] + 1 / model.mc * input_forces[1]
    return xdot
