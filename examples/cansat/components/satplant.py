"""
CanSat Constellation Plant Model

satplant.py

Taken from

> Umberto Ravaioli, James Cunningham, John McCarroll, Vardaan Gangal, Kerianne Hobbs,
> "Safe Reinforcement Learning Benchmark Environments for Aerospace Control Systems,"
> IEEE Aerospace, Big Sky, MT, March 2022.
"""
import numpy as np


def model_output(model, time_t, state_sat, input_forces):
    return []


def model_state_update(model, time_t, state_sat, input_forces):
    input_forces = np.array(input_forces)[model.idx * 2:(model.idx + 1) * 2]
    xdot = np.zeros((4, ))
    xdot[0] = state_sat[2]
    xdot[1] = state_sat[3]
    xdot[2] = 3 * model.n**2 * state_sat[0] + \
        2 * model.n * state_sat[3] + 1 / model.mc * input_forces[0]
    xdot[3] = -2 * model.n * state_sat[2] + 1 / model.mc * input_forces[1]
    return xdot
