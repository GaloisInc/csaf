"""
Dubins Rejoin Centralized Controller

controller.py

Taken from

> Umberto Ravaioli, James Cunningham, John McCarroll, Vardaan Gangal, Kerianne Hobbs,
> "Safe Reinforcement Learning Benchmark Environments for Aerospace Control Systems,"
> IEEE Aerospace, Big Sky, MT, March 2022.
"""
import numpy as np


def get_angles_v(states):
    angs = np.zeros((len(states), len(states)))
    for idx, pidx in enumerate(states):
        for jdx, pjdx in enumerate(states):
            diff = (pjdx[:2] - pidx[:2])
            angle = np.arctan2(diff[1], diff[0])
            angs[idx][jdx] = angle
    return angs


def get_dists_v(states):
    angs = np.zeros((len(states), len(states)))
    for idx, pidx in enumerate(states):
        for jdx, pjdx in enumerate(states):
            diff = (pjdx[:2] - pidx[:2])
            angle = np.sqrt(diff[1]**2 + diff[0]**2)
            angs[idx][jdx] = angle
    return angs


def model_output(model, time_t, states, inputs):
    states = [np.array(inputs[idx*3:(idx+1)*3]) for idx in range(4)]
    angles = get_angles_v(states)
    dists = get_dists_v(states)
    angles[dists < model.rl] = np.pi - angles[dists < model.rl]
    weight = np.exp(-(dists-model.rl)**2/model.tau)
    dangles = weight * model.ti + (1 - weight) * angles
    for idx, (drow, rrow) in enumerate(zip(dangles, dists)):
        rrow[np.abs(rrow) < 1E-8] = 1E12
        ridx = np.argmin(np.abs(rrow))
        mval = drow[ridx]
        drow[drow != mval] = 0.0
        dangles[idx] = drow
    dangles = np.sum(dangles, axis=1)
    cangles = np.array([s[-1] for s in states])
    return 1 * (dangles - cangles)

