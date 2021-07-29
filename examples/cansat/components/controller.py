"""
CanSat Constellation Centralized Controller

controller.py

Taken from

> Umberto Ravaioli, James Cunningham, John McCarroll, Vardaan Gangal, Kerianne Hobbs,
> "Safe Reinforcement Learning Benchmark Environments for Aerospace Control Systems,"
> IEEE Aerospace, Big Sky, MT, March 2022.
"""

import numpy as np
from scipy.spatial.qhull import Delaunay


def graph_from_simplices(tri: Delaunay) -> dict:
    """
    transform simplices to graph represented as
    {
        vertex_id : set({verts})
    }
    """
    graph = {}
    for simplex in tri.simplices:
        for va, vb in list(zip(simplex[:-1],
                               simplex[1:])) + [(simplex[0], simplex[-1])]:
            if va in graph:
                graph[va].add(vb)
            else:
                graph[va] = set({vb})
            if vb in graph:
                graph[vb].add(va)
            else:
                graph[vb] = set({va})

    return graph


def model_output(model, time_t, state_ctrl, input_forces):
    """
    calculate forces to be supplied by the satellites to rejoin
    """
    forces = []
    points = [np.array([0.0, 0.0])
              ] + [np.array(input_forces[i * 4:(i + 1) * 4][:2]) for i in range(0, 4)]
    vels = [np.array([0.0, 0.0])
            ] + [np.array(input_forces[i * 4:(i + 1) * 4][2:]) for i in range(0, 4)]
    tri = Delaunay(points)
    graph = graph_from_simplices(tri)
    for sidx in range(len(points)):
        connections = graph[sidx]
        f = np.array([0.0, 0.0])
        for sother in connections:
            # get unit distance vector and norm
            dist = points[sother] - points[sidx]
            r = np.linalg.norm(dist)
            dist /= r

            vel = (vels[sother] - vels[sidx])
            velp = np.dot(vel, dist) * dist
            velr = np.linalg.norm(vel)

            if not np.isnan(r):
                f += model.kp * (r - model.rest_length) * dist
            if not np.isnan(velr):
                f += model.kd * velp

        forces.append(f)
    return tuple(np.concatenate(forces)[2:])
