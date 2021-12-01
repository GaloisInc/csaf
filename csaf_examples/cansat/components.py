"""
CanSat Constellation Plant Model

cansat/components.py

Taken from

> Umberto Ravaioli, James Cunningham, John McCarroll, Vardaan Gangal, Kerianne Hobbs,
> "Safe Reinforcement Learning Benchmark Environments for Aerospace Control Systems,"
> IEEE Aerospace, Big Sky, MT, March 2022.
"""
from csaf import ContinuousComponent, DiscreteComponent, System
import typing
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


def model_output(model, time_t, state_sat, input_forces):
    """can sat outputs - empty list
    TODO: maybe remove this?
    """
    return []


def model_state_update(model, time_t, state_sat, input_forces):
    """can sat dynamics model"""
    input_forces = np.array(input_forces)[: 2]
    xdot = np.zeros((4, ))
    xdot[0] = state_sat[2]
    xdot[1] = state_sat[3]
    xdot[2] = 3 * model.n**2 * state_sat[0] + \
              2 * model.n * state_sat[3] + 1 / model.mc * input_forces[0]
    xdot[3] = -2 * model.n * state_sat[2] + 1 / model.mc * input_forces[1]
    return xdot


class CanSatInputMessage(typing.NamedTuple):
    """can inputs - controller can apply 2D force vector"""
    xforce: float
    yforce: float


class CanSatOutputMessage(typing.NamedTuple):
    """can outputs - no outputs (states only)"""
    pass


class CanSatStateMessage(typing.NamedTuple):
    """can state - satellite is a 2D point mass (4 states)"""
    x: float
    y: float
    xdot: float
    ydot: float


class EmptyMessage(typing.NamedTuple):
    pass


class CanSatComponent(ContinuousComponent):
    name = "Can Satellite Component"
    sampling_frequency = 30.0
    default_parameters = {
        "n" : 0.001027,
        "mc" : 12
    }
    inputs = (("inputs", CanSatInputMessage),)
    outputs = (("outputs", CanSatOutputMessage),)
    states = CanSatStateMessage
    default_initial_values = {
        "states": [0.0, 0.0, 0.0, 0.0],
        "inputs": [0.0, 0.0]
    }
    flows = {
        "outputs": model_output,
        "states": model_state_update
    }


def generate_cansat_controller(nagents: int) -> typing.Type[DiscreteComponent]:
    """component generator for a satellite controller of nagents"""

    def controller_output(idx):
        def _c_input(model, time_t, state_ctrl, input_forces):
            """calculate forces to be supplied by the satellites to rejoin"""
            forces = []
            points = [np.array([0.0, 0.0])
                      ] + [np.array(input_forces[i * 4:(i + 1) * 4][:2]) for i in range(0, nagents)]
            vels = [np.array([0.0, 0.0])
                    ] + [np.array(input_forces[i * 4:(i + 1) * 4][2:]) for i in range(0, nagents)]
            tri = Delaunay(points)
            graph = graph_from_simplices(tri)
            for sidx in range(len(points)):
                connections = graph[sidx]
                f = np.array([0.0, 0.0])
                for sother in connections:
                    # get unit distance vector and norm
                    dist = points[sother] - points[sidx]
                    r = np.linalg.norm(dist)

                    # weird true divide error
                    dist = dist / r

                    vel = (vels[sother] - vels[sidx])
                    velp = np.dot(vel, dist) * dist
                    velr = np.linalg.norm(vel)

                    if not np.isnan(r):
                        f += model.kp * (r - model.rest_length) * dist
                    if not np.isnan(velr):
                        f += model.kd * velp
                forces.append(f)
            return tuple(np.concatenate(forces)[2:])[idx*2:(idx+1)*2]
        return _c_input

    class _CanSatControllerComponent(DiscreteComponent):
        """controller for nagents satellites"""
        name = f"CanSat {nagents}-Agent Controller Component"
        sampling_frequency = 30.0
        default_parameters = {
            "kp": 2.0,
            "kd": 8.0,
            "rest_length" : 4.0
        }
        inputs = tuple([(f"inputs_sat{idx}", CanSatStateMessage) for idx in range(nagents)])
        outputs = tuple([(f"outputs_sat{idx}", CanSatInputMessage) for idx in range(nagents)])
        states = EmptyMessage
        default_initial_values = {
            "states": [],
            **{f"inputs_sat{idx}": [0.0,]*4 for idx in range(nagents)}
        }
        flows = {
            f"outputs_sat{idx}": controller_output(idx) for idx in range(nagents)
        }

    return _CanSatControllerComponent


def generate_cansat_system(start_states: np.ndarray):
    nagents = len(start_states)
    controller_type = generate_cansat_controller(nagents)

    class _CanSatSystem(System):
        components = {
            **{f"sat{idx}": CanSatComponent for idx in range(nagents)},
            "controller": controller_type
        }

        connections = {
            **{(f"sat{idx}", "inputs") : ("controller", f"outputs_sat{idx}")  for idx in range(nagents)},
            **{("controller", f"inputs_sat{idx}") : (f"sat{idx}", "states")  for idx in range(nagents)}
        }

    sys = _CanSatSystem()
    for idx, sstate in enumerate(start_states):
        sys.set_state(f"sat{idx}", sstate)

    return sys