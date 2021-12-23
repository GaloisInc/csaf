"""
Dubins Rejoin Agent Plant

rejoin/components.py

Taken from

> Umberto Ravaioli, James Cunningham, John McCarroll, Vardaan Gangal, Kerianne Hobbs,
> "Safe Reinforcement Learning Benchmark Environments for Aerospace Control Systems,"
> IEEE Aerospace, Big Sky, MT, March 2022.
"""
from csaf import ContinuousComponent, DiscreteComponent, System
import typing
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


def model_output(model, time_t, state_air, input_forces):
    return []


def model_state_update(model, time_t, state_air, input_rate):
    input_rate = input_rate[0]
    xdot = np.zeros((3,))
    xdot[0] = model.v * np.cos(state_air[-1])
    xdot[1] = model.v * np.sin(state_air[-1])
    xdot[2] = input_rate
    return xdot


class DubinsInputMessage(typing.NamedTuple):
    """dubins input - can only affect angular rate"""
    dtheta: float


class DubinsOutputMessage(typing.NamedTuple):
    """dubins outputs - no outputs (states only)"""
    pass


class DubinsStateMessage(typing.NamedTuple):
    """dubins state - vehicle is a 2D point and heading angle"""
    x: float
    y: float
    theta: float


class EmptyMessage(typing.NamedTuple):
    pass


class DubinsComponent(ContinuousComponent):
    name = "Dubins Vehicle Component"
    sampling_frequency = 30.0
    default_parameters = {
        "v" : 10.0
    }
    inputs = (("inputs", DubinsInputMessage),)
    outputs = (("outputs", DubinsOutputMessage),)
    states = DubinsStateMessage
    default_initial_values = {
        "states": [0.0, 0.0, 0.0],
        "inputs": [0.0,]
    }
    flows = {
        "outputs": model_output,
        "states": model_state_update
    }


def generate_dubins_controller(nagents: int) -> typing.Type[DiscreteComponent]:
    """component generator for a dubins controller of nagents"""

    def controller_output(ai):
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
            return [(1 * (dangles - cangles))[ai]]
        return model_output

    class _DubinsControllerComponent(DiscreteComponent):
        """controller for nagents vehicles"""
        name = f"Dubins {nagents}-Agent Controller Component"
        sampling_frequency = 30.0
        default_parameters = {
            "rl": 4.0,
            "ti": 0.0,
            "tau" : 10.0
        }
        inputs = tuple([(f"inputs_dub{idx}", DubinsStateMessage) for idx in range(nagents)])
        outputs = tuple([(f"outputs_dub{idx}", DubinsInputMessage) for idx in range(nagents)])
        states = EmptyMessage
        default_initial_values = {
            "states": [],
            **{f"inputs_dub{idx}": [0.0,]*4 for idx in range(nagents)}
        }
        flows = {
            f"outputs_dub{idx}": controller_output(idx) for idx in range(nagents)
        }

    return _DubinsControllerComponent


def generate_dubins_system(start_states: np.ndarray):
    nagents = len(start_states)
    controller_type = generate_dubins_controller(nagents)

    class _DubinsSystem(System):
        components = {
            **{f"dub{idx}": DubinsComponent for idx in range(nagents)},
            "controller": controller_type
        }

        connections = {
            **{(f"dub{idx}", "inputs") : ("controller", f"outputs_dub{idx}")  for idx in range(nagents)},
            **{("controller", f"inputs_dub{idx}") : (f"dub{idx}", "states")  for idx in range(nagents)}
        }

    sys = _DubinsSystem()
    for idx, sstate in enumerate(start_states):
        sys.set_state(f"dub{idx}", sstate)

    return sys
