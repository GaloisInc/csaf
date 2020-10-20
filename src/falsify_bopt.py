""" Code for Falsify Neural Network Controlled F16 Demo

Modified code from https://github.com/caffett/falsify_neural_f16gcas
"""

from skopt import gp_minimize
import numpy as np
from numpy import pi, deg2rad


class F16SimulationFunction:
    def __init__(self, initial_time, simulator):
        self.core = lambda x0: simulator(x0, initial_time)
        self.initial_time = initial_time
        self.last_ps = 0

    def reward_func(self, states):
        altitude_min = 0  # ft AGL
        altitude_max = 4000  # ft AGL 45000
        nz_max = 15  # G's original is 9
        nz_min = -3  # G's original is -2
        # ps_max_accel_deg = 500  # /s/s

        v_min = 300  # ft/s
        v_max = 2500  # ft/s
        alpha_min_deg = -10  # deg
        alpha_max_deg = 45  # deg
        beta_max_deg = 30  # deg

        # did not consider the change rate of ps here
        constraints_dim = [0, 1, 2, 11, 13]
        constraints_box = np.array([[v_min, alpha_min_deg, -beta_max_deg, altitude_min, nz_min]
                                       , [v_max, alpha_max_deg, beta_max_deg, altitude_max, nz_max]])

        states = np.array(states)
        dist_to_lb = np.abs(states[:, constraints_dim] - constraints_box[0])
        dist_to_ub = np.abs(states[:, constraints_dim] - constraints_box[1])

        min_dist = np.min(np.array([dist_to_ub, dist_to_lb]), axis=0)
        norm_min_dist = min_dist / (constraints_box[1] - constraints_box[0])

        return np.mean(norm_min_dist)

    def __call__(self, initial_state):
        ret = self.core(initial_state)
        _, states, passed = ret[:3]

        if not passed:
            return False, -1.0
        else:
            return True, self.reward_func(states)

    def __repr__(self):
        return f"t={self.initial_time}"


class EvaluatedFunction:
    def __init__(self, simu_fn: F16SimulationFunction):
        """
        Initialization
        :param simu_fn: wrapped simulation function,
                takes initial state as input,
                return passed indicator and reward
        """
        self.simu_fn = simu_fn
        self._collector = []

    def __call__(self, initial_state):
        ret = self.simu_fn(initial_state)
        passed, reward = ret[0:2]
        if not passed:
            self._collector.append((self.simu_fn.initial_time, initial_state))
        return reward

    def __repr__(self):
        return f"EvaluatedFunction<<{repr(self.simu_fn)}>>"

    def get_collector(self):
        return self._collector


def attack(func: EvaluatedFunction,
           space,
           acq_func="EI",
           n_calls=5,
           n_random_starts=2):
    """
    Attack with BO
    :param func: evaluated function with a collector callback
    :param space: lower and upper boundary
    :param acq_func: BO acquisition function
    :param n_calls: the number of evaluations of f
    :param n_random_starts: the number of random initialization points
    :return: unsafe initial states
    """
    dimensions = np.array(space).T
    dimensions[:, 0] -= 1e-5

    gp_minimize(func, dimensions,
                acq_func=acq_func,
                n_calls=n_calls,
                n_random_starts=n_random_starts,
                noise=0,
                n_jobs=1)

    return func.get_collector()


def initial_space():
    """
    Self-defined initial space
    :return: lower and upper boundary of initial space box
    """
    # Initial Conditions
    power = 9  # Power

    # Default alpha & beta
    alpha = deg2rad(2.1215)  # Trim Angle of Attack (rad)
    beta = 0  # Side slip angle (rad)

    # Initial Attitude
    alt = 3600
    Vt = 540  # Pass at Vtg = 540;    Fail at Vtg = 550;
    phi = (pi / 2) * 0.5  # Roll angle from wings level (rad)
    theta = (-pi / 2) * 0.8  # Pitch angle from nose level (rad)
    psi = -pi / 4  # Yaw angle from North (rad)

    # Build Initial Condition Vectors
    # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow]
    center = np.array([Vt, alpha, beta, phi, theta, psi, 0, 0, 0, 0, 0, alt, power, 0, 0, 0])
    state_delta = np.abs(center) * 0.02
    initial_state_low = center - state_delta
    initial_state_high = center + state_delta

    return initial_state_low, initial_state_high
