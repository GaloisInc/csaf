from math import pi
import numpy as np
import gym

import helpers.llc_helper as lh
from helpers import fops, lqr

from f16llc import model_state_update
from autopilot_helper import FlightLimits

from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines import DDPG
from stable_baselines.common.vec_env import DummyVecEnv


def model_init(model):
    """load trained model"""
    path = fops.prepend_curr_path(('../', 'ddpg_128_128'))
    register_env()
    env = gym.make('F16GCAS-v0')
    env = DummyVecEnv([lambda: env])

    ddpg_model = DDPG.load(path, policy=CustomPolicy, env=env)

    _, xequil, uequil = getattr(lqr, model.lqr_name)()

    def ctrl_fn(x):
        action, states = ddpg_model.predict(x)
        return action

    model.parameters['llc'] = lh.FeedbackController(lh.CtrlLimits(), model, ctrl_fn, xequil, uequil)


def model_output(model, t, state_controller, input_f16):
    """ neural network low level controller output """
    return model.parameters['llc'].output(t, np.array(state_controller), np.array(input_f16))


def register_env():
    gym.envs.register(
        id='F16GCAS-v0',
        entry_point=F16GCAS,
        max_episode_steps=5000,
    )


# Custom MLP policy of two layers of size 16 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, layers=[128, 128], layer_norm=False, feature_extraction="mlp")


class F16GCAS(gym.Env):
    def __init__(self):
        # Initial condition
        self.power_low = 9
        self.power_high = 9
        # Default alpha & beta
        self.alpha_low = np.deg2rad(2.1215)
        self.alpha_high = np.deg2rad(2.1215)
        self.beta_low = 0
        self.beta_high = 0
        # Initial Attitude
        self.alt_low = 3600
        self.alt_high = 3600
        self.Vt_low  = 540
        self.Vt_high = 540                   # Pass at Vtg = 540;    Fail at Vtg = 550;
        self.phi_low  = (pi/2)*0.5
        self.phi_high = (pi/2)*0.5           # Roll angle from wings level (rad)
        self.theta_low  = (-pi/2)*0.8
        self.theta_high = (-pi/2)*0.8        # Pitch angle from nose level (rad)
        self.psi_low = -pi/4
        self.psi_high = -pi/4                # Yaw angle from North (rad)
        # state = [VT, alpha, beta, phi, theta, psi, P, Q, R, pn, pe, h, pow, Nz, Ps, Ny]
        self.initial_state_low = np.array([self.Vt_low, self.alpha_low, self.beta_low, self.phi_low, self.theta_low, self.psi_low, 0, 0, 0, 0, 0, self.alt_low, self.power_low, 0, 0, 0])/1.10 #1.82
        self.initial_state_high = np.array([self.Vt_high, self.alpha_high, self.beta_high, self.phi_high, self.theta_high, self.psi_high, 0, 0, 0, 0, 0, self.alt_high, self.power_high, 0, 0, 0])*1.01 #1.06
        # handle the nagtive situation
        for i in range(len(self.initial_state_low)):
            if self.initial_state_low[i] > self.initial_state_high[i]:
                temp = self.initial_state_low[i]
                self.initial_state_low[i] = self.initial_state_high[i]
                self.initial_state_high[i] = temp

        self.initial_space = gym.spaces.Box(self.initial_state_low, self.initial_state_high, dtype=np.float32)

        # Safety Constrains
        self.flightLimits = FlightLimits()
        self.state_high = np.full(len(self.initial_state_low), 1e50)
        self.state_low = np.full(len(self.initial_state_low), -1e50)

        self.state_low[0] = self.flightLimits.vMin
        self.state_low[1] = self.flightLimits.alphaMinDeg
        self.state_low[2] = -self.flightLimits.betaMaxDeg
        self.state_low[11] = self.flightLimits.altitudeMin
        self.state_low[13] = self.flightLimits.NzMin
        self.state_high[0] = self.flightLimits.vMax
        self.state_high[1] = self.flightLimits.alphaMaxDeg
        self.state_high[2] = self.flightLimits.betaMaxDeg
        self.state_high[11] = self.flightLimits.altitudeMax
        self.state_high[13] = self.flightLimits.NzMax
        self.original = (self.state_low + self.state_high) / 2
        self.center = ((self.state_low + self.state_high) / 2)[([0, 1, 2, 11, 13])]
        self.safe_space = gym.spaces.Box(self.state_low, self.state_high, dtype=np.float32)              # Yaw angle from North (rad)
        self.safe_norm_range = (self.state_high - self.state_low)[([0, 1, 2, 11, 13])]

        ctrl_state_low = np.array([self.state_low[i] for i in [1, 7, 13, 2, 6, 8, 14, 15]], dtype=np.float32)
        ctrl_state_high = np.array([self.state_high[i] for i in [1, 7, 13, 2, 6, 8, 14, 15]], dtype=np.float32)
        self.observation_space = gym.spaces.Box(ctrl_state_low, ctrl_state_high, dtype=np.float32)

        # control limits
        self.ctrlLimits = lh.CtrlLimits()
        self.u_low = np.array([self.ctrlLimits.ThrottleMin, self.ctrlLimits.ElevatorMinDeg,
                               self.ctrlLimits.AileronMinDeg, self.ctrlLimits.RudderMinDeg])
        self.u_high = np.array([self.ctrlLimits.ThrottleMax, self.ctrlLimits.ElevatorMaxDeg,
                                self.ctrlLimits.AileronMaxDeg, self.ctrlLimits.RudderMaxDeg])
        self.action_space = gym.spaces.Box(self.u_low[1:4], self.u_high[1:4], dtype=np.float32)


