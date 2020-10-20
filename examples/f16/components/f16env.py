import enum
import gym
import numpy as np
import matplotlib.pyplot as plt

import autopilot as ap

# CSAF Imports
import csaf.config as cconf
import csaf.system as csys

states = ['vt', 'alpha', 'beta', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'pn', 'pe', 'h', 'power']
State = enum.IntEnum('State', states, start=0)

def state_vector(**x):
    l = [0]*len(State)
    for k, d in x.items():
        l[getattr(State, k)] = d
    return np.array(l)

######################################
# F16 GCAS Gym Environment
######################################

class F16GCAS(gym.Env):
    metadata = {}
    def __init__(self):

        self.x0 = state_vector(vt=540, alpha=np.deg2rad(2.1215), beta=0, phi=0,
                theta=0, psi=0, p=0, q=0, r=0, pn=0, pe=0, h=3600, power=9)
        self.state = self.x0
        self.csaf_init()



    def csaf_init(self):

        # create a csaf configuration out of toml
        my_conf = cconf.SystemConfig.from_toml("../f16_simple_config.toml")

        # termination condition
        def ground_collision(cname, outs):
            """ground collision premature termnation condition"""
            return cname == "plant" and outs["states"][11] <= 0.0

        # create pub/sub components out of the configuration
        my_system = csys.System.from_config(my_conf)

        # create an environment from the system, allowing us to act as the controller
        self.csaf_env = csys.SystemEnv("autopilot", my_system, terminating_conditions=ground_collision)

    def step(self, ctrl_signal):


        done = False
        # step through simulation and collect f16 states
        try:
            comp_input_buffer = self.csaf_env.step({
                "autopilot-states": ["Waiting"],
                "autopilot-fdas": ["Waiting"],
                "autopilot-outputs": ctrl_signal
                })
        except StopIteration: done = True

        pstates = comp_input_buffer['plant-states']
        self.state = pstates
        return pstates, self.reward(pstates), done, {}

    def reward(self, state, action=None):
        return 0

    def reset(self, pstate=None):
        # return current state
        self.state = pstate if pstate is not None else self.x0
        self.csaf_env.reset()
        return self.state

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

######################################
# Simulator code follows
######################################

def plot(pstates):
    # plot the results
    plt.plot(pstates[:, 11], label='F16 Altitude')
    plt.xlabel("Step Index [n]")
    plt.ylabel("[ft]")
    plt.legend()
    plt.show()

# XXX: Get this out from CSAF itself!
class Autopilot:

    def __init__(self):
        class Model:
            NzMax = 5.0
            vt_des = 502.0
        self.model = Model
        self.state = [ap.GcasAutopilot.STATE_START]

    def step(self, t, xf16):
        self.state = ap.model_state_update(self.model, t, self.state, xf16)
        return ap.model_output(self.model, t, self.state, xf16)
def test_model(model, env):
    # XXX: get the time from the simulator?
    dt = 0.1
    obs_trace, t, done = [env.reset()], 0, False
    while not done and t <= 30:
        action = model.step(t, obs_trace[-1])
        obs, reward, done, info = env.step(action)
        obs_trace.append(obs)
        t += dt
    plot(np.array(obs_trace))


def register_env():
    gym.envs.register(
            id='F16GCAS-v0',
            entry_point=F16GCAS,
            max_episode_steps=5000,
            )

def test():
    env = gym.make('F16GCAS-v0')
    env.reset()
    for _ in range(1000):
        #env.render()
        env.step(env.action_space.sample()) # take a random action
    env.close()


if __name__ == "__main__":
    register_env()
    test_model(Autopilot(), gym.make('F16GCAS-v0'))
