import numpy as np 
#from gym.envs.classic_control import rendering
import gym
import sys
import time
import matplotlib.pyplot as pl 
from general.system import * 
from general.utils import * 

class MountainCar(System):
	def __init__(self, n_steps):
		self.max_action = 1.0
		self.min_position = -1.2
		self.max_position = 0.6
		self.max_speed = 0.07
		self.goal_position = 0.45
		self.power_min = 15
		self.power_max = 15 # * 1e-4

		self.dt = 1.0

		self.viewer = None
		self.counter = 0
		self.n_steps = n_steps


		# Data structures for rendering
		self.env = gym.make('MountainCarContinuous-v0')
		self.env.reset()

		self.num_actions = 1
		self.num_cond_features = 3
		self.num_act_features = 1

		self.dt_scale = 0.1
		self.test_dt_scale = 2.0
		self.time_weight = 0.001

		self.infinite_system = False


	def set_inp_limits(self, limits):
		self.power_min = limits[0]
		self.power_max = limits[1]


	def simulate(self, state, action, dt):
		# Step 1: Unpack values
		if dt < -0.01:
			dt = self.dt 
		else:
			dt = dt/self.dt_scale

		ns = np.copy(state)
		p, v, power = state
		force = action[0]/5.0*self.max_action
		
		
		v += (force*power - 0.0025 * np.cos(3*p))*dt
		v = np.clip(v, -self.max_speed, self.max_speed)
		
		p += v*dt
		p = np.clip(p, self.min_position, self.max_position)

		if p == self.min_position and v < 0:
			v = 0
				
		ns[0] = p
		ns[1] = v

		self.counter += 1
		#print(self.counter)
		return ns

	def check_safe(self, state):
		return 0 # no safety property

	def check_goal(self, state):
		p,v, power = state
		if (p < self.goal_position):
			return [(self.goal_position - p)*5.0]
		else:
			return [0.0]


	def check_time(self, total_time):
		return 0.0

	def get_obj(self, state):
		return 0.0


	def abstract_actions(self, a):
		a[a>=0] = 1.0
		a[a<0] = -1.0
		return a 

	def get_act_features(self, state):
		return [] # const action
		'''features = []
		th, thdot, _ = state
		features.append(th)
		features.append(thdot)
		return features'''

	def get_features(self, state):
		p,v,power = state

		features = []
		features.append(p*10.0)
		features.append(v*100.0)
		return features


	def done(self, state):
		return self.counter >= self.n_steps #or self.check_goal(state) < 0.01

	def sample_init_state(self):
		p = -0.6 + rand(-0.04, 0.04)
		v = 0.0 + rand(-0.004, 0.004)
		power = rand(self.power_min, self.power_max)*1e-4
		return np.array([p, v, power])

	def get_neutral_state(self):
		p = -0.6
		v = 0
		power = 0.0015
		return np.array([p, v, power])

	def render(self, state, mode='human'):
		self.env.env.state = state[0:2]
		self.env.render()

	def reset(self):
		self.env.close()
		self.counter = 0

	def get_plot_limits(self):
		return (-2, 1), (-1, 1)

	def plot_init(self, state):
		pl.xlim((-2, 1))
		pl.ylim((-1, 1))

		#plot goal lines
		pl.plot([0.45, 0.45], [-1, 1], "k--")
		

	def plot_states(self, state_actions):
		C = [] 
		states = [x[0] for x in state_actions]
		actions = [x[1] for x in state_actions]

		X, Y = self.get_2d_states(states)

		for i in range(len(actions)):
			a = actions[i] 
			if len(a) == 0:
				C.append("b")
			else:
				C.append('g' if a[0] >= 0 else 'r')

		pl.scatter(X, Y, c = C, s = 1)

	def plot_mode_changes(self, mode_change_states):
		X_mc, Y_mc = self.get_2d_states(mode_change_states)
		pl.scatter(X_mc, Y_mc, c= 'k', s = 10)

	def get_2d_states(self, states):
		X = []
		Y = []
		for s in states:
			X.append(s[0])
			Y.append(s[1]*10.0)
		return X, Y 

	def get_2d_states1(self, states):
		return self.get_2d_states(states)


