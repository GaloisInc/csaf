import numpy as np 
#from gym.envs.classic_control import rendering
import gym
from gym.envs.mujoco import mujoco_env
import sys
import time
import matplotlib.pyplot as pl 

from general.system import * 
from general.utils import * 


class Walker(System):
	def __init__(self, n_steps):
		self.t_max = 1.0
		self.t_min = -1.0

		self.h_min = 0.8
		self.h_max = 2.0

		self.dt = 0.01

		self.counter = 0
		self.n_steps = n_steps

		# Data structures for rendering
		self.env = gym.make('Walker2d-v2')
		self.env.reset()

		self.num_actions = 6
		self.num_cond_features = 18
		self.num_act_features = 1

		self.dt_scale = 5.0
		self.test_dt_scale = 2.0
		self.time_weight = 0.01

		self.infinite_system = True
		self.desired_duration = 4.0




	def simulate(self, state, action, dt):
		if dt < -0.01:
			self.env.model.opt.timestep = self.dt 
		else:
			self.env.model.opt.timestep = dt / self.dt_scale 
		state = np.copy(state)
		s_qpos = state[0:9]
		s_qvel = state[9:18]
		self.env.set_state(s_qpos, s_qvel)

		action = np.array(action)/5.0
		action = np.reshape(action, (6))
		self.env.do_simulation(action, self.env.frame_skip)		
		ns = np.copy(state)
		qpos = self.env.sim.data.qpos
		assert(len(qpos) == 9)
		for i in range(len(qpos)):
			ns[i] = qpos[i]
		
		qvel = self.env.sim.data.qvel 
		assert(len(qvel) == 9)
		for i in range(len(qvel)):
			ns[9+i] = qvel[i]

		self.counter += 1

		return ns

	def abstract_actions(self, a):
		a[a>=0] = 1.0
		a[a<0] = -1.0
		return a 

	def check_safe(self, state):
		height = state[1]
		ang = state[2]
		vel = state[9]

		error = 0.0 
		if height < 0.8:
			error += 0.8 - height
		if height > 2.0:
			error += height - 2.0
		if ang < -1.0:
			error += -1.0 - ang 
		if ang > 1.0:
			error += ang - 1.0

		#if vel < 0.0:
		#	error = 0.0 - vel

		return error 

	def check_goal(self, state):
		return [0] # no end goal


	def get_obj(self, state):
		vel = state[9]
		if vel < 0.0:
			return -vel 
		return 0.0

	def check_time(self, total_time):
		error = 0.0 
		if total_time < self.desired_duration:
			error += (self.desired_duration - total_time)

		return error
		

	def done(self, state):
		return self.counter >= self.n_steps

	def sample_init_state(self):
		state = []
		qpos = self.env.init_qpos
		qvel = self.env.init_qvel 
		for i in range(len(qpos)):
			state.append(qpos[i] + rand(-0.004, 0.004))

		for i in range(len(qvel)):
			state.append(qvel[i] + rand(-0.004, 0.004))

		return np.array(state)

	def get_neutral_state(self):
		state = []
		qpos = self.env.init_qpos
		qvel = self.env.init_qvel 
		for i in range(len(qpos)):
			state.append(qpos[i])

		for i in range(len(qvel)):
			state.append(qvel[i])

		return np.array(state)

	def get_act_features(self, state):
		return [] # const action
		'''features = []
		features.append(state[0])
		features.append(state[1])
		features.append(state[2])
		features.append(state[3])
		return features'''

	def get_features(self, state):
		features = [] 
		for i in range(1, len(state)):
			features.append(state[i])

		return features

	def render(self, state, mode='human'):
		state = np.copy(state)
		s_qpos = state[0:9]
		s_qvel = state[9:18]
		self.env.set_state(s_qpos, s_qvel)
		self.env.render()

	def reset(self):
		self.env.reset()
		self.counter = 0

	
	def get_plot_limits(self):
		return (-1.5, 1.5), (0.5, 2.5)

	def plot_init(self, state):
		
		pl.xlim((-1.5, 1.5))
		pl.ylim((0.5, 2.5))

		# plot boundary lines
		pl.plot([1.0, 1.0], [0.5, 2.5], "k--")
		pl.plot([-1.0, -1.0], [0.5, 2.5], "k--")

		pl.plot([-1.5, 1.5], [0.8, 0.8], "k--")
		pl.plot([-1.5, 1.5], [2.0, 2.0], "k--")

	def plot_states(self, state_actions):
		C = [] 
		states = [x[0] for x in state_actions]
		actions = [x[1] for x in state_actions]

		X, Y = self.get_2d_states(states)

		for i in range(len(actions)):
			a = actions[i] 
			if len(a) == 0:
				C.append('k')
			else:
				if a[0] >= 0 and a[1] >= 0:
					c = 'g'
				if a[0] <= 0 and a[1] >= 0:
					c = 'r'
				if a[0] >= 0 and a[1] <= 0:
					c = 'b'
				if a[0] <= 0 and a[1] <= 0:
					c = 'y'
				C.append(c)

		pl.scatter(X, Y, c = C, s = 1)

	def plot_mode_changes(self, mode_change_states):
		X_mc, Y_mc = self.get_2d_states(mode_change_states)
		pl.scatter(X_mc, Y_mc, c= 'k', s = 10)


	def get_2d_states(self, states):
		X = []
		Y = []
		for s in states:
			X.append(s[2]) # angle 
			Y.append(s[1]) # height 
		return X, Y 

	def get_2d_states1(self, states):
		return self.get_2d_states(states)

	
		


