import numpy as np 
from gym.envs.classic_control import rendering
import gym
from gym.envs.mujoco import mujoco_env
import sys
import time
import matplotlib.pyplot as pl

from general.system import * 
from general.utils import * 

from environments.swimmer.swimmer4.swimmer4Mujoco import Swimmer4Env


class Swimmer4(System):
	def __init__(self, n_steps):
		

		self.dt = 0.04

		self.counter = 0
		self.n_steps = n_steps

		# Data structures for rendering
		self.env = Swimmer4Env()
		self.env.reset()

		self.num_actions = 3
		self.num_cond_features = 11
		self.num_act_features = 1

		self.dt_scale = 1.25
		self.test_dt_scale = 2.0
		self.time_weight = 0.0

		self.infinite_system = True
		self.desired_duration = 50.0
		self.desired_distance = 8.0


	def set_inp_limits(self, limits):
		self.desired_distance = limits[0]


	def get_actual_action(self, action, state):
		if len(action) == 0:
			return []
		action = np.array(action)/5.0 * 2.0
		action = np.reshape(action, (self.num_actions))
		x,y,ang, ang_mid, ang_back, ang_back2 = state[0:6]

		act1 = (ang_mid + action[0])*-1.0
		act2 = (ang_back + action[1])*-1.0
		act3 = (ang_back2 + action[2])*-1.0

		act1 = max(min(act1, 1.0), -1.0)
		act2 = max(min(act2, 1.0), -1.0)
		act3 = max(min(act3, 1.0), -1.0)
		action = np.array([act1, act2, act3])
		return action


	def simulate(self, state, action, dt):
		if dt < -0.01:
			self.env.model.opt.timestep = (self.dt)/float(self.env.frame_skip)
		else:
			self.env.model.opt.timestep = (dt / self.dt_scale )/ float(self.env.frame_skip)
		#print(state)
		state = np.copy(state)
		s_qpos = state[0:6]
		s_qvel = state[6:12]
		self.env.set_state(s_qpos, s_qvel)

		action = self.get_actual_action(action, state)

		self.env.do_simulation(action, self.env.frame_skip)		
		ns = np.copy(state)
		qpos = self.env.sim.data.qpos
		assert(len(qpos) == 6)
		for i in range(len(qpos)):
			ns[i] = qpos[i]
		
		qvel = self.env.sim.data.qvel 
		assert(len(qvel) == 6)
		for i in range(len(qvel)):
			ns[6+i] = qvel[i]

		self.counter += 1

		return ns

	def abstract_actions(self, a):
		a[a>=0] = 1.0
		a[a<0] = -1.0
		return a 

	def check_safe(self, state):
		return 0.0

	def check_goal(self, state):
		if state[0] < self.desired_distance:
			return [(self.desired_distance - state[0])*1.0]
		return [0] 


	def get_obj(self, state):
		error = 0.0;
		vx = state[6]
		if vx < 0.0:
			error += -vx*0.1 

		y = state[1]
		if y < -1.0:
			error += (-1.0 - y)*0.1
		if y > 1.0:
			error += (y - 1.0)*0.1

		return error

	def check_time(self, total_time):
		error = 0.0 
		#if total_time < self.desired_duration:
		#	error += (self.desired_duration - total_time)
		
		return error
		

	def done(self, state):
		return self.counter >= self.n_steps or np.sum(self.check_goal(state)) < 0.01

	def sample_init_state(self):
		state = []
		qpos = self.env.init_qpos
		qvel = self.env.init_qvel 
		
		for i in range(len(qpos)):
			state.append(qpos[i] + rand(-0.04, 0.04))

		for i in range(len(qvel)):
			state.append(qvel[i] + rand(-0.04, 0.04))

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
		return []


	def get_features(self, state):
		features = [] 
		for i in range(2, len(state)):
			features.append(state[i])

		return features

	def render(self, state, mode='human'):
		state = np.copy(state)
		s_qpos = state[0:6]
		s_qvel = state[6:12]
		self.env.set_state(s_qpos, s_qvel)
		self.env.render(mode=mode)

	def reset(self):
		self.env.reset()
		self.counter = 0
		

	
	def get_plot_limits(self):
		return (-2, 10), (-3, 3)

	def plot_init(self, state):
		xlim, ylim = self.get_plot_limits()
		pl.xlim(xlim)
		pl.ylim(ylim)



	def plot_states(self, state_actions):
		C = [] 
		states = [x[0] for x in state_actions]
		actions = [x[1] for x in state_actions]

		X, Y = self.get_2d_states(states)

		colors = ['g', 'r', 'b', 'k']
		act_to_colors = {}


		for i in range(len(actions)):
			act = np.copy(actions[i])
			if len(act) == 0 : continue 
			act = self.abstract_actions(act)
			astr = np.array2string(act)
			if astr not in act_to_colors:
				c = colors[len(act_to_colors)]
				act_to_colors[astr] = c
		print(act_to_colors)

		for i in range(len(actions)):
			a = actions[i] 
			if len(a) == 0:
				C.append('k')
			else:
				act = np.copy(a)
				act = self.abstract_actions(act)
				astr = np.array2string(act)
				c = act_to_colors[astr]
				'''if a[0] >= 0 and a[1] >= 0:
					c = 'g'
				if a[0] <= 0 and a[1] >= 0:
					c = 'r'
				if a[0] >= 0 and a[1] <= 0:
					c = 'b'
				if a[0] <= 0 and a[1] <= 0:
					c = 'y' '''
				C.append(c)

		pl.scatter(X, Y, c = C, s = 1)

	def plot_mode_changes(self, mode_change_states):
		X_mc, Y_mc = self.get_2d_states(mode_change_states)
		pl.scatter(X_mc, Y_mc, c= 'k', s = 10)


	def get_2d_states(self, states):
		X = []
		Y = []
		for s in states:
			X.append(s[0]) 
			Y.append(s[1]) 
		return X, Y 

	def get_2d_states1(self, states):
		return self.get_2d_states(states)

	
		


