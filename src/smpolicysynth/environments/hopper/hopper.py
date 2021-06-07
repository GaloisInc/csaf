import numpy as np 
#from gym.envs.classic_control import rendering
import gym
from gym.envs.mujoco import mujoco_env
import sys
import time
import matplotlib.pyplot as pl 
from general.system import * 
from general.utils import * 

class Hopper(System):
	def __init__(self, n_steps):
		self.t_max = 0.2
		self.t_min = -0.2

		self.h_min = 0.7

		self.dt = 0.008

		self.counter = 0
		self.n_steps = n_steps

		# Data structures for rendering
		self.env = gym.make('Hopper-v2')
		self.env.reset()

		self.num_actions = 3
		self.num_cond_features = 12
		self.num_act_features = 1

		self.dt_scale = 6.25
		self.test_dt_scale = 4.0
		self.time_weight = 0.01

		self.infinite_system = True
		self.desired_duration = 5.0


	def set_inp_limits(self, limits):
		self.desired_duration = limits[0]


	def get_actual_action(self, action, state):
		if len(action) == 0:
			return []
		action = np.array(action)# /5.0 * 2.0
		action = np.reshape(action, (self.num_actions))

		angle = state[2]
		thigh_ang = state[3]
		leg_ang = state[4]
		foot_ang = state[5]

		thigh_ang_vel = state[9]
		leg_ang_vel = state[10]
		foot_ang_vel = state[11]
		height = state[1]
		angle = state[2]

		act_thigh = (thigh_ang + action[0])*-1.0 + (thigh_ang_vel)*-1.0
		act_leg = (leg_ang + action[1])*-1.0 + (leg_ang_vel)*-1.0
		act_foot =  (foot_ang + action[2])*-1.0

		act_foot = max(min(act_foot, 1.0), -1.0)
		act_thigh = max(min(act_thigh, 1.0), -1.0)
		act_leg = max(min(act_leg, 1.0), -1.0)

		action = np.array([act_thigh, act_leg, act_foot])
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

		action = np.array(action)# /5.0 * 2.0
		action = np.reshape(action, (self.num_actions))
		#action = self.get_actual_action(action, state)

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
		height = state[1]
		ang = state[2]
		vel = state[6]

		error = 0.0 
		if height < self.h_min:
			error += self.h_min - height
		
		if ang < self.t_min:
			error += self.t_min - ang 
		if ang > self.t_max:
			error += ang - self.t_max

		#if vel < 0.0:
		#	error = 0.0 - vel

		return error*0.1

	def check_goal(self, state):
		safe_err = self.check_safe(state)*10.0
		if state[0] < 15.0:
			return [(15.0 - state[0])*1.0, safe_err]
		return [0, safe_err] 


	def get_obj(self, state):
		#vel = state[6]
		#if vel < 0.0:
		#	return -vel 
		return 0.0

	def check_time(self, total_time):
		error = 0.0 
		#if total_time < self.desired_duration:
		#	error += (self.desired_duration - total_time)
		
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
		return []

		'''angle = state[2]
		thigh_ang = state[3]
		leg_ang = state[4]
		foot_ang = state[5]

		thigh_ang_vel = state[9]
		leg_ang_vel = state[10]
		foot_ang_vel = state[11]
		return [angle, thigh_ang, leg_ang, foot_ang, thigh_ang_vel, leg_ang_vel, foot_ang_vel]'''

		

	def get_features(self, state):
		features = [] 
		for i in range(1, len(state)):
			features.append(state[i])

		return features

	def render(self, state, mode='human'):
		state = np.copy(state)
		s_qpos = state[0:6]
		s_qvel = state[6:12]
		self.env.set_state(s_qpos, s_qvel)
		self.env.render()

	def reset(self):
		self.env.reset()
		self.counter = 0
		

	
	def get_plot_limits(self):
		return (-0.5, 0.5), (0.5, 2.5)

	def plot_init(self, state):
		
		pl.xlim((-0.5, 0.5))
		pl.ylim((0.5, 2.5))

		# plot boundary lines
		pl.plot([self.t_min, self.t_min], [0.5, 2.5], "k--")
		pl.plot([self.t_max, self.t_max], [0.5, 2.5], "k--")

		pl.plot([-0.5, 0.5], [self.h_min, self.h_min], "k--")


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

	
		


