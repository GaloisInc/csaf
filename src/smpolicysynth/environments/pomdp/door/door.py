import numpy as np 
import gym
import sys
import time
import matplotlib.pyplot as pl 

from general.system import * 
from general.utils import * 
import random


class Door(System):
	def __init__(self, n_steps):
		self.num_doors = 5
		self.door_width = 0.2
		self.door_gap = 0.8

		self.dt = 0.1

		self.num_actions = 2
		self.num_cond_features = 2
		self.num_act_features = 1
		self.num_decode_params = 4
		
		self.counter = 0
		self.n_steps = n_steps

		self.dt_scale = 1.0
		self.test_dt_scale = 1.0

		self.infinite_system = False
		self.time_weight = 0.1

		self.all_states_actions = [] 

		self.state_counter = 0

	def set_inp_limits(self, lim):
		self.num_doors = int(lim[0])

	def get_wall_left(self):
		return 0.0

	def get_wall_right(self):
		return self.num_doors * self.door_width + (self.num_doors + 1)*self.door_gap 


	def simulate(self, state, action, dt):
		# state - x, counter, goal_door
		self.all_states_actions.append((state, action))

		if dt < -0.01:
			dt = self.dt
		else:
			dt = dt/self.dt_scale

		wall_left = self.get_wall_left()
		wall_right = self.get_wall_right() 

		ns = np.copy(state)
		x = ns[0]
		v = action[0]

		new_x = x + v*dt 

		if new_x < wall_left:
			new_x = wall_left
		if new_x > wall_right:
			new_x = wall_right 

		ns[0] = new_x 

		counter = ns[1]
		counter_update = action[1]/5.0
		ns[1] = counter + counter_update


		# update counter
		self.counter += 1

		return ns 


	def abstract_actions(self, a):
		a[a>=0] = 1.0
		a[a<0] = -1.0
		return a 

	def get_act_features(self, state):
		return [] # const action

	def get_door_limits(self, door):
		d_min = door*(self.door_gap + self.door_width) + self.door_gap
		d_max = d_min + self.door_width 
		return d_min, d_max


	def get_features(self, state):
		features = []
		wall_left = self.get_wall_left()
		wall_right = self.get_wall_right()

		x = state[0]

		at_door = False 
		for k in range(self.num_doors):
			d_min, d_max = self.get_door_limits(k)
			if x >= d_min and x <= d_max:
				at_door = True 
				break  

		at_right_wall = x > wall_right - self.door_gap/2.0
		at_left_wall = x < wall_left + self.door_gap/2.0 

		features.append(1.0 if at_door else -1.0)
		features.append(1.0 if at_right_wall else -1.0)
		features.append(1.0 if at_left_wall else -1.0)

		return features


	def check_safe(self, state):
		return 0.0


	def check_goal(self, state):
		# unpack
		x, counter, goal_door = state

		error = 0.0
		
		d_min, d_max = self.get_door_limits(goal_door)

		if x < d_min:
			error = d_min - x 
		if x > d_max:
			error = x - d_max

		#I = self.get_info(self.all_states_actions)
		
		return [error] #, -np.log(I)]

	def get_decode_error(self, state, decode_params):
		th0, th1, th2, th3 = decode_params 

		x, counter, _ = state 

		lower_bound = th0*counter + th1
		upper_bound = th2*counter + th3

		error = 0.0
		# check if x is in between the bounds
		if x < lower_bound:
			error += lower_bound - x 
		if x > upper_bound:
			error += x - upper_bound

		# error for info contained by the bounds 
		diff = max(0, upper_bound - lower_bound)
		error += diff

		return error 


	'''def get_info(self, state_actions):
		for s, a in state_actions:
			obs = self.get_features(s)
			if obs[1] > 0.0 or obs[2] > 0.0:
				return 1.0
		return 0.001'''


	def check_time(self, total_time):
		return 0.0

	def get_obj(self, state):
		return 0.0 
		

	def done(self, state):
		goal_err = self.check_goal(state)
		return self.counter >= self.n_steps #or np.sum(goal_err) < 0.01

	def sample_init_state(self):
		wall_left = self.get_wall_left()
		wall_right = self.get_wall_right()
		x = rand(wall_left, wall_right)
		
		goal_door = random.randint(0, self.num_doors - 1)
		
		return np.array([x, 0, goal_door])


	def get_neutral_state(self):
		x = 5.0
		goal_door = 0
		return np.array([x, 0, goal_door])


	def reset(self):
		self.counter = 0
		self.all_states_actions = [] 

	def get_plot_limits(self):
		wall_left = self.get_wall_left()
		wall_right = self.get_wall_right()
		return (wall_left -0.2, wall_right + 0.2), (0, 1)

	def plot_init(self, state):
		x, counter, goal_door = state[0]

		(x_min, x_max), (y_min, y_max) = self.get_plot_limits()

		pl.plot([x_min, x_max], [1.0, 1.0], "k")

		wall_left = self.get_wall_left()
		wall_right = self.get_wall_right()

		pl.plot([wall_left, wall_left], [y_min, y_max], "k")
		pl.plot([wall_right, wall_right], [y_min, y_max], 'k')

		for k in range(self.num_doors):
			d_min, d_max = self.get_door_limits(k)
			c = "g" if k == goal_door else "r"
			pl.plot([d_min, d_max], [0.95, 0.95], c = c)
			pl.plot([d_min, d_min], [0.95, 1.0], c = c)
			pl.plot([d_max, d_max], [0.95, 1.0], c = c)

		pl.scatter([x], [0.2])
		pl.xlim((x_min, x_max))
		pl.ylim((y_min, y_max))
		

	def plot_states(self, state_actions, line = False):
		C = [] 
		states = [x[0] for x in state_actions]
		actions = [x[1] for x in state_actions]

		X, Y = self.get_2d_states(states)

		prev_act = [] 
		upd = 0.0
		for i in range(len(actions)):
			a = actions[i] 
			if len(a) == 0:
				C.append("k")
			else:
				if a[0] >= 0:
					c = 'g'
				if a[0] <= 0:
					c = 'r'
				C.append(c)
			if len(prev_act) > 0 and len(a) > 0 and a[0]*prev_act[0] < 0:
				upd += 0.1
			prev_act = a 
			Y[i] += upd 

		if line:
			pl.plot(X,Y, c= 'k', label="Trajectory")
		else:
			pl.scatter(X, Y, c = C, s = 1)

	def plot_mode_changes(self, mode_change_states):
		X_mc, Y_mc = self.get_2d_states(mode_change_states)
		pl.scatter(X_mc, Y_mc, c= 'k', s = 10)

		
	def get_2d_states(self, states):
		X = []
		Y = []
		prev_x = states[0][0]
		for s in states:
			X.append(s[0])
			Y.append(0.2)
			prev_x = s[0]
		return X, Y 

	
