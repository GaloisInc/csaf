import numpy as np 
import gym
import sys
import time
import matplotlib.pyplot as pl 
#from gym.envs.classic_control import rendering

from general.system import * 
from general.utils import * 
import random


class Toy(System):
	def __init__(self, n_steps):
		self.dt = 0.1

		self.num_actions = 2
		self.num_cond_features = 5
		self.num_act_features = 1
		
		self.counter = 0
		self.n_steps = n_steps
		self.world_size = 10
		self.viewer = None

		self.dt_scale = 1.0
		self.test_dt_scale = 1.0

		self.infinite_system = False
		self.time_weight = 0.1


	def simulate(self, state, action, dt):

		if dt < -0.01:
			dt = self.dt
		else:
			dt = dt/self.dt_scale
		x,y,_,_ = state
		ns = np.copy(state)
		vx,vy = action
		x += vx*dt/5.0 
		y += vy*dt/5.0 

		ns[0] = x
		ns[1] = y
		
		# update counter
		self.counter += 1

		return ns 

	def abstract_actions(self, a):
		a[a/5.0>=0.5] = 0.75*5.0
		a[a/5.0<0.5] = 0.25*5.0
		return a 

	def get_act_features(self, state):
		return [] # const action

	def get_features(self, state):
		features = []

		x,y,gx,gy = state

		features.append(x)
		features.append(y)
		features.append(gx - x)
		features.append(gy - y)

		return features

	def check_safe(self, state):
		return 0.0

	def check_goal(self, state):
		x,y,gx,gy = state 
		if abs(gx - x) < 0.1:
			error_x = 0
		else:
			error_x = abs(gx - x) - 0.1
		if abs(gy - y) < 0.1:
			error_y = 0
		else:
			error_y = abs(gy - y) - 0.1
		return [error_x*10.0, error_y*10.0] 

	def check_time(self, total_time):
		return 0.0

	def get_obj(self, state):
		return 0.0 
		
	def done(self, state):
		return self.counter >= self.n_steps 

	def sample_init_state(self):
		x = 0
		y = 0
		v = rand(0, 1.0)
		if v < 0.33:
			gx = 1.0
			gy = 0.0
		elif v < 0.66:
			gx = 0.0
			gy = 1.0
		else:
			gx = 1.0
			gy = 1.0
		return np.array([x, y, gx, gy])


	def get_neutral_state(self):
		return np.array([0.0, 0.0, 1.0, 1.0])


	def reset(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None
		self.counter = 0

	def render(self, state, mode='human'):
		"""
		Renders the state in the viewer using openai gym
		"""
		
		# Gets scaling factors between world and screen
		screen_width = 600
		screen_height = 600
		
		world_size = self.world_size
		
		scale = screen_width / world_size
		
		# unpack state
		x,y,_,_ = state

		# Scales objects
		length = 0.2 * scale
		dt = self.dt
		
		if self.viewer is None:
			
			# Launches the viewer
			self.viewer = rendering.Viewer(screen_width, screen_height)
			
			# Creates the my shape
			l,r,t,b = -length/2, length/2, length/2 , -length/2 
			block = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.blocktrans = rendering.Transform()
			block.add_attr(self.blocktrans)
			self.viewer.add_geom(block)
			
			
		
		# Translate 
		x = scale * x + screen_width/2.0
		y = scale * y + screen_height/2.0 
		ang = t
		
		self.blocktrans.set_translation(x, y)		
		time.sleep(dt)
		
		return self.viewer.render(return_rgb_array = mode=='rgb_array')


	def get_plot_limits(self):
		return (-1, 5), (-1, 5)

	def plot_init(self, state):
		(x_min, x_max), (y_min, y_max) = self.get_plot_limits()
		pl.xlim((x_min, x_max))
		pl.ylim((y_min, y_max))
		

	def plot_states(self, state_actions, line = False):
		C = [] 
		states = [x[0] for x in state_actions]
		actions = [x[1] for x in state_actions]

		X, Y = self.get_2d_states(states)

		for i in range(len(actions)):
			a = actions[i] 
			if len(a) == 0:
				C.append("k")
			else:
				if a[0]/5.0 >= 0.5 and a[1]/5.0 >= 0.5:
					c = 'g'
				if a[0]/5.0 <= 0.5 and a[1]/5.0 >= 0.5:
					c = 'y'
				if a[0]/5.0 >= 0.5 and a[1]/5.0 <= 0.5:
					c = 'b'
				if a[0]/5.0 <= 0.5 and a[1]/5.0 <= 0.5:
					c = 'r'
				C.append(c)

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
		for s in states:
			X.append(s[0])
			Y.append(s[1])
		return X, Y 

	def get_2d_states1(self, states):
		return self.get_2d_states(states)

	
