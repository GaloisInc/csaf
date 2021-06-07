import numpy as np 
import gym
import sys
import time
import matplotlib.pyplot as pl 
#from gym.envs.classic_control import rendering

from general.system import * 
from general.utils import * 
import random


class PusherSlider(System):
	def __init__(self, n_steps):
		self.goal = np.array([3.0, 2.0, 0.0, 0.0, 1.57, 0.0])

		self.dt = 0.1

		self.num_actions = 2
		self.num_cond_features = 7
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

		x,y,vx,vy,t,w = state

		ns = np.copy(state)
		a_r = action[0] # acceleration along the angle
		a_t = action[1] # acceleration tanger to the angle

		ax = a_r*np.cos(t) + a_t*np.sin(t)
		ay = a_r*np.sin(t) - a_t*np.cos(t)

		vx += ax*dt 
		vy += ay*dt 
		x += vx*dt 
		y += vy*dt 

		alpha = action[2]/10.0
		w += alpha*dt 
		t += w*dt

		#t = wrap(t, 0, 2.0*np.pi)

		ns[0] = x
		ns[1] = y
		ns[2] = vx
		ns[3] = vy 
		ns[4] = t 
		ns[5] = w
		
		# update counter
		self.counter += 1

		return ns 


	def abstract_actions(self, a):
		a[a>=0] = 1.0
		a[a<0] = -1.0
		return a 

	def get_act_features(self, state):
		return [] # const action

	
	def get_features(self, state):
		features = []

		x,y,vx,vy,t,w = state

		features.append(x)
		features.append(y)
		features.append(vx)
		features.append(vy)
		features.append(t)
		features.append(w)

		return features


	def check_safe(self, state):
		return 0.0


	def check_goal(self, state):
		error = [] 
		goal = self.goal 
		for i in range(len(goal)):
			diff = abs(state[i] - goal[i])
			if diff > 0.01:
				err = diff - 0.01
			else:
				err = 0
			error.append(err)
		return error 

	def check_time(self, total_time):
		return 0.0

	def get_obj(self, state):
		return 0.0 
		

	def done(self, state):
		goal_err = self.check_goal(state)
		return self.counter >= self.n_steps #or np.sum(goal_err) < 0.01

	def sample_init_state(self):
		x = rand(-5, 5)
		y = rand(-5, 5)
		vx = 0.0
		vy = 0.0
		t = rand(-3.14, 3.14)
		w = 0.0

		return np.array([x, y, vx, vy, t, w])


	def get_neutral_state(self):
		return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


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
		x,y,_,_,t,_ = state

		# Scales objects
		length = 0.2 * scale
		dt = self.dt
		
		if self.viewer is None:
			
			# Launches the viewer
			self.viewer = rendering.Viewer(screen_width, screen_height)
			
			# Creates the my car shape
			l,r,t,b = -length/2, length/2, length/2 , -length/2 
			block = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.blocktrans = rendering.Transform()
			block.add_attr(self.blocktrans)
			self.viewer.add_geom(block)
			
			
		
		# Translate and rotate the car
		x = scale * x + screen_width/2.0
		y = scale * y + screen_height/2.0 
		ang = t
		
		self.blocktrans.set_translation(x, y)
		self.blocktrans.set_rotation(t - np.pi / 2.0) 
		
		time.sleep(dt)
		
		return self.viewer.render(return_rgb_array = mode=='rgb_array')


	def get_plot_limits(self):
		return (-5, 5), (-5, 5)

	def plot_init(self, state):
		(x_min, x_max), (y_min, y_max) = self.get_plot_limits()
		pl.xlim((x_min, x_max))
		pl.ylim((y_min, y_max))
		

	def plot_states(self, state_actions, line = False):
		C = [] 
		states = [x[0] for x in state_actions]
		actions = [x[1] for x in state_actions]

		X, Y = self.get_2d_states(states)

		ratio = int(len(X)/len(states))

		for i in range(len(actions)):
			a = actions[i] 
			if len(a) == 0:
				for k in range(ratio):
					C.append("k")
			else:
				if a[0] >= 0:
					c = 'g'
				if a[0] <= 0:
					c = 'r'
				for k in range(ratio):
					C.append(c)

		pl.scatter(X, Y, c = C, s = 1)

	def plot_mode_changes(self, mode_change_states):
		X_mc, Y_mc = self.get_2d_states(mode_change_states)
		pl.scatter(X_mc, Y_mc, c= 'k', s = 10)

		
	def get_2d_states(self, states):
		X = []
		Y = []
		
		for s in states:
			x,y,_,_,t,_ = s

			for l in np.arange(-0.1, 0.1, 0.02):
				x1 = x + l*np.sin(t)
				y1 = y - l*np.cos(t)
				X.append(x1)
				Y.append(y1)
		return X, Y 

def wrap(x, m, M):
	diff = M - m
	while x > M:
		x = x - diff
	while x < m:
		x = x + diff
	return x

	
