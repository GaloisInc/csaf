import numpy as np 
#from gym.envs.classic_control import rendering
import gym
import sys
import time
import random

from general.system import * 
from general.utils import * 
from environments.quadcopter.quad_collision import *

import matplotlib.pyplot as pl 


class Quadcopter(System):
	def __init__(self, n_steps):
		self.l = 0.2
		self.tunnel_y0_lim = (0.5, 7.5) # min, max
		self.tunnel_y1_lim = (2.0, 10.0) 
		self.tunnel_l_lim = (1.0, 1.0)
		self.num_tunnels = 40

		self.x_offset = 3.0
		self.x_start = 0.0

		self.x_lookout = 10.0
		self.y_lookout = 0.6
		
		
		self.dt = 0.05
		self.tol = 0.02
		self.t_max = 0.8
		self.t_min = -0.8

		self.world_size = 90
		self.viewer = None
		self.counter = 0
		self.n_steps = n_steps

		self.num_actions = 2
		self.num_cond_features = 6
		self.num_act_features = 1

		self.dt_scale = 1.0
		self.test_dt_scale = 2.0
		self.time_weight = 0.001

		self.infinite_system = False
		self.desired_duration = (self.num_tunnels*1.0 + self.x_offset)/2.0

	def set_inp_limits(self, lim):
		self.infinite_system = True
		self.num_tunnels = lim[0]
		self.desired_duration = (self.num_tunnels*1.0 + self.x_offset)/2.0



	def simulate(self, state, action, dt):
		# Step 1: Unpack values
		if dt < -0.01:
			dt = self.dt 
		else:
			dt = dt/self.dt_scale
		
		
		alpha = 0.0 #alpha/25.0
		x,y,vx,vy,t,w = state[0:6]
		a = (vy - action[0]/5.0*2.0)*action[1]/5.0*2.0 

		tunnel = state[6:]

		ay = a
		ax = 0.0

		ns = np.copy(state)
		ns[0] = x + vx*dt
		ns[1] = y + vy*dt
		ns[2] = vx + ax*dt
		ns[3] = vy + ay*dt
		ns[4] = t + w*dt
		ns[5] = w + alpha*dt
   
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
		x,y,vx,vy,t,w = state[0:6]
		tunnels = state[6:]
		features.append(x/5.0)
		#features.append(y)

		y_floor = 0 # dist to floor in near neighborhood
		y_roof = 12 # dist to roof in near neighborhood

		tunnels = np.array(tunnels)
		tunnels = np.reshape(tunnels, (len(tunnels)//3,3))

		t_start = self.x_start + self.x_offset
		for tunnel in tunnels:
			t_y0, t_y1, t_l = tunnel 
			t_x0 = t_start
			t_x1 = t_start + t_l

			t_start += t_l

			if (x > t_x1) or (x + self.y_lookout < t_x0):
				# out of range
				continue

			if t_y0 > y_floor:
				y_floor = t_y0 

			if t_y1 < y_roof:
				y_roof = t_y1

		features.append((y - y_floor)*5.0)
		features.append((y_roof - y)*5.0)


		x_floor = 10 # x dist at which floor y > cur y 
		x_roof = 10 # x dist at which roof y < cur y 

		got_x_floor = False 
		got_x_roof = False

		t_start = self.x_start + self.x_offset
		for tunnel in tunnels:
			t_y0, t_y1, t_l = tunnel 
			t_x0 = t_start
			t_x1 = t_start + t_l

			t_start += t_l

			if (x > t_x1) or (x + self.x_lookout < t_x0):
				# out of range
				continue

			if not got_x_floor and t_y0 > y:
				x_floor = max(t_x0 - x, 0)
				got_x_floor = True 

			if not got_x_roof and t_y1 < y:
				x_roof = max(t_x0 - x, 0)
				got_x_roof = True 

		features.append(x_roof)
		features.append(x_floor)

		return features

	def check_safe(self, state):
		e1 = self.check_collision(state)
		e2 = self.check_copter(state)
		return (e1 + e2)

	def check_collision(self, state):
		x,y,_,_,t,_ = state[0:6]
		tunnels = state[6:]

		error = 0.0
		# ground
		e1 = check_collision_with_ground(x, y, t, self.l)
		error += e1
		tunnels = np.array(tunnels)
		tunnels = np.reshape(tunnels, (len(tunnels)//3,3))

		tunnel_idx = -1
		start = self.x_start + self.x_offset
		for i in range(len(tunnels)):
			t_yl, t_yu, t_l = tunnels[i]

			t_x0 = start 
			t_x1 = start + t_l 
			if x >= t_x0 and x <= t_x1:
				e2 = check_collision_with_lower_obj(x, y, t, self.l, start, start + t_l, t_yl)
				e3 = check_collision_with_upper_obj(x, y, t, self.l, start, start + t_l, t_yu)
			
				error += e2 
				error += e3

			start += t_l

		return error

	def check_copter(self, state):
		# unpack
		x,y,_,vy,t,_ = state[0:6]
		tunnels = state[6:]

		error = 0
		if (t > self.t_max):
			error += t - self.t_max

		if (t < self.t_min):
			error += self.t_min - t

		#if (vy > 3.0):
		#	error += vy - 3.0

		#if (vy < -3.0):
		#	error += -3.0 - vy

		return error

	def check_goal(self, state):
		# unpack
		x,y,vx,vy,t,w = state[0:6]
		tunnels = state[6:]

		tunnels = np.array(tunnels)
		tunnels = np.reshape(tunnels, (len(tunnels)//3,3))

		goal_x = self.x_start  +self.x_offset 
		for tunnel in tunnels:
			goal_x += tunnel[2]

		error = 0.0
		# error for x
		if (x < goal_x):
			error += goal_x - x


		return [error];

	def check_time(self, total_time):
		return 0.0

	def get_obj(self, state):
		# try to maximize the distance from obstacles 
		x,y,vx,vy,t,w = state[0:6]
		tunnels = state[6:]

		y_floor = 0 # dist to floor in near neighborhood
		y_roof = 12 # dist to roof in near neighborhood

		tunnels = np.array(tunnels)
		tunnels = np.reshape(tunnels, (len(tunnels)//3,3))

		t_start = self.x_start + self.x_offset
		for tunnel in tunnels:
			t_y0, t_y1, t_l = tunnel 
			t_x0 = t_start
			t_x1 = t_start + t_l

			t_start += t_l

			if (x > t_x1) or (x + self.y_lookout < t_x0):
				# out of range
				continue

			if t_y0 > y_floor:
				y_floor = t_y0 

			if t_y1 < y_roof:
				y_roof = t_y1

		return abs(y - (y_roof + y_floor)/2.0)*0.01

	def done(self, state):
		return self.counter >= self.n_steps

	def sample_init_state(self):
		x = self.x_start + self.x_offset + rand(-0.04, 0.04)
		y = 1.0
		vx = 2.0 + rand(-0.04, 0.04)
		vy = 1.0 + rand(-0.04, 0.04)
		t = 0.0 + rand(-0.0, 0.0)
		w = 0.0 + rand(-0.0, 0.0)

		state = [x, y, vx, vy, t, w]

		# sample tunnel
		# sample tunnel
		old_t_y0 = 1.0
		old_t_y1 = 5.0

		y_low = rand(0.5, 2.0)
		y_high = rand(10.0, 12.0)

		delta = 0.8

		increasing = True 
		for i in range(self.num_tunnels):

			t_l = self.tunnel_l_lim[0]
			t_y0 = old_t_y0 + delta if increasing else old_t_y0 - delta
			t_y1 = old_t_y1 + delta if increasing else old_t_y1 - delta

			
			if (not increasing and t_y0 <= y_low)  or (increasing and t_y1 >= y_high) :
				increasing = not increasing
				y_low = rand(0.5, 2.0)
				y_high = rand(10.0, 12.0)
				

			old_t_y0 = t_y0
			old_t_y1 = t_y1

			state.append(t_y0)
			state.append(t_y1)
			state.append(t_l) 

			if i == 0:
				state[1] = (t_y0 + t_y1)/2.0 + rand(-0.04, 0.04)

		return np.array(state)

	def get_neutral_state(self):
		state = self.sample_init_state()
		for i in range(6):
			state[i] = 0.0
		return state

	def render(self, state, mode='human'):
		"""
		Renders the state in the viewer using openai gym
		"""
		
		# Gets scaling factors between world and screen
		screen_width = 1200
		screen_height = 200
		
		world_size = self.world_size
		
		scale = screen_width / world_size
		
		# unpack state
		ns = np.copy(state)
		ns = np.multiply(ns, scale)
		x,y,vx,vy,t,w = ns[0:6]
		tunnels = ns[6:]
		t = t/scale
		t_start= (self.x_start + self.x_offset)*scale 

		# Scales objects
		ql = self.l * 2.0*scale
		qw = 0.1 * scale
		dt = self.dt


		
		if self.viewer is None:
			
			# Launches the viewer
			self.viewer = rendering.Viewer(screen_width, screen_height)
			
			# Creates the my car shape
			l,r,t,b = -ql/2, ql/2, qw/2, -qw/2
			copter = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.coptertrans = rendering.Transform()
			copter.add_attr(self.coptertrans)
			self.viewer.add_geom(copter)

			tunnels = np.array(tunnels)
			tunnels = np.reshape(tunnels, (len(tunnels)//3, 3))

			for tunnel in tunnels:
				t_y0, t_y1, t_l = tunnel
				# Creates the obstacle1
				l,r,t,b = t_start, t_start + t_l, 0.0, t_y0
				block1 = rendering.PolyLine([(l,b), (l,t), (r,t), (r,b)], True)
				self.viewer.add_geom(block1)

				# Creates the obstacle1
				l,r,t,b = t_start, t_start + t_l, t_y1, screen_height
				block2 = rendering.PolyLine([(l,b), (l,t), (r,t), (r,b)], True)
				self.viewer.add_geom(block2)

				t_start += t_l
			
		
		# Translate and rotate the car	
		self.coptertrans.set_translation(x, y)
		self.coptertrans.set_rotation(t)
		
		time.sleep(dt)
		
		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def reset(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None
		self.counter = 0

	def get_plot_limits(self):
		xmax = self.num_tunnels*1.0 + 10.0
		return (-5, xmax),(-1 , 15)


	def plot_init(self, state):
		xmax = self.num_tunnels*1.0 + 5.0
		pl.xlim((1, xmax))
		pl.ylim((-1 , 13))

		#plot ground lines
		pl.plot([-5.0, xmax], [0, 0], "k")
		
		#plot tunnels
		x,y,_,_,t,_ = state[0][0:6] 
		tunnels = state[0][6:]

		tunnels = np.array(tunnels)
		tunnels = np.reshape(tunnels, (len(tunnels)//3, 3))

		t_start = self.x_start + self.x_offset

		for tunnel in tunnels:
			t_y0, t_y1, t_l = tunnel 
			t_x0 = t_start 
			t_x1 = t_start + t_l 
			pl.plot([t_x0, t_x0], [0.0, t_y0], "k", alpha = 0.1)
			pl.plot([t_x1, t_x1], [0.0, t_y0], "k", alpha = 0.1)
			pl.plot([t_x0, t_x0], [t_y1, 15.0], "k", alpha=0.1)
			pl.plot([t_x1, t_x1], [t_y1, 15.0], "k", alpha = 0.1)
			pl.plot([t_x0, t_x1], [t_y0, t_y0], "k", alpha = 0.1)
			pl.plot([t_x0, t_x1], [t_y1, t_y1], "k", alpha = 0.1)

			t_start += t_l

		pl.text(5.0, 1.0, 'start', horizontalalignment='center', verticalalignment='center', fontsize=20)
		pl.text(xmax - 4.0, 1.0, 'goal', horizontalalignment='center', verticalalignment='center', fontsize=20)


	def plot_states(self, state_actions, color_modes=True):
		C = [] 
		states = [x[0] for x in state_actions]
		actions = [x[1] for x in state_actions]

		X, Y = self.get_2d_states(states)

		for i in range(len(actions)):
			a = actions[i] 
			if len(a) == 0:
				C.append("k")
			else:
				#a0 = a[0] #a[0]/5.0*8.0 + 8.0 
				if color_modes:
					x,y,vx,vy,t,w = states[i][0:6]
					tunnel = states[i][6:]
					ay = (vy - a[0]/5.0*2.0)*a[1]/5.0*2.0  # a0*np.cos(t) - 9.8
					if ay >= 0:# and a[1] >= 0:
						c = 'g'
					if ay <= 0:# and a[1] >= 0:
						c = 'r'
					#if ay >= 0 and a[1] <= 0:
					#	c = 'b'
					#if ay <= 0 and a[1] <= 0:
					#	c = 'y'
					C.append(c)

		if color_modes:
			pl.scatter(X, Y, c = C, s = 1)
		else:
			pl.plot(X, Y, c='k')

	def plot_mode_changes(self, mode_change_states):
		X_mc, Y_mc = self.get_2d_states(mode_change_states)
		pl.scatter(X_mc, Y_mc, c= 'k', s = 10)

	def plot_collision_states(self, states):
		X_mc, Y_mc = self.get_2d_states(states)
		pl.scatter(X_mc, Y_mc, s = 50, facecolors='none', edgecolors='r', label="Collision")

	def get_2d_states(self, states):
		X = []
		Y = []
		for s in states:
			X.append(s[0])
			Y.append(s[1])
		return X, Y 

	def get_2d_states1(self, states):
		X = []
		Y = []
		for s in states:
			X.append(s[0])
			Y.append(s[1])
		return X, Y 


