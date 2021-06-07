import numpy as np 
#from gym.envs.classic_control import rendering
import gym
import sys
import time
import matplotlib.pyplot as pl 
#import torch 

from general.system import * 
from general.utils import * 

from environments.car.collision import *



class CarReversePP(System):
	def __init__(self, n_steps):
		self.height = 5
		self.width = 1.8
		self.dist_min = 15.0
		self.dist_max = 15.0
		
		self.x_lane_1 = -1.5
		self.x_lane_2 = 1.0
		
		self.goal_ang = np.pi/2.0
		self.dt = 0.02
		self.tol = 0.05

		self.num_actions = 2
		self.num_cond_features = 6
		self.num_act_features = 1
		

		self.world_size = 30
		self.viewer = None
		self.counter = 0
		self.n_steps = n_steps

		self.dt_scale = 1.0
		self.test_dt_scale = 10.0

		self.infinite_system = False
		self.time_weight = 0.01

	def set_inp_limits(self, lim):
		self.dist_min = lim[0]
		self.dist_max = lim[1]

	def simulate(self, state, action, dt):
		if dt < -0.01:
			dt = self.dt
		else:
			dt = dt/self.dt_scale
		ns = np.copy(state)
		v, w = action 
		w = w/10.0

		if (v > 5.0):
			v = 5
		if (v < -5.0):
			v = -5
		if (w > 0.5):
			w = 0.5
		if (w < -0.5):
			w = -0.5
	

		x,y,ang, _ = state   
		beta = np.arctan(0.5*np.tan(w))
		dx = v*np.cos(ang + beta)*dt 
		dy = v*np.sin(ang + beta)*dt 
		da = v/(self.height/2.0)*np.sin(beta)*dt 

		ns[0] += dx 
		ns[1] += dy 
		ns[2] += da 

		# update counter
		self.counter += 1

		return ns 

	''' # with torch
	def simulate(self, state, action, dt):
		if dt < -0.01:
			dt = self.dt
		else:
			dt = dt/self.dt_scale

		v, w = action 
		w = w/10.0


		x,y,ang,_ = state   
		beta = torch.atan(0.5*torch.tan(w))
		dx = v*torch.cos(ang + beta)*dt 
		dy = v*torch.sin(ang + beta)*dt 
		da = v/(2.5)*torch.sin(beta)*dt 

		# update counter
		self.counter += 1

		return state + np.array([dx, dy, da, 0])'''


	def abstract_actions(self, a):
		a[a>=0] = 1.0
		a[a<0] = -1.0
		return a 

	def get_act_features(self, state):
		'''features = []
		x,y,ang,d = state 
		features.append(x/3.0)
		features.append(y/10.0)
		features.append(ang)
		return features'''
		return [] # const action


	def get_features(self, state):
		features = []
		x,y,ang, dist = state
		features.append(x)
		features.append(y)
		features.append(ang*5.0)

		d1 = 1e20 # min dist to front car
		d2 = 1e20 # min dist to back car
		d3 = 1e20 # min dist to end of lane 

		vertices = get_all_vertices(x, y, ang, self.width, self.height)
		for v in vertices:
			d = max(dist - self.height/2.0 - v[1], self.x_lane_2 - self.width/2.0 - v[0])
			if d < d1: 
				d1 = d 

			d = max(v[1] - self.height/2.0, self.x_lane_2 - self.width/2.0 - v[0])
			if d < d2:
				d2 = d 

			d = 2.2 - v[0]
			if d < d3:
				d3 = d 

		features.append(d1)
		features.append(d2)
		#features.append(d3)
		

		return features


	def check_safe(self, state):
		e1 = self.check_collision(state)
		e2 = self.check_boundaries(state)
		#assert(e1 >= 0.0)
		#assert(e2 >= 0.0)
		return (e1 + e2)

	def check_collision(self, state):
		x,y,ang,d = state

		# obstacle 1
		bx = self.x_lane_2
		by = 0.0
		e1 = check_collision_box(x, y, ang, bx, by, 'l', self.width, self.height)

		# obstacle 2
		bx = self.x_lane_2
		by = d
		e2 = check_collision_box(x, y, ang, bx, by, 'u', self.width, self.height)

		return e1 + e2

	def check_boundaries(self, state):
		x,y,ang,_ = state

		vertices = get_all_vertices(x, y, ang, self.width, self.height)
		d1 = 1e20 
		d2 = 1e20
		for v in vertices:
			d = 2.5 - v[0]
			if d < d1:
				d1 = d 

			d = v[0] - (-5)
			if d < d2:
				d2 = d 

		err = 0.0
		if d1 < 0.0:
			err += -d1 

		if d2 < 0.0:
			err += -d2 

		return err 
		
		

	def check_goal(self, state):
		# unpack
		x,y,ang, dist = state

		error = 0.0
		error_x = 0.0
		error_y = 0.0
		error_ang = 0.0
		# error for x
		if (x > self.x_lane_2 - self.width):
			error_x += x - self.x_lane_2 + self.width;

		# error for ang
		if (abs(ang - self.goal_ang) > self.tol):
			error_ang += abs(ang - self.goal_ang) - self.tol;
	
		# error for y

		#if (y < dist - self.height):
		#	error_y += dist - self.height - y

		error = error_x + error_y + error_ang 
		return [error_x, 5.0*error_ang];

	def check_time(self, total_time):
		return 0.0

	def get_obj(self, state):
		return 0.0

	def done(self, state):
		goal_err = self.check_goal(state)
		return self.counter >= self.n_steps #or np.sum(goal_err) < 0.01

	def sample_init_state(self):
		x = self.x_lane_2 + rand(-0.04, 0.04)
		#x = 1.0 + rand(-0.04, 0.04)
		ang = np.pi/2.0 + rand(-0.04, 0.04)
		dist = rand(self.dist_min, self.dist_max) 
		#y = 9.7 
		y = self.height + 0.21

		return np.array([x, y, ang, dist])


	def get_neutral_state(self):
		x = 0.0
		ang = np.pi/2.0 
		dist = 15.0
		y = 2.5
		return np.array([x, y, ang, dist])



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
		x,y,ang,dist = state

		# Scales objects
		dist = dist * scale
		w_car = self.width * scale
		h_car = self.height * scale
		dt = self.dt
		vshift = -5 * scale
		
		if self.viewer is None:
			
			# Launches the viewer
			self.viewer = rendering.Viewer(screen_width, screen_height)
			
			# Creates the my car shape
			l,r,t,b = -w_car/2, w_car/2, h_car/2 , -h_car/2 
			car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			car.set_color(0.5, 0.5, 0.8)
			self.cartrans = rendering.Transform()
			car.add_attr(self.cartrans)
			self.viewer.add_geom(car)
			
			# Creates the stationary car 1
			l,r,t,b = -w_car/2, w_car/2, h_car/2 , -h_car/2 
			car1 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.cartrans1 = rendering.Transform()
			car1.add_attr(self.cartrans1)
			self.cartrans1.set_translation(self.x_lane_2*scale + screen_width/2.0 , screen_height/2.0 + vshift)
			self.viewer.add_geom(car1)

			# Creates the stationary car 2
			l,r,t,b = -w_car/2, w_car/2, h_car/2 , -h_car/2 
			car2 = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.cartrans2 = rendering.Transform()
			car2.add_attr(self.cartrans2)
			self.cartrans2.set_translation(self.x_lane_2*scale + screen_width/2.0, dist + screen_height/2.0 + vshift)
			self.viewer.add_geom(car2)
			
		
		# Translate and rotate the car
		x = scale * x + screen_width/2.0
		y = scale * y + screen_height/2.0 + vshift
		ang = ang
		
		self.cartrans.set_translation(x, y)
		self.cartrans.set_rotation(ang - np.pi / 2.0) 
		
		time.sleep(dt)
		
		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def reset(self):
		if self.viewer:
			self.viewer.close()
			self.viewer = None
		self.counter = 0
		self.mode_change_states = []

	def get_plot_limits(self):
		return (-4, 2.2), (-5, 20)

	def plot_init(self, state):
		x,y,ang,dist = state[0] 

		H = self.height
		W = self.width
		l,r,t,b = -W/2.0 + self.x_lane_2, W/2.0 + self.x_lane_2, H/2.0, - H/2.0
		pl.fill([l, l, r, r], [b, t, t, b], "k")
		pl.fill([l, l, r, r], [b + dist, t + dist, t + dist, b + dist], 'k')

		pl.fill([l, l, r, r], [b + H + 0.2, t + H + 0.2, t + H + 0.2, b + H + 0.2], 'b', alpha = 0.1)

		v = get_all_vertices(x, y, ang, self.width, self.height)
		pl.fill([v[2][0], v[1][0], v[0][0], v[3][0] ], [v[2][1], v[1][1], v[0][1], v[3][1] ], 'b', alpha = 0.1)

		x_min = -4.0
		x_max = 2.2

		y_min = -5 # 0 + 2
		y_max = 20 # 14



		pl.xlim((x_min, x_max))
		pl.ylim((y_min, y_max))
		#pl.gca().set_aspect('equal', adjustable='box')
		#pl.plot([self.x_lane_2 + W/2.0 + 0.2, self.x_lane_2 + W/2.0 + 0.2], [y_min, y_max], "k")


	def plot_init_paper(self, state1, state2):
		x,y,ang,dist = state1[0] 

		H = self.height
		W = self.width
		l,r,t,b = -W/2.0 + self.x_lane_2, W/2.0 + self.x_lane_2, H/2.0, - H/2.0
		pl.fill([l, l, r, r], [b, t, t, b], "k")
		pl.fill([l, l, r, r], [b + dist, t + dist, t + dist, b + dist], 'k')

		
		v = get_all_vertices(x, y, ang, self.width, self.height)
		pl.fill([v[2][0], v[1][0], v[0][0], v[3][0] ], [v[2][1], v[1][1], v[0][1], v[3][1] ], 'b', alpha = 0.2)
		pl.text(x, y, 'start', horizontalalignment='center', verticalalignment='center', fontsize=20)

		x,y,ang,dist = state2[0] 
		v = get_all_vertices(x, y, ang, self.width, self.height)
		pl.fill([v[2][0], v[1][0], v[0][0], v[3][0] ], [v[2][1], v[1][1], v[0][1], v[3][1] ], 'b', alpha = 0.2)
		pl.text(x, y, 'goal', horizontalalignment='center', verticalalignment='center', fontsize=20)

		x_min = -4.0
		x_max = 2.2

		y_min = -3 # 0 + 2
		y_max = 18 # 14



		pl.xlim((x_min, x_max))
		pl.ylim((y_min, y_max))
		#pl.gca().set_aspect('equal', adjustable='box')
		#pl.plot([self.x_lane_2 + W/2.0 + 0.2, self.x_lane_2 + W/2.0 + 0.2], [y_min, y_max], "k")


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
				if a[0] >= 0 and a[1] >= 0:
					c = 'g'
				if a[0] <= 0 and a[1] >= 0:
					c = 'y'
				if a[0] >= 0 and a[1] <= 0:
					c = 'b'
				if a[0] <= 0 and a[1] <= 0:
					c = 'r'
				C.append(c)

		if line:
			pl.plot(X,Y, c= 'k', label="Trajectory")
		else:
			pl.scatter(X, Y, c = C, s = 1)

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
			v = get_all_vertices(s[0], s[1], s[2], self.width, self.height)
			x = (v[0][0] + v[1][0])/2.0
			y = (v[0][1] + v[1][1])/2.0
			X.append(x)
			Y.append(y)
		return X, Y 

	def get_2d_states1(self, states):
		X = []
		Y = []
		for s in states:
			v = get_all_vertices(s[0], s[1], s[2], self.width, self.height)
			x = (v[2][0] + v[3][0])/2.0
			y = (v[2][1] + v[3][1])/2.0
			X.append(x)
			Y.append(y)
		return X, Y 
		

	'''def get_2d_cond(self, cond, last_states):
		X = []
		Y = []
		#print(len(last_states))
		for s in last_states[:1]:
			last_x, last_y, _, _ = s
			X1 = np.arange(-10, 10, 0.01)
			#Y1 = (cond[0]*X1 + cond[2]*last_x + cond[3]*last_y + cond[4])/(-cond[1])
			y1 = ( cond[1])/(-cond[0])
			Y1 = [y1]*len(X1)
			X.extend(X1)
			Y.extend(Y1)
			X11 = np.arange(-10, 10, 1)
			if cond[0] > 0.0:
				for y in np.arange(y1, 20, 1.0):
					X.extend(X11)
					Y.extend([y]*len(X11))
			else:
				for y in np.arange(-5, y1, 1.0):
					X.extend(X11)
					Y.extend([y]*len(X11))

		return X, Y 
		'''

	def get_cond_states(self, states):
		X = []
		Y = []
		for s in states:
			features = self.get_features(s)
			X.append(features[0])
			Y.append(features[1])
		return X, Y 

	def get_2d_cond(self, cond):
		X = []
		Y = []
		
		if (abs(cond[1]) >= 0.01):
			X1 = np.arange(-10, 10, 0.01)
			Y1 = (X1*cond[0] + cond[2] )/(-cond[1])
			X.extend(X1)
			Y.extend(Y1)
		else:
			Y1 = np.arange(-10, 10, 0.01)
			X1 = (Y1*cond[1] + cond[2])/(-cond[0])
			X.extend(X1)
			Y.extend(Y1)

		X11 = np.arange(-10, 10, 1)
		Y11 = np.arange(-10, 10, 1)
		for x in X11:
			for y in Y11:
				if cond[0]*x + cond[1]*y + cond[2] > 0.0:
					X.append(x)
					Y.append(y)

		return X, Y 

