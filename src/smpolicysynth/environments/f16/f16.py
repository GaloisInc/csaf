import numpy as np 
#from gym.envs.classic_control import rendering
import gym
import sys
import time
import matplotlib.pyplot as pl 
from general.system import * 
from general.utils import * 

# CSAF Imports
from environments.f16.csaf import config as cconf
from environments.f16.csaf import system as csys

class F16(System):
	def __init__(self, n_steps):
	# create a csaf configuration out of toml
		my_conf = cconf.SystemConfig.from_toml("/home/sdwivedi/csaf_architecture/examples/f16/f16_simple_config.toml")  
		# termination condition
		def ground_collision_condition(cname, outs):
			"""ground collision premature termnation condition"""
			return cname == "plant" and outs["states"][11] <= 0.0

		# create pub/sub components out of the configuration
		my_system = csys.System.from_config(my_conf)

		# create an environment from the system, allowing us to act as the controller
		self.env = csys.SystemEnv("autopilot", my_system, terminating_conditions=ground_collision_condition)	
		self.num_actions = 4
		self.num_cond_features = 13
		self.num_act_features = 1
		self.n_steps = n_steps

		self.dt_scale = 10.0
		self.test_dt_scale = 5.0
		self.time_weight = 0.01

		self.infinite_system = True
		self.desired_duration = 5.0


	def set_inp_limits(self, lim):
		self.desired_duration = lim[0]
		#self.l = lim[1]

	def simulate(self, state, action, dt):
		# Step 1: Unpack values
		if dt < -0.01:
			dt = self.dt
		else:
			dt = dt/self.dt_scale
			ns, r, d, _ = self.env.step(action)
		return ns

	def abstract_actions(self, a):
		a[a>=0] = 1.0
		a[a<0] = -1.0
		return a 

	def check_safe(self, state):
		# unpack
		'''
		x,v,t,w = state

		error = 0
		if (t > self.t_max):
			error += t - self.t_max

		if (t < self.t_min):
			error += self.t_min - t
		'''
		
		'''
		if (x < self.x_min):
			error += self.x_min - x

		if (x > self.x_max):
			error += x - self.x_max
			
		'''

		return 0 #error

	def check_goal(self, state):
		return [0] # no end goal

	def check_time(self, total_time):
		error = 0
		if total_time < self.desired_duration :
			error = (self.desired_duration - total_time)
		return error 

	def get_obj(self, state):
		return 0
		'''err = 0 
		x,v,t,w = state 

		if v < 0.0:
			err = -v 

		return err '''

		

	def done(self, state):
		return self.counter >= self.n_steps

	def sample_init_state(self):
		x = 0.0 + rand(-0.05, 0.05)
		v = 0.0 + rand(-0.05, 0.05)
		t = 0.0 + rand(-0.05, 0.05)
		w = 0.0 + rand(-0.05, 0.05)

		return np.array([x, v, t, w])

	def get_neutral_state(self):
		x = 0
		v = 0 
		t = 0.0
		w = 0.0
		return np.array([x, v, t, w])

	def get_act_features(self, state):
		return [] # const action
		'''features = []
		#features.append(state[0])
		features.append(state[1])
		features.append(state[2])
		features.append(state[3])
		return features'''

	def get_features(self, state):
		features = []
		features.append(state[0])
		features.append(state[1])
		features.append(state[2]*10.0)
		features.append(state[3]*5.0)
		return features

	def reset(self):
		#self.env.close()
		self.counter = 0

	
	def get_plot_limits(self):
		return (-0.3, 0.3), (-3, 3)

	def plot_init(self, state):
		t_min = self.t_min
		t_max = self.t_max 
		
		pl.xlim((-0.3, 0.3))
		pl.ylim((-3, 3))

		# plot boundary lines
		pl.plot([t_max, t_max], [-1, 1], "k--")
		pl.plot([t_min, t_min], [-1, 1], "k--")

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
			X.append(s[2])
			Y.append(s[3])
		return X, Y 

	def get_2d_states1(self, states):
		return self.get_2d_states(states)


	def render(self, state, mode='human'):
		x = state[0]
		
		self.env.env.state = ns
		self.env.render()


	def render(self, state, mode='human'):
		screen_width = 600
		screen_height = 400

		ns = np.copy(state)
		for i in range(10):
			if ns[0] > self.x_max:
				#print("Resetting x to x_min")
				ns[0] = self.x_min + ns[0] - self.x_max
			elif ns[0] < self.x_min:
				#print("Resetting x to x_max")
				ns[0] = self.x_max - (self.x_min -ns[0])
			else:
				break


		world_width = 2.4*2
		scale = screen_width/world_width
		carty = 100 # TOP OF CART
		polewidth = 10.0
		polelen = scale * (2 * self.l)
		cartwidth = 50.0
		cartheight = 30.0

		if self.viewer is None:
			from gym.envs.classic_control import rendering
			self.viewer = rendering.Viewer(screen_width, screen_height)
			l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
			axleoffset =cartheight/4.0
			cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			self.carttrans = rendering.Transform()
			cart.add_attr(self.carttrans)
			self.viewer.add_geom(cart)
			l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
			pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
			pole.set_color(.8,.6,.4)
			self.poletrans = rendering.Transform(translation=(0, axleoffset))
			pole.add_attr(self.poletrans)
			pole.add_attr(self.carttrans)
			self.viewer.add_geom(pole)
			self.axle = rendering.make_circle(polewidth/2)
			self.axle.add_attr(self.poletrans)
			self.axle.add_attr(self.carttrans)
			self.axle.set_color(.5,.5,.8)
			self.viewer.add_geom(self.axle)
			self.track = rendering.Line((0,carty), (screen_width,carty))
			self.track.set_color(0,0,0)
			self.viewer.add_geom(self.track)

			self._pole_geom = pole

		
		# Edit the pole polygon vertex
		pole = self._pole_geom
		l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
		pole.v = [(l,b), (l,t), (r,t), (r,b)]

		x = ns
		cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
		self.carttrans.set_translation(cartx, carty)
		self.poletrans.set_rotation(-x[2])

		return self.viewer.render(return_rgb_array = mode=='rgb_array')