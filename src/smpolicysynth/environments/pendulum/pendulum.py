import numpy as np 
#from gym.envs.classic_control import rendering
import gym
import sys
import time
import matplotlib.pyplot as pl 
from general.system import * 
from general.utils import * 


class Pendulum(System):
	def __init__(self, n_steps):
		self.g = 10.0
		self.l = 1.0
		self.dt = 0.01

		self.max_torque = 2.0
		self.max_speed = 8.0
		

		self.goal = np.array([0.0, 0.0])

		self.world_size = 60
		self.viewer = None
		self.counter = 0
		self.n_steps = n_steps

		self.mass_low = 1.0
		self.mass_high = 1.0

		# Data structures for rendering
		self.env = gym.make('Pendulum-v0')
		self.env.reset()
		self.env.last_u = 1e-10
		self.num_actions = 1
		self.num_cond_features = 3
		self.num_act_features = 1

		self.dt_scale = 5.0
		self.test_dt_scale = 2.0
		self.time_weight = 0.01

		self.infinite_system = False

	def set_inp_limits(self, lim):
		self.mass_low = lim[0]
		self.mass_high = lim[1]



	def simulate(self, state, action, dt):
		# Step 1: Unpack values
		if dt < 0.0:
			dt = self.dt 
		else:
			dt = dt/self.dt_scale
		th, thdot, m = state
		#print(state)
		u = action[0]/5.0*self.max_torque
		u = np.clip(u, -self.max_torque, self.max_torque)
		
		g = self.g
		l = self.l

		self.env.last_u = u # for rendering

		newthdot = thdot + (-3.0*g/(2.0*l) * np.sin(th + np.pi) + 3.0/(m*l*l)*u) * dt		
		newth = th + thdot*dt
		#newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

		ns = np.copy(state)
		
		ns[0] = newth
		ns[1] = newthdot

		self.counter += 1

		return ns

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
		features = []
		th, thdot, m = state
		features.append(th)
		#features.append(np.cos(th)*10.0)
		#features.append(np.sin(th)*10.0)
		features.append(thdot)
		#features.append(m*10.0)
		return features

	def check_safe(self, state):
		return 0 # no safety property

	def check_goal(self, state):
		ang = state[0]
		if ang > np.pi:
			ang = ang - 2.0*np.pi
		th_err = 0.0
		if ang > 0.05:
			th_err = ang - 0.05
		if ang < -0.05:
			th_err = -0.05 - ang 

		#thdot_err =  0.1 * abs(state[1] - self.goal[1]) 
		error =  th_err # + thdot_err
		return [th_err] # , thdot_err]

	def check_time(self, total_time):
		return 0.0

	def get_obj(self, state):
		return 0.0

	def done(self, state):
		err = np.sum(self.check_goal(state))
		return self.counter >= self.n_steps # or err < 0.01

	def sample_init_state(self):
		th = np.pi + rand(-4, 4)/100.0
		thdot = 0.0 + rand(-4, 4)/100.0
		mass = rand(self.mass_low, self.mass_high)

		return np.array([th, thdot, mass])

	def get_neutral_state(self):
		th = np.pi 
		thdot = 0.0 
		mass = 1.0
		return np.array([th, thdot, mass])

	def set_act_for_render(self, action):
		if len(action) == 0: return 
		u = action[0]/5.0*self.max_torque
		u = np.clip(u, -self.max_torque, self.max_torque)
		
		g = self.g
		l = self.l

		self.last_u = u # for rendering

	def render(self, state, mode='human'):
		#self.env.env.state = state[0:2]
		#self.env.render()

		if self.viewer is None:
			self.viewer = rendering.Viewer(500,500)
			self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
			rod = rendering.make_capsule(1, .2)
			rod.set_color(.8, .3, .3)
			self.pole_transform = rendering.Transform()
			rod.add_attr(self.pole_transform)
			self.viewer.add_geom(rod)
			axle = rendering.make_circle(.05)
			axle.set_color(0,0,0)
			self.viewer.add_geom(axle)
			#fname = path.join(path.dirname(__file__), "assets/clockwise.png")
			#self.img = rendering.Image(fname, 1., 1.)
			#self.imgtrans = rendering.Transform()
			#self.img.add_attr(self.imgtrans)

		#self.viewer.add_onetime(self.img)
		self.pole_transform.set_rotation(state[0] + np.pi/2)
		#if self.last_u:
		#	self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

		return self.viewer.render(return_rgb_array = mode=='rgb_array')


	def reset(self):
		self.env.reset()
		self.counter = 0

	def get_plot_limits(self):
		return (-2, 8), (-8, 8)

	def plot_init(self, state):
		pl.xlim((-2, 8))
		pl.ylim((-8, 8))

		#plot goal lines
		pl.plot([0.0, 0.0], [-2, 2], "k--")
		pl.plot([2*np.pi, 2*np.pi], [-2, 2], "k--")
		pl.plot([-2, 8], [0, 0], "k--")
	

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
			Y.append(s[1])
		return X, Y 

	def get_2d_states1(self, states):
		return self.get_2d_states(states)
	

def angle_normalize(x):
	return (((x+np.pi) % (2*np.pi)) - np.pi)



