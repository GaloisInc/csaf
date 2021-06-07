import numpy as np 
#from gym.envs.classic_control import rendering
import gym
import sys
import time
import matplotlib.pyplot as pl
 
from general.system import * 
from general.utils import * 

class Acrobot(System):
	def __init__(self, n_steps):
		self.g = 9.8
		self.l1 = 1.0
		self.l2 = 1.0
		self.lc1 = 0.5
		self.lc2 = 0.5
		self.I = 1.0 # moment of inertia per mass

		self.mass_low = 1.0
		self.mass_high = 1.0


		self.max_v1 = 4*np.pi
		self.max_v2 = 9*np.pi 

		self.max_torque = 1.0

		self.dt = 0.05

		self.world_size = 60
		self.viewer = None
		self.counter = 0
		self.n_steps = n_steps


		# Data structures for rendering
		self.env = gym.make('Acrobot-v1')
		self.env.reset()

		self.num_actions = 1
		self.num_cond_features = 6
		self.num_act_features = 1

		self.dt_scale = 1.0
		self.test_dt_scale = 2.0
		self.time_weight = 0.01


		self.infinite_system = False

	def set_inp_limits(self, lim):
		self.mass_low = lim[0]
		self.mass_high = lim[1]



	def simulate(self, state, action, dt):
		# Step 1: Unpack values
		if dt < -0.01:
			dt = self.dt 
		else:
			dt = dt/self.dt_scale

		
		torque = action[0]/5.0*self.max_torque
		
		ns = np.copy(state)
		ns = np.append(ns, torque)

		ns = rk4(self._dsdt, ns, [0, dt])
		# only care about final timestep of integration returned by integrator
		ns = ns[-1]
		ns = ns[:-1]
		

		ns[0] = wrap(ns[0], -np.pi, np.pi)
		ns[1] = wrap(ns[1], -np.pi, np.pi)
		ns[2] = np.clip(ns[2], -self.max_v1, self.max_v1)
		ns[3] = np.clip(ns[3], -self.max_v2, self.max_v2)

		self.counter += 1

		return ns

	def _dsdt(self, s_augmented, t):	
		torque = s_augmented[-1]
		state = s_augmented[:-1]
		th1, th2, thdot1, thdot2, m1, m2 = state
		
		g = self.g
		l1 = self.l1
		l2 = self.l2
		lc1 = self.lc1
		lc2 = self.lc2
		I1 = self.I * m1
		I2 = self.I * m2


		d1 = m1*lc1*lc1 + m2*(l1*l1 + lc2*lc2 + 2*l1*lc2*np.cos(th2)) + I1 + I2
		d2 = m2*(lc2*lc2 + l1*lc2*np.cos(th2)) + I2
		phi2 = m2*lc2*g*np.cos(th1 + th2 - np.pi/2.0)
		phi1 = -m2*l1*lc2*thdot2*thdot2*np.sin(th2) - 2*m2*l1*lc2*thdot2*thdot1*np.sin(th2) + (m1*lc1 + m2*l1)*g*np.cos(th1 - np.pi/2.0) + phi2

		ddtheta2 = (torque + d2/d1*phi1 - m2*l1*lc2*thdot1*thdot1*np.sin(th2) - phi2)/(m2*lc2*lc2 + I2 - d2*d2/d1)

		ddtheta1 = -(d2*ddtheta2 + phi1)/d1
		return (thdot1, thdot2, ddtheta1, ddtheta2, 0., 0., 0.)

	def abstract_actions(self, a):
		a[a>=0] = 1.0
		a[a<0] = -1.0
		return a 

	def check_safe(self, state):
		return 0 # no safety property

	def check_goal(self, state):
		th1, th2,_,_, _, _ = state
		a = -np.cos(th1) - np.cos(th1 + th2)
		if (a < 1.0):
			return [(1.0 - a)*1.0]
		else:
			return [0.0]

	def check_time(self, total_time):
		return 0.0

	def get_obj(self, state):
		return 0.0

	def get_act_features(self, state):
		return [] # const action

	def get_features(self, state):
		features = []
		th1, th2, thdot1, thdot2, m1, m2 = state
		#features.append(np.cos(th1)*5.0)
		#features.append(np.sin(th1)*5.0)
		#features.append(np.cos(th2)*5.0)
		#features.append(np.sin(th2)*5.0)
		features.append(th1*5.0)
		features.append(th2*5.0)
		features.append(np.cos(th1) + np.cos(th1 + th2))
		features.append(thdot1)
		features.append(thdot2)
		return features

	def done(self, state):
		return self.counter >= self.n_steps 

	def sample_init_state(self):
		th1 = 0.0 + rand(-0.04, 0.04)
		thdot1 = 0.0 + rand(-0.04, 0.04)
		th2 = 0.0 + rand(-0.04, 0.04)
		thdot2 = 0.0 + rand(-0.04, 0.04)
		m1 = rand(self.mass_low, self.mass_high)
		m2 = rand(self.mass_low, self.mass_high)

		return np.array([th1, th2, thdot1, thdot2, m1, m2])

	def get_neutral_state(self):
		th1 = 0.0 
		thdot1 = 0.0 
		th2 = 0.0
		thdot2 = 0.0 
		m1 = 1.0
		m2 = 1.0
		return np.array([th1, th2, thdot1, thdot2, m1, m2])

	def render(self, state, mode='human'):
		self.env.env.state = state[0:4]
		self.env.render()

	def reset(self):
		self.env.close()
		self.counter = 0

	def get_plot_limits(self):
		return (-3.5, 3.5), (-3.5, 3.5)

	def plot_init(self, state):
		pl.xlim((-3.5, 3.5))
		pl.ylim((-3.5, 3.5))

		X = []
		Y = []
		for th1 in np.arange(-4, 8, 0.1):
			for th2 in np.arange(-4, 8, 0.1):
				if -np.cos(th1) - np.cos(th1 + th2) >= 1.0:
					X.append(th1)
					Y.append(th2)


		#plot goal 
		pl.scatter(X, Y, c = "b", alpha = 0.1, s = 1.0)
	

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
		pl.plot(X,Y, "k--", alpha=0.2)

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

def wrap(x, m, M):
	diff = M - m
	while x > M:
		x = x - diff
	while x < m:
		x = x + diff
	return x




def rk4(derivs, y0, t, *args, **kwargs):
	"""
	Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
	This is a toy implementation which may be useful if you find
	yourself stranded on a system w/o scipy.  Otherwise use
	:func:`scipy.integrate`.
	*y0*
		initial state vector
	*t*
		sample times
	*derivs*
		returns the derivative of the system and has the
		signature ``dy = derivs(yi, ti)``
	*args*
		additional arguments passed to the derivative function
	*kwargs*
		additional keyword arguments passed to the derivative function
	Example 1 ::
		## 2D system
		def derivs6(x,t):
			d1 =  x[0] + 2*x[1]
			d2 =  -3*x[0] + 4*x[1]
			return (d1, d2)
		dt = 0.0005
		t = arange(0.0, 2.0, dt)
		y0 = (1,2)
		yout = rk4(derivs6, y0, t)
	Example 2::
		## 1D system
		alpha = 2
		def derivs(x,t):
			return -alpha*x + exp(-t)
		y0 = 1
		yout = rk4(derivs, y0, t)
	If you have access to scipy, you should probably be using the
	scipy.integrate tools rather than this function.
	"""

	try:
		Ny = len(y0)
	except TypeError:
		yout = np.zeros((len(t),), np.float_)
	else:
		yout = np.zeros((len(t), Ny), np.float_)

	yout[0] = y0


	for i in np.arange(len(t) - 1):

		thist = t[i]
		dt = t[i + 1] - thist
		dt2 = dt / 2.0
		y0 = yout[i]

		k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
		k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
		k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
		k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
		yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
	return yout

