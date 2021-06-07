import numpy as np 
from environments.pendulum.pendulum import * 
from general.simulate import *

class PendulumPolicy:
	def __init__(self, env):
		self.cur_mode = 0

	def reset(self, init_state):
		self.cur_mode = 0

	def get_action(self, state):
		th, thdot, m = state

		if self.cur_mode == 0 and thdot < 0.0:
			self.cur_mode = 1

		if self.cur_mode == 1 and thdot > 0.0:
			self.cur_mode = 0

		if self.cur_mode == 0 and  (abs(th - 0.0) < 1.4 or abs(th - 2*np.pi - 0.0) < 1.4) and abs(thdot) > 4.0:
			self.cur_mode = 2

		if self.cur_mode == 1 and  (abs(th - 0.0) < 1.4 or abs(th - 2*np.pi - 0.0) < 1.4) and abs(thdot) > 4.0:
			self.cur_mode = 3


		if (self.cur_mode == 2 or self.cur_mode == 3) and abs(thdot) < 0.05:
			return np.array([])
		
		if self.cur_mode == 0:
			return np.array([3.0])
		
		if self.cur_mode == 1:
			return np.array([-3.0])


		if self.cur_mode == 2:
			return np.array([-5.0])
		
		if self.cur_mode == 3:
			return np.array([5.0])

if __name__ == "__main__":
	env = Pendulum(2000)
	init_state = env.sample_init_state()
	policy = PendulumPolicy(env)
	plot_traj(env, policy, init_state)




