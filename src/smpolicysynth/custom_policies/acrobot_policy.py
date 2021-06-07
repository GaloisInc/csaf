import numpy as np 

from environments.acrobot.acrobot import * 
from general.simulate import *


class Policy:
	def __init__(self, env):
		self.cur_mode = 0
		self.timesteps = 0
		self.prev_mode = 0
		self.num_changes = 0

	def reset(self, init_state):
		self.cur_mode = 0
		self.timesteps = 0
		self.prev_mode = 0
		self.num_changes

	def get_action(self, state):
		th1, th2, thdot1, thdot2, m1, m2 = state
	

		if self.cur_mode == 0 and thdot1 < 0.0:
			self.cur_mode = 1
			

		if self.cur_mode == 1 and thdot1 > 0.0:
			self.cur_mode = 0

		if self.cur_mode != self.prev_mode:
			print(self.timesteps)
			self.timesteps = 0
			self.num_changes += 1

		self.prev_mode = self.cur_mode 
		self.timesteps += 1
			

		if -np.cos(th1) - np.cos(th1 + th2) > 1.0:
			return np.array([])

		if self.cur_mode == 0:
			return np.array([-5.0])
		
		if self.cur_mode == 1:
			return np.array([5.0])

if __name__ == "__main__":
	env = Acrobot(2000)
	env.set_inp_limits((1, 2))
	init_state = env.sample_init_state()
	policy = Policy(env)
	plot_traj(env, policy, init_state)





