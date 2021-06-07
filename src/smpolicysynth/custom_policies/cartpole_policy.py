import numpy as np 

from environments.cartpole.cartpole import * 
from general.simulate import *

class Policy:
	def __init__(self, env):
		self.cur_mode = 0
		self.num_changes = 0
		self.prev_mode = -1
		self.timesteps = 0

	def reset(self, init_state):
		self.cur_mode = 0
		self.num_changes = 0
		self.prev_mode = -1
		self.timesteps = 0

	def get_action(self, state):
		x,v,t,w = state


		if self.cur_mode == 0 and (t < -0.05 or w < -1.0):
			self.cur_mode = 1

		if self.cur_mode == 1 and (t > 0.05 or w > 1.0):
			self.cur_mode = 0

		if self.prev_mode != self.cur_mode:
			self.num_changes += 1 
			print("Timesteps: ", self.timesteps)
			self.timesteps = 0
		self.timesteps += 1


		self.prev_mode = self.cur_mode

	
		if self.cur_mode == 0:
			return np.array([5.0])
		
		if self.cur_mode == 1:
			return np.array([-5.0])


		if self.cur_mode == 2:
			return np.array([-5.0])
		
		if self.cur_mode == 3:
			return np.array([5.0])

if __name__ == "__main__":
	env = CartPole(2000)
	init_state = env.sample_init_state()
	policy = Policy(env)
	plot_traj(env, policy, init_state)


