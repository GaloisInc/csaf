import numpy as np 

from environments.mountain_car.mountain_car import * 
from general.simulate import * 

class Policy:
	def __init__(self, env):
		self.cur_mode = 0
		self.env = env 
		self.num_changes = 0
		self.prev_mode = 0 
		self.time = 0

	def reset(self, init_state):
		self.cur_mode = 0
		self.num_changes = 0 
		self.prev_mode = 0
		self.time = 0

	def get_action(self, state):
		p, v, power = state

		if self.cur_mode == 0 and v < 0.0:
			self.cur_mode = 1

		if self.cur_mode == 1 and v > 0.0:
			self.cur_mode = 0

		if self.cur_mode != self.prev_mode:
			self.num_changes += 1 
			print(self.time)
			self.time = 0

		self.prev_mode = self.cur_mode 
		self.time += 1

		if p > self.env.goal_position:
			print(self.time)
			return []
		
		if self.cur_mode == 0:
			return np.array([5.0])
		
		if self.cur_mode == 1:
			return np.array([-5.0])

if __name__ == "__main__":
	env = MountainCar(2000)
	init_state = env.sample_init_state()
	policy = Policy(env)
	plot_traj(env, policy, init_state)
