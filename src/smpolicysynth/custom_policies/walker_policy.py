import numpy as np 

class Policy:
	def __init__(self, env):
		self.cur_mode = 0

	def reset(self, init_state):
		self.cur_mode = 0

	def get_action(self, state):
		return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])




