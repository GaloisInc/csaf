import sys
from general.utils import *
from synth.policy.condition import * 

class BoolCondD1Grammar:
	def __init__(self, env):
		self.env = env
	
		self.num_cond_features = self.env.num_cond_features

		# ranges for each variable
		self.x_low = []
		self.x_high = []

		# sign 
		self.x_low.append(-1)
		self.x_high.append(1)
		# feature 
		for i in range(self.num_cond_features - 1):
			self.x_low.append(0.0)
			self.x_high.append(1.0)

		# threshold 
		self.x_low.append(-5)
		self.x_high.append(5) 

		self.x_low = np.array(self.x_low)
		self.x_high = np.array(self.x_high)

		self.len = len(self.x_low)


	def get_random_policy_vars(self):
		states = []
		for i in range(len(self.x_low)):
			states.append(rand(self.x_low[i], self.x_high[i]))
		
		return np.array(states)


	def get_cond(self, state):
		nc = self.num_cond_features

		sign = state[0]
		features = state[1:nc]
		threshold = state[nc]

		cond_err = 0.0

		assert(len(state) == nc + 1)
		total_f_sum = 0.0
		cond_p = []
		for i in range(nc - 1):
			f = features[i]
			cond_p.append(f*sign)
			total_f_sum += f

			cond_err += min(f, 1.0 - f)

		cond_err += abs(total_f_sum - 1.0)
		
		cond_p.append(-threshold)

		cond_p = np.array(cond_p)
		return LinearCond(cond_p), cond_err


