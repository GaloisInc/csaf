import sys
from general.utils import *
from synth.policy.condition import * 
from synth.policy_grammars.bool_cond_d1_grammar import *

class BoolCondD2Grammar:
	def __init__(self, env):
		self.env = env
		self.bool1g = BoolCondD1Grammar(env)
	
		#self.num_cond_features = self.env.num_cond_features

		# ranges for each variable
		self.x_low = []
		self.x_high = []

		# leaf 1 
		self.x_low.extend(self.bool1g.x_low)
		self.x_high.extend(self.bool1g.x_high )

		# leaf 2  
		self.x_low.extend(self.bool1g.x_low)
		self.x_high.extend(self.bool1g.x_high )

		# weight for leaf 1  
		self.x_low.append(0.0)
		self.x_high.append(1.0)

		# weight for leaf 2  
		self.x_low.append(0.0)
		self.x_high.append(1.0)

		# weight for leaf 1 and leaf 2 
		self.x_low.append(0.0)
		self.x_high.append(1.0)

		# weight for leaf 1 or leaf 2 
		self.x_low.append(0.0)
		self.x_high.append(1.0)

		self.x_low = np.array(self.x_low)
		self.x_high = np.array(self.x_high)

		self.len = len(self.x_low)


	def get_random_policy_vars(self):
		states = []
		for i in range(len(self.x_low)):
			states.append(rand(self.x_low[i], self.x_high[i]))
		
		return np.array(states)


	def get_cond(self, state):
		leaf_len = self.bool1g.len 
		assert(len(state) == 2*leaf_len + 4) 
		leaf1 = state[0:leaf_len]
		state = state[leaf_len:]

		leaf2 = state[0:leaf_len]
		state = state[leaf_len:] 

		w = state
		assert(len(w) == 4)
		w_sum = 0.0
		cond_err = 0.0
		for i in range(len(w)):
			cond_err += min(w[i], 1.0 - w[i])
			w_sum += w[i]
		cond_err += abs(w_sum - 1.0)

		leaf1_cond, leaf1_err = self.bool1g.get_cond(leaf1)
		leaf2_cond, leaf2_err = self.bool1g.get_cond(leaf2)

		cond_err += leaf1_err
		cond_err += leaf2_err 

		
		return SoftBoolCond2(leaf1_cond, leaf2_cond, w), cond_err


