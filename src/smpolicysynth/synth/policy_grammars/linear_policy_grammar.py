import sys
from general.utils import *
from synth.policy.linear_policy import *
from synth.policy.state_machine import * 

class LinearPolicyGrammar:
	def __init__(self, env, num_ex, timesteps):
		self.env = env
		self.num_ex = num_ex
		self.timesteps = timesteps

		self.num_actions = self.env.num_actions
		self.num_act_features = self.env.num_act_features

		# ranges for each variable
		self.x_low = []
		self.x_high = []

		# actions for each mode
		for i in range(1):
			for j in range(self.num_actions):
				for k in range(self.num_act_features):
					self.x_low.append(-20)
					self.x_high.append(20)

		self.x_low = np.array(self.x_low)
		self.x_high = np.array(self.x_high)

	def get_random_policy_vars(self):
		states = []
		for i in range(len(self.x_low)):
			states.append(rand(self.x_low[i], self.x_high[i]))
		
		return np.array(states)

	def get_mode_dt_indices(self):
		ex = self.num_ex
		na = self.num_actions * self.num_act_features

		i_min = 0
		i_max = na 
		return np.arange(i_min, i_max, 1)

	def get_random_mode_dt_vars(self):
		states = []
		ind = self.get_mode_dt_indices()

		for i in ind:
			states.append(rand(self.x_low[i], self.x_high[i]))
		return np.array(states) 

	def get_random_vars_subset(self, sub_indices):
		states = []
		for i in sub_indices:
			states.append(rand(self.x_low[i], self.x_high[i]))
		return np.array(states)

	def bound(self, x, sub_indices):
		assert(len(sub_indices) == len(x))
		return np.clip(x, self.x_low.take(sub_indices), self.x_high.take(sub_indices))

	def get_bounds(self, sub_indices):
		return self.x_low.take(sub_indices), self.x_high.take(sub_indices)


	def get_policy(self, states):
		ex = self.num_ex
		na = self.num_actions
		naf = self.num_act_features

		states = np.clip(states, self.x_low, self.x_high)

		actions = states[0:na*naf].reshape((na, naf))
		
		policies = []
		for i in range(ex):
			policy = LinearPolicy(self.env, actions, self.timesteps)
			policies.append(policy)
		return policies

	