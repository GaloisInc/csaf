import sys
from general.utils import *
from synth.policy.straight_traj_pomdp import *
from synth.policy.state_machine import * 

class StraightTrajPOMDPGrammar:
	def __init__(self, env, num_modes, max_timesteps):
		self.env = env
		self.num_modes = num_modes
		self.max_timesteps = max_timesteps

		self.num_actions = self.env.num_actions
		self.num_act_features = self.env.num_act_features
		self.num_decode_params = self.env.num_decode_params

		# ranges for each variable
		self.x_low = []
		self.x_high = []

		# actions for each mode
		for i in range(num_modes):
			for j in range(self.num_actions):
				for k in range(self.num_act_features):
					self.x_low.append(-5)
					self.x_high.append(5)
				

		# dts for modes
		for i in range(num_modes):
			self.x_low.append(0)
			self.x_high.append(10)

		# decode params
		for i in range(self.num_decode_params):
			self.x_low.append(-5)
			self.x_high.append(5)

		self.x_low = np.array(self.x_low)
		self.x_high = np.array(self.x_high)


		self.ref_trajs = []
		self.ref_prob = []
		

	def set_ref_trajs(self, ref_trajs):
		self.ref_trajs = ref_trajs 


	def set_ref_prob(self, prob):
		self.ref_prob = prob

	def get_random_policy_vars(self):
		states = []
		for i in range(len(self.x_low)):
			states.append(rand(self.x_low[i], self.x_high[i]))
		
		return np.array(states)

	def get_mode_dt_indices(self):
		nm = self.num_modes
		na = self.num_actions * self.num_act_features

		i_min = 0
		i_max = nm*na + nm
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
		nm = self.num_modes
		na = self.num_actions
		naf = self.num_act_features
		nd = self.num_decode_params

		states = np.clip(states, self.x_low, self.x_high)

		modes = states[0:nm*na*naf].reshape((nm, na, naf))
		states = states[nm*na*naf:]


		dts = states[0:(nm)].reshape((1, (nm)))
		dts = dts/100.0

		states = states[nm:]

		decode_params = states
		assert(len(decode_params) == nd)



		policy = StraightTrajPOMDP(self.env, modes, self.max_timesteps, dts[0], decode_params, self.ref_trajs, self.ref_prob)
		return [policy]

	

	