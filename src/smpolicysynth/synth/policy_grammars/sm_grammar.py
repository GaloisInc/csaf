import sys
from general.utils import *
from synth.policy.state_machine import * 
from synth.policy.condition import * 
from synth.policy_grammars.bool_cond_d1_grammar import * 

class SMGrammar:
	def __init__(self, env, num_modes, cond_grammar, nm_unroll, timesteps):
		self.env = env
		self.num_modes = num_modes
		self.cond_grammar = cond_grammar
		self.nm_unroll = nm_unroll
		self.timesteps = timesteps
		
		self.num_actions = self.env.num_actions
		self.num_act_features = self.env.num_act_features


		# ranges for each variable
		self.x_low = []
		self.x_high = []

		# actions for each mode
		for i in range(num_modes):
			for j in range(self.num_actions):
				for k in range(self.num_act_features):
					self.x_low.append(-5)
					self.x_high.append(5)
				

		# conditions between every pair of modes
		for i in range(num_modes):
			for j in range(num_modes - 1):
				self.x_low.extend(self.cond_grammar.x_low)
				self.x_high.extend(self.cond_grammar.x_high)

		# start conds 
		for i in range(num_modes):
			self.x_low.extend(self.cond_grammar.x_low)
			self.x_high.extend(self.cond_grammar.x_high)

		if not env.infinite_system:
			# end conds
			for i in range(num_modes):
				self.x_low.extend(self.cond_grammar.x_low)
				self.x_high.extend(self.cond_grammar.x_high)
	

		self.x_low = np.array(self.x_low)
		self.x_high = np.array(self.x_high)


	def get_random_policy_vars(self):
		states = []
		for i in range(len(self.x_low)):
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

		cg_len = self.cond_grammar.len

		states = np.clip(states, self.x_low, self.x_high)

		modes = states[0:nm*na*naf].reshape((nm, na, naf))
		states = states[nm*na*naf:]

		conds_vars = states[0:nm*(nm-1)*cg_len].reshape((nm, nm - 1, cg_len))
		states = states[nm*(nm-1)*cg_len:]

		start_cond_vars = states[0:nm*cg_len].reshape((nm, cg_len))
		states = states[nm*cg_len:]

		end_cond_vars = []

		if not self.env.infinite_system:
			end_cond_vars = states[0:nm*cg_len].reshape((nm, cg_len))
			states = states[nm*cg_len:]

		assert(len(states) == 0)

		conds = {} 
		self.grammar_error = 0.0

		for i in range(nm):
			m1 = i
			c = {}
			for j in range(nm -1):
				m2 = j 
				if m2 >= i:
					m2 = m2 +1

				cond, cond_err = self.cond_grammar.get_cond(conds_vars[i][j])
				c[m2] = cond
				self.grammar_error += cond_err
				
			conds[m1] = c

		conds[-1] = {}
		for i in range(len(start_cond_vars)):
			m1 = -1
			m2 = i 
			cond, cond_err = self.cond_grammar.get_cond(start_cond_vars[i])
			conds[m1][m2] = cond 
			self.grammar_error += cond_err 

		for i in range(len(end_cond_vars)):
			m1 = i 
			m2 = -2 
			cond, cond_err = self.cond_grammar.get_cond(end_cond_vars[i])
			conds[m1][m2] = cond 
			self.grammar_error += cond_err 


		policy = StateMachinePolicy(self.env, modes, conds)
		return [policy]

	

	