import sys

import numpy as np 
import matplotlib.pyplot as pl 

class OptWrapper:
	# env - dynamical system
	def __init__(self, env, policy_grammar):
		self.env = env
		self.policy_grammar = policy_grammar
		self.init_states = [] # states used to estimate the cost function
		
		self.num_init_states = 1 # number of states used to estimate the cost function 
		self.sample_env()

	def get_random_x(self):
		# Try different random x and choose the one with the lowest cost 
		min_err = 1e10
		min_x = None
		num_tries = 10
		for i in range(num_tries):
			x = self.policy_grammar.get_random_policy_vars()
			err, _ = self.get_cost(x, False)
			#print("Err:", err)
			if err < min_err:
				min_err = err
				min_x = x
			if min_err < 0.01:
				break
		#print("Err: ", min_err)
		return min_x

	def bound(self, x):
		return self.policy_grammar.bound(x)

	def get_bounds(self):
		return self.policy_grammar.get_bounds()

	def sample_env(self):
		self.init_states = []
		for i in range(self.num_init_states):
			self.init_states.append(self.env.sample_init_state())

	def get_policy(self, x):
		return self.policy_grammar.get_policy(x)

	def get_cost(self, x, vis = False):		
		policies = self.policy_grammar.get_policy(x)

		err = 0.0
		errors = []

		for i in range(self.num_init_states):
			e = 0.0
			init_state = self.init_states[i]
			safe_err, goal_err, total_time, total_obj = policies[i].evaluate(init_state, vis)
				
			e += safe_err + np.sum(goal_err) 
			if self.env.infinite_system:
				# maximize time
				e +=  -total_time*self.env.time_weight 
			else:
				# minimize time
				e += total_time * self.env.time_weight
			# minimize obj 
			e += total_obj
			errors.append(safe_err)
			goal_err = np.array(goal_err)
			errors.extend(goal_err)
			
			err += e 

		err = err/float(self.num_init_states)		
		return err, errors


