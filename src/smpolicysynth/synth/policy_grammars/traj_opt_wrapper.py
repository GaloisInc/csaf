import sys

from enum import Enum
import numpy as np 
import matplotlib.pyplot as pl 

class VarsMode(Enum):
	Mode = 1
	Dt = 2
	All = 3
	Subset = 4

class ErrMode(Enum):
	Safe = 1
	Goal = 2
	SafeGoal = 3
	Cond = 4
	All = 5 


class TrapOptWrapper:
	# env - dynamical system
	def __init__(self, env, policy_grammar, num_ex):
		self.env = env
		self.policy_grammar = policy_grammar

		self.init_states = [] # states used to estimate the cost function
		self.full_x = [] # full list of optimization variables
		self.sub_indices = [] # indices of full_x that are currently optimized over

		self.vars_mode = VarsMode.All 
		self.err_mode = ErrMode.SafeGoal

		
		self.num_init_states = num_ex # number of states used to estimate the cost function # TODO: currently we can only do 1 with policy time grammar
		self.err_params = 1.0

		self.counter = 0
		
		self.sample_env()

	def init_full_policy(self):
		self.full_x = self.policy_grammar.get_random_policy_vars()
		self.sub_indices = np.arange(0, len(self.full_x), 1)

	def set_opt_mode(self, vars_mode, err_mode, vars_params, err_params):
		self.err_params = err_params
		self.vars_mode = vars_mode
		self.err_mode = err_mode

		if vars_mode == VarsMode.Mode: 
			# Only optimize over mode variables 
			self.sub_indices = self.policy_grammar.get_mode_indices()

		elif vars_mode == VarsMode.Dt:
			self.sub_indices = self.policy_grammar.get_dt_indices()

		elif vars_mode == VarsMode.All:
			# Optimize over all variables
			self.sub_indices = np.arange(0, len(self.full_x), 1)

		elif vars_mode == VarsMode.Subset:
			self.sub_indices = vars_params


		else:
			print("Unrecognized vars mode")
			assert(False)

	def set_opt_mode1(self, vars_mode):
		self.vars_mode = vars_mode

		if vars_mode == VarsMode.Mode: 
			# Only optimize over mode variables 
			self.sub_indices = self.policy_grammar.get_mode_indices()

		elif vars_mode == VarsMode.Dt:
			self.sub_indices = self.policy_grammar.get_dt_indices()

		elif vars_mode == VarsMode.All:
			# Optimize over all variables
			self.sub_indices = np.arange(0, len(self.full_x), 1)

		elif vars_mode == VarsMode.Subset:
			self.sub_indices = vars_params


		else:
			print("Unrecognized vars mode")
			assert(False)


	def get_random_x_for_vars_mode(self):
		vars_mode = self.vars_mode 

		if vars_mode == VarsMode.Mode:
			return self.policy_grammar.get_random_mode_vars()
		elif vars_mode == VarsMode.Dt:
			return self.policy_grammar.get_random_dt_vars()
		elif vars_mode == VarsMode.All:
			return self.policy_grammar.get_random_policy_vars()
		elif vars_mode == VarsMode.Subset:
			return self.policy_grammar.get_random_vars_subset(self.sub_indices)
		else:
			print("Unrecognized vars mode")
			assert(False)

	def get_random_x(self):
		# Try different random x and choose the one with the lowest cost 
		min_err = 1e10
		min_x = None
		num_tries = 10
		for i in range(num_tries):
			x = self.get_random_x_for_vars_mode()
			err, _ = self.get_cost(x, False)
			#print("Err:", err)
			if err < min_err:
				min_err = err
				min_x = x
			if min_err < 0.01:
				break
		#print("Err: ", min_err)
		return min_x

	def get_current_x(self):
		return np.copy(self.full_x.take(self.sub_indices))


	def bound(self, x):
		return self.policy_grammar.bound(x, self.sub_indices)

	def get_bounds(self):
		return self.policy_grammar.get_bounds(self.sub_indices)

	def sample_env(self):
		self.init_states = []
		for i in range(self.num_init_states):
			self.init_states.append(self.env.sample_init_state())


	def get_full_x(self, x):
		assert(len(self.sub_indices) == len(x))
		full_x = np.copy(self.full_x)
		np.put(full_x, self.sub_indices, x)
		return full_x

	def set_full_x(self, x):
		assert(len(self.sub_indices) == len(x))
		np.put(self.full_x, self.sub_indices, x)


	def get_policy(self):
		return self.policy_grammar.get_policy(self.full_x)

	def get_cost(self, x, vis = False):
		#print(self.counter)
		self.counter += 1
		full_x = np.copy(self.full_x)
		assert(len(self.sub_indices) == len(x))
		np.put(full_x, self.sub_indices, x)
		policies = self.policy_grammar.get_policy(full_x)

		err = 0.0
		errors = []

		for i in range(self.num_init_states):
			e = 0.0
			init_state = self.init_states[i]
			safe_err, goal_err, mode_err, cond_err, total_time, total_obj = policies[i].evaluate(init_state, vis)

			if vis:
				print(safe_err, goal_err, cond_err)

			if self.err_mode == ErrMode.Safe:
				e += safe_err
				errors.append(safe_err)

			elif self.err_mode == ErrMode.Goal:
				e += np.sum(goal_err)
				errors.extend(goal_err)

			elif self.err_mode == ErrMode.SafeGoal:
				e += safe_err + np.sum(goal_err)
				errors.append(safe_err)
				errors.extend(goal_err)

			elif self.err_mode == ErrMode.Cond:
				for c in self.err_params:
					e += cond_err[c]
					errors.append(cond_err[c])

			elif self.err_mode == ErrMode.All:
				[w1, w2, w3, w4] = self.err_params
				e += w1*safe_err + w2*np.sum(goal_err) + w3*mode_err + w4*cond_err
				if self.env.infinite_system:
					# maximize time
					e +=  -total_time*self.env.time_weight 
				else:
					# minimize time
					e += total_time * self.env.time_weight
				# minimize obj 
				e += total_obj
				# hard constraints
				#errors.append(safe_err*w1)
				#goal_err = np.array(goal_err)*w2
				#errors.extend(goal_err)
				#errors.append(mode_err*w3)
				#errors.append(cond_err*w4)

			else:
				print("Unrecognized goal mode")
				assert(False)
			err += e 

		#if vis:
			#pl.show()
			#pl.close()

		err = err/float(self.num_init_states)

		#print("Err: ", err)
		
		return err, errors


