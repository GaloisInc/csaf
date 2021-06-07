import numpy as np
import sys 
import copy
import matplotlib.pyplot as pl 

from synth.policy.condition import *

class ProbStateMachinePolicy:
	def __init__(self, env, modes, modes_std, conds, conds_std):
		self.modes = modes # idx to action parameters
		self.conds = conds # start idx to end idx to cond parameters

		self.modes_std = modes_std 
		self.conds_std = conds_std

		self.cur_mode_idx = -1
		self.env = env
		self.old_action_noises = [] 
		self.old_cond_noises = {}
		self.action_noises = []
		self.cond_noises = {}# start idx to end idx to cond noises

		self.sample_noises()

	def reset(self, init_state):
		self.cur_mode_idx = -1
		self.sample_noises()

	def sample_noises(self):
		self.old_action_noises = np.copy(self.action_noises)
		self.old_cond_noises = copy.deepcopy(self.cond_noises)


		self.action_noises = []

		for i in range(len(self.modes)):
			noises = [] 
			for j in range(len(self.modes[0])):
				std = np.sqrt(self.modes_std[i][j])
				n = np.random.normal(0, std) 

				noises.append(n)
			self.action_noises.append(noises)

		#print("Action noises: ", self.action_noises)

		self.cond_noises = {}
		for v in self.conds:
			edges = self.conds[v]
			noises = {}
			for v1 in edges:
				std = np.sqrt(self.conds_std[v][v1])
				noises[v1] = np.random.normal(0, std)
			self.cond_noises[v] = noises 

		#print("Cond noises: ", self.cond_noises)

	def get_gaussian_prob(self, x, m, sigma):
		p = 1.0
		sigma = max(sigma, 0.2)
		p *= 1.0/np.sqrt(2*np.pi*sigma)*np.exp(- ((x - m)*(x-m))/(2.0 * sigma) )
		assert (0.0 <= p <= 1.0)
		return p

	def get_prob_for_mode(self, mode_idx):
		prob = 1.0 

		if mode_idx == -2:
			#print("prob for mode %i = %f"%(mode_idx, 1.0))
			return 1.0

		# prob for action
		if mode_idx >= 0:
			for j in range(len(self.modes[mode_idx])):
				x = self.action_noises[mode_idx][j]
				m = 0
				std = self.modes_std[mode_idx][j]
				prob *= self.get_gaussian_prob(x, m, std)


		# prob for conds 
		edges = self.conds[mode_idx]
		for v1 in edges:
			x = self.cond_noises[mode_idx][v1]
			m = 0
			std = self.conds_std[mode_idx][v1]
			prob *= self.get_gaussian_prob(x, m, std)

		#print("prob for mode %i = %f"%(mode_idx, prob))
		return prob


	# select the most positive cond 
	def get_closest_cond(self, state, cur_id):
		features = self.env.get_features(state)

		v = cur_id 
		max_v1 = -1
		max_dist = -1e20 

		edges = self.conds[v] 
		for v1 in edges:
			cond = edges[v1]
			dist = cond.eval(features) + self.cond_noises[v][v1]
			if dist > max_dist:
				max_dist = dist 
				max_v1 = v1 

		return max_v1, max_dist 

	def get_action(self, state):
		if self.cur_mode_idx == -2:
			return []

		features = self.env.get_features(state)

		v = self.cur_mode_idx
		v1, dist = self.get_closest_cond(state, v)
		if (dist >= 0 or self.cur_mode_idx == -1):
			#print("Switching from", v, v1)
			self.cur_mode_idx = v1
			self.sample_noises()


		return self.get_action_for_mode(self.cur_mode_idx, state)


	def get_action_for_mode(self, mode_idx, state):
		if mode_idx < 0:
			return []

		act_weights = self.modes[mode_idx]
		
		act_features = self.env.get_act_features(state)
		acts = []
		for i in range(len(act_weights)):
			aw = act_weights[i]
			assert(len(aw) == len(act_features) + 1)
			act = 0.0
			for j in range(len(act_features)):
				act += aw[j] * act_features[j]
			act += aw[-1]
			act += self.action_noises[mode_idx][i]
			acts.append(act)

		return acts


	def save(self, outname):
		file = open(outname, 'w')

		file.write(str(self.modes.tolist()))
		file.write("\n")

		file.write(str(self.modes_std.tolist()))
		file.write("\n")
		
		cond_str = {}
		for v in self.conds:
			c = {}
			for v1 in self.conds[v]:
				c[v1] = str(self.conds[v][v1])
			cond_str[v] = c
		file.write(str(cond_str))
		file.write("\n")

		cond_std_str = {}
		for v in self.conds_std:
			c = {}
			for v1 in self.conds_std[v]:
				c[v1] = str(self.conds_std[v][v1])
			cond_std_str[v] = c
		file.write(str(cond_std_str))
		file.write("\n")

		file.close()

	def read(self, filename):
		file = open(filename, 'r')
		lines = file.readlines()

		self.modes = np.array(eval(lines[0]))
		
		self.modes_std = np.array(eval(lines[1]))

		conds_str = eval(lines[2])
		self.conds = {}
		for v in conds_str:
			conds = {}
			for v1 in conds_str[v]:
				conds[v1] = eval(conds_str[v][v1])
			self.conds[v] = conds 

		conds_std_str = eval(lines[3])
		self.conds_std = {}
		for v in conds_std_str:
			conds_std = {}
			for v1 in conds_std_str[v]:
				conds_std[v1] = eval(conds_std_str[v][v1])
			self.conds_std[v] = conds_std 

		self.sample_noises()
		file.close()

	
	def get_traj_from_sm(self, env, init_state, max_modes = 10, max_time_per_mode = 40, max_timesteps = 5000):
		states = []

		env.reset()
		state = init_state
		done = False
		self.reset(init_state)
		total_goal_error = 0.0
		total_safe_error = 0.0
		total_obj = 0.0

		prev_mode_idx = self.cur_mode_idx
		mode_change_states = []

		time_in_mode = 0

		actions = []
		times = []
		conds = []
		cond_noises = [] 

		traj_prob = 1.0 
		traj_prob *= self.get_prob_for_mode(self.cur_mode_idx)

		total_time = 0.0 

		for i in range(max_timesteps):
			
			features = env.get_features(state)

			if time_in_mode >= max_time_per_mode:
				next_mode, dist = self.get_closest_cond(state, self.cur_mode_idx)
				#print("Force switch from", self.cur_mode_idx, next_mode)
				self.cur_mode_idx = next_mode
				self.sample_noises()

				mode_change_states.append(state)
				if prev_mode_idx >= 0:
					acts = np.copy(self.modes[prev_mode_idx])
					for a in range(len(acts)):
						acts[a][-1] += self.old_action_noises[prev_mode_idx][a]
					actions.append(acts)
					times.append(time_in_mode)
					conds.append(self.conds[prev_mode_idx][self.cur_mode_idx])
					cond_noises.append(self.old_cond_noises[prev_mode_idx][self.cur_mode_idx])

				prev_mode_idx = self.cur_mode_idx
				time_in_mode = 0

				traj_prob *= self.get_prob_for_mode(self.cur_mode_idx)

			
			action = self.get_action(state)
			states.append((state, action))

			if prev_mode_idx != self.cur_mode_idx:
				mode_change_states.append(state)
				if prev_mode_idx >= 0:
					acts = np.copy(self.modes[prev_mode_idx])
					for a in range(len(acts)):
						acts[a][-1] += self.old_action_noises[prev_mode_idx][a]
					actions.append(acts)
					times.append(time_in_mode)
					conds.append(self.conds[prev_mode_idx][self.cur_mode_idx])
					cond_noises.append(self.old_cond_noises[prev_mode_idx][self.cur_mode_idx])

				prev_mode_idx = self.cur_mode_idx
				time_in_mode = 1

				traj_prob *= self.get_prob_for_mode(self.cur_mode_idx)
			else:
				time_in_mode += 1

			if len(times) >= max_modes:
				action = []


			if len(action) == 0:
				done = True
				break
	 
			next_state, safe_error, goal_error, done = env.step(state, action)
			total_obj += env.get_obj(next_state)
			
			if env.infinite_system: 
				# stop if the safety prop is violated
				if safe_error > 0.05:
					#total_safe_error += env.desired_duration - total_time
					total_safe_error += safe_error 
					done = True 
			else:
				total_safe_error += safe_error

			total_time += env.dt 
			if env.infinite_system:
				if total_time > env.desired_duration:
					done = True 
			if not env.infinite_system and np.sum(goal_error) < 0.01:
				done = True
		   
			state = next_state
			if done:
				mode_change_states.append(state)
				if len(action) > 0:
					acts = np.copy(self.modes[prev_mode_idx])
					for a in range(len(acts)):
						acts[a][-1] += self.action_noises[prev_mode_idx][a]
					actions.append(acts)
					times.append(time_in_mode)
					conds.append(None)
					cond_noises.append(0)
				break
		states.append((state, []))
		total_goal_error = env.check_goal(state)
		total_goal_error.append(env.check_time(total_time))

		traj_prob = traj_prob ** (1.0/float(len(actions) + 1))

		return states, total_safe_error, total_goal_error, total_time, total_obj,  mode_change_states, actions, times, conds, cond_noises, traj_prob

	def evaluate(self, init_state, max_modes = 10, max_time_per_mode = 40, vis = False):
		states, safe_err, goal_err, total_time, total_obj, mode_change_states, actions, times, conds, cond_noises, traj_prob = self.get_traj_from_sm(self.env, init_state, max_modes = max_modes, max_time_per_mode = max_time_per_mode, max_timesteps = 50000)

		if (vis):
			self.env.plot_init(states[0])
			self.env.plot_states(states)
			self.env.plot_mode_changes(mode_change_states)
		
		return safe_err, goal_err, total_time, total_obj, actions, times, conds, cond_noises, traj_prob
	