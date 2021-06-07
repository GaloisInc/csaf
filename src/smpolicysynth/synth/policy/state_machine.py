import numpy as np
import sys 
import matplotlib.pyplot as pl 

from synth.policy.condition import *



class StateMachinePolicy:
	def __init__(self, env, modes, conds):
		self.modes = modes # idx to action parameters
		self.conds = conds # start idx to end idx to cond parameters

		self.cur_mode_idx = -1
		self.env = env

		self.num_mode_changes = 0

	def reset(self, init_state):
		self.cur_mode_idx = -1

	def get_closest_cond(self, state, cur_id):
		features = self.env.get_features(state)

		v = cur_id 
		max_v1 = -2
		max_dist = -1e20 

		edges = self.conds[v] 
		for v1 in edges:
			if self.env.infinite_system and v1 == -2:
				continue
			cond = edges[v1]
			dist = cond.eval(features)
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
			self.num_mode_changes += 1 
			#print("Switching from", v, v1)
			self.cur_mode_idx = v1


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
			acts.append(act)

		return acts


	def save(self, outname):
		file = open(outname, 'w')

		file.write(str(self.modes.tolist()))
		file.write("\n")

		file.write(str([]))
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
		file.write(str(cond_std_str))
		file.write("\n")

		file.close()

	def read(self, filename):
		file = open(filename, 'r')
		lines = file.readlines()

		self.modes = np.array(eval(lines[0]))
		

		conds_str = eval(lines[2])
		self.conds = {}
		for v in conds_str:
			conds = {}
			for v1 in conds_str[v]:
				conds[v1] = eval(conds_str[v][v1])
			self.conds[v] = conds 

		file.close()


	
	def get_traj_from_sm(self, env, init_state, max_modes = 10, max_time_per_mode = 40, max_timesteps = 5000, dt = -1):
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

		total_time = 0.0

		time_safe = 0.0
		safe_sofar = True

		if dt < -0.1:
			dt = env.dt *env.dt_scale


		for i in range(max_timesteps):
			features = env.get_features(state)

			if time_in_mode >= max_time_per_mode:
				next_mode, dist = self.get_closest_cond(state, self.cur_mode_idx)
				#print("Force switch from", self.cur_mode_idx, next_mode)
				self.cur_mode_idx = next_mode

				mode_change_states.append(state)
				if prev_mode_idx >= 0:
					actions.append(self.modes[prev_mode_idx])
					times.append(time_in_mode)
					conds.append(self.conds[prev_mode_idx][self.cur_mode_idx])
					cond_noises.append(0.0)
				prev_mode_idx = self.cur_mode_idx
				time_in_mode = 0

			
			action = self.get_action(state)
			states.append((state,action))
			
			if prev_mode_idx != self.cur_mode_idx:
				mode_change_states.append(state)
				if prev_mode_idx >= 0:
					actions.append(self.modes[prev_mode_idx])
					times.append(time_in_mode)
					conds.append(self.conds[prev_mode_idx][self.cur_mode_idx])
					cond_noises.append(0.0)
				prev_mode_idx = self.cur_mode_idx
				time_in_mode = 1
			else:
				time_in_mode += 1

			if len(times) >= max_modes :
				#print("Breaking because max modes")
				action = []


			if len(action) == 0:
				done = True
				break
	 
			next_state, safe_error, goal_error, done = env.step(state, action, dt)
			total_obj += env.get_obj(next_state)

			if safe_sofar and safe_error > 0.05:
				time_safe = total_time
				safe_sofar = False
			
			#print(next_state, safe_error)
			if env.infinite_system: 
				# stop if the safety prop is violated
				if safe_error > 0.05:
					#total_safe_error += env.desired_duration - total_time 
					#print("Breaking because unsafe")
					total_safe_error += safe_error
					done = True 
			else:
				total_safe_error += safe_error
			
			total_time += dt/env.dt_scale
			if env.infinite_system:
				if total_time > env.desired_duration:
					#print("Breaking because desired duration")
					done = True 

			

			if not env.infinite_system and np.sum(goal_error) < 0.01:
				done = True
		   
			state = next_state
			if done:
				mode_change_states.append(state)
				if len(action) > 0:
					actions.append(self.modes[prev_mode_idx])
					times.append(time_in_mode)
					conds.append(None)
					cond_noises.append(0.0)
				break
		states.append((state, []))
		total_goal_error = env.check_goal(state)
		total_goal_error.append(env.check_time(total_time))
		#print("Total time: ", total_time)

		if safe_sofar:
			time_safe = total_time
			
		return states, total_safe_error, total_goal_error, total_time, time_safe, total_obj, mode_change_states, actions, times, conds, cond_noises

	def evaluate(self, init_state, max_modes = 10, max_time_per_mode = 40, dt = -1, vis = False):
		states, safe_err, goal_err, total_time, time_safe, total_obj, mode_change_states, actions, times, conds, cond_noises = self.get_traj_from_sm(self.env, init_state, max_modes = max_modes, max_time_per_mode = max_time_per_mode, max_timesteps = 50000, dt = dt)

		if (vis):
			self.env.plot_init(states[0])
			self.env.plot_states(states)
			self.env.plot_mode_changes(mode_change_states)

			'''for i in self.conds:
				edges = self.conds[i]
				for j in edges:
					if i == -1: continue
					#if i == 0 and j == 2 or i == 2 and j == 0:
					cond = edges[j]
					X, Y = self.env.get_2d_cond(cond.params)
					pl.scatter(X, Y, s = 1)'''
		
		return safe_err, goal_err, total_time, time_safe, total_obj, actions, times, conds, cond_noises
	