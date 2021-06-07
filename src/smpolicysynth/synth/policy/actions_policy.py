import random
import numpy as np
import sys
from general.utils import *

import matplotlib.pyplot as pl 

# A list of actions to execute at each timestep
class ActionsPolicy:
	# Initialize the policy.
	def __init__(self, env, actions):
		self.actions = actions 
		self.cur_id = -1

		old_action = actions[0]
		time = 0 
		for i in range(1, len(actions)):
			diff = np.linalg.norm(np.array(actions[i]) - np.array(old_action))
			if diff < 0.01:
				time += 1
			else:
				print(old_action, time)
				time = 0

			old_action = actions[i]
		print(old_action, time)



	# Get a action to take.
	#
	# state: np.array([state_dim])
	def get_action(self, state):
		nm = len(self.actions)

		self.cur_id += 1 
		if self.cur_id < nm:
			actions =  self.actions[self.cur_id]
			#if actions[2] < 0.0:
			#	return []
			return actions
		else:
			return [] 

	def reset(self, init_state):
		self.cur_id = -1

	def save(self, outname):
		file = open(outname, 'w')
		file.write("(")
		file.write(str(self.actions.tolist()))
		file.write(")\n")
		file.close()
	

	def get_traj_from_k_mode_policy(self, env, init_state, vis = False):
		states = []

		env.reset()
		state = init_state
		done = False
		self.reset(init_state)
		total_goal_error = 0.0
		total_safe_error = 0.0
		total_obj = 0.0

		prev_idx = self.cur_id
		total_time = 0.0


		while True:
			action = self.get_action(state)
			states.append((state, action))
			
			
			if len(action) == 0:
				done = True
				break
	 
			next_state, safe_error, goal_error, done = env.step(state, action, -1)
			total_safe_error += safe_error
			total_obj += env.get_obj(next_state)
			if env.infinite_system: 
				# stop if the safety prop is violated
				if safe_error > 0.05:
					#total_safe_error += env.desired_duration - total_time 
					break 
			total_time += env.dt 
			if env.infinite_system:
				if total_time > env.desired_duration:
					break 

			state = next_state
		
		states.append((state,[]))
		total_goal_error = env.check_goal(state)
		total_goal_error.append(env.check_time(total_time))

		
		
		return states, total_safe_error, total_goal_error, total_time, total_obj

	def evaluate(self, init_state, vis = False, id = 0):
		states, safe_err, goal_err, total_time, total_obj = self.get_traj_from_k_mode_policy(self.env, init_state, vis)

		if vis or False:
			print(total_time)
			file = open("traj_%i.txt"%id, 'w')
			for s,a in states:
				file.write(str(s.tolist()))
				file.write("\n")
			file.close()

		if (vis):
			self.env.plot_init(states[0])
			self.env.plot_states(states)
			self.env.plot_mode_changes(mode_change_states)
			
		return safe_err, goal_err, mode_err, cond_err, total_time, total_obj


	
