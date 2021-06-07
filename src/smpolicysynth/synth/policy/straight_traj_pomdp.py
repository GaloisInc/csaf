import random
import numpy as np
import sys
from general.utils import *

import matplotlib.pyplot as pl 


# A k mode policy for continuous actions.
class StraightTrajPOMDP:
	# Initialize the policy.
	def __init__(self, env, modes, max_timesteps, dts, decode_params, ref_trajs, ref_prob):
		self.modes = modes
		self.max_timesteps = max_timesteps
		self.dts = dts
		self.decode_params = decode_params 
		self.env = env 
		self.ref_trajs = ref_trajs 
		self.ref_prob = ref_prob 

		self.cur_id = 0
		self.cur_time = 0


	def get_mode_id(self, cur_id):
		nm = len(self.modes)

		if cur_id < nm:
			return cur_id
		else:
			return -3


	# Get a action to take.
	#
	# state: np.array([state_dim])
	def get_action(self, state):
		if (self.cur_time < self.max_timesteps):
			self.cur_time += 1
		else:
			self.cur_time = 1
			self.cur_id += 1
	
		mode_id = self.get_mode_id(self.cur_id)

		if mode_id >= 0:
			dt = self.dts[self.cur_id]

			# compute the action
			act_features = self.env.get_act_features(state)
			acts = []
			act_weights = self.modes[mode_id]
			for i in range(len(act_weights)):
				aw = act_weights[i]
				assert(len(aw) == len(act_features) + 1)
				act = 0.0
				for j in range(len(act_features)):
					act += aw[j] * act_features[j]
				act += aw[-1]
				acts.append(act)

			if self.cur_time != 1:
				acts[1] = 0.0

			return acts, dt
		else:
			return [], -1

	def reset(self, init_state):
		self.cur_id = 0
		self.cur_time = 0

	def save(self, outname):
		file = open(outname, 'w')
		file.write("(")
		file.write(str(self.modes.tolist()))
		file.write(",")
		file.write(str(self.dts.tolist()))
		file.write(")\n")
		file.close()

	# remove zero time modes and modes after quitting
	def cleanup_modes(self, init_state):
		self.env.reset()
		state = init_state
		done = False
		self.reset(init_state)

		prev_idx = -1

		total_time = 0

		new_modes = []
		new_dts = []
		

		while True:	
			action, dt = self.get_action(state)
		
			if self.cur_id != prev_idx:
				if  self.dts[prev_idx] > 0.1/(self.env.dt_scale*100.0):
					if prev_idx >= 0:
						new_modes.append(self.modes[prev_idx])
						new_dts.append(self.dts[prev_idx])

			prev_idx = self.cur_id
		
			if len(action) == 0:
				done = True
				break
			next_state, safe_error, goal_error, done = self.env.step(state, action, dt)
			if self.env.infinite_system and safe_error > 0.05:
				new_modes.append(self.modes[self.cur_id])
				new_dts.append(self.dts[self.cur_id])
				print("Unsafe - Breaking at ", self.cur_id)
				break 

			total_time += dt/self.env.dt_scale
			if self.env.infinite_system and total_time > self.env.desired_duration:
				new_modes.append(self.modes[self.cur_id])
				new_dts.append(self.dts[self.cur_id])
				print("Breaking at ", self.cur_id)
				break 
		   
			state = next_state
			if done:
				break

		#print("Old modes: ", self.modes)
		#print("New modes: ", new_modes)
		self.modes = np.array(new_modes) 

		#print("Old times: ", self.dts)
		#print("New times: ", new_dts)
		self.dts = np.array(new_dts) 

	

	def get_traj_from_k_mode_policy(self, env, init_state, vis = False):
		states = []

		env.reset()
		state = init_state
		done = False
		self.reset(init_state)
		total_goal_error = 0.0
		total_safe_error = 0.0
		total_obj = 0.0

		total_decode_error = 0.0

		nm = len(self.modes)

		prev_idx = self.cur_id
		mode_change_states = []
		total_time = 0.0

		segments = []
		segment_states = []

		while True:
			action, dt = self.get_action(state)
			states.append((state, action))
			segment_states.append(state)      
			
			if prev_idx != self.cur_id:
				#if vis:
				#	print("Mode change from %i to %i"%(mode_idx, self.get_mode_id(self.cur_id)))
				mode_change_states.append(state)
				prev_idx = self.cur_id

				segments.append(segment_states)
				segment_states = []


			if len(action) == 0:
				done = True
				break
	 
			next_state, safe_error, goal_error, done = env.step(state, action, dt)
			total_safe_error += safe_error
			total_obj += env.get_obj(next_state)
			total_decode_error += env.get_decode_error(next_state, self.decode_params)
			if env.infinite_system: 
				# stop if the safety prop is violated
				if total_safe_error > 0.05:
					#total_safe_error += env.desired_duration - total_time 
					break 
			total_time += dt/env.dt_scale
			if env.infinite_system:
				if total_time > env.desired_duration:
					break 

			
			
		   
			state = next_state
			if done:
				mode_change_states.append(state)
				break

		
		states.append((state,[]))
		total_goal_error = env.check_goal(state)
		total_goal_error.append(env.check_time(total_time))

		total_obj += total_decode_error

		mode_err = 0.0
		if len(self.ref_trajs) > 0:

			ref_modes = [t.modes for t in self.ref_trajs]
			weight = 0.0
			count = 0.0 
			for i in range(len(ref_modes)):
				dist = np.linalg.norm(np.array(ref_modes[i]).flatten() - self.modes.flatten())/len(self.modes.flatten())	
				weight += self.ref_prob[i] * np.exp(-dist/1.0)
				count += self.ref_prob[i]
			weight = weight/count
			mode_err += -np.log(weight)
			

		assert(len(segments) <= len(self.modes))

		cond_err = 0.0
		if (len(self.ref_trajs) > 0):
			ref_times = [t.times for t in  self.ref_trajs]
			ref_conds = [t.conds for t in self.ref_trajs]
			ref_cond_noises = [t.cond_noises for t in self.ref_trajs]

			weight = 0.0
			count = 0.0
			for i in range(len(ref_times)):
				dist = np.linalg.norm(np.array(ref_times[i]).flatten() - self.dts/self.env.dt_scale * self.max_timesteps)/len(self.dts)
				cond_dist = self.get_cond_dist(segments, ref_conds[i], ref_cond_noises[i])
				cond_dist = min(cond_dist, 50.0)
				weight += self.ref_prob[i]*np.exp(-dist/1.0)*np.exp(-cond_dist/5.0)
				count += self.ref_prob[i]
			weight = weight/count 
			cond_err += -np.log(weight)

		
		
		return states, total_safe_error, total_goal_error, mode_err, cond_err, total_time, total_obj, mode_change_states

	def evaluate(self, init_state, vis = False, id = 0):
		states, safe_err, goal_err, mode_err, cond_err, total_time, total_obj, mode_change_states = self.get_traj_from_k_mode_policy(self.env, init_state, vis)
		
		if vis:
			s = ""
			for mode in self.modes:
				s += str(mode[1][0]) + " "
			print(s)
			

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


	def get_cond_dist(self, segments, ref_conds, ref_cond_noises):
		#print(len(segments), len(ref_conds), len(ref_cond_noises))
		assert(len(segments) <= len(ref_conds))

		error = 0.0
		num_count = 0
		for k in range(len(segments)):
			if self.dts[k] < 0.1/(self.env.dt_scale*100.0):
				continue
			segment = segments[k]
			ref_cond = ref_conds[k]
			ref_cond_noise = ref_cond_noises[k] 

			if ref_cond == None:
				continue

			for i in range(len(segment)):
				state = segment[i]
				features = self.env.get_features(state)
				cond_dist = ref_cond.eval(features) + ref_cond_noise
				num_count += 1
				if i != len(segment) - 1:
					# cond_dist should be negative
					if cond_dist > 0.0:
						error += cond_dist 
				else:
					# cond_dist should be positive
					if cond_dist < 0.0:
						error += -cond_dist 

		if num_count > 0:
			error = error/float(num_count)
		return error 



	def diff(self, traj1):
		mode_err =  np.linalg.norm(traj1.modes.flatten() - self.modes.flatten())	

		cond_err = np.linalg.norm(traj1.dts * self.max_timesteps - self.dts * self.max_timesteps)

		return mode_err + cond_err


	# should be called after cleanup 
	def get_segments(self, init_state, indices):
		self.env.reset()
		state = init_state
		done = False
		self.reset(init_state)

		prev_idx = -1


		time_in_mode = 0

		segments = {} 
		seg_states = [] 

		segments[-1] = [init_state]
		

		while True:	
			seg_states.append(state)	
			action, dt = self.get_action(state)
			
			if self.cur_id != prev_idx :
				if prev_idx >= 0:
					segments[prev_idx] = seg_states
				#else:
				#	assert(False)
				seg_states = [] 
				prev_idx = self.cur_id
				time_in_mode = 0
				


			time_in_mode += dt/self.env.dt_scale 
			
			if len(action) == 0:
				done = True
				break
			next_state, safe_error, goal_error, done = self.env.step(state, action, dt)
			
		   
			state = next_state
			if done:
				break

		for k in segments:
			if k == -1: continue 
			else:
				states = segments[k]
				extended_states = self.get_traj_states(states[-1], self.modes[k], self.max_timesteps, self.dts[k])
				segments[k] = (states, extended_states)

		return segments

	def get_traj_states(self, start_state, act_weights, timesteps, dt):
		self.env.reset()
		state = start_state
		done = False
		self.reset(start_state)

		all_states = []

		for i in range(timesteps):	
			all_states.append(state)	
			# compute the action
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

			next_state, safe_error, goal_error, done = self.env.step(state, acts, dt)
		   
			state = next_state
			if done:
				break
		all_states.append(state)
		return all_states

