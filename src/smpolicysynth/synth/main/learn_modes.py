import numpy as np 
import matplotlib.pyplot as pl 
import random

import synth.main.globals as globals

from general.utils import * 

def get_random_action(env):
	na = env.num_actions
	naf = env.num_act_features

	actions = [] 
	for i in range(na):
		a = []
		for j in range(naf):
			a.append(rand(-5, 5))
		actions.append(a)
	return actions

def get_zero_action(env):
	na = env.num_actions
	naf = env.num_act_features

	actions = [] 
	for i in range(na):
		a = []
		for j in range(naf):
			a.append(0.0)
		actions.append(a)
	return actions


def plot_actions(env, actions):
	
	for a in actions:
		globals.fig_add_subplot()
		state = env.get_neutral_state()
		
		states = []
		for i in range(200):
			states.append(state)
			state = env.simulate(state, a, -1)

		X,Y = env.get_2d_states1(states)
		pl.plot(X, Y)
		pl.xlim((-5, 5))
		pl.ylim((-5, 5))

		a = a.tolist()
		if len(a) > 2:
			a = a[:2]
		a = [round(x[0], 2) for x in a]
		pl.title(str(a))

	globals.fig_new_row()
	
def init_features_random(env, trajs, weights, nm_sm):
	actions_set = {}
	counts = {}
	for ii in range(len(trajs)):
		traj = trajs[ii]
		weight = weights[ii]
		for kk in range(len(traj.modes)):
			a = np.copy(traj.modes[kk])
			a = env.abstract_actions(a)
			astr = np.array2string(a)
			actions_set[astr] = a 
			if astr in counts:
				counts[astr] = counts[astr] + weight
			else:
				counts[astr] = weight

	counts_arr = [] 
	for s in counts:
		counts_arr.append((counts[s], s))

	counts_arr = sorted(counts_arr, reverse = True)

	actions_mean = [] 
	for k in range(min(nm_sm, len(counts_arr))):
		astr = counts_arr[k][1]
		actions_mean.append(actions_set[astr])
					
	ll = len(actions_mean)
	for k in range(ll, nm_sm, 1):
		actions_mean.append(get_random_action(env))
	actions_mean = np.array(actions_mean)

	actions_std = []
	for i in range(nm_sm):
		a_std = [] 
		for j in range(len(actions_mean[i])):
			a_std.append(1.0)
		actions_std.append(a_std)

	actions_weights = [] 
	for i in range(nm_sm):
		actions_weights.append(1.0/float(nm_sm))

	return actions_mean, actions_std, actions_weights

def init_from_prev_sm(env, trajs, sm, nm_sm):
	actions_mean = np.copy(sm.modes)

	actions_std = []
	for i in range(nm_sm):
		a_std = [] 
		for j in range(len(actions_mean[0])):
			a_std.append(1.0)
		actions_std.append(a_std)

	actions_weights = [] 
	for i in range(nm_sm):
		actions_weights.append(1.0/float(nm_sm))

	return actions_mean, actions_std, actions_weights


def compute_transition_time(env, states, ex_states, cond):
	changed = False 
	for idx in range(len(states)):
		s = states[idx]
		f = env.get_features(s)
		if cond.eval(f) >= 0:
			changed = True
			break 
	if changed:
		return -(len(states) - 1 - idx)
	else:
		for idx in range(len(ex_states)):
			s = ex_states[idx]
			f = env.get_features(s)
			if cond.eval(f) >= 0:
				break 

		return idx 

def compute_cond_times(env, trajs, init_states, sm):
	conds = sm.conds 

	segments = {}
	for k in range(len(trajs)):
		t_segments = trajs[k].get_segments(init_states[k], None)
		segments[k] = t_segments

	cond_times = {} # traj id -> traj mode id -> sm mode id 1 -> sm mode id 2 -> transition time 

	for k in segments:
		t_segments = segments[k]
		t_times = {}
		for i in t_segments:
			if i == -1: continue # we don't need start modes here 
			states, ex_states = t_segments[i]
			i_times = {}
			# comput times for every cond g_{j1}^{j2}
			for j1 in conds:
				if j1 == -1: continue
				j1_times = {}
				for j2 in conds[j1]:
					time = compute_transition_time(env, states, ex_states, conds[j1][j2])
					j1_times[j2] = time 
				i_times[j1] = j1_times
			t_times[i] = i_times
		cond_times[k] = t_times 

	return cond_times 


def learn_modes_n_mapping(env, init_states, trajs, weights, sm, nm_sm):
	if sm == None:
		actions_mean, actions_std, actions_weights = init_features_random(env, trajs, weights, nm_sm)
	else:
		actions_mean, actions_std, actions_weights = init_from_prev_sm(env, trajs, sm, nm_sm)

	if sm == None:
		cond_times = None 
	else:
		cond_times = compute_cond_times(env, trajs, init_states, sm)
	
	for tt in range(3):
		print("Assigning modes")
		globals.fig_add_subplot()
		mode_mapping = learn_structure(env, init_states, trajs, weights, actions_mean, actions_std, actions_weights, cond_times)

	

		print("Merging modes")
		globals.fig_add_subplot()
		actions_mean, actions_std, actions_weights = learn_actions(env, init_states, trajs, mode_mapping, nm_sm, weights)


	for ii in mode_mapping:
		print("Traj %i"%ii)
		print(trajs[ii].modes.flatten())
		print(trajs[ii].dts*100.0)
		ss = "" 
		for kk in mode_mapping[ii]:
			m = np.argmax(mode_mapping[ii][kk])
			ss += "(%i,%i,%f)->"%(kk, m, mode_mapping[ii][kk][m])
		print(ss)

	plot_actions(env, actions_mean)

	return actions_mean, actions_std, mode_mapping
   

# Find best actions given mode_mapping
def learn_actions(env, init_states, trajs, mode_mapping, nm_sm, weights):
	colors = [(31, 119, 180), (255, 127, 14),  
			 (44, 160, 44), (214, 39, 40),   
			 (148, 103, 189), (140, 86, 75),  
			 (227, 119, 194),  (127, 127, 127),   
			 (188, 189, 34),  (23, 190, 207)]

	for i in range(len(colors)):  
		r, g, b = colors[i]  
		colors[i] = [r / 255., g / 255., b / 255.]


	# initialize actions 
	actions = [] 
	for i in range(nm_sm):
		actions.append(get_zero_action(env))
	actions = np.array(actions)

	# initialize stds 
	std = [] 
	for i in range(nm_sm):
		s = [] 
		for j in range(len(actions[i])):
			s.append(0.0)
		std.append(s)
	std = np.array(std)

	# initialize action weights
	actions_weights = [] 
	for i in range(nm_sm):
		actions_weights.append(0)

	# compute actions mean 
	counts = [] 
	for i in range(nm_sm):
		counts.append(0)

	for i in mode_mapping:
		traj = trajs[i]
		mapping = mode_mapping[i]
		for j in mapping:
			for sm_idx in range(len(mapping[j])):
				p = mapping[j][sm_idx]
				actions[sm_idx] += traj.modes[j]*weights[i]*p
				counts[sm_idx] += weights[i]*p
				actions_weights[sm_idx] += p*weights[i]

	for i in range(nm_sm):
		if counts[i] > 0:
			actions[i] = actions[i]/float(counts[i])
	
	print("Avg modes: ", actions.flatten())

	# Compute actions std 
	for i in mode_mapping:
		traj = trajs[i]
		mapping = mode_mapping[i]
		for j in mapping:
			for sm_idx in range(len(mapping[j])):
				p = mapping[j][sm_idx]
				for kk in range(len(std[sm_idx])):
					std[sm_idx][kk] += weights[i]*p*(np.linalg.norm(traj.modes[j][kk] - actions[sm_idx][kk] )**2)

	for i in range(nm_sm):
		if counts[i] > 0:
			std[i] = std[i]/float(counts[i])


	sum_weights = np.sum(actions_weights) + 1e-4
	actions_weights = actions_weights/sum_weights

	# plot
	for i in mode_mapping:
		traj = trajs[i]
		X = []
		Y = [] 
		C = [] 
		mapping = mode_mapping[i]
		weight = weights[i]
		for k in mapping:
			X.append(traj.modes[k][0][0])
			if len(traj.modes[k]) > 1:
				Y.append(traj.modes[k][1][0])
			else:
				Y.append(0.0)
			C.append(colors[np.argmax(mapping[k])] + [weight])

		pl.scatter(X, Y, c = C, s = 20)
		#pl.plot(X, Y, alpha = 0.5)

	for i in range(len(actions)):
		X = [actions[i][0][0]]
		if len(actions[i]) > 1:
			Y = [actions[i][1][0]]
		else:
			Y = [1.0]
		pl.scatter(X, Y, marker = "s", c = [colors[i]], s = 10)

	pl.plot([-5.2, 5.2], [0, 0], "k--")
	pl.plot([0, 0], [-5.2, 5.2], "k--")

	pl.xlim((-5.2, 5.2))
	pl.ylim((-5.2, 5.2))


	return actions, std, actions_weights

def get_gaussian_prob(a, m, sigma):
	assert(len(a) == len(m) == len(sigma))
	prob = 1.0
	for i in range(len(a)):
		std = max(sigma[i], 0.2)
		p = (1.0/np.sqrt(2*np.pi*std)) * np.exp(- ((a[i] - m[i])*(a[i] - m[i]))/(2.0 * std) )
		prob *= p
	return prob 

def get_cond_prob(cond_times, next_mode_mapping):
	if len(next_mode_mapping) == 0:
		# special case, next mode is STOP 
		prob = 1.0
		for j3 in cond_times:
			t = cond_times[j3]
			if j3 == -2: # prob for changing at t
				p = np.exp(-abs(t)/2.0)
				prob *= p 
			else: # prob for changing after t 
				p = np.exp(-relu(-t)/2.0) 
				prob *= p 
		return prob
	else:
		prob = 0.0 
		p_j2_total = 0.0
		# do a summation over possible next mode 
		for j2 in range(len(next_mode_mapping)):
			p_j2 = next_mode_mapping[j2]
			p = 1.0
			for j3 in cond_times: 
				t = cond_times[j3]
				if j3 == j2:  # prob for changing at t
					p_t = np.exp(-abs(t)/10.0)
					p *= p_t 
				else: # prob for changing after t 
					p_t = np.exp(-relu(-t)/10.0)
					p *= p_t 

			prob += p_j2*p
			p_j2_total += p_j2 

		prob = prob/p_j2_total 
		return prob


def get_mode_assignment(a, actions_mean, actions_std, actions_weights, cond_times, next_mode_mapping):
	prob = [] 

	for i in range(len(actions_mean)):
		m = actions_mean[i]
		std = actions_std[i]
		p = get_gaussian_prob(a.flatten(), m.flatten(), std)
		if cond_times != None:
			cond_p = get_cond_prob(cond_times[i], next_mode_mapping)
			p = p*cond_p 
		prob.append(p*actions_weights[i])



	s = np.sum(prob)
	prob = prob/np.sum(prob)
	#assert(abs(np.sum(prob)- 1) < 1e-4)

	return prob

# Learn the best mode mapping given actions and conds from previous iteration
def learn_structure(env, init_states, trajs, weights, actions_mean, actions_std, actions_weights, cond_times ):
	colors = [(31, 119, 180), (255, 127, 14),  
			 (44, 160, 44), (214, 39, 40),   
			 (148, 103, 189), (140, 86, 75),  
			 (227, 119, 194),  (127, 127, 127),   
			 (188, 189, 34),  (23, 190, 207)]

	for i in range(len(colors)):  
		r, g, b = colors[i]  
		colors[i] = [r / 255., g / 255., b / 255.]



	mode_mapping = {} # traj -> unroll idx -> sm idx
	for i in range(len(trajs)):
		traj = trajs[i]
		mapping = {}
		prev_k = -2
		# Iterate from back 
		for k in range(len(traj.modes) - 1, -1, -1):
			ct = None if cond_times == None else cond_times[i][k]
			next_mode_mapping = [] if prev_k == -2 else mapping[prev_k]
			mapping[k] = get_mode_assignment(traj.modes[k], actions_mean, actions_std, actions_weights, ct, next_mode_mapping)

			prev_k = k
		mode_mapping[i] = mapping

	# plot
	for i in mode_mapping:
		traj = trajs[i]
		X = []
		Y = [] 
		C = [] 
		mapping = mode_mapping[i]
		weight = weights[i]
		for k in mapping:
			X.append(traj.modes[k][0][0])
			if len(traj.modes[k]) > 1:
				Y.append(traj.modes[k][1][0])
			else:
				Y.append(0.0)
			C.append(colors[np.argmax(mapping[k])] + [weight])

		pl.scatter(X, Y, c = C, s = 20)
		#pl.plot(X, Y, alpha = 0.5)

	for i in range(len(actions_mean)):
		X = [actions_mean[i][0][0]]
		if len(actions_mean[i]) > 1:
			Y = [actions_mean[i][1][0]]
		else:
			Y = [1.0]
		pl.scatter(X, Y, marker = "s", c = [colors[i]], s = 10)

	pl.plot([-5.2, 5.2], [0, 0], "k--")
	pl.plot([0, 0], [-5.2, 5.2], "k--")

	pl.xlim((-5.2, 5.2))
	pl.ylim((-5.2, 5.2))

	return mode_mapping
