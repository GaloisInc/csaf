import sys 
import numpy as np 
import matplotlib.pyplot as pl 

import random

sys.path.append("../synth")
from constants import *
from simulate import * 

from utils import * 

import globals

sys.path.append("../policy")
from actions_policy import * 

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
		actions.append(0.0)
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
		a = [round(x, 2) for x in a]
		pl.title(str(a))

	globals.fig_new_row()


def get_traj_actions(traj_file_name):
	file = open(traj_file_name )
	actions = []
	init_state = [] 
	first = True
	for l in file.readlines():
		if l[0] == "S" and l[1] == ":":
			if first: 
				first = False
				continue 
			l = l.strip().split(":")
			state = np.array(eval(l[1]))
			act = np.array(eval(l[2]))
			if len(init_state) == 0:
				init_state = state 
			actions.append(act)

	file.close()
	return actions, init_state 
	
def init_features_random(env, traj_actions, nm_sm):
	actions_set = {}
	counts = {}
	for i in range(len(traj_actions)):
		a = np.copy(traj_actions[i])
		a = env.abstract_actions(a)
		astr = np.array2string(a)
		actions_set[astr] = a 
		if astr in counts:
			counts[astr] = counts[astr] + 1 
		else:
			counts[astr] = 1

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



def learn_modes_n_mapping(env, traj_file_name, nm_sm):
	traj_actions, init_state = get_traj_actions(traj_file_name )
	print(init_state)

	#actions1 = [[-1.0, 0.0, 0.0]]*500

	actions_policy = ActionsPolicy(env, traj_actions)
	simulate(env, actions_policy, init_state, 1000, True)
	assert(False)

	actions_mean, actions_std, actions_weights = init_features_random(env, traj_actions, nm_sm)

	#print(actions_mean)
	#assert(False)
	
	for tt in range(3):
		print("Assigning modes")
		globals.fig_add_subplot()
		mode_mapping = learn_structure(env, traj_actions, actions_mean, actions_std, actions_weights)

	

		print("Merging modes")
		globals.fig_add_subplot()
		actions_mean, actions_std, actions_weights = learn_actions(env, traj_actions, mode_mapping, nm_sm)


	plot_actions(env, actions_mean)
	print(actions_mean)

	traj_actions_mean = []
	for i in range(len(traj_actions)):
		mapping = mode_mapping[i]
		idx = np.argmax(mapping)
		traj_actions_mean.append(actions_mean[idx])

	pl.show()
	pl.close()

	actions_policy = ActionsPolicy(env, traj_actions_mean)
	simulate(env, actions_policy, init_state, 1000, True)

	return actions_mean, actions_std, mode_mapping
   

# Find best actions given mode_mapping
def learn_actions(env, traj_actions, mode_mapping, nm_sm):
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
		traj_action = traj_actions[i]
		mapping = mode_mapping[i]
		for sm_idx in range(len(mapping)):
			p = mapping[sm_idx]
			actions[sm_idx] += traj_action*p
			counts[sm_idx] += p
			actions_weights[sm_idx] += p

	for i in range(nm_sm):
		if counts[i] > 0:
			actions[i] = actions[i]/float(counts[i])
	
	print("Avg modes: ", actions.flatten())

	# Compute actions std 
	for i in mode_mapping:
		traj_action = traj_actions[i]
		mapping = mode_mapping[i]
		for sm_idx in range(len(mapping)):
			p = mapping[sm_idx]
			for kk in range(len(std[sm_idx])):
				std[sm_idx][kk] += p*(np.linalg.norm(traj_action[kk] - actions[sm_idx][kk] )**2)

	for i in range(nm_sm):
		if counts[i] > 0:
			std[i] = std[i]/float(counts[i])


	sum_weights = np.sum(actions_weights)
	actions_weights = actions_weights/sum_weights

	plot_actions_mean(traj_actions, actions, mode_mapping)


	return actions, std, actions_weights

def plot_actions_mean(traj_actions, actions_mean, mode_mapping):
	colors = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
			 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
			 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
			 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
			 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

	for i in range(len(colors)):  
		r, g, b = colors[i]  
		colors[i] = (r / 255., g / 255., b / 255.)


	# plot
	for i in mode_mapping:
		traj_action = traj_actions[i]
		X = []
		Y = [] 
		C = [] 
		mapping = mode_mapping[i]
		X.append(traj_action[0])
		if len(traj_action) > 1:
			Y.append(traj_action[1])
		else:
			Y.append(0.0)
		C.append(colors[np.argmax(mapping)])

		pl.scatter(X, Y, c = C, s = 20)
		#pl.plot(X, Y, alpha = 0.5)

	for i in range(len(actions_mean)):
		X = [actions_mean[i][0]]
		if len(actions_mean[i]) > 1:
			Y = [actions_mean[i][1]]
		else:
			Y = [1.0]
		pl.scatter(X, Y, marker = "s", c = [colors[i]], s = 10)

	pl.plot([-1.2, 1.2], [0, 0], "k--")
	pl.plot([0, 0], [-1.2, 1.2], "k--")

	pl.xlim((-1.2, 1.2))
	pl.ylim((-1.2, 1.2))


def get_gaussian_prob(a, m, sigma):
	assert(len(a) == len(m) == len(sigma))
	prob = 1.0
	for i in range(len(a)):
		std = max(sigma[i], 0.2)
		p = (1.0/np.sqrt(2*np.pi*std)) * np.exp(- ((a[i] - m[i])*(a[i] - m[i]))/(2.0 * std) )
		prob *= p
	return prob 


def get_mode_assignment(a, actions_mean, actions_std, actions_weights):
	prob = [] 

	for i in range(len(actions_mean)):
		m = actions_mean[i]
		std = actions_std[i]
		p = get_gaussian_prob(a.flatten(), m.flatten(), std)
		prob.append(p*actions_weights[i])

	s = np.sum(prob)
	prob = prob/np.sum(prob)
	#assert(abs(np.sum(prob)- 1) < 1e-4)

	return prob

# Learn the best mode mapping given actions and conds from previous iteration
def learn_structure(env, traj_actions, actions_mean, actions_std, actions_weights):

	mode_mapping = {} #  unroll idx -> sm idx
	for i in range(len(traj_actions)):
		traj_action = traj_actions[i]
		mode_mapping[i] = get_mode_assignment(traj_action, actions_mean, actions_std, actions_weights)


	plot_actions_mean(traj_actions, actions_mean, mode_mapping)
	
	return mode_mapping

sys.path.append(DIR + "hopper/python")
from hopper import *

if __name__ == '__main__':
	globals.ncol = 6 # max(nex, 2*nm_sm - 1)
	globals.nrow = 5
	globals.plt_idx = 1
	globals.fig = pl.figure(figsize = (10,15))


	env = Hopper(1000)
	learn_modes_n_mapping(env, DIR + "hopper/python/nn_hopper.txt", 3)

	

	
