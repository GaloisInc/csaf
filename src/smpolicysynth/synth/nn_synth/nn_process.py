import sys
sys.path.append("../../general/python")
#from run import *
from hopper import *
sys.path.append("../../general/python/synth")
from condition import * 
from utils import *
from simulate import *

import numpy as np 
from scipy import optimize 

def main():
	n_env_steps = 1000

	env = Hopper(n_env_steps)
	process_nn_from_file(env, "nn_hopper.txt")



def process_nn_from_file(env, filename):
	file = open(filename)
	states = []
	actions = []
	for l in file.readlines():
		if l[0] == "S" and l[1] == ":":
			l = l.strip().split(":")
			state = np.array(eval(l[1]))
			act = np.array(eval(l[2]))
			states.append(state)
			actions.append(act)
	print(len(states))

	features = env.get_features(states[0])
	cond = []
	for i in range(len(features)):
		cond.append(0.0)
	cond[0] = 1.0
	cond.append(-1.2)
	cond = LinearCond(cond)
	mapping = split_states(env, states, actions, cond)
	print(mapping)

	act_params = {}

	for i in [0,1]:
		act_params[i] = {}
		for j in [0,1,2]:
			theta,_ = learn_actions(env, states, actions, mapping, i, j)
			act_params[i][j] = theta

	policy = Policy(env, act_params, cond)
	init_state = states[20]
	simulate(env, policy, init_state, 2000, True)





def split_states(env, states, actions, cond):
	mapping = [] 
	for s in states:
		features = env.get_features(s)
		if cond.eval(features) >= 0.0:
			mapping.append(1)
		else:
			mapping.append(0)
	#print(mapping)
	return mapping


def learn_actions(env, states, actions, mapping, idx, act_idx):
	def obj(theta):
		return cost_fun(env, theta, states, actions, mapping, idx, act_idx)

	theta = []
	for i in range(env.num_act_features):
		theta.append(rand(-5.0, 5.0))
	theta = np.array(theta)
	res = optimize.fmin_bfgs(obj, theta, maxiter = 50, disp=0)
	theta0 = res 
	cost = obj(theta0)
	print(idx, act_idx, theta0, cost)
	return theta0, cost 


def calc_action(theta, features):
	assert(len(theta) == len(features) + 1)
	act = 0.0
	for j in range(len(features)):
		act += theta[j] * features[j]
	act += theta[-1]
	return act 

def calc_dist(env, theta, state, nn_action):
	dist = 0.0
	features = env.get_act_features(state)
	act = calc_action(theta, features)
	dist += (act - nn_action)**2
	return dist


def cost_fun(env, theta, states, actions, mapping, idx, act_idx):
	cost = 0.0 
	count = 0
	for i in range(len(states)):
		if mapping[i] != idx: continue	
		dist = calc_dist(env, theta, states[i], actions[i][act_idx])
		cost += dist
		count += 1
	#print(cost)
	return cost/float(count) + 0.01*np.linalg.norm(theta)


class Policy:
	def __init__(self, env, act_params, cond):
		self.act_params = act_params
		self.cond = cond
		self.env = env 

	def reset(self, init_state):
		pass

	def get_action(self, state):
		act_params = self.act_params
		cond = self.cond 
		env = self.env 

		cond_features = env.get_features(state)
		act_features = env.get_act_features(state) 

		if (cond.eval(cond_features) >= 0.0):
			acts = []
			for j in [0, 1, 2]:
				a = calc_action(act_params[1][j], act_features)
				acts.append(a)
		else:
			acts = []
			for j in [0, 1, 2]:
				a = calc_action(act_params[0][j], act_features)
				acts.append(a)

		return acts 


	
if __name__ == '__main__':
    main()
    
	