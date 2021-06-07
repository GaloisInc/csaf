import sys 
from synth.policy.condition import *
from general.utils import *  

import synth.main.globals as globals 

import scipy
from scipy import optimize 
import random
import numpy as np 

import matplotlib.pyplot as pl 

from multiprocessing import Process, Queue

prev_cond_opt_state = None

def optimize_conds(env, init_states, trajs, mode_mapping, nm_sm, weights, cond_depth):
	global prev_cond_opt_state 
	segments = {}
	for k in range(len(trajs)):
		t_segments = trajs[k].get_segments(init_states[k], mode_mapping[k].keys())
		assert(len(t_segments) == len(mode_mapping[k]) + 1)
		segments[k] = t_segments

	procs = [] 
	queue = Queue()
	for i in range(nm_sm):
		for j in range(nm_sm):
			if i == j: continue 
			scipy.random.seed()
			random.seed()
			np.random.seed()

			proc = Process(target = optimize_cond, args=[env, segments, mode_mapping, weights, i, j, cond_depth, prev_cond_opt_state, queue])
			procs.append(proc)
			proc.start()

		if not env.infinite_system :
			scipy.random.seed()
			random.seed()
			np.random.seed()

			# end condition
			proc = Process(target = optimize_cond, args=[env, segments, mode_mapping, weights, i, -2, cond_depth, prev_cond_opt_state, queue])
			procs.append(proc)
			proc.start()

	# start conditions
	for i in range(nm_sm):
		scipy.random.seed()
		random.seed()
		np.random.seed()

		proc = Process(target = optimize_cond, args=[env, segments, mode_mapping, weights, -1, i, cond_depth, prev_cond_opt_state, queue])
		procs.append(proc)
		proc.start()


	prev_cond_opt_state = {} # i->j-> (L/A/O, f, sign) -> theta

	conds_l = {}
	conds_stds_l = {}
	conds_costs_l = {}
	for k in range(len(procs)):
		i,j,cond_l,std_l,cost_l = queue.get()
		if i not in conds_l:
			conds_l[i] = {}
			conds_stds_l[i] = {} 
			conds_costs_l[i] = {} 
			prev_cond_opt_state[i] = {} 
		conds_l[i][j] = cond_l
		conds_stds_l[i][j] = std_l
		conds_costs_l[i][j] = cost_l

		prev_cond_opt_state[i][j] = {}
		for cond in cond_l:
			if isinstance(cond, LinearCond):
				f, sign, theta = extract_cond_features(cond)
				prev_cond_opt_state[i][j][('L', f, sign)] = theta 
			elif isinstance(cond, AndCond) and isinstance(cond.mother, LinearCond):
				f, sign, theta = extract_cond_features(cond.father)
				prev_cond_opt_state[i][j][('A', f, sign)] = theta 
			elif isinstance(cond, AndCond) and isinstance(cond.mother, AndCond):
				f, sign, theta = extract_cond_features(cond.father)
				prev_cond_opt_state[i][j][('AA', f, sign)] = theta 
			elif isinstance(cond, OrCond) and isinstance(cond.mother, LinearCond):
				f, sign, theta = extract_cond_features(cond.father)
				prev_cond_opt_state[i][j][('O', f, sign)] = theta 
			elif isinstance(cond, OrCond) and isinstance(cond.mother, OrCond):
				f, sign, theta = extract_cond_features(cond.father)
				prev_cond_opt_state[i][j][('OO', f, sign)] = theta 

	# print(prev_cond_opt_state)



	for proc in procs:
		proc.join()

	conds = {}
	conds_stds = {}
	for i in conds_l:
		for j in conds_l[i]:
			print("%i->%i"%(i, j))
			cond_l = conds_l[i][j]
			cost_l = conds_costs_l[i][j]
			std_l = conds_stds_l[i][j]

			# sort by prefering min cost and min std better
			cost_l, std_l, cond_l, = zip(*sorted(zip(cost_l, std_l, cond_l), reverse=False, key=lambda x: (x[0], x[1])))

			for k in range(min(3, len(cond_l))):
				print(cond_l[k], cost_l[k])

			# get the best cond 
			cond = cond_l[0]
			cost = cost_l[0]
			std = std_l[0]

			if i not in conds:
				conds[i] = {}
				conds_stds[i] = {}
			conds[i][j] = cond 
			conds_stds[i][j] = std 

			# plot
			globals.fig_add_subplot()
			print("Cond: ", str(cond))
			print("Cost: ", cost)
			visualize_cond(env, segments, mode_mapping, weights, i, j, cond)
			pl.title("%i->%i "%(i, j) + str(round(cost, 3)))
	globals.fig_new_row()

	'''# visualize opt terrain - comment this when running actual benchmarks
	for i in conds:
		for j in conds[i]:
			globals.fig_add_subplot()
			cond = conds[i][j]
			visualize_opt_terrain(env, segments, mode_mapping, weights, i, j, cond)
	globals.fig_new_row()'''
	return conds, conds_stds

# learn the condition g_i^j
def optimize_cond(env, segments, mode_mapping, weights, i, j, cond_depth, prev_cond_opt_state, result_queue):
	num_features = env.num_cond_features - 1

	final_costs = [] 
	final_conds = [] 
	final_stds = [] 

	# learn linear conds 
	costs = []
	conds = [] 
	stds = []
	for f in range(num_features):
		theta_init = [] 
		if prev_cond_opt_state != None:
			key = ('L', f, 1.0)
			if key in prev_cond_opt_state[i][j]:
				theta_init = [prev_cond_opt_state[i][j][key]]
		cond, std, cost = learn_cond_with_feature(env, segments, mode_mapping, weights, i, j, f, 1.0, num_features, theta_init)
		costs.append(cost)
		conds.append(cond)
		stds.append(std)
		
		theta_init = [] 
		if prev_cond_opt_state != None:
			key = ('L', f, -1.0)
			if key in prev_cond_opt_state[i][j]:
				theta_init = [prev_cond_opt_state[i][j][key]]
		cond, std, cost = learn_cond_with_feature(env, segments, mode_mapping, weights, i, j, f, -1.0, num_features, theta_init)
		costs.append(cost)
		conds.append(cond)
		stds.append(std)

	costs, stds, conds, = zip(*sorted(zip(costs, stds, conds), reverse=False, key=lambda x: (x[0], x[1])))

	final_costs.extend(costs)
	final_stds.extend(stds)
	final_conds.extend(conds)

	# pick the best linear cond 
	best_linear_cond = conds[0] 
	best_f, best_sign, best_theta = extract_cond_features(best_linear_cond)

	if cond_depth >= 2:
		# learn and conditions
		costs = []
		conds = [] 
		stds = []
		for f in range(num_features):
			theta_init = [best_theta] 
			if prev_cond_opt_state != None:
				key = ('A', f, 1.0)
				if key in prev_cond_opt_state[i][j]:
					theta_init = [best_theta, prev_cond_opt_state[i][j][key]]
			cond, std, cost = learn_cond_with_features_and(env, segments, mode_mapping, weights, i, j, best_f, best_sign, f, 1.0, num_features, theta_init)
			costs.append(cost)
			conds.append(cond)
			stds.append(std)
			
			theta_init = [best_theta] 
			if prev_cond_opt_state != None:
				key = ('A', f, -1.0)
				if key in prev_cond_opt_state[i][j]:
					theta_init = [best_theta, prev_cond_opt_state[i][j][key]]
			cond, std, cost = learn_cond_with_features_and(env, segments, mode_mapping, weights, i, j, best_f, best_sign, f, -1.0, num_features, theta_init)
			costs.append(cost)
			conds.append(cond)
			stds.append(std)

		costs, stds, conds, = zip(*sorted(zip(costs, stds, conds), reverse=False, key=lambda x: (x[0], x[1])))
		# pick best and cond
		best_and_cond = conds[0]
		best_and_f1, best_and_sign1, best_and_theta1 = extract_cond_features(best_and_cond.mother)
		best_and_f2, best_and_sign2, best_and_theta2 = extract_cond_features(best_and_cond.father)

		# scale costs a bit higher to prefer smaller conditions
		costs = (np.array(costs)*1.02).tolist()

		final_costs.extend(costs)
		final_stds.extend(stds)
		final_conds.extend(conds)

		# learn or conditions
		costs = []
		conds = [] 
		stds = []
		for f in range(num_features):
			theta_init = [best_theta] 
			if prev_cond_opt_state != None:
				key = ('O', f, 1.0)
				if key in prev_cond_opt_state[i][j]:
					theta_init = [best_theta, prev_cond_opt_state[i][j][key]]
			cond, std, cost = learn_cond_with_features_or(env, segments, mode_mapping, weights, i, j, best_f, best_sign, f, 1.0, num_features, theta_init)
			costs.append(cost)
			conds.append(cond)
			stds.append(std)
			
			theta_init = [best_theta] 
			if prev_cond_opt_state != None:
				key = ('O', f, -1.0)
				if key in prev_cond_opt_state[i][j]:
					theta_init = [best_theta, prev_cond_opt_state[i][j][key]]
			cond, std, cost = learn_cond_with_features_or(env, segments, mode_mapping, weights, i, j, best_f, best_sign, f, -1.0, num_features, theta_init)
			costs.append(cost)
			conds.append(cond)
			stds.append(std)

		costs, stds, conds, = zip(*sorted(zip(costs, stds, conds), reverse=False, key=lambda x: (x[0], x[1])))

		# pick best or condition
		best_or_cond = conds[0]
		best_or_f1, best_or_sign1, best_or_theta1 = extract_cond_features(best_or_cond.mother)
		best_or_f2, best_or_sign2, best_or_theta2 = extract_cond_features(best_or_cond.father)

		# scale costs a bit higher to prefer smaller conditions
		costs = (np.array(costs)*1.02).tolist()

		final_costs.extend(costs)
		final_stds.extend(stds)
		final_conds.extend(conds)


	if cond_depth >= 3:
		# learn and3 condition
		costs = []
		conds = [] 
		stds = []
		for f in range(num_features):
			theta_init = [best_and_theta1, best_and_theta2] 
			if prev_cond_opt_state != None:
				key = ('AA', f, 1.0)
				if key in prev_cond_opt_state[i][j]:
					theta_init = [best_theta, prev_cond_opt_state[i][j][key]]
			cond, std, cost = learn_cond_with_features_and3(env, segments, mode_mapping, weights, i, j, best_and_f1, best_and_sign1, best_and_f2, best_and_sign2, f, 1.0, num_features, theta_init)
			costs.append(cost)
			conds.append(cond)
			stds.append(std)
			
			theta_init = [best_and_theta1, best_and_theta2] 
			if prev_cond_opt_state != None:
				key = ('AA', f, -1.0)
				if key in prev_cond_opt_state[i][j]:
					theta_init = [best_theta, prev_cond_opt_state[i][j][key]]
			cond, std, cost = learn_cond_with_features_and3(env, segments, mode_mapping, weights, i, j, best_and_f1, best_and_sign1, best_and_f2, best_and_sign2, f, -1.0, num_features, theta_init)
			costs.append(cost)
			conds.append(cond)
			stds.append(std)

		costs, stds, conds, = zip(*sorted(zip(costs, stds, conds), reverse=False, key=lambda x: (x[0], x[1])))


		# scale costs a bit higher to prefer smaller conditions
		costs = (np.array(costs)*1.04).tolist()

		final_costs.extend(costs)
		final_stds.extend(stds)
		final_conds.extend(conds)

		# learn or3 condition
		costs = []
		conds = [] 
		stds = []
		for f in range(num_features):
			theta_init = [best_and_theta1, best_and_theta2] 
			if prev_cond_opt_state != None:
				key = ('OO', f, 1.0)
				if key in prev_cond_opt_state[i][j]:
					theta_init = [best_theta, prev_cond_opt_state[i][j][key]]
			cond, std, cost = learn_cond_with_features_or3(env, segments, mode_mapping, weights, i, j, best_and_f1, best_and_sign1, best_and_f2, best_and_sign2, f, 1.0, num_features, theta_init)
			costs.append(cost)
			conds.append(cond)
			stds.append(std)
			
			theta_init = [best_and_theta1, best_and_theta2] 
			if prev_cond_opt_state != None:
				key = ('OO', f, -1.0)
				if key in prev_cond_opt_state[i][j]:
					theta_init = [best_theta, prev_cond_opt_state[i][j][key]]
			cond, std, cost = learn_cond_with_features_or3(env, segments, mode_mapping, weights, i, j, best_and_f1, best_and_sign1, best_and_f2, best_and_sign2, f, -1.0, num_features, theta_init)
			costs.append(cost)
			conds.append(cond)
			stds.append(std)

		costs, stds, conds, = zip(*sorted(zip(costs, stds, conds), reverse=False, key=lambda x: (x[0], x[1])))


		# scale costs a bit higher to prefer smaller conditions
		costs = (np.array(costs)*1.04).tolist()

		final_costs.extend(costs)
		final_stds.extend(stds)
		final_conds.extend(conds)


	# DO final sorting
	final_costs, final_stds, final_conds, = zip(*sorted(zip(final_costs, final_stds, final_conds), reverse=False, key=lambda x: (x[0], x[1])))

	final_costs = final_costs[0:5]
	final_stds = final_costs[0:5]
	final_conds = final_conds[0:5]

	result_queue.put([i, j, final_conds, final_stds, final_costs])
	return 

# learn cond of the form sign * x_f > theta 
def learn_cond_with_feature(env, segments, mode_mapping, weights, i, j, f, sign, num_features, theta_init):
	segments_features = get_segment_features(env, segments, [f], [sign])

	def obj(theta):
		return cost_fun(theta, linear_vc, segments_features, mode_mapping, weights, i, j)

	x0 = rand(-10.0, 10.0) 
	theta = [x0]
	if len(theta_init) > 0:
		theta = theta_init
	res = optimize.fmin_bfgs(obj, theta, maxiter = 50, disp=0)
	theta0 = res[0] 
	cost = obj([theta0])

	cond = create_cond(f, sign, theta0, num_features)

	std = calculate_std([theta0], linear_vc, segments_features, mode_mapping, weights, i, j)

	return cond, std, cost 


# learn cond of the form sign1 * x_f1 > theta1 and sign2 * x_f2 > theta2 
def learn_cond_with_features_and(env, segments, mode_mapping, weights, i, j, f1, sign1, f2, sign2, num_features, theta_init):
	segments_features = get_segment_features(env, segments, [f1, f2], [sign1, sign2])

	def obj(theta):
		return cost_fun(theta, and_vc, segments_features, mode_mapping, weights, i, j)

	x0 = rand(-10.0, 10.0)
	x1 = rand(-10, 10.0)
	theta = [x0, x1]
	if len(theta_init) == 2:
		theta = theta_init
	elif len(theta_init) == 1:
		theta = [theta_init[0], x1]
	res = optimize.fmin_bfgs(obj, np.array(theta), maxiter = 50, disp=0)
	theta1 = res[0]
	theta2 = res[1] 
	cost = obj(np.array([theta1, theta2]))
	
	cond1 = create_cond(f1, sign1, theta1, num_features)
	cond2 = create_cond(f2, sign2, theta2, num_features)
	cond = AndCond(cond1, cond2)

	std = calculate_std([theta1, theta2], and_vc, segments_features, mode_mapping, weights, i, j)

	return cond, std, cost 

# learn cond of the form sign1 * x_f1 > theta1 or sign2 * x_f2 > theta2 
def learn_cond_with_features_or(env, segments, mode_mapping, weights, i, j, f1, sign1, f2, sign2, num_features, theta_init):
	segments_features = get_segment_features(env, segments, [f1, f2], [sign1, sign2])

	def obj(theta):
		return cost_fun(theta, or_vc, segments_features, mode_mapping, weights, i, j)

	x0 = rand(-10.0, 10.0)
	x1 = rand(-10.0, 10.0)
	theta = [x0, x1]
	if len(theta_init) == 2:
		theta = theta_init
	elif len(theta_init) == 1:
		theta = [theta_init[0], x1]
	res = optimize.fmin_bfgs(obj, np.array(theta), maxiter = 50, disp=0)
	theta1 = res[0]
	theta2 = res[1] 
	cost = obj(np.array([theta1, theta2]))
	
	cond1 = create_cond(f1, sign1, theta1, num_features)
	cond2 = create_cond(f2, sign2, theta2, num_features)
	cond = OrCond(cond1, cond2)

	std = calculate_std([theta1, theta2], or_vc, segments_features, mode_mapping, weights, i, j)

	return cond, std, cost 



# learn and3 cond 
def learn_cond_with_features_and3(env, segments, mode_mapping, weights, i, j, f1, sign1, f2, sign2, f3, sign3, num_features, theta_init):
	segments_features = get_segment_features(env, segments, [f1, f2, f3], [sign1, sign2, sign3])

	def obj(theta):
		return cost_fun(theta, and3_vc, segments_features, mode_mapping, weights, i, j)

	x0 = rand(-10.0, 10.0)
	x1 = rand(-10, 10.0)
	x2 = rand(-10, 10)
	theta = [x0, x1, x2]
	if len(theta_init) == 3:
		theta = theta_init
	elif len(theta_init) == 2:
		theta = [theta_init[0], theta_init[1], x2]
	res = optimize.fmin_bfgs(obj, np.array(theta), maxiter = 50, disp=0)
	theta1 = res[0]
	theta2 = res[1]
	theta3 = res[2] 
	cost = obj(np.array([theta1, theta2, theta3]))
	
	cond1 = create_cond(f1, sign1, theta1, num_features)
	cond2 = create_cond(f2, sign2, theta2, num_features)
	cond3 = create_cond(f3, sign3, theta3, num_features)
	cond = AndCond(cond1, cond2)
	cond = AndCond(cond, cond3)

	std = calculate_std([theta1, theta2, theta3], and3_vc, segments_features, mode_mapping, weights, i, j)

	return cond, std, cost 


# learn or3 cond 
def learn_cond_with_features_or3(env, segments, mode_mapping, weights, i, j, f1, sign1, f2, sign2, f3, sign3, num_features, theta_init):
	segments_features = get_segment_features(env, segments, [f1, f2, f3], [sign1, sign2, sign3])

	def obj(theta):
		return cost_fun(theta, or3_vc, segments_features, mode_mapping, weights, i, j)

	x0 = rand(-10.0, 10.0)
	x1 = rand(-10, 10.0)
	x2 = rand(-10, 10)
	theta = [x0, x1, x2]
	if len(theta_init) == 3:
		theta = theta_init
	elif len(theta_init) == 2:
		theta = [theta_init[0], theta_init[1], x2]
	res = optimize.fmin_bfgs(obj, np.array(theta), maxiter = 50, disp=0)
	theta1 = res[0]
	theta2 = res[1]
	theta3 = res[2] 
	cost = obj(np.array([theta1, theta2, theta3]))
	
	cond1 = create_cond(f1, sign1, theta1, num_features)
	cond2 = create_cond(f2, sign2, theta2, num_features)
	cond3 = create_cond(f3, sign3, theta3, num_features)
	cond = OrCond(cond1, cond2)
	cond = OrCond(cond, cond3)

	std = calculate_std([theta1, theta2, theta3], or3_vc, segments_features, mode_mapping, weights, i, j)

	return cond, std, cost 

def create_cond(f, sign, theta, num_features):
	cond = []
	for k in range(num_features):
		cond.append(0.0)
	cond[f] = sign 
	cond.append(-theta)
	cond = np.array(cond)
	cond = LinearCond(cond)
	return cond 

def extract_cond_features(cond):
	cond = cond.params 
	f = -1
	sign = 0
	for k in range(len(cond) - 1): 
		v = cond[k]
		if v != 0:
			f = k
			sign = v 
			break

	theta = -cond[-1]
	return f, sign, theta

# skeys should be sorted
def get_prob(skeys, k, mode_mapping, i, j):
	assert(skeys[0] == -1)
	m1 = skeys[k]
	m2 = skeys[k+1] if k < len(skeys) - 1 else -2
	p_i = 0.0
	if i >= 0 and m1 >= 0:
		p_i = mode_mapping[m1][i] 
	elif i == -1 and m1 == -1:
		p_i = 1.0 

	p_j = 0.0 
	if j >= 0 and m2 >= 0:
		p_j = mode_mapping[m2][j]
	elif j == -2 and m2 == -2:
		p_j = 1.0

	return p_i, p_j

def get_segment_features(env, segments, f, sign):
	segments_features = {}
	for t in segments:
		sv = {}
		for k in segments[t]:
			states = segments[t][k]
			s_features = []
			if k == -1:
				for idx in range(len(states)):
					s = states[idx]
					features = env.get_features(s)
					v_list = []
					for i in range(len(f)):
						v = features[f[i]]*sign[i]
						v_list.append(v)
					s_features.append(v_list)
				sv[k] = s_features
			else:
				states, extended_states = states 
				s_features = []
				ex_features = [] 
				for s in states:
					features = env.get_features(s)
					v_list = []
					for i in range(len(f)):
						v = features[f[i]]*sign[i]
						v_list.append(v)
					s_features.append(v_list)
				for s in extended_states:
					features = env.get_features(s)
					v_list = []
					for i in range(len(f)):
						v = features[f[i]]*sign[i]
						v_list.append(v)
					ex_features.append(v_list)

				sv[k] = (s_features, ex_features)

		segments_features[t] = sv 
	return segments_features

def linear_vc(v, theta):
	return v[0] - theta[0]

def and_vc(v, theta):
	return min(v[0] - theta[0], v[1] - theta[1])

def or_vc(v, theta):
	return max(v[0] - theta[0], v[1] - theta[1])

def and3_vc(v, theta):
	return min(v[0] - theta[0], v[1] - theta[1], v[2] - theta[2])

def or3_vc(v, theta):
	return max(v[0] - theta[0], v[1] - theta[1], v[2] - theta[2])

def cost_fun(theta, value_combinator, segments_features, mode_mapping, weights, i, j):
	cost = 0.0 
	for t in segments_features:
		p_t = weights[t]
		s = segments_features[t].keys()
		s = sorted(s)
		for k1 in range(len(s)):
			p_i, p_j = get_prob(s, k1, mode_mapping[t], i, j)
			assert(0.0 <= p_i <= 1.0)
			assert(0.0 <= p_j <= 1.0)

			segment = segments_features[t][s[k1]]

			# special case for the start
			if i == -1 :
				if p_i == 0.0: continue 

				val = segment[0]
				val_t = value_combinator(val, theta)
				# cost for making the change from i to j 
				cost += p_t*p_i*p_j*relu(-val_t + 0.1)

				# cost for making the change from i to something other than j
				cost += p_t*p_i*(1 - p_j)*relu(val_t + 0.1) 

			else:
				if p_i == 0.0: continue 

				s_vals, ex_vals = segment 
				changed = False
				for idx in range(len(s_vals)):
					val = s_vals[idx]
					val_t = value_combinator(val, theta)
					if val_t >= 0:
						changed = True 
						break 

				if changed:
					if idx == len(s_vals) - 1:
						cost += p_t*p_i*(1 - p_j)
					else:
						cost += p_t*p_i* (len(s_vals) - 1 - idx) 
				
				ex_changed = False
				for idx in range(len(ex_vals)):
					val = ex_vals[idx]
					val_t = value_combinator(val, theta)
					if val_t >= 0:
						ex_changed = True 
						break 

				#cost += 0.0001*p_t*p_i*(1- p_j) * (len(ex_vals) - 1 - idx) 

				if (not changed) :
					cost += p_t*p_i*p_j*idx

				val = s_vals[-1]
				val_t = value_combinator(val, theta)
				cost += p_t*p_i*p_j*abs(val_t)

	return cost 

def calculate_std(theta0, value_combinator, segments_features, mode_mapping, weights, i, j):
	# calculate std dev 
	std_dev = 0.0
	count = 0.0
	for t in segments_features:
		p_t = weights[t]
		s = segments_features[t].keys()
		s = sorted(s)
		for k1 in range(len(s)):
			p_i, p_j = get_prob(s, k1, mode_mapping[t], i, j)
			segment = segments_features[t][s[k1]]
			max_val = -1e20 
			if len(segment) == 1:
				val = segment[0]
				val_t = value_combinator(val, theta0)
				max_val = val_t 
			else:
				s_vals, _ = segment
				for v in s_vals:
					val_t = value_combinator(v, theta0)
					if val_t > max_val:
						max_val = val_t 

			if i != -1:
				std_dev += p_t*p_i*p_j*((max_val)** 2)
			else:
				if max_val < 0.0:
					std_dev += p_t*p_i*p_j*((max_val)** 2)
			count += p_t*p_i*p_j

			if max_val >= 0.0:
				std_dev += p_t*p_i*(1 - p_j)*((max_val)** 2)
			count += p_t*p_i*(1 - p_j)
	if count == 0:
		std = 0.0
	else:
		std = std_dev / count

	return std 

def visualize_opt_terrain(env, segments, mode_mapping, weights, i, j, cond):
	f_list = []
	sign_list = []
	theta_list = []

	if isinstance(cond, LinearCond):
		f, sign, theta = extract_cond_features(cond)
		f_list.append(f)
		sign_list.append(sign)
		theta_list.append(theta)
	else:
		f1, sign1, theta1 = extract_cond_features(cond.mother)
		f_list.append(f1)
		sign_list.append(sign1)
		theta_list.append(theta1)

		f2, sign2, theta2 = extract_cond_features(cond.father)
		f_list.append(f2)
		sign_list.append(sign2)
		theta_list.append(theta2)

	segments_features = get_segment_features(env, segments, f_list, sign_list)

	vc = None 
	if isinstance(cond, LinearCond):
		vc = linear_vc
	elif isinstance(cond, AndCond):
		vc = and_vc 
	elif isinstance(cond, OrCond):
		vc = or_vc

	for k in range(len(f_list)):
		theta = np.copy(np.array(theta_list)).tolist()
		X = np.arange(-20, 20, 0.1)
		Y = []
		for x in X:
			theta[k] = x
			y = cost_fun(theta, vc, segments_features, mode_mapping, weights, i, j)
			Y.append(y)

		pl.plot(X, Y)
		pl.plot([theta_list[k], theta_list[k]], [0, np.max(Y)], "k--")
		pl.plot([-20, 20], [0, 0], "k--")

def visualize_cond(env, segments, mode_mapping, weights, i, j, cond):
	GX = []
	GY = []
	GS = []
	GC = []
	s_min = 1
	s_max = 20

	#env.plot_cond(cond.params)

	x_lim, y_lim = env.get_plot_limits()

	for t in segments:
		if (weights[t] < 0.1):
			continue
		skeys = segments[t].keys()
		skeys = sorted(skeys)
		for k1 in range(len(skeys)):
			p_i, p_j = get_prob(skeys, k1, mode_mapping[t], i, j)

			if p_i < 0.5:
				continue

			change = p_j > 0.5

			states = segments[t][skeys[k1]]
			if len(states) > 1:
				states = states[0]
			for s in states:
				X, Y = env.get_2d_states([s])
				if change:
					GX.append(X[0])
				else:
					GX.append(X[0] + x_lim[1] - x_lim[0])
				GY.append(Y[0])
				f = env.get_features(s)
				e = cond.eval(f)
				if change:
					GC.append('r' if e < -0.001 else 'g')
				else:
					GC.append('k' if e < -0.001 else 'y')
				s = (s_max  - s_min)*min(abs(e), 1.0) + s_min
				GS.append(s)

			X,Y = env.get_2d_states([states[-1]])
			'''if change:
				pl.scatter(X,Y, marker = 'X', c = 'g', s = 25)
			else:
				pl.scatter(X,Y, marker = '+', c = 'b', s = 25)'''

	g_plot = pl.scatter(GX, GY, marker='o', facecolors='none', alpha = 0.5)
	g_plot.set_edgecolors(GC)
	g_plot.set_sizes(GS)


	pl.xlim((x_lim[0], x_lim[1] + x_lim[1] - x_lim[0]))
	pl.ylim(y_lim)
	'''x_min = -4.0
	x_max = 8.0

	y_min = -5 # 0 + 2
	y_max = 20 # 14

	pl.xlim((x_min, x_max))
	pl.ylim((y_min, y_max))'''



