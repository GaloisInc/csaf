from multiprocessing import Process, Queue
import sys 

import numpy as np 
import random

from matplotlib import pyplot as pl 


from synth.policy_grammars.straight_traj_grammar import * 
from synth.policy_grammars.straight_traj_pomdp_grammar import *
from synth.policy_grammars.traj_time_grammar import * 
from synth.policy_grammars.traj_opt_wrapper import * 

from synth.optimizers.scipy_optimize import * 
from synth.optimizers.pyopt_search import *
from synth.main.learn_modes import get_random_action

import synth.main.globals as globals

def init_threads_for_opt(num_threads, envs, nm, timesteps):
	t_pgs = []
	t_opts = []
	t_searches = [] 

	for i in range(num_threads):
		pg = StraightTrajGrammar(envs[i], nm, timesteps)
		#pg = TrajTimeGrammar(envs[i], nm, timesteps)
		safe_live_opt = TrapOptWrapper(envs[i], pg, 1)
		pysearch = PyOptSearch(safe_live_opt)
		#pysearch = ScipySearch(safe_live_opt)

		safe_live_opt.init_full_policy()
		
		t_pgs.append(pg)
		t_opts.append(safe_live_opt)
		t_searches.append(pysearch)

	return t_pgs, t_opts, t_searches

def run_traj_opt1(search, opt, num_iter, random_init, id, queue):
	min_x = None 
	min_cost = 1e30
	if random_init: 
		for i in range(5):
			x1, cost1, it = search.minimize(num_iter, random_init = random_init, vis = False)
			if cost1 < 0.02:
				queue.put([id, x1])
				return
			if cost1 < min_cost:
				min_cost = cost1
				min_x = x1 
		queue.put([id, min_x])
		return
	else:
		x1, cost1, it = search.minimize(num_iter, random_init = random_init, vis = False)
		print(cost1, id)
		queue.put([id, x1])
		return 

	opt.set_full_x(x1)
	policy = opt.get_policy()[0]
	safe_err, goal_err, mode_err, cond_err, total_time, total_obj  = policy.evaluate(opt.init_states[0], vis = True, id = id )
	original_cost = safe_err + np.sum(goal_err) + mode_err + cond_err
	original_dts = np.copy(policy.dts)

	remove_idx = []
	for k in range(len(original_dts)):
		policy.dts[k] = 0.0 
		safe_err, goal_err, mode_err, cond_err, total_time, total_obj  = policy.evaluate(opt.init_states[0], vis = False, id = id )
		cost = safe_err + np.sum(goal_err) + mode_err + cond_err
		policy.dts[k] = original_dts[k]
		if cost < original_cost + 1.0:
			remove_idx.append(k)

	x = np.copy(x1)
	for i in remove_idx:
		x[i-len(original_dts)] = 0
	opt.set_full_x(x)
	x, cost, it = search.minimize(20, random_init = False, vis = False)
	print("Final: ", cost, id)
	queue.put([id, x if cost <= cost1 else x1])

def run_traj_opt(search, opt, num_iter, random_init, id, queue):
	x, cost, it = search.minimize(num_iter, random_init = random_init, vis = False)
	print(cost, id)
	queue.put([id, x])


def joint_sample_trajs(t_pgs, t_opts, nm_sm):
	min_total_cost = 1e30 
	min_min_xs = None
	for i in range(10):
		# sample sm modes
		sm_modes = []
		for t in range(nm_sm):
			mode = get_random_action(t_pgs[0].env)
			sm_modes.append(mode)

		total_cost = 0
		min_xs = []
		for j in range(len(t_pgs)):
			pg = t_pgs[j]
			opt = t_opts[j]
			opt.set_opt_mode(VarsMode.All, ErrMode.All, [], [1, 1, 0, 0] )
			min_cost = 1e30
			min_x = None
			for k in range(5):
				# sample mode order
				modes = []
				for t in range(pg.num_modes):
					modes.append(random.choice(sm_modes))
				modes = np.array(modes)
				# sample times 
				dts = pg.get_random_dt_vars()
				# select min cost 
				x = np.append(modes.flatten(), dts)
				cost, _ = opt.get_cost(x)
				if cost < min_cost:
					min_cost = cost
					min_x = x 


			# sum total min cost 
			total_cost += min_cost 
			min_xs.append(min_x)

		# select min total min cost  
		if total_cost < min_total_cost:
			min_total_cost = total_cost 
			min_min_xs = min_xs

	for j in range(len(t_pgs)):
		t_opts[j].set_full_x(min_min_xs[j])


def optimize_trajs(t_pgs, t_opts, t_searches, mode_weights, cond_weights, random_inits):
	print("Mode weights: ", mode_weights)
	print("Cond weights: ", cond_weights)

	for k in range(len(t_opts)):
		opt = t_opts[k]
		opt.set_opt_mode(VarsMode.All, ErrMode.All, [], [1, 1, mode_weights[k], cond_weights[k]] )

	procs = []
	queue = Queue()
	for k in range(len(t_pgs)):
		random_init = random_inits[k]
		num_iter = 100
		np.random.seed()
		random.seed()

		proc = Process(target = run_traj_opt, args=[t_searches[k], t_opts[k], num_iter, random_init, k, queue])
		procs.append(proc)
		proc.start()

	

	costs = []
	weights = [] 
	for i in range(len(t_pgs)):
		costs.append(0.0)
		weights.append(0.0)

	cur_plot_idx = globals.get_cur_row()

	for i in range(len(procs)):
		r = queue.get()
		#print("From queue: ", r)
		id,x0 = r 
		opt = t_opts[id]
		
		# visualize x after optimizing mode dt 
		opt.set_full_x(x0)
		globals.fig_add_subplot1(cur_plot_idx + id + 1)
		policy = opt.get_policy()[0]
		#policy.cleanup_modes(opt.init_states[0])
		safe_err, goal_err, mode_err, cond_err, total_time, total_obj  = policy.evaluate(opt.init_states[0], vis = True, id = id )

		cost = safe_err + np.sum(goal_err) + mode_err + cond_err
		costs[id] = cost

		weights[id] = safe_err + np.sum(goal_err) 
		random_txt = "R " if random_inits[id] else " " 
		pl.title(random_txt + str(round(safe_err, 1)) + " " + str(round(np.sum(goal_err), 1))  +  " " + str(round(mode_err, 1)) + " " + str(round(cond_err, 1)) + " " + str(round(total_obj, 1)))

		#opt.set_full_x(x1)
		#globals.fig_add_subplot1(cur_plot_idx + 2*id + 2)
		#policy = opt.get_policy()[0]
		#policy.cleanup_modes(opt.init_states[0])
		#safe_err, goal_err, mode_err, cond_err, total_time, total_obj  = policy.evaluate(opt.init_states[0], vis = True, id = id )

	min_cost = np.min(weights)
	for i in range(len(weights)):
		w = weights[i]
		weights[i] = np.exp(-(w - min_cost)*10.0)

	print("Weights: ", weights)


	for proc in procs:
		proc.join()

	globals.fig_new_row()
			
	trajs = []
	init_states = []
	for opt in t_opts:
		traj = opt.get_policy()[0]
		traj.cleanup_modes(opt.init_states[0])
		trajs.append(traj)
		init_states.extend(opt.init_states)
		print("Traj x: ", opt.full_x)

	return trajs, init_states, costs, weights
