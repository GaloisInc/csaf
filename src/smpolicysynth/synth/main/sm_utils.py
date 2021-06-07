
from multiprocessing import Process, Queue
import sys 

import numpy as np 
import random

from matplotlib import pyplot as pl 


from synth.policy_grammars.sm_grammar import * 
from synth.policy_grammars.straight_traj_grammar import * 
from synth.policy_grammars.traj_opt_wrapper import * 

from synth.policy.prob_state_machine import * 
from synth.policy.state_machine import *

from synth.main.learn_modes import get_random_action, get_zero_action


import synth.main.globals as globals


def test_psm(env, opt, pg, nm_unroll, timesteps, name):

	globals.plt_idx = 1
	globals.fig = pl.figure(figsize = (5,10))
	globals.nrow = 2
	globals.ncol = 1

	dsm = StateMachinePolicy(env, [], [])
	dsm.read(name)
	init_state = env.sample_init_state()
	globals.fig_add_subplot()
	safe_err, goal_err, total_time, time_safe, total_obj, ref_modes, ref_times, ref_conds, ref_cond_noises = dsm.evaluate(init_state, max_modes = 10, max_time_per_mode = timesteps *env.test_dt_scale, vis = True)
	

	ref_modes = np.copy(ref_modes[:nm_unroll]).tolist()
	ref_times = (np.copy(ref_times[:nm_unroll])*env.dt).tolist()

	print(ref_modes)
	print(ref_times)

	l = len(ref_modes)

	for tt in range(l, nm_unroll, 1):
		idx = len(ref_modes) # random.randint(0, len(ref_modes))
		ref_modes.insert(idx, get_zero_action(env))
		ref_times.insert(idx, 0.0)
	
	opt.set_opt_mode(VarsMode.All, ErrMode.All, [], [1.0, 1.0, 1.0, 1.0])
			
	modes = np.array(ref_modes).flatten()
	dts = np.array(ref_times)/float(timesteps)*100*env.dt_scale
	print(dts)
	new_x = np.append(modes, dts)
	full_x = opt.get_full_x(new_x)

	policy = pg.get_policy(full_x)[0]
	globals.fig_add_subplot()
	safe_err, goal_err, mode_err, cond_err, total_time, total_obj = policy.evaluate(init_state, vis = True)
	pl.title(str(round(safe_err, 1)) + " " + str(round(np.sum(goal_err), 1)) + " " + str(round(mode_err, 1)) + " " + str(round(cond_err, 1)) + " " + str(round(total_obj, 1)) )


	pl.show()
	pl.close()


def evaluate_state_machine(env, sm, nm_unroll, num_timesteps):
	total_cost = 0.0 

	for i in range(100):
		init_state = env.sample_init_state()
		safe_error, goal_error, total_time, time_safe, total_obj, _, _,_,_  = sm.evaluate(init_state, max_modes = nm_unroll, max_time_per_mode = num_timesteps*env.test_dt_scale, vis = False)

		total_cost += safe_error + np.sum(goal_error)

	total_cost = total_cost/float(100)

	return total_cost 

def visualize_sm(env, sm, init_states, nm_unroll, num_timesteps):
	total_cost = 0.0 

	for init_state in init_states:
		globals.fig_add_subplot()
		safe_error, goal_error, total_time, time_safe, total_obj, _, _ ,_,_ = sm.evaluate(init_state, max_modes = nm_unroll, max_time_per_mode = num_timesteps*env.test_dt_scale, vis = True)
		total_cost += safe_error + np.sum(goal_error)

		pl.title( str(round(safe_error, 2)) + " " + str(round(np.sum(goal_error), 2)) + " " + str(round(np.sum(total_obj), 2)))
	total_cost = total_cost/float(len(init_states))
	print("SM cost on init states: ", total_cost)


def parallel_eval_sm(env, sm, nm_unroll, timesteps, vis, result_queue):
	total_cost = 0.0 

	failed_states = []

	for i in range(10):
		init_state = env.sample_init_state()
		init_state_cp = np.copy(init_state)
		safe_error, goal_error, total_time, time_safe, total_obj, _, _,_,_  = sm.evaluate(init_state, max_modes = nm_unroll, max_time_per_mode = timesteps*env.test_dt_scale, vis = False)

		goal_error = np.sum(goal_error)

		total_cost += safe_error + goal_error 
		failed_states.append((safe_error + goal_error , init_state_cp))

		if vis:
			s = str(init_state);
			s += "\t";
			s += GREEN + "SAFE" + ENDC if safe_error < 0.02 else RED + "UNSAFE" + ENDC
			s += " (" + str(round(safe_error,2)) + ") "
			s += "\t"
			s += GREEN + "PASS" + ENDC if goal_error < 0.1 else RED + "FAIL" + ENDC
			s += " (" + str(round(goal_error,2)) + ") "
			s += "\n"
			print(s)

	total_cost = total_cost/10.0
	result_queue.put((total_cost, failed_states))


def evaluate_state_machine1(envs, actions_mean, conds, nm_unroll, timesteps, vis = False):
	procs = []
	queue = Queue()
	for env in envs:
		random.seed()
		np.random.seed()

		sm = StateMachinePolicy(env, actions_mean, conds)

		proc = Process(target = parallel_eval_sm, args=[env, sm, nm_unroll, timesteps, vis, queue])
		procs.append(proc)
		proc.start()

	
	total_cost = 0.0 
	all_failed_states = []
	for i in range(len(procs)):
		cost, failed_states = queue.get()
		total_cost += cost 
		all_failed_states.extend(failed_states)

	for proc in procs:
		proc.join()


	total_cost = total_cost/float(len(envs))

	all_failed_states = sorted(all_failed_states, reverse=True, key=lambda x: (x[0]))

	return total_cost, all_failed_states 



def parallel_eval_sm_test(env, sm, nm_unroll, timesteps, vis, result_queue):
	total_cost = 0.0 

	num_safe = 0
	total_goal_err = 0
	total_time_safe = 0
	total_time_to_goal = 0
	total_correct = 0
	for i in range(100):
		init_state = env.sample_init_state()
		safe_error, goal_error, total_time, time_safe, total_obj, _, _, _, _  = sm.evaluate(init_state, max_modes = nm_unroll, max_time_per_mode = timesteps*env.test_dt_scale, vis = False)

		goal_error = np.sum(goal_error)

		if safe_error <= 0.05:
			num_safe += 1
			if goal_error < 0.1:
				total_correct += 1 

		total_time_safe += time_safe

		total_goal_err += goal_error

		total_time_to_goal += total_time 

		if vis:
			s = str(init_state);
			s += "\t";
			s += GREEN + "SAFE" + ENDC if safe_error < 0.02 else RED + "UNSAFE" + ENDC
			s += " (" + str(round(safe_error,2)) + ") "
			s += "\t"
			s += GREEN + "PASS" + ENDC if goal_error < 0.1 else RED + "FAIL" + ENDC
			s += " (" + str(round(goal_error,2)) + ") "
			s += "\n"
			print(s)

	total_goal_err = total_goal_err/100.0
	num_safe = num_safe/100.0
	total_time_safe = total_time_safe/100.0
	total_time_to_goal = total_time_to_goal/100.0
	total_correct = total_correct/100.0

	result_queue.put((num_safe, total_time_safe, total_goal_err, total_time_to_goal, total_correct))


def evaluate_state_machine_test(envs, actions_mean, conds, nm_unroll, timesteps, vis = False):
	procs = []
	queue = Queue()
	for env in envs:
		np.random.seed()
		random.seed()

		sm = StateMachinePolicy(env, actions_mean, conds)

		proc = Process(target = parallel_eval_sm_test, args=[env, sm, nm_unroll, timesteps, vis, queue])
		procs.append(proc)
		proc.start()

	for proc in procs:
		proc.join()

	num_safe = 0.0
	total_goal_err = 0.0 
	total_time_safe = 0.0
	total_time_goal = 0.0
	total_correct = 0.0
	while not queue.empty():
		safe, time_safe, goal, time_goal, correct = queue.get()
		num_safe += safe 
		total_goal_err += goal 
		total_time_safe += time_safe 
		total_time_goal += time_goal
		total_correct += correct 

	l = float(len(envs))
	total_goal_err = total_goal_err/l
	num_safe = num_safe/l
	total_time_safe = total_time_safe/l
	total_time_goal = total_time_goal/l
	total_correct = total_correct/l
	print("Total correct", total_correct)

	return num_safe, total_time_safe, total_goal_err, total_time_goal

