from multiprocessing import Process, Queue
import sys 

import numpy as np 
import random

from matplotlib import pyplot as pl 

from synth.policy_grammars.straight_traj_grammar import * 
from synth.policy_grammars.traj_opt_wrapper import * 

from synth.policy.prob_state_machine import * 
from synth.policy.state_machine import *

from synth.main.learn_modes import get_random_action, get_zero_action

import synth.main.globals as globals

class RefTraj:
	def __init__(self, ref_modes, ref_times, ref_conds, ref_cond_noises):
		self.modes = ref_modes
		self.times = ref_times
		self.conds = ref_conds 
		self.cond_noises = ref_cond_noises


def parallel_sample_ref_trajectories(envs, t_opts, t_pgs, actions_mean, actions_std, conds, conds_std, nm_unroll, timesteps):
	procs = []
	queue = Queue()
	for k in range(len(t_opts)):
		np.random.seed()
		random.seed()
		sm = ProbStateMachinePolicy(envs[k], actions_mean, actions_std, conds, conds_std)
		proc = Process(target = sample_ref_traj, args=[envs[k], sm, t_opts[k], nm_unroll, timesteps, k, queue])
		procs.append(proc)
		proc.start()

	ref_traj_results = {}
	for i in range(len(procs)):
		k, prob_cost_arr, sm_cost_arr, traj_prob_arr, ref_trajs_arr = queue.get()
		ref_traj_results[k] = (prob_cost_arr, sm_cost_arr, traj_prob_arr, ref_trajs_arr)


	for proc in procs:
		proc.join()


	return ref_traj_results


def sample_ref_traj(env, sm, opt, nm_unroll, timesteps, i, result_queue):
	prob_cost_arr = []
	sm_cost_arr = []
	traj_prob_arr = []

	ref_trajs_arr = []
	

	# First add the traj obtained by deterministic SM 
	dsm = StateMachinePolicy(env, sm.modes, sm.conds)
	init_goal_err = np.sum(env.check_goal(opt.init_states[0]))
	safe_err, goal_err, total_time, time_safe, total_obj, ref_modes, ref_times, ref_conds, ref_cond_noises = dsm.evaluate(opt.init_states[0], max_modes = nm_unroll, max_time_per_mode = timesteps*env.test_dt_scale, vis = False)
	sm_cost = safe_err + np.sum(goal_err)
	sm_cost_weighted = safe_err *0.01 + np.sum(goal_err)
	prob = 1.0
	prob_cost = prob * np.exp(-sm_cost_weighted)
	if not env.infinite_system:
		prob_cost *= np.exp(-total_time*env.time_weight*1.0)

	ref_modes = np.copy(ref_modes[:nm_unroll]).tolist()
	ref_times = (np.copy(ref_times[:nm_unroll])*env.dt).tolist()
	ref_conds = ref_conds[:nm_unroll]
	ref_cond_noises = ref_cond_noises[:nm_unroll]

	l = len(ref_modes)

	if sm_cost < 0.1:
		# fill left over modes with zeros at the end
		for tt in range(l, nm_unroll, 1):
			idx = len(ref_modes) # random.randint(0, len(ref_modes))
			ref_modes.insert(idx, get_zero_action(env))
			ref_times.insert(idx, 0.0)
			ref_conds.insert(idx, None)
			ref_cond_noises.insert(idx, 0.0)
	else:
		# fill left over modes with random and at random places
		for tt in range(l, nm_unroll, 1):
			idx = random.randint(0, len(ref_modes))
			ref_modes.insert(idx, get_random_action(env))
			#ref_modes.insert(idx, np.copy(random.choice(sm.modes)))
			ref_times.insert(idx, 0.0)
			ref_conds.insert(idx, None)
			ref_cond_noises.insert(idx, 0.0)

	prob_cost_arr.append(prob_cost)
	sm_cost_arr.append(sm_cost)
	traj_prob_arr.append(prob)

	ref_traj = RefTraj(ref_modes, ref_times, ref_conds, ref_cond_noises)
	ref_trajs_arr.append(ref_traj)
	

	
	for r in range(100):
		init_goal_err = np.sum(env.check_goal(opt.init_states[0]))
		safe_err, goal_err, total_time, total_obj, ref_modes, ref_times, ref_conds, ref_cond_noises, prob = sm.evaluate(opt.init_states[0], max_modes = nm_unroll, max_time_per_mode = timesteps*env.test_dt_scale, vis = False)

		sm_cost = safe_err + np.sum(goal_err)
		sm_cost_weighted = safe_err * 0.01 + np.sum(goal_err)
		prob_cost = prob * np.exp(-sm_cost_weighted)
		if not env.infinite_system:
			prob_cost *= np.exp(-total_time*env.time_weight*1.0)

		ref_modes = np.copy(ref_modes[:nm_unroll]).tolist()
		ref_times = (np.copy(ref_times[:nm_unroll])*env.dt).tolist()
		ref_conds = ref_conds[:nm_unroll]
		ref_cond_noises = ref_cond_noises[:nm_unroll]

		'''print("SM cost: ", sm_cost)
		print("Ref modes: ", ref_modes)
		print("Ref times: ", ref_times)'''
		l = len(ref_modes)

		if sm_cost < 0.1:
			# fill left over modes with zeros at the end
			for tt in range(l, nm_unroll, 1):
				idx = len(ref_modes) # random.randint(0, len(ref_modes))
				ref_modes.insert(idx, get_zero_action(env))
				ref_times.insert(idx, 0.0)
				ref_conds.insert(idx, None)
				ref_cond_noises.insert(idx, 0.0)
		else:
			# fill left over modes with random and at random places
			for tt in range(l, nm_unroll, 1):
				idx = random.randint(0, len(ref_modes))
				ref_modes.insert(idx, get_random_action(env))
				ref_times.insert(idx, 0.0)
				ref_conds.insert(idx, None)
				ref_cond_noises.insert(idx, 0.0)

		prob_cost_arr.append(prob_cost)
		sm_cost_arr.append(sm_cost)
		traj_prob_arr.append(prob)

		ref_traj = RefTraj(ref_modes, ref_times, ref_conds, ref_cond_noises) 
		ref_trajs_arr.append(ref_traj)
		

	prob_cost_arr, sm_cost_arr, traj_prob_arr, ref_trajs_arr = zip(*sorted(zip(prob_cost_arr, sm_cost_arr, traj_prob_arr, ref_trajs_arr), reverse = True)[:10])

	result_queue.put([i, prob_cost_arr, sm_cost_arr, traj_prob_arr, ref_trajs_arr])

def set_ref_traj(env, opt, pg, num_timesteps, prob_cost_arr, sm_cost_arr, traj_prob_arr, ref_trajs_arr):

	print("Best prob cost: ", prob_cost_arr)
	print("Best SM cost: ", sm_cost_arr)
	print("Best prob: ", traj_prob_arr)

	ref_modes = ref_trajs_arr[0].modes
	ref_times = ref_trajs_arr[0].times 

	print("Ref modes: ", ref_modes)
	print("Ref times: ", ref_times)
	print("Ref conds: ", ref_trajs_arr[0].conds )
	print("Ref cond noises: ", ref_trajs_arr[0].cond_noises)

	pg.set_ref_trajs(ref_trajs_arr)
	pg.set_ref_prob(traj_prob_arr)



	# plot the traj obtained from sm policy
	opt.set_opt_mode(VarsMode.All, ErrMode.All, [], [1.0, 1.0, 1.0, 1.0])
			
	modes = np.array(ref_modes).flatten()
	dts = np.array(ref_times)/float(num_timesteps)*100*env.dt_scale
	new_x = np.append(modes, dts)
	full_x = opt.get_full_x(new_x)

	policy = pg.get_policy(full_x)[0]
	globals.fig_add_subplot()
	safe_err, goal_err, mode_err, cond_err, total_time, total_obj = policy.evaluate(opt.init_states[0], vis = True)
	pl.title(str(round(safe_err, 1)) + " " + str(round(np.sum(goal_err), 1)) + " " + str(round(mode_err, 1)) + " " + str(round(cond_err, 1)) + " " + str(round(total_obj, 1)) )

	opt.set_full_x(new_x)

	return safe_err + np.sum(goal_err), mode_err, cond_err

