import numpy as np 
import random
import sys 
import time 
import os 
from matplotlib import pyplot as pl 

from general.utils import * 

from synth.main.learn_conds import * 
from synth.main.learn_modes import * 
from synth.main.traj_opt import * 
from synth.main.ref_trajs_sampler import * 

from synth.main.sm_utils import * 
import synth.main.globals as globals

from synth.policy.prob_state_machine import * 
from synth.policy.state_machine import *

from synth.main.bench_params import * 

import cProfile 

def opt_all(dir, synth_params, gen_params, resample_env = False, vis = False):
	print("in opt_all...")
	if not os.path.exists(dir):
		os.makedirs(dir)

	envs = synth_params.envs
	nm_unroll = synth_params.nm_unroll
	nm_sm = synth_params.nm_sm
	timesteps = synth_params.timesteps 
	cond_depth = synth_params.cond_depth

	overall_start_time = time.time()

	traj_opt_time = 0.0 
	learn_modes_time = 0.0
	learn_cond_time = 0.0 
	sm_eval_time = 0.0 
	traj_sampler_time = 0.0

	niters = 20
	nex = 10
	max_threads = len(envs)
	#assert(max_threads >= nex)

	it_weights = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0, 1.0]

	globals.ncol = 10 # max(nex, 2*nm_sm - 1)
	globals.nrow = 6

	num_traj_threads = min(nex, max_threads)

	t_pgs, t_opts, t_searches = init_threads_for_opt(num_traj_threads, envs, nm_unroll, timesteps)

	#test_psm(envs[0], t_opts[0], t_pgs[0], nm_unroll, timesteps, "out/cp_sm_4.txt")
	#assert(False)

	min_cost = 1e30
	min_sm = None 
	min_cost_it = -1
	restart_it = 0

	prev_trajs = [] 
	for k in range(num_traj_threads):
		prev_trajs.append(None)

	mode_weights = []
	for k in range(num_traj_threads):
		mode_weights.append(0.0)

	cond_weights = [] 
	for k in range(num_traj_threads):
		cond_weights.append(0.00)

	prev_sm = None
	sm = None

	reintialize = False

	for it in range(niters):
		print("Iteration %i"%it)

		np.random.random()
		random.random()

		globals.plt_idx = 1
		globals.fig = pl.figure(figsize = (20,15))

		if not reintialize and it > 0:
			if resample_env:
				print("Resampling env")
				for i in range(len(t_opts)):
					t_opt = t_opts[i]
					t_opt.init_states[0] = random.choice(failed_states[:100])[1]
					#t_opt.sample_env()


			print("Get reference trajectories")	
			traj_sampler_start_time = time.time()

			ref_traj_results = parallel_sample_ref_trajectories(envs, t_opts, t_pgs, actions_mean, actions_std, conds, conds_std, nm_unroll, timesteps)

			# process reference trajectories
			for k in range(len(t_opts)):
				prob_cost_arr, sm_cost_arr, traj_prob_arr, ref_trajs_arr = ref_traj_results[k]
				safe_goal_err, mode_err, cond_err = set_ref_traj(envs[k], t_opts[k], t_pgs[k], timesteps, prob_cost_arr, sm_cost_arr, traj_prob_arr, ref_trajs_arr)

				cost = safe_goal_err + mode_err + cond_err
				if cond_err < 0.1 and safe_goal_err < 0.03:
					cond_weights[k] = 1.0
				else:
					cond_weights[k] = 0.01

				if mode_err < 0.1 and safe_goal_err < 0.03:
					mode_weights[k] = 1.0
				else:
					mode_weights[k] = 0.01

				#cond_weights[k] = it_weights[it]
				#mode_weights[k] = it_weights[it]



				# check if trajectories have made progress 
				traj = t_opts[k].get_policy()[0]
				old_traj = prev_trajs[k]
				if cost > 0.01 and old_traj != None:
					diff = traj.diff(old_traj)
					print("Diff: ", diff)
					if  (diff < 0.01):
						prev_trajs[k] = None
					else:
						prev_trajs[k] = traj
				else:
					prev_trajs[k] = traj

			traj_sampler_time += time.time() - traj_sampler_start_time
			globals.fig_new_row()

		#print("Init states")	
		#for t_opt in t_opts:
		#	print(t_opt.init_states[0])

		#if reintialize or it == 0:
		#	print("Initial sampling of trajectories")
		#	joint_sample_trajs(t_pgs, t_opts, nm_sm)

		print("Optimizing trajectories for safe goal")
		traj_opt_start_time = time.time()

		random_inits = [p == None for p in prev_trajs]  
		print("Random inits: ", random_inits)
		trajs, init_states, costs, t_weights = optimize_trajs(t_pgs, t_opts, t_searches, mode_weights, cond_weights, random_inits)
		
		traj_opt_time += time.time() - traj_opt_start_time


		print("Learn mode mapping")
		learn_modes_start_time = time.time()

		actions_mean, actions_std, mode_mapping = learn_modes_n_mapping(envs[0], init_states, trajs, t_weights, None, nm_sm)
		print("Mode mapping: ", mode_mapping)

		learn_modes_time += time.time() - learn_modes_start_time	
		
		print("Learning switch conds")
		learn_cond_start_time = time.time()

		conds, conds_std = optimize_conds(envs[0], init_states, trajs,  mode_mapping, nm_sm, t_weights, cond_depth)

		learn_cond_time += time.time() - learn_cond_start_time



		sm = ProbStateMachinePolicy(envs[0], actions_mean, actions_std, conds, conds_std)
		sm.save(dir + "/sm_%i.txt"%(it))


		print("Evaluating current SM")	
		sm_eval_start_time = time.time()

		dsm = StateMachinePolicy(envs[0], actions_mean, conds)
		visualize_sm(envs[0], dsm, init_states, nm_unroll, timesteps)
		pl.tight_layout()
		pl.savefig(dir + "/%i.png"%(it))
		#pl.show()
		pl.close()
		#assert(False)

		total_cost, failed_states = evaluate_state_machine1(envs, actions_mean, conds, nm_unroll, timesteps)
		print("Total cost: ", total_cost)

		sm_eval_time += time.time() - sm_eval_start_time
		
		reintialize = False

		if total_cost < min_cost:
			min_cost = total_cost 
			min_sm = sm 
			min_cost_it = it 

		if total_cost < 0.01:
			break

		if (total_cost > min_cost and it - min_cost_it >= 2) or (min_cost > 1.0 and it - restart_it >= 3):
			if min_cost < 0.1:
				break 
			else:
				restart_it = it
				min_cost_it = it
				reintialize = True
				print("Reintializing state machine")

		prev_sm = dsm 

		overall_time = time.time() - overall_start_time
		print("Overall time:", overall_time)
		print("Traj opt time:", traj_opt_time)
		print("Learn modes time:", learn_modes_time)
		print("Learn conds time:", learn_cond_time)
		print("SM eval time:", sm_eval_time)
		print("Ref traj sampler time:", traj_sampler_time)

		
	overall_time = time.time() - overall_start_time

	print("Min SM cost: ", min_cost)
	min_sm.save(dir + "/sm_min.txt")

	#min_sm = ProbStateMachinePolicy(envs[0], [], [], [], [])
	#min_sm.read("out/cp_sm_min.txt")

	'''min_dsm = StateMachinePolicy(envs[0], min_sm.modes, min_sm.conds)

	# evaluate generalization error 
	# evaluate error on train distribution
	safe_fraction, avg_time_safe, avg_goal_error, avg_time_goal = evaluate_state_machine_test(envs, min_sm.modes, min_sm.conds, gen_params.max_modes, gen_params.timesteps, vis = False)
	print("Generalization on train set: ", safe_fraction, avg_time_safe, avg_goal_error, avg_time_goal)


	if gen_params.inp_limits != None:
		for env in envs:
			env.set_inp_limits(gen_params.inp_limits)

		safe_fraction, avg_time_safe, avg_goal_error, avg_time_goal = evaluate_state_machine_test(envs, min_sm.modes, min_sm.conds, gen_params.max_modes, gen_params.timesteps, vis = False)
		print("Generalization on test set: ", safe_fraction, avg_time_safe, avg_goal_error, avg_time_goal)

	# visualize some test scenarios 
	globals.plt_idx = 1
	globals.fig = pl.figure(figsize = (15,15))
	globals.nrow = 4
	globals.ncol = 5

	gen_init_states = []
	for i in range(20):
		gen_init_states.append(envs[0].sample_init_state())

	visualize_sm(envs[0], min_dsm, gen_init_states, gen_params.max_modes, gen_params.timesteps)

	pl.tight_layout()
	pl.savefig(dir + "/gen.png")
	pl.close()


	print("Overall time:", overall_time)
	print("Traj opt time:", traj_opt_time)
	print("Learn modes time:", learn_modes_time)
	print("Learn conds time:", learn_cond_time)
	print("SM eval time:", sm_eval_time)
	print("Ref traj sampler time:", traj_sampler_time)'''

	
def run(params):
	print("in run...")
	if len(params) == 4:
		name, run_id, resample_env, vis = params
		run_id = int(run_id)
		resample_env = int(resample_env)
		vis = int(vis)
	elif len(params) == 2:
		name, run_id = params 
		run_id = int(run_id)
		resample_env = True
		vis = False
	else:
		assert(False)
	opt = opt_all

	synth_params, gen_params = get_bench_params(name, num_threads = 1)
	print("here....")
	dirname = "out/%s_%i"%(name, run_id)
	opt_all(dirname, synth_params, gen_params, resample_env = resample_env, vis = vis)


if __name__ == '__main__':
	seed = None
	#seed = 2921275267
	np.random.seed(seed)
	print("np seed: ", np.random.get_state()[1][0])
	seed = random.randrange(sys.maxsize)
	#seed = 1070761973242435720
	random.seed(seed)
	print("random seed: ", seed)
	params = sys.argv[1:]

	run(params)
		#cProfile.run('run("%s")'%p, 'restats')



