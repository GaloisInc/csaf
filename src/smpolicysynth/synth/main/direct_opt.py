import numpy as np 
import random
import sys 
from matplotlib import pyplot as pl 

from general.simulate import *
from general.utils import * 


from synth.main.sm_utils import * 
import synth.main.globals as globals 

from synth.policy.state_machine import *

from synth.policy_grammars.sm_grammar import * 
from synth.policy_grammars.sm_opt_wrapper import * 
from synth.policy_grammars.bool_cond_d1_grammar import * 
from synth.policy_grammars.bool_cond_d2_grammar import * 

from synth.optimizers.pyopt_search import * 

from synth.main.bench_params import * 

def init_threads_for_sm_opt(num_threads, envs, nm_sm, cond_grammar, nm_unroll, timesteps):
	t_pgs = []
	t_opts = []
	t_searches = [] 

	for i in range(num_threads):
		pg = SMGrammar(envs[i], nm_sm, cond_grammar, nm_unroll, timesteps)
		safe_live_opt = SMOptWrapper(envs[i], pg, 10)
		pysearch = PyOptSearch(safe_live_opt)

		safe_live_opt.init_full_policy()
		
		t_pgs.append(pg)
		t_opts.append(safe_live_opt)
		t_searches.append(pysearch)

	return t_pgs, t_opts, t_searches

MAX_TIME = 3600 # 1 hour

def run_sm_opt(env, search, opt, nm_unroll, timesteps, id, queue):
	min_cost = 1e30
	min_total_cost = 1e30
	min_sm = None 
	min_total_sm = None 

	random_init = True 

	start_time = time.time()

	for k in range(100):
		opt.sample_env()
		x, cost, it = search.minimize(500, random_init = True, vis = False)
		opt.set_full_x(x)
		sm = opt.get_policy()[0]
		total_cost = evaluate_state_machine(env, sm, nm_unroll, timesteps)
		print("Total cost: ", total_cost)
		if total_cost < min_total_cost:
			min_total_cost = total_cost 
			min_total_sm = sm

		if total_cost < 0.01:
			break
		elif total_cost < 0.5:
			random_init = False 
		else:
			random_init = True 

		if time.time() - start_time > MAX_TIME: 
			break 

	
	queue.put((id, min_total_sm, min_total_cost))
	return 



def opt_sm(dir, synth_params, gen_params, cond_grammar, resample_env= False, vis = False):
	if not os.path.exists(dir):
		os.makedirs(dir)

	envs = synth_params.envs
	nm_unroll = synth_params.nm_unroll
	nm_sm = synth_params.nm_sm
	timesteps = synth_params.timesteps 
	cond_depth = synth_params.cond_depth

	overall_start_time = time.time()

	max_threads = len(envs)
	num_traj_threads = max_threads

	#cond_grammar = BoolCondD1Grammar(envs[0])

	t_pgs, t_opts, t_searches = init_threads_for_sm_opt(num_traj_threads, envs, nm_sm, cond_grammar, nm_unroll, timesteps)


	for k in range(len(t_opts)):
		opt = t_opts[k]
		opt.set_opt_mode(VarsMode.All, ErrMode.All, [], [1, 1, 0, 0] )

	
	procs = []
	queue = Queue()
	for k in range(len(t_pgs)):
		np.random.seed()
		random.seed()

		proc = Process(target = run_sm_opt, args=[envs[k], t_searches[k], t_opts[k], nm_unroll, timesteps, k, queue])
		procs.append(proc)
		proc.start()

	
	best_cost = 1e30
	best_sm = None 

	for i in range(len(procs)):
		r = queue.get()
		#print("From queue: ", r)
		id, sm, total_cost = r 
		if total_cost < best_cost:
			best_cost = total_cost 
			best_sm = sm 

			if best_cost < 0.01 :
				for proc in procs:
					proc.terminate()
				break
		
	for proc in procs:
		proc.join()

		
	overall_time = time.time() - overall_start_time

	print("Min SM cost: ", best_cost)
	best_sm.save(dir + "/sm_min.txt")

	# evaluate generalization error 
	# evaluate error on train distribution
	safe_fraction, avg_time_safe, avg_goal_error, avg_time_goal = evaluate_state_machine_test(envs, best_sm.modes, best_sm.conds, gen_params.max_modes, gen_params.timesteps, vis = False)
	print("Generalization on train set: ", safe_fraction, avg_time_safe, avg_goal_error, avg_time_goal)

	# visualize some train scenarios 
	globals.plt_idx = 1
	globals.fig = pl.figure(figsize = (15,15))
	globals.nrow = 4
	globals.ncol = 5

	init_states = []
	for i in range(20):
		init_states.append(envs[0].sample_init_state())

	visualize_sm(envs[0], best_sm, init_states, gen_params.max_modes, gen_params.timesteps)

	pl.tight_layout()
	pl.savefig(dir + "/train.png")
	pl.close()


	if gen_params.inp_limits != None:
		for env in envs:
			env.set_inp_limits(gen_params.inp_limits)

		safe_fraction, avg_time_safe, avg_goal_error, avg_time_goal = evaluate_state_machine_test(envs, best_sm.modes, best_sm.conds, gen_params.max_modes, gen_params.timesteps, vis = False)
		print("Generalization on test set: ", safe_fraction, avg_time_safe, avg_goal_error, avg_time_goal)

	# visualize some test scenarios 
	globals.plt_idx = 1
	globals.fig = pl.figure(figsize = (15,15))
	globals.nrow = 4
	globals.ncol = 5

	gen_init_states = []
	for i in range(20):
		gen_init_states.append(envs[0].sample_init_state())

	visualize_sm(envs[0], best_sm, gen_init_states, gen_params.max_modes, gen_params.timesteps)

	pl.tight_layout()
	pl.savefig(dir + "/test.png")
	pl.close()

	print("Overall time:", overall_time)
	
def run(name):
	params = name.split("_")

	run_id = int(params[-3])
	resample_env = bool(int(params[-2]))
	vis = bool(params[-1])
	name = params[0]
	synth_params, gen_params = get_bench_params(name, num_threads = 10)
	cond_grammar = BoolCondD1Grammar(synth_params.envs[0])
	if name in ["cp", "pen", "acrobot"]:
		cond_grammar = BoolCondD2Grammar(synth_params.envs[0])
	dirname = "out/diropt_%s_%i"%(name, run_id)
	opt_sm(dirname, synth_params, gen_params, cond_grammar, resample_env = resample_env, vis = vis)


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

	for p in params:
		run(p)
