import sys 
from matplotlib import pyplot as pl 

from general.simulate import *
from general.utils import * 

from synth.main.sm_utils import * 
import synth.main.globals as globals

from synth.policy.state_machine import *
from synth.policy.random_policy import *
from synth.policy.random_k_mode_policy import *

from synth.main.bench_params import * 


def run(synth_params, gen_params, inp_limits, args):
	envs = synth_params.envs
	env = envs[0]
	action_bound = 5.0
	action_dim = env.num_actions

	if len(inp_limits) > 0.0:
		for e in envs:
			e.set_inp_limits(inp_limits)
	
	if args[0] == "RP":
		policy = RandomPolicy(action_dim, action_bound)
		init_state = env.sample_init_state()
		print(init_state)
		simulate(env, policy, init_state, True)

	if args[0] == "RKMP":
		n_modes = int(args[1])
		max_time = float(args[2])
		policy = RandomKModePolicy(action_dim, action_bound, n_modes, max_time)
		init_state = env.sample_init_state()
		print(init_state)
		simulate(env, policy, init_state, True)

	if args[0] == "SM":
		name = args[1]
		SM = StateMachinePolicy(env, [], [])
		SM.read(name)
		init_state  = env.sample_init_state()
		print(init_state)
		states, safe_err, goal_err, *_ = SM.get_traj_from_sm(env, init_state, max_modes = gen_params.max_modes, max_time_per_mode = gen_params.timesteps*env.test_dt_scale, max_timesteps = 50000)
		simulate_from_states(env, states, True)
		fig = pl.figure(figsize = (3, 6))
		env.plot_init(states[0])
		env.plot_states(states)
		pl.tight_layout()
		pl.show()

	if args[0] == "GIF":
		name = args[1]
		SM = StateMachinePolicy(env, [], [])
		SM.read(name)
		init_state  = env.sample_init_state()
		print(init_state)
		states, safe_err, goal_err, *_ = SM.get_traj_from_sm(env, init_state, max_modes = gen_params.max_modes, max_time_per_mode = gen_params.timesteps*env.test_dt_scale, max_timesteps = 50000)
		simulate_from_states_gif(env, states, True, args[2] + "_video.gif")
		fig = pl.figure(figsize = (6, 3))
		env.plot_init(states[0])
		env.plot_states(states)
		pl.tight_layout()
		pl.savefig(args[2] + "_plot.pdf")

	if args[0] == "TRAJ":
		name = args[1]
		simulate_from_file(env, name)

	if args[0] == "SM_EVAL":
		name = args[1]
		SM = StateMachinePolicy(env, [], [])
		SM.read(name)
		safe_fraction, avg_time_safe, avg_goal_error, avg_time_goal = evaluate_state_machine_test(envs, SM.modes, SM.conds, 10000, 10000, vis = False)
		print("Generalization: ", safe_fraction, avg_time_safe, avg_goal_error, avg_time_goal)
		#evaluate(env, SM, 100)


if __name__ == '__main__':
	name = sys.argv[1]
	synth_params, gen_params = get_bench_params(name, num_threads = 1)
	
	inp_limits = eval(sys.argv[2])
	

	run(synth_params, gen_params, inp_limits, sys.argv[3:])


