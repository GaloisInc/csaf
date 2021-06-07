import sys
import numpy as np 

from synth.policy.state_machine import *
from environments.car.car import *
from environments.quadcopter.quad import * 

def main(args):
	n_env_steps = 50000
	env = CarReversePP(n_env_steps)
	env.tol = 0.001

	policy = StateMachinePolicy(env, [], [])
	policy.read(args[0])

	plot(env, policy, 15.0, args)
	plot(env, policy, 13.0, args)
	plot(env, policy, 12.0, args)
	plot(env, policy, 11.2, args)

def plot(env, policy, dist, args):
	fig = pl.figure(figsize = (1.5, 3))
	
	env.set_inp_limits((dist, dist))
	init_state = env.sample_init_state()

	states, safe_err, goal_err, *_ = policy.get_traj_from_sm(env, init_state, max_modes = 1000, max_time_per_mode = 100*env.test_dt_scale, max_timesteps = 50000, dt = -1)
	
	print("safe_err: ", safe_err)
	print("goal_err: ", goal_err)
	env.plot_init_paper(states[0], states[-1])
	env.plot_states(states)
	pl.tight_layout()
	d_str = str(dist)
	d_str = d_str.replace(".", "_")
	if len(args) > 1:
		pl.savefig(args[1] + "/reverse_pp_%s.pdf"%d_str)
	pl.show()
	pl.close()



if __name__ == '__main__':
	# args[1] - sm policy file name 
	# args[2] - plots save location
    main(sys.argv[1:])
    