import sys
from synth.policy.state_machine import * 

from environments.car.car import *


import numpy as np 

def main(args):
	n_env_steps = 50000
	env = CarReversePP(n_env_steps)
	env.tol = 0.001
	env.set_inp_limits((11.8, 11.8))
	init_state = env.sample_init_state()

	policy = StateMachinePolicy(env, [], [])
	policy.read(args[0])

	old_mode_steering_action = policy.modes[0][1][0]
	policy.modes[0][1][0] = 5.0
	policy.modes[2][1][0] = -5.0
	plot(env, init_state, policy,  "steering")

	cond1 = policy.conds[0][2]
	cond1.params[-1] = 0.1

	cond2 = policy.conds[2][0]
	cond2.params[-1] = 0.1
	plot(env, init_state, policy, "gap")
	
def plot(env, init_state, policy, name):
	fig = pl.figure(figsize = (1.5, 3))
	

	states, safe_err, goal_err, *_ = policy.get_traj_from_sm(env, init_state, max_modes = 1000, max_time_per_mode = 100*env.test_dt_scale, max_timesteps = 50000, dt = -1)
	
	print("safe_err: ", safe_err)
	print("goal_err: ", goal_err)
	env.plot_init_paper(states[0], states[-1])
	env.plot_states(states)
	pl.tight_layout()
	
	pl.savefig("/Users/jinala/papers/policy_synth/figures/rpp_modify_%s.pdf"%name)
	pl.show()
	pl.close()



if __name__ == '__main__':
    main(sys.argv[1:])
    