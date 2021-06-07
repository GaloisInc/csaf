import sys

from synth.policy.state_machine import * 

from environments.pendulum.pendulum import * 
from environments.mountain_car.mountain_car import *
from environments.acrobot.acrobot import *
from environments.car.car import * 
from environments.cartpole.cartpole import * 

from environments.quadcopter.quad import * 
from environments.quadcopter.quad_po import * 
from environments.swimmer.swimmer4.swimmer4 import * 

import numpy as np 
import matplotlib 
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
matplotlib.rcParams.update({'font.size': 20})

def main(args):
	n_env_steps = 50000
	env = Swimmer4(n_env_steps)

	fig = pl.figure(figsize = (6, 3))
	plot_nn(env, args[0])
	plot_sm(env, args[1])

	
	pl.legend(loc='upper right', fontsize=15)
	
	
	pl.xlabel("time",size= 20)
	pl.ylabel("torque3", size = 18)
	pl.tight_layout()
	pl.savefig("/Users/jinala/papers/policy_synth/figures/swimmer_action3.pdf")
	pl.show()
	pl.close()

	

def plot_nn(env, name):

	X = []
	time = 0.0
	acts = []
	nn_file = open(name, 'r')
	for l in nn_file.readlines():
		if l[0] == "S" and l[1] == ":":
			l = l.strip().split(":")
			act = np.array(eval(l[2]))
			acts.append(act[2])
			X.append(time)
			time += 0.1

			if len(X) > 200:
				break

	pl.plot(X, acts, c ='b', label="RL")
	

	


def plot_sm(env, name):
	policy = StateMachinePolicy(env, [], [])
	policy.read(name)
	init_state = env.sample_init_state()
	print(init_state)

	states, safe_err, goal_err, *_ = policy.get_traj_from_sm(env, init_state, max_modes = 1000, max_time_per_mode = 100*env.test_dt_scale, max_timesteps = 50000, dt = -1)
	print(goal_err)
	print(len(states))

	X = []
	time = 0
	acts = []
	for i in range(len(states)):
		state = states[i][0]
		action = states[i][1]
		if len(action) == 0: continue

		action = env.get_actual_action(action, state)

		#x,y,vx,vy,t,w = state[0:6]
		#a = (vy - action[0]/5.0*2.0)*action[1]/5.0*2.0 
		a = action[2]
		acts.append(a) 
		X.append(time)
		time += 0.1

		if len(X) > 200:
			break

	pl.plot(X, acts, c ='r', linewidth=2, label="Ours")






if __name__ == '__main__':
    main(sys.argv[1:])
    