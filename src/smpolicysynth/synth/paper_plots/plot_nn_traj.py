import sys
from environments.car.car import *


import numpy as np 

def main(args):
	n_env_steps = 50000
	env = CarReversePP(n_env_steps)
	env.tol = 0.001

	collision_states = []

	nn_traj = []
	nn_file = open(args[0], 'r')
	for l in nn_file.readlines():
		if l[0] == "S" and l[1] == ":":
			l = l.strip().split(":")
			state = np.array(eval(l[1]))
			act = np.array(eval(l[2]))
			nn_traj.append((state, act))
			if env.check_safe(state) > 0.02:
				collision_states.append(state)

	fig = pl.figure(figsize = (1.5, 3))
	env.plot_init_paper(nn_traj[0], nn_traj[-1])
	env.plot_states(nn_traj, line = True)
	env.plot_collision_states(collision_states)
	pl.legend(loc='best', fontsize=7)

	pl.tight_layout()
	d_str = str(args[1])
	d_str = d_str.replace(".", "_")
	pl.savefig("/Users/jinala/papers/policy_synth/figures/nn_rpp_%s.pdf"%d_str)
	pl.show()
	pl.close()




if __name__ == '__main__':
    main(sys.argv[1:])
    