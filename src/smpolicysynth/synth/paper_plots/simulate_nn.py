import sys
sys.path.append("../../general/python")
#from run import *
from hopper import *
sys.path.append("../../general/python/synth")

import numpy as np 

def main():
	n_env_steps = 1000

	env = Hopper(n_env_steps)
	simulate_from_file(env, "nn_hopper.txt")



def simulate_from_file(env, filename):
	file = open(filename)
	states = []
	for l in file.readlines():
		if l[0] == "S" and l[1] == ":":
			l = l.strip().split(":")
			state = np.array(eval(l[1]))
			act = np.array(eval(l[2]))
			states.append(state)
	print(len(states))

	simulate_from_states(env, states, True)

def simulate_from_states(env, states, render):
	# Step 1: Initialization
	env.reset()
	
	for state in states:
		# Step 2a: Render environment
		if render:
			env.render(state)
			time.sleep(0.01)

		safe_error = env.check_safe(state)
		goal_error = env.check_goal(state)
		
		print("Safe error: ", safe_error, " Goal error: ", goal_error)
		

	time.sleep(2)
	
if __name__ == '__main__':
    main()
    
	