import numpy as np
import sys 
import matplotlib.pyplot as pl

# Linear policy for continuous actions.
class LinearPolicy:
	# Initialize the policy.
	def __init__(self, env, actions, timesteps):
		self.env = env
		self.actions = actions
		self.timesteps = timesteps



	# Get a  action to take.
	#
	# state: np.array([state_dim])
	def get_action(self, state):
		act_features = self.env.get_act_features(state)
		acts = []
		act_weights = self.actions
		for i in range(len(act_weights)):
			aw = act_weights[i]
			assert(len(aw) == len(act_features) + 1)
			act = 0.0
			for j in range(len(act_features)):
				act += aw[j] * act_features[j]
			act += aw[-1]
			acts.append(act)
		return np.array(acts)
		
	def reset(self, init_state):
		return 		

	def get_traj_from_linear_policy(self, env, init_state, timesteps, vis = False):
		states = []

		env.reset()
		state = init_state
		done = False
		self.reset(init_state)
		total_goal_error = 0.0
		total_safe_error = 0.0

		time_until_unsafe = 0.0
		total_time = 0.0
		unsafe = False
		for t in range(timesteps):
			states.append(state)        
			action = self.get_action(state)

			if len(action) == 0:
				done = True
				break
	 
			next_state, safe_error, goal_error, done = env.step(state, action, -1)
			if not unsafe:
				if safe_error < 0.01:
					time_until_unsafe += 1.0;
				else:
					unsafe = True
					total_safe_error += safe_error
			total_time += 1.0;

			#total_safe_error += safe_error
		   
			state = next_state
			if done:
				break

		
		states.append(state)
		total_goal_error = env.check_goal(state)
		total_safe_error += total_time - time_until_unsafe
		return states, total_safe_error, total_goal_error, 

	def evaluate(self, init_state,  vis = False, timesteps = -1):
		if timesteps < 0:
			timesteps = self.timesteps
		states, safe_err, goal_err = self.get_traj_from_linear_policy(self.env, init_state, timesteps, vis)

		if (vis):
			self.env.plot_init()
			X_traj, Y_traj = self.env.get_2d_states(states)
			pl.plot(X_traj, Y_traj)

		return safe_err, goal_err, []

