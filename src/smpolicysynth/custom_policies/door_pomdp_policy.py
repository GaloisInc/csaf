import numpy as np 

from environments.pomdp.door.door import * 
from general.simulate import * 

class Policy:
	def __init__(self, env):
		self.env = env
		self.cur_mode = 0
		self.num_changes = 0
		self.prev_mode = -1
		self.timesteps = 0
		self.counter = -1 

	def reset(self, init_state):
		self.cur_mode = 0
		self.num_changes = 0
		self.prev_mode = -1
		self.timesteps = 0
		self.counter = -1

	def get_action(self, state):
		x,goal_door = state
		#print(goal_door)


		door, right_wall, left_wall = self.env.get_features(state)
		print(x, door, self.counter)


		if self.cur_mode == 0 and left_wall > 0.0:
			self.cur_mode = 1
			self.counter += 0 

		if self.cur_mode == 2 and abs(self.counter - goal_door) < 0.1:
			self.cur_mode = 3 
			self.counter += 0 

		if self.cur_mode == 1 and door > 0.0:
			self.counter += 1
			self.cur_mode = 2

		if self.cur_mode == 2 and door < 0.0:
			self.cur_mode = 1
			self.counter += 0
			


	
		if self.cur_mode == 0:
			return np.array([-1.0])
		
		if self.cur_mode == 1:
			return np.array([1.0])

		if self.cur_mode == 2:
			return np.array([1.0])



		if self.cur_mode == 3:
			return []
if __name__ == "__main__":
	env = Door(2000)
	init_state = env.sample_init_state()
	policy = Policy(env)
	plot_traj(env, policy, init_state)		




