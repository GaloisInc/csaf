import numpy as np 

from environments.quadcopter.quad import * 
from general.simulate import *


class QuadPolicy:
	def __init__(self, env):
		self.cur_mode = 0
		self.env = env 
		self.prev_mode = -1
		self.num_mode_changes = 0
		self.num_timesteps = 0

	def reset(self, state):
		self.cur_mode = 0
		self.prev_mode = -1
		self.num_mode_changes = 0
		self.num_timesteps = 0

	def get_action(self, state):
		x,d_f,d_r ,x_r, x_f= self.env.get_features(state) 
		#vy = vy/5.0
		x = x*5.0
		d_f = d_f/5.0
		d_r = d_r/5.0
		#print( d_f, d_r, x_r, x_f, vy)


		if self.cur_mode == 0 and d_r < 1.2:
			self.cur_mode = 1 

		if self.cur_mode == 1 and d_f < 1.2:
			self.cur_mode = 0

		if x > 35:
			self.cur_mode = 4

		if self.cur_mode != self.prev_mode:
			print("Mode change from %i to %i"%(self.prev_mode, self.cur_mode))
			self.num_mode_changes += 1
			print("Timesteps: ", self.num_timesteps)
			self.num_timesteps = 0

		self.num_timesteps += 1

		self.prev_mode = self.cur_mode
		
		
		#print(d1, d2)

		if self.cur_mode == 0:
			return np.array([5.0, -5.0, 0.0])

		if self.cur_mode == 1:
			return np.array([-5.0, -5.0, 0.0])

		

		if self.cur_mode == 4:
			print("Num mode changes", self.num_mode_changes)
			return []

if __name__ == "__main__":
	env = Quadcopter(2000)
	init_state = env.sample_init_state()
	policy = QuadPolicy(env)
	plot_traj(env, policy, init_state)
		



