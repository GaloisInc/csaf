import numpy as np 
import sys 

from environments.car.car import * 
from general.simulate import *



# reverse policy - tries to retrieve an already parked car
class ReversePPPolicy:
	def __init__(self, env):
		self.cur_mode = 0
		self.env = env 

	def reset(self, state):
		self.cur_mode = 0

	def get_action(self, state):
		x,y,ang,d = state
		x,y,ang,d1,d2 = self.env.get_features(state) 
		
		#print("v4 x: ", x, v4_x)
		
		print(d1, d2)
		if self.cur_mode == 0 and (d1 < 0.2):
			#print("Change from %i to %i"%(0, 1))
			self.cur_mode = 1

		if self.cur_mode == 1 and (d2 < 0.2  ):
			#print("Change from %i to %i"%(1, 0))
			self.cur_mode = 0

		if self.cur_mode == 0 and (x < -1.2):
			#print("dafhapei")
			#print("Change from %i to %i"%(0, 2))
			#print(state, v1_x)
			self.cur_mode = 2


		if self.cur_mode == 2 and (ang < 1.58):
			#print("Change from %i to %i"%(2, 3))
			self.cur_mode = 3


		if self.cur_mode == 0:
			return np.array([5, 5])

		if self.cur_mode == 1:
			return np.array([-5, -5])


		if self.cur_mode == 2:
			return np.array([5, -5])

		if self.cur_mode == 3:
			return []


if __name__ == "__main__":
	env = CarReversePP(2000)
	env.set_inp_limits((12, 13))
	init_state = env.sample_init_state()
	policy = ReversePPPolicy(env)
	plot_traj(env, policy, init_state)





