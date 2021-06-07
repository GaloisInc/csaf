import numpy as np 

class Policy:
	def __init__(self, env):
		self.env = env
		self.cur_mode = 0

	def reset(self, init_state):
		self.cur_mode = 0 

	def get_action(self, state):
		x,y,vx,vy,t,w = self.env.get_features(state)
			
		if self.cur_mode == 0 and abs(t - 1.57) < 0.01 and abs(w - 0.0) < 0.01:
			print("Mode change from %i to %i"%(0, 1))
			self.cur_mode = 1 

		#if self.cur_mode == 1 and (abs(t - 1.57) > 0.1 or abs(w - 0.0) > 0.1) and abs(vx - 0.0) < 0.1 and abs(vy - 0.0) < 0.1:
		#	print("Mode change from %i to %i"%(1, 0))
		#	self.cur_mode = 0 

		if self.cur_mode == 1 and abs(x - 3.0) < 0.01 and abs(vx - 0.0) < 0.01:
			print("Mode change from %i to %i"%(1, 2))
			self.cur_mode = 2

		#if self.cur_mode == 2 and (abs(x - 3.0) > 0.1 or abs(vx - 0.0) > 0.1) and abs(vy - 0.0) < 0.1 :
		#	print("Mode change from %i to %i"%(2, 1))
		#	self.cur_mode = 1

		if self.cur_mode == 2 and abs(y - 2.0) < 0.01 and abs(vy - 0.0) < 0.01:
			print("Mode change from %i to %i"%(2, 3))
			self.cur_mode = 3

		



	
		if self.cur_mode == 0:
			a_r = 0.0
			a_t = 0.0
			alpha = (t - 1.57)*-1.0 + (w - 0)*-1.0
			return np.array([a_r, a_t, alpha])
		
		if self.cur_mode == 1:
			a_r = 0.0
			a_t = (x - 3.0)*-1.0 + (vx - 0)*-1.0
			alpha = 0.0
			return np.array([a_r, a_t, alpha])

		if self.cur_mode == 2:
			a_r = (y - 2.0)*-1.0 + (vy - 0)*-1.0
			a_t = 0.0
			alpha = 0.0
			return np.array([a_r, a_t, alpha])


		if self.cur_mode == 3:
			return []
		




