import numpy as np 
from utils import * 


class Policy:
	def __init__(self, env, params):
		self.cur_mode = 0
		self.prev_mode = self.cur_mode
		self.timesteps = 0
		self.params = params
		self.env = env 

	def reset(self, init_state):
		self.cur_mode = 0
		self.prev_mode = self.cur_mode
		self.timesteps = 0

	def act(self, state, idx):
		params = self.params
		angle = state[2]
		thigh_ang = state[3]
		leg_ang = state[4]
		foot_ang = state[5]

		a = foot_ang*params[idx] + leg_ang*params[idx+1] + thigh_ang*params[idx+2] + angle*params[idx+3] + params[idx+4]

		return a


	def get_action(self, state):
		params = self.params 

		x = state[0]
		y = state[1]
		vy = state[7]
		angle = state[2]
		thigh_ang = state[3]
		leg_ang = state[4]
		foot_ang = state[5]

		v = state[6]
		leg_ang_vel = state[10]
		foot_ang_vel = state[11]
		thigh_ang_vel = state[9]
		#print(angle, foot_ang, leg_ang, thigh_ang)


		'''if y > 1.2:
			self.cur_mode = 0 
			acts = [self.params[2], self.params[1], self.params[0]]

			#act_foot =   (foot_ang + self.params[0])*-1.0
			#act_leg =  (leg_ang + self.params[1])*-1.0 + (leg_ang_vel)*-1.0
			#act_thigh = (thigh_ang + self.params[2])*-1.0  + (thigh_ang_vel)*-1.0
		else:
			self.cur_mode = 1
			acts = [self.params[5], self.params[4], self.params[3]]
			#act_foot =   (foot_ang + self.params[3])*-1.0
			#act_leg =  (leg_ang + self.params[4])*-1.0 + (leg_ang_vel)*-1.0
			#act_thigh = (thigh_ang + self.params[5])*-1.0  + (thigh_ang_vel)*-1.0'''

		#if self.cur_mode != self.prev_mode:
			#print("Mode change from %i to %i"%(self.prev_mode, self.cur_mode))
			#print("Time steps: %i"%self.timesteps)
			#self.timesteps = 0
			#print("x: %f"%x)
		
		#self.timesteps += 1 
		#self.prev_mode = self.cur_mode 
		#return acts 

		if vy > 0.0:
			act_foot =   (foot_ang + leg_ang + thigh_ang - angle  + self.params[0])*-1.0 
			act_leg =  (-thigh_ang + angle + self.params[1])*-1.0 + (leg_ang_vel)*-1.0
			act_thigh = (angle + self.params[2]) * -1.0  + (thigh_ang_vel)*-1.0
		else:
			act_foot =  (foot_ang + leg_ang + thigh_ang - angle  + self.params[3])*-1.0 
			act_leg =  (-thigh_ang + angle + self.params[4])*-1.0 + (leg_ang_vel)*-1.0
			act_thigh = (angle + self.params[5]) * -1.0  + (thigh_ang_vel)*-1.0
		
		'''if y > 1.2:
			act_foot =   (foot_ang + leg_ang + thigh_ang - angle  - 1.0)*-1.0 #+ (foot_ang_vel)*-1.0
			act_leg =  (-thigh_ang + angle - 0.1)*-1.0 + (leg_ang_vel)*-1.0
			act_thigh = (angle - 0.1) * -1.0  + (thigh_ang_vel)*-1.0
		else:
			act_foot =  (foot_ang + leg_ang + thigh_ang - angle  + 1.0)*-1.0 #+ (foot_ang_vel)*-1.0
		
			act_leg =  (-thigh_ang + angle - 0.2)*-1.0 + (leg_ang_vel)*-1.0
			act_thigh = (angle - 0.2) * -1.0  + (thigh_ang_vel)*-1.0'''

		'''act_foot = (foot_ang + leg_ang + thigh_ang - angle  - 1.0)*-1.0 + (foot_ang_vel)*-1.0
		act_leg  = (-thigh_ang + angle + 0.2)*-1.0 + (leg_ang_vel)*-1.0
		act_thigh = (angle + 0.2) * -1.0  + (thigh_ang_vel)*-1.0'''

		act_foot = max(min(act_foot, 1.0), -1.0)
		act_thigh = max(min(act_thigh, 1.0), -1.0)
		act_leg = max(min(act_leg, 1.0), -1.0)
		return [act_thigh, act_leg, act_foot] 

	def evaluate(self, init_state, vis = False):
		env = self.env 

		env.reset()
		state = init_state
		done = False
		self.reset(init_state)
		total_goal_error = 0.0
		total_safe_error = 0.0
		
		for i in range(2000):
			action = self.get_action(state)

			if len(action) == 0:
				done = True
				break
	 
			next_state, safe_error, goal_error, done = env.step(state, action)
			total_safe_error += safe_error
			if total_safe_error > 0.05:
				done = True
		   
			state = next_state
			if done:
				break
		
		total_goal_error = env.check_goal(state)
		#print(total_safe_error, total_goal_error)

		return total_safe_error, total_goal_error, 0.0, 0.0

class PolicyGrammar:
	def __init__(self, env):
		self.env = env 
		self.num_vars = 6
		self.x_low = []
		self.x_high = []
		for i in range(self.num_vars):
			self.x_low.append(-5)
			self.x_high.append(5)
		self.x_low = np.array(self.x_low)
		self.x_high = np.array(self.x_high)

	def get_random_policy_vars(self):
		states = []
		for i in range(len(self.x_low)):
			states.append(rand(self.x_low[i], self.x_high[i]))
		
		return np.array(states)

	def bound(self, x):
		return np.clip(x, self.x_low, self.x_high)

	def get_bounds(self):
		return self.x_low, self.x_high

	def get_policy(self, x):
		policy = Policy(self.env, x)
		policies = [] 
		for i in range(5):
			policies.append(policy)
		return policies





