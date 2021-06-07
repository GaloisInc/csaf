import random
import numpy as np

# Random k mode policy for continuous actions.
class RandomKModePolicy:
    # Initialize the policy.
    def __init__(self, action_dim, action_bound, num_modes, max_time_per_mode):
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.num_modes = num_modes
        self.max_time_per_mode = max_time_per_mode

        self.actions = []
        self.times = []
        for i in range(num_modes):
        	self.actions.append(2*self.action_bound * np.random.random_sample(self.action_dim) - self.action_bound)
        	self.times.append(self.max_time_per_mode*np.random.random_sample())

        print("Actions: ", self.actions)
        print("Times: ", self.times)

        self.cur_id = 0
        self.cur_time = 0

    # Get a random action to take.
    #
    # state: np.array([state_dim])
    def get_action(self, state):
    	if (self.cur_time < self.times[self.cur_id]):
    		self.cur_time += 1.0
    	else:
    		self.cur_time = 0.0
    		if (self.cur_id + 1 < self.num_modes):
    			self.cur_id += 1
    	return self.actions[self.cur_id]

    def reset(self, init_state):
        return
    	


# Random k mode policy for discrete actions.
class RandomKModePolicyDiscrete:
    # Initialize the policy.
    def __init__(self, n_actions):
        self.n_actions = n_actions

    # Get a random action to take.
    #
    # state: np.array([state_dim])
    def get_action(self, state):
        return random.randint(0, self.n_actions-1)

    def reset(self):
        return
        
