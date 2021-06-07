import random
import numpy as np


# Random policy for continuous actions.
class RandomPolicy:
    # Initialize the policy.
    def __init__(self, action_dim, action_bound):
        self.action_dim = action_dim
        self.action_bound = action_bound

    # Get a random action to take.
    #
    # state: np.array([state_dim])
    def get_action(self, state):
    	return 2*self.action_bound * np.random.random_sample(self.action_dim) - self.action_bound

    def reset(self, init_state):
        return


# Random policy for discrete actions.
class RandomPolicyDiscrete:
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
        
