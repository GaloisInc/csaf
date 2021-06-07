
class System:
	def __init__(self):
		pass

	def simulate(self, state, action, dt = -1):
		pass

	def check_safe(self, state):
		pass

	def check_goal(self, state):
		pass

	def sample_init_state(self):
		pass

	def set_act_for_render(self, action):
		pass

	def render(self, state, mode):
		pass

	def reset(self):
		pass

	def done(self):
		pass

	def step(self, state, action, dt = -1):
		new_state = self.simulate(state, action, dt)
		safe_error = self.check_safe(new_state)
		goal_error = self.check_goal(new_state)
		done = self.done(new_state)
		return new_state, safe_error, goal_error, done
