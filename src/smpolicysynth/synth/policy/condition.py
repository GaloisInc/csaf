import numpy as np 
class Condition:
	def __init__(self):
		pass

	def eval(self, state):
		pass

class FalseCond(Condition):
	def eval(self, state):
		return -10

	def __str__(self):
		return "FalseCond()"

class TrueCond(Condition):
	def eval(self, state):
		return 10

	def __str__(self):
		return "TrueCond()"

class LinearCond(Condition):
	def __init__(self, params):
		self.params = params

	def __str__(self):
		return "LinearCond(" + str(self.params.tolist()) + ")"


	def eval(self, state):
		res = 0
		norm = 0
		assert(len(state) == len(self.params) - 1)
		for i in range(len(state)):
			res += state[i]*self.params[i]
			norm += self.params[i]*self.params[i]
		res += self.params[-1]
		if norm > 0.0:
			res = res/np.sqrt(norm)
				
		return res

class AndCond(Condition):
	def __init__(self, mother, father):
		self.mother = mother
		self.father = father

	def __str__(self):
		return "AndCond(" + str(self.mother) + "," + str(self.father) + ")"

	def eval(self, state):
		v1 = self.mother.eval(state)
		v2 = self.father.eval(state)
		return min(v1, v2)
		

class OrCond(Condition):
	def __init__(self, mother, father):
		self.mother = mother
		self.father = father

	def __str__(self):
		return "OrCond(" + str(self.mother) + "," + str(self.father) + ")"

	def eval(self, state):
		v1 = self.mother.eval(state)
		v2 = self.father.eval(state)
		return max(v1, v2)

class SoftBoolCond2(Condition):
	def __init__(self, mother, father, weights):
		self.mother = mother
		self.father = father
		self.weights = weights

	def __str__(self):
		return "SoftBoolCond2(" + str(self.mother) + "," + str(self.father) + "," + str(self.weights)+ ")"

	def eval(self, state):
		v1 = self.mother.eval(state)
		v2 = self.father.eval(state)

		v = 0.0

		# add v1
		v += v1*self.weights[0]

		# add v2 
		v += v2*self.weights[1]

		# add min(v1, v2)
		v += min(v1, v2)*self.weights[2]

		# add max(v1, v2)
		v += max(v1, v2)*self.weights[3]
		
		return v
