import os, sys, time 
import numpy as np 
from general.utils import * 
import matplotlib.pyplot as pl

from scipy.optimize import minimize


class ScipySearch:

	def __init__(self, opt_fun):
		self.opt_fun = opt_fun
		self.min_cost = 1e30
		self.min_x = None


	def minimize(self, max_it, random_init = True, vis = False):
		opt_fun = self.opt_fun
		self.min_cost = 1e30
		self.min_x = None
		self.counter = 0

		if random_init:
			x = opt_fun.get_random_x()
		else:
			x = opt_fun.get_current_x()

		#print(x)
		#opt_fun.get_cost(x, True)
		
		x_low, x_high = opt_fun.get_bounds()
		bounds = list(zip(x_low, x_high))
		init_cost, _ = opt_fun.get_cost(x, vis = vis)
		print("Init cost: ", init_cost)

		self.t_same_cost = 0

		def objfunc(x0):
			cost, _ = opt_fun.get_cost(x0, vis = False)
			fail = 0
			old_cost = self.min_cost 
			if (cost < self.min_cost):
				self.min_x = np.copy(x0)
				self.min_cost = cost
			#print("It%i"%(self.counter), cost, self.min_cost)
			#self.counter += 1

			if abs(old_cost - self.min_cost) < 1e-3:
				self.t_same_cost += 1
			else:
				self.t_same_cost = 0 

			#if self.t_same_cost > 5:
			#	raise ValueError
			return cost #, grad

		try: 
			res = minimize(objfunc, x, bounds = bounds, options={'maxiter': max_it })
			x = res.x
			cost, _ = opt_fun.get_cost(x, vis = False)
			if cost  < self.min_cost:
				self.min_x = x
				self.min_cost = cost
		except ValueError:
			pass
		
		#print(sol)
		print("Final cost: ", self.min_cost)
		return self.min_x, self.min_cost, None
		#return x, cost, it 

	




