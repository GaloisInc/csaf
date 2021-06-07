import os, sys, time 
import numpy as np 
from general.utils import * 
import matplotlib.pyplot as pl
import time 


from pyOpt import Optimization
from pyOpt import SLSQP
from pyOpt import *


class PyOptSearch:

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
		init_cost, init_cost_arr = opt_fun.get_cost(x, vis = vis)
		print("Init cost: ", init_cost)

		start_time = time.time()


		def objfunc(x0):
			#if time.time() - start_time > 60:
			#	print('Terminating because of time constraint')
			#	raise ValueError
			fail = 0
			cost, cost_arr = opt_fun.get_cost(x0, vis = False)
			if (cost < self.min_cost):
				self.min_x = np.copy(x0)
				self.min_cost = cost
			#print("It%i"%(self.counter), cost, self.min_cost)
			#self.counter += 1
			return cost, cost_arr, fail

		
		opt_prob = Optimization('Opt', objfunc)
		for i in range(len(x)):
			opt_prob.addVar('x' + str(i), 'c',  value = x[i], lower = x_low[i], upper = x_high[i])
		opt_prob.addObj('f')
		opt_prob.addConGroup('cons', len(init_cost_arr), type = "i")
		#print(opt_prob)

		opt = SLSQP()
		opt.setOption('IPRINT', -1)
		opt.setOption('MAXIT', max_it)
		try:
			sol = opt(opt_prob)
			x = sol[1]
			cost = sol[0][0] 
			if cost  < self.min_cost:
				self.min_x = x
				self.min_cost = cost
		except (ValueError) as e:
			pass 
		
		#print(sol)
		#print("Final cost: ", self.min_cost)
		return self.min_x, self.min_cost, 0
		#return x, cost, it 

	




