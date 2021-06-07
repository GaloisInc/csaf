import numpy as np 


def rand(a, b):
	if b < a:
		b = a
	return (b-a)*np.random.random() + a
	
def relu(x):
	if x < 0.0:
		return 0.0
	else:
		return x