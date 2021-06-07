import numpy as np 
import matplotlib.pyplot as pl 

def simulate_point_car(state, action, dt):
		
	ns = np.copy(state)
	v, w = action
	w = w/25.0
	if (v > 5.0):
		v = 5
	if (v < -5.0):
		v = -5
	if (w > 0.2):
		w = 0.2
	if (w < -0.2):
		w = -0.2

	# Step 1: Update car x and y
	d = v*dt
	ns[0] += d*np.cos(ns[2])
	ns[1] += d*np.sin(ns[2])

	# Step 2: Update car ang
	ns[2] += v*w* dt

	return ns 

def simulate_bicycle(state, action, dt):
	ns = np.copy(state)
	v, w = action 
	w = w/10.0
	

	x,y,ang = ns  
	beta = np.arctan(0.5*np.tan(w))
	dx = v*np.cos(ang + beta)*dt 
	dy = v*np.sin(ang + beta)*dt 
	da = v/2.5*np.sin(beta)*dt 

	ns[0] += dx 
	ns[1] += dy 
	ns[2] += da 

	return ns 



def get_all_vertices(x, y, ang, w, h):
	res = []
	db = w/2
	da = h/2
	coa = np.cos(ang)
	sia = np.sin(ang)
	res.append((x + da*coa + db*sia, y + da*sia - db*coa))
	res.append((x + da*coa - db*sia, y + da*sia + db*coa))
	res.append((x - da*coa - db*sia, y - da*sia + db*coa))
	res.append((x - da*coa + db*sia, y - da*sia - db*coa))
	return res


def get_traj(s, a, T):
	w = 1.8
	h = 5.0 

	

	X = []
	Y = []
	X1 = []
	Y1 = []  

	
	v = get_all_vertices(s[0], s[1], s[2], w, h)
	
	X.append(s[0])
	Y.append(s[1])
	X1.append((v[2][0] + v[3][0])/2.0)
	Y1.append((v[2][1] + v[3][1])/2.0)

	for i in range(T):
		s = simulate_bicycle(s, a, 0.01)
		v = get_all_vertices(s[0], s[1], s[2], w, h)
		X.append(s[0])
		Y.append(s[1])
		X1.append((v[2][0] + v[3][0])/2.0)
		Y1.append((v[2][1] + v[3][1])/2.0)

	pl.plot(X, Y, "b")
	pl.plot(X1, Y1, "r")
	pl.gca().set_aspect('equal', adjustable='box')

	return s 

	
s = np.array([0.0, 0.0, np.pi/2.0])
s = get_traj(s, (5, 5), 20)
s = get_traj(s, (-5, -5), 20)
s = get_traj(s, (5, 5), 20)
s = get_traj(s, (-5, -5), 20)
s = get_traj(s, (5, 5), 20)

pl.show()
pl.close()
