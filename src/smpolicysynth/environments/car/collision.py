import numpy as np
#import torch

# b_ang = pi/2
def check_collision_box(a_x, a_y, a_ang, b_x, b_y, s, w, h):
	error = 0

	# check the circle around center of a with r = w/2 does not collide with b
	error += check_collision_point(a_x, a_y, w/2.0 + 0.2, b_x - w/2.0, b_x + w/2.0, b_y - h/2.0, b_y + h/2.0, s)

	vertices = get_all_vertices(a_x, a_y, a_ang, w, h)

	# check that all vertices of a are outside b
	for v in vertices:
		error += check_collision_point(v[0], v[1], 0.2, b_x - w/2.0, b_x + w/2.0, b_y - h/2.0, b_y + h/2.0, s)

	top =(vertices[0][0] + vertices[1][0])/2.0, (vertices[0][1] + vertices[1][1])/2.0
	bottom = (vertices[2][0] + vertices[3][0])/2.0, (vertices[2][1] + vertices[3][1])/2.0

	circle1 = a_x*(h/2.0 - w/2.0)/(h/2.0) + top[0]*(w/2.0)/(h/2.0), a_y*(h/2.0 - w/2.0)/(h/2.0) + top[1]*(w/2.0)/(h/2.0)

	# check the circle from top middle 
	error += check_collision_point(circle1[0], circle1[1], w/2.0 + 0.2, b_x - w/2.0, b_x + w/2.0, b_y - h/2.0, b_y + h/2.0, s)


	circle2 = a_x*(h/2.0 - w/2.0)/(h/2.0) + bottom[0]*(w/2.0)/(h/2.0), a_y*(h/2.0 - w/2.0)/(h/2.0) + bottom[1]*(w/2.0)/(h/2.0)

	# check the circle from bottom middle
	error += check_collision_point(circle2[0], circle2[1], w/2.0 + 0.2, b_x - w/2.0, b_x + w/2.0, b_y - h/2.0, b_y + h/2.0, s)

	assert(error >= 0.0)

	return error
	


def get_all_vertices(x, y, ang, w, h):
	res = []
	db = w/2.0
	da = h/2.0
	coa = np.cos(ang)
	sia = np.sin(ang)
	#coa = torch.cos(ang + torch.tensor(0))
	#sia = torch.sin(ang + torch.tensor(0))
	res.append((x + da*coa + db*sia, y + da*sia - db*coa))
	res.append((x + da*coa - db*sia, y + da*sia + db*coa))
	res.append((x - da*coa - db*sia, y - da*sia + db*coa))
	res.append((x - da*coa + db*sia, y - da*sia - db*coa))
	return res


# check for intersections between a circle and a rect
def check_collision_point(a_x, a_y, a_r, b_xl, b_xu, b_yl, b_yu, s ):
	d1 = b_xl - a_x - a_r
	d2 = a_x - b_xu - a_r
	d3 = b_yl - a_y - a_r
	d4 = a_y - b_yu - a_r
	if s == 'l':
		d_max  = np.max([d1, d4])
	else:
		d_max = np.max([d1, d3])
	if (d_max < 0):
		return -d_max
	else:
		return 0

	


