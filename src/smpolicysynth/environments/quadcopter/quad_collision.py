import numpy as np

def check_collision_with_ground(x, y, t, l):
	error = 0.0
	vertices = get_all_vertices(x, y, t, l)
	for v in vertices:
		if y < 0.0:
			error += -y

		if y > 12.0:
			error += y - 12.0

	return error

def check_collision_with_lower_obj(x, y, t, l, t_xl, t_xu, t_yl):
	d1 = t_xl - x
	d2 = x - t_xu
	d3 = y - t_yl
	d_max = np.max([d1, d2, d3])
	if (d_max < 0.0):
		return -d_max
	else:
		return 0.0

def check_collision_with_upper_obj(x, y, t, l, t_xl, t_xu, t_yu):
	d1 = t_xl - x
	d2 = x - t_xu
	d3 = t_yu - y
	d_max = np.max([d1, d2, d3])
	if (d_max < 0.0):
		return -d_max
	else:
		return 0.0

def get_all_vertices(x, y, t, l):
	res = []
	d = l/2.0
	coa = np.cos(t)
	sia = np.sin(t)

	res.append((x + d*coa, y + d*sia))
	res.append((x, y))
	res.append((x - d*coa, y - d*sia))
	return res

