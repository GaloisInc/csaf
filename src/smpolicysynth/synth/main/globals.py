fig = None
plt_idx = 1
ncol = 5
nrow = 4

def fig_add_subplot():
	global fig, ncol, nrow, plt_idx
	fig.add_subplot(nrow, ncol, plt_idx)
	plt_idx += 1

def get_cur_row():
	return (plt_idx-1)//ncol*ncol 

def fig_add_subplot1(idx):
	global fig, ncol, nrow, plt_idx
	fig.add_subplot(nrow, ncol, idx)
	plt_idx += 1

def fig_new_row():
	global fig, ncol, nrow, plt_idx
	while plt_idx % ncol != 1:
		plt_idx += 1
