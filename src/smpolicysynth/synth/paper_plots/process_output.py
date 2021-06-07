import sys 
import numpy as np 
import matplotlib.pyplot as pl 


def process_file(filename):
	file = open(filename, 'r')
	total_costs = [] 

	total_time = None
	traj_opt_time = None 
	mode_learn_time = None 
	cond_learn_time = None 
	sm_eval_time = None 
	traj_sampler_time = None 

	for l in file.readlines():
		l = l.strip()
		if l.startswith("Total cost:"):
			cost = float(l.split(":")[1])
			total_costs.append(cost)

		if l.startswith("Overall time:"):
			total_time = float(l.split(":")[1])

		if l.startswith("Traj opt time:"):
			traj_opt_time = float(l.split(":")[1])

		if l.startswith("Learn modes time:"):
			mode_learn_time = float(l.split(":")[1])

		if l.startswith("Learn conds time:"):
			cond_learn_time = float(l.split(":")[1])

		if l.startswith("SM eval time:"):
			sm_eval_time = float(l.split(":")[1])

		if l.startswith("Ref traj sampler time:"):
			traj_sampler_time = float(l.split(":")[1])


	iterations = np.argmin(total_costs) + 1
	frac = float(iterations)/float(len(total_costs))
	print("Iterations: ", iterations)
	if total_time != None:
		teacher_time = (traj_opt_time + traj_sampler_time) * frac
		student_time = (mode_learn_time + cond_learn_time) * frac 
		misc_time = (sm_eval_time)*frac
		total_time = total_time * frac
		print("Teacher time: ", teacher_time)
		print("Student time: ", student_time)
		print("Misc time: ", misc_time)
		print("Total_time: ", total_time)

	if False:
		costs = total_costs[:iterations]
		x = np.arange(0, iterations, 1)
		pl.plot(x, costs)
		pl.show()

	file.close()

	return iterations, total_time, teacher_time, student_time, misc_time

#X = ['Car','QuadPO', 'Pendulum', 'Cartpole', 'MCar', "Acrobot"]
X = ["Quad", "Swimmer"]
files = {'Car': '/scratch/jinala/state_machines/general/python/synth/out_iclr/rpp0.txt', 
		'Pendulum' : '/scratch/jinala/state_machines/general/python/synth/out_iclr/pen0.txt',
		'Quad' : '/scratch/jinala/state_machines/general/python/synth/out_iclr/quadr3.txt',
		'QuadPO' : '/scratch/jinala/state_machines/general/python/synth/out_iclr/quad5.txt',
		'Cartpole' : 'out_iclr/cp0.txt',
		"Swimmer" : "out/swimmer4_2.txt",
		'MCar' : '/scratch/jinala/state_machines/general/python/synth/out_iclr/mc0.txt', 
		'Acrobot' : '/scratch/jinala/state_machines/general/python/synth/out_iclr/dpen1.txt'
		}


colors = {
	't': (153,216,201),
	's': (227,26,28),
	'm': (152,78,163),
	'Direct-opt': (251,180,174)
}

for key in colors:
	r, g, b = colors[key]
	colors[key] = (r/255., g/255., b/255.)

def process_all():
	teacher_times = []
	student_times = []
	misc_times = [] 
	iterations = []

	pl.figure(figsize=(3,4))

	for x in X:
		file = files[x]
		it, total_time, teacher_time, student_time, misc_time = process_file(file)
		iterations.append(it)
		teacher_times.append(teacher_time)
		student_times.append(student_time)
		misc_times.append(misc_time)

	p1 = pl.bar(X, teacher_times, 0.5, color=colors['t'], hatch='///')
	p2 = pl.bar(X, student_times, 0.5, color=colors['s'], bottom=teacher_times, hatch='|||')
	p3 = pl.bar(X, misc_times, 0.5, color=colors['m'], bottom=[i + j for i,j in zip(teacher_times, student_times)], hatch="\\\\\\")

	ct = 0
	for ct in range(len(iterations)):
		rect = p3[ct]
		height = p1[ct].get_height() + p2[ct].get_height() + p3[ct].get_height()
		pl.text(rect.get_x() + rect.get_width()/2.0, height, 'iterations: %d' % int(iterations[ct]), ha='center', va='bottom')


	#pl.legend((p1[0], p2[0], p3[0]), ("Teacher", "Student", "Misc"))
	#pl.ylabel("Time (s)", fontsize =20)
	#pl.xlabel("Benchmarks", fontsize=20)
	pl.xlim((-1, 2))
	pl.tight_layout()
	#pl.show()
	pl.savefig("out_iclr/runtime1.pdf")



if len(sys.argv) <= 1:
	process_all()
else:
	filenames = sys.argv[1:]
	for file in filenames:
		process_file(file)