import sys 
import numpy as np 
import matplotlib.pyplot as pl 

from synth.policy.state_machine import * 
from environments.car.car import *

from synth.paper_plots.traj1 import * 



env = CarReversePP(1000)
sm1 = StateMachinePolicy(env, [], [])
sm1.read(sm1_file)

sm2 = StateMachinePolicy(env, [], [])
sm2.read(sm2_file)


def get_color(acts):
	a0 = acts[0]/10.0 + 0.5
	a1 = acts[1]/10.0 + 0.5
	#print(a0, a1)
	g_d = np.sqrt((a0 - 1)**2 + (a1 - 1)**2 )
	r_d = np.sqrt((a0)**2 + (a1)**2 )
	b_d = np.sqrt((a0 - 1)**2 + (a1)**2 )
	y_d = np.sqrt((a0)**2 + (a1 - 1)**2 )

	g = np.array([0, 1, 0])
	r = np.array([1, 0, 0])
	b = np.array([0, 0, 0.8])
	y = np.array([1, 1, 0])

	d = 0.7
	c = g*np.exp(-g_d/d) + r*np.exp(-r_d/d) + b*np.exp(-b_d/d) + y*np.exp(-y_d/d) 
	c = c - min(0.3, np.min(c))
	c = c/max(1.0, np.max(c))
	#c = (c - np.min(c))
	c = c.tolist()
	c.append(0.9)
	#c = [min(max(x, 0), 1) for x in c]
	#print(c)

	c = tuple(c)
	return c 



def plot_traj(traj, outfile):
	num_modes = len(traj)//3
	modes = traj[0:num_modes*2].reshape((num_modes, 2))
	times = traj[-num_modes:]

	l = np.sum(times) + (num_modes - 1 ) * 0.5
	fig = pl.figure(figsize = (l/5.0, 0.5))

	x = 0.0 

	for i in range(num_modes):
		acts = modes[i]
		time = times[i]

		c = get_color(acts)

		l,r,t,b = x, x+time, 2, -2
		pl.fill([l, l, r, r], [b, t, t, b], c = c)
		x += time + 0.5

	ax = pl.gca()
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)

	ax.get_xaxis().set_ticks([])
	ax.get_yaxis().set_ticks([])
	pl.tight_layout()
	pl.savefig(outfile)
	pl.show()
	pl.close()

def plot_merge_modes(traj, sm, mode_mapping, outfile):
	num_modes = len(traj)//3
	modes = traj[0:num_modes*2].reshape((num_modes, 2))
	times = traj[-num_modes:]

	l = np.sum(times) + (num_modes - 1 ) * 0.5
	fig = pl.figure(figsize = (l/5.0, 0.5))

	mean_modes = sm.modes.flatten().reshape((len(sm.modes), 2))

	x = 0.0 
	for i in range(num_modes):
		acts = modes[i]
		time = times[i]

		mapping = mode_mapping[i]

		y = 2 
		for j in range(len(mapping)):
			mode_frac = mapping[j]

			c = get_color(mean_modes[j])
			yl = 4*mode_frac
			if yl > 0.1:

				l,r,t,b = x, x+time, y, y - yl 
				y = y - yl
				pl.fill([l, l, r, r], [b, t, t, b], c = c)
		x += time + 0.5

	ax = pl.gca()
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)

	ax.get_xaxis().set_ticks([])
	ax.get_yaxis().set_ticks([])
	pl.tight_layout()
	pl.savefig(outfile)
	pl.show()
	pl.close()


def plot_sm_rollout(sm, init_state, outfile):
	safe_err, goal_err, _, _, _, ref_modes, ref_times, _, _ = sm.evaluate(init_state, max_modes = 8, max_time_per_mode = 400, vis = False)

	print("Reward: ", safe_err + np.sum(goal_err))

	ref_modes = np.copy(ref_modes[:8]).flatten()
	ref_times = (np.copy(ref_times[:8])*0.02).flatten()
	times = np.array(ref_times)/float(40)*100*1.0

	num_modes = len(ref_modes)//2
	modes = ref_modes.reshape((num_modes, 2))

	l = np.sum(times) + (num_modes - 1 ) * 0.5
	fig = pl.figure(figsize = (l/5.0, 0.5))


	x = 0.0 
	for i in range(num_modes):
		acts = modes[i]
		time = times[i]

		c = get_color(acts)

		l,r,t,b = x, x+time, 2, -2
		pl.fill([l, l, r, r], [b, t, t, b], c = c)
		x += time + 0.5


	ax = pl.gca()
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['left'].set_visible(False)

	ax.get_xaxis().set_ticks([])
	ax.get_yaxis().set_ticks([])
	pl.tight_layout()
	pl.savefig(outfile)
	pl.show()
	pl.close()


dir = "/Users/jinala/papers/smsynth/figures/traj/"
plot_traj(traj_11, dir + "/traj_11.pdf")
plot_traj(traj_12, dir + "/traj_12.pdf")
plot_traj(traj_21, dir + "/traj_21.pdf")
plot_traj(traj_22, dir + "/traj_22.pdf")

plot_merge_modes(traj_11, sm1, mode_mapping1[0], dir + "/merge_11.pdf")
plot_merge_modes(traj_12, sm1, mode_mapping1[1], dir + "/merge_12.pdf")
plot_merge_modes(traj_21, sm2, mode_mapping2[0], dir + "/merge_21.pdf")
plot_merge_modes(traj_22, sm2, mode_mapping2[1], dir + "/merge_22.pdf")

plot_sm_rollout(sm1, init_state1, dir + "/sm_11.pdf")
plot_sm_rollout(sm1, init_state2, dir + "/sm_12.pdf")
plot_sm_rollout(sm2, init_state1, dir + "/sm_21.pdf")
plot_sm_rollout(sm2, init_state2, dir + "/sm_22.pdf")
