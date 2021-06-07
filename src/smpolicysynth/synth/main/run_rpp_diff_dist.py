import sys 
import random
import numpy as np 


from environments.car.car import * 

from synth.main.synth import opt_all 
from synth.main.bench_params import SynthParams, GenParams, get_car_envs



def run_rpp_diff_dist(run_id):
	'''synth_params, gen_params = get_dist1(num_threads = 10)
	dirname = "out/%s_%i"%("rpp_dist1", run_id)
	opt_all(dirname, synth_params, gen_params, resample_env = True, vis = True)'''

	synth_params, gen_params = get_dist2(num_threads = 10)
	dirname = "out/%s_%i"%("rpp_dist2", run_id)
	opt_all(dirname, synth_params, gen_params, resample_env = True, vis = True)

	'''synth_params, gen_params = get_dist3(num_threads = 10)
	dirname = "out/%s_%i"%("rpp_dist3", run_id)
	opt_all(dirname, synth_params, gen_params, resample_env = True, vis = True)

	synth_params, gen_params = get_dist4(num_threads = 10)
	dirname = "out/%s_%i"%("rpp_dist4", run_id)
	opt_all(dirname, synth_params, gen_params, resample_env = True, vis = True)

	synth_params, gen_params = get_dist5(num_threads = 10)
	dirname = "out/%s_%i"%("rpp_dist5", run_id)
	opt_all(dirname, synth_params, gen_params, resample_env = True, vis = True)

	synth_params, gen_params = get_dist6(num_threads = 10)
	dirname = "out/%s_%i"%("rpp_dist6", run_id)
	opt_all(dirname, synth_params, gen_params, resample_env = True, vis = True)'''


def get_dist1(num_threads=10):
	d_min = 13.0
	d_max = 13.5
	envs = get_car_envs(d_min, d_max, num_threads = num_threads)
		
	params = SynthParams()
	params.envs = envs 
	params.nm_unroll = 4 
	params.nm_sm = 3
	params.timesteps = 40 
	params.cond_depth = 1


	gen_params = GenParams()
	gen_params.max_modes = 500
	gen_params.inp_limits = (11.0, 12.0)
	gen_params.timesteps = 40

	return params, gen_params 

def get_dist2(num_threads=10):
	d_min = 12.5
	d_max = 13.5
	envs = get_car_envs(d_min, d_max, num_threads = num_threads)
		
	params = SynthParams()
	params.envs = envs 
	params.nm_unroll = 4 
	params.nm_sm = 3
	params.timesteps = 40 
	params.cond_depth = 1


	gen_params = GenParams()
	gen_params.max_modes = 500
	gen_params.inp_limits = (11.0, 12.0)
	gen_params.timesteps = 40

	return params, gen_params 

def get_dist3(num_threads=10):
	d_min = 12.0
	d_max = 13.5
	envs = get_car_envs(d_min, d_max, num_threads = num_threads)
		
	params = SynthParams()
	params.envs = envs 
	params.nm_unroll = 8 
	params.nm_sm = 3
	params.timesteps = 40 
	params.cond_depth = 1


	gen_params = GenParams()
	gen_params.max_modes = 500
	gen_params.inp_limits = (11.0, 12.0)
	gen_params.timesteps = 40

	return params, gen_params 

def get_dist4(num_threads=10):
	d_min = 11.5
	d_max = 13.5
	envs = get_car_envs(d_min, d_max, num_threads = num_threads)
		
	params = SynthParams()
	params.envs = envs 
	params.nm_unroll = 10
	params.nm_sm = 3
	params.timesteps = 40 
	params.cond_depth = 1


	gen_params = GenParams()
	gen_params.max_modes = 500
	gen_params.inp_limits = (11.0, 12.0)
	gen_params.timesteps = 40

	return params, gen_params 

def get_dist5(num_threads=10):
	d_min = 11.2
	d_max = 13.5
	envs = get_car_envs(d_min, d_max, num_threads = num_threads)
		
	params = SynthParams()
	params.envs = envs 
	params.nm_unroll = 10
	params.nm_sm = 3
	params.timesteps = 40 
	params.cond_depth = 1


	gen_params = GenParams()
	gen_params.max_modes = 500
	gen_params.inp_limits = (11.0, 12.0)
	gen_params.timesteps = 40

	return params, gen_params 

def get_dist6(num_threads=10):
	d_min = 11.0
	d_max = 13.5
	envs = get_car_envs(d_min, d_max, num_threads = num_threads)
		
	params = SynthParams()
	params.envs = envs 
	params.nm_unroll = 12
	params.nm_sm = 3
	params.timesteps = 40 
	params.cond_depth = 1


	gen_params = GenParams()
	gen_params.max_modes = 500
	gen_params.inp_limits = (11.0, 12.0)
	gen_params.timesteps = 40

	return params, gen_params 



if __name__ == '__main__':
	seed = None
	#seed = 2921275267
	np.random.seed(seed)
	print("np seed: ", np.random.get_state()[1][0])
	seed = random.randrange(sys.maxsize)
	#seed = 1070761973242435720
	random.seed(seed)
	print("random seed: ", seed)

	run_id = int(sys.argv[1])
	
	run_rpp_diff_dist(run_id)

