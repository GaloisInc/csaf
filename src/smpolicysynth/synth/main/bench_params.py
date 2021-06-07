import sys 

from environments.cartpole.cartpole import *
from environments.pendulum.pendulum import * 
from environments.mountain_car.mountain_car import *
from environments.acrobot.acrobot import *
from environments.car.car import * 
from environments.toy.toy import *
from environments.f16.f16 import *

from environments.quadcopter.quad import * 
from environments.quadcopter.quad_po import * 

#from environments.hopper.hopper import * 

#from environments.swimmer.swimmer3.swimmer import * 
#from environments.swimmer.swimmer4.swimmer4 import *
#from environments.swimmer.swimmer4_small.swimmer4_small import *  
#from environments.swimmer.swimmer5.swimmer5 import *

from environments.pomdp.door.door import * 

class GenParams:
	inp_limits = None 
	max_modes = 10
	timesteps = 20

class SynthParams:
	envs = []
	nm_unroll = 10
	nm_sm = 3
	timesteps = 20
	cond_depth = 1



def get_toy_envs(num_threads = 1):
	envs = []
	for i in range(num_threads):
		env = Toy(100)
		envs.append(env)
	return envs

def get_car_envs(d_min, d_max, num_threads = 1):
	envs = []
	for i in range(num_threads):
		env = CarReversePP(10000)
		env.set_inp_limits((d_min, d_max))
		envs.append(env)
	return envs


def get_pen_inversion_envs(m_min, m_max, num_threads = 1):
	envs = []
	for i in range(num_threads):
		env = Pendulum(10000)
		env.set_inp_limits((m_min, m_max))
		envs.append(env)
	return envs 

def get_quad_envs(num_threads = 1):
	envs = []
	for i in range(num_threads):
		env = Quadcopter(5000)
		envs.append(env)
	return envs 

def get_f16_envs(num_threads = 1):
        envs = []
        for i in range(num_threads):
                env = F16(5000)
                envs.append(env)
        return envs

def get_quad_po_envs(num_threads = 1):
	envs = []
	for i in range(num_threads):
		env = QuadcopterPO(5000)
		envs.append(env)
	return envs 

def get_cartpole_envs(desired_time, length, num_threads = 1):
	envs = []
	for i in range(num_threads):
		env = CartPole(50000)
		env.set_inp_limits((desired_time, length))
		envs.append(env)
	return envs 

def get_acrobot_envs(m_min, m_max, num_threads = 1):
	envs = []
	for i in range(num_threads):
		env = Acrobot(50000)
		env.set_inp_limits((m_min, m_max))
		envs.append(env)
	return envs 

def get_walker_envs(num_threads = 1):
	envs = []
	for i in range(num_threads):
		env = Walker(5000)
		envs.append(env)
	return envs 

def get_hopper_envs(num_threads = 1):
	envs = []
	for i in range(num_threads):
		env = Hopper(5000)
		envs.append(env)
	return envs 

def get_swimmer_envs(num_threads = 1):
	envs = []
	for i in range(num_threads):
		env = Swimmer(5000)
		envs.append(env)
	return envs 

def get_swimmer4_envs(num_threads = 1):
	envs = []
	for i in range(num_threads):
		env = Swimmer4(5000)
		envs.append(env)
	return envs 

def get_swimmer4_small_envs(num_threads = 1):
	envs = []
	for i in range(num_threads):
		env = Swimmer4Small(5000)
		envs.append(env)
	return envs 

def get_swimmer5_envs(num_threads = 1):
	envs = []
	for i in range(num_threads):
		env = Swimmer5(5000)
		envs.append(env)
	return envs 

def get_mountain_car_envs(p_min, p_max, num_threads = 1):
	envs = []
	for i in range(num_threads):
		env = MountainCar(50000)
		env.set_inp_limits((p_min, p_max))
		envs.append(env)
	return envs 

def get_door_envs(num_threads = 1):
	envs = []
	for i in range(num_threads):
		env = Door(5000)
		envs.append(env)
	return envs 


def get_bench_params(name, num_threads = 10):
	if name == "toy":
		envs = get_toy_envs(num_threads = num_threads)
		params = SynthParams()
		params.envs = envs 
		params.nm_unroll = 2
		params.nm_sm = 2
		params.timesteps = 40 
		params.cond_depth = 1

		gen_params = GenParams()
		gen_params.max_modes = 4
		gen_params.timesteps = 40

		return params, gen_params
		

	if name == "car":
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
		

	if name == "pen":
		m_min = 1.0
		m_max = 1.5
		envs = get_pen_inversion_envs(m_min, m_max, num_threads = num_threads)

		params = SynthParams()
		params.envs = envs
		params.nm_unroll = 6
		params.nm_sm = 2
		params.timesteps = 60
		params.cond_depth = 2

		gen_params = GenParams()
		gen_params.max_modes = 100
		gen_params.inp_limits = (1.5, 5)
		gen_params.timesteps = 100
	
		return params, gen_params

	if name == "quadpo":
		envs = get_quad_po_envs(num_threads=num_threads)
		params = SynthParams()
		params.envs = envs 
		params.nm_unroll = 10
		params.nm_sm = 2
		params.timesteps = 50
		params.cond_depth = 1
		
		gen_params = GenParams()
		gen_params.inp_limits = (120, 120)
		gen_params.max_modes = 50
		gen_params.timesteps = 100
		return params, gen_params


	if name == "quad":
		envs = get_quad_envs(num_threads=num_threads)
		params = SynthParams()
		params.envs = envs 
		params.nm_unroll = 8
		params.nm_sm = 2
		params.timesteps = 60
		params.cond_depth = 1

		gen_params = GenParams()
		gen_params.inp_limits = (80, 80)
		gen_params.max_modes = 50
		gen_params.timesteps = 100
		return params, gen_params

	if name == "f16":
		envs = get_f16_envs(num_threads=num_threads)
		params = SynthParams()
		params.envs = envs 
		params.nm_unroll = 8
		params.nm_sm = 2
		params.timesteps = 60
		params.cond_depth = 2

		gen_params = GenParams()
		gen_params.inp_limits = (80, 80)
		gen_params.max_modes = 50
		gen_params.timesteps = 100
		return params, gen_params
    

	if name == "cp":
		desired_time = 5.0
		length = 0.5
		envs = get_cartpole_envs(desired_time, length, num_threads=num_threads)
		params = SynthParams()
		params.envs = envs 
		params.nm_unroll = 40
		params.nm_sm = 2
		params.timesteps = 20
		params.cond_depth = 2

		gen_params = GenParams()
		gen_params.max_modes = 10000
		gen_params.inp_limits = (300,1.0)
		gen_params.timesteps = 50
		
		return params, gen_params
		
	if name == "acrobot":
		m_min = 0.2
		m_max = 0.5
		envs = get_acrobot_envs(m_min, m_max, num_threads=num_threads)

		params = SynthParams()
		params.envs = envs 
		params.nm_unroll = 8
		params.nm_sm = 2
		params.timesteps = 20
		params.cond_depth = 2

		gen_params = GenParams()
		gen_params.max_modes = 50
		gen_params.timesteps = 50
		gen_params.inp_limits = (0.5, 2)
		return params, gen_params 

	if name == "walker":
		envs = get_walker_envs(num_threads=num_threads)

		params = SynthParams()
		params.envs = envs
		params.nm_unroll = 20
		params.nm_sm = 4
		params.timesteps = 20
		params.cond_depth = 2

		gen_params = GenParams()
		gen_params.max_modes = 1000
		gen_params.inp_limits = (10,10)
		gen_params.timesteps = 100
		
		return params, gen_params

	if name == "hopper":
		envs = get_hopper_envs(num_threads=num_threads)

		params = SynthParams()
		params.envs = envs
		params.nm_unroll = 10
		params.nm_sm = 2
		params.timesteps = 50
		params.cond_depth = 2

		gen_params = GenParams()
		gen_params.max_modes = 1000
		gen_params.inp_limits = (10,10)
		gen_params.timesteps = 100
		return params, gen_params

	if name == "swimmer":
		envs = get_swimmer_envs(num_threads=num_threads)

		params = SynthParams()
		params.envs = envs
		params.nm_unroll = 20
		params.nm_sm = 3
		params.timesteps = 20
		params.cond_depth = 2

		gen_params = GenParams()
		gen_params.max_modes = 1000
		gen_params.inp_limits = (20,20)
		gen_params.timesteps = 100
		return params, gen_params

	if name == "swimmer4":
		envs = get_swimmer4_envs(num_threads=num_threads)

		params = SynthParams()
		params.envs = envs
		params.nm_unroll = 20
		params.nm_sm = 4
		params.timesteps = 20
		params.cond_depth = 2

		gen_params = GenParams()
		gen_params.max_modes = 1000
		gen_params.inp_limits = (20,20)
		gen_params.timesteps = 20
		return params, gen_params

	if name == "swimmer4small":
		envs = get_swimmer4_small_envs(num_threads=num_threads)

		params = SynthParams()
		params.envs = envs
		params.nm_unroll = 20
		params.nm_sm = 4
		params.timesteps = 20
		params.cond_depth = 2

		gen_params = GenParams()
		gen_params.max_modes = 1000
		gen_params.inp_limits = (20,20)
		gen_params.timesteps = 20
		return params, gen_params

	if name == "swimmer5":
		envs = get_swimmer5_envs(num_threads=num_threads)

		params = SynthParams()
		params.envs = envs
		params.nm_unroll = 20
		params.nm_sm = 4
		params.timesteps = 20
		params.cond_depth = 2

		gen_params = GenParams()
		gen_params.max_modes = 1000
		gen_params.inp_limits = (20,20)
		gen_params.timesteps = 20
		return params, gen_params

	if name == "mc":
		p_min = 5
		p_max = 10
		envs = get_mountain_car_envs(p_min, p_max, num_threads=num_threads)

		params = SynthParams()
		params.envs = envs 
		params.nm_unroll = 6
		params.nm_sm = 2
		params.timesteps = 50
		params.cond_depth = 1

		gen_params = GenParams()
		gen_params.max_modes = 20
		gen_params.inp_limits = (3, 5)
		gen_params.timesteps = 100
		
		return params, gen_params 

	if name == "door":
		envs = get_door_envs(num_threads=num_threads)

		params = SynthParams()
		params.envs = envs 
		params.nm_unroll = 8
		params.nm_sm = 3
		params.timesteps = 20
		params.cond_depth = 1

		gen_params = GenParams()
		gen_params.max_modes = 20
		gen_params.timesteps = 100
		
		return params, gen_params 


