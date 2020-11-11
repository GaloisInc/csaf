import numpy as np
import matplotlib.pyplot as plt

# CSAF Imports
import csaftop.csaf.config as cconf
import csaftop.csaf.system as csys

# create a csaf configuration out of toml
my_conf = cconf.SystemConfig.from_toml("/home/averma/csaf_architecture/examples/f16/f16_simple_config.toml")  #"/csaf-system/f16_simple_config.toml")

# termination condition
def ground_collision_condition(cname, outs):
        """ground collision premature termnation condition"""
        return cname == "plant" and outs["states"][11] <= 0.0

# create pub/sub components out of the configuration
my_system = csys.System.from_config(my_conf)

# create an environment from the system, allowing us to act as the controller
my_env = csys.SystemEnv("autopilot", my_system, terminating_conditions=ground_collision_condition)

# collect aircraft states
pstates = []

# send signal of zeros
ctrl_signal = [0.,0.,0.,0.]

# step through simulation and collect f16 states
# StopIteration is thrown when the terminating conditions are achieved
do_sim = True
while do_sim:
    try:
        ob, r, d, _ = my_env.step(ctrl_signal)
        print(ob, r, d)
        # pstates.append(comp_input['plant-states'])
        if d:
            do_sim = False
            print('done')
    except StopIteration:
        do_sim = False
pstates = np.array(pstates)

# plot the results
plt.plot(pstates[:, 11], label='F16 Altitude')
plt.xlabel("Step Index [n]")
plt.ylabel("[ft]")
plt.legend()
plt.show()