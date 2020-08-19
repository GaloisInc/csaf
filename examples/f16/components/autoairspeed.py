import os
import toml

import autopilot_helper as ah

parameters ={}


def main(time=0.0, state=None, input=[0]*4, update=False, output=False):
    global parameters
    if len(parameters.keys()) == 0:
        this_path = os.path.dirname(os.path.realpath(__file__))
        info_file = os.path.join(this_path, "autoairspeed.toml")
        with open(info_file, 'r') as ifp:
            info = toml.load(ifp)
        parameters = info["parameters"]

    uref = get_u_ref(time, state, input)
    if output:
        return uref
    else:
        return


def get_u_ref(t, cstate, x_f16):
    global parameters
    xequil = parameters["xequil"]
    vt = x_f16[0]
    vt_des = xequil[0]

    # basic speed control
    throttle = ah.p_cntrl(kp=0.25, e=(vt_des - vt))
    Nz, ps, Ny_r = 0, 0, 0
    return Nz, ps, Ny_r, throttle
