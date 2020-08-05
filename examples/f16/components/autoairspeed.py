import os
import toml

parameters ={}


def main(time=0.0, state=None, input=[0]*4, update=False, output=False):
    global parameters
    if len(parameters.keys()) == 0:
        this_path = os.path.dirname(os.path.realpath(__file__))
        info_file = os.path.join(this_path, "autoairspeed.toml")
        with open(info_file, 'r') as ifp:
            info = toml.load(ifp)
        parameters = info["parameters"]

    uref = get_u_ref(time, state, input, parameters)
    if output:
        return uref
    else:
        return


def get_u_ref(t, cstate, x_f16, parameters):
    setpoint = parameters["setpoint"]
    p_gain = parameters["p_gain"]
    x_dif = setpoint - x_f16[0]
    return [0, 0, 0, p_gain * x_dif]
