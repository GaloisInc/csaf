import os
import toml

parameters = {}

def main(time=0.0, state=None, input=[0]*4, update=False, output=False):
    global parameters
    if len(parameters.keys()) == 0:
        this_path = os.path.dirname(os.path.realpath(__file__))
        info_file = os.path.join(this_path, "autoaltitude.toml")
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
    xequil = parameters["xequil"]
    airspeed = x_f16[0]   # Vt            (ft/sec)
    alpha = x_f16[1]      # AoA           (rad)
    theta = x_f16[4]      # Pitch angle   (rad)
    gamma = theta - alpha # Path angle    (rad)
    h = x_f16[11]         # Altitude      (feet)

    # Proportional Control
    k_alt = 0.025
    h_error = setpoint - h
    Nz = k_alt * h_error # Allows stacking of cmds

    # (Psuedo) Derivative control using path angle
    k_gamma = 25
    Nz = Nz - k_gamma*gamma

    # try to maintain a fixed airspeed near trim point
    K_vt = 0.25
    airspeed_setpoint = 540
    throttle = -K_vt * (airspeed - xequil[0])

    return [Nz, 0, 0, throttle]
