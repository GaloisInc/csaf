import os
import toml
import numpy as np


class GcasAutopilot():
    """The ground-collision avoidance system autopilot logic"""

    STATE_START = 'Waiting'
    STATE_ROLL = 'Roll'
    STATE_PULL = 'Pull'
    STATE_DONE = 'Finished'


parameters = {}
#state = GcasAutopilot.STATE_START


def main(time=0.0, state='Waiting', input=[0]*4, update=False, output=False, fda=False):
    """TODO: actually implement the autopilot"""
    global parameters
    if len(parameters.keys()) == 0:
        this_path = os.path.dirname(os.path.realpath(__file__))
        info_file = os.path.join(this_path, "autopilot.toml")
        with open(info_file, 'r') as ifp:
            info = toml.load(ifp)
        parameters = info["parameters"]

    nstate = advance_discrete_state(time, state, input)
    uref = get_u_ref(time, state, input)
    if output:
        return list(uref)
    elif fda:
        return list([nstate])
    else:
        return list([nstate])


def advance_discrete_state(t, cstate, x_f16):
    """advance the discrete state based on the current aircraft state"""
    #global state
    state = cstate[0]
    #if state == "Finished":
    #    return "Waiting"
    rv = False

    # Pull out important variables for ease of use
    phi = x_f16[3]             # Roll angle    (rad)
    p = x_f16[6]               # Roll rate     (rad/sec)
    theta = x_f16[4]           # Pitch angle   (rad)
    alpha = x_f16[1]           # AoA           (rad)

    eps_phi = np.deg2rad(5)   # Max roll angle magnitude before pulling g's
    eps_p = np.deg2rad(1)     # Max roll rate magnitude before pulling g's
    path_goal = np.deg2rad(0) # Final desired path angle
    man_start = 2 # maneuver starts after 2 seconds

    old_state = state
    if state == GcasAutopilot.STATE_START:
        if t >= man_start:
            state = GcasAutopilot.STATE_ROLL
            rv = True

    elif state == GcasAutopilot.STATE_ROLL:
        # Determine which angle is "level" (0, 180, 360, 720, etc)
        radsFromWingsLevel = round(phi/np.pi)

        # Until wings are "level" & roll rate is small
        if abs(phi - np.pi * radsFromWingsLevel) < eps_phi and abs(p) < eps_p:
            state = GcasAutopilot.STATE_PULL
            rv = True

    elif state == GcasAutopilot.STATE_PULL:
        radsFromNoseLevel = round((theta - alpha) / (2 * np.pi))

        if (theta - alpha) - 2 * np.pi * radsFromNoseLevel > path_goal:
            state = GcasAutopilot.STATE_DONE
            rv = True

    #logger.info(f'State Tr: {old_state} -> {state}')

    return state


def get_u_ref(t, cstate, x_f16):
    """for the current discrete state, get the reference inputs signals"""
    global parameters

    state = cstate[0]

    NzMax = parameters["NzMax"]
    xequil = parameters["xequil"]

    # Zero default commands
    Nz = 0
    ps = 0
    Ny_r = 0
    throttle = 0

    # GCAS logic
    # Concept:
    # Roll until wings level (in the shortest direction)
    # When abs(roll rate) < threshold, pull X g's until pitch angle > X deg
    # Choose threshold values:

    Nz_des = min(5, NzMax) # Desired maneuver g's

    # Pull out important variables for ease of use
    phi = x_f16[3]             # Roll angle    (rad)
    p = x_f16[6]               # Roll rate     (rad/sec)
    q = x_f16[7]               # Pitch rate    (rad/sec)
    theta = x_f16[4]           # Pitch angle   (rad)
    alpha = x_f16[1]           # AoA           (rad)
    # Note: pathAngle = theta - alpha

    if state == GcasAutopilot.STATE_START:
        pass # Do nothing
    elif state == GcasAutopilot.STATE_ROLL:
        # Determine which angle is "level" (0, 180, 360, 720, etc)
        radsFromWingsLevel = round(phi/np.pi)

        # PD Control until phi == pi*radsFromWingsLevel
        K_prop = 4
        K_der = K_prop * 0.3

        ps = -(phi - np.pi * radsFromWingsLevel) * K_prop - p * K_der
    elif state == GcasAutopilot.STATE_PULL:
        Nz = Nz_des
    elif state == GcasAutopilot.STATE_DONE:
        # steady-level hold
        # Set Proportional-Derivative control gains for roll
        K_prop = 1
        K_der = K_prop*0.3

        # Determine which angle is "level" (0, 180, 360, 720, etc)
        radsFromWingsLevel = round(phi/np.pi)
        # PD Control on phi using roll rate
        ps = -(phi-np.pi*radsFromWingsLevel)*K_prop - p*K_der

        # Set Proportional-Derivative control gains for pitch
        K_prop2 = 2
        K_der2 = K_prop*0.3

        # Determine "which" angle is level (0, 360, 720, etc)
        radsFromNoseLevel = round((theta-alpha)/np.pi)

        # PD Control on theta using Nz
        Nz = -(theta - alpha - np.pi*radsFromNoseLevel) * K_prop2 - p*K_der2

    # basic speed control
    K_vt = 0.25
    throttle = -K_vt * (x_f16[0] - xequil[0])

    return Nz, ps, Ny_r, throttle


if __name__ == '__main__':
    import fire
    fire.Fire(main)