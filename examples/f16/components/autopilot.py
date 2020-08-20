import os
import toml
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import autopilot_helper as ah


class GcasAutopilot():
    """The ground-collision avoidance system autopilot logic"""

    STATE_START = 'Waiting'
    STATE_ROLL = 'Roll'
    STATE_PULL = 'Pull'
    STATE_DONE = 'Finished'


parameters = {}


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
    state = cstate[0]

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

    elif state == GcasAutopilot.STATE_ROLL:
        # Determine which angle is "level" (0, 180, 360, 720, etc)
        radsFromWingsLevel = round(phi/np.pi)

        # Until wings are "level" & roll rate is small
        if abs(phi - np.pi * radsFromWingsLevel) < eps_phi and abs(p) < eps_p:
            state = GcasAutopilot.STATE_PULL

    elif state == GcasAutopilot.STATE_PULL:
        radsFromNoseLevel = round((theta - alpha) / (2 * np.pi))

        if (theta - alpha) - 2 * np.pi * radsFromNoseLevel > path_goal:
            state = GcasAutopilot.STATE_DONE

    return state


def get_u_ref(t, cstate, x_f16):
    """for the current discrete state, get the reference inputs signals"""
    global parameters

    state = cstate[0]
    NzMax = parameters["NzMax"]
    xequil = parameters["xequil"]

    Nz_des = min(5, NzMax) # Desired maneuver g's

    # Pull out important variables for ease of use
    phi = x_f16[3]             # Roll angle    (rad)
    p = x_f16[6]               # Roll rate     (rad/sec)
    theta = x_f16[4]           # Pitch angle   (rad)
    alpha = x_f16[1]           # AoA           (rad)
    vt = x_f16[0]
    # Note: pathAngle = theta - alpha
    gamma = theta-alpha

    # Determine which angle is "level" (0, 180, 360, 720, etc)
    radsFromWingsLevel = round(phi/np.pi)
    phi_des = np.pi*radsFromWingsLevel
    p_des = 0

    # Determine "which" angle is level (0, 360, 720, etc)
    radsFromNoseLevel = round(gamma/np.pi)
    gamma_des = np.pi*radsFromNoseLevel

    if state == GcasAutopilot.STATE_START:
        Nz, ps = 0, 0
    elif state == GcasAutopilot.STATE_ROLL:
        Nz, ps = 0, state_roll(phi_des, phi, p)
    elif state == GcasAutopilot.STATE_PULL:
        Nz, ps = state_pull(Nz_des), 0
    elif state == GcasAutopilot.STATE_DONE:
        Nz, ps = state_done(gamma_des, phi_des, p_des, gamma, phi, p)

    # XXX: Because Nz and throttle control are different, what if Nz_des
    # tries to decrease the force and slows speed and maybe altitude gain?

    # basic speed control
    throttle = ah.p_cntrl(kp=0.25, e=(xequil[0]-vt))
    Ny_r = 0
    # New references
    return Nz, ps, Ny_r, throttle


def state_roll(phi_des, phi, p):
    # Determine which angle is "level" (0, 180, 360, 720, etc)

    # PD Control until phi == phi_des
    K_prop = 4#000
    K_der = K_prop * 0.3

    ps = -(phi - phi_des) * K_prop - p * K_der
    return ps


def state_pull(Nz_des):
    Nz = Nz_des
    return Nz


def state_done(gamma_des, phi_des, p_des, gamma, phi, p):
    # steady-level hold
    # Set Proportional-Derivative control gains for roll
    K_prop = 1
    K_der = K_prop*0.3
    e_ps, ed_ps = phi_des - phi, p_des-p
    # PD Control on phi using roll rate
    ps = ah.pd_cntrl(K_prop, K_der, e_ps, ed_ps)

    # Set Proportional-Derivative control gains for pitch

    # Unstability counterexample
    #./rundemo.py --test-id=dc --endtime=19.4 --animate
    #K_prop2 = 78

    K_prop2 = 2
    K_der2 = K_prop*0.3
    #XXX: Why is roll rate (p) being used here? Should be q, the pitch rate
    e_nz, ed_nz = gamma_des-gamma, p_des-p
    # PD Control on theta using Nz
    #Nz = -(gamma - gamma_des) * K_prop2 - p*K_der2
    Nz = ah.pd_cntrl(K_prop2, K_der2, e_nz, ed_nz)
    return Nz, ps


if __name__ == '__main__':
    import fire
    fire.Fire(main)