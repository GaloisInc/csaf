class StateIndex:
    'list of static state indices'
    # TODO: is this somewhere else?

    VT = 0
    VEL = 0  # alias

    ALPHA = 1
    BETA = 2
    PHI = 3  # roll angle
    THETA = 4  # pitch angle
    PSI = 5  # yaw angle

    P = 6
    Q = 7
    R = 8

    POSN = 9
    POS_N = 9

    POSE = 10
    POS_E = 10

    ALT = 11
    H = 11

    POW = 12


def get_state_names():
    'returns a list of state variable names'

    return ['vt', 'alpha', 'beta', 'phi', 'theta', 'psi', 'P', 'Q', 'R', 'pos_n', 'pos_e', 'alt', 'pow']
