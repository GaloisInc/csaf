import numpy as np
from helpers.variables import State

def cond(xf16):

    ret = False

    # Experimental values
    aoa_max = np.deg2rad(35) # Higher for GCAS + dive (init_cond) or noap
    q_max = np.deg2rad(90)

    if xf16[State.alpha] >= aoa_max:
        print('WARNING: AOA exceeded limit! Switching to safe controller.')
        ret = True
    if abs(xf16[State.q]) >= q_max:
        print('WARNING: Pitch Rate exceeded Limit! Switching to safe controller.')
        ret = True
    return ret

def model_output(model, t, state_switch, input_xf16_cperf_crecov):
    xf16 = input_xf16_cperf_crecov[0:len(State)]
    U = input_xf16_cperf_crecov[len(State):]
    cperf_u, crecov_u = U[0:4], U[4:]
    if cond(xf16):
        return crecov_u
    else:
        return cperf_u
