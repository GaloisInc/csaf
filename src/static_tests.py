import numpy as np
from numpy import nan

def overshoot(ref_time, reference, res_time, response,start_time=1.0):
    """Note that the reference and the response are likely being sampled
    at different rates. 
    """
    _, ref_idx = min((val, idx) for (idx, val) in enumerate(ref_time) if val > start_time)
    _, res_idx = min((val, idx) for (idx, val) in enumerate(res_time) if val > start_time)
    ref = reference[ref_idx:]
    res = response[res_idx:]
    return max(res) - max(ref)