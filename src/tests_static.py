import numpy as np
from numpy import nan

def get_variables(trajs, reference_component, reference_index,
             response_component, response_index,
             reference_subtopic="outputs", response_subtopic="outputs"):
    ref_time = trajs[reference_component].times
    reference = np.array(getattr(trajs[reference_component], reference_subtopic))[:, reference_index]

    res_time = trajs[response_component].times
    response = np.array(getattr(trajs[response_component], response_subtopic))[:, response_index]
    return ref_time, reference, res_time, response

def overshoot(trajs, reference_component, reference_index,
             response_component, response_index,
             reference_subtopic="outputs", response_subtopic="outputs",
             start_time=1.0):
    """Note that the reference and the response are likely being sampled
    at different rates.
    """
    ref_time, reference, res_time, response = get_variables(trajs,
            reference_component, reference_index, response_component,
            response_index, reference_subtopic, response_subtopic)

    _, ref_idx = min((val, idx) for (idx, val) in enumerate(ref_time) if val > start_time)
    _, res_idx = min((val, idx) for (idx, val) in enumerate(res_time) if val > start_time)
    ref = reference[ref_idx:]
    res = response[res_idx:]
    return max(res) - max(ref)
