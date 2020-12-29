import numpy as np

def get_variables(trajs, reference_component, reference_index,
             response_component, response_index,
             reference_subtopic="outputs", response_subtopic="outputs"):
    ref_time = trajs[reference_component].times
    reference = np.array(getattr(trajs[reference_component], reference_subtopic))[:, reference_index]

    res_time = trajs[response_component].times
    response = np.array(getattr(trajs[response_component], response_subtopic))[:, response_index]
    return ref_time, reference, res_time, response

def max_norm_deviation(trajs, reference_component, reference_index,
             response_component, response_index,
             reference_subtopic="outputs", response_subtopic="outputs",
             start_time=1.0, threshold = 0.1):
    """Note that the reference and the response are likely being sampled
    at different rates.
    Returns true if the max difference between the normalized reference and
    the normalized response is below threshold.
    The default threshold is 0.1 (10%)
    """
    # Fetch variables
    ref_time, reference, res_time, response = get_variables(trajs,
            reference_component, reference_index, response_component,
            response_index, reference_subtopic, response_subtopic)

    # Get only values *after* start time
    _, ref_idx = min((val, idx) for (idx, val) in enumerate(ref_time) if val > start_time)
    _, res_idx = min((val, idx) for (idx, val) in enumerate(res_time) if val > start_time)
    # Select relevant datapoints
    ref = reference[ref_idx:]
    res = response[res_idx:]
    # Normalize
    norm = max(max(res),max(ref))
    ref = ref/norm
    res = res/norm
    # Calculate max deviation
    max_dev = abs(max(res) - max(ref))
    # Pass/fail?
    return max_dev <= threshold
