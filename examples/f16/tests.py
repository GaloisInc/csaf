import numpy as np

import static_tests


def overshoot_nz(trajs,start_time=1.0):
    """
    F16 specific version of overshoot
    """
    return static_tests.overshoot(
                    trajs['autopilot'].times,
                    np.array(getattr(trajs['autopilot'], 'outputs'))[:, 0],
                    trajs['plant'].times,
                    np.array(getattr(trajs['plant'], 'outputs'))[:, 0])