import numpy as np


def model_output(model, time_t, state_controller, input_pendulum):
    """
    Inverted Pendulum System -- LQR Gain Matrix
    Taken from
    http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlDigital#4
    Run at a sampling period Ts = 1/100 s
    :return: K gain matrix
    """
    ss_inv_pend_lqr = np.array(model.xform)[np.newaxis, :]
    return list(input_pendulum[-1] - (ss_inv_pend_lqr @ np.array(input_pendulum)[:-1, np.newaxis]).flatten())
