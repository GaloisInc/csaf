import pidc as pid
import functools


# SWIG Objects of C Pointers
controller = None
signal = None


def swig(func):
    @functools.wraps(func)
    def inner(model, *args, **kwargs):
        global controller, signal
        if controller is None:
            controller = pid.pid_controller_init(model.kp, model.ki, model.kd)
        if signal is None:
            signal = pid.input_signal_init()
        ret = func(model, *args, **kwargs)
        return ret
    return inner


@swig
def model_state_update(model, time_t, state_controller, input_pendulum):
    """implement model_update by interacting with C types"""
    pid.input_signal_update(signal, *input_pendulum)
    pid.pid_controller_update(controller, signal)
    ret =  [pid.pid_controller_state(controller)]
    return ret


@swig
def model_output(model, time_t, state_controller, input_pendulum):
    """implement model output by interacting with C types"""
    pid.input_signal_update(signal, *input_pendulum)
    return [pid.pid_controller_output(controller, signal)]
