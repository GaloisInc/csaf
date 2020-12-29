import numpy as np

import helpers.llc_helper as lh
from helpers import fops, llc_nn_helper as nnh

class Model:
    default_np_model_path = fops.path(('examples', 'f16', 'components', 'trained_models', 'np', '128_128.npz'))
    def __init__(self, ctrlLimits):
        try:
            p = np.load(Model.default_np_model_path)
            w0, b0, w1, b1, w2, b2 = p['w0'], p['b0'], p['w1'], p['b1'], p['w2'], p['b2']
        except FileNotFoundError:
            raise FileNotFoundError(f'Missing numpy model file {Model.default_np_model_path}')
            #w0, b0, w1, b1, w2, b2 = extract_ddpg_nn.extract_zikangs_model(default_np_model_path)

        self.w0, self.b0, self.w1, self.b1, self.w2, self.b2 = w0, b0, w1, b1, w2, b2
        cl = ctrlLimits
        self.u_high = np.array([cl.ThrottleMax, cl.ElevatorMaxDeg, cl.AileronMaxDeg, cl.RudderMaxDeg])

    def predict(self, input0):
        def relu(x): return np.maximum(x, 0)
        output0 = relu(input0 @ self.w0 + self.b0)
        output1 = relu(output0 @ self.w1 + self.b1)
        output2 = np.tanh(output1 @ self.w2 + self.b2)
        #output2 = np.tanh(relu(relu(x @ w0 + b0) @w1 + b1) @ w2 + b2)
        output = output2 * self.u_high[1:]
        return output

def model_init(model):
    """load trained model"""
    model.parameters['llc'] = nnh.LowLevelControllerNN(Model(lh.CtrlLimits()), model)

def model_output(model, t, state_controller, input_all):
    """ get the reference commands for the control surfaces """
    return model.parameters['llc'].output(t, state_controller, input_all)

def model_state_update(model, t, state_controller, input_all):
    """ get the derivatives of the integrators in the low-level controller """
    #return llc.step(t, state_controller, input_all)
    return model.parameters['llc']._der(t, state_controller, input_all)
