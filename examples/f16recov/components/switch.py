# def model_output(model, time_t, state_switch, input_autopilot):
#     controller = input_autopilot[-1]
#     if controller == 0.0:
#         sidx = int(0.0)
#     else:
#         sidx = model.mapper.index(controller)
#     assert len(input_autopilot) == 13
#     return input_autopilot[4*sidx:4*sidx+4]

from .llcontroller import LLCBase
from .variables import State

import numpy as np

class FeedbackControllerSingleSwitch(LLCBase):
    '''
    Class to demo a basic switch Cperf -> Csafe
    '''

    def __init__(self, ctrlLimits, is_discrete, fcntrls, cond, **kwargs):
        '''
        assume_fcntrl is a tuple <assume, feedback_controller>
        '''
        self.cperf = True
        self.fcntrls = fcntrls
        fc0 = fcntrls[0]
        self.xequil, self.uequil, self.ctrl_fn = fc0.xequil, fc0.uequil, fc0.ctrl_fn
        super().__init__(ctrlLimits, is_discrete, **kwargs)

    def cond(self, x_f16):

        ret = False

        # Experimental values
        aoa_max = np.deg2rad(35) # Higher for GCAS + dive (init_cond) or noap
        q_max = np.deg2rad(90)

        if x_f16[State.alpha] >= aoa_max:
            print('WARNING: AOA exceeded limit! Switching to safe controller.')
            ret = True
        if abs(x_f16[State.q]) >= q_max:
            print('WARNING: Pitch Rate exceeded Limit! Switching to safe controller.')
            ret = True
        return ret

    def compute(self, x_f16, cstate):
        """ get the reference commands for the control surfaces """

        x_ctrl = type(self).permute2xctrl(self.xerror(x_f16), cstate)
        if self.cperf and self.cond(x_f16):
            fc = self.fcntrls[1]
            self.xequil, self.uequil, self.ctrl_fn = fc.xequil, fc.uequil, fc.ctrl_fn
            self.cperf = False

        return self.ctrl_fn(x_ctrl) # Full Control

