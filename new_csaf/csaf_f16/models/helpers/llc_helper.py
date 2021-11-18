import numpy as np

from math import sin, cos

from csaf_f16.models.helpers.variables import State


class FeedbackController:
    def __init__(self, ctrlLimits, model, ctrl_fn, xequil, uequil):
        self.ctrlLimits = ctrlLimits
        self.model = model
        self.ctrl_fn = ctrl_fn
        self.xequil, self.uequil = xequil, uequil

    @staticmethod
    def permute2xctrl(x_f16, cstate):
        state = np.concatenate((x_f16, cstate))
        # Reorder states to match controller:
        # [alpha, q, int_e_Nz, beta, p, r, int_e_ps, int_e_Ny_r]
        return np.array([state[i] for i in [1, 7, 13, 2, 6, 8, 14, 15]])

    def xerror(self, x_f16):
        # Calculate perturbation from trim state
        return x_f16 - self.xequil

    def compute(self, x_f16, cstate):
        """ get the reference commands for the control surfaces """
        # Calculate control
        x_ctrl = type(self).permute2xctrl(self.xerror(x_f16), cstate)
        return self.ctrl_fn(x_ctrl)  # Full Control

    def output(self, t, cstate, u):
        assert len(u) == 21
        # TODO: hard coded indices!
        x_f16, y, u_ref = u[:13], u[13:17], u[17:]

        # Initialize control vectors
        u_deg = np.zeros((4,))  # throt, ele, ail, rud
        u_deg[1:4] = self.compute(x_f16, cstate)

        # Set throttle as directed from output of getOuterLoopCtrl(...)
        u_deg[0] = u_ref[3]

        # Add in equilibrium control
        u_deg[0:4] += self.uequil
        u_deg = clip_u(self.model, u_deg)

        return u_deg

    def ps(self, x_f16):
        xerror = self.xerror(x_f16)
        # Nonlinear (Actual): ps = p * cos(alpha) + r * sin(alpha)
        ps = xerror[State.p] * cos(xerror[State.alpha]) + xerror[State.r] * sin(xerror[State.alpha])
        return ps

    def ps_(self, x_f16):
        ps_ = x_f16[State.p] * cos(x_f16[State.alpha]) + x_f16[State.r] * sin(x_f16[State.alpha])
        xequil = self.xequil
        ps_equil = xequil[State.p] * cos(xequil[State.alpha]) + xequil[State.r] * sin(xequil[State.alpha])
        return ps_ - ps_equil

    def Ny_r(self, x_f16, Ny):
        # Calculate (side force + yaw rate) term
        xerror = self.xerror(x_f16)
        Ny_r = Ny + xerror[State.r]
        return Ny_r

    def _der(self, t, cstate, u):
        'get the derivatives of the integrators in the low-level controller'
        x_f16, y, u_ref = u[:13], u[13:17], u[17:]
        assert len(u_ref) > 2, f"{len(u)}"
        Nz, Ny, az, ay = y

        ps = self.ps(x_f16)
        Ny_r = self.Ny_r(x_f16, Ny)

        return [Nz - u_ref[0], ps - u_ref[1], Ny_r - u_ref[2]]

    def step(self, sampling_period, t, cstate, u):
        'get the next state of the integrators in the low-level controller'

        x_f16, y, u_ref = u[:13], u[13:17], u[17:]
        Nz, Ny, _, _ = y

        ps, Ny_r = self.ps(x_f16), self.Ny_r(x_f16, Ny)

        error = np.array([Nz - u_ref[0], ps - u_ref[1], Ny_r - u_ref[2]])
        # Integrate/Sum the error
        error_ = cstate + error * sampling_period

        return error_


class CtrlLimits:
    'Control Limits'

    def __init__(self):
        self.ThrottleMax = 1  # Afterburner on for throttle > 0.7
        self.ThrottleMin = 0
        self.ElevatorMaxDeg = 25
        self.ElevatorMinDeg = -25
        self.AileronMaxDeg = 21.5
        self.AileronMinDeg = -21.5
        self.RudderMaxDeg = 30
        self.RudderMinDeg = -30
        self.MaxBankDeg = 60  # For turning maneuvers
        self.NzMax = 6  # Should this not be at least 9Gs?
        self.NzMin = -1

        self.check()

    def check(self):
        'check that limits are in bounds'

        ctrlLimits = self

        assert not (ctrlLimits.ThrottleMin < 0 or ctrlLimits.ThrottleMax > 1), 'ctrlLimits: Throttle Limits (0 to 1)'

        assert not (ctrlLimits.ElevatorMaxDeg > 25 or ctrlLimits.ElevatorMinDeg < -25), \
            'ctrlLimits: Elevator Limits (-25 deg to 25 deg)'

        assert not (ctrlLimits.AileronMaxDeg > 21.5 or ctrlLimits.AileronMinDeg < -21.5), \
            'ctrlLimits: Aileron Limits (-21.5 deg to 21.5 deg)'

        assert not (ctrlLimits.RudderMaxDeg > 30 or ctrlLimits.RudderMinDeg < -30), \
            'ctrlLimits: Rudder Limits (-30 deg to 30 deg)'


def clip_u(model, u_deg):
    """ helper to clip controller output within defined control limits
    :param u_deg: controller output
    :param parameters: containing equilibrium state (uequil) and control limits (ctrlLimits)
    :return: saturated control output
    """
    parameters = model.parameters
    ThrottleMin, ThrottleMax = parameters["throttle_min"], parameters["throttle_max"]
    ElevatorMinDeg, ElevatorMaxDeg = parameters["elevator_min"], parameters["elevator_max"]
    AileronMinDeg, AileronMaxDeg = parameters["aileron_min"], parameters["aileron_max"]
    RudderMinDeg, RudderMaxDeg = parameters["rudder_min"], parameters["rudder_max"]

    u_deg[0] = max(min(u_deg[0], ThrottleMax), ThrottleMin)
    u_deg[1] = max(min(u_deg[1], ElevatorMaxDeg), ElevatorMinDeg)
    u_deg[2] = max(min(u_deg[2], AileronMaxDeg), AileronMinDeg)
    u_deg[3] = max(min(u_deg[3], RudderMaxDeg), RudderMinDeg)
    return u_deg
