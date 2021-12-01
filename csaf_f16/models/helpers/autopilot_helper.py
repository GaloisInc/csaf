"""
CSAF F-16 Model

taken from https://github.com/stanleybak/AeroBenchVVPython

Autopilot helper functions
"""


def p_cntrl(kp, e):
    return pid_cntrl(kp=kp, kd=0, ki=0, e=e, ed=0, ei=0)


def pd_cntrl(kp, kd, e, ed):
    return pid_cntrl(kp=kp, kd=kd, ki=0, e=e, ed=ed, ei=0)


def pi_cntrl(kp, ki, e, ei):
    return pid_cntrl(kp=kp, kd=0, ki=ki, e=e, ed=0, ei=ei)


def pid_cntrl(kp, kd, ki, e, ed, ei):
    return e * kp + ed * kd + ei * ki


class FlightLimits:
    'Flight Limits (for pass-fail conditions)'

    def __init__(self):
        self.altitudeMin = 0  # ft AGL
        self.altitudeMax = 45000  # ft AGL
        self.NzMax = 9  # G's
        self.NzMin = -2  # G's
        self.psMaxAccelDeg = 500  # deg/s/s

        self.vMin = 300  # ft/s
        self.vMax = 2500  # ft/s
        self.alphaMinDeg = -10  # deg
        self.alphaMaxDeg = 45  # deg
        self.betaMinDeg = -500  # add nonreachable value to add min/max structure
        self.betaMaxDeg = 30  # deg

        self.check()

    def check(self):
        'check that flight limits are within model bounds'

        flightLimits = self

        assert not (flightLimits.vMin < 300 or flightLimits.vMax > 2500), \
            'flightLimits: Airspeed limits outside model limits (300 to 2500)'

        assert not (flightLimits.alphaMinDeg < -10 or flightLimits.alphaMaxDeg > 45), \
            'flightLimits: Alpha limits outside model limits (-10 to 45)'

        assert not (abs(flightLimits.betaMaxDeg) > 30), 'flightLimits: Beta limit outside model limits (30 deg)'


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
        self.NzMax = 6
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
