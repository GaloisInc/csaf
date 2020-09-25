import numpy as np

def p_cntrl(kp, e):
    return pid_cntrl(kp=kp, kd=0, ki=0, e=e, ed=0, ei=0)

def pd_cntrl(kp, kd, e, ed):
    return pid_cntrl(kp=kp, kd=kd, ki=0, e=e, ed=ed, ei=0)

def pi_cntrl(kp, ki, e, ei):
    return pid_cntrl(kp=kp, kd=0, ki=ki, e=e, ed=0, ei=ei)

def pid_cntrl(kp, kd, ki, e, ed, ei):
    return e*kp + ed*kd + ei*ki

def get_ctrl_law(klong, klat):
    # Hard coded LQR gain matrix from BuildLqrControllers.py
    k = np.zeros((3, 8))
    k[:1, :3] = klong
    k[1:, 3:] = klat

    def ctrl_fn(x):
        return np.dot(-k, x)

    return ctrl_fn

def lqr_original():

    xequil = [502.0, 0.03887505597600522, 0.0, 0.0, 0.03887505597600522, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 9.05666543872074]
    uequil = [0.13946204864060271, -0.7495784725828754, 0.0, 0.0]

    # params used to compute, like cost fn, ...?
    # Longitudinal Gains
    K_lqr_long = [-156.8801506723475, -31.037008068526642, -38.72983346216317]

    # Lateral Gains
    K_lqr_lat = [[30.511411060051355, -5.705403676148551, -9.310178739319714, -33.97951344944365, -10.652777306717681],
                 [-22.65901530645282, 1.3193739204719577, -14.2051751789712, 6.7374079391328845, -53.726328142239225]]
    return get_ctrl_law(K_lqr_long, K_lqr_lat), xequil, uequil

def lqr2():
    # vt   alpha            beta roll  pitch          yaw p  q   r  pe pn  h     pow
    xequil = [650, 0.0191038693912106, 0, 0, 0.0191038693912106, 0, 0, 0, 0, 0, 0, 6000, 14.1035663029855]
    uequil = [0.217178415506398, -0.8420383758729, 0, 0]

    K_lqr_long = [[-176.596484550912, -27.096597715439, -38.7298334620741]]
    K_lqr_lat = [
        [30.471599413636401,  -4.948438468207930, -7.188850992173120, -33.836559844552703,   -11.735333174435199],
        [-18.369883691977400,   1.306793804824020, -12.292148077612501, 7.422076386429880, -53.500298646689700]
    ]
    return get_ctrl_law(K_lqr_long, K_lqr_lat), xequil, uequil

def lqr_zero():
    xequil = [502, 0.038875055905, 0, 0, 0.038875055905, 0, 0, 0, 0, 0, 0, 1000, 9.056665434701]
    uequil = [0.139462048578702,-0.749578472735319,0, 0]
    return get_ctrl_law(xequil, uequil, np.zeros(3), np.zeros((2,5)))

def lqr_zero_1():
    xequil = [1000, 0.00402947047, 0, 0, 0.00402947047, 0, 0, 0, 0, 0, 0, 20000.0, 28.23298113313]
    uequil = [0.434754868080227, -0.912291004757661, 0, 0]
    return get_ctrl_law(xequil, uequil, np.zeros(3), np.zeros((2,5)))

#10 controllers that ChouYi built
def lqr_c1():
    xequil = [450, 0.0545511927426912, 0, 0, 0.0545511927426912, 0, 0, 0, 0, 0, 0, 1000, 7.67521188738874]
    uequil = [0.118189280680455, -0.676010961697619, 0, 0]

    K_lqr_long = [[-148.06431997589, -33.9803056762493, -38.7298334620742]]
    K_lqr_lat = [
        [31.168206546133, -6.304852459746, -10.988073099895, -34.053251856399,  -10.046894794492],
        [-24.886039774476, 1.319363485905, -15.704103338210, 6.354214192537, -53.842918800789]
    ]
    return get_ctrl_law(K_lqr_long, K_lqr_lat), xequil, uequil


def lqr_c2():
    xequil = [450, 0.0673579330594429, 0, 0, 0.0673579330594429, 0, 0, 0, 0, 0, 0, 6000, 8.93656959477461]
    uequil = [0.137612713193326, -0.615739177335373, 0, 0]

    K_lqr_long = [[-145.76558491102, -36.939640889784, -38.729833462074]]
    K_lqr_lat = [
        [32.450577872733, -6.953174688222, -12.408493280742, -34.082869711804, -9.792598251767],
        [-26.343158242747, 1.350653216560, -17.256566810970, 6.193382937313, -53.889748742033]
    ]
    return get_ctrl_law(K_lqr_long, K_lqr_lat), xequil, uequil


def lqr_c3():
    xequil = [550, 0.0281497692824177, 0, 0, 0.0281497692824177, 0, 0, 0, 0, 0, 0, 1000, 10.767121342373]
    uequil = [0.165801067791392, -0.79978019399078 , 0, 0]

    K_lqr_long = [[-164.473038128307, -28.7430768821018, -38.7298334620743]]
    K_lqr_lat = [
        [30.038651255251, -5.234441092149, -8.162247596957, -33.916691161832, -11.142044317911],
        [-20.518614600206, 1.299208570547, -13.063619250891, 7.046847567027, -53.626997383946]
    ]
    return get_ctrl_law(K_lqr_long, K_lqr_lat), xequil, uequil


def lqr_c4():
    xequil = [550, 0.0367639610453042, 0, 0, 0.0367639610453042, 0, 0, 0, 0, 0, 0, 6000, 10.6377269746881]
    uequil = [0.163808545960703, -0.759468289408034, 0, 0]

    K_lqr_long = [[-162.154944219558, -31.2583990388346, -38.729833462074]]
    K_lqr_lat = [
        [31.236757139617, -5.803493339820, -9.097305519523, -33.950330713866, -10.882904531781],
        [-22.502028514456, 1.363005962007, -14.338718658119, 6.882953175719, -53.680186185893]
    ]
    return get_ctrl_law(K_lqr_long, K_lqr_lat), xequil, uequil


def lqr_c5():
    xequil = [650, 0.0129261273828184, 0, 0, 0.0129261273828184, 0, 0, 0, 0, 0, 0, 1000, 15.2300523399245]
    uequil = [0.234524982136195, -0.870854418989971, 0, 0]

    K_lqr_long = [[-178.844194508012, -24.9018382449753, -38.72983346207392]]
    K_lqr_lat = [
        [29.440098852061, -4.450616941903, -6.507869808036, -33.798020117617, -12.009770619080],
        [-16.015772205246, 1.231972443870, -11.184935332346, 7.595645866493, -53.439361987931]
    ]
    return get_ctrl_law(K_lqr_long, K_lqr_lat), xequil, uequil


def lqr_c6():
    xequil = [650, 0.0191038693912106, 0, 0, 0.0191038693912106, 0, 0, 0, 0, 0, 0, 6000, 14.1035663029855]
    uequil = [0.217178415506398, -0.8420383758729, 0, 0]

    K_lqr_long = [[-176.596484550912, -27.096597715439, -38.7298334620741]]
    K_lqr_lat = [
        [30.471599413636, -4.948438468208, -7.188850992173, -33.836559844553, -11.735333174435],
        [-18.369883691977, 1.306793804824, -12.292148077613, 7.422076386430, -53.500298646690]
    ]
    return get_ctrl_law(K_lqr_long, K_lqr_lat), xequil, uequil


def lqr_c7():
    xequil = [750, 0.00336619510876377, 0, 0, 0.00336619510876377, 0, 0, 0, 0, 0, 0, 1000, 21.1996899535954]
    uequil = [0.326450415053825, -0.91537732186812, 0, 0]

    K_lqr_long = [[-191.563445978309, -21.946855930687, -38.7298334620745]]
    K_lqr_lat = [
        [29.240241895675, -3.860399841358, -5.422577482942, -33.692037718197, -12.732497240875],
        [-11.613159015554, 1.156769246109, -9.748445770221, 8.052738316595, -53.271789100903
         ]]
    return get_ctrl_law(K_lqr_long, K_lqr_lat), xequil, uequil


def lqr_c8():
    xequil = [750, 0.00800924918337088, 0, 0, 0.00800924918337088, 0, 0, 0, 0, 0, 0, 6000, 20.0326958123403]
    uequil = [0.308480071024643, -0.893764014234333, 0, 0]

    K_lqr_long = [[-189.44834885888, -23.8962496980423, -38.7298334620738]]
    K_lqr_lat = [
        [30.110808235694, -4.299460341217, -5.956605692246, -33.735283734462, -12.442932869241],
        [-14.242641313511, 1.234052394998, -10.740552158321, 7.869601727875, -53.340167056465]
    ]
    return get_ctrl_law(K_lqr_long, K_lqr_lat), xequil, uequil


def lqr_c9():
    xequil = [850, -0.00274233149497158, 0, 0, -0.00274233149497158, 0, 0, 0, 0, 0, 0, 1000, 27.1711079495481]
    uequil = [0.418403263774993, -0.971795737220678, 0, 0]

    K_lqr_long = [[-231.153648165528, -21.1684218106868, -38.7298334620743]]
    K_lqr_lat = [
        [29.278374814182, -3.413347309734, -4.617901300623, -33.600799444273, -13.321624967110],
        [-7.792656198484, 1.085342307537, -8.540219717970, 8.425335406127, -53.127528723211]
    ]
    return get_ctrl_law(K_lqr_long, K_lqr_lat), xequil, uequil

def lqr_c10():
    xequil = [850, 0.000590771859903035, 0, 0, 0.000590771859903035, 0, 0, 0, 0, 0, 0, 6000, 26.2206679625367]
    uequil = [0.403767600285444, -0.928287383000852, 0, 0]

    K_lqr_long = [[-200.974374213513, -21.3487852150918, -38.7298334620743]]
    K_lqr_lat = [
        [30.035788716498, -3.794104426691, -5.092217065976, -33.643286864005, -13.050790108866],
        [-10.238863798048, 1.161450162410, -9.506618461427, 8.254044401763, -53.194707232340]
    ]
    return get_ctrl_law(K_lqr_long, K_lqr_lat), xequil, uequil


class FlightLimits:
    'Flight Limits (for pass-fail conditions)'

    def __init__(self):
        self.altitudeMin = 0 # ft AGL
        self.altitudeMax = 45000 #ft AGL
        self.NzMax = 9 # G's
        self.NzMin = -2 #G's
        self.psMaxAccelDeg = 500 # deg/s/s

        self.vMin = 300 # ft/s
        self.vMax = 2500 # ft/s
        self.alphaMinDeg = -10 # deg
        self.alphaMaxDeg = 45 # deg
        self.betaMinDeg = -500 # add nonreachable value to add min/max structure
        self.betaMaxDeg = 30 # deg

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
        self.ThrottleMax = 1 # Afterburner on for throttle > 0.7
        self.ThrottleMin = 0
        self.ElevatorMaxDeg = 25
        self.ElevatorMinDeg = -25
        self.AileronMaxDeg = 21.5
        self.AileronMinDeg = -21.5
        self.RudderMaxDeg = 30
        self.RudderMinDeg = -30
        self.MaxBankDeg = 60 # For turning maneuvers
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

