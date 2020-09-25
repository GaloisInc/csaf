"""low level flight controller

gain matrices and equilibrium points are computed by running BuildLqrControllers.py
"""

# External
#import cmdlineparser
import numpy as np
import collections

LinearizedFeedbackControlLaw = collections.namedtuple('LinearizedFeedbackControlLaw', 'xequil, uequil, ctrl_fn')


def get_ctrl_law(xequil, uequil, klong, klat):
    # Hard coded LQR gain matrix from BuildLqrControllers.py
    k = np.zeros((3, 8))
    k[:1, :3] = klong
    k[1:, 3:] = klat

    def ctrl_fn(x):
        return np.dot(-k, x)

    return LinearizedFeedbackControlLaw(np.array(xequil), np.array(uequil), ctrl_fn)

def lqr_original():

    xequil = [502.0, 0.03887505597600522, 0.0, 0.0, 0.03887505597600522, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 9.05666543872074]
    uequil = [0.13946204864060271, -0.7495784725828754, 0.0, 0.0]

    # params used to compute, like cost fn, ...?
    # Longitudinal Gains
    K_lqr_long = [-156.8801506723475, -31.037008068526642, -38.72983346216317]

    # Lateral Gains
    K_lqr_lat = [[30.511411060051355, -5.705403676148551, -9.310178739319714, -33.97951344944365, -10.652777306717681],
                 [-22.65901530645282, 1.3193739204719577, -14.2051751789712, 6.7374079391328845, -53.726328142239225]]
    return get_ctrl_law(xequil, uequil, K_lqr_long, K_lqr_lat)

def lqr_original_faulty_pitch():
    '''
    lqr_original controller with modified gains K_long.
    The control gain for alpha K_lqr_long[0] is modified
    '''

    xequil = [502.0, 0.03887505597600522, 0.0, 0.0, 0.03887505597600522, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 9.05666543872074]
    uequil = [0.13946204864060271, -0.7495784725828754, 0.0, 0.0]

    # params used to compute, like cost fn, ...?
    # Longitudinal Gains
    K_lqr_long = [-25, -10, -38]

    # Lateral Gains
    K_lqr_lat = [[30.511411060051355, -5.705403676148551, -9.310178739319714, -33.97951344944365, -10.652777306717681],
                 [-22.65901530645282, 1.3193739204719577, -14.2051751789712, 6.7374079391328845, -53.726328142239225]]
    return get_ctrl_law(xequil, uequil, K_lqr_long, K_lqr_lat)

def lqr_original_faulty_roll():

    xequil = [502.0, 0.03887505597600522, 0.0, 0.0, 0.03887505597600522, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 9.05666543872074]
    uequil = [0.13946204864060271, -0.7495784725828754, 0.0, 0.0]

    # params used to compute, like cost fn, ...?
    # Longitudinal Gains
    K_lqr_long = [-156.8801506723475, -31.037008068526642, -38.72983346216317]

#     LQR_lat Gains =
#                           beta            p            r     int_e_ps   int_e_Ny_r
#           aileron
#            rudder

    # Lateral Gains
    K_lqr_lat = [[30, -5.7, -9.3, -0.9, -10.6],
                 [-22, +1.3, -14.2, 6.7, -53.7]]
    return get_ctrl_law(xequil, uequil, K_lqr_long, np.array(K_lqr_lat))

# def lqr_original_faulty_pitch():
#     '''
#     lqr_original controller with modified gains K_long.
#     The control gain for alpha K_lqr_long[0] is modified
#     '''

#     xequil = [502.0, 0.03887505597600522, 0.0, 0.0, 0.03887505597600522, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0, 9.05666543872074]
#     uequil = [0.13946204864060271, -0.7495784725828754, 0.0, 0.0]

#     # params used to compute, like cost fn, ...?
#     # Longitudinal Gains
#     K_lqr_long = [-25, -10, -38]

#     # Lateral Gains
#     K_lqr_lat = [[30.511411060051355, -5.705403676148551, -9.310178739319714, -33.97951344944365, -10.652777306717681],
#                  [-22.65901530645282, 1.3193739204719577, -14.2051751789712, 6.7374079391328845, -53.726328142239225]]
#     return get_ctrl_law(xequil, uequil, K_lqr_long, K_lqr_lat)

def lqr2():
             # vt   alpha            beta roll  pitch          yaw p  q   r  pe pn  h     pow
    xequil = [650, 0.0191038693912106, 0, 0, 0.0191038693912106, 0, 0, 0, 0, 0, 0, 6000, 14.1035663029855]
    uequil = [0.217178415506398, -0.8420383758729, 0, 0]

    K_lqr_long = [[-176.596484550912, -27.096597715439, -38.7298334620741]]
    K_lqr_lat = [
        [30.471599413636401,  -4.948438468207930, -7.188850992173120, -33.836559844552703,   -11.735333174435199],
        [-18.369883691977400,   1.306793804824020, -12.292148077612501, 7.422076386429880, -53.500298646689700]
        ]
    return get_ctrl_law(xequil, uequil, K_lqr_long, K_lqr_lat)

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
    return get_ctrl_law(xequil, uequil, K_lqr_long, K_lqr_lat)


def lqr_c2():
    xequil = [450, 0.0673579330594429, 0, 0, 0.0673579330594429, 0, 0, 0, 0, 0, 0, 6000, 8.93656959477461]
    uequil = [0.137612713193326, -0.615739177335373, 0, 0]

    K_lqr_long = [[-145.76558491102, -36.939640889784, -38.729833462074]]
    K_lqr_lat = [
        [32.450577872733, -6.953174688222, -12.408493280742, -34.082869711804, -9.792598251767],
        [-26.343158242747, 1.350653216560, -17.256566810970, 6.193382937313, -53.889748742033]
        ]
    return get_ctrl_law(xequil, uequil, K_lqr_long, K_lqr_lat)


def lqr_c3():
    xequil = [550, 0.0281497692824177, 0, 0, 0.0281497692824177, 0, 0, 0, 0, 0, 0, 1000, 10.767121342373]
    uequil = [0.165801067791392, -0.79978019399078 , 0, 0]

    K_lqr_long = [[-164.473038128307, -28.7430768821018, -38.7298334620743]]
    K_lqr_lat = [
        [30.038651255251, -5.234441092149, -8.162247596957, -33.916691161832, -11.142044317911],
        [-20.518614600206, 1.299208570547, -13.063619250891, 7.046847567027, -53.626997383946]
        ]
    return get_ctrl_law(xequil, uequil, K_lqr_long, K_lqr_lat)


def lqr_c4():
    xequil = [550, 0.0367639610453042, 0, 0, 0.0367639610453042, 0, 0, 0, 0, 0, 0, 6000, 10.6377269746881]
    uequil = [0.163808545960703, -0.759468289408034, 0, 0]

    K_lqr_long = [[-162.154944219558, -31.2583990388346, -38.729833462074]]
    K_lqr_lat = [
        [31.236757139617, -5.803493339820, -9.097305519523, -33.950330713866, -10.882904531781],
        [-22.502028514456, 1.363005962007, -14.338718658119, 6.882953175719, -53.680186185893]
        ]
    return get_ctrl_law(xequil, uequil, K_lqr_long, K_lqr_lat)


def lqr_c5():
    xequil = [650, 0.0129261273828184, 0, 0, 0.0129261273828184, 0, 0, 0, 0, 0, 0, 1000, 15.2300523399245]
    uequil = [0.234524982136195, -0.870854418989971, 0, 0]

    K_lqr_long = [[-178.844194508012, -24.9018382449753, -38.72983346207392]]
    K_lqr_lat = [
        [29.440098852061, -4.450616941903, -6.507869808036, -33.798020117617, -12.009770619080],
        [-16.015772205246, 1.231972443870, -11.184935332346, 7.595645866493, -53.439361987931]
        ]
    return get_ctrl_law(xequil, uequil, K_lqr_long, K_lqr_lat)


def lqr_c6():
    xequil = [650, 0.0191038693912106, 0, 0, 0.0191038693912106, 0, 0, 0, 0, 0, 0, 6000, 14.1035663029855]
    uequil = [0.217178415506398, -0.8420383758729, 0, 0]

    K_lqr_long = [[-176.596484550912, -27.096597715439, -38.7298334620741]]
    K_lqr_lat = [
        [30.471599413636, -4.948438468208, -7.188850992173, -33.836559844553, -11.735333174435],
        [-18.369883691977, 1.306793804824, -12.292148077613, 7.422076386430, -53.500298646690]
        ]
    return get_ctrl_law(xequil, uequil, K_lqr_long, K_lqr_lat)


def lqr_c7():
    xequil = [750, 0.00336619510876377, 0, 0, 0.00336619510876377, 0, 0, 0, 0, 0, 0, 1000, 21.1996899535954]
    uequil = [0.326450415053825, -0.91537732186812, 0, 0]

    K_lqr_long = [[-191.563445978309, -21.946855930687, -38.7298334620745]]
    K_lqr_lat = [
        [29.240241895675, -3.860399841358, -5.422577482942, -33.692037718197, -12.732497240875],
        [-11.613159015554, 1.156769246109, -9.748445770221, 8.052738316595, -53.271789100903
            ]]
    return get_ctrl_law(xequil, uequil, K_lqr_long, K_lqr_lat)


def lqr_c8():
    xequil = [750, 0.00800924918337088, 0, 0, 0.00800924918337088, 0, 0, 0, 0, 0, 0, 6000, 20.0326958123403]
    uequil = [0.308480071024643, -0.893764014234333, 0, 0]

    K_lqr_long = [[-189.44834885888, -23.8962496980423, -38.7298334620738]]
    K_lqr_lat = [
        [30.110808235694, -4.299460341217, -5.956605692246, -33.735283734462, -12.442932869241],
        [-14.242641313511, 1.234052394998, -10.740552158321, 7.869601727875, -53.340167056465]
        ]
    return get_ctrl_law(xequil, uequil, K_lqr_long, K_lqr_lat)


def lqr_c9():
    xequil = [850, -0.00274233149497158, 0, 0, -0.00274233149497158, 0, 0, 0, 0, 0, 0, 1000, 27.1711079495481]
    uequil = [0.418403263774993, -0.971795737220678, 0, 0]

    K_lqr_long = [[-231.153648165528, -21.1684218106868, -38.7298334620743]]
    K_lqr_lat = [
        [29.278374814182, -3.413347309734, -4.617901300623, -33.600799444273, -13.321624967110],
        [-7.792656198484, 1.085342307537, -8.540219717970, 8.425335406127, -53.127528723211]
        ]
    return get_ctrl_law(xequil, uequil, K_lqr_long, K_lqr_lat)

def lqr_c10():
    xequil = [850, 0.000590771859903035, 0, 0, 0.000590771859903035, 0, 0, 0, 0, 0, 0, 6000, 26.2206679625367]
    uequil = [0.403767600285444, -0.928287383000852, 0, 0]

    K_lqr_long = [[-200.974374213513, -21.3487852150918, -38.7298334620743]]
    K_lqr_lat = [
        [30.035788716498, -3.794104426691, -5.092217065976, -33.643286864005, -13.050790108866],
        [-10.238863798048, 1.161450162410, -9.506618461427, 8.254044401763, -53.194707232340]
        ]
    return get_ctrl_law(xequil, uequil, K_lqr_long, K_lqr_lat)

def lqr_c11():
    '''
    altitude = 30,000 ft
    airspeed = 500 ft/s

    State Equilibrium (ft, rads, etc.) =
                        Vt        alpha         beta          phi        theta
                 500.00000      0.14144            0            0      0.14144

                       psi            p            q            r           pn
                         0            0            0            0            0

                        pe          alt          pow
                         0  30000.00000     22.31226


    Control Equilibrium (% & degs) =
                  throttle     elevator      aileron       rudder
                   0.34358     -0.58337            0            0

    LQR_lat Gains =
                          beta            p            r     int_e_ps   int_e_Ny_r
          aileron     39.63904    -10.32529    -23.65289    -34.10563     -9.59249
          rudder    -27.93590      1.31604    -27.36404      6.06682    -53.92573

    LQR_long Gains =
                     alpha            q     int_e_Nz
     elevator   -130.68208    -49.32303    -38.72983
    '''

    xequil = [500.00000, 0.14144, 0, 0, 0.14144, 0, 0, 0, 0, 0, 0, 30000, 22.31226]
    uequil = [0.34358, -0.58337, 0, 0]

#     K_lqr_long = [[-130.68208,    -49.32303,    -38.72983]]
#     K_lqr_lat = [
#         [39.63904,    -10.32529,    -23.65289,    -34.10563,     -9.59249],
#         [-27.93590,      1.31604,    -27.36404,      6.06682,    -53.92573]
#         ]

#     K_lqr_long = np.array([[-26.41456983, -21.68413662,  -3.87298335]])

#     K_lqr_lat = np.array([[ 3.45295062, -1.76937846, -9.3021737 , -3.46399737, -0.04249096],
#            [-4.18487116, -0.29615515, -3.23298322,  0.02687365, -5.47706076]])

    K_lqr_long = np.array([[-61.07338446, -34.55754756, -12.24744871]])

    K_lqr_lat = np.array([[ 15.97671797,  -4.29832586, -18.64257951, -10.93219972, -1.10341436],
	   [-11.24290724,  -0.26901378, -10.03659023,   0.69786051, -17.28532548]])

    return get_ctrl_law(xequil, uequil, K_lqr_long, K_lqr_lat)


def lqr_c12():
    '''
    altitude = 35,000 ft
    airspeed = 500 ft/s

    State Equilibrium =
                            Vt        alpha         beta          phi        theta
                     500.00000      0.17518            0            0      0.17518

                           psi            p            q            r           pn
                             0            0            0            0            0

                            pe          alt          pow
                             0  35000.00000     33.79263


    Control Equilibrium =
                      throttle     elevator      aileron       rudder
                       0.52037     -0.60779            0            0

    LQR_lat Gains =
                          beta            p            r     int_e_ps   int_e_Ny_r
          aileron     41.87692    -11.39725    -29.83327    -34.14429     -9.24223
           rudder    -26.89695      1.13250    -31.96833      5.84530    -53.98686

    LQR_long Gains =
                         alpha            q     int_e_Nz
         elevator   -158.10294    -60.86832    -38.72983

    '''

    xequil = [500.00000, 0.17518, 0, 0, 0.17518, 0, 0, 0, 0, 0, 0, 35000.00000, 33.79263]
    uequil = [0.52037, -0.60779, 0, 0]

    K_lqr_long = [[-158.10294, -60.86832, -38.72983]]
    K_lqr_lat = [
        [41.87692, -11.39725, -29.83327, -34.14429, -9.24223],
        [-26.89695, 1.13250, -31.96833, 5.84530, -53.98686]
        ]
    return get_ctrl_law(xequil, uequil, K_lqr_long, K_lqr_lat)

def lqr_c13_tuned():
    '''
    altitude = 50,000 ft
    airspeed = 500 ft/s
    '''

    xequil = np.array([500.0, 0.33295724742024696, 0.0, 0.0, 0.33295724742024696, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 50000.0, 165.0919622882589])
    uequil = np.array([1.2994385973330522, 0.7725907437135913, 0.0, 0.0])

    #python -m f16.control_design.BuildLqrControllers --altg=50000 --Vtg=500 --q_Nz=10 --q_alpha=150000
    K_lqr_long = np.array([[-360.09781019, -152.47562576,   -3.16227763]])
    #python -m f16.control_design.BuildLqrControllers --altg=50000 --Vtg=500 --q_Nz=10 --q_alpha=1500
    #K_lqr_long = np.array([[-36.13890104, -44.29709601,  -3.16227766]])

    K_lqr_lat = np.array([[ 55.81609195, -13.61736782, -49.25141521, -34.56563093,
              3.61149494],
           [  6.26175494,  -0.67336214, -81.89043418,  -2.28410996,
            -54.65306125]])

    return get_ctrl_law(xequil, uequil, K_lqr_long, K_lqr_lat)


def get_lqr(lqr_id):
    assert lqr_id in _lqr_controllers.keys()
    lqr = _lqr_controllers[lqr_id]
    return lqr

_lqr_controllers = {
        'lqr_original': lqr_original(),
        'lqr_original_faulty_pitch': lqr_original_faulty_pitch(),
        'lqr_original_faulty_roll': lqr_original_faulty_roll(),
        'lqr2': lqr2(),
        'lqr_zero': lqr_zero(),
        'lqr_zero_1': lqr_zero_1(),
        'lqr_c1': lqr_c1(),
        'lqr_c2': lqr_c2(),
        'lqr_c3': lqr_c3(),
        'lqr_c4': lqr_c4(),
        'lqr_c5': lqr_c5(),
        'lqr_c6': lqr_c6(),
        'lqr_c7': lqr_c7(),
        'lqr_c8': lqr_c8(),
        'lqr_c9': lqr_c9(),
        'lqr_c10': lqr_c10(),
        'lqr_c11': lqr_c11(),
        'lqr_c12': lqr_c12(),
        'lqr_c13_tuned': lqr_c13_tuned(),
        }

#parser = cmdlineparser.add_parser('LQR')
#parser.add_argument('--lqr-id', default=None, help='LQR controller', choices=(_lqr_controllers.keys()))
