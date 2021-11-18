""" Helper functions for subf16 model

Created by Stanley Bak
"""
import numpy as np
from numba import jit  # type: ignore


@jit(nopython=True)
def tfac(alt):
    '''
    Non-linearities: None
    '''
    return 1 - .703e-5 * alt


@jit(nopython=True)
def temp_at(alt):
    '''
    Computes the temp at the given altitude.

    Non-linearities: if-else discontinuity (Saturation)
    '''

    # t = 390 in stratosphere else 3 rankine per atmosphere (3 rankine per 1000 ft)
    # XXX: At 35000, 519 * tfac = 391.30005!! It increases before it suddenly
    # decreases. Maybe a better function should be return to saturate it.
    t = 390 if alt >= 35000 else 519 * tfac(alt)
    return t


@jit(nopython=True)
def temp_at_linear_sym(alt):
    '''
    Replaces temp_at() with a linear function with assertion to check if the
    non-linearity is ever exercised. (Do we really care about cases for >=35000
    for GCAS?)

    Non-linearities: None
    '''
    assert alt < 35000
    t = 519 * tfac(alt)
    return t


@jit(nopython=True)
def amach(vt, alt):
    """converts velocity (vt) and altitude (alt) to mach number (amach) and dynamic pressure (qbar)

    See pages 63-65 of Stevens & Lewis, "Aircraft Control and Simulation", 2nd edition
    [adc() was split into amach and qbar]

    Non-linearities: sqrt, bilinear term vt/alt
    """
    # vt = freestream air speed
    t = temp_at(alt)

    # a = speed of sound at the ambient conditions
    # speed of sound in a fluid is the sqrt of the quotient of the modulus of elasticity over the mass density
    a = np.sqrt(1.4 * 1716.3 * t)

    # amach = mach number
    amach = vt / a
    return amach


@jit(nopython=True)
def qbar(vt, alt):
    '''
    Computes the pressure.

    Non-linearities: alt * vt^2
    '''
    ro = 2.377e-3
    # rho = freestream mass density
    rho = ro * tfac(alt) ** 4.14
    # qbar = dynamic pressure
    return .5 * rho * vt * vt


@jit(nopython=True)
def tgear(thtl):
    '''
    Non-linearities: slightly discontinuous w.r.t. throttle and piecewise-linear
    '''

    if thtl <= .77:
        tg = 64.94 * thtl
    else:
        tg = 217.38 * thtl - 117.38

    return tg


@jit(nopython=True)
def rtau_linear_sym(dp):
    '''
    Non-linearities: None
    '''

    rt = 1.9 - .036 * dp

    assert dp > 25 and dp < 50

    return rt


@jit(nopython=True)
def rtau(dp):
    '''
    Non-linearities: discontinuous w.r.t. dp and saturation at both ends
    '''

    if dp <= 25:
        rt = 1.0
    elif dp >= 50:
        rt = .1
    else:
        rt = 1.9 - .036 * dp

    return rt


# XXX: nonlinear
@jit(nopython=True)
def pdot_sym(p3, p1):
    '''
    Equivalent re-write of pdot
    Non-linearities: discontinuous w.r.t. p1 and p3
    '''

    if p1 >= 50 and p3 >= 50:
        p2 = p1
    elif p1 >= 50 and p3 < 50:
        p2 = 60
    elif p1 < 50 and p3 >= 50:
        p2 = 40

    t = 5 if p3 >= 50 else rtau_linear_sym(p2 - p3)

    pd = t * (p2 - p3)

    return pd


@jit(nopython=True)
def pdot(p3, p1):
    '''
    Non-linearities: discontinuous w.r.t. p1 and p3
    '''

    if p1 >= 50:
        if p3 >= 50:
            t = 5
            p2 = p1
        else:
            p2 = 60
            t = rtau(p2 - p3)
    else:
        if p3 >= 50:
            t = 5
            p2 = 40
        else:
            p2 = p1
            t = rtau(p2 - p3)

    pd = t * (p2 - p3)

    return pd


@jit(nopython=True)
def thrust_lookup(power, alt, rmach):
    '''
    thrust lookup-table version

    Non-linearities: lookup table, rounding using fix(), ...
    '''

    # Idle
    thrust_a_table = np.array([
        [1060, 670, 880, 1140, 1500, 1860],
        [635, 425, 690, 1010, 1330, 1700],
        [60, 25, 345, 755, 1130, 1525],
        [-1020, -170, -300, 350, 910, 1360],
        [-2700, -1900, -1300, -247, 600, 1100],
        [-3600, -1400, -595, -342, -200, 700]]).T
    # Military
    thrust_b_table = np.array([
        [12680, 9150, 6200, 3950, 2450, 1400],
        [12680, 9150, 6313, 4040, 2470, 1400],
        [12610, 9312, 6610, 4290, 2600, 1560],
        [12640, 9839, 7090, 4660, 2840, 1660],
        [12390, 10176, 7750, 5320, 3250, 1930],
        [11680, 9848, 8050, 6100, 3800, 2310]]).T
    # Maximum
    thrust_c_table = np.array([
        [20000, 15000, 10800, 7000, 4000, 2500],
        [21420, 15700, 11225, 7323, 4435, 2600],
        [22700, 16860, 12250, 8154, 5000, 2835],
        [24240, 18910, 13760, 9285, 5700, 3215],
        [26070, 21075, 15975, 11115, 6860, 3950],
        [28886, 23319, 18300, 13484, 8642, 5057]]).T

    if alt < 0: alt = 0.01  # uh, why not 0?

    h = .0001 * alt

    i = fix(h)

    if i >= 5: i = 4

    dh = h - i
    rm = 5 * rmach
    m = fix(rm)

    if m >= 5: m = 4
    if m <= 0: m = 0

    dm = rm - m
    cdh = 1 - dh

    def s_and_t(thrust_table):
        s = thrust_table[i, m] * cdh + thrust_table[i + 1, m] * dh
        t = thrust_table[i, m + 1] * cdh + thrust_table[i + 1, m + 1] * dh
        return s, t

    s, t = s_and_t(thrust_b_table)
    tmil = s + (t - s) * dm

    if power < 50:
        s, t = s_and_t(thrust_a_table)
        tidl = s + (t - s) * dm
        thrst = tidl + (tmil - tidl) * power * .02
    else:
        s, t = s_and_t(thrust_c_table)
        tmax = s + (t - s) * dm
        thrst = tmil + (tmax - tmil) * (power - 50) * .02

    return thrst


@jit(nopython=True)
def dampp_lookup(alpha):
    '''
    dampp lookup-table

    Non-linearities: lookup table, rounding using fix(), ...
    '''
    dampp_table = np.array([[-.267, -.110, .308, 1.34, 2.08, 2.91, 2.76, 2.05, 1.50, 1.49, 1.83, 1.21], \
                            [.882, .852, .876, .958, .962, .974, .819, .483, .590, 1.21, -.493, -1.04], \
                            [-.108, -.108, -.188, .110, .258, .226, .344, .362, .611, .529, .298, -2.27], \
                            [-8.80, -25.8, -28.9, -31.4, -31.2, -30.7, -27.7, -28.2, -29.0, -29.8, -38.3, -35.3], \
                            [-.126, -.026, .063, .113, .208, .230, .319, .437, .680, .100, .447, -.330], \
                            [-.360, -.359, -.443, -.420, -.383, -.375, -.329, -.294, -.230, -.210, -.120, -.100], \
                            [-7.21, -.540, -5.23, -5.26, -6.11, -6.64, -5.69, -6.00, -6.20, -6.40, -6.60, -6.00], \
                            [-.380, -.363, -.378, -.386, -.370, -.453, -.550, -.582, -.595, -.637, -1.02, -.840], \
                            [.061, .052, .052, -.012, -.013, -.024, .050, .150, .130, .158, .240, .150]]).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2: k = -1
    if k >= 9: k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    k += 3;
    l += 3  # XXX: Why this increment?

    d = np.zeros(9)

    # offset for 0-based indexing
    k -= 1;
    l -= 1
    d = [dampp_table[k, i] + abs(da) * (dampp_table[l, i] - dampp_table[k, i]) for i in range(9)]

    return np.array(d)


@jit(nopython=True)
def engine(thtlc, power, vt, alt):
    '''
    Non-linearities: Calls non-linear functions
    '''
    # XXX: amach computation uses saturation
    amach_ = amach(vt, alt)

    # XXX: Piecewise linear
    cpow = tgear(thtlc)
    # XXX:
    power_dot = pdot(power, cpow)

    thrust = thrust_lookup(power, alt, amach_)

    return power_dot, thrust


@jit(nopython=True)
def fix(x):
    """round towards zero"""
    return int(np.floor(x) if x >= 0 else np.ceil(x))
    # return int(np.fix(x))


#     assert isinstance(x, float)
# return int(floor(x)) if x > 0 else int(ceil(x))


@jit(nopython=True)
def sign(ele):
    """sign of a number"""

    if ele < 0:
        rv = -1
    elif ele == 0:
        rv = 0
    else:
        rv = 1

    return rv
