""" Helper functions for subf16 model

Created by Stanley Bak
"""
import numpy as np
from numpy import sqrt, floor, ceil

default_aircraft = {
    "s": 300,
    "b": 30,
    "cbar": 11.32,
    "rm": 1.57e-3,
    "xcgr": 0.35,
    "xcg": 0.35,
    "he": 160.0,
    "c1": -.770,
    "c2": 0.02755,
    "c3": 1.055e-4,
    "c4": 1.642e-6,
    "c5": .9604,
    "c6": 1.759e-2,
    "c7": 1.792e-5,
    "c8": -.7336,
    "c9": 1.587e-5,
    "rtod": 57.29578,
    "g": 32.17,
    "equations": "morelli"
}


def adc(vt, alt):
    """converts velocity (vt) and altitude (alt) to mach number (amach) and dynamic pressure (qbar)

    See pages 63-65 of Stevens & Lewis, "Aircraft Control and Simulation", 2nd edition
    """

    # vt = freestream air speed

    ro = 2.377e-3
    tfac = 1 - .703e-5 * alt

    if alt >= 35000:  # in stratosphere
        t = 390
    else:
        t = 519 * tfac  # 3 rankine per atmosphere (3 rankine per 1000 ft)

    # rho = freestream mass density
    rho = ro * tfac ** 4.14

    # a = speed of sound at the ambient conditions
    # speed of sound in a fluid is the sqrt of the quotient of the modulus of elasticity over the mass density
    a = sqrt(1.4 * 1716.3 * t)

    # amach = mach number
    amach = vt / a

    # qbar = dynamic pressure
    qbar = .5 * rho * vt * vt

    return amach, qbar


def tgear(thtl):
    """tgear function"""

    if thtl <= .77:
        tg = 64.94 * thtl
    else:
        tg = 217.38 * thtl - 117.38

    return tg


def rtau(dp):
    """rtau function"""

    if dp <= 25:
        rt = 1.0
    elif dp >= 50:
        rt = .1
    else:
        rt = 1.9 - .036 * dp

    return rt


def pdot(p3, p1):
    """pdot function"""

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


def thrust(power, alt, rmach):
    """thrust lookup-table version"""

    a = np.array([[1060, 670, 880, 1140, 1500, 1860], \
                  [635, 425, 690, 1010, 1330, 1700], \
                  [60, 25, 345, 755, 1130, 1525], \
                  [-1020, -170, -300, 350, 910, 1360], \
                  [-2700, -1900, -1300, -247, 600, 1100], \
                  [-3600, -1400, -595, -342, -200, 700]], dtype=float).T

    b = np.array([[12680, 9150, 6200, 3950, 2450, 1400], \
                  [12680, 9150, 6313, 4040, 2470, 1400], \
                  [12610, 9312, 6610, 4290, 2600, 1560], \
                  [12640, 9839, 7090, 4660, 2840, 1660], \
                  [12390, 10176, 7750, 5320, 3250, 1930], \
                  [11680, 9848, 8050, 6100, 3800, 2310]], dtype=float).T

    c = np.array([[20000, 15000, 10800, 7000, 4000, 2500], \
                  [21420, 15700, 11225, 7323, 4435, 2600], \
                  [22700, 16860, 12250, 8154, 5000, 2835], \
                  [24240, 18910, 13760, 9285, 5700, 3215], \
                  [26070, 21075, 15975, 11115, 6860, 3950], \
                  [28886, 23319, 18300, 13484, 8642, 5057]], dtype=float).T

    if alt < 0:
        alt = 0.01  # uh, why not 0?

    h = .0001 * alt

    i = fix(h)

    if i >= 5:
        i = 4

    dh = h - i
    rm = 5 * rmach
    m = fix(rm)

    if m >= 5:
        m = 4
    elif m <= 0:
        m = 0

    dm = rm - m
    cdh = 1 - dh

    # do not increment these, since python is 0-indexed while matlab is 1-indexed
    # i = i + 1
    # m = m + 1

    s = b[i, m] * cdh + b[i + 1, m] * dh
    t = b[i, m + 1] * cdh + b[i + 1, m + 1] * dh
    tmil = s + (t - s) * dm

    if power < 50:
        s = a[i, m] * cdh + a[i + 1, m] * dh
        t = a[i, m + 1] * cdh + a[i + 1, m + 1] * dh
        tidl = s + (t - s) * dm
        thrst = tidl + (tmil - tidl) * power * .02
    else:
        s = c[i, m] * cdh + c[i + 1, m] * dh
        t = c[i, m + 1] * cdh + c[i + 1, m + 1] * dh
        tmax = s + (t - s) * dm
        thrst = tmil + (tmax - tmil) * (power - 50) * .02

    return thrst


def cx(alpha, el):
    """cx definition"""

    a = np.array([[-.099, -.081, -.081, -.063, -.025, .044, .097, .113, .145, .167, .174, .166], \
                  [-.048, -.038, -.040, -.021, .016, .083, .127, .137, .162, .177, .179, .167], \
                  [-.022, -.020, -.021, -.004, .032, .094, .128, .130, .154, .161, .155, .138], \
                  [-.040, -.038, -.039, -.025, .006, .062, .087, .085, .100, .110, .104, .091], \
                  [-.083, -.073, -.076, -.072, -.046, .012, .024, .025, .043, .053, .047, .040]], dtype=float).T

    s = .2 * alpha
    k = fix(s)
    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = el / 12
    m = fix(s)
    if m <= -2:
        m = -1

    if m >= 2:
        m = 1

    de = s - m
    n = m + fix(1.1 * sign(de))
    k = k + 3
    l = l + 3
    m = m + 3
    n = n + 3
    t = a[k - 1, m - 1]
    u = a[k - 1, n - 1]
    v = t + abs(da) * (a[l - 1, m - 1] - t)
    w = u + abs(da) * (a[l - 1, n - 1] - u)
    cxx = v + (w - v) * abs(de)

    return cxx


def cy(beta, ail, rdr):
    """cy function"""

    return -.02 * beta + .021 * (ail / 20) + .086 * (rdr / 30)


def cz(alpha, beta, el):
    """cz function"""

    a = np.array([.770, .241, -.100, -.415, -.731, -1.053, -1.355, -1.646, -1.917, -2.120, -2.248, -2.229], \
                 dtype=float).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    l = l + 3
    k = k + 3
    s = a[k - 1] + abs(da) * (a[l - 1] - a[k - 1])

    return s * (1 - (beta / 57.3) ** 2) - .19 * (el / 25)


def cl(alpha, beta):
    """For calculating rolling moment coefficient"""

    a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                  [-.001, -.004, -.008, -.012, -.016, -.022, -.022, -.021, -.015, -.008, -.013, -.015], \
                  [-.003, -.009, -.017, -.024, -.030, -.041, -.045, -.040, -.016, -.002, -.010, -.019], \
                  [-.001, -.010, -.020, -.030, -.039, -.054, -.057, -.054, -.023, -.006, -.014, -.027], \
                  [.000, -.010, -.022, -.034, -.047, -.060, -.069, -.067, -.033, -.036, -.035, -.035], \
                  [.007, -.010, -.023, -.034, -.049, -.063, -.081, -.079, -.060, -.058, -.062, -.059], \
                  [.009, -.011, -.023, -.037, -.050, -.068, -.089, -.088, -.091, -.076, -.077, -.076]], dtype=float).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = .2 * abs(beta)
    m = fix(s)
    if m == 0:
        m = 1

    if m >= 6:
        m = 5

    db = s - m
    n = m + fix(1.1 * sign(db))
    l = l + 3
    k = k + 3
    m = m + 1
    n = n + 1
    t = a[k - 1, m - 1]
    u = a[k - 1, n - 1]
    v = t + abs(da) * (a[l - 1, m - 1] - t)
    w = u + abs(da) * (a[l - 1, n - 1] - u)
    dum = v + (w - v) * abs(db)

    return dum * sign(beta)


def cm(alpha, el):
    """cm function"""

    a = np.array([[.205, .168, .186, .196, .213, .251, .245, .238, .252, .231, .198, .192], \
                  [.081, .077, .107, .110, .110, .141, .127, .119, .133, .108, .081, .093], \
                  [-.046, -.020, -.009, -.005, -.006, .010, .006, -.001, .014, .000, -.013, .032], \
                  [-.174, -.145, -.121, -.127, -.129, -.102, -.097, -.113, -.087, -.084, -.069, -.006], \
                  [-.259, -.202, -.184, -.193, -.199, -.150, -.160, -.167, -.104, -.076, -.041, -.005]], dtype=float).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = el / 12
    m = fix(s)

    if m <= -2:
        m = -1

    if m >= 2:
        m = 1

    de = s - m
    n = m + fix(1.1 * sign(de))
    k = k + 3
    l = l + 3
    m = m + 3
    n = n + 3
    t = a[k - 1, m - 1]
    u = a[k - 1, n - 1]
    v = t + abs(da) * (a[l - 1, m - 1] - t)
    w = u + abs(da) * (a[l - 1, n - 1] - u)

    return v + (w - v) * abs(de)


def cn(alpha, beta):
    """cn function"""

    a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                  [.018, .019, .018, .019, .019, .018, .013, .007, .004, -.014, -.017, -.033], \
                  [.038, .042, .042, .042, .043, .039, .030, .017, .004, -.035, -.047, -.057], \
                  [.056, .057, .059, .058, .058, .053, .032, .012, .002, -.046, -.071, -.073], \
                  [.064, .077, .076, .074, .073, .057, .029, .007, .012, -.034, -.065, -.041], \
                  [.074, .086, .093, .089, .080, .062, .049, .022, .028, -.012, -.002, -.013], \
                  [.079, .090, .106, .106, .096, .080, .068, .030, .064, .015, .011, -.001]], dtype=float).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = .2 * abs(beta)
    m = fix(s)

    if m == 0:
        m = 1

    if m >= 6:
        m = 5

    db = s - m
    n = m + fix(1.1 * sign(db))
    l = l + 3
    k = k + 3
    m = m + 1
    n = n + 1
    t = a[k - 1, m - 1]
    u = a[k - 1, n - 1]

    v = t + abs(da) * (a[l - 1, m - 1] - t)
    w = u + abs(da) * (a[l - 1, n - 1] - u)
    dum = v + (w - v) * abs(db)

    return dum * sign(beta)


def dlda(alpha, beta):
    """dlda function"""

    a = np.array([[-.041, -.052, -.053, -.056, -.050, -.056, -.082, -.059, -.042, -.038, -.027, -.017], \
                  [-.041, -.053, -.053, -.053, -.050, -.051, -.066, -.043, -.038, -.027, -.023, -.016], \
                  [-.042, -.053, -.052, -.051, -.049, -.049, -.043, -.035, -.026, -.016, -.018, -.014], \
                  [-.040, -.052, -.051, -.052, -.048, -.048, -.042, -.037, -.031, -.026, -.017, -.012], \
                  [-.043, -.049, -.048, -.049, -.043, -.042, -.042, -.036, -.025, -.021, -.016, -.011], \
                  [-.044, -.048, -.048, -.047, -.042, -.041, -.020, -.028, -.013, -.014, -.011, -.010], \
                  [-.043, -.049, -.047, -.045, -.042, -.037, -.003, -.013, -.010, -.003, -.007, -.008]], dtype=float).T

    s = .2 * alpha
    k = fix(s)
    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = .1 * beta
    m = fix(s)
    if m <= -3:
        m = -2

    if m >= 3:
        m = 2

    db = s - m
    n = m + fix(1.1 * sign(db))
    l = l + 3
    k = k + 3
    m = m + 4
    n = n + 4
    t = a[k - 1, m - 1]
    u = a[k - 1, n - 1]
    v = t + abs(da) * (a[l - 1, m - 1] - t)
    w = u + abs(da) * (a[l - 1, n - 1] - u)

    return v + (w - v) * abs(db)


def dldr(alpha, beta):
    """dldr function"""

    a = np.array([[.005, .017, .014, .010, -.005, .009, .019, .005, -.000, -.005, -.011, .008], \
                  [.007, .016, .014, .014, .013, .009, .012, .005, .000, .004, .009, .007], \
                  [.013, .013, .011, .012, .011, .009, .008, .005, -.002, .005, .003, .005], \
                  [.018, .015, .015, .014, .014, .014, .014, .015, .013, .011, .006, .001], \
                  [.015, .014, .013, .013, .012, .011, .011, .010, .008, .008, .007, .003], \
                  [.021, .011, .010, .011, .010, .009, .008, .010, .006, .005, .000, .001], \
                  [.023, .010, .011, .011, .011, .010, .008, .010, .006, .014, .020, .000]], dtype=float).T

    s = .2 * alpha
    k = fix(s)
    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = .1 * beta
    m = fix(s)

    if m <= -3:
        m = -2

    if m >= 3:
        m = 2

    db = s - m
    n = m + fix(1.1 * sign(db))
    l = l + 3
    k = k + 3
    m = m + 4
    n = n + 4
    t = a[k - 1, m - 1]
    u = a[k - 1, n - 1]

    v = t + abs(da) * (a[l - 1, m - 1] - t)
    w = u + abs(da) * (a[l - 1, n - 1] - u)

    return v + (w - v) * abs(db)


def dnda(alpha, beta):
    """dnda function"""

    a = np.array([[.001, -.027, -.017, -.013, -.012, -.016, .001, .017, .011, .017, .008, .016], \
                  [.002, -.014, -.016, -.016, -.014, -.019, -.021, .002, .012, .016, .015, .011], \
                  [-.006, -.008, -.006, -.006, -.005, -.008, -.005, .007, .004, .007, .006, .006], \
                  [-.011, -.011, -.010, -.009, -.008, -.006, .000, .004, .007, .010, .004, .010], \
                  [-.015, -.015, -.014, -.012, -.011, -.008, -.002, .002, .006, .012, .011, .011], \
                  [-.024, -.010, -.004, -.002, -.001, .003, .014, .006, -.001, .004, .004, .006], \
                  [-.022, .002, -.003, -.005, -.003, -.001, -.009, -.009, -.001, .003, -.002, .001]], dtype=float).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = .1 * beta
    m = fix(s)
    if m <= -3:
        m = -2

    if m >= 3:
        m = 2

    db = s - m
    n = m + fix(1.1 * sign(db))
    l = l + 3
    k = k + 3
    m = m + 4
    n = n + 4
    t = a[k - 1, m - 1]
    u = a[k - 1, n - 1]
    v = t + abs(da) * (a[l - 1, m - 1] - t)
    w = u + abs(da) * (a[l - 1, n - 1] - u)

    return v + (w - v) * abs(db)


def dndr(alpha, beta):
    """dndr function"""

    a = np.array([[-.018, -.052, -.052, -.052, -.054, -.049, -.059, -.051, -.030, -.037, -.026, -.013], \
                  [-.028, -.051, -.043, -.046, -.045, -.049, -.057, -.052, -.030, -.033, -.030, -.008], \
                  [-.037, -.041, -.038, -.040, -.040, -.038, -.037, -.030, -.027, -.024, -.019, -.013], \
                  [-.048, -.045, -.045, -.045, -.044, -.045, -.047, -.048, -.049, -.045, -.033, -.016], \
                  [-.043, -.044, -.041, -.041, -.040, -.038, -.034, -.035, -.035, -.029, -.022, -.009], \
                  [-.052, -.034, -.036, -.036, -.035, -.028, -.024, -.023, -.020, -.016, -.010, -.014], \
                  [-.062, -.034, -.027, -.028, -.027, -.027, -.023, -.023, -.019, -.009, -.025, -.010]], dtype=float).T

    s = .2 * alpha
    k = fix(s)
    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    s = .1 * beta
    m = fix(s)
    if m <= -3:
        m = -2

    if m >= 3:
        m = 2

    db = s - m
    n = m + fix(1.1 * sign(db))
    l = l + 3
    k = k + 3
    m = m + 4
    n = n + 4
    t = a[k - 1, m - 1]
    u = a[k - 1, n - 1]
    v = t + abs(da) * (a[l - 1, m - 1] - t)
    w = u + abs(da) * (a[l - 1, n - 1] - u)
    return v + (w - v) * abs(db)


def dampp(alpha):
    """dampp functon"""

    a = np.array([[-.267, -.110, .308, 1.34, 2.08, 2.91, 2.76, 2.05, 1.50, 1.49, 1.83, 1.21], \
                  [.882, .852, .876, .958, .962, .974, .819, .483, .590, 1.21, -.493, -1.04], \
                  [-.108, -.108, -.188, .110, .258, .226, .344, .362, .611, .529, .298, -2.27], \
                  [-8.80, -25.8, -28.9, -31.4, -31.2, -30.7, -27.7, -28.2, -29.0, -29.8, -38.3, -35.3], \
                  [-.126, -.026, .063, .113, .208, .230, .319, .437, .680, .100, .447, -.330], \
                  [-.360, -.359, -.443, -.420, -.383, -.375, -.329, -.294, -.230, -.210, -.120, -.100], \
                  [-7.21, -.540, -5.23, -5.26, -6.11, -6.64, -5.69, -6.00, -6.20, -6.40, -6.60, -6.00], \
                  [-.380, -.363, -.378, -.386, -.370, -.453, -.550, -.582, -.595, -.637, -1.02, -.840], \
                  [.061, .052, .052, -.012, -.013, -.024, .050, .150, .130, .158, .240, .150]], dtype=float).T

    s = .2 * alpha
    k = fix(s)

    if k <= -2:
        k = -1

    if k >= 9:
        k = 8

    da = s - k
    l = k + fix(1.1 * sign(da))
    k = k + 3
    l = l + 3

    d = np.zeros((9,))

    for i in range(9):
        d[i] = a[k - 1, i] + abs(da) * (a[l - 1, i] - a[k - 1, i])

    return d


def morelli_f16(alpha, beta, de, da, dr, p, q, r, cbar, b, V, xcg, xcgref):
    """desc"""

    # alpha=max(-10*pi/180,min(45*pi/180,alpha)) # bounds alpha between -10 deg and 45 deg
    # beta = max( - 30 * pi / 180, min(30 * pi / 180, beta)) #bounds beta between -30 deg and 30 deg
    # de = max( - 25 * pi / 180, min(25 * pi / 180, de)) #bounds elevator deflection between -25 deg and 25 deg
    # da = max( - 21.5 * pi / 180, min(21.5 * pi / 180, da)) #bounds aileron deflection between -21.5 deg and 21.5 deg
    # dr = max( - 30 * pi / 180, min(30 * pi / 180, dr)) #bounds rudder deflection between -30 deg and 30 deg

    # xcgref = 0.35
    # reference longitudinal cg position in Morelli f16 model

    phat = p * b / (2 * V)
    qhat = q * cbar / (2 * V)
    rhat = r * b / (2 * V)
    ##
    a0 = -1.943367e-2
    a1 = 2.136104e-1
    a2 = -2.903457e-1
    a3 = -3.348641e-3
    a4 = -2.060504e-1
    a5 = 6.988016e-1
    a6 = -9.035381e-1

    b0 = 4.833383e-1
    b1 = 8.644627
    b2 = 1.131098e1
    b3 = -7.422961e1
    b4 = 6.075776e1

    c0 = -1.145916
    c1 = 6.016057e-2
    c2 = 1.642479e-1

    d0 = -1.006733e-1
    d1 = 8.679799e-1
    d2 = 4.260586
    d3 = -6.923267

    e0 = 8.071648e-1
    e1 = 1.189633e-1
    e2 = 4.177702
    e3 = -9.162236

    f0 = -1.378278e-1
    f1 = -4.211369
    f2 = 4.775187
    f3 = -1.026225e1
    f4 = 8.399763
    f5 = -4.354000e-1

    g0 = -3.054956e1
    g1 = -4.132305e1
    g2 = 3.292788e2
    g3 = -6.848038e2
    g4 = 4.080244e2

    h0 = -1.05853e-1
    h1 = -5.776677e-1
    h2 = -1.672435e-2
    h3 = 1.357256e-1
    h4 = 2.172952e-1
    h5 = 3.464156
    h6 = -2.835451
    h7 = -1.098104

    i0 = -4.126806e-1
    i1 = -1.189974e-1
    i2 = 1.247721
    i3 = -7.391132e-1

    j0 = 6.250437e-2
    j1 = 6.067723e-1
    j2 = -1.101964
    j3 = 9.100087
    j4 = -1.192672e1

    k0 = -1.463144e-1
    k1 = -4.07391e-2
    k2 = 3.253159e-2
    k3 = 4.851209e-1
    k4 = 2.978850e-1
    k5 = -3.746393e-1
    k6 = -3.213068e-1

    l0 = 2.635729e-2
    l1 = -2.192910e-2
    l2 = -3.152901e-3
    l3 = -5.817803e-2
    l4 = 4.516159e-1
    l5 = -4.928702e-1
    l6 = -1.579864e-2

    m0 = -2.029370e-2
    m1 = 4.660702e-2
    m2 = -6.012308e-1
    m3 = -8.062977e-2
    m4 = 8.320429e-2
    m5 = 5.018538e-1
    m6 = 6.378864e-1
    m7 = 4.226356e-1

    n0 = -5.19153
    n1 = -3.554716
    n2 = -3.598636e1
    n3 = 2.247355e2
    n4 = -4.120991e2
    n5 = 2.411750e2

    o0 = 2.993363e-1
    o1 = 6.594004e-2
    o2 = -2.003125e-1
    o3 = -6.233977e-2
    o4 = -2.107885
    o5 = 2.141420
    o6 = 8.476901e-1

    p0 = 2.677652e-2
    p1 = -3.298246e-1
    p2 = 1.926178e-1
    p3 = 4.013325
    p4 = -4.404302

    q0 = -3.698756e-1
    q1 = -1.167551e-1
    q2 = -7.641297e-1

    r0 = -3.348717e-2
    r1 = 4.276655e-2
    r2 = 6.573646e-3
    r3 = 3.535831e-1
    r4 = -1.373308
    r5 = 1.237582
    r6 = 2.302543e-1
    r7 = -2.512876e-1
    r8 = 1.588105e-1
    r9 = -5.199526e-1

    s0 = -8.115894e-2
    s1 = -1.156580e-2
    s2 = 2.514167e-2
    s3 = 2.038748e-1
    s4 = -3.337476e-1
    s5 = 1.004297e-1

    ##
    Cx0 = a0 + a1 * alpha + a2 * de ** 2 + a3 * de + a4 * alpha * de + a5 * alpha ** 2 + a6 * alpha ** 3
    Cxq = b0 + b1 * alpha + b2 * alpha ** 2 + b3 * alpha ** 3 + b4 * alpha ** 4
    Cy0 = c0 * beta + c1 * da + c2 * dr
    Cyp = d0 + d1 * alpha + d2 * alpha ** 2 + d3 * alpha ** 3
    Cyr = e0 + e1 * alpha + e2 * alpha ** 2 + e3 * alpha ** 3
    Cz0 = (f0 + f1 * alpha + f2 * alpha ** 2 + f3 * alpha ** 3 + f4 * alpha ** 4) * (1 - beta ** 2) + f5 * de
    Czq = g0 + g1 * alpha + g2 * alpha ** 2 + g3 * alpha ** 3 + g4 * alpha ** 4
    Cl0 = h0 * beta + h1 * alpha * beta + h2 * alpha ** 2 * beta + h3 * beta ** 2 + h4 * alpha * beta ** 2 + h5 * \
          alpha ** 3 * beta + h6 * alpha ** 4 * beta + h7 * alpha ** 2 * beta ** 2
    Clp = i0 + i1 * alpha + i2 * alpha ** 2 + i3 * alpha ** 3
    Clr = j0 + j1 * alpha + j2 * alpha ** 2 + j3 * alpha ** 3 + j4 * alpha ** 4
    Clda = k0 + k1 * alpha + k2 * beta + k3 * alpha ** 2 + k4 * alpha * beta + k5 * alpha ** 2 * beta + k6 * alpha ** 3
    Cldr = l0 + l1 * alpha + l2 * beta + l3 * alpha * beta + l4 * alpha ** 2 * beta + l5 * alpha ** 3 * beta + l6 * beta ** 2
    Cm0 = m0 + m1 * alpha + m2 * de + m3 * alpha * de + m4 * de ** 2 + m5 * alpha ** 2 * de + m6 * de ** 3 + m7 * \
          alpha * de ** 2

    Cmq = n0 + n1 * alpha + n2 * alpha ** 2 + n3 * alpha ** 3 + n4 * alpha ** 4 + n5 * alpha ** 5
    Cn0 = o0 * beta + o1 * alpha * beta + o2 * beta ** 2 + o3 * alpha * beta ** 2 + o4 * alpha ** 2 * beta + o5 * \
          alpha ** 2 * beta ** 2 + o6 * alpha ** 3 * beta
    Cnp = p0 + p1 * alpha + p2 * alpha ** 2 + p3 * alpha ** 3 + p4 * alpha ** 4
    Cnr = q0 + q1 * alpha + q2 * alpha ** 2
    Cnda = r0 + r1 * alpha + r2 * beta + r3 * alpha * beta + r4 * alpha ** 2 * beta + r5 * alpha ** 3 * beta + r6 * \
           alpha ** 2 + r7 * alpha ** 3 + r8 * beta ** 3 + r9 * alpha * beta ** 3
    Cndr = s0 + s1 * alpha + s2 * beta + s3 * alpha * beta + s4 * alpha ** 2 * beta + s5 * alpha ** 2
    ##

    Cx = Cx0 + Cxq * qhat
    Cy = Cy0 + Cyp * phat + Cyr * rhat
    Cz = Cz0 + Czq * qhat
    Cl = Cl0 + Clp * phat + Clr * rhat + Clda * da + Cldr * dr
    Cm = Cm0 + Cmq * qhat + Cz * (xcgref - xcg)
    Cn = Cn0 + Cnp * phat + Cnr * rhat + Cnda * da + Cndr * dr - Cy * (xcgref - xcg) * (cbar / b)

    return Cx, Cy, Cz, Cl, Cm, Cn


def fix(ele):
    """round towards zero"""

    assert isinstance(ele, float)

    if ele > 0:
        rv = int(floor(ele))
    else:
        rv = int(ceil(ele))

    return rv


def sign(ele):
    """sign of a number"""

    if ele < 0:
        rv = -1
    elif ele == 0:
        rv = 0
    else:
        rv = 1

    return rv
