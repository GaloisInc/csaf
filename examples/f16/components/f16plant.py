"""
CSAF F-16 Model

taken from https://github.com/stanleybak/AeroBenchVVPython
"""

import numpy as np
import helpers.f16plant_helper as ph
from helpers.variables import State, Ctrlinput, Output, state_vector
from helpers.stevens_dyn import stevens_f16
from helpers.morelli_dyn import morelli_f16

parameters = {
    's': 300.0,
    'b': 30.0,
    'cbar': 11.32,
    'rm': 1.57e-3,
    'xcgref': 0.35,
    'xcg': 0.35,
    'he': 160.0,
    'c1': -0.770,
    'c2': 0.02755,
    'c3': 1.055e-4,
    'c4': 1.642e-6,
    'c5': 0.9604,
    'c6': 1.759e-2,
    'c7': 1.792e-5,
    'c8': -0.7336,
    'c9': 1.587e-5,
    'rtod': 57.29578,
    'g': 32.17,
    'xcg_mult': 1,
    'cxt_mult': 1,
    'cyt_mult': 1,
    'czt_mult': 1,
    'clt_mult': 1,
    'cmt_mult': 1,
    'cnt_mult': 1,
    'model': "morelli"
    }
def model_output(model, time_t, state_f16, input_controller):
    return subf16df(model, time_t, state_f16, input_controller)[1]


def model_state_update(model, time_t, state_f16, input_controller):
    return subf16df(model, time_t, state_f16, input_controller)[0]


def subf16df(model, t, x, u, adjust_cy=True):
    ''' Calculate state space differential '''
    #if len(f) != 4+4:
    #    raise E.SystemDimensionError("forcing vector must have 4 values")

    #parameters = model.parameters

    thtlc, el, ail, rdr = u[0:4]
    s, b, cbar, rm, xcgref, xcg, he, c1, c2, c3, c4, c5, c6, c7, c8, c9, rtod, g = \
        (parameters[p] for p in 's b cbar rm xcgref xcg he c1 c2 c3 c4 c5 c6 c7 c8 c9 rtod g'.split())

    xcg_mult, cxt_mult, cyt_mult, czt_mult, clt_mult, cmt_mult, cnt_mult = \
        (parameters[p] for p in 'xcg_mult cxt_mult cyt_mult czt_mult clt_mult cmt_mult cnt_mult'.split())

    vt, alpha, beta, phi, theta, psi, p, q, r  = x[0:9]
    alt, power = x[11], x[12]

    #XXX: Whats the rtod multiplier?
    alpha, beta = alpha*rtod, beta*rtod
    xcg *= xcg_mult

    # get air data computer and engine model

    qbar = ph.qbar(vt, alt)

    # XXX: nonlinear
    power_dot, thrust = ph.engine(thtlc, power, vt, alt)

    if parameters['model'] == 'stevens':
        cxt, cyt, czt, clt, cmt, cnt = stevens_f16(alpha=alpha,
                                                   beta=beta, el=el, ail=ail, rdr=rdr, dail=ail/20,
                                                   drdr=rdr/30)
    elif parameters['model'] == 'morelli':
        cxt, cyt, czt, clt, cmt, cnt = morelli_f16(alpha=alpha,
                                                   beta=beta, de=el, da=ail, dr=rdr, p=p, q=q, r=r,
                                                   cbar=cbar, b=b, V=vt, xcg=xcg, xcgref=xcgref)
    else:
        raise NotImplementedError

    cxt *= cxt_mult; cyt *= cyt_mult; czt *= czt_mult; clt *= clt_mult; cmt *= cmt_mult; cnt *= cnt_mult

    tvt = .5 / vt
    b2v = b * tvt
    cq = cbar * q * tvt

    # Add damping derivatives
    # XXX: nonlinear
    d = ph.dampp_lookup(alpha)

    cxt = cxt + cq * d[0]
    cyt = cyt + b2v * (d[1] * r + d[2] * p)
    czt = czt + cq * d[3]
    clt = clt + b2v * (d[4] * r + d[5] * p)
    cmt = cmt + cq * d[6] + czt * (xcgref-xcg)
    cnt = cnt + b2v * (d[7] * r + d[8] * p)-cyt * (xcgref-xcg) * cbar/b

    # Get redy for state equations
    cbta = np.cos(x[2])
    u = vt * np.cos(x[1]) * cbta
    v = vt * np.sin(x[2])
    w = vt * np.sin(x[1]) * cbta
    sth = np.sin(theta)
    cth = np.cos(theta)
    sph = np.sin(phi)
    cph = np.cos(phi)
    spsi = np.sin(psi)
    cpsi = np.cos(psi)
    qs = qbar * s
    qsb = qs * b
    rmqs = rm * qs
    gcth = g * cth
    qsph = q * sph
    ay = rmqs * cyt
    az = rmqs * czt

    # force equations
    udot = r * v-q * w-g * sth + rm * (qs * cxt + thrust)
    vdot = p * w-r * u + gcth * sph + ay
    wdot = q * u-p * v + gcth * cph + az
    dum = (u * u + w * w)

    vt_dot = (u * udot + v * vdot + w * wdot)/vt
    alpha_dot = (u * wdot-w * udot)/dum
    beta_dot = (vt * vdot-v * vt_dot) * cbta/dum

    # kinematics
    phi_dot = p + (sth/cth) * (qsph + r * cph)
    theta_dot = q * cph-r * sph
    psi_dot = (qsph + r * cph)/cth

    # XXX: Looks quite different form the book
    # moments
    p_dot = (c2 * p + c1 * r + c4 * he) * q + qsb * (c3 * clt + c4 * cnt)
    q_dot = (c5 * p-c7 * he) * r + c6 * (r * r-p * p) + qs * cbar * c7 * cmt
    r_dot = (c8 * p-c2 * r + c9 * he) * q + qsb * (c4 * clt + c9 * cnt)

    # navigation
    t1 = sph * cpsi
    t2 = cph * sth
    t3 = sph * spsi
    s1 = cth * cpsi
    s2 = cth * spsi
    s3 = t1 * sth-cph * spsi
    s4 = t3 * sth + cph * cpsi
    s5 = sph * cth
    s6 = t2 * cpsi + t3
    s7 = t2 * spsi-t1
    s8 = cph * cth
    pn_dot = u * s1 + v * s3 + w * s6    # north speed
    pe_dot = u * s2 + v * s4 + w * s7   # east speed
    alt_dot = u * sth-v * s5-w * s8      # vertical speed

    xa = 15.0                  # sets distance normal accel is in front of the c.g. (xa = 15.0 at pilot)
    az = az-xa * q_dot           # moves normal accel in front of c.g.

    if adjust_cy:
        ay = ay+xa*r_dot           # moves side accel in front of c.g.

    # For extraction of Nz
    Nz = (-az / g) - 1 # zeroed at 1 g, positive g = pulling up
    Ny = ay / g

    output = np.array([Nz, Ny, az, ay])

    xdot = np.array(state_vector(vt=vt_dot, alpha=alpha_dot, beta=beta_dot,
                                 phi=phi_dot, theta=theta_dot, psi=psi_dot, p=p_dot, q=q_dot,
                                 r=r_dot, pn=pn_dot, pe=pe_dot, h=alt_dot, power=power_dot))
    return xdot, output
