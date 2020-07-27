import numpy as np

import f16plant_helper as ph

from csaf import message

import os
import toml
import json

def main():
    this_path = os.path.dirname(os.path.realpath(__file__))
    info_file = os.path.join(this_path, "f16plant.toml")
    with open(info_file, 'r') as ifp:
        info = toml.load(ifp)

    parameters = info["parameters"]

    n_states = 13
    n_outputs = 2
    n_inputs = 4

    fs = info["sampling_frequency"]

    x = info["topics"]["states"]["initial"]
    epoch = 0

    msg_writer = message.Message()

    while True:
        ins = input(f"msg at [t={epoch/fs}]>")
        try:
            msg = json.loads(ins)
        except json.decoder.JSONDecodeError as exc:
            raise Exception(f"input <{ins}> couldn't be interpreted as json! {exc}")

        in_epoch = msg["epoch"]
        epoch = in_epoch
        #assert in_epoch == epoch

        f = msg["Output"]
        assert len(f) == n_inputs

        xd, output = subf16df(epoch/fs, x, f, parameters)
        assert len(xd) == n_states
        assert len(output) == n_outputs

        msg = msg_writer.write_message(epoch/fs, output=output, state=x, differential=xd)

        # TODO: for now do bad integration
        x += 1/fs * xd
        epoch += 1

        print(msg)


def subf16df(t, x, f, parameters, mult=None):
    """ Calculate state space differential """

    thtlc, el, ail, rdr = f

    s = parameters["s"]
    b = parameters["b"]
    cbar = parameters["cbar"]
    rm = parameters["rm"]
    xcgr = parameters["xcgr"]
    xcg = parameters["xcg"]
    he = parameters["he"]
    c1 = parameters["c1"]
    c2 = parameters["c2"]
    c3 = parameters["c3"]
    c4 = parameters["c4"]
    c5 = parameters["c5"]
    c6 = parameters["c6"]
    c7 = parameters["c7"]
    c8 = parameters["c8"]
    c9 = parameters["c9"]
    rtod = parameters["rtod"]
    g = parameters["g"]
    equations = parameters["equations"]

    xd = np.zeros((13))
    vt = x[0]
    alpha = x[1]*rtod
    beta = x[2]*rtod
    phi = x[3]
    theta = x[4]
    psi = x[5]
    p = x[6]
    q = x[7]
    r = x[8]
    alt = x[11]
    power = x[12]

    if mult is not None:
        xcg *= mult[0]

    # get air data computer and engine model
    amach, qbar = ph.adc(vt, alt)
    cpow = ph.tgear(thtlc)

    xd[12] = ph.pdot(power, cpow)

    t = ph.thrust(power, alt, amach)
    dail = ail/20
    drdr = rdr/30

    # get cxt, cyt, czt, clt, cmt, cnt
    if equations == "stevens":
        cxt = ph.cx(alpha, el)
        cyt = ph.cy(beta, ail, rdr)
        czt = ph.cz(alpha, beta, el)

        clt = ph.cl(alpha, beta) + ph.dlda(alpha, beta) * dail + ph.dldr(alpha, beta) * drdr
        cmt = ph.cm(alpha, el)
        cnt = ph.cn(alpha, beta) + ph.dnda(alpha, beta) * dail + ph.dndr(alpha, beta) * drdr
    elif equations == "morelli":
        pi = np.pi
        cxt, cyt, czt, clt, cmt, cnt = ph.morelli_f16(alpha * pi / 180, beta * pi / 180, el * pi / 180, ail * pi / 180, rdr * pi / 180, \
                                                      p, q, r, cbar, b, vt, xcg, xcgr)

    if mult is not None:
        cxt *= mult[1]
        cyt *= mult[2]
        czt *= mult[3]

        clt *= mult[4]
        cmt *= mult[5]
        cnt *= mult[6]

    tvt = .5 / vt
    b2v = b * tvt
    cq = cbar * q * tvt

    # get ready for state equations
    d = ph.dampp(alpha)
    cxt = cxt + cq * d[0]
    cyt = cyt + b2v * (d[1] * r + d[2] * p)
    czt = czt + cq * d[3]
    clt = clt + b2v * (d[4] * r + d[5] * p)
    cmt = cmt + cq * d[6] + czt * (xcgr-xcg)
    cnt = cnt + b2v * (d[7] * r + d[8] * p)-cyt * (xcgr-xcg) * cbar/b
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
    udot = r * v-q * w-g * sth + rm * (qs * cxt + t)
    vdot = p * w-r * u + gcth * sph + ay
    wdot = q * u-p * v + gcth * cph + az
    dum = (u * u + w * w)

    xd[0] = (u * udot + v * vdot + w * wdot)/vt
    xd[1] = (u * wdot-w * udot)/dum
    xd[2] = (vt * vdot-v * xd[0]) * cbta/dum

    # kinematics
    xd[3] = p + (sth/cth) * (qsph + r * cph)
    xd[4] = q * cph-r * sph
    xd[5] = (qsph + r * cph)/cth

    # moments
    xd[6] = (c2 * p + c1 * r + c4 * he) * q + qsb * (c3 * clt + c4 * cnt)

    xd[7] = (c5 * p-c7 * he) * r + c6 * (r * r-p * p) + qs * cbar * c7 * cmt
    xd[8] = (c8 * p-c2 * r + c9 * he) * q + qsb * (c4 * clt + c9 * cnt)

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
    xd[9] = u * s1 + v * s3 + w * s6    # north speed
    xd[10] = u * s2 + v * s4 + w * s7   # east speed
    xd[11] = u * sth-v * s5-w * s8      # vertical speed

    xa = 15.0                  # sets distance normal accel is in front of the c.g. (xa = 15.0 at pilot)
    az = az-xa * xd[7]           # moves normal accel in front of c.g.

    # For extraction of Nz
    Nz = (-az / g) - 1 # zeroed at 1 g, positive g = pulling up
    Ny = ay / g

    output = np.array([Nz, Ny])

    return xd, output


if __name__ == "__main__":
    main()