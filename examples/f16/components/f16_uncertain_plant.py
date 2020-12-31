import numpy as np
import time

from DaTaReachControl.interval import *
from DaTaReachControl import compBuilder

from numba import jit

n_f16 = 12  # Number of states
m_f16 = 4   # Number of control

# # Add the LIpschitz costants and dependencies of unknown term in f
infoF = list()
infoF.append(compBuilder(ind=0, term=1, Lip=350.0, vDep=[0,8,9], weightLip=[0.0179, 3.4806e-04, 1.0], nS=n_f16,
                    bound=(-1500,3500), gradBound=[(-6.0,-1.0), (-0.2,0.1),(210,280)]))
infoF.append(compBuilder(ind=0, term=2, Lip=17.0, vDep=[5,7,1], weightLip=[0.0378, 0.0588, 1.0], nS=n_f16,
                    bound=(-1.1,1.5), gradBound=[(-0.2,0.6), (0.4,1.0),(-17,5)]))
infoF.append(compBuilder(ind=0, term=3, Lip=160.0, vDep=[1], weightLip=[], nS=n_f16,
                    bound=(-14.0,3.0), gradBound=[]))

infoF.append(compBuilder(ind=1, term=1, Lip=350.0, vDep=[0,8,9], weightLip=[0.0179, 3.4806e-04, 1.0], nS=n_f16,
                    bound=(-1500,3500), gradBound=[(-6.0,-1.0), (-0.2,0.1),(210,280)]))
infoF.append(compBuilder(ind=1, term=2, Lip=380.0, vDep=[1], weightLip=[], nS=n_f16,
                    bound=(-25,40), gradBound=[(-45.0, 376.0)]))

infoF.append(compBuilder(ind=5, term=1, Lip=55, vDep=[5,7,1], weightLip=[0.1216,0.1863,1.0], nS=n_f16,
                    bound=(-12.0,12.0), gradBound=[(-7,-3),(-5,10),(-52, 26)]))
infoF.append(compBuilder(ind=6, term=1, Lip=250, vDep=[1], weightLip=[], nS=n_f16,
                    bound=(-22,-1), gradBound=[(-62.0, 250)]))
infoF.append(compBuilder(ind=7, term=1, Lip=8.0, vDep=[5,7,1], weightLip=[0.0531,0.24,1.0], nS=n_f16,
                    bound=(-1.0,1.0), gradBound=[(-0.15,0.26),(-1.2,-0.6),(-5, 5)]))

infoF.append(compBuilder(ind=10, term=1, Lip=2370, vDep=[1], weightLip=[], nS=n_f16,
                    bound=(-63,180), gradBound=[(-2.3657*1e3,0.25*1e3)]))

infoG = list()
infoG.append(compBuilder(ind=(9,0), term=0, Lip=780.0, vDep=[9], weightLip=[], nS=n_f16,
                    bound=(6.0,70.0), gradBound=[]))


# Unknown term that are similar
simVarF = {(0,1): [(1,1), (2,1)], (0,2): [(2,2)], (0,3): [(2,3)]}
simVarG = {}

@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def known_f16(x_lb, x_ub, gradF=None, gradG=None):
    # Output of the known part/term of f for the f16
    res_f = dict()
    res_G = dict()

    # Get the states of the f16
    vt = x_lb[0], x_ub[0]
    alpha = x_lb[1], x_ub[1]
    beta = x_lb[2], x_ub[2]
    phi = x_lb[3], x_ub[3]
    theta = x_lb[4], x_ub[4]
    p = x_lb[5], x_ub[5]
    q = x_lb[6], x_ub[6]
    r = x_lb[7], x_ub[7]
    alt = x_lb[8], x_ub[8]
    power = x_lb[9], x_ub[9]
    # int_Nz = x_lb[10], x_ub[10]
    # int_Ps = x_lb[10], x_ub[10]

    # Temporary variables
    t2 = cos_i(*alpha)
    t3 = cos_i(*beta)
    t4 = cos_i(*phi)
    t5 = sin_i(*alpha)
    t6 = sin_i(*beta)
    t7 = cos_i(*theta)
    t8 = sin_i(*phi)
    t9 = sin_i(*theta)

    t10 = pow2_i(*alpha)
    t11 = mul_i(*t10, *alpha)
    t13 = pow2_i(*beta)
    t14 = mul_i(*t13, *beta)
    t15 = pow2_i(*vt)
    t20 = inv_i(*vt)
    t12 = pow2_i(*t10)
    t18 = mul_i(*r, *t4)
    t19 = mul_i(*q, *t8)
    t21 = inv_i(*t3)
    alpha_5 = mul_i(*t12, *alpha)
    t22 = inv_i(*t7)
    t24 = add_i(*t18, *t19)

    inv_g = -1.0/32.17


    cyt_c = add_i(*mul_i(*r,*add_i(*mul_i_scalar(*alpha,1.7844495),*add_i(*mul_i_scalar(*t10,6.266553e+1),*add_i(*mul_i_scalar(*t11,-1.3743354e+2),1.2107472e+1)))),*add_i(*mul_i(*beta,*mul_i_scalar(*vt,-1.145916)),*mul_i(*p,*add_i(*mul_i_scalar(*alpha,1.30196985e+1),*add_i(*mul_i_scalar(*t10,6.390879e+1),*add_i(*mul_i_scalar(*t11,-1.03849005e+2),-1.5100995))))))
    cxt_c = add_i(*mul_i(*q,*add_i(*mul_i_scalar(*alpha,4.892858882e+1),*add_i(*mul_i_scalar(*t10,6.40201468e+1),*add_i(*mul_i_scalar(*t11,-4.201395926e+2),*add_i(*mul_i_scalar(*t12,3.438889216e+2),2.735694778))))),*mul_i(*vt,*add_i(*mul_i_scalar(*alpha,2.136104e-1),*add_i(*mul_i_scalar(*t10,6.988016e-1),*add_i(*mul_i_scalar(*t11,-9.035381e-1),-1.943367e-2)))))
    czt_c = add_i(*mul_i(*vt,*mul_i_scalar(*add_i(*mul_i_scalar(*alpha,4.211369),*add_i(*mul_i_scalar(*t10,-4.775187),*add_i(*mul_i_scalar(*t11,1.026225e+1),*add_i(*mul_i_scalar(*t12,-8.399763),*add_i(*mul_i_scalar(*mul_i_scalar(*t13, -1.0),1.378278e-1),*add_i(*mul_i(*alpha,*mul_i_scalar(*t13,-4.211369)),*add_i(*mul_i(*t10,*mul_i_scalar(*t13,4.775187)),*add_i(*mul_i(*t11,*mul_i_scalar(*t13,-1.026225e+1)),*add_i(*mul_i(*t12,*mul_i_scalar(*t13,8.399763)),1.378278e-1))))))))),-1.0)),*mul_i(*q,*mul_i_scalar(*add_i(*mul_i_scalar(*alpha,2.33888463e+2),*add_i(*mul_i_scalar(*t10,-1.863718008e+3),*add_i(*mul_i_scalar(*t11,3.875989508e+3),*add_i(*mul_i_scalar(*t12,-2.309418104e+3),1.729105096e+2)))),-1.0)))

    cnt_c = add_i(*mul_i(*p,*add_i(*mul_i_scalar(*alpha,-4.947369),*add_i(*mul_i_scalar(*t10,2.889267),*add_i(*mul_i_scalar(*t11,6.0199875e+1),*add_i(*mul_i_scalar(*t12,-6.606453000000001e+1),4.016478e-1))))),*add_i(*mul_i(*vt,*add_i(*mul_i_scalar(*t13,-2.003125e-1),*add_i(*mul_i(*beta,*add_i(*add_i(*mul_i_scalar(*alpha,6.594004000000001e-2),*add_i(*mul_i_scalar(*t10,-2.107885),*mul_i_scalar(*t11,8.476901e-1))),2.993363e-1)),*add_i(*mul_i(*alpha,*mul_i_scalar(*t13,-6.233977e-2)),*mul_i(*t10,*mul_i_scalar(*t13,2.14142)))))),*mul_i(*mul_i_scalar(*r, -1.0),*add_i(*mul_i_scalar(*alpha,1.7513265),*add_i(*mul_i_scalar(*t10,1.14619455e+1),5.548134)))))
    cmt_c = add_i(*mul_i(*q,*mul_i_scalar(*add_i(*mul_i_scalar(*alpha,2.011969256e+1),*add_i(*mul_i_scalar(*t10,2.036827976e+2),*add_i(*mul_i_scalar(*t11,-1.27200293e+3),*add_i(*mul_i_scalar(*t12,2.332480906e+3),*add_i(*mul_i_scalar(*alpha_5,-1.3650505e+3),2.93840598e+1))))),-1.0)),*mul_i(*vt,*add_i(*mul_i_scalar(*alpha,4.660702e-2),-2.02937e-2)))
    clt_c = add_i(*mul_i(*r,*add_i(*mul_i_scalar(*alpha,9.101584500000001),*add_i(*mul_i_scalar(*mul_i_scalar(*t10, -1.0),1.652946e+1),*add_i(*mul_i_scalar(*t11,1.36501305e+2),*add_i(*mul_i_scalar(*mul_i_scalar(*t12, -1.0),1.789008e+2),9.375655500000001e-1))))),*add_i(*mul_i(*mul_i_scalar(*p, -1.0),*add_i(*mul_i_scalar(*alpha,1.784961),*add_i(*mul_i_scalar(*t10,-1.8715815e+1),*add_i(*mul_i_scalar(*t11,1.1086698e+1),6.190209)))),*mul_i(*mul_i_scalar(*vt, -1.0),*add_i(*mul_i(*beta,*add_i(*add_i(*mul_i_scalar(*alpha,5.776677e-1),*add_i(*mul_i_scalar(*t10,1.672435e-2),*add_i(*mul_i_scalar(*t11,-3.464156),*mul_i_scalar(*t12,2.835451)))),1.05853e-1)),*mul_i(*t13,*add_i(*mul_i_scalar(*alpha,-2.172952e-1),*add_i(*mul_i_scalar(*t10,1.098104),-1.357256e-1)))))))

    qbar_k = mul_i_scalar(*powr_i(1.0-0.703e-5*alt[1],1.0-0.703e-5*alt[0], 4.14) , 0.5 * 2.377e-3)

    # vt_dot =
    res_f[(0,0)] = add_i(*mul_i(*qbar_k,*mul_i(*vt,*mul_i_scalar(*add_i(*mul_i(*cyt_c,*t6),*mul_i(*t3,*add_i(*mul_i(*cxt_c,*t2),*mul_i(*czt_c,*t5)))),4.71e-1))),*mul_i_scalar(*add_i(*mul_i(*t2,*mul_i(*t3,*mul_i_scalar(*t9,-1.0))),*mul_i(*t7,*add_i(*mul_i(*t6,*t8),*mul_i(*t3,*mul_i(*t4,*t5))))),3.217e+1))
    res_f[(0,1)] = mul_i_scalar(*mul_i(*t2,*t3), 0.00157) # thrust
    res_f[(0,2)] = mul_i_scalar(*mul_i(*qbar_k,*mul_i(*vt, *t6)), 7.065) # p*d2 + r*d1
    res_f[(0,3)] = mul_i_scalar(*mul_i(*qbar_k,*mul_i(*vt, *mul_i(*q, *t3))), 2.66586) # d0 cos(alpha) + d3 sin(alpha)

    # alpha_dot =
    res_f[(1,0)] = add_i(*q,*mul_i(*t21,*add_i(*mul_i(*qbar_k,*mul_i_scalar(*add_i(*mul_i(*cxt_c,*mul_i_scalar(*t5,-1.0)),*mul_i(*czt_c,*t2)),4.71e-1)),*add_i(*mul_i(*t6,*mul_i_scalar(*add_i(*mul_i(*p,*t2),*mul_i(*r,*t5)),-1.0)),*mul_i(*add_i(*mul_i(*t5,*t9),*mul_i(*t2,*mul_i(*t4,*t7))),*mul_i_scalar(*t20,3.217e+1))))))
    res_f[(1,1)] = mul_i_scalar(*mul_i(*t5,*mul_i(*t21, *t20)), -0.00157) # thrust
    res_f[(1,2)] = mul_i_scalar(*mul_i(*qbar_k,*mul_i(*t21,*q)), -2.66586) # d0 sin(alpha) - d3 cos(alpha)

    # beta_dot =
    res_f[(2,0)] = add_i(*mul_i(*p,*t5),*add_i(*mul_i(*r,*mul_i_scalar(*t2,-1.0)),*add_i(*mul_i(*qbar_k,*mul_i_scalar(*add_i(*mul_i(*cyt_c,*mul_i_scalar(*t3,-1.0)),*mul_i(*t6,*add_i(*mul_i(*cxt_c,*t2),*mul_i(*czt_c,*t5)))),-4.71e-1)),*mul_i(*t20,*mul_i_scalar(*add_i(*mul_i(*t2,*mul_i(*t6,*t9)),*mul_i(*t7,*add_i(*mul_i(*t3,*t8),*mul_i(*mul_i_scalar(*t4, -1.0),*mul_i(*t5,*t6))))),3.217e+1)))))
    res_f[(2,1)] = mul_i_scalar(*mul_i(*t2, *mul_i(*t6,*t20)),-0.00157) # Thrust
    res_f[(2,2)] = mul_i_scalar(*mul_i(*qbar_k, *t3) ,7.065) # p*d2 + r*d1
    res_f[(2,3)] = mul_i_scalar(*mul_i(*qbar_k,*mul_i(*q, *t6)), -2.66586) # d0 cos(alpha) + d3 sin(alpha)

    # phi_dot and theta_dot are completely known
    res_f[(3,0)] = add_i(*p,*mul_i(*t9,*mul_i(*t22,*t24)))
    res_f[(4,0)] = add_i(*mul_i(*q,*t4),*mul_i(*mul_i_scalar(*r, -1.0),*t8))

    # p_dot =
    res_f[(5,0)] = add_i(*mul_i_scalar(*q,2.6272e-4),*add_i(*mul_i(*p,*mul_i_scalar(*q,2.755e-2)),*add_i(*mul_i(*q,*mul_i_scalar(*r,-7.7e-1)),*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*clt_c,9.495e-1),*mul_i_scalar(*cnt_c,1.4778e-2)))))))
    res_f[(5,1)] = mul_i(*qbar_k, *vt) # (14.242500000000000398938694579076*d5*p + 0.22166999999999998739615137115927*d8*p + 14.242500000000000398938694579076*d4*r + 0.22166999999999998739615137115927*d7*r)

    #q_dot
    res_f[(6,0)] = add_i(*mul_i(*r,*add_i(*mul_i_scalar(*p,9.604e-1),-2.8672e-3)),*add_i(*mul_i_scalar(*pow2_i(*p),-1.759e-2),*add_i(*mul_i_scalar(*pow2_i(*r),1.759e-2),*mul_i(*cmt_c,*mul_i(*qbar_k,*mul_i_scalar(*vt,6.085632e-2))))))
    res_f[(6,1)] = mul_i_scalar(*mul_i(*qbar_k, *mul_i(*vt, *q)), 0.34444677120000002348783373420926) # d6

    # r_dot
    res_f[(7,0)] = add_i(*mul_i(*q,*add_i(*add_i(*mul_i_scalar(*p,-7.336e-1),*mul_i_scalar(*r,-2.755e-2)),2.5392e-3)),*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*clt_c,1.4778e-2),*mul_i_scalar(*cnt_c,1.4283e-1)))))
    res_f[(7,1)] = mul_i(*qbar_k, *vt) # (0.22166999999999998739615137115927*d5*p + 2.1424499999999998415872639462298*d8*p + 0.22166999999999998739615137115927*d4*r + 2.1424499999999998415872639462298*d7*r)

    # Altitude dot is also known
    res_f[(8,0)] = mul_i(*vt,*mul_i_scalar(*add_i(*mul_i(*t2,*mul_i(*t3,*mul_i_scalar(*t9,-1.0))),*mul_i(*t7,*add_i(*mul_i(*t6,*t8),*mul_i(*t3,*mul_i(*t4,*t5))))),-1.0))

    # Power dot evolution
    res_f[(9,0)] = mul_i_scalar(*power, -1.0)

    # Nz evolution
    res_f[(10,0)] = sub_i(*mul_i_scalar(*add_i(*mul_i(*qbar_k,*mul_i(*vt,*mul_i_scalar(*czt_c,4.71e-1))), *mul_i_scalar(*res_f[(6,0)],-15.0)) ,inv_g), 1.0)
    res_f[(10,1)] = mul_i_scalar(*mul_i(*qbar_k,*mul_i(*vt,*q)), inv_g) # 2.66586*d3-15*0.34444677120000002348783373420926*d6

    res_f[(11,0)] = add_i(*mul_i(*p,  *t2), *mul_i(*r, *t5)) # P_s

    # Known part of G
    cxt_el = add_i(*add_i(*mul_i_scalar(*alpha,-3.596257905051324e-3),-5.844481091727544e-5), -0.002221111, 0.002221111) # uncertaincy for quad term
    czt_el = (-7.599163563183311e-3, -7.599163563183311e-3)
    cyt_ail = (1.050000026376525e-3,1.050000026376525e-3)
    cyt_rdr = (2.866666644486394e-3, 2.866666644486394e-3)
    clt_ail = add_i(*mul_i_scalar(*alpha,-7.110314292992219e-4),*add_i(*mul_i_scalar(*t10,8.46695697523816e-3),*add_i(*mul_i_scalar(*t11,-5.607861569046917e-3),*add_i(*mul_i(*beta,*add_i(*mul_i_scalar(*alpha,5.199074042303309e-3),*add_i(*mul_i_scalar(*t10,-6.538689292366793e-3),5.677833564088621e-4))),-2.553668023079992e-3))))
    clt_rdr = add_i(*mul_i_scalar(*alpha,-3.827349969990885e-4),*add_i(*mul_i_scalar(*t13,-2.757382853372769e-4),*add_i(*mul_i(*mul_i_scalar(*beta, -1.0),*add_i(*mul_i_scalar(*alpha,1.015398175824037e-3),*add_i(*mul_i_scalar(*t10,-7.88218440935746e-3),*add_i(*mul_i_scalar(*t11,8.602207774962956e-3),5.502850343942174e-5)))),4.600214924029762e-4)))
    cnt_ail = add_i(*mul_i_scalar(*alpha,7.46417107218781e-4),*add_i(*mul_i_scalar(*t10,4.01869565187478e-3),*add_i(*mul_i_scalar(*t11,-4.385795989434503e-3),*add_i(*mul_i(*beta,*add_i(*mul_i_scalar(*alpha,6.171189274408362e-3),*add_i(*mul_i_scalar(*t10,-2.396874624397829e-2),*add_i(*mul_i_scalar(*t11,2.159988066341646e-2),1.147317665605552e-4)))),*add_i(*mul_i(*t14,*add_i(*mul_i_scalar(*alpha,-9.074884824305068e-3),2.771766111738455e-3)),-5.844613736750695e-4)))))
    cnt_rdr = add_i(*mul_i_scalar(*alpha,-2.018612906271602e-4),*add_i(*mul_i_scalar(*t10,1.752828931790149e-3),*add_i(*mul_i(*beta,*add_i(*mul_i_scalar(*alpha,3.558286521844935e-3),*add_i(*mul_i_scalar(*t10,-5.824994490629027e-3),4.388049209498828e-4))),-1.416490720428527e-3)))
    cmt_el = add_i(*add_i(*mul_i_scalar(*alpha,-1.407254961625748e-3),*add_i(*mul_i_scalar(*t10,8.759001173645119e-3),-1.049345702439952e-2)), *add_i(*mul_i(*alpha,-0.00321855415, 0.00321855415),-0.0006336369034,0.002753244479)) # add uncertaincy el^2, el^3


    t16 = mul_i(*cxt_el, *t2)
    t17 = mul_i(*czt_el, *t5)
    t23 = add_i(*t16, *t17)

    res_G[(0,1,0)] = mul_i(*qbar_k,*mul_i(*t3,*mul_i(*t15,*mul_i_scalar(*t23,4.71e-1))))
    res_G[(0,2,0)] = mul_i(*cyt_ail,*mul_i(*qbar_k,*mul_i(*t6,*mul_i_scalar(*t15,4.71e-1))))
    res_G[(0,3,0)] = mul_i(*cyt_rdr,*mul_i(*qbar_k,*mul_i(*t6,*mul_i_scalar(*t15,4.71e-1))))

    res_G[(1,1,0)] = mul_i(*qbar_k,*mul_i(*t21,*mul_i(*vt,*mul_i_scalar(*add_i(*mul_i(*cxt_el,*t5),*mul_i(*mul_i_scalar(*czt_el, -1.0),*t2)),-4.71e-1))))

    res_G[(2,1,0)] = mul_i(*qbar_k,*mul_i(*t6,*mul_i(*t23,*mul_i_scalar(*vt,-4.71e-1))))
    res_G[(2,2,0)] = mul_i(*cyt_ail,*mul_i(*qbar_k,*mul_i(*t3,*mul_i_scalar(*vt,4.71e-1))))
    res_G[(2,3,0)] = mul_i(*cyt_rdr,*mul_i(*qbar_k,*mul_i(*t3,*mul_i_scalar(*vt,4.71e-1))))

    res_G[(5,2,0)] = mul_i(*qbar_k,*mul_i(*t15,*add_i(*mul_i_scalar(*clt_ail,9.495e-1),*mul_i_scalar(*cnt_ail,1.4778e-2))))
    res_G[(5,3,0)] = mul_i(*qbar_k,*mul_i(*t15,*add_i(*mul_i_scalar(*clt_rdr,9.495e-1),*mul_i_scalar(*cnt_rdr,1.4778e-2))))

    res_G[(6,1,0)] = mul_i(*cmt_el,*mul_i(*qbar_k,*mul_i_scalar(*t15,6.085632e-2)))

    res_G[(7,2,0)] = mul_i(*qbar_k,*mul_i(*t15,*add_i(*mul_i_scalar(*clt_ail,1.4778e-2),*mul_i_scalar(*cnt_ail,1.4283e-1))))
    res_G[(7,3,0)] = mul_i(*qbar_k,*mul_i(*t15,*add_i(*mul_i_scalar(*clt_rdr,1.4778e-2),*mul_i_scalar(*cnt_rdr,1.4283e-1))))

    res_G[(9,0,0)] = (1.0,1.0) # 0th Term is unknown
    res_G[(10,1,0)] = mul_i_scalar(*add_i(*mul_i(*qbar_k,*mul_i(*t15,*mul_i_scalar(*czt_el,4.71e-1))), *mul_i_scalar(*res_G[(6,1,0)],-15.0)) ,inv_g)

    # Compute the gradient if asked for
    if gradF is None or gradG is None:
        return res_f, res_G

    dalt_qbar_k = mul_i_scalar(*powr_i(1.0-0.703e-5*alt[1],1.0-0.703e-5*alt[0], 3.14), -0.703e-5 * 4.14 * 2.377e-3 * 0.5)

    t23 = t10 # alpha^2
    t24 = t11 # alpha^3
    t25 = t12 # alpha^4
    t26 = t13 # beta^2
    beta_trois = t14

    t33 = t18
    t34 = t19
    t27 = t15 # vt^2
    t35 = t20
    alpha_cinq = alpha_5
    t36 = inv_i(*t27)
    t38 = t21
    t40 = t22

    dalph_cxt_el = (-3.596257905051324e-3, -3.596257905051324e-3)
    dvt_cxt_c = add_i(*mul_i_scalar(*alpha,2.136104e-1),*add_i(*mul_i_scalar(*t23,6.988016e-1),*add_i(*mul_i_scalar(*t24,-9.035381e-1),-1.943367e-2)))
    dalph_cxt_c = add_i(*mul_i(*q,*add_i(*mul_i_scalar(*alpha,1.280402936e+2),*add_i(*mul_i_scalar(*mul_i_scalar(*t23, -1.0),1.2604187778e+3),*add_i(*mul_i_scalar(*t24,1.3755556864e+3),4.892858882e+1)))),*mul_i(*vt,*add_i(*mul_i_scalar(*alpha,1.3976032),*add_i(*mul_i_scalar(*t23,-2.7106143),2.136104e-1))))
    dq_cxt_c = add_i(*mul_i_scalar(*alpha,4.892858882e+1),*add_i(*mul_i_scalar(*t23,6.40201468e+1),*add_i(*mul_i_scalar(*t24,-4.201395926e+2),*add_i(*mul_i_scalar(*t25,3.438889216e+2),2.735694778))))
    dvt_cyt_c = mul_i_scalar(*beta, -1.145916)
    dalph_cyt_c = add_i(*mul_i(*r,*add_i(*mul_i_scalar(*alpha,1.2533106e+2),*add_i(*mul_i_scalar(*t23,-4.1230062e+2),1.7844495))),*mul_i(*p,*add_i(*mul_i_scalar(*alpha,1.2781758e+2),*add_i(*mul_i_scalar(*t23,-3.11547015e+2),1.30196985e+1))))
    dbeta_cyt_c = mul_i_scalar(*vt, -1.145916)
    dp_cyt_c = add_i(*mul_i_scalar(*alpha,1.30196985e+1),*add_i(*mul_i_scalar(*t23,6.390879e+1),*add_i(*mul_i_scalar(*t24,-1.03849005e+2),-1.5100995)))
    dr_cyt_c = add_i(*mul_i_scalar(*alpha,1.7844495),*add_i(*mul_i_scalar(*t23,6.266553e+1),*add_i(*mul_i_scalar(*t24,-1.3743354e+2),1.2107472e+1)))
    dvt_czt_c = add_i(*mul_i_scalar(*alpha,-4.211369),*add_i(*mul_i_scalar(*t23,4.775187),*add_i(*mul_i_scalar(*t24,-1.026225e+1),*add_i(*mul_i_scalar(*t25,8.399763),*add_i(*mul_i_scalar(*t26,1.378278e-1),*add_i(*mul_i(*alpha,*mul_i_scalar(*t26,4.211369)),*add_i(*mul_i(*t23,*mul_i_scalar(*t26,-4.775187)),*add_i(*mul_i(*t24,*mul_i_scalar(*t26,1.026225e+1)),*add_i(*mul_i(*t25,*mul_i_scalar(*t26,-8.399763)),-1.378278e-1)))))))))
    dalph_czt_c = add_i(*mul_i(*q,*add_i(*mul_i_scalar(*alpha,3.727436016e+3),*add_i(*mul_i_scalar(*t23,-1.1627968524e+4),*add_i(*mul_i_scalar(*t24,9.237672416e+3),-2.33888463e+2)))),*mul_i(*vt,*add_i(*mul_i_scalar(*alpha,9.550374),*add_i(*mul_i_scalar(*t23,-3.078675e+1),*add_i(*mul_i_scalar(*t24,3.3599052e+1),*add_i(*mul_i_scalar(*t26,4.211369),*add_i(*mul_i(*alpha,*mul_i_scalar(*t26,-9.550374)),*add_i(*mul_i(*t23,*mul_i_scalar(*t26,3.078675e+1)),*add_i(*mul_i(*t24,*mul_i_scalar(*t26,-3.3599052e+1)),-4.211369)))))))))
    dbeta_czt_c = mul_i(*vt,*add_i(*mul_i_scalar(*beta,2.756556e-1),*add_i(*mul_i(*alpha,*mul_i_scalar(*beta,8.422738000000001)),*add_i(*mul_i(*beta,*mul_i_scalar(*t23,-9.550374)),*add_i(*mul_i(*beta,*mul_i_scalar(*t24,2.05245e+1)),*mul_i(*beta,*mul_i_scalar(*t25,-1.6799526e+1)))))))
    dq_czt_c = add_i(*mul_i_scalar(*alpha,-2.33888463e+2),*add_i(*mul_i_scalar(*t23,1.863718008e+3),*add_i(*mul_i_scalar(*t24,-3.875989508e+3),*add_i(*mul_i_scalar(*t25,2.309418104e+3),-1.729105096e+2))))
    dalph_clt_ail = add_i(*mul_i_scalar(*alpha,1.693391395047632e-2),*add_i(*mul_i_scalar(*t23,-1.682358470714075e-2),*add_i(*mul_i(*beta,*add_i(*mul_i_scalar(*alpha,-1.307737858473359e-2),5.199074042303309e-3)),-7.110314292992219e-4)))
    dbeta_clt_ail = add_i(*mul_i_scalar(*alpha,5.199074042303309e-3),*add_i(*mul_i_scalar(*t23,-6.538689292366793e-3),5.677833564088621e-4))
    dalph_clt_rdr = add_i(*mul_i(*beta,*add_i(*mul_i_scalar(*alpha,1.576436881871492e-2),*add_i(*mul_i_scalar(*t23,-2.580662332488887e-2),-1.015398175824037e-3))),-3.827349969990885e-4)
    dbeta_clt_rdr = add_i(*mul_i_scalar(*alpha,-1.015398175824037e-3),*add_i(*mul_i_scalar(*beta,-5.514765706745539e-4),*add_i(*mul_i_scalar(*t23,7.88218440935746e-3),*add_i(*mul_i_scalar(*t24,-8.602207774962956e-3),-5.502850343942174e-5))))
    dvt_clt_c = add_i(*mul_i_scalar(*beta,-1.05853e-1),*add_i(*mul_i_scalar(*t26,1.357256e-1),*add_i(*mul_i(*alpha,*mul_i_scalar(*beta,-5.776677e-1)),*add_i(*mul_i(*alpha,*mul_i_scalar(*t26,2.172952e-1)),*add_i(*mul_i(*beta,*mul_i_scalar(*t23,-1.672435e-2)),*add_i(*mul_i(*beta,*mul_i_scalar(*t24,3.464156)),*add_i(*mul_i(*beta,*mul_i_scalar(*t25,-2.835451)),*mul_i(*t23,*mul_i_scalar(*t26,-1.098104)))))))))
    dalph_clt_c = add_i(*mul_i(*vt,*add_i(*mul_i_scalar(*beta,-5.776677e-1),*add_i(*mul_i_scalar(*t26,2.172952e-1),*add_i(*mul_i(*alpha,*mul_i_scalar(*beta,-3.34487e-2)),*add_i(*mul_i(*alpha,*mul_i_scalar(*t26,-2.196208)),*add_i(*mul_i(*beta,*mul_i_scalar(*t23,1.0392468e+1)),*mul_i(*beta,*mul_i_scalar(*t24,-1.1341804e+1)))))))),*add_i(*mul_i(*mul_i_scalar(*p, -1.0),*add_i(*mul_i_scalar(*alpha,-3.743163e+1),*add_i(*mul_i_scalar(*t23,3.3260094e+1),1.784961))),*mul_i(*mul_i_scalar(*r, -1.0),*add_i(*mul_i_scalar(*alpha,3.305892e+1),*add_i(*mul_i_scalar(*t23,-4.09503915e+2),*add_i(*mul_i_scalar(*t24,7.156032e+2),-9.101584500000001))))))
    dbeta_clt_c = mul_i(*vt,*add_i(*mul_i_scalar(*alpha,-5.776677e-1),*add_i(*mul_i_scalar(*beta,2.714512e-1),*add_i(*mul_i_scalar(*t23,-1.672435e-2),*add_i(*mul_i_scalar(*t24,3.464156),*add_i(*mul_i_scalar(*t25,-2.835451),*add_i(*mul_i(*alpha,*mul_i_scalar(*beta,4.345904e-1)),*add_i(*mul_i(*beta,*mul_i_scalar(*t23,-2.196208)),-1.05853e-1))))))))
    dp_clt_c = add_i(*mul_i_scalar(*alpha,-1.784961),*add_i(*mul_i_scalar(*t23,1.8715815e+1),*add_i(*mul_i_scalar(*t24,-1.1086698e+1),-6.190209)))
    dr_clt_c = add_i(*mul_i_scalar(*alpha,9.101584500000001),*add_i(*mul_i_scalar(*t23,-1.652946e+1),*add_i(*mul_i_scalar(*t24,1.36501305e+2),*add_i(*mul_i_scalar(*t25,-1.789008e+2),9.375655500000001e-1))))
    dalph_cmt_el = add_i(*mul_i_scalar(*alpha,1.751800234729024e-2),-0.004626, 0.0018113)
    dvt_cmt_c = add_i(*mul_i_scalar(*alpha,4.660702e-2),-2.02937e-2)
    dalph_cmt_c = add_i(*mul_i_scalar(*vt,4.660702e-2),*mul_i(*q,*add_i(*mul_i_scalar(*alpha,-4.073655952e+2),*add_i(*mul_i_scalar(*t23,3.81600879e+3),*add_i(*mul_i_scalar(*t24,-9.329923624e+3),*add_i(*mul_i_scalar(*t25,6.8252525e+3),-2.011969256e+1))))))
    dq_cmt_c = add_i(*mul_i_scalar(*alpha,-2.011969256e+1),*add_i(*mul_i_scalar(*t23,-2.036827976e+2),*add_i(*mul_i_scalar(*t24,1.27200293e+3),*add_i(*mul_i_scalar(*t25,-2.332480906e+3),*add_i(*mul_i_scalar(*alpha_cinq,1.3650505e+3),-2.93840598e+1)))))
    dalph_cnt_ail = add_i(*mul_i_scalar(*alpha,8.037391303749559e-3),*add_i(*mul_i_scalar(*t23,-1.315738796830351e-2),*add_i(*mul_i(*beta,*add_i(*mul_i_scalar(*alpha,-4.793749248795657e-2),*add_i(*mul_i_scalar(*t23,6.479964199024939e-2),6.171189274408362e-3))),*add_i(*mul_i_scalar(*beta_trois,-9.074884824305068e-3),7.46417107218781e-4))))
    dbeta_cnt_ail = add_i(*mul_i_scalar(*alpha,6.171189274408362e-3),*add_i(*mul_i_scalar(*t23,-2.396874624397829e-2),*add_i(*mul_i_scalar(*t24,2.159988066341646e-2),*add_i(*mul_i(*t26,*mul_i_scalar(*add_i(*mul_i_scalar(*alpha,9.074884824305068e-3), -2.771766111738455e-3),-3.0)),1.147317665605552e-4))))
    dalph_cnt_rdr = add_i(*mul_i_scalar(*alpha,3.505657863580298e-3),*add_i(*mul_i(*beta,*add_i(*mul_i_scalar(*alpha,-1.164998898125805e-2),3.558286521844935e-3)),-2.018612906271602e-4))
    dbeta_cnt_rdr = add_i(*mul_i_scalar(*alpha,3.558286521844935e-3),*add_i(*mul_i_scalar(*t23,-5.824994490629027e-3),4.388049209498828e-4))
    dvt_cnt_c = add_i(*mul_i_scalar(*beta,2.993363e-1),*add_i(*mul_i_scalar(*t26,-2.003125e-1),*add_i(*mul_i(*alpha,*mul_i_scalar(*beta,6.594004000000001e-2)),*add_i(*mul_i(*alpha,*mul_i_scalar(*t26,-6.233977e-2)),*add_i(*mul_i(*beta,*mul_i_scalar(*t23,-2.107885)),*add_i(*mul_i(*beta,*mul_i_scalar(*t24,8.476901e-1)),*mul_i(*t23,*mul_i_scalar(*t26,2.14142))))))))
    dalph_cnt_c = add_i(*mul_i(*r,*add_i(*mul_i_scalar(*alpha,-2.2923891e+1),-1.7513265)),*add_i(*mul_i(*p,*add_i(*mul_i_scalar(*alpha,5.778534),*add_i(*mul_i_scalar(*t23,1.80599625e+2),*add_i(*mul_i_scalar(*t24,-2.6425812e+2),-4.947369)))),*mul_i(*vt,*add_i(*mul_i_scalar(*beta,6.594004000000001e-2),*add_i(*mul_i_scalar(*mul_i_scalar(*t26, -1.0),6.233977e-2),*add_i(*mul_i(*alpha,*mul_i_scalar(*beta,-4.21577)),*add_i(*mul_i(*alpha,*mul_i_scalar(*t26,4.28284)),*mul_i(*beta,*mul_i_scalar(*t23,2.5430703)))))))))
    dbeta_cnt_c = mul_i(*vt,*add_i(*mul_i_scalar(*alpha,6.594004000000001e-2),*add_i(*mul_i_scalar(*beta,-4.00625e-1),*add_i(*mul_i_scalar(*t23,-2.107885),*add_i(*mul_i_scalar(*t24,8.476901e-1),*add_i(*mul_i(*alpha,*mul_i_scalar(*beta,-1.2467954e-1)),*add_i(*mul_i(*beta,*mul_i_scalar(*t23,4.28284)),2.993363e-1)))))))
    dp_cnt_c = add_i(*mul_i_scalar(*alpha,-4.947369),*add_i(*mul_i_scalar(*t23,2.889267),*add_i(*mul_i_scalar(*t24,6.0199875e+1),*add_i(*mul_i_scalar(*t25,-6.606453000000001e+1),4.016478e-1))))
    dr_cnt_c = add_i(*mul_i_scalar(*alpha,-1.7513265),*add_i(*mul_i_scalar(*t23,-1.14619455e+1),-5.548134))

    t10 = mul_i_scalar(*cnt_c, 1.4283e-1)
    t11 = mul_i_scalar(*cnt_ail, 1.4283e-1)
    t12 = mul_i_scalar(*cnt_rdr, 1.4283e-1)
    t13 = mul_i_scalar(*clt_c, 1.4778e-2)
    t14 = mul_i_scalar(*clt_ail, 1.4778e-2)
    t15 = mul_i_scalar(*clt_rdr, 1.4778e-2)
    t16 = mul_i_scalar(*cnt_c, 1.4778e-2)
    t17 = mul_i_scalar(*cnt_ail, 1.4778e-2)
    t18 = mul_i_scalar(*cnt_rdr, 1.4778e-2)
    t19 = mul_i_scalar(*clt_c, 9.495e-1)
    t20 = mul_i_scalar(*clt_ail, 9.495e-1)
    t21 = mul_i_scalar(*clt_rdr, 9.495e-1)
    t22 = mul_i_scalar(*q, 2.755e-2)

    t37 = mul_i_scalar(*mul_i(*cyt_c, *t6), 4.71e-1)
    t28 = mul_i(*cxt_el, *t2)
    t29 = mul_i(*czt_el, *t2)
    t30 = mul_i(*dalph_cxt_el, *t2)
    t31 = mul_i(*cxt_el, *t5)
    t32 = mul_i(*czt_el, *t5)

    t42 = add_i(*t10, *t13)
    t43 = add_i(*t11, *t14)
    t44 = add_i(*t12, *t15)
    t45 = mul_i_scalar(*mul_i(*cxt_c,*mul_i(*t2,*t3)), 4.71e-1)
    t46 = mul_i_scalar(*mul_i(*czt_c,*mul_i(*t3,*t5)), 4.71e-1)
    t47 = add_i(*t16, *t19)
    t48 = add_i(*t17, *t20)
    t49 = add_i(*t18, *t21)
    t39 = pow2_i(*t38)
    t41 = mul_i_scalar(*t31, -1.0)
    t50 = add_i(*t28, *t32)
    t53 = add_i(*t37, *add_i(*t45, *t46))

    # Vt partial derivatves
    gradF[(0,0,0)] = add_i(*mul_i(*qbar_k,*t53),*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i(*dvt_cyt_c,*mul_i_scalar(*t6,4.71e-1)),*add_i(*mul_i(*dvt_cxt_c,*mul_i(*t2,*mul_i_scalar(*t3,4.71e-1))),*mul_i(*dvt_czt_c,*mul_i(*t3,*mul_i_scalar(*t5,4.71e-1))))))))
    gradF[(0,0,1)] = add_i(*mul_i(*t3,*mul_i(*t5,*mul_i_scalar(*t9,3.217e+1))),*add_i(*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i(*dalph_cyt_c,*mul_i_scalar(*t6,4.71e-1)),*add_i(*mul_i(*mul_i_scalar(*cxt_c, -1.0),*mul_i(*t3,*mul_i_scalar(*t5,4.71e-1))),*add_i(*mul_i(*czt_c,*mul_i(*t2,*mul_i_scalar(*t3,4.71e-1))),*add_i(*mul_i(*dalph_cxt_c,*mul_i(*t2,*mul_i_scalar(*t3,4.71e-1))),*mul_i(*dalph_czt_c,*mul_i(*t3,*mul_i_scalar(*t5,4.71e-1))))))))),*mul_i(*t2,*mul_i(*t3,*mul_i(*t4,*mul_i_scalar(*t7,3.217e+1))))))
    gradF[(0,0,2)] = add_i(*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i(*cyt_c,*mul_i_scalar(*t3,4.71e-1)),*add_i(*mul_i(*dbeta_cyt_c,*mul_i_scalar(*t6,4.71e-1)),*add_i(*mul_i(*mul_i_scalar(*cxt_c, -1.0),*mul_i(*t2,*mul_i_scalar(*t6,4.71e-1))),*add_i(*mul_i(*mul_i_scalar(*czt_c, -1.0),*mul_i(*t5,*mul_i_scalar(*t6,4.71e-1))),*mul_i(*dbeta_czt_c,*mul_i(*t3,*mul_i_scalar(*t5,4.71e-1))))))))),*add_i(*mul_i(*t2,*mul_i(*t6,*mul_i_scalar(*t9,3.217e+1))),*add_i(*mul_i(*t3,*mul_i(*t7,*mul_i_scalar(*t8,3.217e+1))),*mul_i(*mul_i_scalar(*t4, -1.0),*mul_i(*t5,*mul_i(*t6,*mul_i_scalar(*t7,3.217e+1)))))))
    gradF[(0,0,3)] = add_i(*mul_i(*t4,*mul_i(*t6,*mul_i_scalar(*t7,3.217e+1))),*mul_i(*t3,*mul_i(*t5,*mul_i(*t7,*mul_i_scalar(*t8,-3.217e+1)))))
    gradF[(0,0,4)] = add_i(*mul_i(*t2,*mul_i(*t3,*mul_i_scalar(*t7,-3.217e+1))),*add_i(*mul_i(*t6,*mul_i(*t8,*mul_i_scalar(*t9,-3.217e+1))),*mul_i(*t3,*mul_i(*t4,*mul_i(*t5,*mul_i_scalar(*t9,-3.217e+1))))))
    gradF[(0,0,5)] = mul_i(*dp_cyt_c,*mul_i(*qbar_k,*mul_i(*t6,*mul_i_scalar(*vt,4.71e-1))))
    gradF[(0,0,6)] = mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i(*dq_cxt_c,*mul_i(*t2,*mul_i_scalar(*t3,4.71e-1))),*mul_i(*dq_czt_c,*mul_i(*t3,*mul_i_scalar(*t5,4.71e-1))))))
    gradF[(0,0,7)] = mul_i(*dr_cyt_c,*mul_i(*qbar_k,*mul_i(*t6,*mul_i_scalar(*vt,4.71e-1))))
    gradF[(0,0,8)] = mul_i(*dalt_qbar_k,*mul_i(*t53,*vt))

    # Alpha dot partial derivatives
    gradF[(1,0,0)] = add_i(*mul_i(*dvt_czt_c,*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t38,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*dvt_cxt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t38,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*t5, -1.0),*mul_i(*t9,*mul_i(*t36,*mul_i_scalar(*t38,3.217e+1)))),*mul_i(*mul_i_scalar(*t2, -1.0),*mul_i(*t4,*mul_i(*t7,*mul_i(*t36,*mul_i_scalar(*t38,3.217e+1))))))))
    gradF[(1,0,1)] = add_i(*mul_i(*cxt_c,*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t38,-4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*czt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t38,4.71e-1)))),*add_i(*mul_i(*dalph_czt_c,*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t38,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*dalph_cxt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t38,4.71e-1)))),*add_i(*mul_i(*p,*mul_i(*t5,*mul_i(*t6,*mul_i_scalar(*t38,1.0)))),*add_i(*mul_i(*mul_i_scalar(*r, -1.0),*mul_i(*t2,*mul_i(*t6,*mul_i_scalar(*t38,1.0)))),*add_i(*mul_i(*t2,*mul_i(*t9,*mul_i(*t35,*mul_i_scalar(*t38,3.217e+1)))),*mul_i(*mul_i_scalar(*t4, -1.0),*mul_i(*t5,*mul_i(*t7,*mul_i(*t35,*mul_i_scalar(*t38,3.217e+1))))))))))))
    gradF[(1,0,2)] = add_i(*mul_i(*p,*mul_i(*t2,*mul_i_scalar(*t39,-1.0))),*add_i(*mul_i(*mul_i_scalar(*r, -1.0),*mul_i(*t5,*mul_i_scalar(*t39,1.0))),*add_i(*mul_i(*dbeta_czt_c,*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t38,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*cxt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i(*t6,*mul_i_scalar(*t39,4.71e-1))))),*add_i(*mul_i(*czt_c,*mul_i(*qbar_k,*mul_i(*t2,*mul_i(*t6,*mul_i_scalar(*t39,4.71e-1))))),*add_i(*mul_i(*t5,*mul_i(*t6,*mul_i(*t9,*mul_i(*t35,*mul_i_scalar(*t39,3.217e+1))))),*mul_i(*t2,*mul_i(*t4,*mul_i(*t6,*mul_i(*t7,*mul_i(*t35,*mul_i_scalar(*t39,3.217e+1))))))))))))
    gradF[(1,0,3)] = mul_i(*t2,*mul_i(*t7,*mul_i(*t8,*mul_i(*t35,*mul_i_scalar(*t38,-3.217e+1)))))
    gradF[(1,0,4)] = add_i(*mul_i(*t5,*mul_i(*t7,*mul_i(*t35,*mul_i_scalar(*t38,3.217e+1)))),*mul_i(*mul_i_scalar(*t2, -1.0),*mul_i(*t4,*mul_i(*t9,*mul_i(*t35,*mul_i_scalar(*t38,3.217e+1))))))
    gradF[(1,0,5)] = mul_i(*t2,*mul_i(*t6,*mul_i_scalar(*t38,-1.0)))
    gradF[(1,0,6)] = add_i(*mul_i(*dq_czt_c,*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t38,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*dq_cxt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t38,4.71e-1)))),1.0))
    gradF[(1,0,7)] = mul_i(*t5,*mul_i(*t6,*mul_i_scalar(*t38,-1.0)))
    gradF[(1,0,8)] = add_i(*mul_i(*cxt_c,*mul_i(*dalt_qbar_k,*mul_i(*t5,*mul_i_scalar(*t38,-4.71e-1)))),*mul_i(*czt_c,*mul_i(*dalt_qbar_k,*mul_i(*t2,*mul_i_scalar(*t38,4.71e-1)))))

    # Beta dot parial derivatives
    gradF[(2,0,0)] = add_i(*mul_i(*dvt_cyt_c,*mul_i(*qbar_k,*mul_i_scalar(*t3,4.71e-1))),*add_i(*mul_i(*mul_i_scalar(*dvt_cxt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t6,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*dvt_czt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t6,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*t2, -1.0),*mul_i(*t6,*mul_i(*t9,*mul_i_scalar(*t36,3.217e+1)))),*add_i(*mul_i(*mul_i_scalar(*t3, -1.0),*mul_i(*t7,*mul_i(*t8,*mul_i_scalar(*t36,3.217e+1)))),*mul_i(*t4,*mul_i(*t5,*mul_i(*t6,*mul_i(*t7,*mul_i_scalar(*t36,3.217e+1))))))))))
    gradF[(2,0,1)] = add_i(*mul_i(*p,*mul_i_scalar(*t2,1.0)),*add_i(*mul_i(*r,*mul_i_scalar(*t5,1.0)),*add_i(*mul_i(*dalph_cyt_c,*mul_i(*qbar_k,*mul_i_scalar(*t3,4.71e-1))),*add_i(*mul_i(*cxt_c,*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t6,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*czt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t6,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*dalph_cxt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t6,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*dalph_czt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t6,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*t5, -1.0),*mul_i(*t6,*mul_i(*t9,*mul_i_scalar(*t35,3.217e+1)))),*mul_i(*mul_i_scalar(*t2, -1.0),*mul_i(*t4,*mul_i(*t6,*mul_i(*t7,*mul_i_scalar(*t35,3.217e+1)))))))))))))
    gradF[(2,0,2)] = add_i(*mul_i(*cyt_c,*mul_i(*qbar_k,*mul_i_scalar(*t6,-4.71e-1))),*add_i(*mul_i(*dbeta_cyt_c,*mul_i(*qbar_k,*mul_i_scalar(*t3,4.71e-1))),*add_i(*mul_i(*mul_i_scalar(*cxt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t3,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*czt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t3,*mul_i_scalar(*t5,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*dbeta_czt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t6,4.71e-1)))),*add_i(*mul_i(*t2,*mul_i(*t3,*mul_i(*t9,*mul_i_scalar(*t35,3.217e+1)))),*add_i(*mul_i(*mul_i_scalar(*t6, -1.0),*mul_i(*t7,*mul_i(*t8,*mul_i_scalar(*t35,3.217e+1)))),*mul_i(*mul_i_scalar(*t3, -1.0),*mul_i(*t4,*mul_i(*t5,*mul_i(*t7,*mul_i_scalar(*t35,3.217e+1))))))))))))
    gradF[(2,0,3)] = add_i(*mul_i(*t3,*mul_i(*t4,*mul_i(*t7,*mul_i_scalar(*t35,3.217e+1)))),*mul_i(*t5,*mul_i(*t6,*mul_i(*t7,*mul_i(*t8,*mul_i_scalar(*t35,3.217e+1))))))
    gradF[(2,0,4)] = add_i(*mul_i(*t2,*mul_i(*t6,*mul_i(*t7,*mul_i_scalar(*t35,3.217e+1)))),*add_i(*mul_i(*mul_i_scalar(*t3, -1.0),*mul_i(*t8,*mul_i(*t9,*mul_i_scalar(*t35,3.217e+1)))),*mul_i(*t4,*mul_i(*t5,*mul_i(*t6,*mul_i(*t9,*mul_i_scalar(*t35,3.217e+1)))))))
    gradF[(2,0,5)] = add_i(*mul_i_scalar(*t5,1.0),*mul_i(*dp_cyt_c,*mul_i(*qbar_k,*mul_i_scalar(*t3,4.71e-1))))
    gradF[(2,0,6)] = add_i(*mul_i(*dq_cxt_c,*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t6,-4.71e-1)))),*mul_i(*mul_i_scalar(*dq_czt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t6,4.71e-1)))))
    gradF[(2,0,7)] = add_i(*mul_i_scalar(*t2,-1.0),*mul_i(*dr_cyt_c,*mul_i(*qbar_k,*mul_i_scalar(*t3,4.71e-1))))
    gradF[(2,0,8)] = add_i(*mul_i(*cyt_c,*mul_i(*dalt_qbar_k,*mul_i_scalar(*t3,4.71e-1))),*add_i(*mul_i(*mul_i_scalar(*cxt_c, -1.0),*mul_i(*dalt_qbar_k,*mul_i(*t2,*mul_i_scalar(*t6,4.71e-1)))),*mul_i(*mul_i_scalar(*czt_c, -1.0),*mul_i(*dalt_qbar_k,*mul_i(*t5,*mul_i_scalar(*t6,4.71e-1))))))

    gradF[(3,0,3)] = mul_i(*t9,*mul_i(*t40,*add_i(*mul_i(*q,*t4),*mul_i(*mul_i_scalar(*r, -1.0),*t8))))
    gradF[(3,0,4)] = mul_i(*add_i(*t33, *t34), *pow2_i(*t40))
    gradF[(3,0,5)] = (1.0,1.0)
    gradF[(3,0,6)] = mul_i(*t8,*mul_i(*t9,*t40))
    gradF[(3,0,7)] = mul_i(*t4,*mul_i(*t9,*t40))

    gradF[(4,0,3)] = mul_i_scalar(*add_i(*t33, *t34), -1.0)
    gradF[(4,0,6)] = t4
    gradF[(4,0,7)] = mul_i_scalar(*t8, -1.0)

    gradF[(5,0,0)] = add_i(*mul_i(*qbar_k,*t47),*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dvt_clt_c,9.495e-1),*mul_i_scalar(*dvt_cnt_c,1.4778e-2)))))
    gradF[(5,0,1)] = mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dalph_clt_c,9.495e-1),*mul_i_scalar(*dalph_cnt_c,1.4778e-2))))
    gradF[(5,0,2)] = mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dbeta_clt_c,9.495e-1),*mul_i_scalar(*dbeta_cnt_c,1.4778e-2))))
    gradF[(5,0,5)] = add_i(*t22,*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dp_clt_c,9.495e-1),*mul_i_scalar(*dp_cnt_c,1.4778e-2)))))
    gradF[(5,0,6)] = add_i(*mul_i_scalar(*p,2.755e-2),*add_i(*mul_i_scalar(*r,-7.7e-1),2.6272e-4))
    gradF[(5,0,7)] = add_i(*mul_i_scalar(*q,-7.7e-1),*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dr_clt_c,9.495e-1),*mul_i_scalar(*dr_cnt_c,1.4778e-2)))))
    gradF[(5,0,8)] = mul_i(*dalt_qbar_k,*mul_i(*t47,*vt))

    gradF[(7,0,0)] = add_i(*mul_i(*qbar_k,*t42),*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dvt_clt_c,1.4778e-2),*mul_i_scalar(*dvt_cnt_c,1.4283e-1)))))
    gradF[(7,0,1)] = mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dalph_clt_c,1.4778e-2),*mul_i_scalar(*dalph_cnt_c,1.4283e-1))))
    gradF[(7,0,2)] = mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dbeta_clt_c,1.4778e-2),*mul_i_scalar(*dbeta_cnt_c,1.4283e-1))))
    gradF[(7,0,5)] = add_i(*mul_i_scalar(*q,-7.336e-1),*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dp_clt_c,1.4778e-2),*mul_i_scalar(*dp_cnt_c,1.4283e-1)))))
    gradF[(7,0,6)] = add_i(*mul_i_scalar(*p,-7.336e-1),*add_i(*mul_i_scalar(*r,-2.755e-2),2.5392e-3))
    gradF[(7,0,7)] = add_i(*mul_i_scalar(*t22,-1.0),*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dr_clt_c,1.4778e-2),*mul_i_scalar(*dr_cnt_c,1.4283e-1)))))
    gradF[(7,0,8)] = mul_i(*dalt_qbar_k,*mul_i(*t42,*vt))

    gradF[(6,0,0)] = add_i(*mul_i(*cmt_c,*mul_i_scalar(*qbar_k,6.085632e-2)),*mul_i(*dvt_cmt_c,*mul_i(*qbar_k,*mul_i_scalar(*vt,6.085632e-2))))
    gradF[(6,0,1)] = mul_i(*dalph_cmt_c,*mul_i(*qbar_k,*mul_i_scalar(*vt,6.085632e-2)))
    gradF[(6,0,5)] = add_i(*mul_i_scalar(*p,-3.518e-2),*mul_i_scalar(*r,9.604e-1))
    gradF[(6,0,6)] = mul_i(*dq_cmt_c,*mul_i(*qbar_k,*mul_i_scalar(*vt,6.085632e-2)))
    gradF[(6,0,7)] = add_i(*mul_i_scalar(*p,9.604e-1),*add_i(*mul_i_scalar(*r,3.518e-2),-2.8672e-3))
    gradF[(6,0,8)] = mul_i(*cmt_c,*mul_i(*dalt_qbar_k,*mul_i_scalar(*vt,6.085632e-2)))

    gradF[(8,0,0)] = add_i(*mul_i(*t2,*mul_i(*t3,*t9)),*add_i(*mul_i(*t6,*mul_i(*t7,*mul_i_scalar(*t8,-1.0))),*mul_i(*t3,*mul_i(*t4,*mul_i(*t5,*mul_i_scalar(*t7,-1.0))))))
    gradF[(8,0,1)] = mul_i_scalar(*mul_i(*vt,*add_i(*mul_i(*t3,*mul_i(*t5,*t9)),*mul_i(*t2,*mul_i(*t3,*mul_i(*t4,*mul_i_scalar(*t7,1.0)))))),-1.0)
    gradF[(8,0,2)] = mul_i(*vt,*add_i(*mul_i(*t2,*mul_i(*t6,*t9)),*add_i(*mul_i(*t3,*mul_i(*t7,*mul_i_scalar(*t8,-1.0))),*mul_i(*t4,*mul_i(*t5,*mul_i(*t6,*t7))))))
    gradF[(8,0,3)] = mul_i(*vt,*add_i(*mul_i(*t4,*mul_i(*t6,*mul_i_scalar(*t7,-1.0))),*mul_i(*t3,*mul_i(*t5,*mul_i(*t7,*t8)))))
    gradF[(8,0,4)] = mul_i(*vt,*add_i(*mul_i(*t2,*mul_i(*t3,*t7)),*add_i(*mul_i(*t6,*mul_i(*t8,*t9)),*mul_i(*t3,*mul_i(*t4,*mul_i(*t5,*t9))))))

    gradF[(9,0,9)] = (-1.0,-1.0)

    gradF[(10,0,0)] = mul_i_scalar(*add_i(*mul_i_scalar(*mul_i(*qbar_k, *add_i(*czt_c, *mul_i(*vt, *dvt_czt_c))),0.471), *mul_i_scalar(*gradF[(6,0,0)], -15.0)), inv_g)
    gradF[(10,0,1)] = mul_i_scalar(*add_i(*mul_i_scalar(*mul_i(*qbar_k,*mul_i(*vt, *dalph_czt_c)),0.471), *mul_i_scalar(*gradF[(6,0,1)], -15.0)), inv_g)
    gradF[(10,0,2)] = mul_i_scalar(*mul_i_scalar(*mul_i(*qbar_k,*mul_i(*vt, *dbeta_czt_c)),0.471), inv_g)
    gradF[(10,0,5)] = mul_i_scalar(*gradF[(6,0,5)], -15.0*inv_g)
    gradF[(10,0,6)] = mul_i_scalar(*add_i(*mul_i_scalar(*mul_i(*qbar_k,*mul_i(*vt, *dq_czt_c)),0.471), *mul_i_scalar(*gradF[(6,0,6)], -15.0)), inv_g)
    gradF[(10,0,7)] = mul_i_scalar(*gradF[(6,0,7)], -15.0*inv_g)
    gradF[(10,0,8)] = mul_i_scalar(*add_i(*mul_i_scalar(*mul_i(*dalt_qbar_k,*mul_i(*vt, *czt_c)),0.471), *mul_i_scalar(*gradF[(6,0,8)], -15.0)), inv_g)

    gradF[(11,0,1)] = sub_i(*mul_i(*r, *t2), *mul_i(*p,  *t5))
    gradF[(11,0,5)] = t2
    gradF[(11,0,7)] = t5

    gradF[(0,1,1)] = mul_i_scalar(*mul_i(*t5,*t3), -0.00157)
    gradF[(0,1,2)] = mul_i_scalar(*mul_i(*t2,*t6), -0.00157)
    gradF[(0,2,0)] = mul_i_scalar(*mul_i(*qbar_k, *t6), 7.065)
    gradF[(0,2,2)] = mul_i_scalar(*mul_i(*qbar_k,*mul_i(*vt, *t3)), 7.065)
    gradF[(0,2,8)] = mul_i_scalar(*mul_i(*dalt_qbar_k,*mul_i(*vt, *t6)), 7.065)
    gradF[(0,3,0)] = mul_i_scalar(*mul_i(*qbar_k,*mul_i(*q, *t3)), 2.66586)
    gradF[(0,3,2)] = mul_i_scalar(*mul_i(*qbar_k,*mul_i(*vt, *mul_i(*q, *t6))), -2.66586)
    gradF[(0,3,6)] = mul_i_scalar(*mul_i(*qbar_k,*mul_i(*vt, *t3)), 2.66586)
    gradF[(0,3,8)] = mul_i_scalar(*mul_i(*dalt_qbar_k,*mul_i(*vt, *mul_i(*q, *t3))), 2.66586)
    gradF[(1,1,0)] = mul_i_scalar(*mul_i(*t5,*mul_i(*t38, *t36)), 0.00157)
    gradF[(1,1,1)] = mul_i_scalar(*mul_i(*t2,*mul_i(*t38, *t35)), -0.00157)
    gradF[(1,1,2)] = mul_i_scalar(*mul_i(*t5,*mul_i(*t39, *mul_i(*t6, *t35))), -0.00157)
    gradF[(1,2,2)] = mul_i_scalar(*mul_i(*qbar_k,*mul_i(*t39,*mul_i(*q, *t6))), -2.66586)
    gradF[(1,2,6)] = mul_i_scalar(*mul_i(*qbar_k, *t38), -2.66586)
    gradF[(1,2,8)] = mul_i_scalar(*mul_i(*dalt_qbar_k,*mul_i(*t38,*q)), -2.66586)
    gradF[(2,1,0)] = mul_i_scalar(*mul_i(*t2, *mul_i(*t6,*t36)),0.00157)
    gradF[(2,1,1)] = mul_i_scalar(*mul_i(*t5, *mul_i(*t6,*t35)),0.00157)
    gradF[(2,1,2)] = mul_i_scalar(*mul_i(*t2, *mul_i(*t3,*t35)),-0.00157)
    gradF[(2,2,2)] = mul_i_scalar(*mul_i(*qbar_k, *t6) ,-7.065)
    gradF[(2,2,8)] = mul_i_scalar(*mul_i(*dalt_qbar_k, *t3) ,7.065)
    gradF[(2,3,2)] = mul_i_scalar(*mul_i(*qbar_k,*mul_i(*q, *t3)), -2.66586)
    gradF[(2,3,6)] = mul_i_scalar(*mul_i(*qbar_k,*t6), -2.66586)
    gradF[(2,3,8)] = mul_i_scalar(*mul_i(*dalt_qbar_k,*mul_i(*q, *t6)), -2.66586)
    gradF[(5,1,0)] = qbar_k
    gradF[(5,1,8)] = mul_i(*dalt_qbar_k, *vt)
    gradF[(6,1,0)] = mul_i_scalar(*mul_i(*qbar_k,  *q), 0.34444677120000002348783373420926)
    gradF[(6,1,6)] = mul_i_scalar(*mul_i(*qbar_k, *vt), 0.34444677120000002348783373420926)
    gradF[(6,1,8)] = mul_i_scalar(*mul_i(*dalt_qbar_k, *mul_i(*vt, *q)), 0.34444677120000002348783373420926)
    gradF[(7,1,0)] = qbar_k
    gradF[(7,1,8)] = mul_i(*dalt_qbar_k, *vt)
    gradF[(10,1,0)] = mul_i_scalar(*mul_i(*qbar_k,*q), inv_g)
    gradF[(10,1,6)] = mul_i_scalar(*mul_i(*qbar_k,*vt), inv_g)
    gradF[(10,1,8)] = mul_i_scalar(*mul_i(*dalt_qbar_k,*mul_i(*vt,*q)), inv_g)

    t51 = add_i(*t29, *t41)
    t52 = add_i(*t30, *t51)
    gradG[(0,1,0,0)] = mul_i(*qbar_k,*mul_i(*t3,*mul_i(*t50,*mul_i_scalar(*vt,9.42e-1))))
    gradG[(0,1,0,1)] = mul_i(*qbar_k,*mul_i(*t3,*mul_i(*t27,*mul_i_scalar(*t52,4.71e-1))))
    gradG[(0,1,0,2)] = mul_i(*qbar_k,*mul_i(*t6,*mul_i(*t27,*mul_i_scalar(*t50,-4.71e-1))))
    gradG[(0,1,0,8)] = mul_i(*dalt_qbar_k,*mul_i(*t3,*mul_i(*t27,*mul_i_scalar(*t50,4.71e-1))))

    gradG[(0,2,0,0)] = mul_i(*cyt_ail,*mul_i(*qbar_k,*mul_i(*t6,*mul_i_scalar(*vt,9.42e-1))))
    gradG[(0,2,0,2)] = mul_i(*cyt_ail,*mul_i(*qbar_k,*mul_i(*t3,*mul_i_scalar(*t27,4.71e-1))))
    gradG[(0,2,0,8)] = mul_i(*cyt_ail,*mul_i(*dalt_qbar_k,*mul_i(*t6,*mul_i_scalar(*t27,4.71e-1))))

    gradG[(0,3,0,0)] = mul_i(*cyt_rdr,*mul_i(*qbar_k,*mul_i(*t6,*mul_i_scalar(*vt,9.42e-1))))
    gradG[(0,3,0,2)] = mul_i(*cyt_rdr,*mul_i(*qbar_k,*mul_i(*t3,*mul_i_scalar(*t27,4.71e-1))))
    gradG[(0,3,0,8)] = mul_i(*cyt_rdr,*mul_i(*dalt_qbar_k,*mul_i(*t6,*mul_i_scalar(*t27,4.71e-1))))

    gradG[(1,1,0,0)] = mul_i(*qbar_k,*mul_i(*t38,*mul_i_scalar(*t51,4.71e-1)))
    gradG[(1,1,0,1)] = mul_i(*qbar_k,*mul_i(*t38,*mul_i(*vt,*mul_i_scalar(*add_i(*t50,*mul_i(*dalph_cxt_el,*t5)),-4.71e-1))))
    gradG[(1,1,0,2)] = mul_i(*qbar_k,*mul_i(*t6,*mul_i(*t39,*mul_i(*t51,*mul_i_scalar(*vt,4.71e-1)))))
    gradG[(1,1,0,8)] = mul_i(*dalt_qbar_k,*mul_i(*t38,*mul_i(*t51,*mul_i_scalar(*vt,4.71e-1))))

    gradG[(2,1,0,0)] = mul_i(*qbar_k,*mul_i(*t6,*mul_i_scalar(*t50,-4.71e-1)))
    gradG[(2,1,0,1)] = mul_i(*qbar_k,*mul_i(*t6,*mul_i(*t52,*mul_i_scalar(*vt,-4.71e-1))))
    gradG[(2,1,0,2)] = mul_i(*qbar_k,*mul_i(*t3,*mul_i(*t50,*mul_i_scalar(*vt,-4.71e-1))))
    gradG[(2,1,0,8)] = mul_i(*dalt_qbar_k,*mul_i(*t6,*mul_i(*t50,*mul_i_scalar(*vt,-4.71e-1))))

    gradG[(2,2,0,0)] = mul_i(*cyt_ail,*mul_i(*qbar_k,*mul_i_scalar(*t3,4.71e-1)))
    gradG[(2,2,0,2)] = mul_i(*cyt_ail,*mul_i(*qbar_k,*mul_i(*t6,*mul_i_scalar(*vt,-4.71e-1))))
    gradG[(2,2,0,8)] = mul_i(*cyt_ail,*mul_i(*dalt_qbar_k,*mul_i(*t3,*mul_i_scalar(*vt,4.71e-1))))

    gradG[(2,3,0,0)] = mul_i(*cyt_rdr,*mul_i(*qbar_k,*mul_i_scalar(*t3,4.71e-1)))
    gradG[(2,3,0,2)] = mul_i(*cyt_rdr,*mul_i(*qbar_k,*mul_i(*t6,*mul_i_scalar(*vt,-4.71e-1))))
    gradG[(2,3,0,8)] = mul_i(*cyt_rdr,*mul_i(*dalt_qbar_k,*mul_i(*t3,*mul_i_scalar(*vt,4.71e-1))))

    gradG[(5,2,0,0)] = mul_i(*qbar_k,*mul_i(*t48,*mul_i_scalar(*vt,2.0)))
    gradG[(5,2,0,1)] = mul_i(*qbar_k,*mul_i(*t27,*add_i(*mul_i_scalar(*dalph_clt_ail,9.495e-1),*mul_i_scalar(*dalph_cnt_ail,1.4778e-2))))
    gradG[(5,2,0,2)] = mul_i(*qbar_k,*mul_i(*t27,*add_i(*mul_i_scalar(*dbeta_clt_ail,9.495e-1),*mul_i_scalar(*dbeta_cnt_ail,1.4778e-2))))
    gradG[(5,2,0,3)] = mul_i(*dalt_qbar_k,*mul_i(*t27,*t48))

    gradG[(5,3,0,0)] = mul_i(*qbar_k,*mul_i(*t49,*mul_i_scalar(*vt,2.0)))
    gradG[(5,3,0,1)] = mul_i(*qbar_k,*mul_i(*t27,*add_i(*mul_i_scalar(*dalph_clt_rdr,9.495e-1),*mul_i_scalar(*dalph_cnt_rdr,1.4778e-2))))
    gradG[(5,3,0,2)] = mul_i(*qbar_k,*mul_i(*t27,*add_i(*mul_i_scalar(*dbeta_clt_rdr,9.495e-1),*mul_i_scalar(*dbeta_cnt_rdr,1.4778e-2))))
    gradG[(5,3,0,3)] = mul_i(*dalt_qbar_k,*mul_i(*t27,*t49))

    gradG[(6,1,0,0)] = mul_i(*cmt_el,*mul_i(*qbar_k,*mul_i_scalar(*vt,1.2171264e-1)))
    gradG[(6,1,0,1)] = mul_i(*dalph_cmt_el,*mul_i(*qbar_k,*mul_i_scalar(*t27,6.085632e-2)))
    gradG[(6,1,0,8)] = mul_i(*cmt_el,*mul_i(*dalt_qbar_k,*mul_i_scalar(*t27,6.085632e-2)))

    gradG[(7,2,0,0)] = mul_i(*qbar_k,*mul_i(*t43,*mul_i_scalar(*vt,2.0)))
    gradG[(7,2,0,1)] = mul_i(*qbar_k,*mul_i(*t27,*add_i(*mul_i_scalar(*dalph_clt_ail,1.4778e-2),*mul_i_scalar(*dalph_cnt_ail,1.4283e-1))))
    gradG[(7,2,0,2)] = mul_i(*qbar_k,*mul_i(*t27,*add_i(*mul_i_scalar(*dbeta_clt_ail,1.4778e-2),*mul_i_scalar(*dbeta_cnt_ail,1.4283e-1))))
    gradG[(7,2,0,3)] = mul_i(*dalt_qbar_k,*mul_i(*t27,*t43))

    gradG[(7,3,0,0)] = mul_i(*qbar_k,*mul_i(*t44,*mul_i_scalar(*vt,2.0)))
    gradG[(7,3,0,1)] = mul_i(*qbar_k,*mul_i(*t27,*add_i(*mul_i_scalar(*dalph_clt_rdr,1.4778e-2),*mul_i_scalar(*dalph_cnt_rdr,1.4283e-1))))
    gradG[(7,3,0,2)] = mul_i(*qbar_k,*mul_i(*t27,*add_i(*mul_i_scalar(*dbeta_clt_rdr,1.4778e-2),*mul_i_scalar(*dbeta_cnt_rdr,1.4283e-1))))
    gradG[(7,3,0,3)] = mul_i(*dalt_qbar_k,*mul_i(*t27,*t44))

    gradG[(10,1,0,0)] = mul_i_scalar(*add_i(*mul_i_scalar(*mul_i(*qbar_k,*mul_i(*vt, *czt_el)),0.942), *mul_i_scalar(*gradG[(6,1,0,0)], -15.0)), inv_g)
    gradG[(10,1,0,1)] = mul_i_scalar(*gradG[(6,1,0,1)], -15.0 * inv_g)
    gradG[(10,1,0,8)] = mul_i_scalar(*add_i(*mul_i_scalar(*mul_i(*dalt_qbar_k,*mul_i(*t27, *czt_el)),0.471), *mul_i_scalar(*gradG[(6,1,0,8)], -15.0)), inv_g)

    return res_f, res_G

# @jit(nopython=True, parallel=False, fastmath=True, cache=True)
# def known_grad_f16(x_lb, x_ub):
#     # Output of the known part of f for the f16
#     res_f = dict()
#     res_G = dict()

#     # Get the states of the f16
#     vt = x_lb[0], x_ub[0]
#     alpha = x_lb[1], x_ub[1]
#     beta = x_lb[2], x_ub[2]
#     phi = x_lb[3], x_ub[3]
#     theta = x_lb[4], x_ub[4]
#     p = x_lb[5], x_ub[5]
#     q = x_lb[6], x_ub[6]
#     r = x_lb[7], x_ub[7]
#     alt = x_lb[8], x_ub[8]
#     power = x_lb[9], x_ub[9]
#     # int_Nz = x_lb[10], x_ub[10]

#     qbar_k = mul_i_scalar(*powr_i(1.0-0.703e-5*alt[1],1.0-0.703e-5*alt[0], 4.14) , 0.5 * 2.377e-3)
#     dalt_qbar_k = mul_i_scalar(*powr_i(1.0-0.703e-5*alt[1],1.0-0.703e-5*alt[0], 3.14), -0.703e-5 * 4.14 * 2.377e-3 * 0.5)

#     t2 = cos_i(*alpha)
#     t3 = cos_i(*beta)
#     t4 = cos_i(*phi)
#     t5 = sin_i(*alpha)
#     t6 = sin_i(*beta)
#     t7 = cos_i(*theta)
#     t8 = sin_i(*phi)
#     t9 = sin_i(*theta)
#     t23 = pow2_i(*alpha) # alpha^2
#     t24 = mul_i(*t23, *alpha) # alpha^3
#     t25 = pow2_i(*t23) # alpha^4
#     t26 = pow2_i(*beta) # beta^2
#     beta_trois = mul_i(*t26, *beta)

#     t33 = mul_i(*r, *t4)
#     t34 = mul_i(*q, *t8)
#     t27 = pow2_i(*vt) # vt^2
#     t35 = inv_i(*vt)
#     alpha_cinq = mul_i(*t25, *alpha)
#     t36 = inv_i(*t27)
#     t38 = inv_i(*t3)
#     t40 = inv_i(*t7)

#     dalph_cxt_el = (-3.596257905051324e-3, -3.596257905051324e-3)
#     dvt_cxt_c = add_i(*mul_i_scalar(*alpha,2.136104e-1),*add_i(*mul_i_scalar(*t23,6.988016e-1),*add_i(*mul_i_scalar(*t24,-9.035381e-1),-1.943367e-2)))
#     dalph_cxt_c = add_i(*mul_i(*q,*add_i(*mul_i_scalar(*alpha,1.280402936e+2),*add_i(*mul_i_scalar(*mul_i_scalar(*t23, -1.0),1.2604187778e+3),*add_i(*mul_i_scalar(*t24,1.3755556864e+3),4.892858882e+1)))),*mul_i(*vt,*add_i(*mul_i_scalar(*alpha,1.3976032),*add_i(*mul_i_scalar(*t23,-2.7106143),2.136104e-1))))
#     dq_cxt_c = add_i(*mul_i_scalar(*alpha,4.892858882e+1),*add_i(*mul_i_scalar(*t23,6.40201468e+1),*add_i(*mul_i_scalar(*t24,-4.201395926e+2),*add_i(*mul_i_scalar(*t25,3.438889216e+2),2.735694778))))
#     dvt_cyt_c = mul_i(*beta, -1.145916)
#     dalph_cyt_c = add_i(*mul_i(*r,*add_i(*mul_i_scalar(*alpha,1.2533106e+2),*add_i(*mul_i_scalar(*t23,-4.1230062e+2),1.7844495))),*mul_i(*p,*add_i(*mul_i_scalar(*alpha,1.2781758e+2),*add_i(*mul_i_scalar(*t23,-3.11547015e+2),1.30196985e+1))))
#     dbeta_cyt_c = mul_i(*vt, -1.145916)
#     dp_cyt_c = add_i(*mul_i_scalar(*alpha,1.30196985e+1),*add_i(*mul_i_scalar(*t23,6.390879e+1),*add_i(*mul_i_scalar(*t24,-1.03849005e+2),-1.5100995)))
#     dr_cyt_c = add_i(*mul_i_scalar(*alpha,1.7844495),*add_i(*mul_i_scalar(*t23,6.266553e+1),*add_i(*mul_i_scalar(*t24,-1.3743354e+2),1.2107472e+1)))
#     dvt_czt_c = add_i(*mul_i_scalar(*alpha,-4.211369),*add_i(*mul_i_scalar(*t23,4.775187),*add_i(*mul_i_scalar(*t24,-1.026225e+1),*add_i(*mul_i_scalar(*t25,8.399763),*add_i(*mul_i_scalar(*t26,1.378278e-1),*add_i(*mul_i(*alpha,*mul_i_scalar(*t26,4.211369)),*add_i(*mul_i(*t23,*mul_i_scalar(*t26,-4.775187)),*add_i(*mul_i(*t24,*mul_i_scalar(*t26,1.026225e+1)),*add_i(*mul_i(*t25,*mul_i_scalar(*t26,-8.399763)),-1.378278e-1)))))))))
#     dalph_czt_c = add_i(*mul_i(*q,*add_i(*mul_i_scalar(*alpha,3.727436016e+3),*add_i(*mul_i_scalar(*t23,-1.1627968524e+4),*add_i(*mul_i_scalar(*t24,9.237672416e+3),-2.33888463e+2)))),*mul_i(*vt,*add_i(*mul_i_scalar(*alpha,9.550374),*add_i(*mul_i_scalar(*t23,-3.078675e+1),*add_i(*mul_i_scalar(*t24,3.3599052e+1),*add_i(*mul_i_scalar(*t26,4.211369),*add_i(*mul_i(*alpha,*mul_i_scalar(*t26,-9.550374)),*add_i(*mul_i(*t23,*mul_i_scalar(*t26,3.078675e+1)),*add_i(*mul_i(*t24,*mul_i_scalar(*t26,-3.3599052e+1)),-4.211369)))))))))
#     dbeta_czt_c = mul_i(*vt,*add_i(*mul_i_scalar(*beta,2.756556e-1),*add_i(*mul_i(*alpha,*mul_i_scalar(*beta,8.422738000000001)),*add_i(*mul_i(*beta,*mul_i_scalar(*t23,-9.550374)),*add_i(*mul_i(*beta,*mul_i_scalar(*t24,2.05245e+1)),*mul_i(*beta,*mul_i_scalar(*t25,-1.6799526e+1)))))))
#     dq_czt_c = alpha.*-2.33888463e+2+t23.*1.863718008e+3+t24.*-3.875989508e+3+t25.*2.309418104e+3+-1.729105096e+2
#     dalph_clt_ail = add_i(*mul_i_scalar(*alpha,1.693391395047632e-2),*add_i(*mul_i_scalar(*t23,-1.682358470714075e-2),*add_i(*mul_i(*beta,*add_i(*mul_i_scalar(*alpha,-1.307737858473359e-2),5.199074042303309e-3)),-7.110314292992219e-4)))
#     dbeta_clt_ail = add_i(*mul_i_scalar(*alpha,5.199074042303309e-3),*add_i(*mul_i_scalar(*t23,-6.538689292366793e-3),5.677833564088621e-4))
#     dalph_clt_rdr = add_i(*mul_i(*beta,*add_i(*mul_i_scalar(*alpha,1.576436881871492e-2),*add_i(*mul_i_scalar(*t23,-2.580662332488887e-2),-1.015398175824037e-3))),-3.827349969990885e-4)
#     dbeta_clt_rdr = add_i(*mul_i_scalar(*alpha,-1.015398175824037e-3),*add_i(*mul_i_scalar(*beta,-5.514765706745539e-4),*add_i(*mul_i_scalar(*t23,7.88218440935746e-3),*add_i(*mul_i_scalar(*t24,-8.602207774962956e-3),-5.502850343942174e-5))))
#     dvt_clt_c = add_i(*mul_i_scalar(*beta,-1.05853e-1),*add_i(*mul_i_scalar(*t26,1.357256e-1),*add_i(*mul_i(*alpha,*mul_i_scalar(*beta,-5.776677e-1)),*add_i(*mul_i(*alpha,*mul_i_scalar(*t26,2.172952e-1)),*add_i(*mul_i(*beta,*mul_i_scalar(*t23,-1.672435e-2)),*add_i(*mul_i(*beta,*mul_i_scalar(*t24,3.464156)),*add_i(*mul_i(*beta,*mul_i_scalar(*t25,-2.835451)),*mul_i(*t23,*mul_i_scalar(*t26,-1.098104)))))))))
#     dalph_clt_c = add_i(*mul_i(*vt,*add_i(*mul_i_scalar(*beta,-5.776677e-1),*add_i(*mul_i_scalar(*t26,2.172952e-1),*add_i(*mul_i(*alpha,*mul_i_scalar(*beta,-3.34487e-2)),*add_i(*mul_i(*alpha,*mul_i_scalar(*t26,-2.196208)),*add_i(*mul_i(*beta,*mul_i_scalar(*t23,1.0392468e+1)),*mul_i(*beta,*mul_i_scalar(*t24,-1.1341804e+1)))))))),*add_i(*mul_i(*mul_i_scalar(*p, -1.0),*add_i(*mul_i_scalar(*alpha,-3.743163e+1),*add_i(*mul_i_scalar(*t23,3.3260094e+1),1.784961))),*mul_i(*mul_i_scalar(*r, -1.0),*add_i(*mul_i_scalar(*alpha,3.305892e+1),*add_i(*mul_i_scalar(*t23,-4.09503915e+2),*add_i(*mul_i_scalar(*t24,7.156032e+2),-9.101584500000001))))))
#     dbeta_clt_c = mul_i(*vt,*add_i(*mul_i_scalar(*alpha,-5.776677e-1),*add_i(*mul_i_scalar(*beta,2.714512e-1),*add_i(*mul_i_scalar(*t23,-1.672435e-2),*add_i(*mul_i_scalar(*t24,3.464156),*add_i(*mul_i_scalar(*t25,-2.835451),*add_i(*mul_i(*alpha,*mul_i_scalar(*beta,4.345904e-1)),*add_i(*mul_i(*beta,*mul_i_scalar(*t23,-2.196208)),-1.05853e-1))))))))
#     dp_clt_c = add_i(*mul_i_scalar(*alpha,-1.784961),*add_i(*mul_i_scalar(*t23,1.8715815e+1),*add_i(*mul_i_scalar(*t24,-1.1086698e+1),-6.190209)))
#     dr_clt_c = add_i(*mul_i_scalar(*alpha,9.101584500000001),*add_i(*mul_i_scalar(*t23,-1.652946e+1),*add_i(*mul_i_scalar(*t24,1.36501305e+2),*add_i(*mul_i_scalar(*t25,-1.789008e+2),9.375655500000001e-1))))
#     dalph_cmt_el = add_i(*mul_i_scalar(*alpha,1.751800234729024e-2),*add_i(*mul_i_scalar(*el,1.287421659820075e-4),-1.407254961625748e-3))
#     dvt_cmt_c = add_i(*mul_i_scalar(*alpha,4.660702e-2),-2.02937e-2)
#     dalph_cmt_c = add_i(*mul_i_scalar(*vt,4.660702e-2),*mul_i(*q,*add_i(*mul_i_scalar(*alpha,-4.073655952e+2),*add_i(*mul_i_scalar(*t23,3.81600879e+3),*add_i(*mul_i_scalar(*t24,-9.329923624e+3),*add_i(*mul_i_scalar(*t25,6.8252525e+3),-2.011969256e+1))))))
#     dq_cmt_c = add_i(*mul_i_scalar(*alpha,-2.011969256e+1),*add_i(*mul_i_scalar(*t23,-2.036827976e+2),*add_i(*mul_i_scalar(*t24,1.27200293e+3),*add_i(*mul_i_scalar(*t25,-2.332480906e+3),*add_i(*mul_i_scalar(*alpha_cinq,1.3650505e+3),-2.93840598e+1)))))
#     dalph_cnt_ail = add_i(*mul_i_scalar(*alpha,8.037391303749559e-3),*add_i(*mul_i_scalar(*t23,-1.315738796830351e-2),*add_i(*mul_i(*beta,*add_i(*mul_i_scalar(*alpha,-4.793749248795657e-2),*add_i(*mul_i_scalar(*t23,6.479964199024939e-2),6.171189274408362e-3))),*add_i(*mul_i_scalar(*beta_trois,-9.074884824305068e-3),7.46417107218781e-4))))
#     dbeta_cnt_ail = add_i(*mul_i_scalar(*alpha,6.171189274408362e-3),*add_i(*mul_i_scalar(*t23,-2.396874624397829e-2),*add_i(*mul_i_scalar(*t24,2.159988066341646e-2),*add_i(*mul_i(*t26,*mul_i_scalar(*add_i(*mul_i_scalar(*alpha,9.074884824305068e-3),*mul_i_scalar(*2.771766111738455e-3, -1.0)),-3.0)),1.147317665605552e-4))))
#     dalph_cnt_rdr = alpha.*3.505657863580298e-3+beta.*(alpha.*-1.164998898125805e-2+3.558286521844935e-3)+-2.018612906271602e-4
#     dbeta_cnt_rdr = add_i(*mul_i_scalar(*alpha,3.558286521844935e-3),*add_i(*mul_i_scalar(*t23,-5.824994490629027e-3),4.388049209498828e-4))
#     dvt_cnt_c = add_i(*mul_i_scalar(*beta,2.993363e-1),*add_i(*mul_i_scalar(*t26,-2.003125e-1),*add_i(*mul_i(*alpha,*mul_i_scalar(*beta,6.594004000000001e-2)),*add_i(*mul_i(*alpha,*mul_i_scalar(*t26,-6.233977e-2)),*add_i(*mul_i(*beta,*mul_i_scalar(*t23,-2.107885)),*add_i(*mul_i(*beta,*mul_i_scalar(*t24,8.476901e-1)),*mul_i(*t23,*mul_i_scalar(*t26,2.14142))))))))
#     dalph_cnt_c = add_i(*mul_i(*r,*add_i(*mul_i_scalar(*alpha,-2.2923891e+1),-1.7513265)),*add_i(*mul_i(*p,*add_i(*mul_i_scalar(*alpha,5.778534),*add_i(*mul_i_scalar(*t23,1.80599625e+2),*add_i(*mul_i_scalar(*t24,-2.6425812e+2),-4.947369)))),*mul_i(*vt,*add_i(*mul_i_scalar(*beta,6.594004000000001e-2),*add_i(*mul_i_scalar(*mul_i_scalar(*t26, -1.0),6.233977e-2),*add_i(*mul_i(*alpha,*mul_i_scalar(*beta,-4.21577)),*add_i(*mul_i(*alpha,*mul_i_scalar(*t26,4.28284)),*mul_i(*beta,*mul_i_scalar(*t23,2.5430703)))))))))
#     dbeta_cnt_c = mul_i(*vt,*add_i(*mul_i_scalar(*alpha,6.594004000000001e-2),*add_i(*mul_i_scalar(*beta,-4.00625e-1),*add_i(*mul_i_scalar(*t23,-2.107885),*add_i(*mul_i_scalar(*t24,8.476901e-1),*add_i(*mul_i(*alpha,*mul_i_scalar(*beta,-1.2467954e-1)),*add_i(*mul_i(*beta,*mul_i_scalar(*t23,4.28284)),2.993363e-1)))))))
#     dp_cnt_c = add_i(*mul_i_scalar(*alpha,-4.947369),*add_i(*mul_i_scalar(*t23,2.889267),*add_i(*mul_i_scalar(*t24,6.0199875e+1),*add_i(*mul_i_scalar(*t25,-6.606453000000001e+1),4.016478e-1))))
#     dr_cnt_c = add_i(*mul_i_scalar(*alpha,-1.7513265),*add_i(*mul_i_scalar(*t23,-1.14619455e+1),-5.548134))


#     t10 = mul_i_scalar(*cnt_c, 1.4283e-1)
#     t11 = mul_i_scalar(*cnt_ail, 1.4283e-1)
#     t12 = mul_i_scalar(*cnt_rdr, 1.4283e-1)
#     t13 = mul_i_scalar(*clt_c, 1.4778e-2)
#     t14 = mul_i_scalar(*clt_ail, 1.4778e-2)
#     t15 = mul_i_scalar(*clt_rdr, 1.4778e-2)
#     t16 = mul_i_scalar(*cnt_c, 1.4778e-2)
#     t17 = mul_i_scalar(*cnt_ail, 1.4778e-2)
#     t18 = mul_i_scalar(*cnt_rdr, 1.4778e-2)
#     t19 = mul_i_scalar(*clt_c, 9.495e-1)
#     t20 = mul_i_scalar(*clt_ail, 9.495e-1)
#     t21 = mul_i_scalar(*clt_rdr, 9.495e-1)
#     t22 = mul_i_scalar(*q, 2.755e-2)

#     t37 = mul_i_scalar(*mul_i(*cyt_c, *t6), 4.71e-1)
#     t28 = mul_i(*cxt_el, *t2)
#     t29 = mul_i(*czt_el, *t2)
#     t30 = mul_i(*dalph_cxt_el, *t2)
#     t31 = mul_i(*cxt_el, *t5)
#     t32 = mul_i(*czt_el, *t5)

#     t42 = add_i(*t10, *t13)
#     t43 = add_i(*t11, *t14)
#     t44 = add_i(*t12, *t15)
#     t45 = mul_i_scalar(*mul_i(*cxt_c,*mul_i(*t2,*t3)), 4.71e-1)
#     t46 = mul_i_scalar(*mul_i(*czt_c,*mul_i(*t3,*t5)), 4.71e-1)
#     t47 = add_i(*t16, *t19)
#     t48 = add_i(*t17, *t20)
#     t49 = add_i(*t18, *t21)
#     t39 = pow2_i(*t38)
#     t41 = mul_i_scalar(*t31, -1.0)
#     t50 = add_i(*t28, *t32)
#     t53 = add_i(*t37, *add_i(*t45, *t46))

#     # Vt partial derivatves
#     res_f[(0,0,0)] = add_i(*mul_i(*qbar_k,*t53),*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i(*dvt_cyt_c,*mul_i_scalar(*t6,4.71e-1)),*add_i(*mul_i(*dvt_cxt_c,*mul_i(*t2,*mul_i_scalar(*t3,4.71e-1))),*mul_i(*dvt_czt_c,*mul_i(*t3,*mul_i_scalar(*t5,4.71e-1))))))))
#     res_f[(0,0,1)] = add_i(*mul_i(*t3,*mul_i(*t5,*mul_i_scalar(*t9,3.217e+1))),*add_i(*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i(*dalph_cyt_c,*mul_i_scalar(*t6,4.71e-1)),*add_i(*mul_i(*mul_i_scalar(*cxt_c, -1.0),*mul_i(*t3,*mul_i_scalar(*t5,4.71e-1))),*add_i(*mul_i(*czt_c,*mul_i(*t2,*mul_i_scalar(*t3,4.71e-1))),*add_i(*mul_i(*dalph_cxt_c,*mul_i(*t2,*mul_i_scalar(*t3,4.71e-1))),*mul_i(*dalph_czt_c,*mul_i(*t3,*mul_i_scalar(*t5,4.71e-1))))))))),*mul_i(*t2,*mul_i(*t3,*mul_i(*t4,*mul_i_scalar(*t7,3.217e+1))))))
#     res_f[(0,0,2)] = add_i(*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i(*cyt_c,*mul_i_scalar(*t3,4.71e-1)),*add_i(*mul_i(*dbeta_cyt_c,*mul_i_scalar(*t6,4.71e-1)),*add_i(*mul_i(*mul_i_scalar(*cxt_c, -1.0),*mul_i(*t2,*mul_i_scalar(*t6,4.71e-1))),*add_i(*mul_i(*mul_i_scalar(*czt_c, -1.0),*mul_i(*t5,*mul_i_scalar(*t6,4.71e-1))),*mul_i(*dbeta_czt_c,*mul_i(*t3,*mul_i_scalar(*t5,4.71e-1))))))))),*add_i(*mul_i(*t2,*mul_i(*t6,*mul_i_scalar(*t9,3.217e+1))),*add_i(*mul_i(*t3,*mul_i(*t7,*mul_i_scalar(*t8,3.217e+1))),*mul_i(*mul_i_scalar(*t4, -1.0),*mul_i(*t5,*mul_i(*t6,*mul_i_scalar(*t7,3.217e+1)))))))
#     res_f[(0,0,3)] = add_i(*mul_i(*t4,*mul_i(*t6,*mul_i_scalar(*t7,3.217e+1))),*mul_i(*t3,*mul_i(*t5,*mul_i(*t7,*mul_i_scalar(*t8,-3.217e+1)))))
#     res_f[(0,0,4)] = add_i(*mul_i(*t2,*mul_i(*t3,*mul_i_scalar(*t7,-3.217e+1))),*add_i(*mul_i(*t6,*mul_i(*t8,*mul_i_scalar(*t9,-3.217e+1))),*mul_i(*t3,*mul_i(*t4,*mul_i(*t5,*mul_i_scalar(*t9,-3.217e+1))))))
#     res_f[(0,0,5)] = mul_i(*dp_cyt_c,*mul_i(*qbar_k,*mul_i(*t6,*mul_i_scalar(*vt,4.71e-1))))
#     res_f[(0,0,6)] = mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i(*dq_cxt_c,*mul_i(*t2,*mul_i_scalar(*t3,4.71e-1))),*mul_i(*dq_czt_c,*mul_i(*t3,*mul_i_scalar(*t5,4.71e-1))))))
#     res_f[(0,0,7)] = mul_i(*dr_cyt_c,*mul_i(*qbar_k,*mul_i(*t6,*mul_i_scalar(*vt,4.71e-1))))
#     res_f[(0,0,8)] = mul_i(*dalt_qbar_k,*mul_i(*t53,*vt))

#     # Alpha dot partial derivatives
#     res_f[(1,0,0)] = add_i(*mul_i(*dvt_czt_c,*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t38,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*dvt_cxt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t38,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*t5, -1.0),*mul_i(*t9,*mul_i(*t36,*mul_i_scalar(*t38,3.217e+1)))),*mul_i(*mul_i_scalar(*t2, -1.0),*mul_i(*t4,*mul_i(*t7,*mul_i(*t36,*mul_i_scalar(*t38,3.217e+1))))))))
#     res_f[(1,0,1)] = add_i(*mul_i(*cxt_c,*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t38,-4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*czt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t38,4.71e-1)))),*add_i(*mul_i(*dalph_czt_c,*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t38,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*dalph_cxt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t38,4.71e-1)))),*add_i(*mul_i(*p,*mul_i(*t5,*mul_i(*t6,*mul_i_scalar(*t38,1.0)))),*add_i(*mul_i(*mul_i_scalar(*r, -1.0),*mul_i(*t2,*mul_i(*t6,*mul_i_scalar(*t38,1.0)))),*add_i(*mul_i(*t2,*mul_i(*t9,*mul_i(*t35,*mul_i_scalar(*t38,3.217e+1)))),*mul_i(*mul_i_scalar(*t4, -1.0),*mul_i(*t5,*mul_i(*t7,*mul_i(*t35,*mul_i_scalar(*t38,3.217e+1))))))))))))
#     res_f[(1,0,2)] = add_i(*mul_i(*p,*mul_i(*t2,*mul_i_scalar(*t39,-1.0))),*add_i(*mul_i(*mul_i_scalar(*r, -1.0),*mul_i(*t5,*mul_i_scalar(*t39,1.0))),*add_i(*mul_i(*dbeta_czt_c,*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t38,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*cxt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i(*t6,*mul_i_scalar(*t39,4.71e-1))))),*add_i(*mul_i(*czt_c,*mul_i(*qbar_k,*mul_i(*t2,*mul_i(*t6,*mul_i_scalar(*t39,4.71e-1))))),*add_i(*mul_i(*t5,*mul_i(*t6,*mul_i(*t9,*mul_i(*t35,*mul_i_scalar(*t39,3.217e+1))))),*mul_i(*t2,*mul_i(*t4,*mul_i(*t6,*mul_i(*t7,*mul_i(*t35,*mul_i_scalar(*t39,3.217e+1))))))))))))
#     res_f[(1,0,3)] = mul_i(*t2,*mul_i(*t7,*mul_i(*t8,*mul_i(*t35,*mul_i_scalar(*t38,-3.217e+1)))))
#     res_f[(1,0,4)] = add_i(*mul_i(*t5,*mul_i(*t7,*mul_i(*t35,*mul_i_scalar(*t38,3.217e+1)))),*mul_i(*mul_i_scalar(*t2, -1.0),*mul_i(*t4,*mul_i(*t9,*mul_i(*t35,*mul_i_scalar(*t38,3.217e+1))))))
#     res_f[(1,0,5)] = mul_i(*t2,*mul_i(*t6,*mul_i_scalar(*t38,-1.0)))
#     res_f[(1,0,6)] = add_i(*mul_i(*dq_czt_c,*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t38,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*dq_cxt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t38,4.71e-1)))),1.0))
#     res_f[(1,0,7)] = mul_i(*t5,*mul_i(*t6,*mul_i_scalar(*t38,-1.0)))
#     res_f[(1,0,8)] = add_i(*mul_i(*cxt_c,*mul_i(*dalt_qbar_k,*mul_i(*t5,*mul_i_scalar(*t38,-4.71e-1)))),*mul_i(*czt_c,*mul_i(*dalt_qbar_k,*mul_i(*t2,*mul_i_scalar(*t38,4.71e-1)))))

#     # Beta dot parial derivatives
#     res_f[(2,0,0)] = add_i(*mul_i(*dvt_cyt_c,*mul_i(*qbar_k,*mul_i_scalar(*t3,4.71e-1))),*add_i(*mul_i(*mul_i_scalar(*dvt_cxt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t6,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*dvt_czt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t6,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*t2, -1.0),*mul_i(*t6,*mul_i(*t9,*mul_i_scalar(*t36,3.217e+1)))),*add_i(*mul_i(*mul_i_scalar(*t3, -1.0),*mul_i(*t7,*mul_i(*t8,*mul_i_scalar(*t36,3.217e+1)))),*mul_i(*t4,*mul_i(*t5,*mul_i(*t6,*mul_i(*t7,*mul_i_scalar(*t36,3.217e+1))))))))))
#     res_f[(2,0,1)] = add_i(*mul_i(*p,*mul_i_scalar(*t2,1.0)),*add_i(*mul_i(*r,*mul_i_scalar(*t5,1.0)),*add_i(*mul_i(*dalph_cyt_c,*mul_i(*qbar_k,*mul_i_scalar(*t3,4.71e-1))),*add_i(*mul_i(*cxt_c,*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t6,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*czt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t6,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*dalph_cxt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t6,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*dalph_czt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t6,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*t5, -1.0),*mul_i(*t6,*mul_i(*t9,*mul_i_scalar(*t35,3.217e+1)))),*mul_i(*mul_i_scalar(*t2, -1.0),*mul_i(*t4,*mul_i(*t6,*mul_i(*t7,*mul_i_scalar(*t35,3.217e+1)))))))))))))
#     res_f[(2,0,2)] = add_i(*mul_i(*cyt_c,*mul_i(*qbar_k,*mul_i_scalar(*t6,-4.71e-1))),*add_i(*mul_i(*dbeta_cyt_c,*mul_i(*qbar_k,*mul_i_scalar(*t3,4.71e-1))),*add_i(*mul_i(*mul_i_scalar(*cxt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t3,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*czt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t3,*mul_i_scalar(*t5,4.71e-1)))),*add_i(*mul_i(*mul_i_scalar(*dbeta_czt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t6,4.71e-1)))),*add_i(*mul_i(*t2,*mul_i(*t3,*mul_i(*t9,*mul_i_scalar(*t35,3.217e+1)))),*add_i(*mul_i(*mul_i_scalar(*t6, -1.0),*mul_i(*t7,*mul_i(*t8,*mul_i_scalar(*t35,3.217e+1)))),*mul_i(*mul_i_scalar(*t3, -1.0),*mul_i(*t4,*mul_i(*t5,*mul_i(*t7,*mul_i_scalar(*t35,3.217e+1))))))))))))
#     res_f[(2,0,3)] = add_i(*mul_i(*t3,*mul_i(*t4,*mul_i(*t7,*mul_i_scalar(*t35,3.217e+1)))),*mul_i(*t5,*mul_i(*t6,*mul_i(*t7,*mul_i(*t8,*mul_i_scalar(*t35,3.217e+1))))))
#     res_f[(2,0,4)] = add_i(*mul_i(*t2,*mul_i(*t6,*mul_i(*t7,*mul_i_scalar(*t35,3.217e+1)))),*add_i(*mul_i(*mul_i_scalar(*t3, -1.0),*mul_i(*t8,*mul_i(*t9,*mul_i_scalar(*t35,3.217e+1)))),*mul_i(*t4,*mul_i(*t5,*mul_i(*t6,*mul_i(*t9,*mul_i_scalar(*t35,3.217e+1)))))))
#     res_f[(2,0,5)] = add_i(*mul_i_scalar(*t5,1.0),*mul_i(*dp_cyt_c,*mul_i(*qbar_k,*mul_i_scalar(*t3,4.71e-1))))
#     res_f[(2,0,6)] = add_i(*mul_i(*dq_cxt_c,*mul_i(*qbar_k,*mul_i(*t2,*mul_i_scalar(*t6,-4.71e-1)))),*mul_i(*mul_i_scalar(*dq_czt_c, -1.0),*mul_i(*qbar_k,*mul_i(*t5,*mul_i_scalar(*t6,4.71e-1)))))
#     res_f[(2,0,7)] = add_i(*mul_i_scalar(*t2,-1.0),*mul_i(*dr_cyt_c,*mul_i(*qbar_k,*mul_i_scalar(*t3,4.71e-1))))
#     res_f[(2,0,8)] = add_i(*mul_i(*cyt_c,*mul_i(*dalt_qbar_k,*mul_i_scalar(*t3,4.71e-1))),*add_i(*mul_i(*mul_i_scalar(*cxt_c, -1.0),*mul_i(*dalt_qbar_k,*mul_i(*t2,*mul_i_scalar(*t6,4.71e-1)))),*mul_i(*mul_i_scalar(*czt_c, -1.0),*mul_i(*dalt_qbar_k,*mul_i(*t5,*mul_i_scalar(*t6,4.71e-1))))))

#     res_f[(3,0,3)] = mul_i(*t9,*mul_i(*t40,*add_i(*mul_i(*q,*t4),*mul_i(*mul_i_scalar(*r, -1.0),*t8))))
#     res_f[(3,0,4)] = mul_i(*add_i(*t33, *t34), *pow2_i(*t40))
#     res_f[(3,0,5)] = (1.0,1.0)
#     res_f[(3,0,6)] = mul_i(*t8,*mul_i(*t9,*t40))
#     res_f[(3,0,7)] = mul_i(*t4,*mul_i(*t9,*t40))

#     res_f[(4,0,3)] = mul_i_scalar(*add_i(*t33, *t34), -1.0)
#     res_f[(4,0,6)] = t4
#     res_f[(4,0,7)] = mul_i_scalar(*t8, -1.0)

#     res_f[(5,0,0)] = add_i(*mul_i(*qbar_k,*t47),*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dvt_clt_c,9.495e-1),*mul_i_scalar(*dvt_cnt_c,1.4778e-2)))))
#     res_f[(5,0,1)] = mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dalph_clt_c,9.495e-1),*mul_i_scalar(*dalph_cnt_c,1.4778e-2))))
#     res_f[(5,0,2)] = mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dbeta_clt_c,9.495e-1),*mul_i_scalar(*dbeta_cnt_c,1.4778e-2))))
#     res_f[(5,0,5)] = add_i(*t22,*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dp_clt_c,9.495e-1),*mul_i_scalar(*dp_cnt_c,1.4778e-2)))))
#     res_f[(5,0,6)] = add_i(*mul_i_scalar(*p,2.755e-2),*add_i(*mul_i_scalar(*r,-7.7e-1),2.6272e-4))
#     res_f[(5,0,7)] = add_i(*mul_i_scalar(*q,-7.7e-1),*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dr_clt_c,9.495e-1),*mul_i_scalar(*dr_cnt_c,1.4778e-2)))))
#     res_f[(5,0,8)] = mul_i(*dalt_qbar_k,*mul_i(*t47,*vt))

#     res_f[(7,0,0)] = add_i(*mul_i(*qbar_k,*t42),*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dvt_clt_c,1.4778e-2),*mul_i_scalar(*dvt_cnt_c,1.4283e-1)))))
#     res_f[(7,0,1)] = mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dalph_clt_c,1.4778e-2),*mul_i_scalar(*dalph_cnt_c,1.4283e-1))))
#     res_f[(7,0,2)] = mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dbeta_clt_c,1.4778e-2),*mul_i_scalar(*dbeta_cnt_c,1.4283e-1))))
#     res_f[(7,0,5)] = add_i(*mul_i_scalar(*q,-7.336e-1),*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dp_clt_c,1.4778e-2),*mul_i_scalar(*dp_cnt_c,1.4283e-1)))))
#     res_f[(7,0,6)] = add_i(*mul_i_scalar(*p,-7.336e-1),*add_i(*mul_i_scalar(*r,-2.755e-2),2.5392e-3))
#     res_f[(7,0,7)] = add_i(*mul_i_scalar(*t22,-1.0),*mul_i(*qbar_k,*mul_i(*vt,*add_i(*mul_i_scalar(*dr_clt_c,1.4778e-2),*mul_i_scalar(*dr_cnt_c,1.4283e-1)))))
#     res_f[(7,0,8)] = mul_i(*dalt_qbar_k,*mul_i(*t42,*vt))

#     res_f[(6,0,0)] = add_i(*mul_i(*cmt_c,*mul_i_scalar(*qbar_k,6.085632e-2)),*mul_i(*dvt_cmt_c,*mul_i(*qbar_k,*mul_i_scalar(*vt,6.085632e-2))))
#     res_f[(6,0,1)] = mul_i(*dalph_cmt_c,*mul_i(*qbar_k,*mul_i_scalar(*vt,6.085632e-2)))
#     res_f[(6,0,5)] = add_i(*mul_i_scalar(*p,-3.518e-2),*mul_i_scalar(*r,9.604e-1))
#     res_f[(7,0,6)] = mul_i(*dq_cmt_c,*mul_i(*qbar_k,*mul_i_scalar(*vt,6.085632e-2)))
#     res_f[(7,0,7)] = add_i(*mul_i_scalar(*p,9.604e-1),*add_i(*mul_i_scalar(*r,3.518e-2),-2.8672e-3))
#     res_f[(7,0,8)] = mul_i(*cmt_c,*mul_i(*dalt_qbar_k,*mul_i_scalar(*vt,6.085632e-2)))

#     res_f[(8,0,0)] = add_i(*mul_i(*t2,*mul_i(*t3,*t9)),*add_i(*mul_i(*t6,*mul_i(*t7,*mul_i_scalar(*t8,-1.0))),*mul_i(*t3,*mul_i(*t4,*mul_i(*t5,*mul_i_scalar(*t7,-1.0))))))
#     res_f[(8,0,1)] = mul_i_scalar(*mul_i(*vt,*add_i(*mul_i(*t3,*mul_i(*t5,*t9)),*mul_i(*t2,*mul_i(*t3,*mul_i(*t4,*mul_i_scalar(*t7,1.0)))))),-1.0)
#     res_f[(8,0,2)] = mul_i(*vt,*add_i(*mul_i(*t2,*mul_i(*t6,*t9)),*add_i(*mul_i(*t3,*mul_i(*t7,*mul_i_scalar(*t8,-1.0))),*mul_i(*t4,*mul_i(*t5,*mul_i(*t6,*t7))))))
#     res_f[(8,0,3)] = mul_i(*vt,*add_i(*mul_i(*t4,*mul_i(*t6,*mul_i_scalar(*t7,-1.0))),*mul_i(*t3,*mul_i(*t5,*mul_i(*t7,*t8)))))
#     res_f[(8,0,4)] = mul_i(*vt,*add_i(*mul_i(*t2,*mul_i(*t3,*t7)),*add_i(*mul_i(*t6,*mul_i(*t8,*t9)),*mul_i(*t3,*mul_i(*t4,*mul_i(*t5,*t9))))))

#     t51 = add_i(*t29, *t41)
#     t52 = add_i(*t30, *t51)
#     res_G[(0,1,0,0)] = mul_i(*qbar_k,*mul_i(*t3,*mul_i(*t50,*mul_i_scalar(*vt,9.42e-1))))
#     res_G[(0,1,0,1)] = mul_i(*qbar_k,*mul_i(*t3,*mul_i(*t27,*mul_i_scalar(*t52,4.71e-1))))
#     res_G[(0,1,0,2)] = mul_i(*qbar_k,*mul_i(*t6,*mul_i(*t27,*mul_i_scalar(*t50,-4.71e-1))))
#     res_G[(0,1,0,8)] = mul_i(*dalt_qbar_k,*mul_i(*t3,*mul_i(*t27,*mul_i_scalar(*t50,4.71e-1))))

#     res_G[(0,2,0,0)] = mul_i(*cyt_ail,*mul_i(*qbar_k,*mul_i(*t6,*mul_i_scalar(*vt,9.42e-1))))
#     res_G[(0,2,0,2)] = mul_i(*cyt_ail,*mul_i(*qbar_k,*mul_i(*t3,*mul_i_scalar(*t27,4.71e-1))))
#     res_G[(0,2,0,8)] = mul_i(*cyt_ail,*mul_i(*dalt_qbar_k,*mul_i(*t6,*mul_i_scalar(*t27,4.71e-1))))

#     res_G[(0,3,0,0)] = mul_i(*cyt_rdr,*mul_i(*qbar_k,*mul_i(*t6,*mul_i_scalar(*vt,9.42e-1))))
#     res_G[(0,3,0,2)] = mul_i(*cyt_rdr,*mul_i(*qbar_k,*mul_i(*t3,*mul_i_scalar(*t27,4.71e-1))))
#     res_G[(0,3,0,8)] = mul_i(*cyt_rdr,*mul_i(*dalt_qbar_k,*mul_i(*t6,*mul_i_scalar(*t27,4.71e-1))))

#     res_G[(1,1,0,0)] = mul_i(*qbar_k,*mul_i(*t38,*mul_i_scalar(*t51,4.71e-1)))
#     res_G[(1,1,0,1)] = mul_i(*qbar_k,*mul_i(*t38,*mul_i(*vt,*mul_i_scalar(*add_i(*t50,*mul_i(*dalph_cxt_el,*t5)),-4.71e-1))))
#     res_G[(1,1,0,2)] = mul_i(*qbar_k,*mul_i(*t6,*mul_i(*t39,*mul_i(*t51,*mul_i_scalar(*vt,4.71e-1)))))
#     res_G[(1,1,0,8)] = mul_i(*dalt_qbar_k,*mul_i(*t38,*mul_i(*t51,*mul_i_scalar(*vt,4.71e-1))))

#     res_G[(2,1,0,0)] = mul_i(*qbar_k,*mul_i(*t6,*mul_i_scalar(*t50,-4.71e-1)))
#     res_G[(2,1,0,1)] = mul_i(*qbar_k,*mul_i(*t6,*mul_i(*t52,*mul_i_scalar(*vt,-4.71e-1))))
#     res_G[(2,1,0,2)] = mul_i(*qbar_k,*mul_i(*t3,*mul_i(*t50,*mul_i_scalar(*vt,-4.71e-1))))
#     res_G[(2,1,0,8)] = mul_i(*dalt_qbar_k,*mul_i(*t6,*mul_i(*t50,*mul_i_scalar(*vt,-4.71e-1))))

#     res_G[(2,2,0,0)] = mul_i(*cyt_ail,*mul_i(*qbar_k,*mul_i_scalar(*t3,4.71e-1)))
#     res_G[(2,2,0,2)] = mul_i(*cyt_ail,*mul_i(*qbar_k,*mul_i(*t6,*mul_i_scalar(*vt,-4.71e-1))))
#     res_G[(2,2,0,8)] = mul_i(*cyt_ail,*mul_i(*dalt_qbar_k,*mul_i(*t3,*mul_i_scalar(*vt,4.71e-1))))

#     res_G[(2,3,0,0)] = mul_i(*cyt_rdr,*mul_i(*qbar_k,*mul_i_scalar(*t3,4.71e-1)))
#     res_G[(2,3,0,2)] = mul_i(*cyt_rdr,*mul_i(*qbar_k,*mul_i(*t6,*mul_i_scalar(*vt,-4.71e-1))))
#     res_G[(2,3,0,8)] = mul_i(*cyt_rdr,*mul_i(*dalt_qbar_k,*mul_i(*t3,*mul_i_scalar(*vt,4.71e-1))))

#     res_G[(5,2,0,0)] = mul_i(*qbar_k,*mul_i(*t48,*mul_i_scalar(*vt,2.0)))
#     res_G[(5,2,0,1)] = mul_i(*qbar_k,*mul_i(*t27,*add_i(*mul_i_scalar(*dalph_clt_ail,9.495e-1),*mul_i_scalar(*dalph_cnt_ail,1.4778e-2))))
#     res_G[(5,2,0,2)] = mul_i(*qbar_k,*mul_i(*t27,*add_i(*mul_i_scalar(*dbeta_clt_ail,9.495e-1),*mul_i_scalar(*dbeta_cnt_ail,1.4778e-2))))
#     res_G[(5,2,0,3)] = mul_i(*dalt_qbar_k,*mul_i(*t27,*t48))

#     res_G[(5,3,0,0)] = mul_i(*qbar_k,*mul_i(*t49,*mul_i_scalar(*vt,2.0)))
#     res_G[(5,3,0,1)] = mul_i(*qbar_k,*mul_i(*t27,*add_i(*mul_i_scalar(*dalph_clt_rdr,9.495e-1),*mul_i_scalar(*dalph_cnt_rdr,1.4778e-2))))
#     res_G[(5,3,0,2)] = mul_i(*qbar_k,*mul_i(*t27,*add_i(*mul_i_scalar(*dbeta_clt_rdr,9.495e-1),*mul_i_scalar(*dbeta_cnt_rdr,1.4778e-2))))
#     res_G[(5,3,0,3)] = mul_i(*dalt_qbar_k,*mul_i(*t27,*t49))

#     res_G[(6,1,0,0)] = mul_i(*cmt_el,*mul_i(*qbar_k,*mul_i_scalar(*vt,1.2171264e-1)))
#     res_G[(6,1,0,1)] = mul_i(*dalph_cmt_el,*mul_i(*qbar_k,*mul_i_scalar(*t27,6.085632e-2)))
#     res_G[(6,1,0,8)] = mul_i(*cmt_el,*mul_i(*dalt_qbar_k,*mul_i_scalar(*t27,6.085632e-2)))


#     res_G[(7,2,0,0)] = mul_i(*qbar_k,*mul_i(*t43,*mul_i_scalar(*vt,2.0)))
#     res_G[(7,2,0,1)] = mul_i(*qbar_k,*mul_i(*t27,*add_i(*mul_i_scalar(*dalph_clt_ail,1.4778e-2),*mul_i_scalar(*dalph_cnt_ail,1.4283e-1))))
#     res_G[(7,2,0,2)] = mul_i(*qbar_k,*mul_i(*t27,*add_i(*mul_i_scalar(*dbeta_clt_ail,1.4778e-2),*mul_i_scalar(*dbeta_cnt_ail,1.4283e-1))))
#     res_G[(7,2,0,3)] = mul_i(*dalt_qbar_k,*mul_i(*t27,*t43))

#     res_G[(7,3,0,0)] = mul_i(*qbar_k,*mul_i(*t44,*mul_i_scalar(*vt,2.0)))
#     res_G[(7,3,0,1)] = mul_i(*qbar_k,*mul_i(*t27,*add_i(*mul_i_scalar(*dalph_clt_rdr,1.4778e-2),*mul_i_scalar(*dalph_cnt_rdr,1.4283e-1))))
#     res_G[(7,3,0,2)] = mul_i(*qbar_k,*mul_i(*t27,*add_i(*mul_i_scalar(*dbeta_clt_rdr,1.4778e-2),*mul_i_scalar(*dbeta_cnt_rdr,1.4283e-1))))
#     res_G[(7,3,0,3)] = mul_i(*dalt_qbar_k,*mul_i(*t27,*t44))
