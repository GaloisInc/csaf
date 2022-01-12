import numpy as np

from numba import jit, types, prange, typeof
from numba.typed import Dict

from .interval import *

from numpy import float64 as realN
from numba import float64 as real
from numba import int64 as indType
from numba import boolean as bol

# Threshold to detect change when applying the contraction algorithm
tolChange = 1e-4

@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def buildJacF(nS, LipF, scalingLip, vDep, nvDep, knownGrad,
                unkEval, kEval, gradBounds):
    """ Compute the enclosure of the Jacobian matrice Jf based only on the
        LIpschitz constant LipF, the non-dependent variables of f, scaling factor
        for the weighted norm associated to LIpschitz constant, and
        side information such as gradient bounds
    """
    # Upper bound given by the Lipschitz constants
    Jf_init_lb = np.zeros((nS, nS), dtype=realN)
    Jf_init_ub = np.zeros((nS, nS), dtype=realN)
    for (k,p), (lb,ub) in kEval.items():
        if (k,p) in LipF:
            currDep = vDep[(k,p)]
            currScale = scalingLip[(k,p)]
            Lf = LipF[(k,p)]
            for j in prange(currDep.shape[0]):
                currB = Lf * currScale[j]
                gradB = (-currB, currB)
                if (k,p,currDep[j]) in gradBounds:
                    gradB = and_i(*gradB, *gradBounds[(k,p,currDep[j])])
                    assert gradB[0] <= gradB[1]
                if (k,p,currDep[j]) in knownGrad:
                    t_lb, t_ub = add_i(
                        *mul_i(*unkEval[(k,p)], *knownGrad[(k,p,currDep[j])]),
                        *mul_i(*gradB, lb, ub))
                else:
                    t_lb, t_ub = mul_i(*gradB, lb, ub)
                Jf_init_lb[k,currDep[j]] += t_lb
                Jf_init_ub[k,currDep[j]] += t_ub
            currnDep = nvDep[(k,p)]
            for j in prange(currnDep.shape[0]):
                if (k,p,currnDep[j]) in knownGrad:
                    t_lb, t_ub = mul_i(*unkEval[(k,p)], *knownGrad[(k,p,currnDep[j])])
                    Jf_init_lb[k,currnDep[j]] += t_lb
                    Jf_init_ub[k,currnDep[j]] += t_ub
        else:
            for j in prange(nS):
                if (k,p,j) in knownGrad:
                    t_lb, t_ub = knownGrad[(k,p,j)]
                    Jf_init_lb[k,j] += t_lb
                    Jf_init_ub[k,j] += t_ub
    return Jf_init_lb, Jf_init_ub


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def buildJacG(nS, nC, LipG, scalingLip, vDep, nvDep, knownGrad,
                unkEval, kEval, gradBounds):
    """ Compute the enclosure of the Jacobian matrice Jf based only on the
        LIpschitz constant LipF, the non-dependent variables of f, scaling factor
        for the weighted norm associated to LIpschitz constant, and
        side information such as gradient bounds
    """
    # Upper bound given by the Lipschitz constants
    JG_init_lb = np.zeros((nS, nC, nS), dtype=realN)
    JG_init_ub = np.zeros((nS, nC, nS), dtype=realN)
    for (k,l,p), (lb,ub) in kEval.items():
        if (k,l,p) in LipG:
            currDep = vDep[(k,l,p)]
            currScale = scalingLip[(k,l,p)]
            LG = LipG[(k,l,p)]
            for j in prange(currDep.shape[0]):
                currB = LG * currScale[j]
                gradB = (-currB, currB)
                if (k,l,p,currDep[j]) in gradBounds:
                    gradB = and_i(*gradB, *gradBounds[(k,l,p,currDep[j])])
                    assert gradB[0] <= gradB[1]
                if (k,l,p,currDep[j]) in knownGrad:
                    t_lb, t_ub = add_i(
                        *mul_i(*unkEval[(k,l,p)], *knownGrad[(k,l,p,currDep[j])]),
                        *mul_i(*gradB, lb, ub))
                else:
                    t_lb, t_ub = mul_i(*gradB, lb, ub)
                JG_init_lb[k,l,currDep[j]] += t_lb
                JG_init_ub[k,l,currDep[j]] += t_ub
            currnDep = nvDep[(k,l,p)]
            for j in prange(currnDep.shape[0]):
                if (k,l,p,currnDep[j]) in knownGrad:
                    t_lb, t_ub = mul_i(*unkEval[(k,l,p)], *knownGrad[(k,l,p,currnDep[j])])
                    JG_init_lb[k,l,currnDep[j]] += t_lb
                    JG_init_ub[k,l,currnDep[j]] += t_ub
        else:
            for j in prange(nS):
                if (k,l,p,j) in knownGrad:
                    t_lb, t_ub = knownGrad[(k,l,p,j)]
                    JG_init_lb[k,l,j] += t_lb
                    JG_init_ub[k,l,j] += t_ub
    return JG_init_lb, JG_init_ub

@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def lipOverApprox(x_lb, x_ub, L, dataState, dataFun_lb, dataFun_ub, scalingLip):
    """ Function to over-approximate a real-valued function at the given
        input x based on the Lipschitz constants of the function, and
        interval-based samples of the values of such function

        Parameters
        ----------
        :param x=(x_lb,x_ub): The point to evaluate the function based on data
        :param L: An upper bound on the LIpschitz constant of that function
        :param dataState: 2d array providing the historic of the state
        :param dataFun=(dataFun_lb, dataFun_ub): arrays providing an
                        over-approximation of the function at every point of dataState

        Returns
        -------
        An overapproximation of the unknown function at given x
    """
    normValLip_lb = np.empty(dataState.shape[0], dtype=realN)
    normValLip_ub = np.empty(dataState.shape[0], dtype=realN)
    for i in prange(normValLip_lb.shape[0]):
        normValLip_lb[i], normValLip_ub[i] = \
            mul_i_lip(*norm_i(
                        *mul_iv_sv(
                            *sub_i(x_lb, x_ub, dataState[i]),
                             scalingLip
                            )),
                       L)
    res_lb, res_ub = and_iv(*add_i(dataFun_lb, dataFun_ub, normValLip_lb, normValLip_ub))
    assert res_lb <= res_ub
    return res_lb, res_ub

@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def hc4Revise(xdot_lb, xdot_ub, Gx_i_lb, Gx_i_ub, u, tol):
    """ Given an equation constraint of the type
        xdot_i = sum_k (Gx_i)_k u_k where (Gx_i)_k are unknown for
        all k, this function returns set of possible values of (Gx_i)_k
        satisfying the constraints above. The returned sets are contraction of
        the original sets (Gx_i_lb, Gx_i_ub) that
        over-approximates the unknonw variables (Gx_i)_k.
        Assumption: u is nonzero
    """
    # Find the control that are non-zero --> Reduce also the complexity
    (indNZu,) = np.nonzero(u)
    # If the size of indNZu is zero then only f can be update
    if indNZu.shape[0] == 0:
        return False
    if indNZu.shape[0] == 1:
        new_val_lb, new_val_ub = mul_i_scalar(Gx_i_lb[indNZu[0]], Gx_i_ub[indNZu[0]], u[indNZu[0]])

        # new_val_lb, new_val_ub = mul_i_scalar(xdot_lb, xdot_ub, 1.0/u[indNZu[0]])
        hasChanged = (xdot_lb - new_val_lb > tol) or \
                     (xdot_ub - new_val_ub < -tol)
        # if hasChanged:
        #     print('HAS: ', new_val_lb, new_val_ub, Gx_i_lb[indNZu[0]], Gx_i_ub[indNZu[0]])
        if hasChanged:
            temp_lb, temp_ub = \
                    mul_i_scalar(*and_i(xdot_lb, xdot_ub, new_val_lb, new_val_ub, tol),
                                 1.0/u[indNZu[0]])
            # Gx_i_lb[indNZu[0]], Gx_i_ub[indNZu[0]] = \
            #         mul_i_scalar(*and_i(xdot_lb, xdot_ub, new_val_lb, new_val_ub, tol),
            #                      1.0/u[indNZu[0]])
            assert temp_lb <= temp_ub
            Gx_i_lb[indNZu[0]], Gx_i_ub[indNZu[0]] = temp_lb, temp_ub
        return hasChanged

    # Boolean variable checking if a tighter set was obtained
    hasChanged = False

    # Compute the elementwise product Gx_i u and use it as the nodes of the
    # tree for the HC4revise algorithm
    u_red = u[indNZu]
    nGu_lb, nGu_ub = mul_iv_sv(Gx_i_lb[indNZu], Gx_i_ub[indNZu], u_red)

    # Store the forward and backward interval result of the nodes representing
    # the 2-ary operation (Here addition) -> There's len(indZu) additions
    plusArray_lb = np.empty(indNZu.shape[0]-1, dtype=realN)
    plusArray_ub = np.empty(indNZu.shape[0]-1, dtype=realN)

    # Forward Evaluation of the tree to update the addition node
    plusArray_lb[0], plusArray_ub[0] = add_i(nGu_lb[0], nGu_ub[0], nGu_lb[1], nGu_ub[1])
    for i in range(1, plusArray_lb.shape[0]):
        plusArray_lb[i], plusArray_ub[i] = add_i(plusArray_lb[i-1], plusArray_ub[i-1],\
                                                 nGu_lb[i+1], nGu_ub[i+1])

    # Check if the sum is not already included
    hasChanged_t = xdot_lb - plusArray_lb[-1] > tol or \
                    xdot_ub - plusArray_ub[-1] < -tol
    if not hasChanged_t:
        return False

    # Backward evaluation of the tree to tighten the addition values
    plusArray_lb[-1], plusArray_ub[-1] = and_i(xdot_lb, xdot_ub,
                                            plusArray_lb[-1], plusArray_ub[-1], tol)
    assert plusArray_lb[-1] <= plusArray_ub[-1]

    for i in range(plusArray_lb.shape[0]-1, -1, -1):
        lTerm_lb, lTerm_ub = nGu_lb[i+1], nGu_ub[i+1]
        if i == 0:
            rTerm_lb, rTerm_ub = nGu_lb[0], nGu_ub[0]
        else:
            rTerm_lb, rTerm_ub = plusArray_lb[i-1], plusArray_ub[i-1]

        oTerm_lb, oTerm_ub = sub_i(plusArray_lb[i], plusArray_ub[i], rTerm_lb, rTerm_ub)
        lTerm_lb, lTerm_ub = and_i(oTerm_lb, oTerm_ub, lTerm_lb, lTerm_ub)
        assert lTerm_lb <= lTerm_ub
        if (lTerm_lb != -np.inf and lTerm_lb - nGu_lb[i+1] > tol) or \
            (lTerm_ub != np.inf and lTerm_ub - nGu_ub[i+1] < -tol):
            hasChanged = True
        nGu_lb[i+1], nGu_ub[i+1] = lTerm_lb, lTerm_ub

        oTerm_lb, oTerm_ub = sub_i(plusArray_lb[i], plusArray_ub[i], lTerm_lb, lTerm_ub)
        hasChanged_t = (oTerm_lb != -np.inf and  oTerm_lb - rTerm_lb > epsTolInt) or \
            (oTerm_ub != np.inf and  oTerm_ub - rTerm_ub < -epsTolInt)
        if not hasChanged_t:
            break
        rTerm_lb, rTerm_ub = and_i(oTerm_lb, oTerm_ub, rTerm_lb, rTerm_ub)
        assert rTerm_lb <= rTerm_ub
        if i == 0:
            if (rTerm_lb != -np.inf and rTerm_lb - nGu_lb[0] > tol) or \
                (rTerm_ub != np.inf and rTerm_ub - nGu_ub[0] < -tol):
                hasChanged = True
            nGu_lb[0], nGu_ub[0] = rTerm_lb, rTerm_ub
        else:
            plusArray_lb[i-1], plusArray_ub[i-1] = rTerm_lb, rTerm_ub

    # POst processing and return correct value format
    Gx_i_lb[indNZu], Gx_i_ub[indNZu] = mul_iv_sv(nGu_lb, nGu_ub, 1.0/u_red)
    return hasChanged
