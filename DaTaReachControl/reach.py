import numpy as np

from .interval import *
from .overapprox_functions import *

from numba import jit
from numba.core import types
from numba.typed import List, Dict

from numba import uint16
from numpy import float64 as realN
from numba import float64 as real
from numba import int64 as indType
from numpy import int64 as indTypeN

from collections import namedtuple

spec = ['nS', 'nC', 'verbose', 'Lf', 'vDepF', 'nvDepF', 'bf', 'bGf',
        'LG', 'vDepG', 'nvDepG', 'bG', 'bGG', 'xTraj', 'xDot_lb', 'xDot_ub',
        'uVal', 'fOverTraj_lb', 'fOverTraj_ub', 'fCoeff',
        'GOverTraj_lb', 'GOverTraj_ub', 'GCoeff', 'nbData',
        'fixpointWidenCoeff','zeroDiameter', 'widenZeroInterval', 'tolChange',
        'maxInvariantIter', 'scalingLipF', 'scalingLipG', 'termF', 'termG',
        'simVarF', 'simVarG', 'ignoreInconsistent']

ReachDyn = namedtuple('ReachDyn', spec)

# Define some types for empty dictionary
uniTupleInt2 = types.UniTuple(types.int64,2)
uniTupleInt3 = types.UniTuple(types.int64,3)
uniTupleInt4 = types.UniTuple(types.int64,4)
uniTupleFloat2 = types.UniTuple(types.float64,2)

# Dummy function when no gradient information or known part are known
@jit(nopython=True, fastmath=True, cache=True)
def knownDynamics(x_lb, x_ub, gradF=None, gradG=None):
    return Dict.empty(uniTupleInt2, uniTupleFloat2), Dict.empty(uniTupleInt3, uniTupleFloat2)

def compBuilder(ind=0, term=0, Lip=0.0, vDep=[], weightLip=[], nS=0,
                    bound=None, gradBound=None):
    """ Utils function to build the parameters needed in order
        to construct the differential inclusion
    """
    assert (len(vDep) > 0) and nS > 0 and Lip > 0
    nvDep = set([i for i in range(nS)]) - set(vDep)
    nvDep = np.array([np.int64(val) for val in nvDep])
    vDep = np.array([np.int64(vDep[i]) for i in range(len(vDep))])
    if len(weightLip) == 0:
        weightLip = [1.0 for i in range(len(vDep))]
    assert(len(vDep) == len(weightLip))
    weightLip = np.array([np.float64(weightLip[i]) for i in range(len(weightLip))])
    if gradBound is None:
        gradBound = (np.array([-np.inf for i in range(len(vDep))]), \
                        np.array([np.inf for i in range(len(vDep))]))
    else:
        gradBound = (np.array([np.float64(val[0]) for val in gradBound]),\
                        np.array([np.float64(val[1]) for val in gradBound]))
    bound = (-np.inf, np.inf) if bound is None else (np.float64(bound[0]),np.float64(bound[1]))
    if type(ind) is tuple:
        ind = (np.int64(ind[0]), np.int64(ind[1]))
    else:
        ind = np.int64(ind)
    return (ind, np.int64(term), np.float(Lip), nvDep, vDep, weightLip, bound, gradBound)


def initOverApprox(nS, nC, listInfoF, listInfoG, xTraj=None, xDotTraj=None, uTraj=None,
                    verbose=False, fixpointWidenCoeff=0.2, zeroDiameter=1e-5,
                    widenZeroInterval=1e-3, maxData=20, tolChange=tolChange,
                    maxInvariantIter=10, ignoreInconsistent = False,
                    simVarF={}, simVarG={}):
    """ Initialize the builder of the differential inclusion
    """
    _nS = nS
    _nC = nC

    # Take into account the variable that are similar to each others
    simVarFD = Dict.empty(uniTupleInt2, uniTupleInt2)
    simVarGD = Dict.empty(uniTupleInt3, uniTupleInt3)
    newListInfoF = list()
    newListInfoG = list()
    unkVariableF = dict()
    unkVariableG = dict()
    for (k2,p2,Lip, nvDep, vDep, weightLip, bound, gradBound) in listInfoF:
        unkVariableF[(k2,p2)] = (Lip, nvDep, vDep, weightLip, bound, gradBound)
    for (k2,p2,Lip, nvDep, vDep, weightLip, bound, gradBound) in listInfoG:
        unkVariableG[(k2[0], k2[1], p2)] = (Lip, nvDep, vDep, weightLip, bound, gradBound)
    for (k,p), varTerm in simVarF.items():
        assert (k,p) in unkVariableF
        for (k2,p2) in varTerm:
            unkVariableF[(k2,p2)] = unkVariableF[(k,p)]
            simVarFD[(k2,p2)] = (k,p)
    for (k,l,p), varTerm in simVarG.items():
        assert (k,l,p) in unkVariableG
        for (k2,l2,p2) in varTerm:
            unkVariableG[(k2,l2,p2)] = unkVariableG[(k,l,p)]
            simVarGD[(k2,l2,p2)] = (k,l,p)

    # Fill back the new List Info
    for (k,p), (Lip, nvDep, vDep, weightLip, bound, gradBound) in unkVariableF.items():
        newListInfoF.append((k,p,Lip, nvDep, vDep, weightLip, bound, gradBound))
    for (k,l,p), (Lip, nvDep, vDep, weightLip, bound, gradBound) in unkVariableG.items():
        newListInfoG.append(((k,l),p,Lip, nvDep, vDep, weightLip, bound, gradBound))

    # COmpute all the decoupling + Lipschitz + scaling from the list of Info
    _Lf, _LG,_vDepF, _nvDepF, _vDepG, _nvDepG, _scalinLipF, _scalinLipG, _bf,\
        _bGf, _bG, _bGG, _fOverTraj_lb, _fOverTraj_ub, _fCoeff,\
            _GOverTraj_lb, _GOverTraj_ub, _GCoeff, _termF, _termG = \
                computeDecoupling(maxData, List(newListInfoF), List(newListInfoG),\
                                    simVarFD, simVarGD)
    # Update the trajectory data
    _nbData = np.array([0,0], dtype=np.int64) #(nbData, dataCounter)
    _xTraj = np.empty((maxData, _nS), dtype=realN)
    _uVal = np.empty((maxData, _nC), dtype=realN)
    _xDot_lb = np.empty((maxData, _nS), dtype=realN)
    _xDot_ub = np.empty((maxData, _nS), dtype=realN)

    # Coefficient parameter for computing the apriori enclosure
    _fixpointWidenCoeff = fixpointWidenCoeff
    _zeroDiameter = zeroDiameter
    _widenZeroInterval = widenZeroInterval
    _tolChange = tolChange
    _maxInvariantIter = maxInvariantIter

    # Build the object to store all the information
    overApprox = ReachDyn(_nS, _nC, verbose, _Lf, _vDepF, _nvDepF,
                    _bf, _bGf, _LG, _vDepG, _nvDepG, _bG, _bGG, _xTraj, _xDot_lb,_xDot_ub,
                    _uVal, _fOverTraj_lb, _fOverTraj_ub, _fCoeff,
                    _GOverTraj_lb, _GOverTraj_ub, _GCoeff, _nbData, _fixpointWidenCoeff,
                    _zeroDiameter, _widenZeroInterval, _tolChange, _maxInvariantIter,
                    _scalinLipF, _scalinLipG, _termF, _termG, simVarFD, simVarGD,
                    ignoreInconsistent)

    # If the trajectory data is empty -> Nothing to do
    if uTraj is None:
        return overApprox

    # If not update the trajectory list based on HC4-Revise
    for i in range(uTraj.shape[0]):
        # if knownPart is None:
        #     feval = Dict.empty(uniTupleInt2, uniTupleFloat2)
        # else:
        #     feval, geval = knownPart(xTraj[i,:], xTraj[i,:])
        feval, geval = knownDynamics(xTraj[i,:], xTraj[i,:])
        # Fill in 1.0 for coefficient that haven't been entered by the user
        for (k,p) in _Lf:
            if (k,p) not in feval:
                feval[(k,p)] = (1.0,1.0)
        for (k,l,p) in _LG:
            if (k,l,p) not in geval:
                geval[(k,l,p)] = (1.0,1.0)
        updateApprox(overApprox, xTraj[i,:], xDotTraj[i,:], uTraj[i,:], feval, geval)
        # print('--> ', i)
    return overApprox


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def computeDecoupling(maxData, listInfoF, listInfoG, simVarFD, simVarGD):
    """ Compute and store the variable for which f and G depends on.
        TODO: Inefficient approach -> But not important since computed only once
    """
    countF = dict()
    countG = dict()

    _Lf = dict()
    _vDepF = dict()
    _nvDepF = dict()
    _scalinLipF = dict()
    _bf = Dict.empty(key_type=uniTupleInt2, value_type=uniTupleFloat2)
    _bGf = Dict.empty(key_type=uniTupleInt3, value_type=uniTupleFloat2)
    for (ind, term, Lip, nvDep, vDep, weightLip, bound, gradBound) in listInfoF:
        if ind in countF:
            countF[ind] += 1
        else:
            countF[ind] = 1
        _Lf[(ind,term)] = Lip
        _vDepF[(ind,term)] = vDep
        _nvDepF[(ind,term)] = nvDep
        _scalinLipF[(ind, term)] = weightLip
        if bound[0] != -np.inf:
            _bf[(ind,term)] = bound
        for i in range(vDep.shape[0]):
            if gradBound[0][i] != -np.inf:
                _bGf[(ind,term,vDep[i])] = (gradBound[0][i], gradBound[1][i])

    _LG = dict()
    _vDepG = dict()
    _nvDepG = dict()
    _scalinLipG = dict()
    _bG = Dict.empty(key_type=uniTupleInt3, value_type=uniTupleFloat2)
    _bGG = Dict.empty(key_type=uniTupleInt4, value_type=uniTupleFloat2)
    for (ind, term, Lip, nvDep, vDep, weightLip, bound, gradBound) in listInfoG:
        if ind[0] in countG:
            countG[ind[0]] += 1
        else:
            countG[ind[0]] = 1
        _LG[(ind[0],ind[1],term)] = Lip
        _nvDepG[(ind[0],ind[1],term)] = nvDep
        _vDepG[(ind[0], ind[1], term)] = vDep
        _scalinLipG[(ind[0],ind[1], term)] = weightLip
        if bound[0] != -np.inf:
            _bG[(ind[0],ind[1],term)] = bound
        for i in range(vDep.shape[0]):
            if gradBound[0][i] != -np.inf:
                _bGG[(ind[0],ind[1],term,vDep[i])] = (gradBound[0][i], gradBound[1][i])

    # Initialze the array for saving the trajectory
    _fOverTraj_lb = dict()
    _fOverTraj_ub = dict()
    _fCoeff = dict()
    for (k,p) in _Lf:
        _fCoeff[(k,p)] = np.empty(maxData, dtype=realN)
        if (k,p) in simVarFD:
            continue
        _fOverTraj_lb[(k,p)] = np.empty(maxData, dtype=realN)
        _fOverTraj_ub[(k,p)] = np.empty(maxData, dtype=realN)

    _GOverTraj_lb = dict()
    _GOverTraj_ub = dict()
    _GCoeff = dict()
    for (k,l,p) in _LG:
        _GCoeff[(k,l,p)] = np.empty(maxData, dtype=realN)
        if (k,l,p) in simVarGD:
            continue
        _GOverTraj_lb[(k,l,p)] = np.empty(maxData, dtype=realN)
        _GOverTraj_ub[(k,l,p)] = np.empty(maxData, dtype=realN)

    _termF = dict()
    for k, val in countF.items():
        _termF[k] = np.zeros(val, dtype=np.int64)
        iterCount = 0
        for (ind, term, _, _, _, _, _, _) in listInfoF:
            if ind == k:
                _termF[ind][iterCount] = term
                iterCount += 1

    _termG = dict()
    for k, val in countG.items():
        _termG[k] = (np.zeros(val, dtype=np.int64),np.zeros(val, dtype=np.int64))
        iterCount = 0
        for (ind, term, _, _, _, _, _, _) in listInfoG:
            if ind[0] == k:
                _termG[ind[0]][0][iterCount] = ind[1]
                _termG[ind[0]][1][iterCount] = term
                iterCount += 1


    return _Lf, _LG, _vDepF, _nvDepF, _vDepG, _nvDepG, _scalinLipF, _scalinLipG,\
            _bf, _bGf, _bG, _bGG, _fOverTraj_lb, _fOverTraj_ub, _fCoeff,\
            _GOverTraj_lb, _GOverTraj_ub, _GCoeff, _termF, _termG


@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def diffInclusion(overApprox, x_lb, x_ub, fknownX, gknonwX, fukn=None, gukn=None):
    """ Compute the differential inclusion for the system
        given the known part
    """
    resf_lb = np.zeros(overApprox.nS, dtype=np.float64)
    resf_ub = np.zeros(overApprox.nS, dtype=np.float64)
    resG_lb = np.zeros((overApprox.nS,overApprox.nC), dtype=np.float64)
    resG_ub = np.zeros((overApprox.nS,overApprox.nC), dtype=np.float64)
    if fukn is None or gukn is None:
        fuknx, guknx = estimateUnknownFun(overApprox, x_lb, x_ub)
    else:
        fuknx, guknx = fukn, gukn
    for (k,p), (lb,ub) in fknownX.items():
        if (k,p) in fuknx:
            temp_v_lb, temp_v_ub = mul_i(lb, ub, *fuknx[(k,p)])
            resf_lb[k] += temp_v_lb
            resf_ub[k] += temp_v_ub
        else:
            resf_lb[k] += lb
            resf_ub[k] += ub
    for (k,l,p), (lb,ub) in gknonwX.items():
        if (k,l,p) in guknx:
            temp_v_lb, temp_v_ub = mul_i(lb, ub, *guknx[(k,l,p)])
            resG_lb[k,l] += temp_v_lb
            resG_ub[k,l] += temp_v_ub
        else:
            resG_lb[k,l] += lb
            resG_ub[k,l] += ub
    return resf_lb, resf_ub, resG_lb, resG_ub

@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def estimateUnknownFun(overApprox, x_lb, x_ub):
    """ Compute the unknown term of f and G based on the current saved
        trajectory and the LIpschitz bounds with variable dependency of
        that function. The estimation is made at x = (x_lb, x_ub)
    """
    if overApprox.nbData[0] == 0:
        resf = dict()
        resG = dict()
        for (k,p) in overApprox.Lf:
            resf[(k,p)] = overApprox.bf[(k,p)] if (k,p) in overApprox.bf else (-np.inf, np.inf)
        for (k,l,p) in overApprox.LG:
            resG[(k,l,p)] = overApprox.bG[(k,l,p)] if (k,l,p) in overApprox.bG else (-np.inf, np.inf)
        return resf, resG
    f_val = dict()
    G_val = dict()
    for (k,p), Lip in overApprox.Lf.items():
        if (k,p) in overApprox.simVarF:
            continue
        depVar = overApprox.vDepF[(k,p)]
        temp_v = lipOverApprox(x_lb[depVar], x_ub[depVar], Lip,
                    overApprox.xTraj[:overApprox.nbData[0], depVar],
                    overApprox.fOverTraj_lb[(k,p)][:overApprox.nbData[0]],
                    overApprox.fOverTraj_ub[(k,p)][:overApprox.nbData[0]],
                    overApprox.scalingLipF[(k,p)])
        f_val_lb,f_val_ub = and_i(*temp_v, *overApprox.bf[(k,p)])\
                                    if (k,p) in overApprox.bf else temp_v
        assert f_val_lb <= f_val_ub
        f_val[(k,p)] = (f_val_lb,f_val_ub)
    for (k,p), (k1,p1) in overApprox.simVarF.items():
        f_val[(k,p)] = f_val[(k1,p1)]
    for (k,l,p), Lip in overApprox.LG.items():
        if (k,l,p) in overApprox.simVarG:
            continue
        depVar = overApprox.vDepG[(k,l,p)]
        temp_v = lipOverApprox(x_lb[depVar], x_ub[depVar], Lip,
                        overApprox.xTraj[:overApprox.nbData[0], depVar],
                        overApprox.GOverTraj_lb[(k,l,p)][:overApprox.nbData[0]],
                        overApprox.GOverTraj_ub[(k,l,p)][:overApprox.nbData[0]],
                        overApprox.scalingLipG[(k,l,p)])
        G_val_lb, G_val_ub = and_i(*temp_v, *overApprox.bG[(k,l,p)])\
                                    if (k,l,p) in overApprox.bG else temp_v
        assert G_val_lb <= G_val_ub
        G_val[(k,l,p)] = (G_val_lb, G_val_ub)
    for (k,l,p), (k1,l1,p1) in overApprox.simVarG.items():
        G_val[(k,l,p)] = G_val[k1,l1,p1]
    return f_val, G_val

@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def updateTraj(overApprox, xdot, u, fxR, GxR, fxRk=None, GxRk=None, index=0):
    """ Given new data point, update your knowledge of f and G based on HC4revise"""
    hasChanged = False
    # print(fxRk, GxRk)
    for k in prange(overApprox.nS):
        # Check if the current axis is fully known
        isKnownF = k not in overApprox.termF
        isKnownG = k not in overApprox.termG
        if isKnownF and isKnownG:
            continue
        # COunt the number of unknown terms
        nIncF = 0 if isKnownF else overApprox.termF[k].shape[0]
        nIncG = 0 if isKnownG else overApprox.termG[k][0].shape[0]

        coeffVal = np.empty(nIncF+nIncG, dtype=np.float64)
        funVal_lb = np.empty(nIncF+nIncG, dtype=np.float64)
        funVal_ub = np.empty(nIncF+nIncG, dtype=np.float64)

        if not isKnownF:
            indTable = overApprox.termF[k]
            for i in prange(nIncF):
                if fxRk is None:
                    coeffVal[i] = overApprox.fCoeff[(k,indTable[i])][index]
                else:
                    coeffVal[i] = fxRk[(k,indTable[i])][0]
                funVal_lb[i], funVal_ub[i] = fxR[(k,indTable[i])]
        if not isKnownG:
            indJ, indTerm = overApprox.termG[k]
            for i in prange(nIncG):
                if GxRk is None:
                    coeffVal[i+nIncF] = overApprox.GCoeff[(k,indJ[i],indTerm[i])][index]*u[indJ[i]]
                else:
                    coeffVal[i+nIncF] = GxRk[(k,indJ[i],indTerm[i])][0] * u[indJ[i]]
                funVal_lb[i+nIncF], funVal_ub[i+nIncF] = GxR[(k,indJ[i],indTerm[i])]
        # print(k, xdot[0][k],xdot[1][k], funVal_lb, funVal_ub, coeffVal)
        try:
            changed = hc4Revise(xdot[0][k],xdot[1][k], funVal_lb, funVal_ub, coeffVal, overApprox.tolChange)
        except:
            if overApprox.verbose:
                print(k, xdot[0][k],xdot[1][k], funVal_lb, funVal_ub, coeffVal)
            if overApprox.ignoreInconsistent:
                changed = False
            else:
                assert False

        if changed:
            hasChanged = True

        if not isKnownF:
            indTable = overApprox.termF[k]
            for i in prange(nIncF):
                fxR[(k,indTable[i])] = (funVal_lb[i], funVal_ub[i])
            # Intersectionf of the range of the values that are duplicated
            for (k0,p), (k1,p1) in overApprox.simVarF.items():
                fxR[(k1,p1)] = and_i(*fxR[(k0,p)], *fxR[(k1,p1)])
            for (k0,p), (k1,p1) in overApprox.simVarF.items():
                fxR[(k0,p)] = fxR[(k1,p1)]

        if not isKnownG:
            indJ, indTerm = overApprox.termG[k]
            for i in prange(nIncG):
                GxR[(k,indJ[i],indTerm[i])] = (funVal_lb[i+nIncF],funVal_ub[i+nIncF])
            # Intersectionf of the range of the values that are duplicated
            for (k0,l,p), (k1,l1,p1) in overApprox.simVarG.items():
                GxR[(k1,l1,p1)] = and_i(*GxR[(k0,l,p)], *GxR[(k1,l1,p1)])
            for (k0,l,p), (k1,l1,p1) in overApprox.simVarG.items():
                GxR[(k0,l,p)] = GxR[(k1,l1,p1)]
    return hasChanged

@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def updateApprox(overApprox, xVal, xDot, uVal, knownf, knownG):
    """ Update the over-approximation based on the new xVal, xDot, and
        the control u. The update is based on the HC4revise algorithm.
        KNnown coefficient associated to unknown term should be SET TO 1
    """
    xDotO = (xDot+0.0, xDot+0.0)
    # Substract the known part of f and G
    for (k,p), (val_lb, val_ub) in knownf.items():
        if (k,p) not in overApprox.Lf:
            xDotO[0][k] = xDotO[0][k] - val_ub
            xDotO[1][k] = xDotO[1][k] - val_lb
    for(k,l,p), (val_lb, val_ub) in knownG.items():
        if (k,l,p) not in overApprox.LG:
            t_lb, t_ub = mul_i_scalar(val_lb, val_ub, uVal[l])
            xDotO[0][k] = xDotO[0][k] - t_ub
            xDotO[1][k] = xDotO[1][k] - t_lb

    # Compute the unknown term based on the current trajectory and Lipschitz constants
    foverx, Goverx = estimateUnknownFun(overApprox, xVal, xVal)

    if overApprox.verbose:
        print('***************************')
        print('uVal :  ', uVal)
        print('xVal : ', xVal)
        print('xDot - xDotKnown : ', xDot, xDotO)
        print('-------- foverx :')
        print (foverx)
        print('--------- Goverx :')
        print(Goverx)
        print('------------')

    # Obtain tighter over-approximation based on the current measurement
    updateTraj(overApprox, xDotO, uVal, foverx, Goverx, knownf, knownG)

    if overApprox.verbose:
        print('-------- foverx tight :')
        print (foverx)
        print('--------- Goverx tight :')
        print(Goverx)
        print('***************************')

    overApprox.xTraj[overApprox.nbData[1]] = xVal
    overApprox.xDot_lb[overApprox.nbData[1]] = xDotO[0]
    overApprox.xDot_ub[overApprox.nbData[1]] = xDotO[1]
    overApprox.uVal[overApprox.nbData[1]] = uVal
    for (k,p) in overApprox.Lf:
        overApprox.fCoeff[(k,p)][overApprox.nbData[1]] = knownf[(k,p)][0]
        if (k,p) in overApprox.simVarF:
            continue
        currVal = foverx[(k,p)]
        overApprox.fOverTraj_lb[(k,p)][overApprox.nbData[1]] = currVal[0]
        overApprox.fOverTraj_ub[(k,p)][overApprox.nbData[1]] = currVal[1]
    for (k,l,p) in overApprox.LG:
        overApprox.GCoeff[(k,l,p)][overApprox.nbData[1]] = knownG[(k,l,p)][0]
        if (k,l,p) in overApprox.simVarG:
            continue
        currVal = Goverx[(k,l,p)]
        overApprox.GOverTraj_lb[(k,l,p)][overApprox.nbData[1]] =  currVal[0]
        overApprox.GOverTraj_ub[(k,l,p)][overApprox.nbData[1]] =  currVal[1]

    if overApprox.nbData[0] < overApprox.xTraj.shape[0]:
        overApprox.nbData[0] += 1
    overApprox.nbData[1] = \
        (overApprox.nbData[1] + 1) % overApprox.xTraj.shape[0]

    # Check how the new data can make tighter the over-approx of the
    # already present datas
    computeInvariantTraj(overApprox)

@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def computeInvariantTraj(overApprox):
    """ Compute an invariant over the measured trajectory
        by updating the over-approximation of the apst measurements
    """
    for iterVal in range(overApprox.maxInvariantIter):
        isInvariant = True
        for i in range(overApprox.nbData[0]):
            xVal = overApprox.xTraj[i]
            xDot = (overApprox.xDot_lb[i],overApprox.xDot_ub[i])
            uVal = overApprox.uVal[i]
            foverx, Goverx = estimateUnknownFun(overApprox, xVal, xVal)
            # print(foverx, Goverx)
            changed = updateTraj(overApprox, xDot, uVal, foverx, Goverx, index=i)
            # print(changed)
            # print(foverx, Goverx)
            for (k,p) in overApprox.Lf:
                if (k,p) in overApprox.simVarF:
                    continue
                currVal = foverx[(k,p)]
                overApprox.fOverTraj_lb[(k,p)][i] = currVal[0]
                overApprox.fOverTraj_ub[(k,p)][i] = currVal[1]
            for (k,l,p) in overApprox.LG:
                if (k,l,p) in overApprox.simVarG:
                    continue
                currVal = Goverx[(k,l,p)]
                overApprox.GOverTraj_lb[(k,l,p)][i] =  currVal[0]
                overApprox.GOverTraj_ub[(k,l,p)][i] =  currVal[1]
            if changed:
                isInvariant = False
        if isInvariant:
            if overApprox.verbose:
                print('Iteration INAVRIANT: ', iterVal)
            break


@jit(nopython=True, parallel=False, fastmath=True)
def fixpoint(overApprox, x_lb, x_ub, dt, uOver_lb, uOver_ub, hOver=None):
    """ Compute an a priori enclosure, i.e. a loose over-approximation of
        the state for all time between t and  t+dt, that ensures the existence
        of a solution to the unknown dynamical system. The fixpoint
        (solution of the Picard Linderloof operator) using either the recursive
        approach or the Gronwall method developed in the paper
    """
    if hOver is None:
        foverx, Goverx = knownDynamics(x_lb, x_ub)
        # Fill in 1.0 for coefficient that haven't been entered by the user
        for (k,p) in overApprox.Lf:
            if (k,p) not in foverx:
                foverx[(k,p)] = (1.0,1.0)
        for (k,l,p) in overApprox.LG:
            if (k,l,p) not in Goverx:
                Goverx[(k,l,p)] = (1.0,1.0)
        # Compute the differential inclusion
        f_lb, f_ub, G_lb, G_ub = diffInclusion(overApprox, x_lb, x_ub, foverx, Goverx)
        hVal = add_i(f_lb, f_ub, *mul_iMv(G_lb, G_ub, uOver_lb, uOver_ub))
    else:
        hVal = hOver
    r_lb, r_ub = add_i(x_lb, x_ub, *mul_iv_0c(*hVal, dt))
    #for l in range(overApprox.maxFixpointIteration):
    countIter = 0
    while True:
        countIter += 1
        foverx, Goverx = knownDynamics(r_lb, r_ub)
        # Fill in 1.0 for coefficient that haven't been entered by the user
        for (k,p) in overApprox.Lf:
            if (k,p) not in foverx:
                foverx[(k,p)] = (1.0,1.0)
        for (k,l,p) in overApprox.LG:
            if (k,l,p) not in Goverx:
                Goverx[(k,l,p)] = (1.0,1.0)
        # COmpute the differential inclusion
        f_lb, f_ub, G_lb, G_ub = diffInclusion(overApprox, r_lb, r_ub, foverx, Goverx)
        hVal = add_i(f_lb, f_ub, *mul_iMv(G_lb, G_ub, uOver_lb, uOver_ub))
        newX_lb, newX_ub = add_i(x_lb, x_ub, *mul_iv_0c(*hVal, dt))
        isIn = True
        for i in prange(overApprox.nS):
            if not contains_i(newX_lb[i], newX_ub[i], r_lb[i], r_ub[i]):
                widR = r_ub[i] - r_lb[i]
                if widR < overApprox.zeroDiameter:
                    maxR = np.maximum(np.abs(r_lb[i]), np.abs(r_ub[i]))
                    radAdd = overApprox.widenZeroInterval \
                                if maxR <= overApprox.zeroDiameter \
                                else maxR * overApprox.fixpointWidenCoeff
                else:
                    radAdd = widR * overApprox.fixpointWidenCoeff
                r_lb[i] -= radAdd
                r_ub[i] += radAdd
                isIn = False
        if isIn:
            r_lb, r_ub = newX_lb, newX_ub
            break
    if overApprox.verbose:
        print('COUNTER: ', countIter)
    return r_lb, r_ub

@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def canApproximate(overApprox):
    """ Check if data currenlty measured can provide a
        differential inclusion that is not the set R
    """
    randX = np.random.random(overApprox.nS)
    foverx, Goverx = estimateUnknownFun(overApprox, randX, randX)
    canApproxf = True
    canApproxG = True
    for (k,p), (lb,ub) in foverx.items():
        if lb == -np.inf or ub == np.inf:
            canApproxf = False
            break
    for (k,l,p), (lb, ub) in Goverx.items():
        if lb == -np.inf or ub == np.inf:
            canApproxG = False
            break
    return canApproxf, canApproxG

@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def nextStateOverApprox(b_lb, b_ub, A1_lb, A1_ub, A2_lb, A2_ub, uVal):
    """ Compute the next state over-approximation based on the
        control-affine linearization given by b, A1, and A2
    """
    t1_lb, t1_ub = mul_Ms_i(A1_lb, A1_ub, uVal)
    t2_lb, t2_ub = mul_Ms_i(A2_lb, A2_ub, uVal)
    res_lb = np.empty(b_lb.shape[0], dtype=realN)
    res_ub = np.empty(b_lb.shape[0], dtype=realN)
    for i in prange(res_lb.shape[0]):
        res_lb[i], res_ub[i] = and_i(t1_lb[i], t1_ub[i], t2_lb[i], t2_ub[i])
        assert res_lb[i] <= res_ub[i]
    return b_lb + res_lb, b_ub + res_ub

@jit(nopython=True, parallel=False, fastmath=True)
def updateAndComputeAffineLinearization(overApprox, dt, x0, u_lb, u_ub, update=None):
    """ Update the over-approximation if required and
    """
    fx, Gx = knownDynamics(x0, x0)
    for (k,p) in overApprox.Lf:
        if ((k,p) in fx) == False:
            fx[(k,p)] = (1.0,1.0)
    for (k,l,p) in overApprox.LG:
        if ((k,l,p) in Gx) == False:
            Gx[(k,l,p)] = (1.0,1.0)
    if update is not None :
        updX, derUpdX, ctrlVal = update
        fx0, Gx0 = knownDynamics(updX, updX)
        for (k,p) in overApprox.Lf:
            if ((k,p) in fx0) == False:
                fx0[(k,p)] = (1.0,1.0)
        for (k,l,p) in overApprox.LG:
            if ((k,l,p) in Gx0) == False:
                Gx0[(k,l,p)] = (1.0,1.0)
        updateApprox(overApprox, updX, derUpdX, ctrlVal, fx0, Gx0)
    fovx, Govx = estimateUnknownFun(overApprox, x0, x0)
    fx_lb, fx_ub, Gx_lb, Gx_ub = diffInclusion(overApprox, x0, x0, fx, Gx, fovx, Govx)
    hval = add_i(fx_lb, fx_ub, *mul_iMv(Gx_lb, Gx_ub, u_lb, u_ub))

    S_lb, S_ub = fixpoint(overApprox, x0, x0, dt, u_lb, u_ub, hval)

    gradknownfSx = Dict.empty(uniTupleInt3, uniTupleFloat2)
    gradknownGsx = Dict.empty(uniTupleInt4, uniTupleFloat2)
    fknownSx, GknownSx = knownDynamics(S_lb, S_ub, gradknownfSx, gradknownGsx)
    for (k,p) in overApprox.Lf:
        if ((k,p) in fknownSx) == False:
            fknownSx[(k,p)] = (1.0,1.0)
    for (k,l,p) in overApprox.LG:
        if ((k,l,p) in GknownSx) == False:
            GknownSx[(k,l,p)] = (1.0,1.0)
    # gradknownfSx, gradknownGsx = knownDerPart(S_lb, S_ub)
    return controlAffineOverApprox(overApprox, dt, x0, S_lb, S_ub, fx_lb, fx_ub,
            Gx_lb, Gx_ub, fknownSx, GknownSx, gradknownfSx, gradknownGsx, u_lb, u_ub)

@jit(nopython=True, parallel=False, fastmath=True, cache=True)
def controlAffineOverApprox(overApprox, dt, x0, Si_lb, Si_ub, fx_lb, fx_ub,
    Gx_lb, Gx_ub, fknownSx, GknownSx, gradknownfSx, gradknownGsx, uOver_lb, uOver_ub):
    """ Compute a control affine linearization of the over-approximation
        based on interval Taylor methods
    """
    dt_2 = 0.5 * dt**2

    # Compute f and G at the a priori enclosure
    foverSx, GoverSx = estimateUnknownFun(overApprox, Si_lb, Si_ub)
    fSi_lb, fSi_ub, GSi_lb, GSi_ub = diffInclusion(overApprox, Si_lb, Si_ub,
                                        fknownSx, GknownSx, foverSx, GoverSx)
    hSi_lb, hSi_ub = add_i(fSi_lb, fSi_ub, *mul_iMv(GSi_lb, GSi_ub, uOver_lb, uOver_ub))

    # Obtain the approximation of the Jacobian of f
    Jf_lb, Jf_ub = buildJacF(overApprox.nS, overApprox.Lf, overApprox.scalingLipF,
                        overApprox.vDepF, overApprox.nvDepF, gradknownfSx,
                        foverSx, fknownSx, overApprox.bGf)

    # Obtain the approximation of the Jacobian of G
    JG_lb, JG_ub = buildJacG(overApprox.nS, overApprox.nC, overApprox.LG,
                        overApprox.scalingLipG, overApprox.vDepG, overApprox.nvDepG,
                        gradknownGsx, GoverSx, GknownSx, overApprox.bGG)

    # Compute Bi
    fx_lb *= dt
    fx_ub *= dt
    JfFsi_lb, JfFsi_ub = mul_iMv(Jf_lb, Jf_ub, fSi_lb, fSi_ub)
    JfFsi_lb *= dt_2
    JfFsi_ub *= dt_2
    b_lb = x0 + fx_lb + JfFsi_lb
    b_ub = x0 + fx_ub + JfFsi_ub

    # Compute Ai
    JG_t_lb = np.transpose(JG_lb,(0,2,1))
    JG_t_ub = np.transpose(JG_ub,(0,2,1))
    Gx_lb *= dt
    Gx_ub *= dt

    # Compute the term Gx*dt + ((Jf + JG U) GSi + JG^T fSi) * 0.5 * dt**2
    sT1_lb, sT1_ub = mul_MM( *add_i(
                        *mul_iTv(JG_lb, JG_ub, uOver_lb, uOver_ub),
                        Jf_lb, Jf_ub
                        ),
                    GSi_lb, GSi_ub)
    sT2_lb, sT2_ub = mul_iTv(JG_t_lb, JG_t_ub, fSi_lb, fSi_ub)
    A1_lb = Gx_lb + (sT1_lb + sT2_lb) * dt_2
    A1_ub = Gx_ub + (sT1_ub + sT2_ub) * dt_2

    # Compute the term Gx*dt + (Jf GSi + JG^T (fSi + GSi U)) dt**2
    sT3_lb, sT3_ub = mul_iTv(JG_t_lb, JG_t_ub, hSi_lb, hSi_ub)
    sT4_lb, sT4_ub = mul_MM(Jf_lb, Jf_ub, GSi_lb, GSi_ub)
    A2_lb = Gx_lb + (sT3_lb + sT4_lb) * dt_2
    A2_ub = Gx_ub + (sT3_ub + sT4_ub) * dt_2

    return b_lb, b_ub, A1_lb, A1_ub, A2_lb, A2_ub

@jit(nopython=True, parallel=False, fastmath=True)
def DaTaReach(overApprox, x0_lb, x0_ub, t0, nPoint, dt, uOver, uDer):
    """ Compute an over-approximation of the reachable set at time
        t0, t0+dt...,t0 + nPoint*dt.

    Parameters
    ----------
    :param x0 : Intial state
    :param t0 : Initial time
    :param nPoint : Number of point
    :param dt : Integration time
    :param uOver : interval extension of the control signal u
    :param uDer : Interval extension of the derivative of the control signal u

    Returns
    -------
    list
        a list of different time at which the over-approximation is computed
    list
        a list of the over-approximation of the state at the above time
    """
    # Save the integration time
    integTime = np.array([t0 + i*dt for i in range(nPoint+1)])
    # Save the tube of over-approximation of the reachable set
    x_lb = np.zeros((nPoint+1, x0_lb.shape[0]), realN)
    x_ub = np.zeros((nPoint+1, x0_lb.shape[0]), realN)
    # Store the initial point in the trajectory
    x_lb[0,:] = x0_lb
    x_ub[0,:] = x0_ub
    # Constant to not compute everytime
    dt_2 = (0.5* dt**2)
    for i in range(1, nPoint+1):
        # Fetch the previous over-approximation as the current uncertain state
        lastX_lb = x_lb[i-1,:]
        lastX_ub = x_ub[i-1,:]
        # COmpute the control to apply at time t and the control range
        # between t and t + dt --> Ut_lb = Ut_ub in this case
        #print(integTime[i-1], integTime[i-1])
        Ut_lb, Ut_ub = uOver(integTime[i-1], integTime[i-1])
        Ur_lb, Ur_ub = uOver(integTime[i-1], integTime[i])

        # COmpute the known part of fx and Gx
        fx, Gx = knownDynamics(lastX_lb, lastX_ub)
        # Fill in 1.0 for coefficient that haven't been entered by the user
        for (k,p) in overApprox.Lf:
            if ((k,p) in fx) == False:
                fx[(k,p)] = (1.0,1.0)
        for (k,l,p) in overApprox.LG:
            if ((k,l,p) in Gx) == False:
                Gx[(k,l,p)] = (1.0,1.0)
        fovx, Govx = estimateUnknownFun(overApprox, lastX_lb, lastX_ub)
        fx_lb, fx_ub, Gx_lb, Gx_ub = diffInclusion(overApprox, lastX_lb, lastX_ub,
                                        fx, Gx, fovx, Govx)
        # Compute the function f(x_t) + G(x_t) u_t
        hx_lb, hx_ub = add_i(fx_lb, fx_ub, *mul_iMv(Gx_lb, Gx_ub, Ut_lb, Ut_ub))

        # Compute the a priori enclosure
        rEncl_lb, rEncl_ub = fixpoint(overApprox, lastX_lb, lastX_ub, dt, Ur_lb, Ur_ub)
        # print(rEncl_lb, rEncl_ub)
        gradknownfSx = Dict.empty(uniTupleInt3, uniTupleFloat2)
        gradknownGsx = Dict.empty(uniTupleInt4, uniTupleFloat2)
        fknownSx, GknownSx = knownDynamics(rEncl_lb, rEncl_ub, gradknownfSx, gradknownGsx)
        foverSx, GoverSx = estimateUnknownFun(overApprox, rEncl_lb, rEncl_ub)
        # Fill in 1.0 for coefficient that haven't been entered by the user
        for (k,p) in overApprox.Lf:
            if ((k,p) in fknownSx) == False:
                fknownSx[(k,p)] = (1.0,1.0)
        for (k,l,p) in overApprox.LG:
            if ((k,l,p) in GknownSx) == False:
                GknownSx[(k,l,p)] = (1.0,1.0)
        fSi_lb, fSi_ub, GSi_lb, GSi_ub = diffInclusion(overApprox, rEncl_lb, rEncl_ub,
                                        fknownSx, GknownSx, foverSx, GoverSx)
        hr = add_i(fSi_lb, fSi_ub, *mul_iMv(GSi_lb, GSi_ub, Ur_lb, Ur_ub))
        # Compute G at the a priori enclosure rEncl for efficiency
        GEncl = (GSi_lb, GSi_ub)

        # Compute the gradient at the enclosure
        # gradknownfSx, gradknownGsx = gradKnownPart(rEncl_lb, rEncl_ub)

        # Obtain the approximation of the Jacobian of f
        Jf_lb, Jf_ub = buildJacF(overApprox.nS, overApprox.Lf, overApprox.scalingLipF,
                        overApprox.vDepF, overApprox.nvDepF, gradknownfSx,
                        foverSx, fknownSx, overApprox.bGf)

        # Obtain the approximation of the Jacobian of G
        JG_lb, JG_ub = buildJacG(overApprox.nS, overApprox.nC, overApprox.LG,
                        overApprox.scalingLipG, overApprox.vDepG, overApprox.nvDepG,
                        gradknownGsx, GoverSx, GknownSx, overApprox.bGG)

        # Compute the second order term (Jf + Jg Ur) * (hr)
        s_lb, s_ub = mul_iMv(
                        *add_i( Jf_lb, Jf_ub ,
                                *mul_iTv(JG_lb, JG_ub, Ur_lb, Ur_ub)
                                ),
                        *hr
                    )
        # Compute the second order term G(rEncl) * \dot{Ur}
        sx_lb, sx_ub = mul_iMv(*GEncl,
                                *uDer(integTime[i-1], integTime[i])
                                )
        x_lb[i,:] = x_lb[i-1,:] + hx_lb * dt + (s_lb + sx_lb) * dt_2
        x_ub[i,:] = x_ub[i-1,:] + hx_ub * dt + (s_ub + sx_ub) * dt_2
        # print(integTime[i], Ut_lb, x_lb[i,9], x_ub[i,9])
    return integTime, x_lb, x_ub
