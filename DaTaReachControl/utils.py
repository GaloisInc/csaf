import numpy as np
from scipy.integrate import solve_ivp
import scipy.optimize as spo

def synthTrajNonAffine(dynFun, xInitT, evalTime, atol=1e-10, rtol=1e-10):
    """Compute a solution of the ODE dynFun(t,x) with the initial
    state given by xInit, and evalTime contains the different time at which the
    state and its derivative (xDotVal) should be returned.
    """
    xInit = np.squeeze(xInitT)
    nState = xInit.shape[0]
    t0 = evalTime[0] # Init time
    tend = evalTime[-1] # End time
    # Numerical solution of the ODE
    solODE = solve_ivp(dynFun, t_span=(t0,tend), y0=xInit,
                        t_eval=evalTime, atol=atol, rtol=rtol)
    xDotVal = np.zeros((evalTime.shape[0], nState))
    # Compute xDot
    for i in range(evalTime.shape[0]):
        xDotVal[i,:] = dynFun(solODE.t[i], solODE.y[:,i])
    return solODE.t, solODE.y.T, xDotVal

def synthTraj(fFun, GFun, uFun, xInitT, evalTime, atol=1e-10, rtol=1e-10):
    """Compute a solution of the ODE fFun(x) + Gfun(x) u(t) with the initial
    state given by xInit, and evalTime contains the different time at which the
    state and its derivative (xDotVal) should be returned.
    For this function xInit, fFun, and GFun takes as input (n,) array
    and returns (n,) array. uFun return a (n,1) array.
    """
    xInit = np.squeeze(xInitT)
    nState = xInit.shape[0]
    t0 = evalTime[0] # Init time
    tend = evalTime[-1] # End time
    def dynFun (t , x):
        uFunVal = np.squeeze(uFun(t))
        return fFun(x) + np.dot(GFun(x), uFunVal)
    # Numerical solution of the ODE
    solODE = solve_ivp(dynFun, t_span=(t0,tend), y0=xInit,
                        t_eval=evalTime, atol=atol, rtol=rtol)
    xDotVal = np.zeros((evalTime.shape[0], nState))
    # Compute xDot
    for i in range(evalTime.shape[0]):
        xDotVal[i,:] = dynFun(solODE.t[i], solODE.y[:,i])
    return solODE.t, solODE.y.T, xDotVal


def synthNextState(fFun, GFun, samplingTime=0.1, atol=1e-10, rtol=1e-10):
    """Compute the value of the state at time t+samplingTime of the dynamics
    fFun(x) + Gfun(x) uVal given the current state xVal and the constant
    control uVal. For this function x, currX , fFun, and GFun takes as input (n,)
    array and returns (n,) array. uFun return a (n,1) array.

    Returns
    -------
    a function taking as input the current state and returning the state
    at time t+samplingTime and the derivatives between t and t+samplingTime
    """
    def nextState(currXT, currUT, dt=samplingTime):
        currX = np.squeeze(currXT)
        currU = np.squeeze(currUT)
        def dynFun (t, x):
            return fFun(x) + np.dot(GFun(x), currU)
        solODE = solve_ivp(dynFun, t_span=(0,dt), y0=currX, \
                                atol=atol, rtol=rtol)
        return solODE.y[:,-1], dynFun(solODE.t[0],currX)
    return nextState

def synthNextStateNonAffine(dynFunAux, samplingTime=0.1, atol=1e-10, rtol=1e-10):
    """Compute the value of the state at time t+samplingTime of the dynamics
    dynFunAux(t, xVal, uVal) given the current state xVal and the constant
    control uVal. For this function x, currX , fFun, and GFun takes as input (n,)
    array and returns (n,) array. uFun return a (n,1) array.

    Returns
    -------
    a function taking as input the current state and returning the state
    at time t+samplingTime and the derivatives between t and t+samplingTime
    """
    def nextState(currXT, currUT, dt=samplingTime):
        currX = np.squeeze(currXT)
        currU = np.squeeze(currUT)
        def dynFun (t, x):
            return dynFunAux(t, x, currU)
        solODE = solve_ivp(dynFun, t_span=(0,dt), y0=currX, \
                                atol=atol, rtol=rtol)
        return solODE.y[:,-1], dynFun(solODE.t[0],currX)
    return nextState

def generateTraj(fFun, GFun, xInitT, uSeq, dt, nbPoint=1):
    """
    Generate a trajectory/measurements based on the control sequences
    uSeq. xInit is the initial state and dt is the sampling time when
    nbPoint is 1 and dt/nbPoint gives the general sampling time.
    uSeq[i,:] is applied every dt.
    The last nbPoint points are obtained by applying control value
    uSeq[-1,:]
    """
    xInit = np.squeeze(xInitT)
    newDt = dt / nbPoint
    nextState = synthNextState(fFun, GFun, newDt)
    currX = np.zeros((nbPoint*uSeq.shape[0]+1, xInit.shape[0]), dtype=np.float64)
    currX[0,:] = xInit
    currXdot = np.zeros((nbPoint*uSeq.shape[0], xInit.shape[0]))
    currInd = 0
    for i in range(uSeq.shape[0]):
        for j in range(nbPoint):
            nX , cXdot = nextState(currX[currInd,:], uSeq[i,:])
            currXdot[currInd,:] = cXdot
            currX[currInd+1,:] = nX
            currInd += 1
    return currX, currXdot

def generateTrajNonAffine(dynFun, xInitT, uSeq, dt, nbPoint=1):
    """
    Generate a trajectory/measurements based on the control sequences
    uSeq. xInit is the initial state and dt is the sampling time when
    nbPoint is 1 and dt/nbPoint gives the general sampling time.
    uSeq[i,:] is applied every dt.
    The last nbPoint points are obtained by applying control value
    uSeq[-1,:]
    """
    xInit = np.squeeze(xInitT)
    newDt = dt / nbPoint
    nextState = synthNextStateNonAffine(dynFun, newDt)
    currX = np.zeros((nbPoint*uSeq.shape[0]+1, xInit.shape[0]), dtype=np.float64)
    currX[0,:] = xInit
    currXdot = np.zeros((nbPoint*uSeq.shape[0], xInit.shape[0]))
    currInd = 0
    for i in range(uSeq.shape[0]):
        for j in range(nbPoint):
            nX , cXdot = nextState(currX[currInd,:], uSeq[i,:])
            currXdot[currInd,:] = cXdot
            currX[currInd+1,:] = nX
            currInd += 1
    return currX, currXdot
