import numpy as np
from numpy import float64 as realN

from numba import jit

zeroTol = 1e-8
MAX_ITERATION = 10000

betaArr = np.zeros(MAX_ITERATION+1, dtype=realN)
thet_k = 1.0
for i in range(MAX_ITERATION):
    thet_k1 = (-thet_k**2 + np.sqrt(thet_k**4 + 4*thet_k**2)) / 2
    # print(thet_k1**2, (1-thet_k1)* (thet_k**2))
    assert np.abs(thet_k1**2  - (1-thet_k1)* (thet_k**2)) < 1e-12
    betaArr[i+1] = thet_k * (1 - thet_k) /(thet_k**2 + thet_k1)
    thet_k = thet_k1

from numba import jit
@jit(nopython=True, parallel=False, fastmath=True)
def upperBoundEigen(A):
    res = 0
    for i in range(A.shape[0]):
        s = 0
        for j in range(A.shape[1]):
            s += np.abs(A[i,j])
        res = np.maximum(s, res)
    return res

@jit(nopython=True, parallel=False, fastmath=True)
def projBox(x, U_lb, U_ub):
    """ Compute the projection of the vector x onto the box
        given by the constraint U_lb <= v <= U_ub for every
        v inside that box
    """
    res = np.empty(x.shape[0], dtype=realN)
    for k in range(res.shape[0]):
        if x[k] <= U_lb[k]:
            res[k] = U_lb[k]
        elif x[k] <= U_ub[k]:
            res[k] = x[k]
        else:
            res[k] = U_ub[k]
    return res

@jit(nopython=True, parallel=False, fastmath=True)
def acceleratedProjGradWithGradRestartScheme(Q, q, U_lb, U_ub, t=1.0, eps2=1e-12):
    """ Perform Accelerated gradient descent with adaptative restart to
        compute optimal solution of the proble x^T Q x + q x when
        x is constrained on the box U_lb, U_ub
    """
    curr_t = t
    xk = np.zeros(Q.shape[0], dtype=realN)
    yk = np.zeros(Q.shape[0], dtype=realN)
    diff_xk = np.zeros(Q.shape[0], dtype=realN)
    indexRestart = 0
    nIteration = 0
    for k in range(MAX_ITERATION):
        # Save the number of iterations and the indexRestart
        nIteration += 1
        indexRestart += 1
        # Projection of yk - t * gradient at yk
        xk1 = projBox(yk - curr_t*(2*np.dot(Q, yk) + q), U_lb, U_ub)
        c_diff = xk1 - xk
        # Generalized gradient scheme
        Gyk = (1/curr_t) * (yk - xk1)
        # print(curr_t, xk1, c_diff)
        # If generalized gradient is close to zero then we stop the algorithm
        if np.dot(Gyk, Gyk) <= eps2:
            xk = xk1
            break
        # if np.dot(c_diff, c_diff) <= eps2:
        #     xk = xk1
        #     break
        # if np.dot(Gyk , xk1-xk) > 0:
        #     print('Restart', nIteration)
        #     indexRestart = 0
        #     xk = xk1
        #     yk = xk1
        #     continue
        if np.dot(Gyk , xk1-xk) > 0:
            # print('Restart', nIteration)
            indexRestart = 0
            xk = xk1
            yk = xk1
            continue
        tempV = c_diff + diff_xk
        if np.dot(tempV, tempV) <= eps2:
            curr_t *= 0.5
            # print('Shrinkk : ', nIteration)
        # Compute y_{k+1} if no restart is needed
        yk = xk1 + betaArr[indexRestart] * (xk1 - xk)
        xk = xk1
        diff_xk = c_diff
    # print(nIteration)
    return xk, np.dot(xk, np.dot(Q, xk) + q)

@jit(nopython=True, parallel=False, fastmath=True)
def solveIdealisticProblemAPGDAR(A1_lb, A1_ub, A2_lb, A2_ub, b_lb, b_ub, U_lb, U_ub,
    learnInd, Q, S, R, q, r, w1, w2, w3, epsTol):
    """ Solve the one-step optimal control problem using accelearted projected
        gradient descent.
        epsTol : provides a desired accuracy to the optimal cost |f - f*| <= sqrt(epsTol)
    """
    # Weighted center matrices A and weight vector B
    coeffSup = w3*w1 + (1-w3)*w2
    coeffInf = w3*(1-w1) + (1-w3)*(1-w2)
    midB = coeffSup * b_ub + coeffInf * b_lb
    midA = w3*w1*A1_ub + w3*(1-w1)*A1_lb + (1-w3)*w2*A2_ub + (1-w3)*(1-w2)*A2_lb

    # Compute Qt and qt matrices
    temp1 = np.dot(Q, midA) + S
    Qt = np.dot(midA.T, temp1 + S) + R
    qt = 2 * np.dot(temp1.T, midB) + np.dot(midA.T, q) + r
    pt = np.dot(midB, np.dot(Q, midB)) + np.dot(q, midB)

    # In case we are learning the function f, the control value should be 0
    retZero = True
    justOneToLearn = True
    new_U_lb = U_lb.copy()
    new_U_ub = U_ub.copy()
    for j in range(U_lb.shape[0]):
        if not (U_ub[j] >= 0 and U_lb[j] <= 0):
            retZero = False
            if j != learnInd:
                justOneToLearn = False
        else:
            if learnInd >= 0 and j != learnInd:
                new_U_lb[j] = 0
                new_U_ub[j] = 0

    if learnInd == -1 and retZero:
        return np.zeros(U_lb.shape[0]), pt

    # In case we are learning G, 0 value is imposed for all components
    # except the component with index learnInd
    # In case we are doing the learning -> Only one component (l0) of
    # the control is valid and the problem is reduced to
    # Qt_{l0}{l0} u_{l0}**2 + qt_{l0} u_{l0}
    if learnInd >= 0 and justOneToLearn:
        resU = np.zeros(U_lb.shape[0], dtype=realN)
        a = Qt[learnInd,learnInd]
        b = qt[learnInd]
        c = 0
        if np.abs(a) < zeroTol:
            if b >= 0:
                resU[learnInd] = U_lb[learnInd]
            else:
                resU[learnInd] = U_ub[learnInd]
            c = b * resU[learnInd]
        else:
            minVal = - b / (2*a)
            if U_ub[learnInd] <= minVal:
                resU[learnInd] = U_ub[learnInd]
            elif U_lb[learnInd] >= minVal:
                resU[learnInd] = U_lb[learnInd]
            else:
                resU[learnInd] = minVal
            c =  a* resU[learnInd]**2 +  b * resU[learnInd]
        return resU, c+pt

    # If we are not learning, solve the optimization problem
    tVal = 1.0/upperBoundEigen(Qt)
    xsol, csol = acceleratedProjGradWithGradRestartScheme(Qt, qt, new_U_lb, new_U_ub, tVal, epsTol)
    return xsol, csol+pt
