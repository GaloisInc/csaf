#
import gurobipy as gp
import numpy as np

from DaTaReachControl.reach import one_step_pred

# The gurobi model to optimize
mOpt = None
# THe constraints each component of the next state
dictConstr = None
dictGradBound = None
# The variable delta u and delta x, plus the slack variable sInfes
uVar = None
xVar = None
sInfes = None
sInfesPos = None
mLambda, mDevU, mRho0, mRho1, mRho2, mAlpha, mBeta = 10, 0.01, 1e-5, 0.25, 0.7, 2.0, 3.5
# Objective function
objPb = 0
# Cost function parameters
Qc, Sc, Rc, qc, rc = None, None, None, None, None
mGradUbound = None


def setCostFunction(Q=None, S=None, R=None, q=None, r=None):
    """
    Store the matrices associated to the cost function
    """
    global Qc, Sc, Rc, qc, rc
    Qc, Sc, Rc, qc, rc = Q, S, R, q, r

def setQuadraticCost(lam, xpast, upast, discount=1.0 ,Q=None, S=None, R=None, q=None, r=None):
    """ COmpute the quaadratic cost function given by
        [x_var+xpast u+upast]^T [Q S; S^T R] [x_var+xpast u+upast] + q^T (x_var+xpast) + r^T (u_var+u_past)
    """
    # Load the Gurobi optimization problem variables
    global mOpt
    global uVar
    global xVar
    global sInfesPos

    # Initialize the cost function
    res = 0
    for i, x_var in xVar.items():
        # temp variable
        res_t = 0.0
        # Obtain the symbolic vector variables representing x_k and u_{k-1}
        xVect = gp.MVar(x_var)
        uVect = gp.MVar(uVar[i-1])
        # x and u value at the previous iteration
        xPastVect = xpast[i]
        uPastVect = upast[i-1]
        if Q  is not None:
            res_t += xVect @ Q @ xVect + 2 *(xPastVect@Q) @ xVect + np.dot(xPastVect, np.dot(Q, xPastVect))
        if R is not None:
            res_t += uVect @ R @ uVect + 2 *(uPastVect@R) @ uVect + np.dot(uPastVect, np.dot(R, uPastVect))
        if S is not None:
            res_t +=  2 * (xVect @ S @ uVect + (xPastVect@S) @ uVect + (uPastVect@ S.T) @ xVect + np.dot(xPastVect, np.dot(S, uPastVect)))
        if q is not None:
            res_t += q @ xVect + np.dot(q,xPastVect)
        if r is not None:
            res_t += r @ uVect + np.dot(r,uPastVect)
        # Add the slack variable for dynamics constraints
        res += discount**(i-1) * (res_t + np.full(len(x_var), lam) @ gp.MVar(sInfesPos[i-1]))
    mOpt.setObjective(res)
    return res

def computeActualPenalizedCost(lam, xk, uk, fk, discount, Q, S, R, q, r):
    """ Compute the quaadratic cost function given by
        [x_k]^T [Q S; S^T R] [x_k] + q^T x_k + r^T u_k + lam * \sum_{i} |xk_{i+1} - fk_{i}|
        where fk = next state computed using uk and xk
        :param lam : penalization params for non satisfiability of dynamics equation
        :xk : state in the last iterate
        :uk : control in the last iterate
        :fk : f(uk,xk) computed by calling DaTaReach
        :Q,S,R,q,r : Quadratic cost function of the prolem
    """
    # Initialize the cost function
    res = 0
    for tStep, x_val in xk.items():
        res_t = 0.0
        if Q  is not None:
            res_t += np.dot(x_val, np.dot(Q, x_val))
        if R is not None:
            res_t += np.dot(uk[tStep-1], np.dot(R, uk[tStep-1]))
        if S is not None:
            res_t +=  2 * np.dot(x_val, np.dot(S, uk[tStep-1]))
        if q is not None:
            res_t += np.dot(q, x_val)
        if r is not None:
            res_t += np.dot(r, uk[tStep-1])
        # Add the penalization due to not satisfying the constraint x_{i+1} = f(x_i,u_i)
        res_t += lam * np.sum(np.abs(fk[tStep-1]- x_val))
        res += discount**(tStep-1) * res_t
    return res

def createMultiStepIdeProblemGrb(Nstep, nState, nControl, gradUbound=None, lamda=10, devU=0.01, rho0=1e-5, rho1 = 0.25, rho2= 0.7, alpha=2.0, beta=3.5):
    """
    Create the N step receding horizon control approximate problem.
    :param gradUbound : Gradient bound on the control |u_{i+1} - u_i| <= gradUBound
    :param lamda : Penalty cost for violating the state evolution constraint
    :param devU : Initial deviation between two consecutive iterates in control
    :param rho0 : rho0 < rho1 <<1, Threshold to consider the aproximation too inaccurate -> shrink
    :param rho1 : rho1 < rho2, Threshold to consider the approximation good (so accept current step) but need to shrink the trust region
    :param rho2 : 0 << rho2 < 1, Threshold to consider the approx good and increment the trust region
    :param alpha : alpha > 1, factor used to shrink the trust region
    :param beta : beta > 1, factor used to inflate the trust region
    """

    # LOad the global variables
    global dictGradBound, dictConstr, mOpt, uVar, xVar, sInfes, sInfesPos, objPb
    global mLambda, mDevU, mRho0, mRho1, mRho2, mAlpha, mBeta, mGradUbound

    # Dynamics constraint dictionary
    dictConstr = dict()

    # Create the Gurobi model
    mOpt = gp.Model('Nstep MPC Idealistic controller')

    # Variable diff u between two iterates for control at different time step
    uVar = dict()
    for tStep in range(Nstep):
        uVar[tStep]  = [mOpt.addVar(lb=-devU, ub=devU, name='W[{}][{}]'.format(tStep,i)) for i in range(nControl)]

    # Variable diff x between two iterates for the state at different time step
    xVar = dict()
    for tStep in range(Nstep):
        xVar[tStep+1] = [mOpt.addVar(lb=-gp.GRB.INFINITY, name='D[{}][{}]'.format(tStep+1,i)) for i in range(nState)]

    # Create the slack variables for artificial infeasibility -> Penalty to not satisfy equality constraint
    sInfes = dict()
    for tStep in range(Nstep):
        sInfes[tStep] = [mOpt.addVar(lb=-gp.GRB.INFINITY, name='V[{}][{}]'.format(tStep,i)) for i in range(nState)]

    # Create the slack variables for absolute value of artificial infeasibility -> Used in the cost function
    sInfesPos = dict()
    for tStep in range(Nstep):
        sInfesPos[tStep] = [mOpt.addVar(lb=0, name='|V|[{}][{}]'.format(tStep,i)) for i in range(nState)]

    # Impose the constraint on x_1 and x_0 relation ->  x^k_{1} + xVar[1] = f(x^k_0, u^k_{0}) + B^k_{0} uVar[0] + sInfes[0]
    coeffS = [0 for i in range(nState)]
    coeffU = [0 for i in range(nControl)]
    # Add the first constraint since x0 is known
    for i, xV, sInfes_i in zip(range(nState), xVar[1], sInfes[0]):
        dictConstr[(1,i)] = mOpt.addLConstr(\
                                        gp.LinExpr([-1, 1] + coeffU,\
                                                    [xV, sInfes_i]+uVar[0]),\
                                        gp.GRB.EQUAL,\
                                        0,\
                                        name='C_Dyn[{}][{}]'.format(1, i))

    # Add the rest of the constraints -> x^k_{i+1} + xVar[i+1] = f(x^k_i, u^k_{i}) + A^k_{i} xVar[i] + B^k_{i} uVar[i] + sInfes[i]
    for tStep in range(1,Nstep):
        for i, xV, sInfes_i in zip(range(nState), xVar[tStep+1], sInfes[tStep]):
            dictConstr[(tStep+1,i)] =  mOpt.addLConstr(\
                                        gp.LinExpr([-1,1] + coeffS + coeffU,\
                                                    [xV, sInfes_i]+xVar[tStep]+uVar[tStep]),\
                                        gp.GRB.EQUAL,\
                                        0,\
                                        name='C_Dyn[{}][{}]'.format(tStep+1, i))

    # Impose the constraint on the absolute value of the artificial infeasibility |sInfes| <= sInfesPos
    for tStep in range(Nstep):
        for i, sInfesPosv, sInfesv in zip(range(nState), sInfesPos[tStep], sInfes[tStep]):
            mOpt.addConstr(sInfesv <= sInfesPosv, name='C+_|V|[{}][{}]'.format(tStep,i))
            mOpt.addConstr(sInfesv >= -sInfesPosv, name='C-_|V|[{}][{}]'.format(tStep,i))

    # Add the gradient bound constraint if given
    dictGradBound = dict()
    if gradUbound is not None:
        mGradUbound = gradUbound.copy()
        for tStep in range(1, Nstep):
            for k, bVal, Unext_k, Upast_k in zip(range(nControl), gradUbound, uVar[tStep],uVar[tStep-1]):
                dictGradBound[(tStep,k)] = (\
                        mOpt.addConstr(Unext_k - Upast_k <= bVal, name='C+_gradB[{}][{}]'.format(tStep, k)),\
                        mOpt.addConstr(Unext_k - Upast_k >= -bVal, name='C-_gradB[{}][{}]'.format(tStep, k)))

    # Set the objective to zero for now
    objPb = gp.QuadExpr()
    objPb.addConstant(0)
    mOpt.setObjective(objPb)

    # Update the problem if needed
    mOpt.update()

    # Set the parameters of this optimzation model
    mLambda, mDevU, mRho0, mRho1, mRho2, mAlpha, mBeta = lamda, devU, rho0, rho1, rho2, alpha, beta


def update_opt_step_k(trustRegion, xk, uk, Ak, Bk, fk, xRange, uRange):
    """
    Update the optimization problem based on the given xk, uk, estimates of
    the jacobians Ak, Bk and the range of x and u
    :trustRegion : the trust region for the control deviation uVar
    :xk : past iterate of the state
    :uk : past iterate of the control
    :Ak : Jacobian of the unknown x_{i+1} with respect to x computed at xk and uk
    :Bk : Jacobian of the unknown x_{i+1} with respect to u computed at xk and uk
    :fk : estimate of f(xk,uk) where x_i+1 = f(x_i, u_i)
    :xRange : Dict containing the full range of x at each time step
    :uRange : Dict containing the full range of u at each time step
    """
    global mOpt
    global dictConstr, dictGradBound
    global xVar
    global uVar
    global mGradUbound
    # Set the constraint limits on xk -> xVar + xk \in xRange
    for tStep, xvar in xVar.items():
        for i, x_i in enumerate(xvar):
            x_i.lb = xRange[tStep][0][i] - xk[tStep][i]
            x_i.ub = xRange[tStep][1][i] - xk[tStep][i]
    # Set the constraint limits on uk and update the trust region -> uVar + uk \in uRange
    for tStep, uvar in uVar.items():
        for k, u_k in enumerate(uvar):
            u_k.lb = np.maximum(uRange[tStep][0][k] - uk[tStep][k], -trustRegion)
            u_k.ub = np.minimum(uRange[tStep][1][k] - uk[tStep][k], trustRegion)

    # Set the constraints corresponding to the dynamics
    for (t1, i), c_i in dictConstr.items():
        # Set the right hand side of the constraints
        c_i.RHS = xk[t1][i] - fk[t1-1][i]
        for j, u_j in enumerate(uVar[t1-1]):
            mOpt.chgCoeff(c_i, u_j, Bk[t1-1][i,j])
        # At time step 1 there is no constraint involving Ak since d0 =0
        if t1 == 1:
            continue
        # Impose the constraint Ak xVar
        for j, x_j in enumerate(xVar[t1-1]):
            mOpt.chgCoeff(c_i, x_j, Ak[t1-1][i,j])
    # Update the gradBound if given
    for (tStep,k), (c1_k, c2_k) in dictGradBound.items():
        c1_k.RHS = mGradUbound[k] + (uk[tStep-1][k] - uk[tStep][k] )
        c2_k.RHS = - mGradUbound[k] +  (uk[tStep-1][k] - uk[tStep][k])

def sequential_optimization(x0, dt, uRange, overApprox, discount=1.0, weight = 0.5, epstol=1e-5, maxIter=10, verbose=False):
    """
    Perform sequential convexification of the data-driven Nstep optimal control problem
    :param x0 : the current state of the system
    :param dt : the time step
    :param overApprox : the object proving the differential inclusion
    :param uRange :  Dict containg (u_lb, u_ub) at each time step
    :param weight : a parameeter to pick an idealistic trajectory in the reachable set
    :param maxIter : The maximum number of iteration of the algorithm
    :param epsTol : The desired tolerance of the optimal solution
    """
    # print(weight, epstol, maxIter)
    # print(uRange)
    # Load the module global variables
    global xVar, uVar, sInfes, mOpt, Qc, Sc, Rc, qc, rc, mLambda, mDevU, mRho0, mRho1, mRho2, mAlpha, mBeta

    # Sanity check
    assert len(uRange) == len(uVar), 'uRange does not have the appropriate dimension'

    # Get the time horizon
    Nstep = len(uVar)
    nState = x0.shape[0]

    # Define the k-th step dict to store x-xk
    dictXk = dict()
    xRange = dict()

    # Compute an over-approximation of the reachable set using the full uRange to have xRange
    lastX_lb = x0.copy()
    lastX_ub = x0.copy()
    for tStep in range(Nstep):
        xnext_lb, xnext_ub, tA1, tA2 , tB1, tB2 = one_step_pred(overApprox, lastX_lb, lastX_ub, dt, uRange[tStep][0], uRange[tStep][1])
        xRange[tStep+1] = (xnext_lb, xnext_ub)
        lastX_lb, lastX_ub = xnext_lb, xnext_ub

    # Taking midpoint as 'random' intiialization of the optimization algorithm
    xk = dict()
    uk = dict()
    for tStep in range(Nstep):
        xk[tStep+1] = 0.5 * (xRange[tStep+1][0] + xRange[tStep+1][1])
        uk[tStep] = 0.5 * (uRange[tStep][0] + uRange[tStep][1])

    # Get the Jacobian of x_i+1 wrt x_i and u_i on the chosen xk and uk
    Aik = dict()
    Bik = dict()
    fik = dict()
    for tStep in range(Nstep):
        xnext_lb, xnext_ub, Ai_lb, Ai_ub, Bi_lb, Bi_ub = \
            one_step_pred(overApprox, x0 if tStep==0 else xk[tStep], x0 if tStep==0 else xk[tStep],\
                                dt, uk[tStep], uk[tStep])
        Aik[tStep] = weight * Ai_lb + (1-weight) * Ai_ub
        Bik[tStep] = weight * Bi_lb + (1-weight) * Bi_ub
        fik[tStep] = weight * xnext_lb + (1-weight) * xnext_ub

    # Initialize the trust region scalar
    rk = mDevU
    fikn_int = dict()

    # Start the sequential convex optimization loop
    for iterV in range(maxIter):
        # Now update the optimization problem and solve the otpimization problem
        update_opt_step_k(rk, xk, uk, Aik, Bik, fik, xRange, uRange)

        # Update the cost function
        setQuadraticCost(mLambda, xk, uk, discount, Qc, Sc, Rc, qc, rc)

        # Solve the convex subproblem
        mOpt.Params.OutputFlag = False
        mOpt.optimize()

        # mOpt.display()

        # Extract the newt states xk and control uk and deviations
        xkn = dict()
        ukn = dict()
        slackn = dict()
        for  tStep in range(Nstep):
            xkn[tStep+1] = xk[tStep+1] + np.array([solx.x for solx in xVar[tStep+1]])
            ukn[tStep] = uk[tStep] + np.array([solu.x for solu in uVar[tStep]])
            slackn[tStep] = np.array([solv.x for solv in sInfes[tStep]])

        # Compute the new Jacobian wrt to the new x and u obtained while optimizaing
        Aikn = dict()
        Bikn = dict()
        fikn = dict()

        # Get the next state using the computed new control and states
        for tStep in range(Nstep):
            xnext_lb, xnext_ub, Ai_lb, Ai_ub, Bi_lb, Bi_ub = \
                    one_step_pred(overApprox, x0 if tStep==0 else xkn[tStep], x0 if tStep==0 else xkn[tStep],\
                        dt, ukn[tStep], ukn[tStep])
            Aikn[tStep] = weight * Ai_lb + (1-weight) * Ai_ub
            Bikn[tStep] = weight * Bi_lb + (1-weight) * Bi_ub
            fikn[tStep] = weight * xnext_lb + (1-weight) * xnext_ub
            fikn_int[tStep+1] = (xnext_lb, xnext_ub)

        # Get the optimal cost
        Lk1 = mOpt.objVal
        # Get the actual cost and the cost at the pas uk and xk
        Jk = computeActualPenalizedCost(mLambda, xk, uk, fik, discount, Qc, Sc, Rc, qc, rc)
        # Get the cost of the next ukn and xkn
        Jk1 = computeActualPenalizedCost(mLambda, xkn, ukn, fikn, discount, Qc, Sc, Rc, qc, rc)

        deltaJk = Jk - Jk1
        deltaLk = Jk - Lk1

        if verbose:
            print('Iter ---->', iterV, rk)
            print('deltaJk', deltaJk)
            print('deltaLk', deltaLk)
            print('Jk, Jk1', Jk, Jk1)

        # print(slackn)

        # If the solution is good enough
        if np.abs(deltaJk) < epstol:
            xk = xkn
            uk = ukn
            break

        # If the approximation  is too loose -> contract the trust region and restart the optimization
        rhok = deltaJk / deltaLk
        if rhok < mRho0:
            rk = rk / mAlpha
            continue

        # Accept this step
        xk = xkn
        uk = ukn
        Aik = Aikn
        Bik = Bikn
        fik = fikn

        # Update the Trust region if needed
        rk =  rk / mAlpha if rhok < mRho1 else (mBeta * rk if mRho2 < rhok else rk)
        if rk <= 1e-5:
            break
        # rk = np.maximum(rk, 1e-5) # Don't consider too low value of the trust region

    return uk, fikn_int, Jk1
