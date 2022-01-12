import numpy as np
import copy

import gurobipy as gp

# Store the different models to optimize
mModels = dict()

# Store the constraints of the original problem
dictConstr = dict()

# Store the control, state and associate sign of control
dictU = dict()

def partitionRange(U_lb, U_ub):
    """ Partition the initial range set U init
        positive and negative orthant subdivisions
    """
    listConstr = [dict()]
    for i in range(U_ub.shape[0]):
        if U_lb[i] >= 0 or U_ub[i] <= 0:
            for d in listConstr:
                d[i] = U_lb[i] >= 0
            continue
        list_p = copy.deepcopy(listConstr)
        for d in list_p:
            d[i] =  True
        list_n = copy.deepcopy(listConstr)
        for d in list_n:
            d[i] = False
        listConstr = list_p + list_n
    return listConstr

def getQuadraticCost(x_var, u, Q, S, R, q, r):
    """ COmpute the quaadratic cost function given by
        [x_var u]^T [Q S; S^T R] [x_var u] + q^T x + r^T u
    """
    res = gp.QuadExpr()
    listCoeff, listVar1, listVar2 = list(), list(), list()
    if Q is not None:
        # Compute th term x^T Q x
        for i, xi in enumerate(x_var):
            for j, xj in enumerate(x_var):
                listCoeff.append(Q[i,j])
                listVar1.append(xi)
                listVar2.append(xj)
                # res += x_var[i] * Q[i,j] * x_var[j]
    if S is not None:
        # Compute the term 2 x^T S u
        for i, xi in enumerate(x_var):
            for j, uj in enumerate(u):
                listCoeff.append(S[i,j])
                listVar1.append(xi)
                listVar2.append(uj)
                # res += 2 * x_var[i] * S[i,j] * u[j]
    if R is not None:
        # COmpute the term u^T R u
        for i, ui in enumerate(u):
            for j, uj in enumerate(u):
                listCoeff.append(R[i,j])
                listVar1.append(ui)
                listVar2.append(uj)
                # res += u[i] * R[i,j] * u[j]
    res.addTerms(listCoeff, listVar1, listVar2)
    # Coeff linear term
    linearCoeff = list()
    linearVar = list()
    if q is not None:
        # Compute the term q^T x
        for i, xi in enumerate(x_var):
            linearCoeff.append(q[i])
            linearVar.append(xi)
            # res += q[i] * x_var[i]
    if r is not None:
        # COmpute the term r^T u
        for i, ui in enumerate(u):
            linearCoeff.append(r[i])
            linearVar.append(ui)
            # res += r[i] * u[i]
    res.addTerms(linearCoeff, linearVar)
    return res

def initOptimisticProblemGrb(nS, nC, Q, S, R, q, r, U_lb, U_ub):
    """ Build the initial Gurobi model based on separating the
        control input U into positve and negative orthants.
        The separation creates 2^q problems to solve, q is the number
        of component of the control that can take both positve and
        negative values.
    """
    global mModels, dictConstr, dictU
    mModels.clear()
    dictConstr.clear()
    dictU.clear()
    # Obtain the different partitions from the range of U
    listConstr = partitionRange(U_lb, U_ub)
    # Create the optimization problem variables
    counterPb = 0

    for d in listConstr:
        # Create the gurobi model
        mOpt = gp.Model('Optimistic Model No' + str(counterPb))
        # Variables of the sub problems are u and x_var
        u = [ mOpt.addVar(lb=-gp.GRB.INFINITY) for i in range(nC) ]
        x_var = [ mOpt.addVar(lb=-gp.GRB.INFINITY) for i in range(nS)]
        # Constraints on the variable x_var
        for i, xi in enumerate(x_var):
            coeffSup = [0 for l in range(nC)]
            coeffInf = [0 for l in range(nC)]
            dictConstr[(counterPb, i)] = \
                (mOpt.addLConstr(gp.LinExpr(coeffSup + [-1], u + [xi]),
                                gp.GRB.GREATER_EQUAL, 0),
                mOpt.addLConstr(gp.LinExpr(coeffInf + [-1], u + [xi]),
                                gp.GRB.LESS_EQUAL, 0),
                mOpt.addLConstr(gp.LinExpr(coeffSup + [-1], u + [xi]),
                                gp.GRB.GREATER_EQUAL, 0),
                mOpt.addLConstr(gp.LinExpr(coeffInf + [-1], u + [xi]),
                                gp.GRB.LESS_EQUAL, 0),
                )
        # For constraints on the control variable u
        for j, uj in enumerate(u):
            if d[j]:
                uj.lb = np.maximum(0, U_lb[j])
                uj.ub = np.maximum(0, U_ub[j])
            else:
                uj.lb = np.minimum(0, U_lb[j])
                uj.ub = np.minimum(0, U_ub[j])
            # When learning the dynamics, encode some that some components
            # of the control must be zero
            dictConstr[(counterPb,-j-1)] = \
                mOpt.addLConstr(gp.LinExpr([0], [uj]), gp.GRB.EQUAL, 0)
        # Save the input and states variables
        dictU[counterPb] = (u, x_var, d)
        # Create the cost function for the current subproblems
        costVal = getQuadraticCost(x_var, u, Q, S, R, q, r)
        mOpt.setObjective(costVal)
        mModels[counterPb] = (mOpt, costVal)
        # Increment the counter variable
        counterPb = counterPb + 1

def updateCost(Q, S, R, q, r):
    """ Update the cost function, need to be applied only once when
        the parameters Q, R , Q, q, and r have changed
    """
    global dictU, mModels
    for nbProblem, (u, x_var, d) in dictU.items():
        (mOpt, costVal) = mModels[nbProblem]
        costVal = getQuadraticCost(x_var, u, Q, S, R, q, r)
        mOpt.setObjective(costVal)
        mModels[nbProblem] = (mOpt, costVal)

def solveOptimisticProblemGrb(A1_lb, A1_ub, A2_lb, A2_ub, b_lb, b_ub,
            U_lb, U_ub, learnInd, verbose=True):
    """ Given the values of Al, bl, the possible imposed learnConstr,
        compute a solution of the near-optimal control problem
    """
    # In case we are learning the function f, the control value should be 0
    if learnInd == -1:
        retZero = True
        for j in range(U_lb.shape[0]):
            if not (U_ub[j] >= 0 and U_lb[j] <= 0):
                retZero = False
                break
        if retZero:
            return np.zeros(U_lb.shape[0]), 0.0

    global dictU, mModels, dictConstr
    minCost = np.inf
    uOpt = None

    learnConstr = np.zeros(U_lb.shape[0])
    if learnInd >= 0:
        learnConstr = np.full(U_lb.shape[0],1)
        learnConstr[learnInd] = 0

    for nbProblem, (u, x_var, d) in dictU.items():
        (mOpt, costVal) = mModels[nbProblem]
        # Constraints on the control variable u
        nextIter = False
        for j, uj in enumerate(u):
            if d[j]:
                if  U_ub[j] < 0:
                    nextIter = True
                    break
                uj.lb = np.maximum(0, U_lb[j])
                uj.ub = np.maximum(0, U_ub[j])
            else:
                if U_lb[j] > 0:
                    nextIter = True
                    break
                uj.lb = np.minimum(0, U_lb[j])
                uj.ub = np.minimum(0, U_ub[j])
            if (learnConstr[j] == 0) or\
                (learnConstr[j] == 1 and (U_ub[j] >= 0 and U_lb[j] <= 0)):
                c3 = dictConstr[(nbProblem, -j-1)]
                mOpt.chgCoeff(c3, uj, learnConstr[j])
        if nextIter:
            continue
        for i , x_i in enumerate(x_var):
            (c1, c2, c3, c4) = dictConstr[(nbProblem,i)]
            c1.RHS = -b_ub[i]
            c2.RHS = -b_lb[i]
            c3.RHS = -b_ub[i]
            c4.RHS = -b_lb[i]
            for j, uj in enumerate(u):
                if d[j]:
                    mOpt.chgCoeff(c1, uj, A1_ub[i,j])
                    mOpt.chgCoeff(c2, uj, A1_lb[i,j])
                    mOpt.chgCoeff(c3, uj, A2_ub[i,j])
                    mOpt.chgCoeff(c4, uj, A2_lb[i,j])
                else:
                    mOpt.chgCoeff(c1, uj, A1_lb[i,j])
                    mOpt.chgCoeff(c2, uj, A1_ub[i,j])
                    mOpt.chgCoeff(c3, uj, A2_lb[i,j])
                    mOpt.chgCoeff(c4, uj, A2_ub[i,j])
        mOpt.Params.OutputFlag = verbose
        mOpt.optimize()
        currCost = costVal.getValue()
        if minCost > currCost:
            minCost = currCost
            uOpt = u
    return np.array([ uVal.x for uVal in uOpt]), minCost
