import numpy as np
import gurobipy as gp

# The gurobi model to optimize
mOpt = None
# THe constraints when doing excitation based
dictConstr = None
# The variable u
uVar = None
# Objective function
objPb = 0
# # Cost function parameters
# Qc, Rc, Sc, qc, rc = None, None, None, None, None

def updateControlRange(U_lb, U_ub):
    global uVar
    for i, ui in enumerate(uVar):
        ui.lb = U_lb[i]
        ui.ub = U_ub[i]

def initIdealisticProblemGrb(U_lb, U_ub):
    global dictConstr, mOpt, uVar, objPb
    # , Qc, Rc, Sc, qc, rc
    dictConstr = dict()
    # Create the Gurobi model
    mOpt = gp.Model('Midpoint Model')
    # Variable u for the midpoint problem
    uVar = [mOpt.addVar(lb=U_lb[i], ub=U_ub[i]) for i in range(U_lb.shape[0])]
    # Set the objective to zero for now
    objPb = gp.QuadExpr()
    objPb.addConstant(0)
    mOpt.setObjective(objPb)
    # Add learning constraints
    for j, uj in enumerate(uVar):
        dictConstr[-j-1] = \
                mOpt.addLConstr(gp.LinExpr([0], [uj]), gp.GRB.EQUAL, 0)
    # Update the cost function parameters
    # Qc, Rc, Sc, qc, rc = Q, R, S, q, r

def updateCost(u, A1_lb, A1_ub, A2_lb, A2_ub, b_lb, b_ub, Qc, Sc, Rc, qc, rc, w1, w2, w3):
    # global Qc, Rc, Sc, qc, rc
    # if Qc is None:
    #     Qc = np.zeros((A1_lb.shape[0], A1_lb.shape[0]))
    # if Rc is None:
    #     Rc = np.zeros((A1_lb.shape[1], A1_lb.shape[1]))
    # if Sc is None:
    #     Sc = np.zeros((A1_lb.shape[0], A1_lb.shape[1]))
    # if qc is None:
    #     qc = np.zeros(A1_lb.shape[0])
    # if rc is None:
    #     rc = np.zeros(A1_lb.shape[1])

    # Weighted center matrices A and weight vector B
    coeffSup = w3*w1 + (1-w3)*w2
    coeffInf = w3*(1-w1) + (1-w3)*(1-w2)
    midB = coeffSup * b_ub + coeffInf * b_lb
    midA = w3*w1*A1_ub + w3*(1-w1)*A1_lb + (1-w3)*w2*A2_ub + (1-w3)*(1-w2)*A2_lb

    # Compute Qt matrix
    temp1 = np.matmul(Qc, midA) + Sc
    Qt = np.matmul(midA.T, temp1 + Sc) + Rc
    qt = 2 * np.dot(temp1.T, midB) + np.dot(midA.T, qc) + rc
    # pt = np.dot(midB, np.dot(Qc, midB)) + np.dot(qc, midB)
    # print(Qt)
    # print(qt)
    # Build the quadratic expression
    res = gp.QuadExpr()
    listCoeff, listVar1, listVar2 = list(), list(), list()
    # COmpute the term u^T R u
    for i, ui in enumerate(u):
        for j, uj in enumerate(u):
            listCoeff.append(Qt[i,j])
            listVar1.append(ui)
            listVar2.append(uj)
            # res += u[i] * R[i,j] * u[j]
    res.addTerms(listCoeff, listVar1, listVar2)
    # Coeff linear term
    linearCoeff = list()
    linearVar = list()
    # COmpute the term r^T u
    for i, ui in enumerate(u):
        linearCoeff.append(qt[i])
        linearVar.append(ui)
        # res += r[i] * u[i]
    res.addTerms(linearCoeff, linearVar)
    # res.addConstant(pt)
    return res

def solveIdealisticProblemGrb(A1_lb, A1_ub, A2_lb, A2_ub, b_lb, b_ub, U_lb, U_ub,
                            learnInd, Q, S, R, q, r, w1, w2, w3, verbose=True):
    global uVar, objPb, dictConstr, mOpt
    if learnInd == -1:
        retZero = True
        for j in range(U_lb.shape[0]):
            if not (U_ub[j] >= 0 and U_lb[j] <= 0):
                retZero = False
                break
        if retZero:
            return np.zeros(U_lb.shape[0]), 0.0

    for i, ui in enumerate(uVar):
        ui.lb = U_lb[i]
        ui.ub = U_ub[i]

    objPb = updateCost(uVar, A1_lb, A1_ub, A2_lb, A2_ub, b_lb, b_ub,
                        Q, S, R, q, r, w1, w2, w3)
    mOpt.setObjective(objPb)

    learnConstr = np.zeros(U_lb.shape[0])
    if learnInd >= 0:
        learnConstr = np.full(U_lb.shape[0],1)
        learnConstr[learnInd] = 0

    for j, uj in enumerate(uVar):
        if (learnConstr[j] == 0) or\
            (learnConstr[j] == 1 and (U_ub[j] >= 0 and U_lb[j] <= 0)):
            c3 = dictConstr[-j-1]
            mOpt.chgCoeff(c3, uj, learnConstr[j])
    mOpt.Params.OutputFlag = verbose
    mOpt.optimize()
    currCost = objPb.getValue()
    return np.array([uVal.x for uVal in uVar]), currCost
