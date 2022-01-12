import numpy as np
from numpy import float64 as realN

# Import the optimizations tools
import DaTaReachControl.idealisticControlAPGDAR as ideAPG

from numba.typed import Dict
from DaTaReachControl.reach import uniTupleInt3, uniTupleInt4, uniTupleFloat2

# Import the necessary function from reach module
from .interval import norm_i, add_i, mul_iMv
from DaTaReachControl.reach import initOverApprox, diffInclusion, estimateUnknownFun,\
                    updateApprox, nextStateOverApprox,controlAffineOverApprox,\
                    canApproximate, fixpoint, updateAndComputeAffineLinearization

# Import the necessary types
# from DaTaReachControl.reach import knownDynamics
from DaTaReachControl.overapprox_functions import tolChange

from numba import jit

# The different Optimization method
OPTIMISTIC_GRB = 0 # optimistic problem using Gurobi
IDEALISTIC_GRB = 1 # Idealistic problem using Gurobi
IDEALISTIC_APG = 2 # Idealistic problem using Approximated proximal
                   # gradient with restart scheme


class DaTaControl:
    """
    Main class for the systhesis of a 1-step "optimal" control
    Params provide the extra parameters for the solver used in the optimization
    problem.
    If params is None, the idealisticAPG problme is chosen by default.
    with the weight w1, w2, w3 taken as 0.7 each with a stopiing criteria of 0.8:
        params = (IDEALISTIC_APG, 0.7, 0.7, 0.7, 1e-8)
        params[0] is the method to use between OPTIMISTIC_GRB, IDEALISTIC_GRB, IDEALISTIC_APG
        params[1],params[2],params[3] represents the weighted if idealistic solved
    """
    def __init__(self, dt, nS, nC, listInfoF, listInfoG, U_lb, U_ub,
        Q=None, S=None, R=None, q=None, r=None, xTraj=None, xDotTraj=None, uTraj=None,
        verbOverApprox=False, simVarF={}, simVarG={}, ignoreInconsistent=False,
        fixpointWidenCoeff=0.2, cTerm=0, zeroDiameter=1e-5, widenZeroInterval=1e-3,
        maxData=20, gurobiSolver=False, tolChange=tolChange, maxInvariantIter=10,
        verbSolver=False, verbCtrl=False, threshUpdateApprox=0.1, coeffLearning=0.1,
        probLearning=[], params=None):

        # Import gurobi for solving the idealistic and optimistic problem
        if gurobiSolver:
            import DaTaReachControl.optimisticControlGrb as optGrb
            self.optGrb = optGrb
            import DaTaReachControl.idealisticControlGrb as ideGrb
            self.ideGrb = ideGrb

        # Build the Overaproximation model
        self.overApprox = initOverApprox(nS, nC, listInfoF, listInfoG, xTraj,
                    xDotTraj, uTraj, verbose=verbOverApprox, fixpointWidenCoeff=fixpointWidenCoeff,
                    zeroDiameter=zeroDiameter, widenZeroInterval=widenZeroInterval,
                    maxData=maxData, tolChange=tolChange, maxInvariantIter=maxInvariantIter,
                    simVarF=simVarF, simVarG=simVarG, ignoreInconsistent=ignoreInconsistent)

        # Save the verbose parameters
        self.optVerb = verbSolver
        self.ctrlVerb = verbCtrl

        # Update the threshold for updating over-approx
        self.threshUpdateApprox = threshUpdateApprox

        # When learning the dynamics just apply(small pertubations) a ratio of the full range of u
        self.coeffLearning = coeffLearning

        # Label and prob of learning different components
        self.labLearning = np.array([-1]+[i for i in range(self.overApprox.nC)])
        self.probLearning = np.full(self.overApprox.nC+1, 1.0/(self.overApprox.nC+1)) \
                                if len(probLearning)==0 else probLearning

        # Delta time
        self.dt = dt

        # Range of the control u
        self.updateRangeControl(U_lb, U_ub)

        # Variable for updating the over-approximations f and G
        self.indexUpdate = -2
        self.updateMeas = False

        # Select which optimization tool to use
        self.initializeOptimizer(params)

        # Update cost function and create underlying optimizations problems
        self.updateCost(Q, S, R, q, r)
        self.cTerm = cTerm

        # Check if there's sufficient data to approximate f and G
        self.canDoApprox = canApproximate(self.overApprox)

        # Some temporarty variable in the problem
        self.nextStateOverApprox_lb = None
        self.nextStateOverApprox_ub = None
        self.currentX = None
        self.currentU = None

    # def initializeMixStrategy(self, mixStrategy):
    #     # if mixStrategy is None:
    #     #     self.mixStrategy = None
    #     #     return
    #     self.optGrb.initOptimisticProblemGrb(self.overApprox.nS, self.overApprox.nC,
    #             None, None, None, None, None, self.U_lb, self.U_ub)
    #     # self.ideGrb.initIdealisticProblemGrb(self.U_lb, self.U_ub)
    #     # self.mixStrategy = mixStrategy

    # def computeControlMixStrategy(self, A1_lb, A1_ub, A2_lb, A2_ub, b_lb, b_ub):
    #     Q, S, R, q, r = self.mixStrategyCost
    #     grbParams = (self.optVerb,)
    #     ideParams = (Q, S, R, q, r, *self.mixStrategy[1:-1])
    #     tol = self.mixStrategy[-1]
    #     uOpt, cost = self.optGrb.solveOptimisticProblemGrb(A1_lb, A1_ub, A2_lb, A2_ub, b_lb, b_ub,
    #         self.U_lb, self.U_ub, self.indexUpdate, *grbParams)
    #     # print('Cost 1', cost, uOpt)
    #     if cost+self.cTerm > tol:
    #         return uOpt, cost
    #     uOpt, cost = ideAPG.solveIdealisticProblemAPGDAR(A1_lb, A1_ub, A2_lb, A2_ub, b_lb, b_ub,
    #         self.U_lb, self.U_ub, self.indexUpdate, *ideParams)
    #     # print('Cost 2', cost, uOpt)
    #     return uOpt, cost

    def updateRangeControl(self, U_lb, U_ub):
        self.U_lb = U_lb
        self.U_ub = U_ub

    def initializeOptimizer(self, params):
        if params is None:
            params = (IDEALISTIC_APG, 0.7, 0.7, 0.7, 1e-8)
        self.method = params[0]
        if params[0] == OPTIMISTIC_GRB:
            self.optGrb.initOptimisticProblemGrb(self.overApprox.nS, self.overApprox.nC,
                None, None, None, None, None, self.U_lb, self.U_ub)
            self.optSolve = self.optGrb.solveOptimisticProblemGrb
            self.extra_params = (self.optVerb,)
        elif params[0] == IDEALISTIC_GRB:
            self.ideGrb.initIdealisticProblemGrb(self.U_lb, self.U_ub)
            self.extra_params = (*params[1:], self.optVerb)
            self.optSolve = self.ideGrb.solveIdealisticProblemGrb
        elif params[0] == IDEALISTIC_APG:
            self.extra_params = params[1:]
            self.optSolve = ideAPG.solveIdealisticProblemAPGDAR


    def updateCost(self, Q, S, R, q, r):
        """
        Initialize the optimizations problems that are going to be used for
        the synthesis of a controller
        """
        # Save the target function
        if Q is None:
            Q = np.zeros((self.overApprox.nS, self.overApprox.nS), dtype=np.float64)
        if R is None:
            R = np.zeros((self.overApprox.nC, self.overApprox.nC), dtype=np.float64)
        if S is None:
            S = np.zeros((self.overApprox.nS, self.overApprox.nC), dtype=np.float64)
        if q is None:
            q = np.zeros(self.overApprox.nS, dtype=np.float64)
        if r is None:
            r = np.zeros(self.overApprox.nC, dtype=np.float64)
        # if self.mixStrategy is not None:
        #     self.optGrb.updateCost(Q, S, R, q, r)
        #     self.mixStrategyCost = (Q, S, R, q, r)
        #     return
        if self.method == OPTIMISTIC_GRB:
            self.optGrb.updateCost(Q, S, R, q, r)
            self.solve_params = self.extra_params
        elif self.method == IDEALISTIC_GRB or self.method == IDEALISTIC_APG:
            self.solve_params = (Q, S, R, q, r, *self.extra_params)


    def noInitialData(self):
        """ When there's no initial data, perform control to generate
            useful point for over-approximation of f and G
            This routine assumes CONTROL VALLUES OF 0 can be applied
        """
        canApproxf, canApproxG = canApproximate(self.overApprox)
        if not canApproxf:
            self.currentU = np.zeros(self.overApprox.nC, dtype=realN)
            self.indexUpdate = -1
            return
        if not (canApproxf and canApproxG):
            self.indexUpdate = np.random.choice(self.labLearning,
                                    p=self.probLearning)
            self.currentU = np.zeros(self.overApprox.nC, dtype=realN)
            if self.indexUpdate != -1:
                while True:
                    self.currentU[self.indexUpdate] = self.coeffLearning * \
                            np.random.uniform(self.U_lb[self.indexUpdate],
                                              self.U_ub[self.indexUpdate])
                    if not (np.abs(self.currentU[self.indexUpdate]) < 1e-8):
                        break
        else:
            self.indexUpdate = -2
            self.canDoApprox = (True, True)

    def shouldUpdateTraj(self, nextState):
        """
        Check if the over-approxmations of f and G need to be updated. That's
        done by comparing the over-approximation at the next time-step given the
        synthesized control and the tre state and the next time step that will
        be received
        """
        if self.nextStateOverApprox_lb is None:
            return False
        d_lb, d_ub =norm_i(self.nextStateOverApprox_lb-nextState,
                        self.nextStateOverApprox_ub-nextState)
        # for i in range(nextState.shape[0]):
        #     print (i, nextState[i]-self.nextStateOverApprox_lb[i], self.nextStateOverApprox_ub[i]-nextState[i])
        #     assert nextState[i] - self.nextStateOverApprox_lb[i]>=-1e-2 and self.nextStateOverApprox_ub[i]- nextState[i]>=-1e-2
        return (d_ub-d_lb) > self.threshUpdateApprox

    def synthControlUpdate(self):
        """
        When updating over-approximations of f and G, impose constraints
        on the optimal control value such that it's either 0 or all
        components are equal to 0 except from one.
        """
        self.indexUpdate = np.random.choice(self.labLearning, p=self.probLearning)

    def __call__(self, currX, currXdot):
        """
        Function to call when trying to obtain the one-step optimal control
        to minimize the distance to the setpoint. This function is assumed
        to be called every dt
        """

        # print ('No init Data : ', self.noInitData)
        if not (self.canDoApprox[0] and self.canDoApprox[1]):
            if self.currentX is not None:
                fx, Gx = knownDynamics(self.currentX, self.currentX)
                for (k,p) in self.overApprox.Lf:
                    if ((k,p) in fx) == False:
                        fx[(k,p)] = (1.0,1.0)
                for (k,l,p) in self.overApprox.LG:
                    if ((k,l,p) in Gx) == False:
                        Gx[(k,l,p)] = (1.0,1.0)
                updateApprox(self.overApprox, self.currentX, currXdot, self.currentU, fx, Gx)
            self.currentX = currX
            self.noInitialData()
            return self.currentU

        # Check if we need to update f and G
        self.updateMeas = self.shouldUpdateTraj(currX)

        # Do some printing if required
        if self.ctrlVerb:
            nDataPoint = self.overApprox.nbData[0]
            print('Sampling time: ', self.dt)
            print('No. of data points: ', nDataPoint)
            print('Control bounds: ', self.U_lb, self.U_ub)


        # Set the index to update to no excitation based control
        self.indexUpdate = -2

        # Synthesis of the controller
        if self.updateMeas:
            print('Update Next --->')
            self.synthControlUpdate()

        b_lb, b_ub, A1_lb, A1_ub, A2_lb, A2_ub = \
            updateAndComputeAffineLinearization(self.overApprox, self.dt, currX,
                self.U_lb, self.U_ub, update= None if (self.currentX is None) \
                                                    else (self.currentX, currXdot, self.currentU)
            )

        # Synthesize control
        # uOpt, optCost = self.optSolve(A1_lb, A1_ub, A2_lb, A2_ub, b_lb, b_ub,
        #         learnConstr=self.learningConstr, verbose = self.optVerb,
        #         w1=self.weight1, w2=self.weight2, w3=self.weight3)
        uOpt, optCost = self.optSolve(A1_lb, A1_ub, A2_lb, A2_ub, b_lb, b_ub,
                self.U_lb, self.U_ub, self.indexUpdate, *self.solve_params)
        # if self.mixStrategy is None:
        #     uOpt, optCost = self.optSolve(A1_lb, A1_ub, A2_lb, A2_ub, b_lb, b_ub,
        #         self.U_lb, self.U_ub, self.indexUpdate, *self.solve_params)
        # else:
        #     uOpt, optCost = self.computeControlMixStrategy(A1_lb, A1_ub,
        #                                         A2_lb, A2_ub, b_lb, b_ub)
        optCost += self.cTerm
        if self.ctrlVerb:
            print('Approximate optimal cost : ', optCost)
            print('Update over-approximations: ', self.updateMeas, self.indexUpdate)
            print('State: ', currX)

        # Do some logging
        self.currentX = currX
        self.currentU = uOpt
        self.nextStateOverApprox_lb, self.nextStateOverApprox_ub = \
            nextStateOverApprox(b_lb, b_ub, A1_lb, A1_ub, A2_lb, A2_ub, uOpt)

        # print(self.nextStateOverApprox_lb)
        # print(self.nextStateOverApprox_ub)

        if self.ctrlVerb:
            print('Next state overapprox:')
            print(self.nextStateOverApprox_lb)
            print(self.nextStateOverApprox_ub)

        return self.currentU
