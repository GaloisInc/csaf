import time

import numpy as np

from f16_uncertain_plant import *
from DaTaReachControl import DaTaControl
from DaTaReachControl import OPTIMISTIC_GRB, IDEALISTIC_GRB, IDEALISTIC_APG
from f16plant import subf16df
from helpers import llc_helper as lh

# seed = np.random.randint(0,2000)
# np.random.seed(seed)
# print(seed)
# np.random.seed(430) # 607

def buildObjective(setpoints, target_index):
    """ Get the quadratic matrices/vector for the cost function based on
        the setpoints
    """
    Qtarget = np.zeros((n_f16, n_f16), dtype=np.float64)
    Rtarget = np.zeros((m_f16, m_f16), dtype=np.float64)
    Starget = np.zeros((n_f16, m_f16), dtype=np.float64)
    qtarget = np.zeros(n_f16, dtype=np.float64)
    rtarget = np.zeros(m_f16, dtype=np.float64)
    for i in range(target_index.shape[0]):
        Qtarget[target_index[i],target_index[i]] = 1.0
        qtarget[target_index[i]] = -2.0*setpoints[target_index[i]]
    return Qtarget, Rtarget, Starget, qtarget, rtarget

class DaTaControlF16(lh.FeedbackController):
    def __init__(self, ctrlLimits, model):
        super().__init__(ctrlLimits, model, None, None, None)
        # Save the F16 plant to be used for computing the state derivatives
        # Save the equilibrium points
        self.xequil = np.array([502.0, 0.03887505597600522, 0.0, 0.0, 0.03887505597600522, 0.0, 0.0, 0.0, \
                           0.0, 0.0, 0.0, 1000.0, 9.05666543872074])
        self.uequil = np.array([0.13946204864060271, -0.7495784725828754, 0.0, 0.0])

        self.step_size = 0.01
        self.step_size = 0.01#self.sampling_period
        self.model = model

        # Parameters for DaTaControl
        maxData = 10
        threshUpdateLearn = 0.5
        params_solver = (IDEALISTIC_APG, 0.7, 0.7, 0.5, 1e-14)
        gurobiSolver=False
        # params_solver = (IDEALISTIC_GRB, 0.2, 0.7, 0.5)
        # params_solver = (OPTIMISTIC_GRB,)
        # gurobiSolver = True

        self.target_position = np.array([502.0, 0.03887505597600522, 0.0, 0.0, 0.03887505597600522, 0.0, 0.0, \
                           0.0, 1000.0, 9.05666543872074, 0, 0])
        # self.target_index = np.array([1, 6, 10, 11])
        # self.state_names = ['alpha', 'q', 'int_Nz','int_Ps']

        self.target_index = np.array([10, 11])
        self.state_names = ['int_Nz','int_Ps']

        # Build the initial quadratic coefficient
        Q, R, S, q, r = buildObjective(self.target_position, self.target_index)

        self.uRange_lb = np.array([0, ctrlLimits.ElevatorMinDeg, ctrlLimits.AileronMinDeg, ctrlLimits.RudderMinDeg], dtype=np.float64)
        self.uRange_ub = np.array([0.5, ctrlLimits.ElevatorMaxDeg, ctrlLimits.AileronMaxDeg, ctrlLimits.RudderMaxDeg], dtype=np.float64)

        # Index of interests
        self.indexInterest = np.array([0, 1, 2, 3, 4, 6, 7, 8, 11, 12])

        # Build the controller --> No initial trajectory
        self.synth_control = DaTaControl(self.step_size, n_f16, m_f16, infoF, infoG,
            self.uRange_lb, self.uRange_ub, Q=Q, S=S, R=R, q=q, r=r,
            xTraj=None, xDotTraj=None, uTraj=None, verbOverApprox=False,
            simVarF=simVarF, simVarG=simVarG, knownPart=known_f16,
            fixpointWidenCoeff=0.01, maxData=maxData, gurobiSolver=gurobiSolver,
            tolChange=1e-3, maxInvariantIter=2, verbCtrl=False,
            threshUpdateApprox=threshUpdateLearn, coeffLearning=0.1,
            probLearning=[0.75, 0.1, 0.05, 0.05, 0.05], params=None,)

        # Save the last control values applied
        # self.lastControl = np.copy(self.uequil)
        self.lastDerivative = np.zeros(self.xequil.shape[0], dtype=np.float64)
        self.lastDerivativeIntNz = 0
        self.lastDerivativeIntPs = 0
        self.lastTime = -1
        self.lastControl = self.uequil
        self.intNz = 0.0
        self.intPs = 0.0

    def output(self, t, cstate, u):
        if t == self.lastTime:
            return self.lastControl

        assert len(u) == 21

        #todo: hard coded indices!
        x_f16, y, u_ref = u[:13], u[13:17], u[17:]
        Nz, Ny, _, _ = y

        # Get the current integrator state --> Prob need ode solver to get a precise value of int_Nz --> augment Nz in state space?
        curr_state = np.concatenate((x_f16[self.indexInterest],\
                        np.array([self.intNz, self.intPs])))
        last_der = np.concatenate((self.lastDerivative[self.indexInterest],\
                                        np.array([self.lastDerivativeIntNz, self.lastDerivativeIntPs])))

        # The setpoints for q and \alpha are in target_position --> update target Nz
        self.target_position[-2] += u_ref[0] * self.step_size
        self.target_position[-1] += u_ref[1] * self.step_size
        Q, R, S, q, r = buildObjective(self.target_position, self.target_index)

        # Update the objective inside DaTaControl
        self.synth_control.updateCost(Q, S, R, q, r)

        # Store the synthesized control value
        u_deg = np.zeros((4,), dtype=np.float64)  # throt, ele, ail, rud
        # set throttle as directed from output of getouterloopctrl(...)
        u_deg[0] = u_ref[3]
        # add in equilibrium control
        u_deg[0:4] += self.uequil
        u_deg = lh.clip_u(self.model, u_deg)
        u_deg[0] = min(max(u_deg[0], self.uRange_lb[0]), self.uRange_ub[0])

        # if self.synth_control.canDoApprox[0] and self.synth_control.canDoApprox[0]:
        #     uRange_lb = self.uRange_lb + 0.0
        #     uRange_ub = self.uRange_ub + 0.0
        #     uRange_lb[2:] = u_deg[2:]
        #     uRange_ub[2:] = u_deg[2:]
        #     uRange_lb[0] = u_deg[0]
        #     uRange_ub[0] = u_deg[0]
        #     self.synth_control.updateRangeControl(uRange_lb, uRange_ub)

        # Sanity check for DaTaCOntrol to gather some data initially
        if self.synth_control.canDoApprox[0] and self.synth_control.canDoApprox[0]:
            uRange_lb = self.uRange_lb + 0.0
            uRange_ub = self.uRange_ub + 0.0
            # uRange_lb[3:] = u_deg[3:]
            # uRange_ub[3:] = u_deg[3:]
            uRange_lb[0] = u_deg[0]
            uRange_ub[0] = u_deg[0]
            self.synth_control.updateRangeControl(uRange_lb, uRange_ub)

        print('\n[Nz] : Time = ' + str(t) + '.')
        print('Control state: ', {'{}'.format(self.state_names[i]) : curr_state[self.target_index[i]] \
                                     for i in range(self.target_index.shape[0])})
        print('Target state: ', {'{}'.format(self.state_names[i]) : self.target_position[self.target_index[i]] \
                                     for i in range(self.target_index.shape[0])})


        query_timer_start = time.time()
        # Synthesize the near-optimal control value
        uOpt = self.synth_control(curr_state, last_der)
        query_timer_end = time.time()
        query_time = query_timer_end - query_timer_start
        print('Best action = {:s} | Solver Time = '
            '{:1.4f} s '.format(np.array_str(uOpt, precision=2), query_time))

        u_deg = uOpt
        # # print ('Control: ', u_deg)
        # print('---------- CHECK ----------')
        # fx, Gx = known_f16(curr_state, curr_state)
        # print(fx)
        # print(Gx)
        # print('--')
        # xdot1, _ = self.f16Dyn.subf16df_m(t, x_f16, np.zeros(4), np.zeros(9), 0)
        # xdot2, _ = self.f16Dyn.subf16df_m(t, x_f16, np.zeros(4), None, 0)
        # xdot3, _ = self.f16Dyn.subf16df_m(t, x_f16, np.zeros(4), np.zeros(9), None)
        # xdot4, _ = self.f16Dyn.subf16df_m(t, x_f16, np.array([0,1,0,0]), np.zeros(9), 0)
        # print('CST: ', xdot1[1])
        # print('term2: ', xdot2[1]-xdot1[1])
        # print('thrust: ', xdot3[1]-xdot1[1])
        # print('ail: ', xdot4[1]-xdot1[1])
        # Compute the derivative of alpha, q and int_Nz based on the current control value

        #XXX: Why is adjust_cy None!!?
        xder, output = subf16df(self.model, t, x_f16, u_deg, None)

        self.lastDerivative = xder
        self.lastDerivativeIntNz = output[0]
        self.lastDerivativeIntPs = x_f16[6]*np.cos(x_f16[1]) + x_f16[8]*np.sin(x_f16[1])
        self.intNz += self.lastDerivativeIntNz * self.step_size
        self.intPs += self.lastDerivativeIntPs * self.step_size
        self.lastTime = t
        self.lastControl = u_deg

        return u_deg


def model_init(model):
    """load trained model"""
    model.parameters['llc'] = DaTaControlF16(lh.CtrlLimits(), model)


def model_output(model, t, state_controller, input_all):
    assert len(input_all) == 21
    #TODO: hard coded indices!
    """ get the reference commands for the control surfaces """
    return model.parameters['llc'].output(t, np.array(state_controller), np.array(input_all))


def model_state_update(model, t, state_controller, input_all):
    """ get the derivatives of the integrators in the low-level controller """
    #return llc.step(t, 1/model.sampling_frequency, state_controller, input_all)
    return model.parameters['llc']._der(t, np.array(state_controller), np.array(input_all))
