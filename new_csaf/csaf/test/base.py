from os import stat
import csaf
from f16lib.acas import AcasScenario, AcasShieldScenario, AcasScenarioViewer

from collections.abc import Callable

from abc import ABC, abstractmethod

import numpy as np
import typing
import GPyOpt
import GPy

class Optimizer(ABC):
    """
    Generic optimizer
    """
    @abstractmethod
    def prepare(self,
                configuration_space: dict,
                obj_func: Callable[[[float]], float]):
        pass

    @abstractmethod
    def run_optimization(self,
                         max_iter: int,
                         max_time: float,
                         tolerance: float,
                         verbose: bool = False):
        pass

class Scenario(ABC):
    """
    - system_under_test is the system being tested
    - configuration_space is a dictionary specifying
    the configuration space of the problem
    """
    configuration_space: dict = {}

    @abstractmethod
    def get_system_under_test(self, states: typing.List[float]) -> csaf.System:
        pass

    # bounds, constraints
    def get_configuration_space(self) -> dict:
        return self.configuration_space

class Goal(ABC):
    """
    - objective_func takes configuration space states (a list),
    returns the objective value (float)
    - properties_func takes the TimeTrace resulting from the simulation
    and returns True/False depending on whether the property (such as
    a ground collision or mid-air collision) was reached
    """
    optimizer: Optimizer # requires obj_func and  scenario.feasible_region
    scenario: Scenario # gives feasible region, obj_func requires scenario as well (get_system)

    def __init__(self, optimizer: Optimizer, scenario: Scenario) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.scenario = scenario
        self.optimizer.prepare(self.scenario.configuration_space, self.objective_function)

    # -> needed by optimizer, needs Scenario.system_under_test and Scenario.
    @abstractmethod
    def objective_function(self, states: typing.List[float]) -> float:
        pass

    #  -> for filtering results
    @abstractmethod
    def properties_func(ctraces: csaf.TimeTrace) -> bool:
        pass



class GPOptimizer(Optimizer):
    """
    Gausian Optimization using GPyOpt
    extends generic Optimizer
    """
    objective_function: Callable[[[float]], float]

    @staticmethod
    def cost_func(x):
        """weakly penalize searching too far away (this is optional)"""
        cost_f  = np.atleast_2d(.001*x[:,0]**2 +.001*x[:,1]**2).T
        cost_df = np.array([0.002*x[:,0],0.002*x[:,1]]).T
        return cost_f, cost_df

    def objective_function_opt(self, x):
        """GPyOpt Objective"""
        return np.array([self.objective_function(xi) for xi in x])

    def prepare(self,
                configuration_space: dict,
                obj_func: Callable[[[float]], float]):
        self.objective_function = obj_func

        # get feasible region from initial states and constraints
        feasible_region = GPyOpt.Design_space(space = configuration_space['bounds'], constraints = configuration_space['constraints'])

        # generate initial designs
        initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 10)
        # incorporate "good guesses"
        initial_design = np.vstack((
            # "good guess" 1
            #[0.0, 8000, np.pi, 0.0],
            # "good guess" 2
            [7000, 1000.0, -np.pi/2, -100.0],
            initial_design))

        # get GPyOpt objective from function
        objective = GPyOpt.core.task.SingleObjective(self.objective_function_opt)

        # custom kernel!
        # we know that the relative heading angle is periodic, so we can set and fix it
        # we know that the lengthscales and periods will be different along each natural axis, so we turn on ARD
        # we know prior variance is low, so we can set low as well
        k = GPy.kern.StdPeriodic(4,  # dimension
                                 ARD1=True, ARD2=True,
                                 variance=1E-2,
                                 period=[1E10, 1E10, 2*np.pi, 1E8],
                                 lengthscale=[200.0, 200.0, 0.06, 4.0])
        k.period.fix()
        k.lengthscale.fix()
        #k.variance.fix()

        # get GPModel from the covariance (kernel)
        model = GPyOpt.models.GPModel(exact_feval=True,optimize_restarts=0,verbose=False,kernel=k)

        # get the GPOpt acquisition optimizer
        aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

        # get the type of acquisition
        acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

        # get the collection method
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

        # get a cost model from the cost func
        cost = GPyOpt.core.task.cost.CostModel(GPOptimizer.cost_func)

        # build the modular Bayesian optimization model
        self.bo = GPyOpt.methods.ModularBayesianOptimization(model,
                                                feasible_region,
                                                objective,
                                                acquisition,
                                                evaluator,
                                                initial_design,
                                                cost=cost)

    def run_optimization(self,
                         max_iter: int,
                         max_time: float,
                         tolerance: float,
                         verbose: bool = False):
        # Run the optimization
        self.bo.run_optimization(max_iter = max_iter,
                            max_time = max_time,
                            eps = tolerance,
                            verbosity=False)



class ScenarioAcas(Scenario):
    """
    CP3.2 scenario
    One intruder, one static balloon, one F16 (ownship)
    """
    def __init__(self, bounds, constraints, intruder_airspeed) -> None:
        super().__init__()
        self.configuration_space['bounds'] = bounds
        self.configuration_space['constraints'] = constraints
        self.intruder_airspeed = intruder_airspeed

    def get_system_under_test(self, states) -> csaf.System:
        # scenario
        scen = AcasScenario(
            [-3000.0, 12000], # balloon position
            750.0, # ownship airspeed
            ((0.0, 21000.0, 1000.0),), # own waypoints
            ((*states[:2], 1000.0),), # intruder waypoints -- none for now
            intruder_velocity = self.intruder_airspeed
        )
        # scenario, system
        return scen, scen.create_system(
            [*states[:2], # relative position
            states[2], # relative heading
            states[3]]) # relative airspeed


class FalsifyAcas(Goal):
    """
    Falsify CP 3.2 problem
    """
    def objective_function(self, states: typing.List[float]) -> float:
        """obj: configuration space -> real number"""
        # run simulation
        _, sys = self.scenario.get_system_under_test(states)
        trajs, _p = sys.simulate_tspan((0.0, 30.0), return_passed=True)

        # get distances between ownship and intruder
        intruder_pos = np.array(trajs['intruder_plant'].states)[:, 9:11]
        ownship_pos = np.array(trajs['plant'].states)[:, 9:11]
        rel_pos = intruder_pos - ownship_pos

        # get distances between ownship and balloon
        dists = np.linalg.norm(rel_pos, axis=1)
        ballon_dists = ownship_pos - np.tile(np.array(trajs['balloon'].states)[-1, 9:11][:], (len(ownship_pos), 1))
        bdists = np.linalg.norm(ballon_dists, axis=1)

        # get objective (min distance to obstacles)
        print("OBJECTIVE (min distance): ", min(np.hstack((dists, bdists))), ",", np.sqrt(min(dists) * min(bdists)))
        # geometric mean of min dists
        return np.sqrt(min(dists) * min(bdists))
        # minimum distance
        #return min(np.hstack((dists, bdists)))


    def properties_func(self, ctraces: csaf.TimeTrace) -> bool:
        """
        air collision condition
        """
        # get the aircraft states
        sa, sb, sc = ctraces['plant']['states'], ctraces['intruder_plant']['states'], ctraces['balloon']['states']
        if sa and sb and sc:
            # look at distance between last state
            dab =  (np.linalg.norm(np.array(sa[-1][9:11]) - np.array(sb[-1][9:11])))
            dac = (np.linalg.norm(np.array(sa[-1][9:11]) - np.array(sc[-1][9:11])))
            return dab < 500.0 or dac < 500.0

    def falsify(self,max_time=60.0*2,max_iter=100,tolerance=10):
        self.optimizer.run_optimization(max_iter,max_time, tolerance)