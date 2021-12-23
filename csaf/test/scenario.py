from csaf.core.base import CsafBase
from csaf.core.system import System
from csaf.core.trace import TimeTrace
import typing


class FiniteSet(tuple):
    pass


class IntervalSet(tuple):
    pass


class Scenario(CsafBase):
    """
    CSAF Scenario

    A csaf scenario captures the variability of a system, providing a new coordinates
    (called configuration space) to generate systems from. The configuration space may
    be desirable for reducing the number of variables considered in a system or capturing
    symmetries present in a system.
    """
    configuration_space: typing.Type[typing.Tuple]

    system_type: typing.Type[System]

    bounds: typing.Optional[typing.Sequence[typing.Tuple[float, float]]] = None

    def generate_system(self, conf: typing.Sequence) -> System:
        return self.system_type()


class Goal(CsafBase):
    """
    CSAF Goal

    A CSAF goal captures a desirable property for a scenario and an
    implementation to test it (test_goal)
    """
    scenario_type: typing.Type[Scenario]

    should_fail = False

    def test_goal(self) -> bool:
        return True

    def test(self, **kwargs) -> bool:
        orig_kwargs = {}
        for k, v in kwargs.items():
            if hasattr(self, k):
                orig_kwargs[k] = getattr(self, k)
                setattr(self, k, v)
        r = self.test_goal()
        for k, v in orig_kwargs.items():
            setattr(self, k, v)
        return r if not self.should_fail else not r

    def validate(self) -> None:
        assert hasattr(self, 'scenario_type')
        assert issubclass(self.scenario_type,
                          Scenario), f"scenario type {self.scenario_type} must be a subclass of a CSAF scenario"


class SimGoal(Goal):
    """
    Goals involving simulations
    """
    terminating_conditions: typing.Optional[typing.Callable[[], bool]] = None

    terminating_conditions_all: typing.Optional[typing.Callable[[TimeTrace], bool]] = None

    sim_kwargs: typing.Dict[str, typing.Any] = {}

    tspan: typing.Tuple[float, float] = (0.0, 1.0)

    @classmethod
    def run_sim(cls, conf: typing.Sequence, timespan=None):
        if timespan is None:
            timespan = cls.tspan
        sys = cls.scenario_type().generate_system(conf)
        return sys.simulate_tspan(timespan,
                                  terminating_conditions=cls.terminating_conditions,
                                  terminating_conditions_all=cls.terminating_conditions_all,
                                  return_passed=True,
                                  **cls.sim_kwargs)


class FixedSimGoal(SimGoal):
    """
    Test a collection of fixed configurations specified in fixed_configurations
    """
    fixed_configurations: typing.Sequence[typing.Sequence]

    tspans: typing.Sequence[typing.Tuple[float, float]]

    def test_goal(self) -> bool:
        """
        Run fixed simulations. If any of them fails, the goal is considered failed.
        """
        for ts, c in zip(self.tspans, self.fixed_configurations):
            t, p = self.run_sim(c, timespan=ts)
            if not p:
                return False
        return True


class BOptFalsifyGoal(SimGoal):
    import GPy  # type: ignore
    import numpy as np

    max_iter = 500
    max_time = 120.0
    tolerance = 5.0
    kernel = GPy.kern.StdPeriodic(4,  # dimension
                                 ARD1=True, ARD2=True,
                                 variance=1E-2,
                                 period=[1E10, 1E10, 2 * np.pi, 1E8],
                                 lengthscale=[200.0, 200.0, 0.06, 4.0])

    constraints: typing.Sequence[typing.Dict] = tuple()

    @staticmethod
    def property(ctraces: TimeTrace) -> bool:
        pass

    def objective_function(self, conf: typing.Sequence):
        pass

    def gen_optimizer(self):
        import numpy as np
        import GPy  # type: ignore
        import GPyOpt  # type: ignore

        def to_gpy_space(bounds):
            print([isinstance(b, FiniteSet) for b in bounds])
            return [{'name': f'x{idx}', 'type':
                'discrete' if isinstance(b, FiniteSet) else 'continuous',
                     'domain': b} for idx, b in enumerate(bounds)]

        # get feasible region from initial states and constraints
        feasible_region = GPyOpt.Design_space(space=to_gpy_space(self.scenario_type.bounds),
                                              constraints=self.constraints)

        # generate initial designs
        initial_design = GPyOpt.experiment_design.initial_design('random', feasible_region, 1)
        # incorporate "good guesses"
        # initial_design = np.vstack((
        #    # "good guess" 1
        #    #[0.0, 8000, np.pi, 0.0],
        #    # "good guess" 2
        #    [7000, 1000.0, -np.pi/2, -100.0],
        #    initial_design))

        # get GPyOpt objective from function
        objective = GPyOpt.core.task.SingleObjective(self.objective_function)

        # custom kernel!
        # we know that the relative heading angle is periodic, so we can set and fix it
        # we know that the lengthscales and periods will be different along each natural axis, so we turn on ARD
        # we know prior variance is low, so we can set low as well
        k = self.kernel
        k.period.fix()
        k.lengthscale.fix()

        # get GPModel from the covariance (kernel)
        model = GPyOpt.models.GPModel(exact_feval=True, optimize_restarts=0, verbose=False, kernel=k)

        # get the GPOpt acquisition optimizer
        aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(feasible_region)

        # get the type of acquisition
        acquisition = GPyOpt.acquisitions.AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

        # get the collection method
        evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

        # get a cost model from the cost func
        cost = None  # GPyOpt.core.task.cost.CostModel(self.cost_func)

        # build the modular Bayesian optimization model
        return GPyOpt.methods.ModularBayesianOptimization(model,
                                                          feasible_region,
                                                          objective,
                                                          acquisition,
                                                          evaluator,
                                                          initial_design,
                                                          cost=cost)

    def __init__(self):
        self.optimizer = self.gen_optimizer()

    def test_goal(self) -> bool:
        self.optimizer.run_optimization(max_iter=self.max_iter,
                                        max_time=self.max_time,
                                        eps=self.tolerance,
                                        verbosity=False)
        t, p = self.run_sim(self.optimizer.x_opt)
        return self.property(t)
