"""
New CSAF Goals

TODO: consolidate with original goals
"""
import typing
import numpy as np
from csaf_f16.components import f16_xequil
from csaf_f16.acas import generate_acas_goal
from csaf.test.scenario import FiniteSet, IntervalSet
import csaf
from csaf import System, Scenario
import csaf_f16.acas as f16a



class AcasRejoinCoord(typing.NamedTuple):
    altitude: float
    speed: float
    rel_dist: float
    rel_speed: float
    rel_heading: float


class AcasAirportScenario(Scenario):
    configuration_space = AcasRejoinCoord

    system_type = f16a.F16AcasIntruderBalloon

    bounds = [FiniteSet((15E3, 35E3)),
              FiniteSet((800.0, 1000.0)),
              IntervalSet((-1E4, 1E4)),
              IntervalSet((-100.0, 100.0)), # this is low as I keep getting simulation errors
              IntervalSet((-np.pi, np.pi))]

    def __init__(self):
        # this is necessary for the viewer
        # TODO: let's clean up this ugliness
        self.intruder_waypoints = ((1.0, 1.0, 0.0),)
        self.own_waypoints = tuple()
        self.balloon_pos = (1E8, 1E8, 0.0)

    def generate_system(self, conf: typing.Sequence) -> System:
        # create an initial waypoint for the intruder
        iwaypoints = [(*(conf[2]/2, 0.0), conf[0]), ] + list([(*w[:2], conf[0]) for w in self.intruder_waypoints])
        owaypoints = [(1.0, 1.0, conf[0])]

        # copy the states over so we can modify them
        sys = self.system_type()
        ownship_states = f16_xequil.copy()
        intruder_states = f16_xequil.copy()
        balloon_states = f16_xequil.copy()

        # set the positions
        ownship_states[10], ownship_states[9], ownship_states[11] = (-conf[2]/2, 0.0, conf[0])
        balloon_states[10], balloon_states[9], balloon_states[11] = (1E8, 1E8, conf[0])
        intruder_states[10], intruder_states[9], intruder_states[11] = (conf[2]/2, 0.0, conf[0])

        # set the airspeed
        ownship_states[0] = conf[1]
        intruder_states[0] = ownship_states[0] + conf[3]

        # reset the balloon airspeed
        balloon_states[0] = 0.0
        balloon_states[0:9] = [0.0, ] * 9

        # set the relative heading angle
        intruder_states[5] = conf[4]

        # set the states
        sys.set_state("balloon", balloon_states)
        sys.set_state("plant", ownship_states)
        sys.set_state("intruder_plant", intruder_states)

        # set the waypoints
        sys.set_component_param("intruder_autopilot", "waypoints", iwaypoints)
        sys.set_component_param("waypoint", "waypoints", owaypoints)
        sys.set_component_param("intruder_autopilot", "airspeed", lambda t: ownship_states[0] + conf[3])

        return sys


class AcasHeadOnScenario(Scenario):
    configuration_space = AcasRejoinCoord

    system_type = f16a.F16AcasIntruderBalloon


    bounds = [FiniteSet((15E3, 35E3)),
              FiniteSet((600.0, 1000.0)),
              IntervalSet((-1E4, 1E4)),
              IntervalSet((-400.0, 400.0)),
              IntervalSet((-np.pi, np.pi))]


    def __init__(self):
        # this is necessary for the viewer
        # TODO: let's clean up this ugliness
        self.intruder_waypoints = ((1.0, 1.0, 0.0),)
        self.own_waypoints = tuple()
        self.balloon_pos = (1E8, 1E8, 0.0)

    def generate_system(self, conf: typing.Sequence) -> System:
        # create an initial waypoint for the intruder
        iwaypoints = [(*(0.0, conf[2]), conf[0]),]
        owaypoints = [(0.0, 1E5, conf[0])]

        # copy the states over so we can modify them
        sys = self.system_type()
        ownship_states = f16_xequil.copy()
        intruder_states = f16_xequil.copy()
        balloon_states = f16_xequil.copy()

        # set the positions
        ownship_states[10], ownship_states[9], ownship_states[11] = (0.0, 0.0, conf[0])
        balloon_states[10], balloon_states[9], balloon_states[11] = (1E8, 1E8, conf[0])
        intruder_states[10], intruder_states[9], intruder_states[11] = (0.0, conf[2], conf[0])

        # set the airspeed
        ownship_states[0] = conf[1]
        intruder_states[0] = ownship_states[0] + conf[3]

        # reset the balloon airspeed
        balloon_states[0] = 0.0
        balloon_states[0:9] = [0.0, ] * 9

        # set the relative heading angle
        intruder_states[5] = conf[4]

        # set the states
        sys.set_state("balloon", balloon_states)
        sys.set_state("plant", ownship_states)
        sys.set_state("intruder_plant", intruder_states)

        # set the waypoints
        sys.set_component_param("intruder_autopilot", "waypoints", iwaypoints)
        sys.set_component_param("waypoint", "waypoints", owaypoints)
        sys.set_component_param("intruder_autopilot", "airspeed", lambda t: ownship_states[0] + conf[3])

        return sys


class AcasRejoinScenario(Scenario):
    configuration_space = None

    system_type = f16a.F16AcasIntruderBalloon

    bounds = [FiniteSet((15E3, 35E3)),
              FiniteSet((600.0, 1000.0)),
              IntervalSet((6000.0, 1E4)),
              IntervalSet((-400.0, 400.0)),
              IntervalSet((-np.pi, np.pi))]

    def __init__(self):
        # this is necessary for the viewer
        # TODO: let's clean up this ugliness
        self.intruder_waypoints = ((1.0, 1.0, 0.0),)
        self.own_waypoints = tuple()
        self.balloon_pos = (1E8, 1E8, 0.0)

    def generate_system(self, conf: typing.Sequence) -> System:
        # create an initial waypoint for the intruder
        iwaypoints = [(*(conf[2], 0.0), conf[0]), ] + list([(*w[:2], conf[0]) for w in self.intruder_waypoints])
        owaypoints = [(0.0, 1E5, conf[0])]

        # copy the states over so we can modify them
        sys = self.system_type()
        ownship_states = f16_xequil.copy()
        intruder_states = f16_xequil.copy()
        balloon_states = f16_xequil.copy()

        # set the positions
        ownship_states[10], ownship_states[9], ownship_states[11] = (0.0, 0.0, conf[0])
        balloon_states[10], balloon_states[9], balloon_states[11] = (1E8, 1E8, conf[0])
        intruder_states[10], intruder_states[9], intruder_states[11] = (conf[2], 0.0, conf[0])

        # set the airspeed
        ownship_states[0] = conf[1]
        intruder_states[0] = ownship_states[0] + conf[3]

        # reset the balloon airspeed
        balloon_states[0] = 0.0
        balloon_states[0:9] = [0.0, ] * 9

        # set the relative heading angle
        intruder_states[5] = conf[4]

        # set the states
        sys.set_state("balloon", balloon_states)
        sys.set_state("plant", ownship_states)
        sys.set_state("intruder_plant", intruder_states)

        # set the waypoints
        sys.set_component_param("intruder_autopilot", "waypoints", iwaypoints)
        sys.set_component_param("waypoint", "waypoints", owaypoints)
        sys.set_component_param("intruder_autopilot", "airspeed", lambda t: ownship_states[0] + conf[3])

        return sys


import GPy
# this might be a bad choice as the kernel is mutable -- we should pass a way to create the kernel
# so that each cobnsumer of it gets a fresh copy
kernel = [GPy.kern.StdPeriodic(5,  # dimension
                              ARD1=True, ARD2=True,
                              variance=1E-2,
                              period=[1E10, 1E10, 1E8, 1E8, 2 * np.pi],
                              lengthscale=[200.0, 20.0, 200.0, 20.0, 0.05]) for _ in range(3)]
constraints = [
            # keep intruder initial position at least 7000 ft away
            {'name': 'min_distance_constr', 'constraint': '-(np.abs(x[:, 2]) - 7000)'},
            # keep the simulation in a stable plave (min airspeed of intruder)
            {'name': 'min_speed_constr', 'constraint': '-(x[:, 1] + x[:, 3] - 600)'},
            {'name': 'max_speed_constr', 'constraint': '(x[:, 1] + x[:, 3] - 1100)'}
        ]

AcasAirportGoal = generate_acas_goal(AcasAirportScenario, gpkernel=kernel[0], gpconstraints=constraints)

AcasHeadOnGoal = generate_acas_goal(AcasHeadOnScenario, gpkernel=kernel[1], gpconstraints=constraints)

AcasRejoin = generate_acas_goal(AcasRejoinScenario, gpkernel=kernel[2], gpconstraints=constraints)
