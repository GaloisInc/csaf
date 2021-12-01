"""
F16Lib Scenario Goals
"""
import csaf_f16.acas as f16a
from csaf.test.scenario import FixedSimGoal
import numpy as np


## GLOBAL PARAMETERS

# maximum position range to consider
dist_max = 1E4

# bounds of space to search
bounds = [
    (-dist_max, dist_max), # rel x
    (-dist_max, dist_max), # rel y
    (-np.pi, np.pi), # rel heading angle
    (-200.0, 200.0) # rel velocity
]


def intruder_airspeed(t):
    """airspeed for intruder aircraft (when specified to the scenario generator)"""
    if t <= 7.0:
        return 800.0
    else:
        return 1000.0


### SCENARIOS


AcasSimpleScenario = f16a.generate_acas_scenario(
    f16a.F16AcasIntruderBalloon,    # system to consider
    bounds,                         # configuration domain
    (-3000.0, 12000.0),             # balloon position
    ((0.0, 22000.0, 1000.0),),      # ownship waypoint
    800.0,                          # ownship airspeed
    tuple())                        # intruder waypoints


AcasAirspeedScenario = f16a.generate_acas_scenario(
    f16a.F16AcasIntruderBalloon,
    [
         (-dist_max, dist_max),
         (-dist_max, dist_max),
         (-np.pi, np.pi),
         (0.0, 0.0)
     ],
     (-3000.0, 12000.0),
     ((0.0, 22000.0, 1000.0),),
     800.0,
     tuple(),
     intruder_airspeed=intruder_airspeed
     )


AcasNoBalloonSimpleScenario = f16a.generate_acas_scenario(
    f16a.F16AcasIntruderBalloon,
    bounds,
    (-1E6, 1E6),
    ((0.0, 22000.0, 1000.0),),
    800.0,
    tuple())


AcasNoBalloonAirspeedScenario = f16a.generate_acas_scenario(
    f16a.F16AcasIntruderBalloon,
    [
        (-dist_max, dist_max),
        (-dist_max, dist_max),
        (-np.pi, np.pi),
        (0.0, 0.0)
    ],
    (-1E6, 1E6),
    ((0.0, 22000.0, 1000.0),),
    800.0,
    tuple(),
    intruder_airspeed=intruder_airspeed
    )


AcasSimpleShieldScenario = f16a.generate_acas_scenario(
    f16a.F16AcasShieldIntruderBalloon,
    bounds,
    (-3000.0, 12000.0),
    ((0.0, 22000.0, 1000.0),),
    800.0,
    tuple())


### GOALS


# goal: falsify the simple acas scenario
AcasSimpleFalsifyGoal = f16a.generate_acas_goal(AcasSimpleScenario)


# goal: falsify the shield acas scenario
AcasSimpleShieldFalsifyGoal = f16a.generate_acas_goal(AcasSimpleShieldScenario)


# goal: falsify the custom intruder airspeed acas scenario
AcasAirspeedFalsifyGoal = f16a.generate_acas_goal(AcasAirspeedScenario)


# goal: falsify the simple acas scenario with no balloon
AcasNoBalloonSimpleFalsifyGoal = f16a.generate_acas_goal(AcasNoBalloonSimpleScenario)


# goal: falsify the custom intruder airspeed acas scenario with no balloon
AcasNoBalloonAirspeedFalsifyGoal = f16a.generate_acas_goal(AcasNoBalloonAirspeedScenario)


class FixedSimAcasGoal(FixedSimGoal):
    """class with defaults for f16 acas fixed simulation goals"""
    terminating_conditions_all = f16a.collision_condition

    tspan = (0.0, 30.0)

    tspans = [(0.0, 30.0)]


class AcasSimpleCollideWithBalloonGoal(FixedSimAcasGoal):
    """cases where we know collisions occur"""
    scenario_type = AcasSimpleScenario

    should_fail = True

    fixed_configurations = [
        #[-5.38009090e+03,  4.66800612e+03,  1.10750142e+00, -1.49572862e+02],
        [ 8.46779582e+03,  4.52959138e+03, -1.43116897e+00, -2.80331323e+01], # intruder causes ownship to collide with balloon
    ]


class AcasSimpleCollideAvoidBalloonGoal(FixedSimAcasGoal):
    scenario_type = AcasSimpleScenario

    should_fail = False

    fixed_configurations = [
        [ 2.88149643e+03,  8.36264870e+03, -2.74357068e+00, -1.57079289e+02] # interesting near miss of both
    ]


class AcasShieldAvoidBalloonGoal(FixedSimAcasGoal):
    """cases where we know that a collision are successfully avoided"""
    scenario_type = AcasSimpleShieldScenario

    should_fail = False

    fixed_configurations = [
        #[-5.38009090e+03,  4.66800612e+03,  1.10750142e+00, -1.49572862e+02],
        [ 8.46779582e+03,  4.52959138e+03, -1.43116897e+00, -2.80331323e+01], # intruder causes ownship to collide with balloon
    ]


class AcasAirspeedAvoidNoBalloonGoal(FixedSimAcasGoal):
    """cases where we know that a collision are successfully avoided"""
    scenario_type = AcasNoBalloonAirspeedScenario

    should_fail = False

    fixed_configurations = [
        [ 7.61069193e+03,  7.63179351e+02, -7.56855508e-01,  0.00000000e+00],
        [7600, 1000, -np.pi/4+0.05, 0.0] # large path deviation
    ]


class AcasAirspeedCollideNoBalloonGoal(FixedSimAcasGoal):
    """cases where we know collisions occur"""
    scenario_type = AcasNoBalloonAirspeedScenario

    should_fail = True

    fixed_configurations = [
        [ 4.32487353e+03,  7.51223147e+03, -2.09140656e+00,  0.00000000e+00] # easy collision found
    ]
