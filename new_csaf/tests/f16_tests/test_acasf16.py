import pytest
import csaf
import f16lib.systems as f16s
import f16lib.components as f16c
import f16lib.acas as f16acas
import numpy as np


@pytest.fixture
def acas_shield_balloon_f16():
    """create F16Simple for tests"""
    sys = f16s.F16AcasShieldIntruderBalloon()
    yield sys


@pytest.fixture
def acas_shield_f16():
    sys = f16s.F16AcasShield()
    yield sys


@pytest.fixture
def acas_balloon_f16():
    sys = f16s.F16AcasIntruderBalloon()
    yield sys


@pytest.fixture
def acas_scenario():
    scen = f16acas.AcasScenario(
        [-1000, 5000],  # balloon position
        500.0,  # ownship airspeed
        ((0.0, 20000, 1000.0), (5000.0, 30000, 1000.0), (10000, 20000, 1000.0)),  # own waypoints
        ((-0.0, 4000.0, 1000.0), (-4000.0, 2500.0, 1000.0), (-8000, 4000, 1000.0))  # intruder waypoints
    )
    sys = scen.create_system([6000, 8000,  # relative position
                              -3.14 / 2,  # relative heading
                              -200.0])  # relative airspeed
    yield sys


def air_collision_condition(ctraces):
    """air collision termination condition"""
    # get the aircraft states
    sa, sb, sc = ctraces['plant']['states'], ctraces['intruder_plant']['states'], ctraces['balloon']['states']
    if sa and sb and sc:
        # look at distance between last state
        dab = (np.linalg.norm(np.array(sa[-1][9:11]) - np.array(sb[-1][9:11])))
        dac = (np.linalg.norm(np.array(sa[-1][9:11]) - np.array(sc[-1][9:11])))
        return dab < 500.0 or dac < 500.0


def test_acas_shield_balloon(acas_shield_balloon_f16: csaf.System):
    acas_shield_balloon_f16.check()


def test_acas_balloon(acas_balloon_f16: csaf.System):
    acas_balloon_f16.check()


def test_acas_shield(acas_shield_f16: csaf.System):
    acas_shield_f16.check()


def test_acas_scenario(acas_shield_balloon_f16: csaf.System):
    trajs, p = acas_shield_balloon_f16.simulate_tspan((0.0, 20.0),
                                                      terminating_conditions_all=air_collision_condition,
                                                      return_passed=True)
    assert not p, f"{acas_shield_balloon_f16.__class__.__name__} did not collide"


def test_example_collision():
    def air_collision_condition(ctraces):
        """ground collision premature termination condition
        """
        # get the aircraft states
        sa, sb, sc = ctraces['plant']['states'], ctraces['intruder_plant']['states'], ctraces['balloon']['states']
        if sa and sb and sc:
            # look at distance between last state
            dab =  (np.linalg.norm(np.array(sa[-1][9:11]) - np.array(sb[-1][9:11])))
            dac = (np.linalg.norm(np.array(sa[-1][9:11]) - np.array(sc[-1][9:11])))
            return dab < 500.0 or dac < 500.0

    y_offset = 1700
    int_way = (-7000, 0.0+y_offset, 1000.0)

    # construct a scenario
    scen = f16acas.AcasShieldScenario(
        [-2000.0, 5000+y_offset], # balloon position
        500.0, # ownship airspeed
        ((0.0, 15000.0+y_offset, 1000.0),(0.0, 20000.0+y_offset, 1000.0)),
        (int_way,)
    )

    m = (8000+y_offset - int_way[1]) / (6000 - int_way[0])
    theta = np.arctan2((6000 - int_way[0]), (8000+y_offset - int_way[1]))
    x = 9000

    sys = scen.create_system([x, (x-int_way[0])*m+int_way[1], # relative position
                              (theta+np.pi), # relative heading
                              -200.0]) # relative airspeed
    sys.check()
    _, p = sys.simulate_tspan((0.0, 30.0),
                              terminating_conditions_all=air_collision_condition,
                              return_passed=True)
    if issubclass(sys.components["llc"], f16c.F16NNLlcComponent):
        assert p, f"aircraft should not have collided!"
    else:
        # this should hold for the LLC
        assert not p, f"aircraft should have collided!"

