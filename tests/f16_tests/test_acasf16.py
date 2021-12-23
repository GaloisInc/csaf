import pytest
import csaf
import csaf_f16.systems as f16s
import csaf_f16.components as f16c
import csaf_f16.acas as f16acas
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
