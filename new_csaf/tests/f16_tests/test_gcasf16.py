"""
F16 Simple GCAS Tests
"""
import pytest
import f16lib.systems as f16s
import f16lib.components as f16c


@pytest.fixture
def simple_f16():
    """create F16Simple for tests"""
    sys = f16s.F16Simple()
    yield sys


@pytest.fixture
def shield_f16():
    sys = f16s.F16Shield()
    yield sys


@pytest.fixture
def airspeed_f16():
    sys = f16s.F16AirspeedSimple()
    yield sys


def ground_collision_condition(cname, outs):
    """ground collision premature termnation condition"""
    return cname == "plant" and outs["states"][11] <= 0.0


def test_system(simple_f16: f16s.F16Simple):
    simple_f16.check()


def test_shield(shield_f16):
    shield_f16.check()


def test_airspeed(airspeed_f16):
    airspeed_f16.check()


def test_gcas_scenario(simple_f16: f16s.F16Simple):
    """check that GCAS works for the ICs provided"""
    simple_f16.set_state("plant", f16c.f16_gcas_scen)
    t, p = simple_f16.simulate_tspan((0.0, 20.0), terminating_conditions=ground_collision_condition, return_passed=True)
    assert p, f"{simple_f16.__class__.__name__} collided with the ground"

    f16_fail_state = list(f16c.f16_gcas_scen.copy())
    f16_fail_state[11] = 200.0
    simple_f16.set_state("plant", f16_fail_state)
    t, p = simple_f16.simulate_tspan((0.0, 20.0), terminating_conditions=ground_collision_condition, return_passed=True)
    assert not p, f"{simple_f16.__class__.__name__} did not collide with the ground"
