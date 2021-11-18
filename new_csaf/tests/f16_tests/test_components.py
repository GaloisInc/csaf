import csaf_f16.components as f16c
import csaf
import pytest

# extract component types from csaf_f16
components = [(name, oj) for name, oj in f16c.__dict__.items() if
              isinstance(oj, type) and issubclass(oj, csaf.Component)]
# TODO: we should probably designate this via subclass
abstract_components = {f16c.ContinuousComponent,
                       f16c.DiscreteComponent,
                       f16c.F16AutopilotComponent}


@pytest.mark.parametrize("comp_name,comp", components)
def test_component(comp_name, comp):
    if comp in abstract_components:
        with pytest.raises(AssertionError):
            comp().check()
    else:
        comp().check()


@pytest.mark.parametrize("comp_name,comp", components)
def test_defaults_component(comp_name, comp):
    if comp not in abstract_components:
        compi = comp()
        compi.solve_default()
