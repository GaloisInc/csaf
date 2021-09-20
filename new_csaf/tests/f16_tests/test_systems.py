import f16lib.systems as f16s
import csaf
import pytest


systems = [(name, oj) for name, oj in f16s.__dict__.items() if isinstance(oj, type) and issubclass(oj, csaf.System)]
abstract_systems = {f16s.System}


@pytest.mark.parametrize("system_name,system", systems)
def test_system(system_name, system):
        if system in abstract_systems:
                with pytest.raises(Exception):
                        system().check()
        else:
                system().check()

@pytest.mark.parametrize("system_name,system", systems)
def test_simulate_system(system_name, system):
        if system not in abstract_systems:
                sys = system()
                t = sys.simulate_tspan((0.0, 1.0))
                assert isinstance(t, dict)


@pytest.mark.parametrize("system_name,system", systems)
def test_validate_system(system_name, system):
        if system not in abstract_systems:
                sys = system()
                p = sys.validate_tspan((0.0, 1.0))
                assert isinstance(p, bool)


@pytest.mark.parametrize("system_name,system", systems)
def test_plot_system(system_name, system):
        if system not in abstract_systems:
                sys = system()
                sys.plot_config("test.tmp.png")


@pytest.mark.parametrize("system_name,system", systems)
def test_env_system(system_name, system):
        if system not in abstract_systems:
                class _SysEnv(csaf.SystemEnv):
                        system_type = system
                        agents = [list(system.components.keys())[0]]
                if system not in abstract_systems:
                        env = _SysEnv()
                        env.check()
                        env.reset()
