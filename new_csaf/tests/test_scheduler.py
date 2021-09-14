from csaf.scheduler import Scheduler
from f16lib.components import F16PlantComponent, F16AutopilotComponent


def test_scheduler():
    a = F16PlantComponent()
    b = F16AutopilotComponent()
    s = Scheduler({"a": a, "b": b}, ["a", "b"])
    assert len(s.get_schedule_tspan([1.0-0.01+1e-08, 1.0])) == 1
    assert len(s.get_schedule_tspan([1.0-0.1+1e-08, 1.0-0.1+0.01+1e-08])) == 2
    assert len(s.get_schedule_tspan([0.0, 1e-08])) == 0
    assert len(s.get_schedule_tspan([1e-08, 1e-08*2])) == 2
