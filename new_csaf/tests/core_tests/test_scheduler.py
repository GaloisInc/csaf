from csaf.core.scheduler import Scheduler
from f16lib.components import F16PlantComponent, F16GcasComponent


def test_scheduler():
    a = F16PlantComponent()
    b = F16GcasComponent()
    a.check()
    b.check()
    s = Scheduler({"a": a, "b": b}, ["a", "b"])
    assert len(s.get_schedule_tspan([1.0-1/30.0, 1.0])) == 1
    assert len(s.get_schedule_tspan([0.0, 1e-08])) == 2
