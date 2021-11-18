import csaf_f16.goals as f16g
from csaf.test.scenario import BOptFalsifyGoal
import typing
import csaf
import pytest

# extract goal types from csaf_f16
goals = [(name, oj) for name, oj in f16g.__dict__.items() if
         isinstance(oj, type) and issubclass(oj, csaf.Goal)]
abstract_goals: typing.Set[typing.Type[csaf.Goal]] = {f16g.FixedSimAcasGoal, f16g.FixedSimGoal}


@pytest.mark.parametrize("goal_name,goal", goals)
def test_goal(goal_name, goal):
    if goal in abstract_goals:
        with pytest.raises(AssertionError):
            goal().check()
    else:
        goal().check()


@pytest.mark.parametrize("goal_name,goal", goals)
def test_goal_test(goal_name, goal):
    if goal in abstract_goals or issubclass(goal, BOptFalsifyGoal):
        return
    else:
        assert goal().test()
