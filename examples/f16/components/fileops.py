import os
import inspect


def construct_path(path):
    assert(len(path) >= 1)
    return os.path.join(path[0], *path[1:])


def prepend_curr_path(path):
    callers_path = inspect.stack()[1].filename
    return os.path.join(os.path.dirname(callers_path), construct_path(path))
