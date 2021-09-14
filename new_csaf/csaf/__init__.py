"""
Control Systems Analysis Framework (CSAF)

CSAF is a framework to minimize the effort required to evaluate, implement, and verify controller design
(classical and learning enabled) with respect to the system dynamics. Its key features are:

* Component based controller design
* Native support for python and C language executables
* Compatibility with external hardware and software processes
* Ease of deployment
"""
__author__ = ["Ethan Lew", "Michal Podhradsky", "Aditya Zutshi"]
__copyright__ = "Copyright 2021, Galois Inc."
__credits__ = ["Ethan Lew", "Michal Podhradsky", "Aditya Zutshi"]
__license__ = "BSD"
__version__ = "0.1"
__maintainer__ = "Ethan Lew"
__email__ = "elew@galois.com"
__status__ = "prototype"

# import major CSAF objects
from csaf.component import Component, DiscreteSolver, ContinuousComponent
from csaf.scheduler import Scheduler
from csaf.system_env import SystemEnv
from csaf.system import ComponentComposition, System
from csaf.solver import LSODASolver, DiscreteSolver
from csaf.trace import TimeTrace
