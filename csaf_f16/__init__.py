"""
F16 Component Models

Models F16 control components via control systems analysis franmework (CSAF) components.

"""
__author__ = ["Ethan Lew", "Michal Podhradsky", "Aditya Zutshi"]
__copyright__ = "Copyright 2021, Galois Inc."
__credits__ = ["Ethan Lew", "Michal Podhradsky", "Aditya Zutshi"]
__license__ = "GPLv3"
__version__ = "0.1"
__maintainer__ = "Ethan Lew"
__email__ = "elew@galois.com"
__status__ = "prototype"


# import the CSAF parts
from csaf_f16.components import *
from csaf_f16.systems import *
from csaf_f16.messages import *