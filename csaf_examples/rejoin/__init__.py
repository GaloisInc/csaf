"""
Dubins Rejoin Agent Example

rejoin/__init__.py

Model is taken from

> Umberto Ravaioli, James Cunningham, John McCarroll, Vardaan Gangal, Kerianne Hobbs,
> "Safe Reinforcement Learning Benchmark Environments for Aerospace Control Systems,"
> IEEE Aerospace, Big Sky, MT, March 2022.
"""
from csaf_examples.rejoin.components import (DubinsComponent,
                                             generate_dubins_controller,
                                             generate_dubins_system)

from csaf_examples.rejoin.plot import plot_air_anim, plot_aircrafts