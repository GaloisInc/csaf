"""
CanSat Constellation

cansat/__init__.py

The model is taken from

> Umberto Ravaioli, James Cunningham, John McCarroll, Vardaan Gangal, Kerianne Hobbs,
> "Safe Reinforcement Learning Benchmark Environments for Aerospace Control Systems,"
> IEEE Aerospace, Big Sky, MT, March 2022.
"""

from csaf_examples.cansat.components import (CanSatComponent,
                                             generate_cansat_controller,
                                             generate_cansat_system)

from csaf_examples.cansat.plot import plot_sats, plot_sats_anim