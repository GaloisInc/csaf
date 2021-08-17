"""
LearJet Measurements

learjet_data.py

> Berger, T., Tischler, M., Hagerott, S. G., Cotting, M. C., Gray, W. R., Gresham, J., ... & Howland, J. (2017).
> Development and Validation of a Flight-Identified Full-Envelope Business Jet Simulation Model Using a Stitching
> Architecture. In AIAA Modeling and Simulation Technologies Conference (p. 1550).
"""
from collections import namedtuple
import typing as typ
import numpy as np


learjet_data = {
  'xu': (-0.009725, -0.009896), ## Longitudinal A matrix
  'xw': (0.08642, 0.08382), 
  'xq': (0.0, 0.0),
  'zu': (-0.1119, -0.08969),
  'zw': (-1.432, -1.142),
  'zq': (0.0, 0.0),
  'mu': (0.0004093, 0.0008353),
  'mw': (-0.02352, -0.01783),
  'mq': (-1.65, -1.554),
  'xde': (0.07084, 0.07811), ## Longitudinal B matrx
  'xdt': (0.002289, 0.001944),
  'zde': (-1.244, -0.9845),
  'zdt': (-0.001053, -0.0009109),
  'mde': (-0.1919, -0.1856),
  'mdt': (-3.826E-05, -3.444E-05),
  'yv': (-0.1698, -0.1454),   ## Lateral A matrix
  'yp': (0.8673, 0.527),
  'yr': (0.0, 0.0),
  'lv': (-0.01918, -0.009209),
  'lp': (-2.278, -1.035),
  'lr': (0.8487, 0.7406),
  'nv': (0.005268, 0.004034),
  'np': (-0.2258, -0.1004),
  'nr': (-0.2719, -0.1859),
  'yda': (-0.0132, -0.02062), ## Lateral A matrix
  'ydr': (0.3073, 0.2368),
  'lda': (-0.1623, -0.07185),
  'ldr': (0.03301, 0.0173),
  'nda': (-0.01127, -0.003775),
  'ndr': (-0.03732, -0.02688),
   'vtot': (530.0, 530.0), # Misc
   'g': (32.146, 32.146),
   'w0': (17, 17),
    'u0' : (525.0, 525.0),
   'theta0': (2.378, 2.882) # deg
}


learjet_states = ['u', 'w', 'q', 'theta', 'v', 'p', 'r', 'phi']
learjet_states_descr =  ['perturb x vel', 'perturb z vel', 'pitch rate', 'pitch', 'perturb y vel', 'roll rate', 
                         'yaw rate', 'roll']
permute_states_idxs = [0, 4, 1, 5, 2, 6, 7, 3]
assert set(permute_states_idxs) == set(list(range(8)))

learjet_outputs = ['q', 'alpha', 'ax', 'az', 'udot', 'wdot', 'p', 'r', 'ay', 'beta', 'vdot']
learjet_outputs_descr = ['pitch', 'angle of attack', 'x accel', 'z accel', 'perturb x accel', 'perturb z accel',
                        'roll', 'yaw', 'y accel', 'angle of sideslip', 'perturb y accel']
permute_outputs_idxs = [6, 0, 7, 1, 9, 2, 8, 3, 4, 10, 5]
assert set(permute_outputs_idxs) == set(list(range(11)))


learjet_inputs = ['de', 'dT', 'da', 'dr']
learjet_inputs_descr = ['elevator', 'aileron', 'rudder', 'throttle']
permute_inputs_idxs = [0, 2, 3, 1]
assert set(permute_inputs_idxs) == set(list(range(4)))


#learjet_inputs_trim = (-4.128, 0.0, 0.0, 1366.3), (-3.968, 0.0, 0.0, 1455.9)
learjet_inputs_trim = (-4.128, 0.0, 1366.3, 0.0), (-1.862, 0.0, 1455.9, 0.0)

LearJetParams = namedtuple('LearJetParams',  list(learjet_data.keys()))


ljparam0 = LearJetParams(**{k: v[0] for k, v in learjet_data.items()})
ljparam1 = LearJetParams(**{k: v[1] for k, v in learjet_data.items()})
