# CSAF CHANGELOG

## Internals
* CSAF currently removed the use of ROSmsg serialization and
ZMQ for component communication--simulation is done by 
calling models implemented as python functions. 
    * *Implementing a 
protocol (ZMQ + ROSmsgs) to utilize non-python component implementations is 
being planned. Currently, nothing in the F16 library needs such
compatibility, and integration with Tensorflow and OpenAI Gym remains in tact
 by this change.*
    * The current architectural approach lessens the internal networking resources
    required by CSAF for easier usage with common
    parallelization tools in the python ecosystem. Systems that are 
    composed purely of python models shouldn't require such sockets to
    be used, for example.
    
* Systems no longer needs an `Component.evaluation_order` to start a simulation, but
now require initial values for the inputs.
    * fixing cyclic dependencies through evaluation order wasn't general enough,
    and systems could be created that failed due to the wrong order being defined.
    * every component must have initial inputs provided to start the system in an
    accurate state. The system object is aware that if an initial state is modified for a component that 
    serves as an input to other components, it will update the initial inputs for the
     other components. This reduces the initialization effort for some of the component inputs.  
    
##  Interface Changes
* messages can now be described entirely in python
```python
from typing import NamedTuple

class ComponentMessage(NamedTuple):
    i_r: float
    j_r: float
```
* models functions remain unchanged
```python
def model_state_update(model, t, states, inputs): ...

def model_output(model, t, states, inputs): ...
```
* components can now be described entirely in python
```python
import csaf


class MyComponent(csaf.DiscreteComponent):
    name = "My Component"
    sampling_frequency = 25.0
    parameters = {"length": 3.0}
    inputs = (("voltage", VoltageMessage),)
    outputs = (("torque", TorqueMessage),)
    states = ComponentMessage
    default_initial_values = {
        "states" : [0.5, 1.0],
        "voltage" : [0.0]
    }
    flows = {
        "states" : model_state_update,
        "torque": model_output
    }
    
```
* currently, *the old description format is incompatible* but a translation layer to the new description format
 can be implemented such that the old library is fully functional
* CSAF elements can be checked via `CsafObj.check()` and find the error location
```python
import csaf
import f16lib.systems as f16

# modified a component inside of F16AcasShield to have have all default initial values defined

class F16Env(csaf.SystemEnv):
    """create an environment so we can implement predictor externally"""
    system_type = f16.F16AcasShield
    agents = ["predictor"]


my_env = F16Env()
my_env.check()

"""
Raises:
AssertionError: |SystemEnv <F16AcasShieldEnv>| |System <F16AcasShield>||Component <F16AcasSwitchComponent>| initial 
values must reference all inputs and states 
"""
```
* CSAF elements can be type checked
```python
class F16AcasSwitchComponent(DiscreteComponent):
    ...
    default_initial_values = {
        "inputs": [0.0,]*4,
        "inputs_recovery": [0.0,]*4,
        "states": [],
        "inputs_monitors": False # oh no! this should be a sequence!
    }

"""
$ mypy components.py

f16lib/components.py:306: error: Dict entry 3 has incompatible type "str": "bool"; expected "str": "Sequence[Any]"
"""
```
* generated block diagrams are cleaner and display more system information

## F16lib Changes
* library components have been translated fully to the new representation and checked
* aircraft plant functions have been decorated in some places with `numba.jit` in order
to speed up simulation time.

| Benchmark      | Original Time | JIT Optimized Time |
| ----------- | ----------- | ----------- |
| F16 ACAS Shield     |  5.37 s | **2.45 s** |
| F16 GCAS Shield     |  2.30 s | **1.03 s** |
