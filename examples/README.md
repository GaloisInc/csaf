# CSAF Examples

The code in these examples are licensed under a standard [BSD 3-Clause License](../LICENSE.txt) unless otherwise noted.


## F16

A F-16 autopilot example is provided that was taken from 
[AeroBenchVVPython](https://github.com/stanleybak/AeroBenchVVPython). The model was designed to test autopilot and 
analysis methods. No claim is made about its accuracy; the F-16 model is based on a common aircraft model with
additional controllers placed on top of it.


### Citation
The code in this example is licensed under [GPL license](LICENSE.txt).

> Heidlauf, P., Collins, A., Bolender, M., & Bak, S. (2018, September). Verification Challenges in F-16 Ground Collision 
> Avoidance and Other Automated Maneuvers. In ARCH@ ADHS (pp. 208-217).

## CanSat Docking

This system provides a multi-agent task scenario, where a group of satellites attempt to rejoin in formation while 
avoiding collision with one another.  The spacecraft model was taken from the [AerospaceRL repository on GitHub](https://github.com/act3-ace/aerospaceRL). The simple 2D model uses applied forces to move a satellite
agent around on a plane. The coordinate system is relative to a "chief" satellite, where chaser satellites are attempting to
approach the chief without crashing into it. To create a multi-agent example, four chaser satellites were instantiated 
around the chief, who attempt to approach the chief satellite without crashing into one another.

### Citations

> * Umberto Ravaioli, James Cunningham, John McCarroll, Vardaan Gangal, Kerianne Hobbs, "Safe Reinforcement Learning Benchmark Environments for Aerospace Control Systems," IEEE Aerospace, Big Sky, MT, March 2022.
> * Kyle Dunlap, Kelly Cohen, and Kerianne Hobbs, “Comparing the Explainability and Performance of Reinforcement Learning and Genetic Fuzzy Systems for Safe Satellite Docking,” North American Fuzzy Information processing Society NAFIPS’2021, Virtual, June 7-9, 2021.
> * Kyle Dunlap, Mark Mote, Kaiden Delsing, Kerianne Hobbs, "Run-Time Assured Reinforcement Learning for Safe Satellite Docking" AIAA SciTech 2022, San Diego, CA, January 2022.
> * Christopher D. Petersen, Sean Phillips, Kerianne L. Hobbs, and Kendra Lang, “Challenge Problem: Assured Satellite Proximity Operations” 1st AAS/AIAA Space Flight Mechanics Meeting, Virtual, February 1-4, 2021.

## Dubins Aircraft Rejoin

This system provides a multi-agent task scenario,  where a group of Dubins aircraft attempt to rejoin in formation 
and collectively fly at a specific heading angle. Dubins aircraft presents a dynamically simple 2D aircraft model, 
taken from the [AerospaceRL repository on GitHub](https://github.com/act3-ace/aerospaceRL).
 
### Citations

> Umberto Ravaioli, James Cunningham, John McCarroll, Vardaan Gangal, Kerianne Hobbs, "Safe Reinforcement Learning Benchmark Environments for Aerospace Control Systems," IEEE Aerospace, Big Sky, MT, March 2022.

## Inverted Pendulum

This system is a *linearized* cart-pole inverted pendulum model.

### Citations

[Inverted Pendulum: System Modeling](https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling)

## Learjet

This system is a model of a Calspan learjet model is described in [Berger, T., Tischler, et al.]. The model was fit from
flight data using linear parameter-varying (LPV) system identification. It contains parameters for both full and nearly 
empty fuel pods.

### Citations

> Berger, T., Tischler, M., Hagerott, S. G., Cotting, M. C., Gray, W. R., Gresham, J., ... & Howland, J. (2017). Development and Validation of a Flight-Identified Full-Envelope Business Jet Simulation Model Using a Stitching Architecture. In AIAA Modeling and Simulation Technologies Conference (p. 1550).