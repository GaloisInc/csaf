# CSAF

More detailed documentation is coming soon. For now, here's a quick guide for 
getting started with the system. CSAF itself is a middleware framework that
makes creating and evaluating control systems as simple as possible. Control 
loop topologies and component implementations are specified independently of 
the middleware. To show off the power of the system, we've ported the F16 code 
into CSAF as a collection of independent components. To see the system in 
action, simply run the following command:

`./run-example.sh f16-shield`

The only software dependency is docker itself. Also, note that first run of the 
script will be slow, as the docker image is built and cached.

Once the simulation completes, navigate to `examples/f16/output` to view the 
generated heidlauf run:

![f16-shield-heidlauf-run](/uploads/edbd5d4567c74af2ee824ecfe6f3d853/f16-shield-heidlauf-run.png)

and loop topology graph:

![image](/uploads/27e47ebbb19aa11d144db1b01435afb0/image.png)

Here's a quick glance at what's going on behind the scenes. The overall system
and loop topology is defined in `examples/f16/f16_shield_config.toml`. This file
dictates which components are in the system and how they are connected together. 
Individual components are defined in `examples/f16/components`, where a component
consists of a configuration file and an implementation. For example, the F16 plant
is defined by `examples/f16/components/f16plant.py` and 
`examples/f16/components/f16plant.toml`. The middleware message formats that
each component speaks are defined in the ROS message format. The F16 messages 
can be found in `examples/f16/components/msg`.
