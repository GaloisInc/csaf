import csaf.tests as tests
import csaf.component_lib as lib

controller_component = 'controller'
plant_component = 'plant'

system = get_system(controller_component, plant_component)


tests.test(system,
        lib.signal_generators.step(),
        [tests.overshoot(x0, xref), tests.settling_time(tf, delta)]
        )

tests.test(system,
        lib.signal_generators.gaussian(mu, sigma),
        tests.io_stable(tf, delta))

tests.test(controller_component,
        lib.signal_generators.gaussian(mu, sigma),
        tests.single_step_robust_io(tf, delta))

tests.test(controller_component,
        lib.signal_generators.uniform_random(mu, sigma),
        tests.single_step_robust_io(tf, delta))

tests.test(controller_component,
        lib.signal_generators.uniform_random(mu, sigma),
        tests.single_step_robust_io(tf, delta))
