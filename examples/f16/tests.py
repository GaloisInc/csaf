import csaf.tests as tests
import csaf.component_lib as lib

component_name = 'controller'

tests.test(component_name,
        lib.signal_generators.step(),
        [tests.overshoot(x0, xref), tests.settling_time(tf, delta)]
        )

tests.test(component_name,
        lib.signal_generators.gaussian_noise(mu, sigma),
        tests.robustness(tf, delta))
