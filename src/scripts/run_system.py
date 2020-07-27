import csaf.system as csys

my_system = csys.System.from_toml("../../examples/config/config.toml")
my_system.simulate_tspan([0, 10])
