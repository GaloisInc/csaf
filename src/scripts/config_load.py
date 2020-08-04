import csaf.config as cconf

s = cconf.SystemConfig.from_toml('../../examples/config/f16_shield_config.toml')
#s = cconf.SystemConfig.from_toml('../../examples/config/inv_pendulum_config.toml')
v, e, el  = s.build_device_graph()
s.assert_io_widths()
s.plot_config()
print(e)
print(el)
