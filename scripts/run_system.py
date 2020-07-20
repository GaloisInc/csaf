import csaf.system as csys
import csaf.config as cconf

cconf.check_system_io("config.toml")
my_system = csys.System.from_toml("config.toml")
my_system.activate_system()

idx = 0
my_idx  = my_system.names.index("combiner")
my_system.components[my_idx].send_message(0, {"epoch" : -1, "Output" : [0,]*15})
while True:
    pass
    #from csaf.component import Component
    #compa = Component(0, 1)

    #compa.bind([], ['5505'])
    #mesg = {'epoch': idx, 'Output': [0.0,]*19}
    #compa.send_message(0, mesg)
    #idx += 1
