### State Description
```
states = ['vt', 'alpha', 'beta', 'phi', 'theta', 'psi', 'p', 'q', 'r', 'pn', 'pe', 'h', 'power']

State.vt: 'air speed (ft/s)',
State.alpha: 'angle of attack (rad)',
State.beta: 'angle of sideslip (rad)',
State.phi: 'roll angle (rad)',
State.theta: 'pitch angle (rad)',
State.psi: 'yaw angle (rad)',
State.p: 'roll rate (rad/s)',
State.q: 'pitch rate (rad/s)',
State.r: 'yaw rate (rad/s)',
State.pn: 'northward horizontal displacement (ft)',
State.pe: 'eastward horizontal displacement (ft)',
State.h: 'altitude (ft)',
State.power: 'power'
```

### Outputs Description
```
plant outputs = ['Nz', 'Ny', 'az', 'ay']

Output.Nz: 'elevator'      
Output.Ny: 'aileron'
Output.az: 'rudder'
Output.ay: 'throttle'
```

### Notes
- In the file: `f16_shield_config.toml#line11`:
    ```
    # Order to Construct/Evaluate Components
    evaluation_order = ["autopilot", "autoairspeed", "autoaltitude", "monitor_ap", "switch", "controller", "plant"]
    ```
    are in accordance to the diagram in the README.md where the components communications are shown.
- By running the following piece of code, inputs of "plant", "controller", and "autopilot" according to the diagram in the README.md can be seen:
    ```
    model_conf = cconf.SystemConfig.from_toml(config_file)
    model_conf.get_component_settings('autopilot')['sub']
    model_conf.get_component_settings('controller')['sub']
    model_conf.get_component_settings('plant')['sub']
    ```
    To match components name with the ones in the diagram, use this:
    ```
    model_conf.get_component_settings('autopilot')['config']['system_name']
    model_conf.get_component_settings('controller')['config']['system_name']
    model_conf.get_component_settings('plant')['config']['system_name']
    ```
    To get each component's sampling frequency:
    ```
    model_conf.get_component_settings('autopilot')['config']['sampling_frequency']
    model_conf.get_component_settings('controller')['config']['sampling_frequency']
    model_conf.get_component_settings('plant')['config']['sampling_frequency']
    ``` 
- To see the meaning of states of outputs of each component, go to:
    ```
    csaf_architecture/examples/f16/components/msg
    ```  
- Some components do not use all the values from their inputs. E.g. for LLC, it can be seen from:
  ```
  csaf_architecture/examples/f16/components/f16llc.py#L44
  ```  
  and same parent directory for other components.
   


