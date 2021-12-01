import csaf_f16.models.f16 as f16
import csaf_f16.models.llc as llc
import csaf_f16.models.autopilot as auto
import csaf_f16.models.autoairspeed as autoair
import csaf_f16.models.autoaltitude as autoalt
import csaf_f16.models.autowaypoint as awaypoint
import csaf_f16.models.switch as switch
import csaf_f16.models.autoacas as acas
import csaf_f16.models.monitor_ap as monitor
import csaf_f16.models.acas_switch as aswitch
import csaf_f16.models.dummy_predictor as predictor
import csaf_f16.models.nnllc as nnllc

from csaf_f16.messages import *
from csaf import ContinuousComponent, DiscreteComponent
import typing

f16_gcas_scen = [540.0,
                 0.037027160081059704,
                 0.0,
                 0.7853981633974483,
                 -1.2566370614359172,
                 -0.7853981633974483,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 0.0,
                 3600.0,
                 9.0]

f16_xequil = [502.0,
              0.03887505597600522,
              0.0,
              0.0,
              0.03887505597600522,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              1000.0,
              9.05666543872074]


class F16PlantComponent(ContinuousComponent):
    name = "F16 Plant Model"
    sampling_frequency = 30.0
    default_parameters = {
        "s": 300.0,
        "b": 30.0,
        "cbar": 11.32,
        "rm": 1.57e-3,
        "xcgref": 0.35,
        "xcg": 0.35,
        "he": 160.0,
        "c1": -0.770,
        "c2": 0.02755,
        "c3": 1.055e-4,
        "c4": 1.642e-6,
        "c5": 0.9604,
        "c6": 1.759e-2,
        "c7": 1.792e-5,
        "c8": -0.7336,
        "c9": 1.587e-5,
        "rtod": 57.29578,
        "g": 32.17,
        "xcg_mult": 1,
        "cxt_mult": 1,
        "cyt_mult": 1,
        "czt_mult": 1,
        "clt_mult": 1,
        "cmt_mult": 1,
        "cnt_mult": 1,
        "model": "morelli"
    }
    inputs = (
        ("inputs", F16ControllerOutputMessage),
    )
    outputs = (
        ("outputs", F16PlantOutputMessage),
    )
    states = F16PlantStateMessage
    default_initial_values = {
        "states": f16_xequil,
        "inputs": [0.0, 0.0, 0.0, 0.0]
    }
    flows = {
        "outputs": f16.model_output,
        "states": f16.model_state_update
    }


class F16LlcComponent(ContinuousComponent):
    name = "F16 Low Level Controller"
    sampling_frequency = 30.0
    default_parameters = {
        "lqr_name": "lqr_original",
        "throttle_max": 1,
        "throttle_min": 0,
        "elevator_max": 25,
        "elevator_min": -25,
        "aileron_max": 21.5,
        "aileron_min": -21.5,
        "rudder_max": 30.0,
        "rudder_min": -30.0
    }
    inputs = (
        ("inputs_pstates", F16PlantStateMessage),
        ("inputs_poutputs", F16PlantOutputMessage),
        ("inputs_coutputs", F16ControllerOutputMessage)
    )
    outputs = (
        ("outputs", F16ControllerOutputMessage),
    )
    states = F16LlcStateMessage
    default_initial_values = {
        "states": [0.0, 0.0, 0.0],
        "inputs_pstates": f16_xequil,
        "inputs_poutputs": [0.0, 0.0, 0.0, 0.0],
        "inputs_coutputs": [0.0, 0.0, 0.0, 0.0]
    }
    flows = {
        "outputs": llc.model_output,
        "states": llc.model_state_update
    }
    initialize = llc.model_init


class F16NNLlcComponent(F16LlcComponent):
    name = "F16 NN Low Level Controller"
    flows = {
        "outputs": nnllc.model_output,
        "states": nnllc.model_state_update
    }
    initialize = nnllc.model_init


class F16AutopilotComponent(DiscreteComponent):
    name = ""
    sampling_frequency = 10.0
    inputs = (
        ("inputs_pstates", F16PlantStateMessage),
        ("inputs_poutputs", F16PlantOutputMessage),
    )


class F16GcasComponent(F16AutopilotComponent):
    name = "F16 GCAS Autopilot"
    default_parameters = {
        "NzMax": 9.0,
        "vt_des": 502.0
    }
    states = F16AutopilotOutputMessage
    default_initial_values = {
        "states": ["Waiting"],
        "inputs_pstates": f16_xequil,
        "inputs_poutputs": [0.0, 0.0, 0.0, 0.0],
    }
    outputs = (
        ("outputs", F16ControllerOutputMessage),
        ("fdas", F16AutopilotOutputMessage)
    )
    flows = {
        "outputs": auto.model_output,
        "fdas": auto.model_info,
        "states": auto.model_state_update
    }


class F16AutoAirspeedComponent(F16AutopilotComponent):
    name = "F16 Airspeed Autopilot"
    default_parameters = {
        "setpoint": 800.0,  # setpoint in airspeed (ft/s)
        "p_gain": 0.01,  # P controller gain value
        "xequil": [502.0, 0.03887505597600522, 0.0, 0.0, 0.03887505597600522, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0,
                   9.05666543872074]
    }
    states = EmptyMessage
    default_initial_values = {
        "states": [],
        "inputs_pstates": f16_xequil,
        "inputs_poutputs": [0.0, 0.0, 0.0, 0.0],
    }
    outputs = (
        ("outputs", F16ControllerOutputMessage),
    )
    flows = {
        "outputs": autoair.model_output,
    }


class F16AutoAltitudeComponent(F16AutopilotComponent):
    name = "F16 Altitude Autopilot"
    default_parameters = {
        "setpoint": 2500,
        "xequil": f16_xequil
    }
    states = EmptyMessage
    default_initial_values = {
        "states": [],
        "inputs_pstates": f16_xequil,
        "inputs_poutputs": [0.0, 0.0, 0.0, 0.0],
    }
    outputs = (
        ("outputs", F16ControllerOutputMessage),
    )
    flows = {
        "outputs": autoalt.model_output
    }


class F16MonitorComponent(DiscreteComponent):
    name = "F16 Autopilot Monitor"
    sampling_frequency = 10.0
    default_parameters: typing.Dict[str, typing.Any] = {

    }
    default_initial_values = {
        "states": [],
        "inputs_pstates": f16_xequil,
        "inputs_poutputs": [0.0, 0.0, 0.0, 0.0],
        "inputs_gcas": ["Waiting"]
    }
    states = EmptyMessage
    inputs = (
        ("inputs_pstates", F16PlantStateMessage),
        ("inputs_poutputs", F16PlantOutputMessage),
        ("inputs_gcas", F16AutopilotOutputMessage),
    )
    outputs = (
        ("outputs", F16MonitorOutputMessage),
    )
    flows = {
        "outputs": monitor.model_output
    }


class F16SwitchComponent(DiscreteComponent):
    name = "F16 Autopilot Selector"
    sampling_frequency = 10.0
    default_initial_values = {
        "inputs_0": [0.0, 0.0, 0.0, 0.0],
        "inputs_1": [0.0, 0.0, 0.0, 0.0],
        "inputs_2": [0.0, 0.0, 0.0, 0.0],
        "inputs_monitors": ["gcas"],
        "states": []
    }
    default_parameters = {
        "mapper": ["gcas", "altitude", "airspeed"]
    }
    states = EmptyMessage
    inputs = (
        ("inputs_0", F16ControllerOutputMessage),
        ("inputs_1", F16ControllerOutputMessage),
        ("inputs_2", F16ControllerOutputMessage),
        ("inputs_monitors", F16MonitorOutputMessage)
    )
    outputs = (
        ("outputs", F16ControllerOutputMessage),
    )
    flows = {
        "outputs": switch.model_output
    }


def create_collision_predictor(nagents: int) -> typing.Type[DiscreteComponent]:
    class _F16CollisionPredictor(DiscreteComponent):
        name = "F16 Collision Predictor"
        sampling_frequency = 10.0
        default_parameters: typing.Dict[str, typing.Any] = {
            "intruder_waypoints" : ((0.0, 0.0, 1000.0),),
            "own_waypoints" : ((0.0, 0.0, 1000.0),)
        }
        inputs = (
            ("inputs_own", F16PlantStateMessage),
            *[(f"inputs_intruder{idx}", F16PlantStateMessage) for idx in range(nagents)]
        )
        outputs = (
            ("outputs", PredictorOutputMessage),
        )
        states = EmptyMessage
        default_initial_values = {
            "inputs_own": f16_xequil,
            "states": [],
            **{f"inputs_intruder{idx}": f16_xequil for idx in range(nagents)}
        }
        flows = {
            "outputs": predictor.model_output
        }
        initialize = predictor.model_init

    return _F16CollisionPredictor


def create_nagents_acas_xu(nagents: int) -> typing.Type[DiscreteComponent]:
    class _F16AcasComponent(DiscreteComponent):
        name = "F16 Acas Xu Controller"
        sampling_frequency = 10.0
        default_parameters = {
            "roll_rates": (0, -1.5, 1.5, -3.0, 3.0),
            "gains": "nominal",
            "setpoint": 2500.0,
            "xequil": [502.0, 0.03887505597600522, 0.0, 0.0, 0.03887505597600522, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000.0,
                       9.05666543872074]
        }
        inputs = (
            ("inputs_own", F16PlantStateMessage),
            *[(f"inputs_intruder{idx}", F16PlantStateMessage) for idx in range(nagents)]
        )
        outputs = (
            ("outputs", F16ControllerOutputMessage),
        )
        states = F16AutopilotOutputMessage
        default_initial_values = {
            "states": ['clear'],
            "inputs_own": f16_xequil,
            **{f"inputs_intruder{idx}": f16_xequil for idx in range(nagents)}
        }
        flows = {
            "outputs": acas.model_output,
            "states": acas.model_state_update
        }
        initialize = acas.model_init

    return _F16AcasComponent


def switch_model_output(*args):
    inputs = args[-1]
    if inputs[-1] == "clear":
        return inputs[:4]
    else:
        return inputs[4:8]


def switch_model_state(*args):
    inputs = args[-1]
    return [inputs[-1]]


class F16AcasRecoverySwitchComponent(DiscreteComponent):
    name = "F16 Acas Recovery Selector"
    sampling_frequency = 10.0
    default_parameters = {
        "mapper": ["acas", "acas_recovery"]
    }
    inputs = (
        ("inputs", F16ControllerOutputMessage),
        ("inputs_state", F16MonitorOutputMessage),
        ("inputs_recovery", F16PlantOutputMessage),
        ("inputs_recovery_state", F16MonitorOutputMessage),
        ("inputs_select", PredictorOutputMessage),
    )
    outputs = (
        ("outputs", F16ControllerOutputMessage),
        ("outputs_state", F16MonitorOutputMessage)
    )
    states = EmptyMessage
    default_initial_values = {
        "inputs": [0.0, ] * 4,
        "inputs_recovery": [0.0, ] * 4,
        "states": [],
        "inputs_select": [False],
        "inputs_state": ["clear"],
        "inputs_recovery_state": ["clear"]
    }
    flows = {
        "outputs": aswitch.model_output,
        "outputs_state": aswitch.model_output_state
    }
    initialize = None


class F16CollisionPredictor(DiscreteComponent):
    name = "F16 Collision Predictor"
    sampling_frequency = 10.0
    default_parameters: typing.Dict[str, typing.Any] = {
    }
    inputs = (
        ("inputs_own", F16PlantStateMessage),
        ("inputs_intruder0", F16PlantStateMessage),
    )
    outputs = (
        ("outputs", PredictorOutputMessage),
    )
    states = EmptyMessage
    default_initial_values = {
        "inputs_own": f16_xequil,
        "inputs_intruder0": f16_xequil,
        "states": []
    }
    flows = {
        "outputs": predictor.model_output
    }
    initialize = None


class F16AutoWaypointComponent(F16AutopilotComponent):
    name = "F16 Waypoint Autopilot"
    default_parameters = {
        "waypoints": [(5000.0, -1000.0, 1000.0)],
        "airspeed": None
    }
    states = F16AutopilotOutputMessage
    default_initial_values = {
        "states": ['Waiting 1'],
        "inputs_pstates": f16_xequil,
        "inputs_poutputs": [0.0, 0.0, 0.0, 0.0],
    }
    outputs = (
        ("outputs", F16ControllerOutputMessage),
    )
    flows = {
        "outputs": awaypoint.model_output,
        "states": awaypoint.model_state_update
    }
    initialize = awaypoint.model_init


class StaticObject(DiscreteComponent):
    name = "Static Object"
    sampling_frequency = 1.0
    default_parameters: typing.Dict[str, typing.Any] = {}
    inputs = ()
    outputs = ()
    states = F16PlantStateMessage
    default_initial_values = {
        "states": f16_xequil
    }
    flows = {
        "states": lambda m, t, s, i: s
    }


class F16AcasSwitchComponent(DiscreteComponent):
    name = "F16 Acas Monitor"
    sampling_frequency = 10.0
    default_initial_values = {
        "inputs": [0.0, 0.0, 0.0, 0.0],
        "inputs_recovery": [0.0, 0.0, 0.0, 0.0],
        "inputs_select": ["clear"],
        "states": []
    }
    default_parameters = {
        "mapper": ["gcas", "altitude", "airspeed"]
    }
    states = EmptyMessage
    inputs = (
        ("inputs", F16ControllerOutputMessage),
        ("inputs_recovery", F16ControllerOutputMessage),
        ("inputs_select", F16MonitorOutputMessage)
    )
    outputs = (
        ("outputs", F16ControllerOutputMessage),
        ("outputs_state", F16MonitorOutputMessage),
    )
    flows = {
        "outputs": switch_model_output,
        "outputs_state": switch_model_state
    }
