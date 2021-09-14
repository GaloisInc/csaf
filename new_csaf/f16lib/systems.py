from csaf.system import System
import typing
import f16lib.components as f16c


class F16Simple(System):
    components = {
        "plant": f16c.F16PlantComponent,
        "llc": f16c.F16LlcComponent,
        "autopilot": f16c.F16GcasComponent
    }

    connections = {
        ("llc", "inputs_pstates"): ("plant", "states"),
        ("llc", "inputs_poutputs"): ("plant", "outputs"),
        ("llc", "inputs_coutputs"): ("autopilot", "outputs"),
        ("plant", "inputs"): ("llc", "outputs"),
        ("autopilot", "inputs_poutputs"): ("plant", "outputs"),
        ("autopilot", "inputs_pstates"): ("plant", "states")
    }

    priority: typing.Optional[typing.Sequence[str]] = None


class F16AirspeedSimple(F16Simple):
    components = {
        "plant": f16c.F16PlantComponent,
        "llc": f16c.F16LlcComponent,
        "autopilot": f16c.F16AutoAirspeedComponent
    }


class F16Shield(System):
    components = {
        "llc": f16c.F16LlcComponent,
        "plant": f16c.F16PlantComponent,
        "autopilot": f16c.F16GcasComponent,
        "autoairspeed": f16c.F16AutoAirspeedComponent,
        "autoaltitude": f16c.F16AutoAltitudeComponent,
        "monitor": f16c.F16MonitorComponent,
        "switch": f16c.F16SwitchComponent
    }

    connections = {
        ("plant", "inputs"): ("llc", "outputs"),

        ("llc", "inputs_pstates"): ("plant", "states"),
        ("llc", "inputs_poutputs"): ("plant", "outputs"),
        ("llc", "inputs_coutputs"): ("switch", "outputs"),

        ("autopilot", "inputs_pstates"): ("plant", "states"),
        ("autopilot", "inputs_poutputs"): ("plant", "outputs"),
        ("autoairspeed", "inputs_pstates"): ("plant", "states"),
        ("autoairspeed", "inputs_poutputs"): ("plant", "outputs"),
        ("autoaltitude", "inputs_pstates"): ("plant", "states"),
        ("autoaltitude", "inputs_poutputs"): ("plant", "outputs"),

        ("monitor", "inputs_pstates"): ("plant", "states"),
        ("monitor", "inputs_poutputs"): ("plant", "outputs"),
        ("monitor", "inputs_gcas"): ("autopilot", "fdas"),

        ("switch", "inputs_0"): ("autopilot", "outputs"),
        ("switch", "inputs_1"): ("autoairspeed", "outputs"),
        ("switch", "inputs_2"): ("autoaltitude", "outputs"),
        ("switch", "inputs_monitors"): ("monitor", "outputs"),
    }


class F16MultiAgentCentral(System):
    from csaf.component import DiscreteComponent

    class CentralController(DiscreteComponent):
        """NOTE: this doesn't do much as it will be used in a SystemEnv"""

        from f16lib.messages import (EmptyMessage, F16PlantStateMessage,
                                     F16PlantOutputMessage, F16ControllerOutputMessage)

        name = "2 F16 Central Controller"
        sampling_frequency = 10.0
        default_initial_values = {
            "inputs_poutputs_0": [0.0,] * 4,
            "inputs_pstates_0": [0.0,] * 13,
            "inputs_poutputs_1": [0.0,] * 4,
            "inputs_pstates_1": [0.0,] * 13,
            "states": []
        }
        parameters: typing.Dict[str, typing.Any] = {}
        states = EmptyMessage
        inputs = (
            ("inputs_pstates_0", F16PlantStateMessage),
            ("inputs_poutputs_0", F16PlantOutputMessage),
            ("inputs_pstates_1", F16PlantStateMessage),
            ("inputs_poutputs_1", F16PlantOutputMessage),
        )
        outputs = (
            ("outputs_0", F16ControllerOutputMessage),
            ("outputs_1", F16ControllerOutputMessage),
        )
        flows = {
            "outputs_0": lambda m, t, s, y: [0.0,] * 4,
            "outputs_1": lambda m, t, s, y: [0.0,] * 4
        }

    components = {
        "plant_a": f16c.F16PlantComponent,
        "llc_a": f16c.F16LlcComponent,
        "plant_b": f16c.F16PlantComponent,
        "llc_b": f16c.F16LlcComponent,
        "autopilot": CentralController,
    }

    connections = {
        ("llc_a", "inputs_pstates"): ("plant_a", "states"),
        ("llc_a", "inputs_poutputs"): ("plant_a", "outputs"),
        ("llc_a", "inputs_coutputs"): ("autopilot", "outputs_0"),
        ("plant_a", "inputs"): ("llc_a", "outputs"),
        ("autopilot", "inputs_poutputs_0"): ("plant_a", "outputs"),
        ("autopilot", "inputs_pstates_0"): ("plant_a", "states"),

        ("llc_b", "inputs_pstates"): ("plant_b", "states"),
        ("llc_b", "inputs_poutputs"): ("plant_b", "outputs"),
        ("llc_b", "inputs_coutputs"): ("autopilot", "outputs_1"),
        ("plant_b", "inputs"): ("llc_b", "outputs"),
        ("autopilot", "inputs_poutputs_1"): ("plant_b", "outputs"),
        ("autopilot", "inputs_pstates_1"): ("plant_b", "states")
    }


class F16AcasShield(System):
    components = {
        "plant": f16c.F16PlantComponent,
        "llc": f16c.F16LlcComponent,
        "autopilot": f16c.F16AcasComponent,
        "autopilot_recovery": f16c.F16AcasComponent,
        "switch": f16c.F16AcasSwitchComponent,
        "predictor": f16c.F16CollisionPredictor,
        "intruder_llc": f16c.F16LlcComponent,
        "intruder_plant": f16c.F16PlantComponent,
        "intruder_autopilot": f16c.F16AutoAirspeedComponent
    }

    connections = {
        ("plant", "inputs"): ("llc", "outputs"),

        ("llc", "inputs_pstates"): ("plant", "states"),
        ("llc", "inputs_poutputs"): ("plant", "outputs"),
        ("llc", "inputs_coutputs"): ("switch", "outputs"),

        ("autopilot", "inputs_own_pstates"): ("plant", "states"),
        #("autopilot", "inputs_own_lstates"): ("llc", "states"),
        ("autopilot", "inputs_other_pstates"): ("intruder_plant", "states"),
        #("autopilot", "inputs_other_lstates"): ("intruder_llc", "states"),

        ("autopilot_recovery", "inputs_own_pstates"): ("plant", "states"),
        #("autopilot_recovery", "inputs_own_lstates"): ("llc", "states"),
        ("autopilot_recovery", "inputs_other_pstates"): ("intruder_plant", "states"),
        #("autopilot_recovery", "inputs_other_lstates"): ("intruder_llc", "states"),

        ("predictor", "inputs_own_pstates"): ("plant", "states"),
        #("predictor", "inputs_own_lstates"): ("llc", "states"),
        ("predictor", "inputs_other_pstates"): ("intruder_plant", "states"),
        #("predictor", "inputs_other_lstates"): ("intruder_llc", "states"),

        ("switch", "inputs"): ("autopilot", "outputs"),
        ("switch", "inputs_recovery"): ("autopilot_recovery", "outputs"),
        ("switch", "inputs_monitors"): ("predictor", "outputs"),

        # setup the intruder plane
        ("intruder_llc", "inputs_pstates"): ("intruder_plant", "states"),
        ("intruder_llc", "inputs_poutputs"): ("intruder_plant", "outputs"),
        ("intruder_llc", "inputs_coutputs"): ("intruder_autopilot", "outputs"),
        ("intruder_plant", "inputs"): ("intruder_llc", "outputs"),
        ("intruder_autopilot", "inputs_poutputs"): ("intruder_plant", "outputs"),
        ("intruder_autopilot", "inputs_pstates"): ("intruder_plant", "states")
    }