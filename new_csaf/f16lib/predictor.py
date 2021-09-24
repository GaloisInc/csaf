from f16lib.components import (f16_xequil,
                               F16PlantOutputMessage,
                               F16PlantComponent,
                               F16LlcComponent,
                               F16AutoWaypointComponent,
                               F16AcasSwitchComponent,
                               create_nagents_acas_xu,
                               StaticObject,
                               F16PlantStateMessage)
import GPy
import csaf
from collections import deque
import typing
import numpy as np


class F16AcasShieldSurrogate(csaf.System):
    class F16SurrogatePlaceholderComponent(csaf.DiscreteComponent):
        name = "F16 Surrogate Placeholder"
        sampling_frequency = 30.0
        default_parameters: typing.Dict[str, typing.Any] = {}
        inputs = ()
        outputs = (
            ("outputs", F16PlantOutputMessage),
        )
        states = F16PlantStateMessage
        default_initial_values = {
            "states": f16_xequil
        }
        flows = {
            "outputs": lambda m, t, s, i: [0.0, ] * 4,
            "states": lambda m, t, s, i: f16_xequil
        }

    components = {
        "plant": F16PlantComponent,
        "llc": F16LlcComponent,
        "autopilot": create_nagents_acas_xu(2),
        "waypoint": F16AutoWaypointComponent,
        "switch": F16AcasSwitchComponent,
        "intruder_plant": F16SurrogatePlaceholderComponent,
        "balloon": StaticObject
    }

    connections = {
        ("plant", "inputs"): ("llc", "outputs"),

        ("waypoint", "inputs_poutputs"): ("plant", "outputs"),
        ("waypoint", "inputs_pstates"): ("plant", "states"),

        ("switch", "inputs"): ("waypoint", "outputs"),
        ("switch", "inputs_recovery"): ("autopilot", "outputs"),
        ("switch", "inputs_select"): ("autopilot", "states"),

        ("llc", "inputs_pstates"): ("plant", "states"),
        ("llc", "inputs_poutputs"): ("plant", "outputs"),
        ("llc", "inputs_coutputs"): ("switch", "outputs"),

        ("autopilot", "inputs_own"): ("plant", "states"),
        ("autopilot", "inputs_intruder0"): ("intruder_plant", "states"),
        ("autopilot", "inputs_intruder1"): ("balloon", "states")
    }



def generate_surrogate_system(predictors: typing.Tuple[GPy.models.GPRegression, GPy.models.GPRegression]):
    """create the 'digital twin' used by the predictor component"""
    # import relevant f16 objects
    from f16lib.messages import F16ControllerOutputMessage, F16PlantOutputMessage, F16PlantStateMessage

    # infer flows from the GP predictors
    def surrogate_state_update(model, t, states, inputs):
        state = list(f16_xequil.copy())
        state[9:11] = [predictor.predict(np.array([[t]]))[0][0][0] for predictor in predictors]
        return state

    def surrogate_output(model, t, states, inputs):
        return [0.0, 0.0, 0.0, 0.0]

    # create the surrogate model component
    class _IntruderSurrogateComponent(csaf.DiscreteComponent):
        name = "F16 Surrogate Model"
        sampling_frequency = 10.0
        default_parameters = {}
        inputs = ()
        outputs = (
            ("outputs", F16PlantOutputMessage),
        )
        states = F16PlantStateMessage
        default_initial_values = {
            "states": f16_xequil
        }
        flows = {
            "outputs": surrogate_output,
            "states": surrogate_state_update
        }

    # bring the component into a system
    class _SurrogateSystem(F16AcasShieldSurrogate):
        components = {**F16AcasShieldSurrogate.components,
                      **{"intruder_plant": _IntruderSurrogateComponent}}

    return _SurrogateSystem


class PredictorBuffer:
    import numpy as np

    # number of steps to take before re-running predictor
    n_steps = 5

    def __init__(self):
        self.pstates = [] #deque(maxlen=5*10)
        self.times = []
        self.init_out = [0.,0.,0.,0.7]
        self._finished = False

    def step(self, t, comp_input):
        """step through the simulation for n steps and collect a buffer for prediction"""
        # get the states and track them over time
        if len(self.times) == 0 or not np.isclose(t, self.times[-1]):
            self.pstates.append((comp_input))
            self.times.append(t)

    @property
    def buffer(self) -> np.array:
        """get the buffer as a numpy array"""
        return np.array(self.pstates)

    @property
    def tbuffer(self) -> np.array:
        return np.array(self.times)

    @property
    def is_finished(self):
        """if simulation terminated"""
        return self._finished


class CollisionPredictor:
    surrogate_type = F16AcasShieldSurrogate

    @staticmethod
    def prod_kernel():
        kern0 = GPy.kern.RBF(1, lengthscale=40, variance=5)
        kern1 = GPy.kern.Spline(1, c=20, variance=5)
        kern0.lengthscale.fix()
        kern = GPy.kern.Prod([kern0, kern1])
        return kern

    @staticmethod
    def make_predictors(tspan, pstates, idx=0):
        tt = np.array(tspan)[:, np.newaxis]
        xt = (pstates[:, 9+13*idx])[:, np.newaxis]
        yt = (pstates[:, 10+13*idx])[:, np.newaxis]
        mx = GPy.models.GPRegression(tt, xt, CollisionPredictor.prod_kernel(), normalizer=True)
        my = GPy.models.GPRegression(tt, yt, CollisionPredictor.prod_kernel(), normalizer=True)
        mx.optimize()
        my.optimize()
        return mx, my

    @staticmethod
    def predict_intruder(tt, tspan, pstates, predictors, idx=0):
        #tt = (np.arange(0, len(pstates), 1) / 10)[:, np.newaxis]
        xt = (pstates[:, 10+13*idx])[:, np.newaxis]
        yt = (pstates[:, 9+13*idx])[:, np.newaxis]
        my, mx = predictors
        x, xv = mx.predict((tspan)[:, np.newaxis])
        y, yv = my.predict((tspan)[:, np.newaxis])
        return (tt.flatten(), xt.flatten(), yt.flatten()), (x.flatten(), xv.flatten()), (y.flatten(), yv.flatten())

    @staticmethod
    def predict_ownship(tt, tspan, pstates, waypoints, idx=0):
        #tt = (np.arange(0, len(pstates), 1) / 10)[:, np.newaxis]
        xt = (pstates[:, 10+idx*13])[:, np.newaxis]
        yt = (pstates[:, 9+idx*13])[:, np.newaxis]

        tr = min(tspan), max(tspan)

        # create pub/sub components out of the configuration
        alt_system = CollisionPredictor.surrogate_type()

        # set the scenario states
        alt_system.set_state('plant', pstates[-1, :13])
        alt_system.set_state('intruder_plant', pstates[-1, 13:26])
        alt_system.set_state('balloon', pstates[-1, 26:39])
        alt_system.set_component_param('waypoint', 'waypoints', waypoints)
        print("WAYPOINTS", waypoints)
        trajs = alt_system.simulate_tspan(tr - min(tr),
                                         show_status=False)

        x = np.array(trajs['plant'].states)[:, 10]
        xv = np.zeros(x.shape)
        y = np.array(trajs['plant'].states)[:, 9]
        yv = np.zeros(y.shape)
        times = np.array(trajs['plant'].times) + min(tr)

        x = np.interp(tspan, times, x)
        xv = np.interp(tspan, times, xv)
        y = np.interp(tspan, times, y)
        yv = np.interp(tspan, times, yv)

        return (tt.flatten(), xt.flatten(), yt.flatten()), \
                (x.flatten(), xv.flatten()), \
                (y.flatten(), yv.flatten())


    def __init__(self, waypoints, own_waypoints):
        self.intruder_waypoints = waypoints
        self.own_waypoints = own_waypoints
        self.pbuffer = PredictorBuffer()
        self.step_count = 0
        self.prev_ret = False
        self.predictors = None

    def step(self, t, comp_input):
        if len(self.pbuffer.times) == 0 or not np.isclose(t, self.pbuffer.times[-1]):
            self.pbuffer.step(t, comp_input)
            self.step_count += 1

    def build_waypoints(self):
        t0 = self.pbuffer.times[-1]
        intruder_airpseed = self.pbuffer.buffer[-1][13]
        balloon_state = self.pbuffer.buffer[-1][-13:]
        inty, intx = self.pbuffer.buffer[-1][9], self.pbuffer.buffer[-1][10]
        intruder_state = self.pbuffer.buffer[-1].copy()[13:13*2]
        times = []
        obstacles = []
        for wp in self.intruder_waypoints:
            d = np.linalg.norm([intx - wp[0], inty - wp[1]])
            t_est = d / intruder_airpseed + t0
            times.append(t_est)
            intruder_state[9] = wp[1]
            intruder_state[10] = wp[0]
            obstacles.append([*balloon_state, *intruder_state, *balloon_state])
        return [], []#times, obstacles

    def train_predictors(self):
        #t = np.arange(0, len(self.pbuffer.buffer), 1) / 10.0
        t = self.pbuffer.tbuffer
        wt, wp = self.build_waypoints()
        predictors = self.make_predictors(
            [*t, *wt],
            np.vstack((
                self.pbuffer.buffer,
                *wp
            )),
            1)
        self.predictors = predictors
        return predictors

    def make_pos_prediction(self):
        t = self.pbuffer.tbuffer
        tspan = np.arange(max(t), max(t) + 10.0, 0.1)
        #predictors = self.train_predictors()
        if self.predictors is None: self.train_predictors()
        _, (intx, _), (inty, _) = self.predict_intruder(t, tspan, self.pbuffer.buffer, self.predictors, idx=1)
        _, (ownx, _), (owny, _) = self.predict_ownship(t, tspan, self.pbuffer.buffer, self.own_waypoints)
        return (ownx, owny), (intx, inty)

    def make_prediction(self):
        if self.step_count % 20 == 0:
            self.train_predictors()
            (ownx, owny), (intx, inty) = self.make_pos_prediction()
            d = min(np.linalg.norm([intx - ownx, inty - owny], axis=0))
            r = (d < 500.0)
            self.prev_ret = r
        return bool(self.prev_ret)