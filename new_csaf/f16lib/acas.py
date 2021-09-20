from f16lib.components import f16_xequil
import f16lib.systems as f16sys
import csaf
import typing
import numpy as np

from svgpath2mpl import parse_path  # type: ignore
import matplotlib as mpl  # type: ignore
import matplotlib.pyplot as plt # type: ignore
from matplotlib.collections import LineCollection  # type: ignore
import matplotlib.animation as animation  # type: ignore
from numpy import sin, cos


class AcasScenario:
    """
    air collision avoidance scenario with one intruder and a balloon (stationary vehicle)

    configuration space:
        * (x_int, y_int) relative position for balloon and intruder
        * (theta) relative heading angle for intruder
        * (v_r) relative airspeed for intruder

    properties:
        * (x_ball, y_ball) relative position for the balloon
        * (v_s) ownship airspeed
        * ((x0, y0, z0) ... (xn, yn, zn)) intruder waypoints
    """

    @staticmethod
    def abs_to_rel(ownship: typing.Sequence[float],
                   intruder: typing.Sequence[float],
                   _: typing.Sequence[float]) -> typing.Tuple[float, float, float, float]:
        """
        Get relative coordinates from the absolute vehicle states
        """
        rel_vel = intruder[0] - ownship[0]
        x, y = (intruder[10] - ownship[10]), (intruder[9] - ownship[9])
        theta = intruder[5] - ownship[5]
        return x, y, theta, rel_vel

    system_type: typing.Type[csaf.System] = f16sys.F16AcasIntruderBalloon

    def __init__(self, balloon_pos: typing.Sequence[float],
                 ownship_airspeed: float,
                 own_waypoints: typing.Sequence[typing.Tuple[float, float, float]],
                 intruder_waypoints: typing.Sequence[typing.Tuple[float, float, float]]):
        self.balloon_pos, self.ownship_airspeed = balloon_pos, ownship_airspeed
        self.waypoints: typing.Sequence[typing.Tuple[float, float, float]] = intruder_waypoints
        self.own_waypoints: typing.Sequence[typing.Tuple[float, float, float]] = own_waypoints

    def create_system(self, coord):
        """create a system from the relative coordinates coord"""
        sys = self.system_type()
        ownship, intruder, balloon = self.rel_to_abs(coord)
        sys._components['intruder_autopilot'].parameters["waypoints"] = self.waypoints
        sys._components['waypoint'].parameters["waypoints"] = self.own_waypoints
        sys.set_state("balloon", balloon)
        sys.set_state("plant", ownship)
        sys.set_state("intruder_plant", intruder)
        return sys

    def rel_to_abs(self, coord: typing.Sequence[float]) -> \
            typing.Tuple[typing.Sequence[float], typing.Sequence[float], typing.Sequence[float]]:
        """
        Create absolute coordinates for the vehicles that satisfies the relative coordinates
        """
        # copy out the f16 trim states
        ownship_states = f16_xequil.copy()
        intruder_states = f16_xequil.copy()
        balloon_pos = f16_xequil.copy()

        # set the positions
        balloon_pos[10], balloon_pos[9] = self.balloon_pos
        intruder_states[10], intruder_states[9] = coord[:2]

        # set the ownship and intruder airspeeds
        ownship_states[0] = self.ownship_airspeed
        intruder_states[0] = self.ownship_airspeed + coord[3]

        # set the relative ehading angle
        intruder_states[5] = coord[2]

        # reset the balloon airspeed
        balloon_pos[0] = 0.0
        balloon_pos[0:9] = [0.0, ] * 9

        ownship_states[11], balloon_pos[11], intruder_states[11] = [1000.0] * 3

        return ownship_states, intruder_states, balloon_pos


class AcasShieldScenario(AcasScenario):
    system_type = f16sys.F16AcasShieldIntruderBalloon


class AcasScenarioViewer:
    @staticmethod
    def create_f16_marker(theta):
        t = parse_path(
            """M 372.74513,979.43631 C 371.65529,965.92359 369.95779,948.56184 368.97287,940.85467 L 367.18212,926.84165 L 349.19283,925.96456 L 331.20356,925.08746 L 327.87949,913.19903 C 325.96331,906.34592 322.47402,900.90981 319.64122,900.36428 C 315.82936,899.63016 248.84768,915.74382 215.29064,925.46768 C 210.6617,926.80902 210.08428,924.54182 210.1256,905.18649 L 210.17207,883.39515 L 262.24096,800.13058 L 314.30987,716.86597 L 314.19174,691.43277 L 314.07357,665.99953 L 231.55162,750.23936 C 186.16455,796.57125 147.65634,834.42935 145.97784,834.36847 C 140.86463,834.18303 125.69304,770.54619 127.92525,758.64747 C 129.05661,752.61685 147.83573,720.75969 172.55121,682.94354 C 200.18189,640.66685 229.45816,590.48182 255.60553,540.57291 L 295.98723,463.49405 L 298.08663,429.99131 L 300.18604,396.48856 L 317.26626,396.48856 L 334.34644,396.48856 L 336.24026,382.67182 C 337.28184,375.07257 338.13405,356.2719 338.13405,340.89249 C 338.13405,306.85496 346.08391,196.28826 350.30292,171.6479 C 353.90673,150.60058 374.67947,74.862184 376.84833,74.862184 C 379.01998,74.862184 399.79443,150.62154 403.38857,171.6479 C 407.7049,196.89905 415.54975,303.20976 415.5849,336.92811 C 415.60112,352.48827 416.45944,372.25489 417.49232,380.85393 L 419.37025,396.48856 L 436.49812,396.48856 L 453.62594,396.48856 L 455.70958,429.99131 L 457.79321,463.49405 L 498.3161,540.92263 C 523.99736,589.99277 553.30295,640.40526 578.3308,678.56664 C 623.49827,747.43608 627.00218,753.61651 627.00218,764.41604 C 627.00218,776.60416 611.0839,834.24507 607.68646,834.35927 C 606.02576,834.41508 567.53592,796.55697 522.15341,750.23015 L 439.6398,665.99953 L 439.55978,691.31272 L 439.47978,716.62591 L 491.54868,799.89049 L 543.61757,883.15508 L 543.61757,905.06645 C 543.61757,924.52436 543.03402,926.80873 538.40603,925.46768 C 514.86526,918.64623 439.34504,899.77426 435.58838,899.77426 C 430.94458,899.77426 429.87625,901.89355 426.20575,918.38691 C 424.45136,926.27017 423.67573,926.57646 405.46648,926.57646 L 386.54971,926.57646 L 384.7414,940.72207 C 383.7468,948.50215 382.04138,965.92359 380.95153,979.43631 C 379.86169,992.94911 378.01527,1004.005 376.84833,1004.005 C 375.68138,1004.005 373.83496,992.94911 372.74513,979.43631 z """)
        t.vertices -= np.array(t.vertices).mean(axis=0)
        theta *= -1
        theta += np.pi
        t.vertices = [(x * cos(theta) - y * sin(theta), x * sin(theta) + y * cos(theta)) for x, y in t.vertices]
        return t

    @staticmethod
    def create_balloon_marker():
        t = parse_path(
            """M53.083,5c-20.547,0-37.204,8.896-37.204,31.387c0,13.679,18.33,31.619,29.373,43.503    c1.214,1.306,2.343,2.542,3.347,3.688c0.292,0.334,0.524,0.654,0.725,0.967c0.115,0.178,0.275,0.32,0.474,0.394    c1,0.369,2.112,0.578,3.286,0.578s2.285-0.209,3.286-0.578c0.199-0.073,0.359-0.215,0.474-0.394    c0.201-0.313,0.433-0.633,0.725-0.967c1.004-1.146,2.133-2.382,3.347-3.688c11.044-11.883,29.373-29.824,29.373-43.503    C90.287,13.896,73.631,5,53.083,5z""")
        t.vertices -= np.array(t.vertices).mean(axis=0)
        theta = np.pi
        t.vertices = [(x * cos(theta) - y * sin(theta), x * sin(theta) + y * cos(theta)) for x, y in t.vertices]
        return t

    def __init__(self, trajs, scenario):
        self.scenario = scenario
        self.own_states = np.array(trajs["plant"].states)
        self.acas_states = np.array(trajs["acas_switch"].outputs_state)
        self.acas_times = np.array(trajs["acas_switch"].times)
        self.plant_times = np.array(trajs["plant"].times)
        self.intruder_states = np.array(trajs["intruder_plant"].states)
        self.balloon_states = np.array(trajs["balloon"].states)
        self.own_pos = self.own_states[:, 9:11][:, ::-1]
        self.intruder_pos = self.intruder_states[:, 9:11][:, ::-1]
        self.balloon_pos = self.balloon_states[:, 9:11][:, ::-1]
        self.ownship_heading = np.array(trajs["plant"].states)[:, 5]
        self.intruder_heading = np.array(trajs["intruder_plant"].states)[:, 5]

    def summary_plot(self):
        fig, ax = plt.subplots(figsize=(10.0, 10.0))
        plt.figure(figsize=(10.0, 10.0))

        ax.plot(*self.own_pos.T, 'k', label='Ownship Path')

        ax.plot(*self.intruder_pos.T, 'r', label='Intruder Path')

        ax.scatter(*self.balloon_pos.T)

        self.plot_static(ax)

        ax.plot(*self.own_pos[-1],
                marker=self.create_f16_marker(self.ownship_heading[-1]),
                c='k',
                markersize=30,
                linestyle='None')

        ax.plot(*self.intruder_pos[-1],
                marker=self.create_f16_marker(self.intruder_heading[-1]),
                c='r',
                markersize=30,
                linestyle='None')

        ax.grid()
        ax.legend()
        ax.axis('equal')
        plt.tight_layout()
        return fig, ax

    def plot_static(self, ax: plt.Axes):
        ax.plot(*self.balloon_pos[-1],  # type: ignore
                marker=self.create_balloon_marker(),
                c='grey',
                markersize=30,
                label="Balloon")

        ax.scatter(*np.array(self.scenario.waypoints)[:, :2].T,  # type: ignore
                   marker='x',
                   c='r',
                   s=200,
                   label='Intruder Waypoints')

        ax.scatter(*np.array(self.scenario.own_waypoints)[:, :2].T,  # type: ignore
                   marker='x',
                   c='k',
                   s=200,
                   label='Own Waypoints')

    def compute_bounds(self):
        pos = np.vstack((self.own_pos, self.intruder_pos))

        xmin, xmax = min(pos[:, 0]), max(pos[:, 0])
        ymin, ymax = min(pos[:, 1]), max(pos[:, 1])
        xmin -= (xmax - xmin) * 0.05
        xmax += (xmax - xmin) * 0.05
        ymin -= (ymax - ymin) * 0.05
        ymax += (ymax - ymin) * 0.05
        return (xmin, xmax), (ymin, ymax)
        # return min(xmin, ymin), max(xmax, ymax)

    def summary_video(self):
        fig = plt.figure(figsize=(10, 10))
        xbs, ybs = self.compute_bounds()
        ax = plt.axes(xlim=xbs, ylim=ybs)

        line, = ax.plot([], [], 'r', lw=2, label='Intruder')

        cmap = mpl.cm.coolwarm
        norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        lineown = LineCollection([], cmap=cmap, norm=norm, lw=2)
        ax.add_collection(lineown)

        own = ax.scatter([], [], marker='x', s=0.001, zorder=300, color='k')
        intruder = ax.scatter([], [], marker='x', s=0.001, zorder=300, color='r')
        skip_size = 3

        self.plot_static(ax)

        # initialization function
        def init():
            # creating an empty plot/frame
            # line.set_data([], [])
            lineown.set_segments([])
            line.set_data([], [])
            own.set_offsets(np.c_[[], []])
            intruder.set_offsets(np.c_[[], []])
            return line,

        # lists to store x and y axis points
        xdata, ydata = [], []
        xowndata, yowndata = [], []
        modes = []

        # animation function
        def animate(i):
            i *= skip_size
            # t is a parameter
            # x, y values to be plotted
            x, y = self.intruder_pos[i]

            # appending new points to x, y axes points list
            xdata.append(x)
            ydata.append(y)
            line.set_data(xdata, ydata)

            x, y = self.own_pos[i]
            xowndata.append(x)
            yowndata.append(y)
            points = np.array([xowndata, yowndata]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lineown.set_segments(segments)

            own.set_offsets(np.c_[[xowndata[-1]], [yowndata[-1]]])
            own.set_paths([self.create_f16_marker(self.ownship_heading[i])])
            idx = np.argmin(np.abs(self.plant_times[i] - self.acas_times))
            mode = self.acas_states[idx][0]
            mode_map = {"clear": 0.0, "strong-left": 1.0, "strong-right": 1.0, "weak-left": 0.75, "weak-right": 0.75}
            modes.append(mode_map[mode])
            lineown.set_array(np.array(modes))

            intruder.set_offsets(np.c_[[xdata[-1]], [ydata[-1]]])
            intruder.set_paths([self.create_f16_marker(self.intruder_heading[i])])

            return line,

        # setting a title for the plot
        plt.grid()
        plt.legend()
        plt.xlabel("East / West Position (ft)")
        plt.ylabel("North / South Position (ft)")

        # call the animator
        return animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=len(self.intruder_pos) // skip_size,
                                       interval=20)
