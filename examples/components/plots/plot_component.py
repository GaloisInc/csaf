"""
Rolling Plot Component

Ethan Lew
06/20/2020
"""
import collections

import numpy as np

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from csaf.component import Component


class RollingPlotWidget(pg.PlotWidget):
    """implement notion of streaming data for pyqtgraph"""
    def __init__(self, M, N, *args, **kwargs):
        super(RollingPlotWidget, self).__init__(*args, **kwargs)
        self.data = collections.deque(maxlen=N)
        self.M = M
        self.curves = None
        self.offset = 0

        self.generate_curves()

    def push_data(self, data):
        self.data.append(data)
        self.offset += 1
        if self.curves is None:
            self.generate_curves()

    def generate_curves(self):
        self.curves = []
        for i in range(0, self.M):
            self.curves.append(self.plot([0]))

    def set_pen(self, idx, attr):
        self.curves[idx].setPen(attr)

    def update_plot(self):
        data = np.array(self.data)
        for i in range(0, self.M):
            self.curves[i].setData(data[:, i])
            self.curves[i].setPos(self.offset, 0)


class RollingPlotComponent(Component):
    def __init__(self, widg):
        super().__init__(1, 0)
        self.widg = widg

    def subscriber_iteration(self, recv, sock_num):
        mesg = self.deserialize(recv)
        t = mesg['t']
        angle = mesg['states']['angle']
        ang_rate = mesg['states']['angular_rate']
        self.widg.push_data([angle, ang_rate])
        self.widg.update_plot()

    def activate_subscribers(self):
        super().activate_subscribers()


if __name__ == "__main__":
    app = QtGui.QApplication([])
    mw = QtGui.QMainWindow()
    mw.setWindowTitle('Telemetry')
    mw.resize(1600,800)
    cw = QtGui.QWidget()
    mw.setCentralWidget(cw)
    l = QtGui.QVBoxLayout()
    cw.setLayout(l)

    widg = RollingPlotWidget(2, 100)
    widg.set_pen(0, 'r')
    l.addWidget(widg)


    # run simple pendulum model through pub/sub component
    from csaf.dynamics import DynamicalSystem, Dynamics, dict_to_vec, vec_to_dict
    pendulum_state = lambda t, x, u: np.array([x[1], -9.81/1*np.sin(x[0]) + 0*x[1]])
    pendulum_output = lambda t, x, u: np.array([x[0]])
    pend_dyn = Dynamics(pendulum_state, pendulum_output, None)
    pend_dynsys = DynamicalSystem(["torque"], ["angular_rate", "angle"], ["angle"], pend_dyn)
    compa = Component(0, 1)
    compb = RollingPlotComponent(widg)

    compa.bind([], ['1234'])
    pend_dynsys.bind(['1234'], ['1235'])
    compb.bind(['1235'], [])

    pend_dynsys.activate_subscribers()
    pend_dynsys.debug_node = True
    compb.activate_subscribers()

    import sys

    def update():
        mesg = {'t': 0.0, 'inputs': {'torque': 1.0}, 'states': {'angle': 0.1, 'angular_rate': -0.1}}
        x = np.array([0.1, -10])
        dt = 0.1
        compa.send_message(0, mesg)
        x = dict_to_vec(mesg['states'], pend_dynsys._names_states)
        mesg_x = vec_to_dict(x, pend_dynsys._names_states)
        mesg = {'t': dt, 'inputs': {'torque': 0.0}, 'states':  mesg_x}

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(10)
    mw.show()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
