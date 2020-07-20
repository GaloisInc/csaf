
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

import collections
import numpy as np
import sys
import time

class RollingPlotWidget(pg.PlotWidget):
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

if __name__ == "__main__":

    app = QtGui.QApplication([])
    mw = QtGui.QMainWindow()
    mw.setWindowTitle('TWIP Telemetry')
    mw.resize(1600,800)
    cw = QtGui.QWidget()
    mw.setCentralWidget(cw)
    l = QtGui.QVBoxLayout()
    cw.setLayout(l)

    widg = RollingPlotWidget(2, 1000)
    widg.set_pen(0, 'r')
    l.addWidget(widg)

    def update():
        t = time.time()
        widg.push_data([np.sin(10*t),np.cos(t)])
        widg.update_plot()

    timer = pg.QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(10)

    mw.show()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

    while(True):
        widg.push_data([3,4])
        widg.update_plot()