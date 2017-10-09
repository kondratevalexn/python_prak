import sys

from numpy import arange, sin, pi

import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QTabWidget, \
    QColorDialog, QSlider, QHBoxLayout, QLabel, QLineEdit, QFileDialog
from PyQt5.QtGui import QTextLine, QColor
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

import random


def isConvertibleToFloat(value):
    try:
        float(value)
        return True
    except:
        return False


class Button(QPushButton):
    def __init__(self, title, parent=0, move_x=0, move_y=0, size_x=0, size_y=0):
        super().__init__(title, parent)
        self.move(move_x, move_y)
        self.resize(size_x, size_y)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.left = 500
        self.top = 20
        self.title = 'Task 3 GUI'
        self.width = 640
        self.height = 860
        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)

        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.show()


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)

        self.axes.set_xlim([-100, 100])
        self.axes.set_ylim([-100, 100])

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Fixed,
                                   QSizePolicy.Fixed)
        FigureCanvas.updateGeometry(self)
        self.plot()

    def plot(self):
        self.draw()

    def zoomIn(self):
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        self.axes.set_xlim(np.divide(xlim, 1.5))
        self.axes.set_ylim(np.divide(ylim, 1.5))
        self.plot()

    def zoomOut(self):
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        self.axes.set_xlim(np.multiply(xlim, 1.5))
        self.axes.set_ylim(np.multiply(ylim, 1.5))
        self.plot()


class MyTableWidget(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)

        # curr mouse position in plot coordinates
        self.curr_x = 0
        self.curr_y = 0
        # circle color (color picker)
        self.color = QColor()
        # circles size (slider)
        self.curr_size = 10

        # list with circles
        self.circles = []

        self.fileName = 'some_qml.qml'
        # gui stuff
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tabs.resize(300, 200)

        # Add tabs
        self.tabs.addTab(self.tab1, "Edit")
        self.tabs.addTab(self.tab2, "Model")

        # Create first tab
        self.tab1.layout = QVBoxLayout(self)
        self.tab1.setLayout(self.tab1.layout)

        # Creating widgets
        self.m = PlotCanvas(self.tab1, width=5, height=5)
        self.m.move(0, 0)

        pb_plus = Button('+', self.tab1, 500, 0, 140, 100)
        pb_minus = Button('-', self.tab1, 500, 100, 140, 100)
        pb_choose_color = Button('Choose Color', self.tab1)

        self.le_x_coord = QLineEdit()
        self.le_y_coord = QLineEdit()
        self.le_slider = QLineEdit()
        self.le_x_coord.setDisabled(True)
        self.le_y_coord.setDisabled(True)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1.0, 100.0)
        pb_save_qml = Button('Save to xml', self.tab1)
        pb_open_qml = Button('Open xml', self.tab1)

        # Adding widgets to layout
        self.tab1.layout.addWidget(self.m)
        self.tab1.layout_coords = QHBoxLayout()
        self.tab1.layout.addLayout(self.tab1.layout_coords)
        self.tab1.layout_coords.addWidget(self.le_x_coord)
        self.tab1.layout_coords.addWidget(self.le_y_coord)
        self.tab1.layout_coords.addWidget(pb_plus)
        self.tab1.layout_coords.addWidget(pb_minus)

        self.tab1.layout_size = QHBoxLayout()
        self.tab1.layout.addLayout(self.tab1.layout_size)
        self.tab1.layout.addWidget(pb_choose_color)
        self.tab1.label_size = QLabel('Choose size ')
        self.tab1.layout_size.addWidget(self.tab1.label_size)
        self.tab1.layout_size.addWidget(self.slider)
        self.tab1.layout_size.addWidget(self.le_slider)

        layout_save = QHBoxLayout()
        layout_open = QHBoxLayout()
        self.tab1.le_saveFileName = QLineEdit()
        self.tab1.le_openFileName = QLineEdit()
        self.tab1.layout.addLayout(layout_save)
        self.tab1.layout.addLayout(layout_open)
        layout_save.addWidget(pb_save_qml)
        layout_save.addWidget(self.tab1.le_saveFileName)
        layout_open.addWidget(pb_open_qml)
        layout_open.addWidget(self.tab1.le_openFileName)

        # connecting slots
        pb_plus.clicked.connect(self.m.zoomIn)
        pb_minus.clicked.connect(self.m.zoomOut)
        pb_choose_color.clicked.connect(self.showColorPicker)
        pb_save_qml.clicked.connect(self.showSaveDialog)
        pb_open_qml.clicked.connect(self.showOpenDialog)
        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.le_slider.textChanged.connect(self.textSliderValueChanged)
        self.m.mpl_connect('motion_notify_event', self.changeCoords)
        self.m.mpl_connect('button_press_event', self.drawCircle)

        self.slider.setValue(10)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def drawCircle(self, event):
        if (event.inaxes):
            circle = customCircle(self.curr_x, self.curr_y, self.curr_size, self.color.name())
            self.m.axes.add_artist(circle)
            self.circles.append(circle)
            self.m.draw()

    def changeCoords(self, event):
        if (event.inaxes):
            self.le_x_coord.setText(str(event.xdata))
            self.le_y_coord.setText(str(event.ydata))
            self.curr_x = event.xdata
            self.curr_y = event.ydata
        else:
            self.le_x_coord.clear()
            self.le_y_coord.clear()

    def showColorPicker(self):
        self.color = QColorDialog.getColor()

    def sliderValueChanged(self, val):
        self.le_slider.setText(str(val))
        self.curr_size = val

    def textSliderValueChanged(self):
        if isConvertibleToFloat(self.le_slider.text()):
            val = float(self.le_slider.text())
            self.slider.setValue(val)
            self.curr_size = val

    def showSaveDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "", "All Files (*);;Text Files (*.txt)", options=options)

    def showOpenDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Python Files (*.py)", options=options)


class customCircle(Circle):
    def __init__(self, x, y, size, color):
        Circle.__init__(self, (x, y), size)
        self.set_color(color)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
