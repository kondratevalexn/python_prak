import sys
import time

from numpy import arange, sin, pi

import numpy as np

from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QSizePolicy, QMessageBox, QWidget, QPushButton, QTabWidget, \
    QColorDialog, QSlider, QHBoxLayout, QLabel, QLineEdit, QFileDialog, QRadioButton, QButtonGroup, QGroupBox
from PyQt5.QtGui import QTextLine, QColor
from PyQt5 import QtGui
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import matplotlib.animation as animation

import matplotlib.pyplot as plt
import math

from verlet import bodyObject
from verlet import verletIteration

import xml.etree.cElementTree as ET

from sphinx.ext.graphviz import graphviz


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
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        self.axes.set_xlim([-100, 100])
        self.axes.set_ylim([-100, 100])

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Fixed,
                                   QSizePolicy.Fixed)
        FigureCanvas.updateGeometry(self)
        self.plot()

    def plot(self):
        self.draw()

    def getFig(self):
        return self.fig

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
        # artist for the plot
        self.artists = []

        self.fileName = 'some_qml.xml'
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
        pb_draw_system = Button('Draw Sun system', self.tab1)
        pb_start_animation = Button('Start animation', self.tab1)

        # tab Edit
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
        self.tab1.le_saveFileName.setText(self.fileName)
        layout_open.addWidget(pb_open_qml)
        layout_open.addWidget(self.tab1.le_openFileName)
        self.tab1.le_openFileName.setText(self.fileName)
        self.tab1.layout.addWidget(pb_draw_system)
        self.tab1.layout.addWidget(pb_start_animation)

        # connecting slots
        pb_plus.clicked.connect(self.m.zoomIn)
        pb_minus.clicked.connect(self.m.zoomOut)
        pb_choose_color.clicked.connect(self.showColorPicker)
        pb_save_qml.clicked.connect(self.showSaveDialog)
        pb_open_qml.clicked.connect(self.showOpenDialog)
        pb_draw_system.clicked.connect(self.drawSunEarthMoonSystem)
        pb_start_animation.clicked.connect(self.startAnimation)
        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.le_slider.textChanged.connect(self.textSliderValueChanged)
        self.m.mpl_connect('motion_notify_event', self.changeCoords)
        self.m.mpl_connect('button_press_event', self.drawCircleMouseClick)

        self.slider.setValue(10)

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        # tab Model
        self.tab2.layout = QVBoxLayout(self)
        self.tab2.setLayout(self.tab2.layout)

        rb_scipy = QRadioButton('scipy')
        rb_verlet = QRadioButton('verlet')
        rb_verletThreading = QRadioButton('verlet-threading')
        rb_verletMultiprocessing = QRadioButton('verlet-multiprocessing')
        rb_verletCython = QRadioButton('verlet-cython')
        rb_verletOpenCL = QRadioButton('verlet-opencl')
        rb_verlet.setChecked(True)
        vbox = QVBoxLayout()
        vbox.addWidget(rb_scipy)
        vbox.addWidget(rb_verlet)
        vbox.addWidget(rb_verletThreading)
        vbox.addWidget(rb_verletCython)
        vbox.addWidget(rb_verletMultiprocessing)
        vbox.addWidget(rb_verletOpenCL)
        groupBox = QGroupBox()
        groupBox.setTitle('Select mode')
        groupBox.setLayout(vbox)
        self.tab2.layout.addWidget(groupBox)
        # self.drawSunEarthMoonSystem()

    def drawCircleMouseClick(self, event):
        if (event.inaxes):
            circle = customCircle(self.curr_x, self.curr_y, self.curr_size, self.color.name())
            self.m.axes.add_artist(circle)
            self.circles.append(circle)
            self.m.draw()

    def drawCircle(self, customCircle):
        a = self.m.axes.add_artist(customCircle)
        self.artists.append(a)
        self.m.draw()

    def drawCirclesList(self):
        for circ in self.circles:
            self.m.axes.add_artist(circ)
        self.m.draw()

    def scaleCircles(self, scale):
        for circ in self.circles:
            circ.scaleCircle(scale)

    def scaleCirclesRadius(self, scale):
        for circ in self.circles:
            circ.radius *= scale

    def setCentersAfterIteration(self):
        for circ in self.circles:
            circ.setCenterFromRadiusVector()

    def drawSunEarthMoonSystem(self):  # all velocities are [0,v]
        self.circles.clear()
        self.m.axes.clear()
        r_earth_sun = 1.496 * (10 ** 11)
        r_moon_earth = 3.844 * (10 ** 8)
        r_moon_sun = r_earth_sun + r_moon_earth
        sun = getSunCircle(0, 0)
        earth = getEarthCircle(r_earth_sun, 0)
        moon = getMoonCircle(r_moon_sun, 0)
        self.circles.append(sun)
        self.circles.append(earth)
        self.circles.append(moon)
        self.scaleCircles(1e-9)
        self.scaleCirclesRadius(0.5)
        print(self.circles[1].x)
        self.drawCirclesList()

    def startAnimation(self):
        t = 0
        iters = 20
        dt = 3600 * 24 *1e-4 # I don't know about dt, mb it should be normalized somehow
        for i in range(iters):
            print(self.circles[1].r)
            print(self.circles[1].v)
            verletIteration(self.circles, dt, t)
            self.setCentersAfterIteration()
            self.m.axes.clear()
            self.m.draw()
            self.drawCirclesList()
            t += 1

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
        self.save2Xml()
        return
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "", "All Files (*);;Text Files (*.txt)",
                                                       options=options)

    def showOpenDialog(self):
        self.openXml()
        return
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Python Files (*.py)",
                                                       options=options)

    def save2Xml(self):
        root = ET.Element("data")
        params = ET.SubElement(root, "parameters")
        plot_size = ET.SubElement(params, "plot_size")
        plot_size.text = str(self.m.axes.get_xlim())
        color = ET.SubElement(params, "color")
        color.text = self.color.name()
        slider = ET.SubElement(params, "slider_data")

        slider_min = ET.SubElement(slider, "min_value")
        slider_min.text = str(self.slider.minimum())

        slider_max = ET.SubElement(slider, "max_value")
        slider_max.text = str(self.slider.maximum())

        slider_value = ET.SubElement(slider, "value")
        slider_value.text = str(self.slider.value())

        circles = ET.SubElement(root, "circles")
        for cir in self.circles:
            circle = ET.SubElement(circles, "circle")
            ET.SubElement(circle, "x").text = str(cir.x)
            ET.SubElement(circle, "y").text = str(cir.y)
            ET.SubElement(circle, "size").text = str(cir.radius)
            ET.SubElement(circle, "color").text = str(cir.color)

        tree = ET.ElementTree(root)
        tree.write(self.fileName)

    def openXml(self):
        tree = ET.parse(self.fileName)
        parameters = tree.find("parameters")
        plot_size = parameters.find("plot_size")
        spl = plot_size.text.split('(')[1].split(')')[0].split(',')
        lim = (float(spl[0]), float(spl[1]))

        color = parameters.find('color').text

        slider = parameters.find('slider_data')
        min_v = slider.find('min_value').text
        max_v = slider.find('max_value').text
        v = slider.find('value').text

        self.m.axes.clear()
        self.m.axes.set_xlim(lim)
        self.m.axes.set_ylim(lim)
        self.color.setNamedColor(color)
        self.slider.setMinimum(int(min_v))
        self.slider.setMaximum(int(max_v))
        self.slider.setValue(int(v))

        self.circles.clear()
        circles_root = tree.find("circles")
        circles = circles_root.getiterator("circle")
        for c in circles:
            x = c.find("x").text
            y = c.find("y").text
            radius = c.find("size").text
            color = c.find("color").text
            newCircle = customCircle(float(x), float(y), float(radius), color)
            self.circles.append(newCircle)
            self.m.axes.add_artist(newCircle)
        self.m.draw()


class customCircle(Circle, bodyObject):
    def __init__(self, x, y, radius, color, speed_v=np.zeros(2)):
        mass = math.exp(radius)
        Circle.__init__(self, (x, y), radius)
        bodyObject.__init__(self, mass, np.array([x, y]), speed_v)
        self.set_color(color)
        self.x = x
        self.y = y
        self.color = color

    def setCenterFromRadiusVector(self):
        self.center = (self.r[0], self.r[1])

    def setMassAndRadiusByMass(self, mass):
        self.mass = mass
        self.radius = math.log10(mass)

    def scaleCircle(self, scale):
        # if we decrease r in n than we should decrease mass in n^2 since acceleration is ~m/r/r
        # this function should be called for all objects in list before the iteration starts (because all radiuses should be changed by scale)
        self.mass *= scale * scale
        self.x *= scale
        self.y *= scale
        self.r = np.array([self.x, self.y])
        self.center = (self.x, self.y)
        self.set_radius(self.radius)
        self.v *= scale


def getSunCircle(x, y):
    sun = customCircle(x, y, 10, 'yellow', np.zeros(2))
    m_sun = 1.98892 * (10 ** 30)
    sun.setMassAndRadiusByMass(m_sun)
    return sun


def getEarthCircle(x, y):
    v_earth_sun = np.array([0, 29.783 * 1000])  # m/sec
    earth = customCircle(x, y, 10, 'green', v_earth_sun)
    m_earth = 5.972 * (10 ** 24)  # kg
    earth.setMassAndRadiusByMass(m_earth)
    # r_earth_sun = 1.496 * (10 ** 11)
    return earth


def getMoonCircle(x, y):
    v_moon_earth = np.array([0, 1.023 * 1000])
    v_moon_sun = np.array([0, 29.783 * 1000]) + v_moon_earth  # equals to v_earth_sun + v_moon_earth, same velocity direction as v_earth_sun
    moon = customCircle(x, y, 10, 'gray', v_moon_sun)
    m_moon = 7.3477 * (10 ** 22)
    moon.setMassAndRadiusByMass(m_moon)
    return moon


def getMercuryCircle(x, y):
    v_mercury_sun = np.array([0, 47.36 * 1000])
    mercury = customCircle(x, y, 10, 'red', v_mercury_sun)
    m_mercury = 3.33022 * (10 ** 23)
    mercury.setMassAndRadiusByMass(m_mercury)
    return mercury


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
