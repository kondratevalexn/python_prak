# -*- coding: utf-8 -*-
# Импортируем все необходимые библиотеки:
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import sys
import math

# Из модуля random импортируем одноименную функцию random
from random import random

# объявляем массив pointcolor глобальным (будет доступен во всей программе)
global pointcolor


def setList(l, v):
    """Set all elements of a list to the same bvalue"""
    for i in range(len(l)):
        l[i] = v


class renderParam(object):
    """Class holding current parameters for rendering.

    Parameters are modified by user interaction"""

    def __init__(self):
        self.initialColor = [1, 1, 1]
        self.drawColor = self.initialColor
        self.tVec = [0, 0, 0]
        self.mouseButton = None
        self.angle = 10
        self.dx = 1
        self.dy = 1

    def reset(self):
        self.drawColor = self.initialColor
        setList(self.tVec, 0)
        self.mouseButton = None
        self.angle = 10
        self.dx = 1
        self.dy = 1


rP = renderParam()

oldMousePos = [0, 0]


def mouseButton(button, mode, x, y):
    """Callback function (mouse button pressed or released).

    The current and old mouse positions are stored in
    a	global renderParam and a global list respectively"""

    global rP, oldMousePos
    if mode == GLUT_DOWN:
        rP.mouseButton = button
    else:
        rP.mouseButton = None
    oldMousePos[0], oldMousePos[1] = x, y
    glutPostRedisplay()


def mouseMotion(x, y):
    """Callback function (mouse moved while button is pressed).

    The current and old mouse positions are stored in
    a	global renderParam and a global list respectively.
    The global translation vector is updated according to
    the movement of the mouse pointer."""

    global rP, oldMousePos
    deltaX = x - oldMousePos[0]
    deltaY = y - oldMousePos[1]
    if rP.mouseButton == GLUT_LEFT_BUTTON:
        factor = 0.01
        rP.tVec[0] += deltaX * factor
        rP.tVec[1] -= deltaY * factor
        oldMousePos[0], oldMousePos[1] = x, y
    if rP.mouseButton == GLUT_RIGHT_BUTTON:
        rP.angle = 100
        dist = math.sqrt(deltaX * deltaX + deltaY * deltaY)
        rP.dx = deltaX / dist
        rP.dy = deltaY / dist
        oldMousePos[0], oldMousePos[1] = x, y
    glutPostRedisplay()


def drawCone(position=(0, -1, 0), radius=1, height=2, slices=50, stacks=10):
    glPushMatrix()
    try:
        # glTranslatef(*position)

        glRotatef(250, 1, 0, 0)
        glutSolidCone(radius, height, slices, stacks)
    finally:
        glPopMatrix()


def drawCube():
    verticies = (
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, -1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, -1, 1),
        (-1, 1, 1)
    )

    edges = (
        (0, 1),
        (0, 3),
        (0, 4),
        (2, 1),
        (2, 3),
        (2, 7),
        (6, 3),
        (6, 4),
        (6, 7),
        (5, 1),
        (5, 4),
        (5, 7)
    )
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(verticies[vertex])
    glEnd()
    # print(repr(verticies[0][0])+','+repr(verticies[0][1]));

    glBegin(GL_POINTS);
    glColor3f(1, 1, 1);
    for i in range(0, 8):
        glVertex3f(verticies[i][0], verticies[i][1], verticies[i][2]);
    glEnd();


def display(swap=1, clear=1):
    """Callback function for displaying the scene

    This defines a unit-square environment in which to draw,
    i.e. width is one drawing unit, as is height
    """
    if clear:
        glClearColor(0.5, 0.5, 0.5, 0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # establish the projection matrix (perspective)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    x, y, width, height = glGetDoublev(GL_VIEWPORT)
    gluPerspective(
        45,  # field of view in degrees
        width / float(height or 1),  # aspect ratio
        .25,  # near clipping plane
        200,  # far clipping plane
    )


    glTranslatef(rP.tVec[0], rP.tVec[1], rP.tVec[2])
    glRotatef(rP.angle, rP.dx, rP.dy, 0)

    drawCone()
    drawCube()
    if swap:
        glutSwapBuffers()


def main():
    glutInitWindowSize(300, 300)
    glutInitWindowPosition(50, 50)

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    glutCreateWindow(b"title")

    glutDisplayFunc(display)
    # glutKeyboardFunc(key_pressed)
    glutIdleFunc(display)
    glutMouseFunc(mouseButton)
    glutMotionFunc(mouseMotion)
    glClearColor(0.2, 0.2, 0.2, 1)

    # note need to do this to properly render faceted geometry
    glutMainLoop()


if __name__ == "__main__":
    main()
