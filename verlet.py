import numpy as np
import scipy as sp
from scipy.constants import G

import matplotlib.pyplot as plt
import math


# class representing moving object. Mass is scalar, r,v are vectors (numpy arrays of 3)
class bodyObject():
    def __init__(self, mass, r, v):
        self.mass = mass  # mass
        self.r = r  # radius-vector
        self.v = v  # speed
        self.a = None  # acceleration (for storing a_n)

    # returns gravitational acceleration towards object obj
    def computeAccelerationTowardObject(self, obj):
        dist3 = math.sqrt(np.dot(obj.r - self.r, obj.r - self.r))
        dist3 = dist3 * dist3 * dist3
        return G * obj.mass * (obj.r - self.r) / dist3

    def getAcceleration(self, mass_j, r_j):
        dist3 = math.sqrt(np.dot(r_j - self.r, r_j - self.r))
        dist3 = dist3 * dist3 * dist3
        return G * mass_j * (r_j - self.r) / dist3

    def addCoordTupleToList(self, a):
        a.append((self.r[0], self.r[1]))


def verletIteration(objectsList, dt, t):
    # initialization of object parameters should be in the objects constructor
    new_r = np.zeros((len(objectsList), 3))
    new_v = np.zeros((len(objectsList), 3))

    # update r in all objects
    for i in range(len(objectsList)):
        obj = objectsList[i]
        x_n = obj.r
        v_n = obj.v
        xdata[i][t] = x_n[0]
        ydata[i][t] = x_n[1]
        a_n = 0
        for j in range(len(objectsList)):
            if i != j:
                obj_j = objectsList[j]
                a_n += obj.computeAccelerationTowardObject(obj_j)

        if i == 3:
            print('a')
            print(a_n)
            print('r')
            print(x_n)
            print('v')
            print(v_n)

        x_n1 = x_n + v_n * dt + a_n * dt * dt / 2
        obj.a = a_n
        new_r[i] = x_n1

    # objectsList = list(tempList)

    for i in range(len(objectsList)):
        objectsList[i].r = new_r[i]

    # update v in all objects
    for i in range(len(objectsList)):
        obj = objectsList[i]
        v_n = obj.v

        a_n1 = 0
        for j in range(len(objectsList)):
            if i != j:
                obj_j = objectsList[j]
                a_n1 += obj.computeAccelerationTowardObject(obj_j)
        # print(a_n1)
        v_n1 = v_n + (a_n1 + obj.a) * dt / 2
        obj.a = a_n1
        obj.v = v_n1


iters = 300
objCount = 2
xdata = np.zeros((objCount, iters))
ydata = np.zeros((objCount, iters))

m_sun = 1.98892 * (10 ** 30)
m_earth = 5.972 * (10 ** 24)

r_earth_sun = 1.496 * (10 ** 11)

v_earth_sun = 29.783 * 1000

sun = bodyObject(m_sun, np.zeros(3), np.array([0, 0, 0]))
earth = bodyObject(m_earth, np.array([r_earth_sun, 0, 0]), np.array([0, v_earth_sun, 0]))

#sun = bodyObject(10**12, np.zeros(3), np.array([0, 0, 0]))
#earth = bodyObject(1, np.array([1, 0, 0]), np.array([0, 1, 0]))

objList = [sun, earth]

axes = plt.subplot(111)

t = 0
dt = 3600*24  # I don't know about dt, mb it should be normalized somehow
for i in range(iters):
    verletIteration(objList, dt, t)
    t += 1
# print(xdata)
# print(ydata)
plt.plot(xdata[0], ydata[0], 'r+')
plt.plot(xdata[1], ydata[1], 'g*')

plt.show()
