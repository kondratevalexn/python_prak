import numpy as np
import scipy as sp
from scipy.constants import G

import matplotlib.pyplot as plt
import math
import threading


# class representing moving object. Mass is scalar, r,v are vectors (numpy arrays of 3)
class bodyObject():
    def __init__(self, mass, r, v):
        self.mass = mass  # mass
        self.r = r  # radius-vector
        self.v = v  # speed
        self.a = None  # acceleration (for storing a_n)

        self.temp_r = r  # for storing temporary r

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


# verlet update r
def verlet_update_r(objectsList, dt, n_from=-1, n_to=-1):
    # initialization of object parameters should be in the objects constructor
    n_from = int(n_from)
    n_to = int(n_to)

    if n_from == -1 or n_to == -1:
        obj_range = range(len(objectsList))
    else:
        obj_range = range(n_from, n_to)

    # print(obj_range)
    for i in obj_range:  # change to [n1,n2]
        obj = objectsList[i]
        x_n = obj.r
        v_n = obj.v
        a_n = 0
        for j in range(len(objectsList)):
            if i != j:
                obj_j = objectsList[j]
                a_n += obj.computeAccelerationTowardObject(obj_j)

        x_n1 = x_n + v_n * dt + a_n * dt * dt / 2

        obj.a = a_n
        obj.temp_r = x_n1



# updates v (should be called after updating r)
def verlet_update_v(objectsList, dt, n_from=-1, n_to=-1):
    n_from = int(n_from)
    n_to = int(n_to)

    if n_from == -1 or n_to == -1:
        obj_range = range(len(objectsList))
    else:
        obj_range = range(n_from, n_to)

    for i in obj_range:
        obj = objectsList[i]
        obj.r = obj.temp_r

    for i in obj_range:
        obj = objectsList[i]
        v_n = obj.v

        a_n1 = 0
        for j in range(len(objectsList)):
            if i != j:
                obj_j = objectsList[j]
                a_n1 += obj.computeAccelerationTowardObject(obj_j)
        v_n1 = v_n + (a_n1 + obj.a) * dt / 2
        obj.a = a_n1
        obj.v = v_n1


# 1 verlet iteration. dt is obvious, n_from, n_to indicates what range of objects to compute (for threading)
def verletIteration(objectsList, dt, n_from=-1, n_to=-1):
    # initialization of object parameters should be in the objects constructor
    n_from = int(n_from)
    n_to = int(n_to)
    new_r = np.zeros((len(objectsList), 2))

    if n_from == -1 or n_to == -1:
        obj_range = range(len(objectsList))
    else:
        obj_range = range(n_from, n_to)

    for i in obj_range:  # change to [n1,n2]
        obj = objectsList[i]
        x_n = obj.r
        v_n = obj.v
        a_n = 0
        for j in range(len(objectsList)):
            if i != j:
                obj_j = objectsList[j]
                a_n += obj.computeAccelerationTowardObject(obj_j)

        x_n1 = x_n + v_n * dt + a_n * dt * dt / 2
        obj.a = a_n
        obj.temp_r = x_n1
        new_r[i] = x_n1

    # objectsList = list(tempList)
    # for i in range(len(objectsList)):
    #     objectsList[i].r = new_r[i]

    for obj in objectsList:
        obj.r = obj.temp_r

    # update v in all objects
    for i in obj_range:  # change to [n1,n2]
        obj = objectsList[i]
        v_n = obj.v

        a_n1 = 0
        for j in range(len(objectsList)):
            if i != j:
                obj_j = objectsList[j]
                a_n1 += obj.computeAccelerationTowardObject(obj_j)
        v_n1 = v_n + (a_n1 + obj.a) * dt / 2
        obj.a = a_n1
        obj.v = v_n1


# N objects in k threads -> N/k + 1 objects int first k-1 threads, N % obj_in_threads in the last one
def verlet_threading(objectsList, dt, numIterations, nThreads):
    thread_list_r = []
    thread_list_v = []

    N = len(objectsList)

    xdata = np.zeros((N, iters))
    ydata = np.zeros((N, iters))
    objects_in_thread = 0
    last_n_objects = 0
    numThreads = int(nThreads)
    numIterations = int(numIterations)
    if N % numThreads != 0:
        objects_in_thread = N / numThreads + 1
        last_n_objects = N % objects_in_thread
    else:
        objects_in_thread = N / numThreads
        last_n_objects = objects_in_thread

    for t in range(numIterations):
        # fill threads with functions with ranges to compute
        thread_list_r.clear()
        thread_list_v.clear()

        for i in range(numThreads - 1):
            thread = threading.Thread(target=verlet_update_r, args=(objectsList, dt, i * objects_in_thread, (i + 1) * objects_in_thread))
            thread_list_r.append(thread)
        last_thread = threading.Thread(target=verlet_update_r, args=(objectsList, dt,
                                                                     (numThreads - 1) * objects_in_thread,
                                                                     (numThreads - 1) * objects_in_thread + last_n_objects))
        thread_list_r.append(last_thread)

        for i in range(numThreads - 1):
            thread = threading.Thread(target=verlet_update_v, args=(objectsList, dt, i * objects_in_thread, (i + 1) * objects_in_thread))
            thread_list_v.append(thread)
        last_thread = threading.Thread(target=verlet_update_v, args=(objectsList, dt,
                                                                     (numThreads - 1) * objects_in_thread,
                                                                     (numThreads - 1) * objects_in_thread + last_n_objects))
        thread_list_v.append(last_thread)

        for j in range(N):
            xdata[j, t] = objectsList[j].r[0]
            ydata[j, t] = objectsList[j].r[1]

        # calculate r first
        for thr in thread_list_r:
            thr.start()
        for thr in thread_list_r:
            thr.join()

        # then calculate v
        for thrd in thread_list_v:
            thrd.start()
        for thrd in thread_list_v:
            thrd.join()

    return xdata, ydata


def getVerletDataArrays(objectsList, dt, iterations):
    count = len(objectsList)
    xdata = np.zeros((count, iters + 1))
    ydata = np.zeros((count, iters + 1))
    # print(xdata)

    for i in range(count):
        xdata[i, 0] = objectsList[i].r[0]
        ydata[i, 0] = objectsList[i].r[1]

    t = 0
    for i in range(iters):
        verletIteration(objList, dt)
        t += 1
        for j in range(count):
            xdata[j, t] = objectsList[j].r[0]
            ydata[j, t] = objectsList[j].r[1]

    return xdata, ydata


if __name__ == '__main__':
    objCount = 2
    # xdata = np.zeros((objCount, iters))
    # ydata = np.zeros((objCount, iters))

    m_sun = 1.98892 * (10 ** 30)
    m_earth = 5.972 * (10 ** 24)
    m_moon = 7.3477 * (10 ** 22)

    r_earth_sun = 1.496 * (10 ** 11)

    v_earth_sun = 29.783 * 1000

    v_moon_earth = 1.023 * 1000

    sun = bodyObject(m_sun, np.zeros(2), np.array([0, 0]))
    earth = bodyObject(m_earth, np.array([r_earth_sun, 0]), np.array([0, v_earth_sun]))

    # sun = bodyObject(10**12, np.zeros(3), np.array([0, 0, 0]))
    # earth = bodyObject(1, np.array([1, 0, 0]), np.array([0, 1, 0]))

    objList = [sun, earth]

    axes = plt.subplot(111)

    iters = 300
    # t = 0
    dt = 3600 * 24  # I don't know about dt, mb it should be normalized somehow
    # for i in range(iters):
    #     verletIteration(objList, dt, t)
    #     t += 1

    # xdata, ydata = getVerletDataArrays(objList, dt, iters)
    xdata, ydata = verlet_threading(objList, dt, iters, 2)
    # print(xdata)
    # print(ydata)
    plt.plot(xdata[0], ydata[0], 'r+')
    plt.plot(xdata[1], ydata[1], 'g*')

    plt.show()
