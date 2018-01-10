from __future__ import absolute_import  # включение абсолютных путей по умолчанию для импорта
from __future__ import print_function
import numpy as np
import numpy.linalg as nlg
import scipy as sp
from scipy.constants import G

import matplotlib.pyplot as plt
import math
import threading
import copy
import multiprocessing as mp
import time
from multiprocessing import Process, Queue

import verlet_cython
from scipy.integrate import odeint
import pickle

import pyopencl as cl
import pyopencl.cltypes

import random

from scipy.integrate import odeint


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


def VerletMethod(list_of_radius_and_velocity, list_of_mass_all, N, M, T):
    tau = T / M
    N = len(list_of_radius_and_velocity)
    result = np.zeros((M, N, 6))
    result[0] = copy.copy(list_of_radius_and_velocity)

    # acceleration
    a = np.zeros((N, 3))
    for i in range(0, N):
        accelaration_for_i_body(a, N, list_of_radius_and_velocity, list_of_mass_all, i)

    for i in range(1, M):

        list_of_radius_and_velocity_new = np.zeros((N, 6))
        for j in range(0, N):
            Velocity_form_for_x(list_of_radius_and_velocity, list_of_radius_and_velocity_new, a[j], tau, j)

        a_new = np.zeros((N, 3))

        for k in range(0, N):
            accelaration_for_i_body(a_new, N, list_of_radius_and_velocity_new, list_of_mass_all, k)
        for j in range(0, N):
            Velocity_form_for_v(list_of_radius_and_velocity, list_of_radius_and_velocity_new, a[j], a_new[j], tau, j)
        list_of_radius_and_velocity = copy.copy(list_of_radius_and_velocity_new)
        a = copy.copy(a_new)
        result[i] = copy.copy(list_of_radius_and_velocity)

    return result


def ThreadingWork(M, N, list_of_thread, th_ev):
    for i in range(1, M):
        for elem in list_of_thread:
            elem.wait()
            elem.clear()
        th_ev.set()
        for elem in list_of_thread:
            elem.wait()
            elem.clear()
        th_ev.set()


def ThreadMethod(result, list_of_radius_and_velocity, list_of_radius_and_velocity_new, list_of_mass_all, tau, j, M, list_of_thread, th_ev):
    N = len(list_of_radius_and_velocity)
    a = np.zeros((N, 3))
    accelaration_for_i_body(a, N, list_of_radius_and_velocity, list_of_mass_all, j)
    for i in range(1, M):
        list_of_radius_and_velocity_new[j] = np.zeros(6)
        Velocity_form_for_x(list_of_radius_and_velocity, list_of_radius_and_velocity_new, a[j], tau, j)
        list_of_thread[j].set()
        th_ev.wait()
        th_ev.clear()
        a_new = np.zeros((N, 3))
        accelaration_for_i_body(a_new, N, list_of_radius_and_velocity_new, list_of_mass_all, j)
        Velocity_form_for_v(list_of_radius_and_velocity, list_of_radius_and_velocity_new, a[j], a_new[j], tau, j)
        list_of_radius_and_velocity[j] = copy.copy(list_of_radius_and_velocity_new[j])
        a = copy.copy(a_new)
        result[i, j] = copy.copy(list_of_radius_and_velocity[j])
        list_of_thread[j].set()
        th_ev.wait()
        th_ev.clear()


def VerletMethodThreading(list_of_radius_and_velocity, list_of_mass_all, M, T):
    tau = T / M
    N = len(list_of_radius_and_velocity)
    result = np.zeros((M, N, 6))
    result[0] = copy.copy(list_of_radius_and_velocity)

    th_ev = threading.Event()
    list_of_radius_and_velocity_new = np.zeros((N, 6))
    list_of_thread = []
    for j in range(0, N):
        th = threading.Event()
        list_of_thread.append(th)
    Threads = threading.Thread(target=ThreadingWork, name="ThreadingWork", args=(M, N, list_of_thread, th_ev))
    Threads.start()
    for j in range(0, N):
        t = threading.Thread(target=ThreadMethod, name="thread" + str(j), args=(
            result, list_of_radius_and_velocity, list_of_radius_and_velocity_new, list_of_mass_all, tau, j, M, list_of_thread, th_ev))
        t.start()
    Threads.join()

    return result


def solveForOneBody(q, q_out, list_of_radius_and_velocity, body, shared_pos, tau, list_of_mass_all, M, events1, events2):
    result = np.zeros((M, 6))
    N = len(list_of_radius_and_velocity)
    a = np.zeros((N, 3))
    accelaration_for_i_body(a, N, list_of_radius_and_velocity, list_of_mass_all, body)
    for j in range(1, M):
        list_of_radius_and_velocity_new = np.zeros((N, 6))
        Velocity_form_for_x(list_of_radius_and_velocity, list_of_radius_and_velocity_new, a[body], tau, body)

        q.put([body, list_of_radius_and_velocity_new[body, 0:3]])
        events1[body].set()
        if (body == 0):
            for i in range(N):
                events1[i].wait()
                events1[i].clear()
            for i in range(0, N):
                tmp = q.get()
                shared_pos[tmp[0] * 6:tmp[0] * 6 + 3] = tmp[1]

            for i in range(0, N):
                events2[i].set()
        else:
            events2[body].wait()
            events2[body].clear()

        arr = np.frombuffer(shared_pos.get_obj())
        list_of_radius_and_velocity_new = arr.reshape((N, 6))
        a_new = np.zeros((N, 3))
        accelaration_for_i_body(a_new, N, list_of_radius_and_velocity_new, list_of_mass_all, body)
        Velocity_form_for_v(list_of_radius_and_velocity, list_of_radius_and_velocity_new, a[body], a_new[body], tau, body)
        list_of_radius_and_velocity[body] = copy.copy(list_of_radius_and_velocity_new[body])
        a = copy.copy(a_new)
        result[j] = list_of_radius_and_velocity[body]

    q_out.put([body, result])


def Velocity_form_for_v(list, list_new, a, a_new, tau, j):
    list_new[j, 3:6] = list[j, 3:6] + 0.5 * (a + a_new) * tau


def Velocity_form_for_x(list, list_new, a, tau, j):
    list_new[j, 0:3] = list[j, 0:3] + list[j, 3:6] * tau + 0.5 * a * tau ** 2


def accelaration_for_i_body(a, N, list_of_data, list_of_mass, i):
    G = 6.67 * 10 ** (-11) * MassSun / pow(r_norm, 3)  # gravitation const

    for j in range(0, N):
        if i != j:
            a[i] += G * list_of_mass[j] * (list_of_data[j, 0:3] - list_of_data[i, 0:3]) / nlg.norm(list_of_data[j, 0:3] - list_of_data[i, 0:3],
                                                                                                   2) ** 3


def VerletMethodMultiprocessing(list_of_radius_and_velocity, list_of_mass_all, M, T):
    tau = T / M
    N = len(list_of_radius_and_velocity)
    pos = np.asarray(list_of_radius_and_velocity)
    shared_pos = mp.Array('d', np.zeros(6 * N))

    if __name__ == 'verlet':
        result = np.zeros((M, N, 6))

        events1 = []
        events2 = []

        for elem in list_of_mass_all:
            events1.append(mp.Event())
            events2.append(mp.Event())
            events1[-1].clear()
            events2[-1].clear()

        q = mp.Queue()
        q_out = mp.Queue()
        processes = []
        for body in range(0, N):
            processes.append(mp.Process(target=solveForOneBody,
                                        args=(q, q_out, list_of_radius_and_velocity, body, shared_pos, tau, list_of_mass_all, M, events1, events2)))

            processes[-1].start()

        for i in range(0, N):
            tmp = q_out.get()
            result[:, tmp[0], :] = tmp[1]

        result[0] = copy.copy(list_of_radius_and_velocity)

        for process in processes:
            process.join()

        return result


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


def verlet_simple_array(listRadiusAndVelocity, listMass, N, M, T):
    N = len(listRadiusAndVelocity)
    result = np.zeros((M, N, 6))
    result[0] = copy.copy(listRadiusAndVelocity)
    return result


def cverlet_no_tmv(listRadiusAndVelocityAll, listMassAll, M, T):
    result = verlet_cython.cverletnotypedmemoryview(np.asarray(listRadiusAndVelocityAll), np.asarray(listMassAll), M, T)
    return np.asarray(result)


def cverlet_tmv(listRadiusAndVelocityAll, listMassAll, M, T):
    result = verlet_cython.cverlettypedmemoryview(np.asarray(listRadiusAndVelocityAll), np.asarray(listMassAll), M, T)
    return np.asarray(result)


def cverlet_omp_no_tmv(listRadiusAndVelocityAll, listMassAll, M, T):
    result = verlet_cython.cverlet_openmp(np.asarray(listRadiusAndVelocityAll), np.asarray(listMassAll), M, T)
    return np.asarray(result)


def cverlet_omp_tmv(listRadiusAndVelocityAll, listMassAll, M, T):
    result = verlet_cython.cverlettypedmemoryview_openmp(np.asarray(listRadiusAndVelocityAll), np.asarray(listMassAll), M, T)
    return np.asarray(result)


MassSun = 1.99 * pow(10, 30)
MassEarth = 5.98 * pow(10, 24)
MassMoon = 7.32 * pow(10, 22)
MassMerc = 3.285 * pow(10, 23)
M = 500
T = 5 * 12 * 2592000
r_norm = 1.496 * 10 ** 11


def g(list_of_data, time_span, list_of_mass, N):
    G = 6.67 * 10 ** (-11) * MassSun / pow(r_norm, 3)
    mass_of_funct = np.zeros(6 * N)
    for i in range(0, N):
        f1 = list_of_data[6 * i + 3: 6 * i + 6]
        f2 = np.zeros(3)
        for j in range(0, N):
            if (i != j):
                f2 += G * list_of_mass[j] * (list_of_data[6 * j:6 * j + 3] - list_of_data[6 * i:6 * i + 3]) / nlg.norm(
                    list_of_data[6 * j:6 * j + 3] - list_of_data[6 * i:6 * i + 3], 2) ** 3
        mass_of_funct[6 * i:6 * i + 3] = f1
        mass_of_funct[6 * i + 3:6 * i + 6] = f2
    return mass_of_funct


# Реализовать функцию, генерирующую произвольный набор из K частиц c небольшими скоростями, достаточно большими массами и достаточно
# разнесенные в пространстве. Расчетное время: 10 * линейный размер расчетной области / максимальная скорость.
def TaskOfKRandomBodies(K, type):
    list_of_mass_all = np.zeros((K, 1))
    list_of_radius_and_velocity_all = np.zeros((K, 6))
    step = random.randrange(10 ** 3, 10 ** 10)
    max_velocity = 0  # максимальная скорость
    for i in range(0, K):
        list_of_mass_all[i] = random.randrange(pow(10, 24), pow(10, 30))
        temp = random.randrange(200, 500)
        if (temp > max_velocity):
            max_velocity = temp
        list_of_radius_velocity = [0, i * step, 0,
                                   temp, 0, 0]
        list_of_radius_and_velocity_all[i, :] = list_of_radius_velocity

    N = len(list_of_radius_and_velocity_all)
    init = list_of_radius_and_velocity_all.reshape((6 * N))

    asize = step * K  # линейный размер расчетной области
    T = max_velocity * asize * 10  # Расчетное время
    M = 100
    result = np.zeros(1)
    if (type == "verlet"):
        # result = VerletMethod(list_of_radius_and_velocity_all, list_of_mass_all, N, M, T)
        return result
    if (type == "scipy"):
        time_span = np.linspace(0, T, M)
        result = odeint(g, init, time_span, args=(list_of_mass_all, N))
        result2 = result.reshape((M, N, 6))
        return result2
    if (type == "verlet-threading"):
        # result = VerletMethodThreading(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return result
    if (type == "verlet-cython-no-tmv"):
        result = cverlet_no_tmv(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return np.asarray(result)
    if (type == "verlet-cython-tmv"):
        result = cverlet_tmv(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return np.asarray(result)
    if (type == "verlet-cython-no-tmv-omp"):
        result = cverlet_omp_no_tmv(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return np.asarray(result)
    if (type == "verlet-cython-tmv-omp"):
        result = cverlet_omp_tmv(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return np.asarray(result)


def TaskOfNbodiesVerlet(type):
    list_of_mass_all = [MassMoon / MassSun, MassEarth / MassSun, 1, MassMerc / MassSun]
    list_of_radius_and_velocity_all = np.zeros((4, 6))

    # first body moon
    list_of_radius_velocity = [0, 1.496 * 10 ** 11 / r_norm + 384467000 / r_norm, 0, 1022 / r_norm + 29.783 * 10 ** 3 / r_norm, 0, 0]
    list_of_radius_and_velocity_all[0, :] = list_of_radius_velocity

    # second body earth
    list_of_radius_velocity = [0, 1.496 * 10 ** 11 / r_norm, 0, 29.783 * 10 ** 3 / r_norm, 0, 0]
    list_of_radius_and_velocity_all[1, :] = list_of_radius_velocity

    # third sun
    list_of_radius_velocity = [0, 0, 0, 0, 0, 0]
    list_of_radius_and_velocity_all[2, :] = list_of_radius_velocity

    # fourth mercury
    list_of_radius_velocity = [0, 57910000 * 1000 / r_norm, 0, 47.36 * 1000 / r_norm, 0, 0]
    list_of_radius_and_velocity_all[3, :] = list_of_radius_velocity

    N = len(list_of_radius_and_velocity_all)
    init = list_of_radius_and_velocity_all.reshape((6 * N))

    if (type == "verlet"):
        result = VerletMethod(list_of_radius_and_velocity_all, list_of_mass_all, N, M, T)
        return result
    if (type == "scipy"):
        time_span = np.linspace(0, T, M)
        result = odeint(g, init, time_span, args=(list_of_mass_all, N))
        result2 = result.reshape((M, N, 6))
        return result2
    if (type == "verlet-threading"):
        result = VerletMethodThreading(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return result
    if (type == "verlet-cython-no-tmv"):
        result = cverlet_no_tmv(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return np.asarray(result)
    if (type == "verlet-cython-tmv"):
        result = cverlet_tmv(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return np.asarray(result)
    if (type == "verlet-cython-no-tmv-omp"):
        result = cverlet_omp_no_tmv(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return np.asarray(result)
    if (type == "verlet-cython-tmv-omp"):
        result = cverlet_omp_tmv(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return np.asarray(result)
    if (type == "verlet-multiprocessing"):
        result = VerletMethodMultiprocessing(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return result
    if (type == "verlet-opencl"):
        result = Verlet_OpenCl(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return result


def solveForOneBodyKbodies(q, q_out, list_of_radius_and_velocity, body, shared_pos, tau, list_of_mass_all, n1, n2, M, events1,
                           events2):
    N = len(list_of_radius_and_velocity)
    result = np.zeros((M, N, 6))
    a = np.zeros((N, 3))

    for i in range(n1, n2):
        accelaration_for_i_body(a, N, list_of_radius_and_velocity, list_of_mass_all, i)
    for j in range(1, M):
        list_of_radius_and_velocity_new = np.zeros((N, 6))

        for i in range(n1, n2):
            Velocity_form_for_x(list_of_radius_and_velocity, list_of_radius_and_velocity_new, a[i], tau, i)

        for i in range(n1, n2):
            q.put([i, list_of_radius_and_velocity_new[i, 0:3]])

        events1[body].set()

        if (body == 0):
            for i in range(0, 4):
                events1[i].wait()
                events1[i].clear()

            for i in range(0, N):
                tmp = q.get()
                shared_pos[tmp[0] * 6:tmp[0] * 6 + 3] = tmp[1]
            for i in range(0, 4):
                events2[i].set()
        else:
            events2[body].wait()
            events2[body].clear()

        arr = np.frombuffer(shared_pos.get_obj())
        list_of_radius_and_velocity_new = arr.reshape((N, 6))
        a_new = np.zeros((N, 3))

        for i in range(n1, n2):
            accelaration_for_i_body(a_new, N, list_of_radius_and_velocity_new, list_of_mass_all, i)

        for i in range(n1, n2):
            Velocity_form_for_v(list_of_radius_and_velocity, list_of_radius_and_velocity_new, a[i], a_new[i], tau, i)

        for i in range(n1, n2):
            list_of_radius_and_velocity[i] = copy.copy(list_of_radius_and_velocity_new[i])
        a = copy.copy(a_new)

        for i in range(n1, n2):
            result[j,] = list_of_radius_and_velocity[i]

    for i in range(n1, n2):
        q_out.put([i, result[:, i, :]])


def verletMethodMultiprocessing_K_bodies(list_of_radius_and_velocity, list_of_mass_all, M, T):
    tau = T / M
    N = len(list_of_radius_and_velocity)
    pos = np.asarray(list_of_radius_and_velocity)
    shared_pos = mp.Array('d', np.zeros(6 * N))

    if __name__ == 'verlet':
        result = np.zeros((M, N, 6))

        events1 = []
        events2 = []

        for elem in range(0, 4):
            events1.append(mp.Event())
            events2.append(mp.Event())
            events1[-1].clear()
            events2[-1].clear()

        q = mp.Queue()
        q_out = mp.Queue()
        processes = []

        for body in range(0, 4):
            processes.append(mp.Process(target=solveForOneBodyKbodies, args=(
                q, q_out, list_of_radius_and_velocity, body, shared_pos, tau, list_of_mass_all, int(body * N / 4), int(body * N / 4 + N / 4), M,
                events1,
                events2)))
            processes[-1].start()

        for i in range(0, N):
            tmp = q_out.get()
            result[:, tmp[0], :] = tmp[1]

        result[0] = copy.copy(list_of_radius_and_velocity)

        for process in processes:
            process.join()

        return result


# Error between verlet with threading and scipy
def Calculate_Defect():
    result_odeint = TaskOfNbodiesVerlet("scipy")
    result_solver = TaskOfNbodiesVerlet("verlet-threading")
    defect = np.max(np.max(np.max(result_solver - result_odeint)))
    print("Error = " + repr(defect))


# average time of calculation for N bodies in M calculations
def average_time(M_iter, type):
    av_time = 0
    for i in range(0, M_iter):
        t = time.time()
        result = TaskOfNbodiesVerlet(type)
        if (type == 'verlet-opencl'):
            result, t_cl = TaskOfNbodiesVerlet(type)
            av_time += t_cl
        else:
            av_time += time.time() - t
    return av_time / M_iter


def getAverageTime(type_list):
    M_iter = 3
    for elem in type_list:
        if (elem[0] == "verlet-multiprocessing"):
            continue
        print("Average calculation time" + repr(elem[0]) + "= " + repr(average_time(M_iter, elem[0])))


# The slowest and the fastest method and speed-up
def Time_of_all_methods(type_list):
    min_method = "scipy"
    max_method = "scipy"
    t = time.time()
    result = TaskOfNbodiesVerlet("scipy")
    min_time = 10*20
    max_time = -1
    list = []
    for elem in type_list:
        if (elem[0] == "scipy" or elem[0] == "verlet-multiprocessing"):
            continue
        t = time.time()
        if (elem[0] == 'verlet-opencl'):
            result,t_cl = TaskOfNbodiesVerlet(elem[0])
            time_for_method = t_cl
        else:
            result= TaskOfNbodiesVerlet(elem[0])
            time_for_method = time.time() - t
        list.append((elem[0], time_for_method))
        if (time_for_method < min_time):
            min_time = time_for_method
            min_method = elem[0]
        if (time_for_method > max_time):
            max_time = time_for_method
            max_method = elem[0]

    print("The fastest - " + repr(min_method) + ", time = " + repr(min_time))
    print("The slowest - " + repr(max_method) + ", time = " + repr(max_time))

    print("Speed-up:")
    for elem in list:
        if (elem[0] == "max_method"):
            continue
        print(repr(elem[0]) + " = " + repr(elem[1] / max_time))


# Generate random K bodies with low velocities, big masses and far from each other
# time = 10*space/max_v
def TaskOfKRandomBodies(K, type):
    list_of_mass_all = np.zeros(K)
    list_of_radius_and_velocity_all = np.zeros((K, 6))
    step = random.randrange(10 ** 3, 10 ** 10)

    for i in range(0, K):
        list_of_mass_all[i] = random.randrange(pow(10, 24), pow(10, 30))
        temp = random.randrange(200, 500)

        list_of_radius_velocity = [0, i * step, 0,
                                   temp, 0, 0]
        list_of_radius_and_velocity_all[i, :] = list_of_radius_velocity

    N = len(list_of_radius_and_velocity_all)
    init = list_of_radius_and_velocity_all.reshape((6 * N))

    T = 500 * step * 10  # Расчетное время
    M = 100

    t = time.time()
    if (type == "verlet"):
        result = VerletMethod(list_of_radius_and_velocity_all, list_of_mass_all, N, M, T)
        return result, time.time() - t
    if (type == "scipy"):
        time_span = np.linspace(0, T, M)
        result = odeint(g, init, time_span, args=(list_of_mass_all, N))
        result2 = result.reshape((M, N, 6))
        return result2, time.time() - t
    if (type == "verlet-threading"):
        result = VerletMethodThreading(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return result, time.time() - t
    if (type == "verlet-cython-no-tmv"):
        result = cverlet_no_tmv(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return np.asarray(result), time.time() - t
    if (type == "verlet-cython-tmv"):
        result = cverlet_tmv(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return np.asarray(result), time.time() - t
    if (type == "verlet-cython-no-tmv-omp"):
        result = cverlet_omp_no_tmv(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return np.asarray(result), time.time() - t
    if (type == "verlet-cython-tmv-omp"):
        result = cverlet_omp_tmv(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return np.asarray(result), time.time() - t
    if (type == "verlet-multiprocessing"):
        result = verletMethodMultiprocessing_K_bodies(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return result, time.time() - t
    if (type == "verlet-opencl"):
        result, time_opencl = Verlet_OpenCl(list_of_radius_and_velocity_all, list_of_mass_all, M, T)
        return result, time_opencl


# Calculate for K = 10 50 100 200 500 1000.
def GetTimeSlow(list_of_K):
    time_scipy = []
    for K in list_of_K:
        print(repr('verlet') + " " + repr(K))
        result, time = TaskOfKRandomBodies(K, 'verlet')
        time_scipy.append(time)
    return time_scipy


def GetTimeForKBodies(list_of_K, method):
    time_values = []
    i = 0
    for K in list_of_K:
        if (method == "verlet" or method == "scipy"):
            continue
        print(repr(method) + " " + repr(K))
        result, time = TaskOfKRandomBodies(K, method)
        time_values.append(time)
        i = i + 1

    return time_values


def Verlet_OpenCl(list_of_radius_and_velocity, list_of_mass, M_body, T_body):
    N_body = len(list_of_radius_and_velocity)
    tau_body = T_body / M_body
    N = np.array(N_body)
    M = np.array(M_body)
    T = np.array(T_body)
    tau = np.array(tau_body)

    platforms = cl.get_platforms()
    gpu_devices = [plat.get_devices(cl.device_type.GPU) for plat in platforms]
    gpu_devices = [dev for devices in gpu_devices for dev in devices]  # Flatten to 1d if multiple GPU devices exists
    # use_gpu = False
    if gpu_devices:
        from operator import attrgetter
        dev = max(gpu_devices, key=attrgetter('global_mem_size'))

    list_of_mass_all = np.array(list_of_mass, dtype=cl.cltypes.float)
    list_of_radius_and_velocity_all = np.array(list_of_radius_and_velocity, dtype=cl.cltypes.float)
    list_of_radius_and_velocity_new = np.zeros((N, 6), dtype=cl.cltypes.float)

    result = np.zeros((M, N, 6), dtype=cl.cltypes.float)
    a = np.zeros((N, 3), dtype=cl.cltypes.float)
    a_new = np.zeros((N, 3), dtype=cl.cltypes.float)

    kernel_src = """
                     float norm(__global float *list_of_radius_and_velocity, int i, int j)
                     {
                         double temp=0;
                         for (int k=0; k<3; ++k)
                             temp+=(list_of_radius_and_velocity[6*i+k]-list_of_radius_and_velocity[6*j+k])*(list_of_radius_and_velocity[6*i+k]-list_of_radius_and_velocity[6*j+k]);
                         return sqrt(temp);
                     }
                     void acceleration(__global float *list_of_radius_and_velocity, __global float *list_of_mass_all,__global float *a, int N)
                     {
                         double MassSun = 1.99 * pow(10.0, 30);
                         double r_norm = 1.496 * pow(10.0, 11);
                         double G = 6.67 * pow(10.0,-11)*MassSun/pow(r_norm,3);

                         for (int i=0; i<N; ++i)
                         {
                             for (int k=0; k<3; ++k)
                                 a[3*i+k]=0;
                             for (int j=0; j<N; ++j)
                                 if (i!=j)
                                     for (int k=0; k<3; ++k)
                                         a[3*i+k]+=G*list_of_mass_all[j]*(list_of_radius_and_velocity[6*j+k]-list_of_radius_and_velocity[6*i+k])/pow(norm(list_of_radius_and_velocity,i,j),3);
                         }
                     }
                     __kernel void verlet_cl(__global float *list_of_radius_and_velocity, __global float *list_of_mass_all, __global float *result, __global double *T_cl,__global double *tau_cl, __global int *M_cl, __global int *N_cl, __global float *a, __global float *a_new, __global float *list_of_radius_and_velocity_new)
                     {
                         double T=*T_cl;

                         int M=*M_cl;
                         int N=*N_cl;

                         double tau=*tau_cl;
                         double MassSun = 1.99 * pow(10.0, 30);
                         double r_norm = 1.496 * pow(10.0, 11);
                         double G = 6.67 * pow(10.0,-11)*MassSun/pow(r_norm,3);

                         for (int i=0; i<N; ++i)
                         {
                             for (int k=0; k<3; ++k)
                                 a[3*i+k]=0;
                             for (int j=0; j<N; ++j)
                                 if (i!=j)
                                     for (int k=0; k<3; ++k)
                                         a[3*i+k]+=G*list_of_mass_all[j]*(list_of_radius_and_velocity[6*j+k]-list_of_radius_and_velocity[6*i+k])/pow(norm(list_of_radius_and_velocity,i,j),3);
                         }


                         for (int j=0; j<N; ++j)
                             for (int k=0; k<6; ++k)
                                 result[6*j+k]=list_of_radius_and_velocity[6*j+k];

                         for (int i=1; i<M; ++i)
                         {
                             for (int j=0; j<N; ++j)
                                 for (int k=0; k<3; ++k)
                                     {
                                        list_of_radius_and_velocity_new[6*j+k]=list_of_radius_and_velocity[6*j+k]+list_of_radius_and_velocity[6*j+k+3]*tau+0.5*a[3*j+k]*tau*tau;
                                     }


                             acceleration(list_of_radius_and_velocity_new,list_of_mass_all,a_new,N);
                             for (int j=0; j<N; ++j)
                                 for (int k=0; k<3; ++k)
                                     list_of_radius_and_velocity_new[6*j+k+3]=list_of_radius_and_velocity[6*j+k+3]+0.5*(a[3*j+k]+a_new[3*j+k])*tau;
                             for (int j=0; j<N; ++j)
                                 for (int k=0; k<3; ++k)
                                 {
                                     list_of_radius_and_velocity[6*j+k]=list_of_radius_and_velocity_new[6*j+k];
                                     list_of_radius_and_velocity_new[6*j+k]=0;
                                     list_of_radius_and_velocity[6*j+k+3]=list_of_radius_and_velocity_new[6*j+k+3];
                                     list_of_radius_and_velocity_new[6*j+k+3]=0;
                                     a[3*j+k]=a_new[3*j+k];
                                     result[6*N*i+6*j+k]=list_of_radius_and_velocity[6*j+k];
                                     result[6*N*i+6*j+k+3]=list_of_radius_and_velocity[6*j+k+3];
                                 }
                         }
                     }"""

    ctx = cl.Context(devices=[dev], properties=None, dev_type=None, cache_dir=None)
    print(ctx)
    queue = cl.CommandQueue(ctx, dev, properties=None)
    dev_extensions = dev.extensions.strip().split(' ')
    if 'cl_khr_fp64' not in dev_extensions:
        raise RuntimeError('Device does not support double precision float')

    # Build program in the specified context using the kernel source code
    prog = cl.Program(ctx, kernel_src)

    mf = cl.mem_flags
    buff_list = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=list_of_radius_and_velocity_all)
    buff_of_list_new = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=list_of_radius_and_velocity_new)
    buff_of_a = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
    buff_of_a_new = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a_new)
    buff_of_mass = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=list_of_mass_all)
    buff_of_result = cl.Buffer(ctx, mf.WRITE_ONLY, result.nbytes)
    T_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=T)
    M_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=M)
    N_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=N)
    tau_cl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tau)

    prog.build(options=['-Werror'], devices=[dev], cache_dir=None)
    # try:
    #     prg.build()
    # except:
    #     print(ctx.devices)
    #     print("Error:")
    #     # print(prg.get_build_info(ctx.devices[0], cl.program_build_info.LOG))
    #     raise
    # prg.build()

    t0 = time.time()
    prog.verlet_cl(queue, (1,), None, buff_list, buff_of_mass, buff_of_result, T_cl, tau_cl, M_cl, N_cl, buff_of_a, buff_of_a_new, buff_of_list_new)
    cl.enqueue_read_buffer(queue, buff_of_result, result).wait()
    cl.enqueue_read_buffer(queue, buff_of_a, a).wait()
    t_res = time.time() - t0
    return result, t_res


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
