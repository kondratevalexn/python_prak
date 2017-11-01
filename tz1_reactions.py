# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from gevent._socket2 import socket
from scipy.integrate import odeint
from sympy import Symbol, solve, lambdify, Matrix
import sys

# var5
# dx/dt = k1*z - km1*x - k3*x*z + km3*y - k2*z*z*x
# dy/dt = k3*x*z - km3*y
# z = 1-x-2*y
# x,y,z \in [0,1]
# k1 = 0.12, km1 = 0.01, k3 = 0.0032, k2 = 0.95, km3 = 0.002 - базовый

# 1: x_c, y_c from (k1) or (k2) for different km1
# 2: -//- for different km3
# 3: find bifurcation points and plot them

# 4: on (k1,k2) or (k1, km1) or (km1, k2) get phase portret, lines (кратности и нейтральности). Bifurcation points ко-размерности-2: С
# (3-х кратный корень) и ТВ (когда два с.з. равны нулю)

# 5: Задать параметры из области автоколебаний. Решить систему (1), задав некоторые начальные данные. Нарисовать графики установившихся
# колебаний х(t) и у(t). На фазовой плоскости построить фазовый портрет системы:
# отметить стационарную точку, нарисовать предельный цикл, нарисовать несколько траекторий, которые наматываются на цикл.

eps = 1e-12

k1_val = 0.12
km1_val = 0.01
k3_val = 0.0032
k2_val = 0.95
km3_val = 0.002

k1 = Symbol("k1")
km1 = Symbol("km1")
k2 = Symbol("k2")
k3 = Symbol("k3")
km3 = Symbol("km3")
x = Symbol("x")
y = Symbol("y")
# dx/dt = k1*z - km1*x - k3*x*z + km3*y - k2*z*z*x
# dy/dt = k3*x*z - km3*y
# z = 1-x-2*y

eq1 = k1 * (1 - x - 2 * y) - km1 * x - k3 * x * (1 - x - 2 * y) + km3 * y - k2 * (1 - x - 2 * y) ** 2 * x
eq2 = k3 * x * (1 - x - 2 * y) - km3 * y

solution = solve([eq1, eq2], k1, y)
eq_k1 = solution[0][0]
eq_y = solution[0][1]

A = Matrix([eq1, eq2])
var_vector = Matrix([x, y])
jacA = A.jacobian(var_vector)
detA = jacA.det()
traceA = jacA.trace()
discrA = traceA ** 2 - 4 * detA

N = 100
xarray = np.linspace(0, 1, N)


# k1 = 0.12, km1 = 0.01, k3 = 0.0032, k2 = 0.95, km3 = 0.002 - базовый
def dependenceOnParameterK1(km1_new, km3_new):
    func_k1 = lambdify((x, km1, k2, km3, k3), eq_k1)
    func_y = lambdify((x, km3, k3), eq_y)
    func_detA = lambdify((x, km1, k2, km3, k3), detA.subs(y, eq_y).subs(k1, eq_k1))

    detA_list = list(func_detA(xarray, km1_new, k2_val, km3_new, k3_val))
    sarray = []  # bifurcation points x coordinate
    for i in range(len(xarray) - 1):
        if detA_list[i] * detA_list[i - 1] <= 0:
            sarray.append(xarray[i] - detA_list[i] * (xarray[i + 1] - xarray[i]) / (detA_list[i + 1] - detA_list[i]))
    sarray = np.array(sarray)
    sn = func_k1(sarray, km1_new, k2_val, km3_new, k3_val)

    # x(k1)
    plt.plot(func_k1(xarray, km1_new, k2_val, km3_new, k3_val), xarray, color='b', label="$x_{k_1}$")
    # y(k1)
    plt.plot(func_k1(xarray, km1_new, k2_val, km3_new, k3_val), func_y(xarray, km3_new, k3_val), color='g', label="$y_{k_1}$")
    # bifurcation on x(k1)
    plt.plot(sn, sarray, color='b', linestyle='', marker='^')
    # bifurcation on y(k1)
    plt.plot(sn, func_y(sarray, km3_new, k3_val), color='g', linestyle='', marker='o')

    plt.xlabel('$k_2$')
    plt.ylabel("x,y")
    plt.xlim([0.0, 0.7])
    plt.ylim([-0.05, 1.1])
    plt.legend(loc=2)
    plt.show()
    return


def twoParameterAnalysisK1K2(km1val, k3val, km3val):
    # Neutral lines
    eq_k1Trace = solve(traceA.subs(y, eq_y), k1)[0]
    eq_k2JointTrace = solve(eq_k1Trace - eq_k1, k2)[0]
    eq_k1JointTrace = eq_k1.subs(k2, eq_k2JointTrace)

    k1Trace_of_x = lambdify((x, k3, km3, km1), eq_k1JointTrace)
    k2Trace_of_x = lambdify((x, k3, km3, km1), eq_k2JointTrace)

    # Multiplicity lines
    eq_k1Det = solve(detA.subs(y, eq_y), k1)[0]
    eq_k2JointDet = solve(eq_k1Det - eq_k1, k2)[0]
    eq_k1JointDet = eq_k1.subs(k2, eq_k2JointDet)

    k1Det_of_x = lambdify((x, k3, km3, km1), eq_k1JointDet)
    k2Det_of_x = lambdify((x, k3, km3, km1), eq_k2JointDet)

    xx = xarray[k1Det_of_x(xarray, k3val, km3val, km1val) > 0]

    plt.plot(k1Trace_of_x(xx, k3val, km3val, km1val), k2Trace_of_x(xx, k3val, km3val, km1val), color='r', linestyle=':', linewidth=2, label='neutral')
    plt.plot(k1Det_of_x(xx, k3val, km3val, km1val), k2Det_of_x(xx, k3val, km3val, km1val), color='b', linewidth=1, linestyle='-',
             label='multiplicity')
    plt.xlim([0.0, 1])
    plt.ylim([0.3, 10])
    plt.xlabel(r'$k_1$')
    plt.ylabel(r'$k_2$')
    plt.legend(loc=0)
    plt.show()
    return


def solveSystem(init, k1val, km1val, k3val, km3val, k2val, dt, iterations):
    f1 = lambdify((x, y, k1, km1, k3, km3, k2), eq1)
    f2 = lambdify((x, y, k3, km3), eq2)

    def rhs(xy, t):
        return [f1(xy[0], xy[1], k1val, km1val, k3val, km3val, k2val), f2(xy[0], xy[1], k3val, km3val)]

    times = np.arange(iterations) * dt
    return odeint(rhs, init, times), times


def streamplot(k1val, km1val, k3val, km3val, k2val):
    f1 = lambdify((x, y, k1, km1, k3, km3, k2), eq1)
    f2 = lambdify((x, y, k3, km3), eq2)

    Y, xarray = np.mgrid[0:.5:1000j, 0:1:2000j]
    U = f1(xarray, Y, k1val, km1val, k3val, km3val, k2val)
    V = f2(xarray, Y, k3val, km3val)
    velocity = np.sqrt(U * U + V * V)
    plt.streamplot(xarray, Y, U, V, density=[2.5, 1.8], color=velocity)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


km1_list = [0.001, 0.005, 0.01, 0.015, 0.02]
km3_list = [0.0005, 0.001, 0.002, 0.003, 0.004]

if (False):
    # one parameter analysis

    for e in km1_list:
        dependenceOnParameterK1(e, 0.002)

    for e in km3_list:
        dependenceOnParameterK1(0.01, e)

# авто-колебания
twoParameterAnalysisK1K2(km1_val, k3_val, km3_list[1])
# sys.exit()
streamplot(k1_val, km3_val, k3_val, km3_val, k2_val)

res, times = solveSystem([0.1, 0.5], k1_val, km1_val, k3_val, km3_list[1], k2_val, 1e-2, 1e6)
ax = plt.subplot(211)
plt.plot(times, res[:, 0], color='r',)
plt.setp(ax.get_xticklabels(), visible=False)
plt.ylabel('x')
plt.grid()
ax1 = plt.subplot(212, sharex=ax)
plt.plot(times, res[:, 1], color='b')
plt.xlabel('t')
plt.ylabel('y')
plt.grid()
plt.show()
