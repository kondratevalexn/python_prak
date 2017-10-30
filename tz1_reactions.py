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


km1_list = [0.001, 0.005, 0.01, 0.015, 0.02]
km3_list = [0.0005, 0.001, 0.002, 0.003, 0.004]


def z_f(x, km3, k3):
    return (1 - x) * km3 / (km3 + 2 * k3 * x)


def k2_f(x, z, k1, km1):
    if z == 0 or x == 0:
        return 0
    return (k1 * z - km1 * x) / z / z / x


eq1 = k1 * (1 - x - 2 * y) - km1 * x - k3 * x * (1 - x - 2 * y) + km3 * y - k2 * (1 - x - 2 * y) ** 2 * x
eq2 = k3 * x * (1 - x - 2 * y) - km3 * y

solution = solve([eq1, eq2], k1, x)
print(solution)
eq_k1 = solution[0][0]
eq_x = solution[0][1]
# print(eq_k2)
# print(eq_x)

A = Matrix([eq1, eq2])
var_vector = Matrix([x, y])
jacA = A.jacobian(var_vector)
detA = jacA.det()
traceA = jacA.trace()
DA = traceA ** 2 - 4 * detA

N = 10
yarray = np.linspace(0, 1, N)



# k1 = 0.12, km1 = 0.01, k3 = 0.0032, k2 = 0.95, km3 = 0.002 - базовый
def dependenceOnParameterK1(km1_new, km3_new):
    func_k1 = lambdify(y, eq_k1.subs(km1, km1_new).subs(k2, 0.95).subs(km3, km3_new).subs(k3, 0.0032))
    func_x = lambdify(y, eq_x.subs(km1, km1_new).subs(k2, 0.95).subs(km3, km3_new).subs(k3, 0.0032))
    func_detA = lambdify(y, detA.subs(x, eq_x).subs(k1, eq_k1).subs(km1, km1_new).subs(k2, 0.95).subs(km3, km3_new).subs(k3, 0.0032))
    func_DA = lambdify(y, DA.subs(x, eq_x).subs(k1, eq_k1).subs(km1, km1_new).subs(k2, 0.95).subs(km3, km3_new).subs(k3, 0.0032))
    k1_array = np.zeros(N)
    for i in range(N):
        print(func_x(yarray[i]))
    plt.plot(k1_array, yarray, color='b', label="$y_{k_1}$")
    plt.plot(func_k1(yarray), func_x(yarray), color='g', label="$x_{k_2}$")
    DAoneparam = list(func_DA(yarray))
    detAoneparam = list(func_detA(yarray))
    DAarray = []
    sarray = []
    for i in range(1, len(yarray)):
        if detAoneparam[i] * detAoneparam[i - 1] <= 0:
            sarray = sarray + [yarray[i]]
        if DAoneparam[i] * DAoneparam[i - 1] <= 0:
            DAarray = DAarray + [yarray[i]]
    DAarray = np.array(DAarray)
    sarray = np.array(sarray)
    ymax = 1.075 * max(sarray)
    ymin = min(sarray) - 0.15 * (max(sarray) - min(sarray))
    k2max = 1.075 * max(func_k1(sarray))
    x_or_y_max = 1.06 * max(max(func_x(sarray)), ymax)
    sn = func_k1(sarray)
    plt.plot(func_k1(sarray), sarray, color='b', linestyle='', marker='^')
    plt.plot(func_k1(sarray), func_x(sarray), color='g', linestyle='', marker='o')
    plt.title('Dependence on Parameter $k_2$ for $k_1=1$, $k_{-1}=$' + str(km1_new) + ', $k_{-2}=$' + str(km3_new) + ', $k_3=10$')
    plt.xlabel('$k_2$')
    plt.ylabel("x,y")
    plt.xlim([0.0, k2max])
    plt.ylim([ymin, x_or_y_max])
    plt.legend(loc=2)
    plt.show()
    return sn

for e in km1_list:
    dependenceOnParameterK1(e, 0.002)

for e in km3_list:
    dependenceOnParameterK1(0.01, e)

#x_array = solve(eq_x, )

sys.exit()
z = z_f(y_array, km3, k3)
k2 = np.zeros(N)
for i in range(N):
    k2[i] = k2_f(y_array[i], z[i], k1, km1)
