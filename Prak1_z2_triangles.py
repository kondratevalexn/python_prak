import numpy as np
import math

eps = 1e-12


# n_i are vectors in 3d
def vectorProduct(u, v):
    res = np.zeros(3)
    res[0] = u[1] * v[2] - u[2] * v[1]
    res[1] = - (u[0] * v[2] - u[2] * v[0])
    res[2] = u[0] * v[1] - u[1] * v[0]
    return res


# n_i are vectors
def scalarProduct(n1, n2):
    size = len(n1)
    res = 0
    for i in range(size):
        res += n1[i] * n2[i]
    return res


# vector length
def module(n):
    res = 0
    for i in range(len(n)):
        res += n[i] * n[i]
    res = math.sqrt(res)
    return res


def isInRange(v, min, max):
    return v >= min and v <= max


# get barycentric coordinates of point p in triangle abc
def getBarycentricCoordinates(a, b, c, p):
    area_x2 = module(vectorProduct(b - a, c - a))
    if area_x2 <= eps:
        return (-1, -1, -1)
    alpha = module(vectorProduct(b - p, b - c)) / area_x2
    beta = module(vectorProduct(c - p, a - p)) / area_x2
    gamma = 1 - alpha - beta
    return (alpha, beta, gamma)


def isPointInTriangle(a, b, c, p):
    area_x2 = module(vectorProduct(b - a, c - a))

    if area_x2 == 0:  # not necessary, lets assume the triangles are actually triangles
        if a == b and b == c and a == p:  # same points
            return True
        else:  # check if p belong to AB and AC
            if module(vectorProduct(b - a, p - a)) != 0 or module(vectorProduct(c - a, p - a)) != 0:  # p is not on the same line
                return False
            else:  # p is on the line ABC
                return True  # not correct, but checking further is a pain

    alpha, beta, gamma = getBarycentricCoordinates(a, b, c, p)
    if isInRange(alpha, 0, 1) and isInRange(beta, 0, 1) and isInRange(gamma, 0, 1):
        return True
    return False


# check if point p is on segment [a,b]
def isPointInSegment(a, b, p):
    if (module(vectorProduct(b - a, p - a)) >= 1e-12):  # p is not on line
        return False
    d = module(p - a) / module(b - a)
    sign = 1
    if scalarProduct(p - a, b - a) < 0:
        sign = -1
    return isInRange(sign * d, 0, 1)


def isSegmentsIntesecting3D(a1, b1, a2, b2):
    e = b1 - a1
    f = b2 - a2
    g = a2 - a1
    if module(g) <= eps:
        return True
    h = module(vectorProduct(f, g))
    k = module(vectorProduct(f, e))
    if math.fabs(h) <= eps or math.fabs(k) <= eps:
        return False
    l = 1.0 * h / k * e
    sign = 1
    if scalarProduct(vectorProduct(f, g), vectorProduct(f, e)) < 0:
        sign = -1
    m = a1 + sign * l
    return isPointInSegment(a2, b2, m) and isPointInSegment(a1, b1, m)


# return true only if a2b2c2 <= a1b1c1
def checkTriangleIsInside(a1, b1, c1, a2, b2, c2):
    res = True
    alpha, beta, gamma = getBarycentricCoordinates(a1, b1, c1, a2)
    res = res and isInRange(alpha, 0, 1) and isInRange(beta, 0, 1) and isInRange(gamma, 0, 1)
    alpha, beta, gamma = getBarycentricCoordinates(a1, b1, c1, b2)
    res = res and isInRange(alpha, 0, 1) and isInRange(beta, 0, 1) and isInRange(gamma, 0, 1)
    alpha, beta, gamma = getBarycentricCoordinates(a1, b1, c1, c2)
    res = res and isInRange(alpha, 0, 1) and isInRange(beta, 0, 1) and isInRange(gamma, 0, 1)
    return res


# a_i,b_i,c_i are 3-d point, i.e. a1 = [0,0,0]
def isIntersecting(a1, b1, c1, a2, b2, c2):
    ab1 = a1 - b1
    bc1 = b1 - c1  # n*b1 is d
    n1 = vectorProduct(ab1, bc1)  # normal vector of the first triangle

    ab2 = a2 - b2
    bc2 = b2 - c2
    n2 = vectorProduct(ab2, bc2)  # normal vector of the second triangle

    # find on which side from p2 is triangle1
    s1 = scalarProduct(n2, a1 - a2)
    s2 = scalarProduct(n2, b1 - a2)
    s3 = scalarProduct(n2, c1 - a2)

    # check if triangle1 intersects p2
    if s1 * s2 > 0 and s2 * s3 > 0 and s1 * s3 > 0:
        return False
    if s1 * s2 < 0 and s2 * s3 < 0 and s1 * s3 < 0:
        return False
    if math.fabs(s1 * s2) < eps and math.fabs(s2 * s3) < eps and math.fabs(s1 * s3) < eps:
        # 2d problem
        if (checkTriangleIsInside(a1, b1, c1, a2, b2, c2) or checkTriangleIsInside(a2, b2, c2, a1, b1, c1)):
            return True
        return isSegmentsIntesecting3D(a1, b1, a2, b2) or isSegmentsIntesecting3D(a1, b1, a2, c2) or isSegmentsIntesecting3D(a1, b1, b2,
                                                                                                                             c2) or isSegmentsIntesecting3D(
            a1, c1, a2, b2) or isSegmentsIntesecting3D(a1, c1, a2, c2) or isSegmentsIntesecting3D(a1, c1, b2, c2) or isSegmentsIntesecting3D(c1, b1,
                                                                                                                                             a2,
                                                                                                                                             b2) or isSegmentsIntesecting3D(
            c1, b1, a2, c2) or isSegmentsIntesecting3D(c1, b1, b2, c2)

    if s1 * s2 < 0:  # a1b1 intersects p2
        r = scalarProduct(a2 - a1, n2) / scalarProduct(b1 - a1, n2)
        point = a1 + r * (b1 - a1)  # point of intersection
        if isPointInTriangle(a2, b2, c2, point):
            return True
    if s2 * s3 < 0:  # c1b1 intersects p2
        r = scalarProduct(a2 - c1, n2) / scalarProduct(b1 - c1, n2)
        point = c1 + r * (b1 - c1)  # point of intersection
        if isPointInTriangle(a2, b2, c2, point):
            return True
    if s1 * s3 < 0:  # a1c1 intersects p2
        r = scalarProduct(a2 - a1, n2) / scalarProduct(c1 - a1, n2)
        point = a1 + r * (c1 - a1)  # point of intersection
        if isPointInTriangle(a2, b2, c2, point):
            return True
    # else no couple of points from triangle1 intersect p2 in point belonging to triangle2
    return False


a1 = np.array([1, 0, 0])
b1 = np.array([0, 1, 0])
c1 = np.array([0, 0, 1])

a2 = np.array([1, 1, 1])
b2 = np.array([0, 0, 0])
c2 = np.array([1, 1, 0])

print(isIntersecting(a1, b1, c1, a2, b2, c2))
