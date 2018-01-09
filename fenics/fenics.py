from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri
from math import pi,sinh,cosh
from mshr import *
from dolfin import *


# Create mesh and define function space
tol = 1e-14
box = Rectangle(Point(0, 0), Point(pi,pi))
mesh = generate_mesh(box, 128)
V = FunctionSpace(mesh, 'P', 1)


# Define boundary condition
u_L = Constant(0.0)
u_R = Expression('cos(2*x[1])', degree=1)

g_L = Expression('sin(2*x[0])', degree=1)
g_R = Expression('sin(3*x[0])', degree=1)

def boundary_L(x, on_boundary):
    return on_boundary and near(x[0], 0, tol)
def boundary_R(x, on_boundary):
    return on_boundary and near(x[0], pi, tol)

class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return  on_boundary and near(x[1], 0, tol)

class Top(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], pi, tol)

bc_L = DirichletBC(V, u_L, boundary_L)
bc_R = DirichletBC(V, u_R, boundary_R)
bcs = [bc_L, bc_R]

top = Top()
bottom = Bottom()

boundaries = FacetFunction("size_t", mesh)
boundaries.set_all(0)


top.mark(boundaries, 1)
bottom.mark(boundaries, 2)

ds = Measure('ds', domain=mesh, subdomain_data=boundaries)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0.0)
a = dot(grad(u), grad(v))*dx
L = -g_L*v*ds(2) + g_R*v*ds(1)
# Compute solution
u = Function(V)
solve(a == L, u, bcs)

# Plot solution and mesh fenics
#plot(u)
#plot(mesh)


u_solve = Expression('sinh(2*x[0])*cos(2*x[1])/sinh(2*pi) + cosh(2*x[1] - 2*pi)*sin(2*x[0])/(2*sinh(-2*pi)) + cosh(3*x[1])*sin(3*x[0])/(3*sinh(3*pi))', degree=2)

# Compute error in L2 norm
error_L2 = errornorm(u_solve, u, 'L2')
# Compute maximum error at vertices
vertex_values_u_solve = u_solve.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
error_max = np.max(np.abs(vertex_values_u_solve - vertex_values_u))
# Print errors
print('error_L2 =', error_L2)
print('error_max =', error_max)

# Plot solution and mesh matplotlib
n = mesh.num_vertices()
d = mesh.geometry().dim()
mesh_coordinates = mesh.coordinates().reshape((n, d))
triangles = np.asarray([cell.entities(0) for cell in cells(mesh)])
triangulation = tri.Triangulation(mesh_coordinates[:, 0],
mesh_coordinates[:, 1], triangles)
plt.figure(1)
plt.title("Analytical solution")
zfaces = np.asarray([u_solve(cell.midpoint()) for cell in cells(mesh)])
plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')

plt.figure(2)
plt.title("Numerical solution")
zfaces = np.asarray([u(cell.midpoint()) for cell in cells(mesh)])
plt.tripcolor(triangulation, facecolors=zfaces, edgecolors='k')
plt.show()