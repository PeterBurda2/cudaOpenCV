import gmsh
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import ufl
import dolfinx
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
# from dolfinx.io import gmshio, XDMFFile
from dolfinx.io import XDMFFile
from dolfinx.io.gmsh import model_to_mesh, read_from_msh

comm = MPI.COMM_WORLD
data_out = read_from_msh("mesh.msh", comm, rank=0, gdim=3)
domain, cell_tags, facet_tags, *_ = data_out

#### BCs facets ID
DIRICHLET_ID = 11
ROBIN_ID = 12

OMEGA_ID = 1

heated_facets = facet_tags.find(DIRICHLET_ID)
newton_cooling_facets = facet_tags.find(ROBIN_ID)

fdim = domain.topology.dim - 1

# Parameters
k = fem.Constant(domain, PETSc.ScalarType(400.0))          # W/(m K)
h = fem.Constant(domain, PETSc.ScalarType(100.0))           # W/(m^2 K)  (choose your test value)
T_inf = fem.Constant(domain, PETSc.ScalarType(200.15))     # K
T_heat = PETSc.ScalarType(500.0) 


V = fem.functionspace(domain, ("CG", 2))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

heated_dofs = fem.locate_dofs_topological(V, fdim, heated_facets)

bc = fem.dirichletbc(T_heat, heated_dofs, V)

a = ufl.inner(k * ufl.grad(u), ufl.grad(v)) * ufl.dx + h * u * v * ds(ROBIN_ID)
L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx + h * T_inf * v * ds(ROBIN_ID)

problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options_prefix="Temperature_Laplace",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu", "ksp_error_if_not_converged": True},
)

Th = problem.solve()
# Th.name = "Temperature"

V1 = fem.functionspace(domain, ("CG", 1))
T1 = fem.Function(V1)
T1.interpolate(Th)   # T2 is your CG2 solution
T1.name = "Temperature"


vals = Th.x.array[heated_dofs]
print("max |T - T_heat| on heated dofs =", np.max(np.abs(vals - float(T_heat))))

with XDMFFile(comm, "steady_cube_robin.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(T1)

# print(f"heated_facets: {heated_facets}, \n newton_cooling_facets: {newton_cooling_facets}")



