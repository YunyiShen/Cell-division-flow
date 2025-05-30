import numpy as np
import matplotlib.pyplot as plt
from fipy import (
    Grid2D,
    CellVariable,
    TransientTerm,
    DiffusionTerm,
    ConvectionTerm,
    LinearLUSolver,
)
from fipy import FaceVariable
from fipy.tools import numerix
from tqdm import tqdm

# -------------------------
# Parameters
# -------------------------
L = 2.0       # Domain size
N = 100       # Number of cells per side
dx = L / N
mu = 0.01     # Viscosity
rho = 1.0     # Density
dt = 0.01    # Time step
steps = 50  # Number of time steps

# -------------------------
# Mesh & Geometry
# -------------------------
mesh = Grid2D(dx=dx, dy=dx, nx=N, ny=N)
x, y = mesh.cellCenters
center = L / 2
radius = 1
inside = ((x - center)**2 + (y - center)**2) < radius**2

# -------------------------
# Variables
# -------------------------
u = CellVariable(name="u", mesh=mesh, value=0.0)
v = CellVariable(name="v", mesh=mesh, value=0.0)
p = CellVariable(name="p", mesh=mesh, value=0.0)

# -------------------------
# Initial pressure bump
# -------------------------
sigma = 0.04
A = 20
p_ext = CellVariable(mesh=mesh, value=0.0)
p_ext.value = A * np.exp(-(((x - center)/4)**2 + (y - center)**2) / (2 * sigma**2))


# -------------------------
# Boundary conditions (no-slip)
# -------------------------
for var in [u, v]:
    var.constrain(0.0, mesh.exteriorFaces)

# -------------------------
# Solver
# -------------------------
solver = LinearLUSolver()

# -------------------------
# Time-stepping loop
# -------------------------
for step in tqdm(range(steps)):
    velocity = FaceVariable(name="velocity", mesh=mesh, rank=1)
    velocity[:] = numerix.array([u.arithmeticFaceValue, v.arithmeticFaceValue])
    
    # Add pressure gradient as explicit source term in momentum eq
    u_star_eq = (
      TransientTerm(var=u)
      == DiffusionTerm(coeff=mu / rho, var=u)
      - ConvectionTerm(coeff=velocity, var=u)
      - (1.0 / rho) * (p_ext.grad[0])
    )
    v_star_eq = (
      TransientTerm(var=v)
      == DiffusionTerm(coeff=mu / rho, var=v)
      - ConvectionTerm(coeff=velocity, var=v)
      - (1.0 / rho) * (p_ext.grad[1])
    )
    u_star_eq.solve(dt=dt, solver=solver)
    v_star_eq.solve(dt=dt, solver=solver)

    # Solve pressure Poisson eq to enforce incompressibility
    velocity[:] = numerix.array([u.arithmeticFaceValue, v.arithmeticFaceValue])
    div_u_star = velocity.divergence
    pressure_eq = DiffusionTerm(var=p) == (rho / dt) * div_u_star
    pressure_eq.solve(var=p, solver=solver)

    # Correct velocity to be divergence-free
    u.value[:] -= dt / rho * p.grad[0]
    v.value[:] -= dt / rho * p.grad[1]

    # Apply mask only after correction (inside domain)
    u.setValue(0.0, where=~inside)
    v.setValue(0.0, where=~inside)
    p.setValue(0.0, where=~inside)

    #if step % 20 == 0:
    #    max_vel = np.max(np.sqrt(u.value**2 + v.value**2))
    #    print(f"Step {step}: max |u| = {max_vel:.4f}")


