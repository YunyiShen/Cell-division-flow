import numpy as np
import matplotlib.pyplot as plt
from fipy import (
    Grid3D,
    CellVariable,
    TransientTerm,
    DiffusionTerm,
    ConvectionTerm,
    LinearLUSolver,
)
from fipy import FaceVariable
from fipy.tools import numerix
from tqdm import tqdm
import matplotlib.gridspec as gridspec
import copy


class cellBoundary3D():
    def __init__(self, center, radius = 1.):
        self.radius = radius
        self.center = center

    def inside(self, x, y, z):
        return ((x - self.center[0])**2 + (y - self.center[1])**2 + (z-self.center[2])**2) < self.radius**2

class growinggaussianbump3D():
    def __init__(self, center = None, precision = np.array([[25, 0, 0], [0, 25, 0],[0, 0, 25**2]]), 
                 maxp = 20, timescale = 1.):
        self.center = center
        self.precision = precision
        self.maxp = maxp
        self.timescale = timescale
    
    def __call__(self, x, y, z, t):
        assert self.center is not None, "center need to be set"
        dx = x - self.center[0]
        dy = y - self.center[1]
        dz = z - self.center[2]
        xy = np.stack([dx, dy, dz], axis=0)  # shape (2, ...)
        quadform = np.einsum("i...,ij,j...->...", xy, self.precision, xy)
        return self.maxp * (1.-np.exp(-t/(self.timescale))) * np.exp(-0.5 * quadform)


class celldivflow3D():
    def __init__(self, domain_size = 2, N = 100, 
                cell_radius = 1.0, 
                mu = 0.01, rho = 1.0,
                pressure_field = None
                ):
        self.L = domain_size
        L = domain_size
        self.N = N
        self.dx = L / N
        self.mu = mu
        self.rho = rho
        self.cellcenter = [L / 2, L / 2, L/2]
        self.cellradius = cell_radius
        self.thecell = cellBoundary3D(self.cellcenter, cell_radius)

        if pressure_field is None:
            self.pressure_field = growinggaussianbump3D(self.cellcenter)
        else:
            if pressure_field.center is None:
                pressure_field.center = self.cellcenter
            self.pressure_field = pressure_field
        
        self.mesh = Grid3D(dx=self.dx, dy=self.dx, dz=self.dx, nx=N, ny=N, nz=N)# Grid2D(dx=self.dx, dy=self.dx, nx=N, ny=N)

        self.saved = None
        
    
    def solve(self, dt, steps, save_every = 5):
        u = CellVariable(name="u", mesh=self.mesh, value=0.0)
        v = CellVariable(name="v", mesh=self.mesh, value=0.0)
        w = CellVariable(name="w", mesh=self.mesh, value=0.0)
        p = CellVariable(name="p", mesh=self.mesh, value=0.0)
        x, y, z = self.mesh.cellCenters
        for var in [u, v, w]:
            var.constrain(0.0, self.mesh.exteriorFaces)
        p_ext = CellVariable(mesh=self.mesh, value=0.0)
        # -------------------------
        # Solver
        # -------------------------
        solver = LinearLUSolver()
        u_save = []
        v_save = []
        w_save = []
        p_save = []
        p_ext_save = []
        t_save = []
        #p_ext.value = self.pressure_field(x, y, 0)
        for step in tqdm(range(steps)):
            velocity = FaceVariable(name="velocity", mesh=self.mesh, rank=1)
            velocity[:] = numerix.array([u.arithmeticFaceValue, 
                                         v.arithmeticFaceValue, 
                                         w.arithmeticFaceValue ])
            #p_ext = CellVariable(mesh=self.mesh, value=0.0)
            p_ext.value = self.pressure_field(x, y, z, step*dt)
            # Add pressure gradient as explicit source term in momentum eq
            u_star_eq = (
                TransientTerm(var=u)
                == DiffusionTerm(coeff=self.mu / self.rho, var=u)
                - ConvectionTerm(coeff=velocity, var=u)
                - (1.0 / self.rho) * (p_ext.grad[0])
            )
            v_star_eq = (
                TransientTerm(var=v)
                == DiffusionTerm(coeff=self.mu / self.rho, var=v)
                - ConvectionTerm(coeff=velocity, var=v)
                - (1.0 / self.rho) * (p_ext.grad[1])
            )
            w_star_eq = (
                TransientTerm(var=w)
                == DiffusionTerm(coeff=self.mu / self.rho, var=w)
                - ConvectionTerm(coeff=velocity, var=w)
                - (1.0 / self.rho) * (p_ext.grad[2])
            )
            u_star_eq.solve(dt=dt, solver=solver)
            v_star_eq.solve(dt=dt, solver=solver)
            w_star_eq.solve(dt=dt, solver=solver)

            # Solve pressure Poisson eq to enforce incompressibility
            velocity[:] = numerix.array([u.arithmeticFaceValue, 
                                        v.arithmeticFaceValue, 
                                        w.arithmeticFaceValue])
            
            div_u_star = velocity.divergence
            pressure_eq = DiffusionTerm(var=p) == (self.rho / dt) * div_u_star
            pressure_eq.solve(var=p, solver=solver)

            # Correct velocity to be divergence-free
            u.value[:] -= dt / self.rho * p.grad[0]
            v.value[:] -= dt / self.rho * p.grad[1]
            w.value[:] -= dt / self.rho * p.grad[2]

            # Apply mask only after correction (inside domain)
            inside = self.thecell.inside(x, y, z)
            u.setValue(0.0, where=~inside)
            v.setValue(0.0, where=~inside)
            w.setValue(0.0, where=~inside)
            p.setValue(0.0, where=~inside)

            #breakpoint()

            if step % save_every == 0:
                u_save.append(copy.deepcopy(u.value))
                v_save.append(copy.deepcopy(v.value))
                w_save.append(copy.deepcopy(w.value))
                p_tmp = copy.deepcopy(p.value)
                p_tmp[np.logical_not(inside)] = np.nan
                p_save.append(p_tmp)
                p_ext_tmp = copy.deepcopy(p_ext.value)
                p_ext_tmp[np.logical_not(inside)] = np.nan
                p_ext_save.append(p_ext_tmp)
                t_save.append(dt * step)
        self.saved = {"u": u_save, "v": v_save, "w": w_save,"p": p_save, "p_ext": p_ext_save, 't': t_save}
        #breakpoint()
        return u_save, v_save, w_save, p_save, p_ext_save, t_save, x, y, z, self.N
