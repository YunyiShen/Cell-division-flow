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
import matplotlib.gridspec as gridspec
import copy


def rotate_precision_matrix(Lambda, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return R.T @ Lambda @ R



class cellBoundary2D():
    def __init__(self, center, radius = 1):
        self.radius = radius
        self.center = center

    def inside(self, x, y):
        return ((x - self.center[0])**2 + (y - self.center[1])**2) < self.radius**2

class gaussianbump2D():
    def __init__(self, center = None, precision = np.array([[25, 0],[0, 25**2]]), 
                maxp = 50, theta = 0):
        self.center = center
        precision = rotate_precision_matrix(precision, theta)
        self.precision = precision
        self.maxp = maxp
    
    def __call__(self, x, y, t=None):
        assert self.center is not None, "center need to be set"
        dx = x - self.center[0]
        dy = y - self.center[1]
        xy = np.stack([dx, dy], axis=0)  # shape (2, ...)
        quadform = np.einsum("i...,ij,j...->...", xy, self.precision, xy)
        return self.maxp - self.maxp * np.exp(-0.5 * quadform)


class growinggaussianbump2D():
    def __init__(self, center = None, precision = np.array([[25, 0],[0, 25**2]]), 
                 maxp = 50, timescale = 1., theta = 0):
        self.center = center
        precision = rotate_precision_matrix(precision, theta)
        #breakpoint()
        self.precision = precision
        self.maxp = maxp
        self.timescale = timescale
    
    def __call__(self, x, y, t):
        assert self.center is not None, "center need to be set"
        dx = x - self.center[0]
        dy = y - self.center[1]
        xy = np.stack([dx, dy], axis=0)  # shape (2, ...)
        quadform = np.einsum("i...,ij,j...->...", xy, self.precision, xy)
        return self.maxp - self.maxp * (1.-np.exp(-t/(self.timescale))) * np.exp(-0.5 * quadform)


class celldivflow2D():
    def __init__(self, domain_size = 2, N = 100, 
                cell_radius = 1.0, 
                mu = 0.01, rho = 1.0,
                stress_field = None
                ):
        self.L = domain_size
        L = domain_size
        self.N = N
        self.dx = L / N
        self.mu = mu
        self.rho = rho
        self.cellcenter = [L / 2, L / 2]
        self.cellradius = cell_radius
        self.thecell = cellBoundary2D(self.cellcenter, cell_radius)

        if stress_field is None:
            self.stress_field = growinggaussianbump2D(self.cellcenter)
        else:
            if stress_field.center is None:
                stress_field.center = self.cellcenter
            self.stress_field = stress_field
        
        self.mesh = Grid2D(dx=self.dx, dy=self.dx, nx=N, ny=N)

        self.saved = None
        
    
    def solve(self, dt, steps, save_every = 5):
        u = CellVariable(name="u", mesh=self.mesh, value=0.0)
        v = CellVariable(name="v", mesh=self.mesh, value=0.0)
        p = CellVariable(name="p", mesh=self.mesh, value=0.0)
        x, y = self.mesh.cellCenters
        for var in [u, v]:
            var.constrain(0.0, self.mesh.exteriorFaces)
        stress_ext = CellVariable(mesh=self.mesh, value=0.0)
        # -------------------------
        # Solver
        # -------------------------
        solver = LinearLUSolver()
        u_save = []
        v_save = []
        p_save = []
        stress_ext_save = []
        t_save = []
        #stress_ext.value = self.stress_field(x, y, 0)
        for step in tqdm(range(steps)):
            velocity = FaceVariable(name="velocity", mesh=self.mesh, rank=1)
            velocity[:] = numerix.array([u.arithmeticFaceValue, v.arithmeticFaceValue])
            #stress_ext = CellVariable(mesh=self.mesh, value=0.0)
            stress_ext.value = self.stress_field(x, y, step*dt)
            # Add pressure gradient as explicit source term in momentum eq
            u_star_eq = (
                TransientTerm(var=u)
                == DiffusionTerm(coeff=self.mu / self.rho, var=u)
                - ConvectionTerm(coeff=velocity, var=u)
                + (1.0 / self.rho) * (stress_ext.grad[0])
            )
            v_star_eq = (
                TransientTerm(var=v)
                == DiffusionTerm(coeff=self.mu / self.rho, var=v)
                - ConvectionTerm(coeff=velocity, var=v)
                + (1.0 / self.rho) * (stress_ext.grad[1])
            )
            u_star_eq.solve(dt=dt, solver=solver)
            v_star_eq.solve(dt=dt, solver=solver)

            # Solve pressure Poisson eq to enforce incompressibility
            velocity[:] = numerix.array([u.arithmeticFaceValue, v.arithmeticFaceValue])
            div_u_star = velocity.divergence
            pressure_eq = DiffusionTerm(var=p) == (self.rho / dt) * div_u_star
            pressure_eq.solve(var=p, solver=solver)

            # Correct velocity to be divergence-free
            u.value[:] -= dt / self.rho * p.grad[0]
            v.value[:] -= dt / self.rho * p.grad[1]

            # Apply mask only after correction (inside domain)
            inside = self.thecell.inside(x, y)
            u.setValue(0.0, where=~inside)
            v.setValue(0.0, where=~inside)
            p.setValue(0.0, where=~inside)

            #breakpoint()

            if step % save_every == 0:
                u_save.append(copy.deepcopy(u.value))
                v_save.append(copy.deepcopy(v.value))
                p_tmp = copy.deepcopy(p.value)
                p_tmp[np.logical_not(inside)] = np.nan
                p_save.append(p_tmp)
                stress_ext_tmp = copy.deepcopy(stress_ext.value)
                stress_ext_tmp[np.logical_not(inside)] = np.nan
                stress_ext_save.append(stress_ext_tmp)
                t_save.append(dt * step)
        self.saved = {"u": u_save, "v": v_save, "p": p_save, "stress_ext": stress_ext_save, 't': t_save}
        #breakpoint()
        return u_save, v_save, p_save, stress_ext_save, t_save, x, y, self.N
    

    def plot_vel_p_end(self, idx = -1, thinning=2, scale = 15):
        if self.saved is None:
            raise RuntimeError("Run solve at least once")
        assert np.abs(idx) < len(self.saved['t']), "index has to be within saved range"

        X = self.mesh.cellCenters[0].value
        Y = self.mesh.cellCenters[1].value

        p = self.saved['p'][idx]
        stress_ext = self.saved['stress_ext'][idx]
        u = self.saved['u'][idx]
        v = self.saved['v'][idx]
        inside = self.thecell.inside(X, Y)
        stress_ext[np.logical_not(inside)] = np.nan

        U, V = np.where(inside, u, np.nan), np.where(inside, v, np.nan)
        nx, ny = self.N, self.N  # grid dims
        X2 = X.reshape((nx, ny))
        Y2 = Y.reshape((nx, ny))
        U2 = U.reshape((nx, ny))
        V2 = V.reshape((nx, ny))
        stress_ext = stress_ext.reshape((nx, ny))
        p = p.reshape((nx, ny))

        step = thinning

        fig = plt.figure(figsize=(18, 5))
        gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 0.05, 1, 0.05])

        # Quiver plot
        if scale is None:
            scale = v.std() * 60
        ax0 = fig.add_subplot(gs[0])
        ax0.quiver(
            X2[::step, ::step], Y2[::step, ::step],
            U2[::step, ::step], V2[::step, ::step],
            scale=scale,
            width=0.005,
            color='black'
        )
        ax0.set_title("velocity field")
        ax0.set_aspect('equal')

        # Pressure p + stress_ext
        ax1 = fig.add_subplot(gs[1])
        im1 = ax1.contourf(X2, Y2, p - stress_ext, levels=50, cmap='viridis')
        ax1.set_title('Overall pressure p')
        ax1.set_aspect('equal')

        cax1 = fig.add_subplot(gs[2])
        fig.colorbar(im1, cax=cax1)

        # External pressure stress_ext
        ax2 = fig.add_subplot(gs[3])
        im2 = ax2.contourf(X2, Y2, stress_ext, levels=np.linspace(0, np.nanmax(self.saved['stress_ext']), 50), cmap='viridis')
        ax2.set_title('External stress bump')
        ax2.set_aspect('equal')

        cax2 = fig.add_subplot(gs[4])
        fig.colorbar(im2, cax=cax2)

        fig.tight_layout()
        return fig, (ax0, ax1, ax2)
