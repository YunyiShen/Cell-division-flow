import fipy as fp
from fipy import CellVariable, FaceVariable, Grid2D, TransientTerm, DiffusionTerm, HybridConvectionTerm, ImplicitSourceTerm, LinearLUSolver
import fipy.tools.numerix as numerix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import copy


# some useful actin models 
def rotate_precision_matrix(Lambda, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return R.T @ Lambda @ R

class growinggaussianbump2Dconc():
    def __init__(self, precision = np.array([[100, 0],[0, 50**2]]), 
                 timescale = 1., theta = 0):
        precision = rotate_precision_matrix(precision, theta)
        #breakpoint()
        self.precision = precision
        
        self.timescale = timescale
        
    def __call__(self, x, y, t):
        #assert self.center is not None, "center need to be set"
        xy = np.stack([x, y], axis=0)  # shape (2, ...)
        quadform = np.einsum("i...,ij,j...->...", xy, self.precision, xy)
        conc = (1.-(1.-np.exp(-t/(self.timescale))) * np.exp(-0.5 * quadform)) 
        return conc


# =============================================================================
# 1. THE BIOLOGY CLASS (Constitutive Laws)
# =============================================================================
class ActinModel:
    """
    Defines the "Physics of the Material".
    Decoupled from the solver mesh.
    
    Physics:
    - Active Stress scales as A^n
    - Hydraulic Resistance scales as A^m
    """
    def __init__(self, 
                actin = None,
                stress_range = [1e-5, 1e4],
                drag_range = [1e6, 1e10],
                stress_power = 1.,
                drag_power = 2.,
                mu = 20.0, rho=1.0
                ):           # Min Cytosol Drag (Solvent floor)
        
        if actin is None:
            def actin(x, y, t): return 1.0
        
        self.actin = actin
        #self.boundary = boundary
        self.stress_range = stress_range
        self.drag_range = drag_range
        self.stress_power = stress_power
        self.drag_power = drag_power
        # Geometry Placeholders (Set by Solver)
        self.mu = mu
        self.rho = rho
        self.center = None
        self.domain_size = None

    def set_geometry(self, domain_size, center):
        """Called by the solver to sync the biological pattern to the mesh size"""
        self.center = center
        self.domain_size = domain_size

    def get_actin(self, x, y, t):
        """
        Calculates Actin Concentration A(x,y,t)
        Returns: Numpy array (0.0 to 1.0)
        """
        if self.center is None:
            raise ValueError("Geometry not set! Run solver.solve() first.")

        A = self.actin(x-self.center[0], y-self.center[1], t)
        
        # Safety clamp
        return np.maximum(A, 0.0001)

    def get_stress(self, x, y, t):
        """ Constitutive Law: Sigma ~ A^n """
        Apow = self.get_actin(x, y, t) ** self.stress_power
        return self.stress_range[1] * (Apow) + self.stress_range[0] * (1-Apow)

    def get_drag(self, x, y, t):
        """ Constitutive Law: Alpha ~ A^m """
        Apow = self.get_actin(x, y, t) ** self.drag_power
        # Interpolate between Cytosol Viscosity (Gap) and Cortex Drag (Bulk)
        return self.drag_range[0] + (self.drag_range[1] - self.drag_range[0]) * (Apow)


# =============================================================================
# 2. THE SOLVER CLASS (Navier-Stokes-Brinkman)
# =============================================================================
class CellDivFlow2D:
    '''
    Stable implementation of Navier-Stokes-Brinkman solver.
    '''
    def __init__(self, domain_size=1.0, N=100, cell_radius=0.5,
                 ):

        self.L = domain_size
        self.N = N
        self.dx = domain_size / N
        
        self.cellcenter = [self.L/2.0, self.L/2.0]
        self.cellradius = cell_radius
        
        # FiPy Mesh
        self.mesh = Grid2D(dx=self.dx, dy=self.dx, nx=N, ny=N)
        self.saved = None

        # --- Geometry Masks ---
        x, y = self.mesh.cellCenters
        x0, y0 = self.cellcenter
        rdist = numerix.sqrt((x - x0)**2 + (y - y0)**2)
        
        # 1. Smooth Transition Mask (Chi) for Numerical Stability
        # 0.0 = Inside Cell, 1.0 = Outside (Wall)
        epsilon = 1.5 * self.dx
        smooth_mask = 0.5 * (1 + numerix.tanh((rdist - self.cellradius) / epsilon))
        self.chi = CellVariable(mesh=self.mesh, value=smooth_mask)
        
        # 2. Binary Mask for Plotting (Hiding the outside)
        self.plotting_mask = (rdist >= self.cellradius)

    def solve(self, biology_model, dt, steps, 
              save_every=10, alpha_wall=1e14):
        """
        Main Loop.
        biology_model: Instance of ActinModel
        alpha_wall: Drag coefficient of the exterior wall (Must be >> biology.D_max)
        """
        
        # Sync Geometry
        biology_model.set_geometry(self.L, self.cellcenter)
        
        # Fields
        u = CellVariable(name="u", mesh=self.mesh, value=0.0)
        v = CellVariable(name="v", mesh=self.mesh, value=0.0)
        p = CellVariable(name="p", mesh=self.mesh, value=0.0)
        
        # Driven Fields
        stress_ext = CellVariable(mesh=self.mesh, value=0.0)
        total_drag = CellVariable(mesh=self.mesh, value=0.0)

        # Boundary Conditions (Box Walls)
        for var in [u, v]:
            var.constrain(0.0, self.mesh.exteriorFaces)
        
        # Pin Pressure (Gauge freedom)
        x, y = self.mesh.cellCenters
        p.constrain(0.0, where=(x == x.min()) & (y == y.min()))

        solver = LinearLUSolver()

        # Storage
        u_save, v_save, p_save, stress_save, drag_save, t_save = [], [], [], [], [], []

        print(f"Starting Simulation: {steps} steps, dt={dt}")
        
        for step in tqdm(range(steps)):
            t = step * dt
            
            # --- 1. UPDATE PHYSICS ---
            # Get raw constitutive values
            bio_stress = biology_model.get_stress(x, y, t)
            bio_drag   = biology_model.get_drag(x, y, t)
            
            # Apply Masks
            # Stress is zero outside the cell
            stress_ext.value = bio_stress * (1.0 - self.chi.value)
            
            # Drag is Bio (Inside) + Wall (Outside)
            total_drag.value = (bio_drag * (1.0 - self.chi.value)) + (alpha_wall * self.chi.value)

            # --- 2. PREDICTOR STEP (Momentum) ---
            # Explicit Velocity for linearization
            velocity = FaceVariable(mesh=self.mesh, rank=1)
            velocity[:] = numerix.array([u.arithmeticFaceValue, v.arithmeticFaceValue])
            
            # Eq: Inertia = Diffusion - Convection + StressGrad - Drag*u
            
            u_star_eq = (TransientTerm(var=u)
                         == DiffusionTerm(coeff=biology_model.mu/biology_model.rho, var=u)
                         - HybridConvectionTerm(coeff=velocity, var=u)
                         + (1.0/biology_model.rho) * stress_ext.grad[0] 
                         - ImplicitSourceTerm(coeff=total_drag, var=u))
            
            v_star_eq = (TransientTerm(var=v)
                         == DiffusionTerm(coeff=biology_model.mu/biology_model.rho, var=v)
                         - HybridConvectionTerm(coeff=velocity, var=v)
                         + (1.0/biology_model.rho) * stress_ext.grad[1] 
                         - ImplicitSourceTerm(coeff=total_drag, var=v))

            u_star_eq.solve(dt=dt, solver=solver)
            v_star_eq.solve(dt=dt, solver=solver)

            # --- 3. CORRECTOR STEP (Pressure Projection) ---
            # Div(u*)
            velocity[:] = numerix.array([u.arithmeticFaceValue, v.arithmeticFaceValue])
            div_u_star = velocity.divergence
            
            # Poisson Eq: Laplacian(p) = rho/dt * Div(u*)
            pressure_eq = DiffusionTerm(var=p) == (biology_model.rho/dt) * div_u_star
            pressure_eq.solve(var=p, solver=solver)

            # Update Velocity: u_new = u* - dt/rho * Grad(p)
            u.value[:] -= (dt/biology_model.rho) * p.grad[0]
            v.value[:] -= (dt/biology_model.rho) * p.grad[1]
            
            # Hard kill velocity inside the wall (prevents leakage)
            u.value[:] *= (1.0 - self.chi.value)
            v.value[:] *= (1.0 - self.chi.value)
            
            # --- 4. SAVE ---
            if step % save_every == 0:
                u_save.append(u.value.copy())
                v_save.append(v.value.copy())
                
                # Visual Masking for Pressure
                p_tmp = p.value.copy()
                p_tmp[self.plotting_mask] = np.nan 
                p_save.append(p_tmp)
                
                drag_tmp = total_drag.value.copy()
                drag_tmp[self.plotting_mask] = np.nan 
                drag_save.append(drag_tmp)
                
                stress_save.append(stress_ext.value.copy())
                t_save.append(t)

        self.saved = dict(u=u_save, v=v_save, p=p_save,
                          stress_ext=stress_save, drag = drag_save,
                          t=t_save)
        
        return self.saved # Return dict for external processing
    
    def plot_vel_p_end(self, idx=-1, thinning=2, scale=None):
        """
        Visualize the state at a specific time index.
        """
        if self.saved is None:
            raise RuntimeError("Run solve at least once")
        
        # Unpack
        X = self.mesh.cellCenters[0].value
        Y = self.mesh.cellCenters[1].value
        u = self.saved['u'][idx]
        v = self.saved['v'][idx]
        p = self.saved['p'][idx]
        s = self.saved['stress_ext'][idx]
        
        # Reshape for Matplotlib (N x N)
        nx, ny = self.N, self.N
        X2 = X.reshape((nx, ny))
        Y2 = Y.reshape((nx, ny))
        U2 = u.reshape((nx, ny))
        V2 = v.reshape((nx, ny))
        P2 = p.reshape((nx, ny))
        S2 = s.reshape((nx, ny))

        # Setup Plot
        fig = plt.figure(figsize=(18, 5))
        gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 0.05, 1, 0.05])

        # 1. Quiver (Velocity)
        if scale is None:
            scale = np.nanstd(v) * 50 # Auto-scale hint
            
        ax0 = fig.add_subplot(gs[0])
        ax0.quiver(
            X2[::thinning, ::thinning], Y2[::thinning, ::thinning],
            U2[::thinning, ::thinning], V2[::thinning, ::thinning],
            scale=scale, width=0.005, color='black'
        )
        ax0.set_title("Velocity Field")
        ax0.set_aspect('equal')
        ax0.set_xlim(0, self.L)
        ax0.set_ylim(0, self.L)

        # 2. Effective Pressure (P - Stress)
        # This shows the "Hydrostatic Drive"
        ax1 = fig.add_subplot(gs[1])
        im1 = ax1.contourf(X2, Y2, P2 - S2, levels=50, cmap='coolwarm')
        ax1.set_title('Effective Potential (P - Stress)')
        ax1.set_aspect('equal')
        cax1 = fig.add_subplot(gs[2])
        fig.colorbar(im1, cax=cax1)

        # 3. Active Stress
        ax2 = fig.add_subplot(gs[3])
        im2 = ax2.contourf(X2, Y2, S2, levels=50, cmap='viridis')
        ax2.set_title('Active Stress ($\Sigma$)')
        ax2.set_aspect('equal')
        cax2 = fig.add_subplot(gs[4])
        fig.colorbar(im2, cax=cax2)

        plt.tight_layout()
        return fig
