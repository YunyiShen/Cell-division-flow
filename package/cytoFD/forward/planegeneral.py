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
    def __init__(self, precision = np.array([[25, 0],[0, 25**2]]), 
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
import fipy as fp
from fipy import CellVariable, FaceVariable, Grid2D, DiffusionTerm, ImplicitSourceTerm, HybridConvectionTerm, LinearLUSolver, TransientTerm
import fipy.tools.numerix as numerix
import numpy as np
from tqdm import tqdm


class CellDivFlow2D:

    def __init__(self, domain_size=1.0, N=100, cell_radius=0.5):
        self.L = domain_size
        self.N = N
        self.dx = domain_size / N
        self.cellcenter = [self.L/2.0, self.L/2.0]
        self.cellradius = cell_radius
        
        # Mesh
        self.mesh = Grid2D(dx=self.dx, dy=self.dx, nx=N, ny=N)
        self.saved = None

        # --- Geometry Masks ---
        x, y = self.mesh.cellCenters
        x0, y0 = self.cellcenter
        rdist = numerix.sqrt((x - x0)**2 + (y - y0)**2)
        
        # Smooth Transition Mask (Chi)
        epsilon = 1.5 * self.dx
        smooth_mask = 0.5 * (1 + numerix.tanh((rdist - self.cellradius) / epsilon))
        self.chi = CellVariable(mesh=self.mesh, value=smooth_mask)
        
        # Binary Mask (For hard constraints/plotting)
        self.wall_mask = (rdist >= self.cellradius)
        self.plotting_mask = self.wall_mask
        
        # Variables
        

    def solve(self, biology_model, dt, steps, alpha_wall=1e4, save_every = 10):
        x, y = self.mesh.cellCenters
        # Sync Geometry
        biology_model.set_geometry(self.L, self.cellcenter)
        # 1. Setup
        u = CellVariable(mesh=self.mesh, name="u", value=0.)
        v = CellVariable(mesh=self.mesh, name="v", value=0.)
        p = CellVariable(mesh=self.mesh, name="p", value=0.)
        for var in [u, v]:
            var.constrain(0.0, self.mesh.exteriorFaces)

        
        # Fields
        total_drag = CellVariable(mesh=self.mesh, value=0.0)
        stress_ext = CellVariable(mesh=self.mesh, value=0.0)
        
        solver = LinearLUSolver()
        u_save, v_save, p_save, stress_save, drag_save, t_save = [], [], [], [], [], []

        for step in tqdm(range(steps)):
            t = step * dt
            velocity = FaceVariable(mesh=self.mesh, rank=1)
            velocity[:] = numerix.array([u.arithmeticFaceValue, v.arithmeticFaceValue])
            # --- A. PHYSICS UPDATE ---
            # 1. Stress
            raw_stress = biology_model.get_stress(self.mesh.x, self.mesh.y, t)
            stress_ext.value = raw_stress * (1.0 - self.chi.value)
            
            # 2. Drag 
            # Bio (Inside) + Wall (Outside)
            # alpha_wall = 1e4 is PLENTY. Do not use 1e9.
            bio_drag = biology_model.get_drag(self.mesh.x, self.mesh.y, t)
            total_drag.value = bio_drag * (1.0 - self.chi.value) + alpha_wall * self.chi.value
            #breakpoint()
            # --- B. PREDICTOR STEP (Implicit Drag) ---
            # We use the 'face_velocity' from the PREVIOUS step to advect.
            # This is much more stable than interpolating u/v every time.
            
            u_star_eq = (TransientTerm(var=u) + ImplicitSourceTerm(coeff=total_drag/ biology_model.rho, var=u)
                         == DiffusionTerm(coeff=biology_model.mu/biology_model.rho, var=u)
                         - HybridConvectionTerm(coeff=velocity, var=u)
                         + (1.0/biology_model.rho) * stress_ext.grad[0] 
                         ) # Implicit Drag
            
            v_star_eq = (TransientTerm(var=v) + ImplicitSourceTerm(coeff=total_drag/ biology_model.rho, var=v)
                         == DiffusionTerm(coeff=biology_model.mu/biology_model.rho, var=v)
                         - HybridConvectionTerm(coeff=velocity, var=v)
                         + (1.0/biology_model.rho) * stress_ext.grad[1] 
                         ) # Implicit Drag

            u_star_eq.solve(dt=dt, solver=solver)
            v_star_eq.solve(dt=dt, solver=solver)

            # --- C. SIMPLIFIED PRESSURE PROJECTION ---
            velocity[:] = numerix.array([u.arithmeticFaceValue, v.arithmeticFaceValue])
            div_u_star = velocity.divergence
            
            
            pressure_eq = DiffusionTerm(var=p) == (biology_model.rho/dt) * div_u_star
            pressure_eq.solve(var=p, solver=solver)

            # --- VELOCITY CORRECTION ---
            u.value[:] -= (dt/biology_model.rho) * p.grad[0]
            v.value[:] -= (dt/biology_model.rho) * p.grad[1]

            
            # soft 0 on boundary
            u.value[:] *= (1.-self.chi)
            v.value[:] *= (1.-self.chi)
            
                
            
            # --- SAVE ---
            if step % save_every == 0:
                u_save.append(u.value.copy())
                v_save.append(v.value.copy())
                p_tmp = p.value.copy()
                p_tmp[self.plotting_mask] = np.nan 
                p_save.append(p_tmp)
                drag_tmp = total_drag.value.copy()
                drag_tmp[self.plotting_mask] = np.nan 
                drag_save.append(drag_tmp)
                stress_save.append(stress_ext.value.copy())
                t_save.append(t)

        self.saved = dict(u=u_save, v=v_save, p=p_save,
                          stress_ext=stress_save, drag=drag_save,
                          t=t_save, x=x, y=y, steps=steps)
        
        return self.saved
    
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
        im1 = ax1.contourf(X2, Y2, P2 - S2, levels=50, cmap='viridis')
        ax1.set_title('Effective Potential (P - Stress)')
        ax1.set_aspect('equal')
        cax1 = fig.add_subplot(gs[2])
        fig.colorbar(im1, cax=cax1)

        # 3. Active Stress
        ax2 = fig.add_subplot(gs[3])
        im2 = ax2.contourf(X2, Y2, S2, levels=50, cmap='viridis')
        ax2.set_title('Active Stress')
        ax2.set_aspect('equal')
        cax2 = fig.add_subplot(gs[4])
        fig.colorbar(im2, cax=cax2)

        plt.tight_layout()
        return fig, (ax0, ax1, ax2)

class CellDivFlow2DPISO:
    '''
    Stable PISO implementation for Transient Navier-Stokes-Brinkman.
    Iterates to enforce divergence-free flow without large matrix crashes.
    '''
    def __init__(self, domain_size=1.0, N=100, cell_radius=0.5):
        self.L = domain_size
        self.N = N
        self.dx = domain_size / N
        self.cellcenter = [self.L/2.0, self.L/2.0]
        self.cellradius = cell_radius
        
        # Mesh
        self.mesh = Grid2D(dx=self.dx, dy=self.dx, nx=N, ny=N)
        self.saved = None

        # --- Geometry Masks ---
        x, y = self.mesh.cellCenters
        x0, y0 = self.cellcenter
        rdist = numerix.sqrt((x - x0)**2 + (y - y0)**2)
        
        # Smooth Transition Mask (Chi)
        epsilon = 1.5 * self.dx
        smooth_mask = 0.5 * (1 + numerix.tanh((rdist - self.cellradius) / epsilon))
        self.chi = CellVariable(mesh=self.mesh, value=smooth_mask)
        
        # Binary Mask (For hard constraints/plotting)
        self.wall_mask = (rdist >= self.cellradius)
        self.plotting_mask = self.wall_mask
        
        # Variables
        self.u = CellVariable(mesh=self.mesh, name="u", value=0., hasOld=True)
        self.v = CellVariable(mesh=self.mesh, name="v", value=0., hasOld=True)
        self.p = CellVariable(mesh=self.mesh, name="p", value=0., hasOld=True)

    def solve(self, biology_model, dt, steps, 
              save_every=10, alpha_wall=1e9, sweeps=5):
        """
        sweeps: Number of PISO iterations per time step (3-5 is usually enough)
        """
        x, y = self.mesh.cellCenters
        # Sync Geometry
        biology_model.set_geometry(self.L, self.cellcenter)
        
        # Fields
        total_drag = CellVariable(mesh=self.mesh, value=0.0)
        stress_x   = CellVariable(mesh=self.mesh, value=0.0)
        stress_y   = CellVariable(mesh=self.mesh, value=0.0)
        
        # Helper Variables
        velocity_vector = FaceVariable(mesh=self.mesh, rank=1)
        p_corr = CellVariable(mesh=self.mesh, value=0.0)
        
        solver = LinearLUSolver()

        # Storage
        u_save, v_save, p_save, stress_save, drag_save, t_save = [], [], [], [], [], []

        print(f"Starting PISO Simulation: {steps} steps, dt={dt}, wall={alpha_wall:.0e}")
        
        for step in tqdm(range(steps)):
            t = step * dt
            
            # A. Update Old Values (Inertia)
            self.u.updateOld()
            self.v.updateOld()
            
            # B. Update Physics
            bio_drag_val = biology_model.get_drag(self.mesh.x, self.mesh.y, t)
            
            # Blend Bio Drag (Inside) and Wall Drag (Outside)
            # Safe to use 1e9 here because matrices are separate
            total_drag.value = bio_drag_val * (1.0 - self.chi.value) + alpha_wall * self.chi.value
            
            # Stress (Bio Only)
            s_val = biology_model.get_stress(self.mesh.x, self.mesh.y, t) * (1.0 - self.chi.value)
            s_var = CellVariable(mesh=self.mesh, value=s_val)
            stress_x.value = s_var.grad[0]
            stress_y.value = s_var.grad[1]
            
            # --- PISO INNER LOOP ---
            # This iteration fixes the "Wall Leakage" issue
            for sweep in range(sweeps):
                
                # 1. Update Convection Guess
                velocity_vector[:] = numerix.array([self.u.faceValue, self.v.faceValue])
                
                # 2. MOMENTUM PREDICTOR (Solve for u*, v*)
                # Implicit Drag on LHS ensures stability
                # We include grad(p) from previous sweep/step
                
                eq_u = (TransientTerm(var=self.u) 
                        + ImplicitSourceTerm(coeff=total_drag, var=self.u)
                        == DiffusionTerm(coeff=biology_model.mu/biology_model.rho, var=self.u)
                        - HybridConvectionTerm(coeff=velocity_vector, var=self.u)
                        + (1.0/biology_model.rho) * stress_x
                        - (1.0/biology_model.rho) * self.p.grad[0])
                
                eq_v = (TransientTerm(var=self.v) 
                        + ImplicitSourceTerm(coeff=total_drag, var=self.v)
                        == DiffusionTerm(coeff=biology_model.mu/biology_model.rho, var=self.v)
                        - HybridConvectionTerm(coeff=velocity_vector, var=self.v)
                        + (1.0/biology_model.rho) * stress_y
                        - (1.0/biology_model.rho) * self.p.grad[1])
                
                eq_u.solve(dt=dt, solver=solver)
                eq_v.solve(dt=dt, solver=solver)
                
                # 3. PRESSURE CORRECTOR
                # Calculate Variable Mobility (Beta)
                # Small at wall, Large in gap
                beta = dt / (biology_model.rho + total_drag.value * dt)
                beta_coeff = CellVariable(mesh=self.mesh, value=beta)
                
                # Calculate Divergence Error
                velocity_vector[:] = numerix.array([self.u.faceValue, self.v.faceValue])
                div_u = velocity_vector.divergence
                
                # Solve Poisson: div(beta * grad(P_prime)) = div(u*)
                p_corr.value = 0.0
                p_eq = (DiffusionTerm(coeff=beta_coeff, var=p_corr) == div_u)
                p_eq.solve(var=p_corr, solver=solver)
                
                # 4. UPDATE FIELDS
                # Correct Pressure
                self.p.value[:] += p_corr.value
                self.p.value[:] -= self.p.value.mean()
                
                # Correct Velocity
                self.u.value[:] -= beta * p_corr.grad[0]
                self.v.value[:] -= beta * p_corr.grad[1]
                
                # Safety Clamp: Explicitly kill wall velocity inside the loop
                # This helps the pressure solver "learn" the boundary faster
                self.u.value[self.wall_mask] = 0.0
                self.v.value[self.wall_mask] = 0.0
            
            # --- SAVE ---
            if step % save_every == 0:
                u_save.append(self.u.value.copy())
                v_save.append(self.v.value.copy())
                p_tmp = self.p.value.copy()
                p_tmp[self.plotting_mask] = np.nan 
                p_save.append(p_tmp)
                drag_tmp = total_drag.value.copy()
                drag_tmp[self.plotting_mask] = np.nan 
                drag_save.append(drag_tmp)
                stress_save.append(s_val.copy())
                t_save.append(t)

        self.saved = dict(u=u_save, v=v_save, p=p_save,
                          stress_ext=stress_save, drag=drag_save,
                          t=t_save, x=x, y=y, steps=steps)
        
        return self.saved
    
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
        im1 = ax1.contourf(X2, Y2, P2 - S2, levels=50, cmap='viridis')
        ax1.set_title('Effective Potential (P - Stress)')
        ax1.set_aspect('equal')
        cax1 = fig.add_subplot(gs[2])
        fig.colorbar(im1, cax=cax1)

        # 3. Active Stress
        ax2 = fig.add_subplot(gs[3])
        im2 = ax2.contourf(X2, Y2, S2, levels=50, cmap='viridis')
        ax2.set_title('Active Stress')
        ax2.set_aspect('equal')
        cax2 = fig.add_subplot(gs[4])
        fig.colorbar(im2, cax=cax2)

        plt.tight_layout()
        return fig, (ax0, ax1, ax2)



class QuasiStaticSolver:
    def __init__(self, domain_size=2.0, N=60, mu=6000.0):
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
        # (Update chi based on rdist as before for smooth drag blending)

    def solve(self, bio, total_time, dt_bio):
        steps = int(total_time / dt_bio)
        
        # --- DEFINE EQUATIONS ONCE ---
        # We use a large "Pseudo-Transient" coefficient for the pressure 
        # to help the coupled solver, or solve directly if using LinearLUSolver.
        
        # Coefficients (Will be updated in loop)
        drag_coeff = CellVariable(mesh=self.mesh, value=1.0)
        stress_x   = CellVariable(mesh=self.mesh, value=0.0)
        stress_y   = CellVariable(mesh=self.mesh, value=0.0)

        # 1. Momentum X
        # Viscosity + Drag + PressureGrad = Stress
        eq_u = (DiffusionTerm(coeff=biology_model.mu, var=self.u) 
                - ImplicitSourceTerm(coeff=drag_coeff, var=self.u) 
                - self.p.grad[0] 
                + stress_x == 0) # Steady State

        # 2. Momentum Y
        eq_v = (DiffusionTerm(coeff=biology_model.mu, var=self.v) 
                - ImplicitSourceTerm(coeff=drag_coeff, var=self.v) 
                - self.p.grad[1] 
                + stress_y == 0)

        # 3. Continuity (div u = 0)
        # We map this to the Pressure variable.
        # A tiny diffusion term stabilizes the checkerboard pressure nodes.
        eq_p = (self.u.grad[0] + self.v.grad[1] 
                - DiffusionTerm(coeff=1e-10, var=self.p) == 0)

        # 4. COUPLED SYSTEM
        # We solve for u, v, p simultaneously
        eq_system = eq_u & eq_v & eq_p
        
        solver = LinearLUSolver() # Direct solver is best for coupled 2D

        print("Starting Quasi-Static Loop...")
        
        for step in tqdm(range(steps)):
            t = step * dt_bio
            
            # --- Update Physics ---
            bio_drag = bio.get_drag(self.mesh.x, self.mesh.y, t)
            bio_stress = bio.get_stress(self.mesh.x, self.mesh.y, t)
            
            # Update Terms
            # Use Smooth Drag + Wall Penalty
            # Inside: ~1e7. Outside: ~1e9 (Penalty method for wall)
            drag_val = bio_drag * (1.0 - self.chi.value) + 1e9 * self.chi.value
            drag_coeff.value = drag_val
            
            # Update Stress
            s_val = bio_stress * (1.0 - self.chi.value)
            # We need the Force Vector (Divergence of Stress Scalar)
            # F = grad(Sigma)
            # FiPy calculates gradients better on variables
            s_var = CellVariable(mesh=self.mesh, value=s_val)
            stress_x.value = s_var.grad[0]
            stress_y.value = s_var.grad[1]

            # --- SOLVE COUPLED ---
            # This solves u, v, and p in one shot.
            # No predictors. No correctors. No beta.
            eq_system.solve(solver=solver)
            
            # --- Enforce Hard Wall (Visual Cleanup) ---
            # The 1e9 drag handled the physics, this cleans the noise
            self.u.value[self.wall_mask] = 0.0
            self.v.value[self.wall_mask] = 0.0
            
            # (Save data here...)