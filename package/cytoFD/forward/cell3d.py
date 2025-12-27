import fipy as fp
from fipy import CellVariable, FaceVariable, Grid3D, TransientTerm, DiffusionTerm, HybridConvectionTerm, ImplicitSourceTerm, LinearLUSolver
import fipy.tools.numerix as numerix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import copy



class growinggaussianbump3Dconc():
    def __init__(self, precision = np.array([[25, 0, 0],[0, 25, 0],[0, 0, 25**2]]), 
                 timescale = 1.):
        self.precision = precision
        
        self.timescale = timescale
        
    def __call__(self, x, y, z, t):
        #assert self.center is not None, "center need to be set"
        xy = np.stack([x, y, z], axis=0)  # shape (2, ...)
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
                stress_range = [1e-5, 1e3],
                visc_range = [20., 500],
                drag_range = [0., 0.,],
                stress_power = 1.,
                visc_power = 1., 
                drag_power = 2.,
                rho=1.0
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
        self.visc_power = visc_power
        self.visc_range = visc_range
        self.rho = rho
        self.center = None
        self.domain_size = None

    def set_geometry(self, domain_size, center):
        """Called by the solver to sync the biological pattern to the mesh size"""
        self.center = center
        self.domain_size = domain_size

    def get_actin(self, x, y, z, t):
        """
        Calculates Actin Concentration A(x,y,t)
        Returns: Numpy array (0.0 to 1.0)
        """
        if self.center is None:
            raise ValueError("Geometry not set! Run solver.solve() first.")

        A = self.actin(x-self.center[0], y-self.center[1], z-self.center[2], t)
        
        # Safety clamp
        return np.maximum(A, 0.0001)

    def get_stress(self, x, y, z, t):
        """ Constitutive Law: Sigma ~ A^n """
        Apow = self.get_actin(x, y, z, t) ** self.stress_power
        return self.stress_range[1] * (Apow) + self.stress_range[0] * (1-Apow)
    
    def get_viscosity(self, x, y, z, t):
        Apow = self.get_actin(x, y, z, t) ** self.visc_power
        # Interpolate between Cytosol Viscosity (Gap) and Cortex Drag (Bulk)
        return self.visc_range[0] + (self.visc_range[1] - self.visc_range[0]) * (Apow)

    def get_drag(self, x, y, z, t):
        """ Constitutive Law: Alpha ~ A^m """
        Apow = self.get_actin(x, y, z, t) ** self.drag_power
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


class CellDivFlow3D:

    def __init__(self, domain_size=1.0, N=100, cell_radius=0.5, Stokes = False):
        self.L = domain_size
        self.N = N
        self.dx = domain_size / N
        self.cellcenter = [self.L/2.0, self.L/2.0, self.L/2.0]
        self.cellradius = cell_radius
        self.Stokes = Stokes
        
        # Mesh
        self.mesh = Grid3D(dx=self.dx, dy=self.dx, dz = self.dx, nx=N, ny=N, nz = N)
        self.saved = None

        # --- Geometry Masks ---
        x, y , z= self.mesh.cellCenters
        x0, y0, z0 = self.cellcenter
        rdist = numerix.sqrt((x - x0)**2 + (y - y0)**2 + (z-z0)**2)
        
        # Smooth Transition Mask (Chi)
        epsilon = 1.5 * self.dx
        smooth_mask = 0.5 * (1 + numerix.tanh((rdist - self.cellradius) / epsilon))
        self.chi = CellVariable(mesh=self.mesh, value=smooth_mask)
        
        # Binary Mask (For hard constraints/plotting)
        self.wall_mask = (rdist >= self.cellradius)
        self.plotting_mask = self.wall_mask
        
        # Variables
        

    def solve(self, biology_model, dt, steps, alpha_wall=1e4, save_every = 10):
        x, y, z = self.mesh.cellCenters
        # Sync Geometry
        biology_model.set_geometry(self.L, self.cellcenter)
        # 1. Setup
        u = CellVariable(mesh=self.mesh, name="u", value=0.)
        v = CellVariable(mesh=self.mesh, name="v", value=0.)
        w = CellVariable(mesh=self.mesh, name="w", value=0.)
        p = CellVariable(mesh=self.mesh, name="p", value=0.)
        for var in [u, v, w]:
            var.constrain(0.0, self.mesh.exteriorFaces)

        
        # Fields
        total_drag = CellVariable(mesh=self.mesh, value=0.0)
        stress_ext = CellVariable(mesh=self.mesh, value=0.0)
        viscosity = CellVariable(mesh=self.mesh, value=0.0)
        
        solver = LinearLUSolver()
        u_save, v_save, w_save, p_save, stress_save, drag_save, t_save = [], [], [], [], [], [], []

        for step in tqdm(range(steps)):
            t = step * dt
            velocity = FaceVariable(mesh=self.mesh, rank=1)
            velocity[:] = numerix.array([u.arithmeticFaceValue, v.arithmeticFaceValue, w.arithmeticFaceValue])
            # --- A. PHYSICS UPDATE ---
            # 1. Stress
            raw_stress = biology_model.get_stress(self.mesh.x, self.mesh.y, self.mesh.z, t)
            stress_ext.value = raw_stress * (1.0 - self.chi.value)
            
            # 2. Drag 
            # Bio (Inside) + Wall (Outside)
            # alpha_wall = 1e4 is PLENTY. Do not use 1e9.
            bio_drag = biology_model.get_drag(self.mesh.x, self.mesh.y, self.mesh.z, t)
            total_drag.value = bio_drag * (1.0 - self.chi.value) + alpha_wall * self.chi.value
            
            raw_visc = biology_model.get_viscosity(self.mesh.x, self.mesh.y, self.mesh.z, t)
            viscosity.value = raw_visc * (1.0 - self.chi.value)
            #breakpoint()
            # --- B. PREDICTOR STEP (Implicit Drag) ---
            # We use the 'face_velocity' from the PREVIOUS step to advect.
            # This is much more stable than interpolating u/v every time.
            
            if self.Stokes:
                u_star_eq = (ImplicitSourceTerm(coeff=total_drag/ biology_model.rho, var=u)
                         == DiffusionTerm(coeff=viscosity/biology_model.rho, var=u)
                         
                         + (1.0/biology_model.rho) * stress_ext.grad[0] 
                         ) # Implicit Drag
            
                v_star_eq = (ImplicitSourceTerm(coeff=total_drag/ biology_model.rho, var=v)
                         == DiffusionTerm(coeff=viscosity/biology_model.rho, var=v)
                         
                         + (1.0/biology_model.rho) * stress_ext.grad[1] 
                         ) # Implicit Drag
            
                w_star_eq = (ImplicitSourceTerm(coeff=total_drag/ biology_model.rho, var=w)
                         == DiffusionTerm(coeff=viscosity/biology_model.rho, var=w)
                         
                         + (1.0/biology_model.rho) * stress_ext.grad[2] 
                         ) # Implicit Drag
            
            else:
            
            
                u_star_eq = (TransientTerm(var=u) + ImplicitSourceTerm(coeff=total_drag/ biology_model.rho, var=u)
                         == DiffusionTerm(coeff=viscosity/biology_model.rho, var=u)
                         - HybridConvectionTerm(coeff=velocity, var=u)
                         + (1.0/biology_model.rho) * stress_ext.grad[0] 
                         ) # Implicit Drag
            
                v_star_eq = (TransientTerm(var=v) + ImplicitSourceTerm(coeff=total_drag/ biology_model.rho, var=v)
                         == DiffusionTerm(coeff=viscosity/biology_model.rho, var=v)
                         - HybridConvectionTerm(coeff=velocity, var=v)
                         + (1.0/biology_model.rho) * stress_ext.grad[1] 
                         ) # Implicit Drag
            
                w_star_eq = (TransientTerm(var=w) + ImplicitSourceTerm(coeff=total_drag/ biology_model.rho, var=w)
                         == DiffusionTerm(coeff=viscosity/biology_model.rho, var=w)
                         - HybridConvectionTerm(coeff=velocity, var=w)
                         + (1.0/biology_model.rho) * stress_ext.grad[2] 
                         ) # Implicit Drag

            u_star_eq.solve(dt=dt, solver=solver)
            v_star_eq.solve(dt=dt, solver=solver)
            w_star_eq.solve(dt=dt, solver=solver)
            

            # --- C. SIMPLIFIED PRESSURE PROJECTION ---
            velocity[:] = numerix.array([u.arithmeticFaceValue, v.arithmeticFaceValue, w.arithmeticFaceValue])
            div_u_star = velocity.divergence
            
            
            pressure_eq = DiffusionTerm(var=p) == (biology_model.rho/dt) * div_u_star
            pressure_eq.solve(var=p, solver=solver)

            # --- VELOCITY CORRECTION ---
            u.value[:] -= (dt/biology_model.rho) * p.grad[0]
            v.value[:] -= (dt/biology_model.rho) * p.grad[1]
            w.value[:] -= (dt/biology_model.rho) * p.grad[2]

            
            # soft 0 on boundary
            u.value[:] *= (1.-self.chi)
            v.value[:] *= (1.-self.chi)
            w.value[:] *= (1.-self.chi)
            #breakpoint()
                
            
            # --- SAVE ---
            if step % save_every == 0:
                u_save.append(u.value.copy())
                v_save.append(v.value.copy())
                w_save.append(w.value.copy())
                p_tmp = p.value.copy()
                #p_tmp[self.plotting_mask] = np.nan 
                p_save.append(p_tmp)
                drag_tmp = total_drag.value.copy()
                #drag_tmp[self.plotting_mask] = np.nan 
                drag_save.append(drag_tmp)
                stress_save.append(stress_ext.value.copy())
                t_save.append(t)

        self.saved = dict(u=u_save, v=v_save, w = w_save,p=p_save,
                          stress_ext=stress_save, drag=drag_save,
                          chi = self.chi.value.copy(), 
                          t=t_save, x=x, y=y, z=z,steps=steps,
                          N = self.N)
        
        return self.saved
    