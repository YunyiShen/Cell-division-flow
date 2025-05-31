import numpy as np
from fipy import CellVariable, DiffusionTerm, Grid2D
from fipy.tools import numerix
import math
from tqdm import tqdm


def project_to_divfree(u_meas, dx=1.0):
    """
    Project a 2D velocity field (u_meas) to a divergence-free field using Helmholtz-Hodge projection.

    Parameters:
        u_meas: np.ndarray of shape (2, N, N)
            The measured velocity field (u, v).
        dx: float
            Grid spacing.

    Returns:
        u_divfree: np.ndarray of shape (2, N, N)
            The divergence-free projection of the input velocity field.
    """
    assert u_meas.shape[0] == 2, "Input velocity must have shape (2, N, N)"
    N = u_meas.shape[1]
    assert u_meas.shape[2] == N, "Only square grids (N x N) are supported"

    u, v = u_meas[0], u_meas[1]

    # Create FiPy mesh
    mesh = Grid2D(dx=dx, dy=dx, nx=N, ny=N)

    # Compute divergence of input field
    du_dx = np.gradient(u, dx, axis=0)
    dv_dy = np.gradient(v, dx, axis=1)
    div_u = du_dx + dv_dy

    # Solve Poisson equation ∇²φ = div(u)
    phi = CellVariable(name="phi", mesh=mesh, value=0.0)
    rhs = CellVariable(name="rhs", mesh=mesh, value=div_u.flatten())
    poisson_eq = DiffusionTerm(var=phi) == rhs
    poisson_eq.solve(var=phi)

    # Compute ∇φ
    phi_array = phi.value.reshape(N, N)
    dphi_dx = np.gradient(phi_array, dx, axis=0)
    dphi_dy = np.gradient(phi_array, dx, axis=1)

    # Subtract gradient to get divergence-free field
    u_proj = u - dphi_dx
    v_proj = v - dphi_dy

    u_divfree = np.stack([u_proj, v_proj], axis=0)
    #breakpoint()
    return u_divfree

def pressure_recon(u, L = 78, rho = 1., mu = 0.01, dt = 10):
    timesteps = len(u)
    N = int(math.sqrt(u[0].shape[0]))
    #breakpoint()
    # Parameters
    u = np.array(u)
    u_data = u.transpose(0, 2, 1).reshape(timesteps, 2, N, N).transpose(0,1,3,2)
    dx = L / N
    mesh = Grid2D(dx=dx, dy=dx, nx=N, ny=N)
    x, y = mesh.cellCenters
    center_mask = (np.abs(x - 0.5) < dx / 2) & (np.abs(y - 0.5) < dx / 2)   

    # Setup variables
    p = CellVariable(name="pressure", mesh=mesh, value=0.0)
    p.setValue(0.0, where=center_mask)
    
    p_save = []
    u_data[0] = project_to_divfree(u_data[0], dx)
    u_data[1] = project_to_divfree(u_data[1], dx)
    #breakpoint()
    # Compute pressure for each time step
    for t in tqdm(range(1, timesteps - 1)):
        u_data[t+1] = project_to_divfree(u_data[t+1], dx)
        u_prev = u_data[t - 1]
        u_curr = u_data[t]
        u_next = u_data[t + 1]
        # Central difference in time
        du_dt = (u_next - u_prev) / (2 * dt)

        # Velocity at current step
        u = u_curr[0]
        v = u_curr[1]

        # Compute convective term
        u_x = np.gradient(u, dx, axis=0)
        u_y = np.gradient(u, dx, axis=1)
        v_x = np.gradient(v, dx, axis=0)
        v_y = np.gradient(v, dx, axis=1)

        conv_u = u * u_x + v * u_y
        conv_v = u * v_x + v * v_y

        div_accel = np.gradient(du_dt[0] + conv_u, dx, axis=0) + np.gradient(du_dt[1] + conv_v, dx, axis=1)

        # Optional viscous term (can be added if data is smooth enough)
        lap_u = np.gradient(np.gradient(u, dx, axis=0), dx, axis=0) + np.gradient(np.gradient(u, dx, axis=1), dx, axis=1)
        lap_v = np.gradient(np.gradient(v, dx, axis=0), dx, axis=0) + np.gradient(np.gradient(v, dx, axis=1), dx, axis=1)
        div_visc = np.gradient(lap_u, dx, axis=0) + np.gradient(lap_v, dx, axis=1)

        rhs = -rho * div_accel + mu * div_visc

        # Solve Poisson equation in FiPy
        rhs_var = CellVariable(name="rhs", mesh=mesh, value=rhs.flatten())
        p_eq = DiffusionTerm(var=p) == rhs_var
        p_eq.solve(var=p)

        # Store p.value or visualize here
        #p.setValue(0.0, where=~inside)
        p_save.append(p.value)
        #p.setValue(0.0, where=center_mask)
    p_save = np.array(p_save)

    return (p_save - p_save.mean()).reshape((timesteps-2, N, N)), u_data[1:-1], x.reshape(N,N), y.reshape(N,N)