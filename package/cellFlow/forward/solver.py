import numpy as np
from .pressure_poisson import build_up_b, pressure_poisson, build_up_b_with_hydrostatic_tress
from tqdm import tqdm


def velocity_u_update(u, dx, dy, dt, rho, nu, p, un, vn):
    
    u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                    un[1:-1, 1:-1] * dt / dx *
                    (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                    vn[1:-1, 1:-1] * dt / dy *
                    (un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
                    dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                    nu * (dt / dx**2 *
                    (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                    dt / dy**2 *
                    (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))
    return u

def velocity_v_update(v, dx, dy, dt, rho, nu, p, un, vn):
    
    v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                    un[1:-1, 1:-1] * dt / dx *
                    (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                    vn[1:-1, 1:-1] * dt / dy *
                    (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                    dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                    nu * (dt / dx**2 *
                    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                    dt / dy**2 *
                    (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
    return v



#### upwind scheme ####

def compute_F(c):
    denom = abs(c) + 1e-6
    pos_part = np.maximum(c/denom, 0)
    neg_part = np.maximum(-c/denom, 0)
    return pos_part, neg_part

def velocity_u_upwind_update(u, dx, dy, dt, rho, nu, p, un, vn):
    #F = lambda c: (max(c/(abs(c)+1e-6), 0), max(-c/(abs(c)+1e-6), 0))
    #F_vectorized = np.vectorize(F) # vectorize function F to support element-wise operations on arrays
    fe1, fe2 = compute_F(un)       
    fw1, fw2 = fe1, fe2
    ue = un[1:-1, 1:-1] * fe1[1:-1, 1:-1] + un[1:-1, 2:] * fe2[1:-1, 1:-1]     
    uw = un[1:-1, 0:-2] * fw1[1:-1, 1:-1] + un[1:-1, 1:-1]* fw2[1:-1, 1:-1]

    fnorth1, fnorth2 = compute_F(vn)       
    fs1, fs2 = fnorth1, fnorth2
    unorth = un[1:-1, 1:-1] * fnorth1[1:-1, 1:-1] + un[2:, 1:-1] * fnorth2[1:-1, 1:-1]     
    us = un[0:-2, 1:-1] * fs1[1:-1, 1:-1] + un[1:-1, 1:-1]* fs2[1:-1, 1:-1]
   
    u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
                    un[1:-1, 1:-1] * dt / dx *
                    (ue - uw) -
                    vn[1:-1, 1:-1] * dt / dy *
                    (unorth - us) -
                    dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                    nu * (dt / dx**2 *
                    (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                    dt / dy**2 *
                    (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))
    return u

def velocity_v_upwind_update(v, dx, dy, dt, rho, nu, p, un, vn):
    
    #F = lambda c: (max(c/(abs(c)+1e-6), 0), max(-c/(abs(c)+1e-6), 0))
    #F_vectorized = np.vectorize(F) # vectorize function F to support element-wise operations on arrays
    fe1, fe2 = compute_F(un)       
    fw1, fw2 = fe1, fe2
    ve = vn[1:-1, 1:-1] * fe1[1:-1, 1:-1] + vn[1:-1, 2:] * fe2[1:-1, 1:-1]     
    vw = vn[1:-1, 0:-2] * fw1[1:-1, 1:-1] + vn[1:-1, 1:-1]* fw2[1:-1, 1:-1]

    fnorth1, fnorth2 = compute_F(vn)       
    fs1, fs2 = fnorth1, fnorth2
    vnorth = vn[1:-1, 1:-1] * fnorth1[1:-1, 1:-1] + vn[2:, 1:-1] * fnorth2[1:-1, 1:-1]     
    vs = vn[0:-2, 1:-1] * fs1[1:-1, 1:-1] + vn[1:-1, 1:-1]* fs2[1:-1, 1:-1]
    
    v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                    un[1:-1, 1:-1] * dt / dx *
                    (ve - vw) -
                    vn[1:-1, 1:-1] * dt / dy *
                    (vnorth - vs) -
                    dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                    nu * (dt / dx**2 *
                    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                    dt / dy**2 *
                    (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
    return v




def naivesolver_with_extra_hydrostatic(nt, u, v, 
               dt, dx, dy, 
               X, Y, hydro_stress : callable,
               mask,
               rho = 1, nu = 0.6, upwind = False,
               save_every = 10, save_from = 20, 
               kwargs = {"tol": 1e-3, "maxit": 500}):
    '''
    Solve the Navier-Stokes equation with hydrostatic stress term
    nt: number of time steps
    u, v: initial velocity field
    dt: time step
    dx, dy: grid spacing
    X, Y: grid points
    hydro_stress: hydrostatic stress term, a callable function taking X, Y, t returns a 2D array of extra hydro static stress
    mask: boundary condition
    rho: density
    nu: viscosity
    save_every: save every n time steps
    save_from: start saving from n time steps
    kwargs: additional parameters to pressure_poisson

    return: u, v, p, stress
    '''

    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros_like(X)
    p = np.zeros_like(X)

    u_save = []
    v_save = []
    stress_inner_save = []
    pressure_save = []

    #p = concentration(X, Y, dt, scale_x = 0.15, scale_y = 0.5, scale_t = 1)

    for n in tqdm(range(nt)):

        t = (n) * dt
        
        un = u.copy()
        vn = v.copy()
        stress_inner = hydro_stress(X, Y, t)
        b = build_up_b_with_hydrostatic_tress(b, rho, dt, u, v, stress_inner, dx, dy)
        p = pressure_poisson(p , dx, dy, b, mask, **kwargs)  # the correction pressure
        #stress = p - concentration(X, Y, t, scale_x = 0.15, scale_y = 0.5, scale_t = 1) # add stress
        #stress = p+stress_inner
        if upwind:
            u = velocity_u_upwind_update(u, dx, dy, dt, rho, nu, p - stress_inner, un, vn)
            v = velocity_v_upwind_update(v, dx, dy, dt, rho, nu, p - stress_inner, un, vn)
        else:
            u = velocity_u_update(u, dx, dy, dt, rho, nu, p - stress_inner, un, vn)
            v = velocity_v_update(v, dx, dy, dt, rho, nu, p - stress_inner, un, vn)
        # set BC
        u = u * mask
        v = v * mask
        if n % save_every == 0 and n > save_from:
            u_save.append(u.copy())
            v_save.append(v.copy())
            stress_inner_save.append(stress_inner.copy())
            pressure_save.append(p.copy())

    return u, v, p, stress_inner, u_save, v_save, pressure_save, stress_inner_save 





def chorin_projection_with_extra_hydrostatic(nt, u, v, 
               dt, dx, dy, 
               X, Y, hydro_stress : callable,
               mask,
               rho = 1, nu = 0.6, upwind = False, 
               save_every = 10, save_from = 20,kwargs = {}):
    
    '''
    Solve the Navier-Stokes equation with hydrostatic stress term using Chorin's projection method
    nt: number of time steps
    u, v: initial velocity field
    dt: time step
    dx, dy: grid spacing
    X, Y: grid points
    hydro_stress: hydrostatic stress term, a callable function taking X, Y, t returns a 2D array of extra hydro static stress
    mask: boundary condition
    rho: density
    nu: viscosity
    save_every: save every n time steps
    save_from: start saving from n time steps
    kwargs: additional parameters to pressure_poisson

    return: u, v, p, stress
    '''
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros_like(X)
    p = np.zeros_like(X)

    u_save = []
    v_save = []
    stress_inner_save = []
    pressure_save = []

    #p = concentration(X, Y, dt, scale_x = 0.15, scale_y = 0.5, scale_t = 1)

    for n in tqdm(range(nt)):

        t = (n) * dt
        
        un = u.copy()
        vn = v.copy()
        stress_inner = hydro_stress(X, Y, t)
        if upwind:
            ustar = velocity_u_upwind_update(u, dx, dy, dt, rho, nu, -stress_inner, un, vn)
            vstar = velocity_v_upwind_update(v, dx, dy, dt, rho, nu, -stress_inner, un, vn)
        else:
            ustar = velocity_u_update(u, dx, dy, dt, rho, nu, -stress_inner, un, vn)
            vstar = velocity_v_update(v, dx, dy, dt, rho, nu, -stress_inner, un, vn)
        b = build_up_b(b, rho, dt, ustar, vstar, dx, dy)
        p = pressure_poisson(p , dx, dy, b, mask, **kwargs)
        # set BC
        u = ustar - dt / (2 * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) # projection, central difference in space
        v = vstar - dt / (2 * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1])
        u = u * mask
        v = v * mask
        if n % save_every == 0 and n > save_from:
            u_save.append(u.copy())
            v_save.append(v.copy())
            stress_inner_save.append(stress_inner.copy())
            pressure_save.append(p.copy())

    return u, v, p, stress_inner, u_save, v_save, pressure_save, stress_inner_save 