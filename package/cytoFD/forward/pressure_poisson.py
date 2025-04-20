import numpy as np


# source term in pressure poisson equation without external pressure
def build_up_b(b, rho, dt, u, v, dx, dy):
    
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    return b

# source term in pressure poisson equation with external pressure
def build_up_b_with_hydrostatic_tress(b, rho, dt, u, v, stress, dx, dy):
    
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2)) + \
                    ((stress[1:-1, 2:] + stress[1:-1, 0:-2] - 2 * stress[1:-1, 1:-1])/(dy**2) + 
                     (stress[2:, 1:-1] + stress[0:-2,1:-1] - 2 * stress[1:-1, 1:-1])/(dx**2))

    return b

def build_up_b_convection_only(rho, dt, u, v, dx, dy):
    b = np.zeros_like(u)
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                                (
                                    (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) + 
                                    (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)
                                ) 
                            )
                    )


    return b


def pressure_poisson(p, dx, dy, b, mask, tol = 1e-10, maxit = 500):

    '''
    Solving pressure Poisson equation, with boundary condition set by mask
    p: initial pressure field
    dx, dy: grid spacing
    b: source term
    mask: boundary condition, outside of the domain is 0
    tol: tolerance
    maxit: maximum iteration

    return: (dynamic) pressure field

    '''

    p = p * mask
    pn = np.empty_like(p)
    pn = p.copy()
    l1norm = 1
    for q in range(maxit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1,1:-1])
        p = p * mask
        '''
        if np.abs((pn - p)/(max(p.std(), 1e-8))).max() < tol:
            #breakpoint()
            break  
        '''    
        
        l1norm = (np.sum(np.abs(p[:]-pn[:])) / (np.sum(np.abs(pn[:]))+1e-8))
        if l1norm < tol:
            break
         
    return p * mask # set BC