import numpy as np 
from cytoFD.forward.planecells import gaussianbump2D, cellBoundary2D, celldivflow2D, growinggaussianbump2D
import matplotlib.pyplot as plt
from cytoFD.forward.plot_utils import velocity_animation
import scipy.io


setups = []
for stress in [1e5, 5e4, 1e4]:
    for cell_radius in [1./2, 0.5/2, 0.2/2]:
        for mu in [5, 10, 20]:
            setups.append({"stress": stress, 
                          "cell_radius": cell_radius,
                          "mu": mu
                          })
# in total 27

def run(run_id, dx = 0.04, tmax = 10, dt = None):
    setup  = setups[run_id]
    cell_radius = setup["cell_radius"]
    stress_max = setup["stress"]
    mu = setup["mu"]
    print(setup)
    print(f"dx {dx}, tmax {tmax}, dt {dt}")
    stress = growinggaussianbump2D(theta = (90/90) *  np.pi/2, # rotate the gap?
                                precision = np.array([[25, 0],[0, 25**2]]) * ((cell_radius * 2) ** 2),
                                timescale = tmax,
                                maxp = stress_max) # maximum stress? mg/(mm s^2)
    myflow = celldivflow2D(stress_field = stress,
                                  mu = mu, # dynamic viscosity mg/(mm s)
                                  rho = 1., # density mg/mm^3, so that the cell is of size 1mm-ish
                                  domain_size = 2 * cell_radius, # 1mm 
                                  cell_radius = cell_radius,
                                  N=int(2 * cell_radius/dx))
    # N determines number of girds
    if dt is None:
        dt = 1./(mu/(dx ** 2))
    uu, v, p, stress_ext, t, x, y, N = myflow.solve(dt = dt, steps = int(tmax/dt), save_every = max(int(tmax/dt/100), 1))

    np.savez(f"./modelcell2Dmax{stress_max}_size{cell_radius}_visc{mu}_dt{dt}_dx{dx}_tmax{tmax}", u = uu, v = v, p = p, stress_ext = stress_ext, t = t, x = x, y = y, N = N)

    #breakpoint()
    fig, axs = myflow.plot_vel_p_end(thinning = 1, scale = None, idx = -1)
    fig.savefig(f"hydrostatic_norot{stress_max}_size{cell_radius}_visc{mu}_dt{dt}_dx{dx}_tmax{tmax}.png")

import fire
if __name__ == "__main__":
    fire.Fire(run)
