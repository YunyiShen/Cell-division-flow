import numpy as np 
from cytoFD.forward.planegeneral import growinggaussianbump2Dconc, ActinModel, CellDivFlow2D, CellDivFlow2DPISO
import matplotlib.pyplot as plt
from cytoFD.forward.plot_utils import velocity_animation
import scipy.io


setups = []
for stress in [ 1e3, 1e4, 5e3, 5e2, 1e2]:
    for cell_radius in [1./2]:
        for visc_range in [
                           #[200, 500],
                           #[200, 1000], 
                           #[200, 3000],
                           #[500, 3000],  
                           [5000, 50000],
                           [10000, 50000],
                           
                           [1000, 10000],
                           [2000, 10000],

                           [500, 5000],
                           [1000, 5000],
                           
                           ]:
            setups.append({"stress": stress, 
                          "cell_radius": cell_radius,
                          "visc_range": visc_range
                          })
# in total 27

def run(run_id, dx = None, tmax = 10, dt = None, N = 51):
    # N determines number of girds
    
    
    
        
        
    setup  = setups[run_id]
    cell_radius = setup["cell_radius"]
    stress_max = setup["stress"]
    visc_range = setup["visc_range"]
    
    if N is not None:
        dx = cell_radius * 2 / N
    if dx is None and N is None:
        dx = cell_radius * 2 / 100
        N = 100
    if dt is None:
        dt = 1./(mu/(dx ** 2))
    print(setup)
    print(f"dx {dx}, tmax {tmax}, dt {dt}")
    
    biology = ActinModel(actin = growinggaussianbump2Dconc(theta = (90/90) *  np.pi/2,
                                precision = np.array([[25, 0],[0, 30**2]]) * ((cell_radius * 2) ** 2),
                                timescale = tmax/2.),
                        stress_range = [1e-5, stress_max],
                        drag_range = [0, 0],
                        visc_range = visc_range,
                        stress_power = 1.,
                        drag_power = 1.,
                        visc_power = 1., 
                        rho=1.0
                        )
    
    
    myflow = CellDivFlow2D(
                                  domain_size = 2 * cell_radius, # 1mm 
                                  cell_radius = cell_radius,
                                  N=int(2 * cell_radius/dx))
    
    
    
    
    res = myflow.solve(biology, dt = dt, 
                                                    steps = int(tmax/dt), 
                                                    save_every = max(int(tmax/dt/100), 1), 
                                                    alpha_wall=1e9)

    np.savez(f"./simulations/modelcell2D_maxstress{stress_max}_drag{0}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_dt{dt}_dx{dx}_tmax{tmax}", 
             **res)
             #u = uu, v = v, p = p, stress_ext = stress_ext, t = t, x = x, y = y, N = N)

    #breakpoint()
    print(res["u"][-1].max()*1000*60)
    fig, axs = myflow.plot_vel_p_end(thinning = 2, scale = None, idx = -1)
    fig.savefig(f"./simulations/modelcell2D_maxstress{stress_max}_drag{0}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_dt{dt}_dx{dx}_N{N}_tmax{tmax}.png")

import fire
if __name__ == "__main__":
    fire.Fire(run)
