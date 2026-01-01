import numpy as np 
from cytoFD.forward.planegeneral import growinggaussianbump2Dconc, cortex2Dconc, ActinModel, CellDivFlow2D
import matplotlib.pyplot as plt
from cytoFD.forward.plot_utils import velocity_animation
import scipy.io
import os


setups = []

for stress in [ 1e3, 5e2, 1e2]:
    for cell_radius in np.linspace(0.1/2, 1./2, num = 20): 
        for visc_range in [
                           #[4000, 20000],
                           #[4000, 4000],
                           #[20000, 20000]
                           [3000, 10000],
                           [3000, 3000],
                           [10000, 10000]
                           
                           ]:
            setups.append({"stress": stress, 
                          "cell_radius": cell_radius,
                          "visc_range": visc_range,
                          "drag_range": [0,0],
                          "aspect_ratio": 1.0
                          })
##### 180 setups for size ####

'''
for stress in [ 1e3]:
    for cell_radius in [1./2]:
        for drag_range in [[3.e4, 1.e6], [3.e5, 1.e7]]:
            for visc_range in [
                           [5, 20],
                           [500, 500], 
                           ]:
                for aspect_ratio in [0.25, 0.5, 2, 4]:
                    setups.append({"stress": stress, 
                          "cell_radius": cell_radius,
                          "visc_range": visc_range,
                          "drag_range": drag_range,
                          "aspect_ratio": aspect_ratio
                          })

#### 16 settings ####

##### size and drag ####
for stress in [ 1e3, 5e2, 1e2]:
    for cell_radius in np.linspace(0.1/2, 1./2, num = 20): 
        for drag_range in [
                            
                            [20000, 20000],
                            [50000, 100000]
                           
                           ]:
            setups.append({"stress": stress, 
                          "cell_radius": cell_radius,
                          "visc_range": [500, 500],
                          "drag_range": drag_range,
                          "aspect_ratio": 1.0
                          })

# 100 setups
'''

# geometry
for stress in [ 1e3]:
    for cell_radius in [1./2]:
        for drag_range in [[0,0]]:
            for visc_range in [
                           #[4000, 20000],
                           #[4000, 4000],
                           #[20000, 20000]
                           [3000, 10000],
                           [3000, 3000],
                           [10000, 10000]
                           ]:
                for aspect_ratio in [0.25, 0.5, 2, 4]:
                    setups.append({"stress": stress, 
                          "cell_radius": cell_radius,
                          "visc_range": visc_range,
                          "drag_range": drag_range,
                          "aspect_ratio": aspect_ratio
                          })
# 9 settings



def run(run_id, dx = None, tmax = 10, dt = None, N = 51, Stokes = False, which_biology = "bulk"):
    # N determines number of girds
    os.makedirs(f"./simulations/{which_biology}", exist_ok=True)
    setup  = setups[run_id]
    cell_radius = setup["cell_radius"]
    stress_max = setup["stress"]
    visc_range = setup["visc_range"]
    drag_range = setup['drag_range']
    aspect_ratio = setup['aspect_ratio']
    
    if N is not None:
        dx = cell_radius * 2 / N
    if dx is None and N is None:
        dx = cell_radius * 2 / 100
        N = 100
    if dt is None:
        dt = 1./(mu/(dx ** 2))
    print(setup)
    print(f"dx {dx}, tmax {tmax}, dt {dt}")
    if Stokes:
        print("running Stokes equation instead of Navier-Stokes")
    
    if which_biology == "bulk":
        actin = growinggaussianbump2Dconc(theta = (90/90) *  np.pi/2,
                                precision = np.array([[25, 0],[0, 30**2]]) / ((cell_radius * 2) ** 2),
                                timescale = tmax/2.)
    
    elif which_biology == "cortex":
        actin = cortex2Dconc(R = 0.95*cell_radius, width = 0.02, 
                             aspect_ratio = aspect_ratio, 
                             phase_rot = 0., timescale = tmax/2)
    
    else:
        raise("Unknown biology")
        return
    
    
    biology = ActinModel(actin = actin,
                        stress_range = [1e-5, stress_max],
                        drag_range = drag_range,#[0, 0],
                        visc_range = visc_range,
                        stress_power = 1.,
                        drag_power = 1.,
                        visc_power = 1., 
                        rho=1.0,
                        domain_size = 2 * cell_radius * (aspect_ratio if aspect_ratio > 1. else 1.), # 1mm 
                        cell_radius = cell_radius,
                        aspect_ratio = aspect_ratio
                        )
    
    myflow = CellDivFlow2D(
                                  
                                  N=int(2 * cell_radius/dx),
                            Stokes = Stokes      
                            )
    
    
    
    
    res = myflow.solve(biology, dt = dt, 
                                                    steps = int(tmax/dt), 
                                                    save_every = max(int(tmax/dt/100), 1), 
                                                    alpha_wall=1e9)

    if Stokes:
        np.savez(f"./simulations/{which_biology}/modelcell2D_Stokes_maxstress{stress_max}_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_aspectratio{aspect_ratio}_dt{dt}_N{N}_tmax{tmax}", 
             **res, **setup)
    else:
        np.savez(f"./simulations/{which_biology}/modelcell2D_maxstress{stress_max}_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_aspectratio{aspect_ratio}_dt{dt}_N{N}_tmax{tmax}", 
             **res, **setup)
             #u = uu, v = v, p = p, stress_ext = stress_ext, t = t, x = x, y = y, N = N)

    #breakpoint()
    print(res["u"][-1].max()*1000*60)
    fig, axs = myflow.plot_vel_p_end(thinning = 2, scale = None, idx = -1)
    if Stokes:
        fig.savefig(f"./simulations/{which_biology}/modelcell2D{which_biology}_Stokes_maxstress{stress_max}_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_aspectratio{aspect_ratio}_dt{dt}_N{N}_tmax{tmax}.png")
    else:
        fig.savefig(f"./simulations/{which_biology}/modelcell2D{which_biology}_maxstress{stress_max}_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_aspectratio{aspect_ratio}_dt{dt}_N{N}_tmax{tmax}.png")

import fire
if __name__ == "__main__":
    fire.Fire(run)
