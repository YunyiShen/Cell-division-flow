import numpy as np 
from cytoFD.forward.planegeneral import growinggaussianbump2Dconc, ActinModel, CellDivFlow2D, CellDivFlow2DPISO
import matplotlib.pyplot as plt
from cytoFD.forward.plot_utils import velocity_animation
import scipy.io


setups = []


#exploding and literature settings
for stress in [ 1e3, 2e3, 5e3, 5e2, 1e2]:
    for cell_radius in [1./2]:
        for drag_range in [[0,0], [3.e4, 1.e6], [3.e5, 1.e7]]:
            for visc_range in [
                            
                           [5, 20],
                           [50, 200],
                           [4000, 20000],
                           [8000, 40000] # this is a literature value
                           
                           
                           ]:
                setups.append({"stress": stress, 
                          "cell_radius": cell_radius,
                          "visc_range": visc_range,
                          "drag_range": drag_range
                          })
# in total 45



# grid w/o drag
'''
for stress in [ 1e3, 2e3, 5e3, 5e2, 1e2]:
    for cell_radius in [1./2]:
        for drag_range in [[0,0]]:
            for visc_range in [
                            
                           #[5000, 50000],
                           #[10000, 50000],
                           
                           #[1000, 10000],
                            [2000, 2000],
                            [3000, 3000],
                            [5000, 5000],
                            [2000, 10000],
                            [3000, 10000],
                            [5000, 10000]
                           #[500, 5000],
                           #[1000, 5000],
                           
                           ]:
                setups.append({"stress": stress, 
                          "cell_radius": cell_radius,
                          "visc_range": visc_range,
                          "drag_range": drag_range
                          })
#### 30 setups for viscosity ####
'''
'''
for stress in [1e3]: #[ 1e3, 2e3, 5e3, 5e2, 1e2]:
    for cell_radius in np.linspace(0.05/2, 1./2, num = 10): #[(1./2)/2, (1./(2**2))/2, (1./(2**3))/2]: #[1./2]:
        for visc_range in [
                            
                           #[5000, 50000],
                           #[10000, 50000],
                           
                           #[1000, 10000],
                            [2000, 2000],
                            [3000, 3000],
                            [5000, 5000],
                            [2000, 10000],
                            [3000, 10000],
                            [5000, 10000]
                           #[500, 5000],
                           #[1000, 5000],
                           
                           ]:
            setups.append({"stress": stress, 
                          "cell_radius": cell_radius,
                          "visc_range": visc_range,
                          "drag_range": [0,0]
                          })
##### 18 setups for size ####

##### size and drag ####
for stress in [1e3]: #[ 1e3, 2e3, 5e3, 5e2, 1e2]:
    for cell_radius in np.linspace(0.05/2, 1./2, num = 10): #[(1./2)/2, (1./(2**2))/2, (1./(2**3))/2]: #[1./2]:
        for drag_range in [
                            
                           #[5000, 50000],
                           #[10000, 50000],
                           
                           #[1000, 10000],
                            [20000, 20000],
                            [30000, 30000],
                            [50000, 50000],
                            [20000, 100000],
                            [30000, 100000],
                            [50000, 100000]
                           #[500, 5000],
                           #[1000, 5000],
                           
                           ]:
            setups.append({"stress": stress, 
                          "cell_radius": cell_radius,
                          "visc_range": [500, 500],
                          "drag_range": drag_range
                          })

'''

'''
for stress in [ 1e3, 2e3, 5e3, 5e2, 1e2]:
    for cell_radius in [1./2]:
        for visc_range in [[1000, 1000], [500, 500]]:
            for drag_range in [
                            
                           #[5000, 50000],
                           #[10000, 50000],
                           
                           #[1000, 10000],
                            [20000, 20000],
                            [30000, 30000],
                            [50000, 50000],
                            [20000, 100000],
                            [30000, 100000],
                            [50000, 100000]
                           #[500, 5000],
                           #[1000, 5000],
                           
                           ]:
                setups.append({"stress": stress, 
                          "cell_radius": cell_radius,
                          "visc_range": visc_range,
                          "drag_range": drag_range
                          })
##### 60 setups for drag #####
'''
def run(run_id, dx = None, tmax = 10, dt = None, N = 51, Stokes = False):
    # N determines number of girds
    
    
    
        
        
    setup  = setups[run_id]
    cell_radius = setup["cell_radius"]
    stress_max = setup["stress"]
    visc_range = setup["visc_range"]
    drag_range = setup['drag_range']
    
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
    
    biology = ActinModel(actin = growinggaussianbump2Dconc(theta = (90/90) *  np.pi/2,
                                precision = np.array([[25, 0],[0, 30**2]]) / ((cell_radius * 2) ** 2),
                                timescale = tmax/2.),
                        stress_range = [1e-5, stress_max],
                        drag_range = drag_range,#[0, 0],
                        visc_range = visc_range,
                        stress_power = 1.,
                        drag_power = 1.,
                        visc_power = 1., 
                        rho=1.0
                        )
    
    
    myflow = CellDivFlow2D(
                                  domain_size = 2 * cell_radius, # 1mm 
                                  cell_radius = cell_radius,
                                  N=int(2 * cell_radius/dx),
                            Stokes = Stokes      
                            )
    
    
    
    
    res = myflow.solve(biology, dt = dt, 
                                                    steps = int(tmax/dt), 
                                                    save_every = max(int(tmax/dt/100), 1), 
                                                    alpha_wall=1e9)

    if Stokes:
        np.savez(f"./simulations/modelcell2D_Stokes_maxstress{stress_max}_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_dt{dt}_dx{dx}_tmax{tmax}", 
             **res)
    else:
        np.savez(f"./simulations/modelcell2D_maxstress{stress_max}_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_dt{dt}_dx{dx}_tmax{tmax}", 
             **res)
             #u = uu, v = v, p = p, stress_ext = stress_ext, t = t, x = x, y = y, N = N)

    #breakpoint()
    print(res["u"][-1].max()*1000*60)
    fig, axs = myflow.plot_vel_p_end(thinning = 2, scale = None, idx = -1)
    if Stokes:
        fig.savefig(f"./simulations/modelcell2D_Stokes_maxstress{stress_max}_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_dt{dt}_dx{dx}_N{N}_tmax{tmax}.png")
    else:
        fig.savefig(f"./simulations/modelcell2D_maxstress{stress_max}_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_dt{dt}_dx{dx}_N{N}_tmax{tmax}.png")

import fire
if __name__ == "__main__":
    fire.Fire(run)
