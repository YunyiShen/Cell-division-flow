import numpy as np 
from cytoFD.forward.planecells import gaussianbump2D, cellBoundary2D, celldivflow2D, growinggaussianbump2D
import matplotlib.pyplot as plt
from cytoFD.forward.plot_utils import velocity_animation
import scipy.io


pressure = growinggaussianbump2D(theta = (10/90) *  np.pi/2)
myflow = celldivflow2D(pressure_field = pressure, N=300)
v, p, p_ext, t = myflow.solve(dt = 0.01, steps = 501)
np.savez("./modelcell2D", v = v, p = p, p_ext = p_ext, t = t)


#breakpoint()
fig, axs = myflow.plot_vel_p_end(thinning = 10)
fig.savefig("hydrostatic.png")



