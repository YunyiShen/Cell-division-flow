import numpy as np 
from cytoFD.forward.planecells import gaussianbump2D, cellBoundary2D, celldivflow2D, growinggaussianbump2D
import matplotlib.pyplot as plt
from cytoFD.forward.plot_utils import velocity_animation
import scipy.io


pressure = growinggaussianbump2D(timescale = 1.)
myflow = celldivflow2D(pressure_field = pressure)
u, v, p, p_ext, t = myflow.solve(dt = 0.01, steps = 200)
#breakpoint()
fig, axs = myflow.plot_vel_p_end(thinning = 3)
fig.savefig("hydrostatic.png")



