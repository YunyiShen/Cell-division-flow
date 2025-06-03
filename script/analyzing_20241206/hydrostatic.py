import numpy as np 
from cytoFD.forward.planecells import gaussianbump2D, cellBoundary2D, celldivflow2D, growinggaussianbump2D
import matplotlib.pyplot as plt
from cytoFD.forward.plot_utils import velocity_animation
import scipy.io


stress = growinggaussianbump2D(theta = (0/90) *  np.pi/2, maxp = 20)
myflow = celldivflow2D(stress_field = stress, N=200)
u, v, p, stress_ext, t, x, y, N = myflow.solve(dt = 0.01, steps = 501)
np.savez("./modelcell2Dmax20_norot", u = u, v = v, p = p, stress_ext = stress_ext, t = t, x = x, y = y, N = N)

#breakpoint()
#breakpoint()
fig, axs = myflow.plot_vel_p_end(thinning = 10)
fig.savefig("hydrostatic_norot.png")



