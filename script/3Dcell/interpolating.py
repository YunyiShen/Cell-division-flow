
import pyvista as pv
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from cytoFD.forward.solinoidal_interpolating import solinoidal_interpolating, simple_interpolate


visc = [4000, 20000]
refine = 2
simures = np.load(f"./simulations/modelcell3D_Stokes_maxstress1000.0_drag0_size0.5_visc{visc[0]}-{visc[1]}_dt0.05_dx0.03225806451612903_tmax60.npz")
#breakpoint()

xu = np.unique(simures['x'])
x = simures['x']
y = simures['y']
N = simures['N']
d = xu[1] - xu[0]
x0 = xu[0] - 0.5 * d
k = np.repeat(np.arange(N), N*N)
z = x0 + (k + 0.5) * d
chi_thr = 0.4
u = simures['u'][-1]
v = simures['v'][-1]
w = simures['w'][-1]
stress = simures['stress_ext'][-1]
chi = simures['chi']



xf, yf, zf, stressf = simple_interpolate(stress, x, y, z, refine=refine)
#breakpoint()
xff, yff, zff, uf, vf, wf, chif = solinoidal_interpolating(x, y, z, u, v, w, chi, refine = refine)
#breakpoint()

np.savez(f"./simulations/modelcell3D_Stokes_maxstress1000.0_drag0_size0.5_visc{visc[0]}-{visc[1]}_dt0.05_dx0.03225806451612903_tmax60_interpolated{refine}.npz",
         x = xf, y = yf, z = zf,
         stress = stressf, chi = chif,
         u = uf, v = vf, w = wf,
         N = simures['N'] * refine
         )


