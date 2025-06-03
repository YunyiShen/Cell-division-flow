import numpy as np 
from cytoFD.forward.cell3d import celldivflow3D
import matplotlib.pyplot as plt
import scipy.io

N = 36
steps = 500
myflow = celldivflow3D(N=N)
u, v, w, p, stress_ext, t, x, y, z, N = myflow.solve(dt = 0.01, steps = steps)
np.savez(f"./modelcell3Dmax20_gird{N}_steps{steps}.npz", u = u, v = v, w = w, 
                    p = p, stress_ext = stress_ext, t = t, 
                    x = x, y = y, z = z, N = N)
