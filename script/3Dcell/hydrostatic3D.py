import numpy as np 
from cytoFD.forward.cell3d import celldivflow3D
import matplotlib.pyplot as plt
import scipy.io


myflow = celldivflow3D()
u, v, w, p, p_ext, t, x, y, z, N = myflow.solve(dt = 0.01, steps = 1)
np.savez("./modelcell3Dmax20", u = u, v = v, w = w, 
                    p = p, p_ext = p_ext, t = t, 
                    x = x, y = y, z = z, N = N)
