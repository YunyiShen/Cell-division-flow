import scipy.io
import numpy as np 
from cytoFD.inverse.pressure import pressure_recon
from matplotlib import pyplot as plt

denoised_data = np.load("./res/20241206_inducing_res_spVel1.npz")['denoised']
psave, u_proj, x, y = pressure_recon(denoised_data)
X_mesh = np.load("./res/20241206_inducing_res_spVel1.npz")['X_mesh']

this_frame = denoised_data[0].reshape(120, 120, 2)
fig, axs = plt.subplots(1, 2, figsize=(14, 8))
axs[0].contourf(x, y, psave[2], levels=100, cmap='coolwarm')
axs[0].quiver(x[::4, ::4], y[::4, ::4], 
                u_proj[1][0][::4, ::4], 
                u_proj[1][1][::4, ::4], 
                scale = u_proj.std() * 60, width = 0.005, )
axs[1].quiver(X_mesh[::4,0], 
                 X_mesh[::4,1], 
                 denoised_data[2][::4,0], 
                 denoised_data[2][::4,1], 
                 scale = 0.2,
                 width = 0.005, 
                 color = "blue")
plt.show()
plt.savefig("pressurerec.png")
plt.close()

breakpoint()
