import numpy as np 
from cytoFD.forward.solver import naivesolver_with_extra_hydrostatic, chorin_projection_with_extra_hydrostatic
import matplotlib.pyplot as plt
from cytoFD.forward.plot_utils import velocity_animation
import scipy.io




#### experimental data #####
#### concentration data ####
conc_data = scipy.io.loadmat('../data/20241230_xyuv/20241018_ctrl_GFP_z2_raw.mat')
conc_data = conc_data[list(conc_data.keys())[-1]]
n_frames = conc_data.shape[0]
which_frame = n_frames//2
conc_plot = conc_data[which_frame][0]
#conc_plot = (conc_plot-np.min(conc_plot)) / (np.max(conc_plot) - np.min(conc_plot))
X_range_conc = np.linspace(-3, 3, conc_plot.shape[0])
Y_range_conc = np.linspace(-3, 3, conc_plot.shape[1])
X_conc, Y_conc = np.meshgrid(X_range_conc, Y_range_conc)

#### velocity field ####
#vel_data = scipy.io.loadmat("../data/20241230_xyuv/20241018_ctrl001_z2_xyuv_dt=10s.mat")
#vel_data = vel_data[list(vel_data.keys())[-1]][which_frame,0]

denoised_data = np.load("../data/20241230_xyuv/20241018_ctrl001_z2_xyuv_dt=10s_denoised.npy")
when = np.quantile(denoised_data[:, 0], 0.5)
vel_data = denoised_data[denoised_data[:, 0] == when]
#breakpoint()

##### simulation parameters #####
def concentration(x,y,t, scale_x = 0.12, scale_y = 0.12, scale_t = 0.1, t0 = -0.01, rho = 0.91):
    local_t = max( t - t0, 0)
    gau = 1 - ( min((local_t)/scale_t,1)) * np.exp(-.5 * ((x / scale_x)**2+np.abs(y / scale_y)**2 - 2 * rho * (x / scale_x) * (y / scale_y)))
    gau *= (gau >= 0)
    return gau

def concentration2(x,y,t, scale_x = 0.12, scale_y = 0.12, rho = 0.91,
                   #scale_x2 = 1, scale_y2 = .5, max_high = .0000001, x2_0 = 0., y2_0 = -1.2,
                   scale_x2 = 0.4, scale_y2 = 0.2, max_high = .01, x2_0 = 0.1, y2_0 = -1.,
                   scale_t = 0.1, t0 = -0.01):
    local_t = 100#max( t - t0, 0)
    gau = 1 - ( min((local_t)/scale_t,1)) * np.exp(-.5 * ((x / scale_x)**2+np.abs(y / scale_y)**2 - 2 * rho * (x / scale_x) * (y / scale_y)))
    gau2 = max_high * np.exp(-.5 * ((np.abs(x - x2_0) / scale_x2)**2+np.abs((y-y2_0) / scale_y2)**2))
    gau = gau + gau2
    gau *= (gau >= 0)
    gau = (gau - np.min(gau)) / (np.max(gau) - np.min(gau))
    #breakpoint()
    return gau



dt = 0.001
nt = 300
dx = 0.05
dy = 0.05
x = np.arange(-3, 3, dx)
y = np.arange(-3, 3, dy)
X, Y = np.meshgrid(x, y)
mask = 1 * ((X/1.) ** 2 + Y ** 2 < 2.4 ** 2)
#mask = 1 * (np.logical_and(np.logical_and(X > -2, X < 2), np.logical_and(Y > -2, Y < 2)))

u = np.zeros_like(X)
v = np.zeros_like(X)
p = np.zeros_like(X)
b = np.zeros_like(X)


u, v, p, stress_inner, \
u_save, v_save, pressure_save,\
stress_inner_save = \
            naivesolver_with_extra_hydrostatic(nt = nt, u = u, v = v, 
               dt = dt, dx = dx, 
               dy = dy, mask = mask,
                X = X, Y = Y, hydro_stress = concentration,
                upwind = True,
               rho = 1, nu = 0.6, save_every = 2, save_from =150, #150,
               kwargs = {"tol": 1e-10, "maxit": 1000})
#naivesolver_with_extra_hydrostatic(nt = nt, u = u, v = v, 
#chorin_projection_with_extra_hydrostatic(nt = nt, u = u, v = v,
# plot size
stress_inner[np.logical_not( mask)] = np.nan



###### plotting ####

plt.rcParams.update({'font.size': 16})
fig, axs = plt.subplots(2, 3, figsize=(14, 8))

# plotting actin concentration
axs[0, 0].pcolormesh(X_conc, Y_conc, conc_plot, cmap='viridis')#,  vmin=0, vmax=1)
axs[0, 0].set_xlabel('')
axs[0, 0].set_ylabel('')
axs[0, 0].arrow(1, -0.2, -0.4, 0.4, head_width=0.2, head_length=0.2, fc='black', ec='black')

axs[0, 0].set_title('Exp. actin concentration')

# plotting velocity field
axs[0, 1].quiver(vel_data[::3,1], 
                 vel_data[::3,2], 
                 vel_data[::3,3], 
                 vel_data[::3,4], 
                 scale = 0.5,
                 width = 0.005, 
                 color = "blue")
axs[0, 1].set_title('Exp. velocity field, denoised')

axs[0, 2].set_title('Exp. pressure reconstruction')
axs[0, 2].set_xlabel('')
axs[0, 2].set_ylabel('')



# plotting hydrostatic stress
axs[1, 0].pcolormesh(X, Y, -stress_inner, alpha=0.5, cmap='viridis')
axs[1, 0].set_title('Model stress due to actin')
axs[1, 0].set_xlabel('')
axs[1, 0].set_ylabel('')
axs[1, 0].arrow(.5, -0.5, -0.25, 0.25, head_width=0.2, head_length=0.2, fc='black', ec='black')

#cbar = plt.colorbar(axs[1,0].collections[0])
#cbar.set_label('Hydrostatic stress', rotation=270, labelpad=15)
# plotting velocity field
axs[1, 1].contourf(X, Y, stress_inner-stress_inner, alpha=0.5, cmap='viridis')
axs[1, 1].quiver(X[::5, ::5], 
           Y[::5, ::5], 
           u[::5, ::5], 
           v[::5, ::5], scale = v.std() * 60, color = "blue", width = 0.005) 
'''
axs[1].streamplot(X, Y, u, v, color='blue')
'''
axs[1, 1].set_title('Model velocity field')
axs[1, 1].set_xlabel('')
axs[1, 1].set_ylabel('')

axs[1, 2].pcolormesh(X, Y, p-stress_inner, alpha=0.5, cmap='viridis')
axs[1, 2].set_title('Model dynamic pressure')
axs[1, 2].set_xlabel('')
axs[1, 2].set_ylabel('')

for a in axs.flat:
    a.set_xticks([])
    a.set_yticks([])

plt.tight_layout()
plt.show()
plt.savefig("hydrostatic.png", dpi=500)
plt.close()




#velocity_animation(X,Y,u_save,v_save,pressure_save,stress_inner_save,mask)