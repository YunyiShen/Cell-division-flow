import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import matplotlib.gridspec as gridspec


########## time series plot ################
simulation = np.load("./modelcell2Dmax20_norot.npz")
u, v, p, p_ext_save, t = simulation['u'], simulation['v'], simulation['p'], simulation['p_ext'], simulation['t']
X, Y, N = simulation['x'], simulation['y'], simulation['N']
nx, ny = N, N

n_frame = len(t)
n_plot_time_series = 5
u_ts = u[::(n_frame//n_plot_time_series)]
v_ts = v[::(n_frame//n_plot_time_series)]
p_ext_ts = p_ext_save[::(n_frame//n_plot_time_series)]
t_ts = t[::(n_frame//n_plot_time_series)]

#x,y = np.meshgrid(np.linspace(0, 2, n_grid), np.linspace(0, 2, n_grid))

fig = plt.figure(figsize=(26, 5))
gs = gridspec.GridSpec(1, 7, width_ratios=[1, 1, 1, 1, 1, 1, 0.03])

step = 7
scale = 60 * v.std()

for i in range(6):

    U = u_ts[i]
    V = v_ts[i]
    X2 = X.reshape((nx, ny))
    Y2 = Y.reshape((nx, ny))
    U2 = U.reshape((nx, ny))
    V2 = V.reshape((nx, ny))
    p_ext = p_ext_ts[i].reshape((nx, ny))

    #breakpoint()
    ax0 = fig.add_subplot(gs[i])
    cf = ax0.contourf(X2, Y2, np.nanmax(p_ext_save) - p_ext, 
                levels=np.linspace(0, np.nanmax(p_ext_save), 20), cmap='viridis')
    
    ax0.quiver(
            X2[::step, ::step], Y2[::step, ::step],
            U2[::step, ::step], V2[::step, ::step],
            scale=10,
            width=0.005,
            color='black'
        )
    ax0.set_title(f"model t={t_ts[i]}")
    ax0.set_aspect('equal')
    ax0.set_xticks([])
    ax0.set_yticks([])

cax1 = fig.add_subplot(gs[6])
cbar = fig.colorbar(cf, cax=cax1)
cbar.set_label('Modeled actin stress') 

fig.tight_layout()
fig.savefig("timeseries.pdf")
plt.close()

########################################################
############# anime would be fun #######################
########################################################
from matplotlib.animation import FuncAnimation, PillowWriter
from celluloid import Camera

# Load data
simulation = np.load("./modelcell2Dmax20_norot.npz")
u_ts, v_ts, p, p_ext_ts, t_ts = simulation['u'], simulation['v'], simulation['p'], simulation['p_ext'], simulation['t']
X, Y, N = simulation['x'], simulation['y'], simulation['N']
nx, ny = N, N

X2 = X.reshape((nx, ny))
Y2 = Y.reshape((nx, ny))

p_ext_max = np.nanmax(p_ext_ts)

#fig, ax = plt.subplots(figsize=(6,6))
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.03])
camera = Camera(fig)

ax0 = fig.add_subplot(gs[0])
cax1 = fig.add_subplot(gs[1])


fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.03])  
camera = Camera(fig)

# Create the axes only once
ax0 = fig.add_subplot(gs[0]) 
cax1 = fig.add_subplot(gs[1]) 

# Loop to generate each frame
for frame in range(len(t_ts)):
    #ax0.clear()
    #cax1.clear()

    p_ext = p_ext_ts[frame].reshape((nx, ny))
    U2 = u_ts[frame].reshape((nx, ny))
    V2 = v_ts[frame].reshape((nx, ny))

    # Contour plot for this frame
    cf = ax0.contourf(X2, Y2, np.nanmax(p_ext_ts) - p_ext, 
                levels=np.linspace(0, np.nanmax(p_ext_ts), 20), cmap='viridis')

    # Quiver plot for this frame
    ax0.quiver(X2[::step, ::step], Y2[::step, ::step], U2[::step, ::step], V2[::step, ::step],
               scale=10, width=0.005, color='black')

    # Set title and remove axis ticks
    #ax0.set_title(f"Model t={t_ts[frame]:.2f}")
    ax0.set_aspect('equal')
    ax0.set_xticks([])  # Remove x-axis ticks
    ax0.set_yticks([])  # Remove y-axis ticks

    # Create the colorbar for this frame
    cbar = fig.colorbar(cf, cax=cax1)
    cbar.set_label('Modeled Actin Stress')

    # Capture the frame with the Camera
    camera.snap()

# After the loop, generate the animation
animation = camera.animate()

# Save the animation as a video (with `writer='ffmpeg'`)
animation.save("velocity_video_celluloid.mp4", writer='ffmpeg', fps=20)

# Finally show the plot (after saving)
plt.show()

# Close the figure after showing
plt.close()





##########################################################
############# compare to real data kinda plot ################
##########################################################

### load simulation ###
simulation = np.load("./modelcell2Dmax20.npz")
u_ts, v_ts, p, p_ext_ts, t_ts = simulation['u'], simulation['v'], simulation['p'], simulation['p_ext'], simulation['t']
X, Y, N = simulation['x'], simulation['y'], simulation['N']
nx, ny = N, N

### load experimental data ### 

#### experimental data #####
#### concentration data ####
conc_data = scipy.io.loadmat('../../newdata/xyuv-cen_files/20241206_ctrl_GFP_z2_raw.mat')
conc_data = conc_data[list(conc_data.keys())[-1]]
n_frames = conc_data.shape[0]
which_frame = n_frames//2
conc_plot = conc_data[which_frame][0]

#### velocity field ####
denoised_data = np.load("./res/20241206_inducing_res_spVel.npz")['denoised']
vel_data = denoised_data[which_frame]
X_mesh = np.load("./res/20241206_inducing_res_spVel.npz")['X_mesh']


#conc_plot = (conc_plot-np.min(conc_plot)) / (np.max(conc_plot) - np.min(conc_plot))
X_range_conc = np.linspace(-3,3, conc_plot.shape[0])
Y_range_conc = np.linspace(-3,3, conc_plot.shape[1])
X_conc, Y_conc = np.meshgrid(X_range_conc, Y_range_conc)



##### plot ####
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.03])

ax0 = fig.add_subplot(gs[0, 0])
ax0.quiver(X_mesh[::4,0], 
                 X_mesh[::4,1], 
                 vel_data[::4,0], 
                 vel_data[::4,1], 
                 scale = 0.2,
                 width = 0.005, 
                 color = "black")
ax0.set_title("velocity")
ax0.set_aspect('equal')
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_ylabel("Experimental")

### actin conc ###
ax0 = fig.add_subplot(gs[0, 1])
#breakpoint()
from matplotlib import colors
norm = colors.PowerNorm(gamma=1.3, vmin=conc_plot.min(), vmax=conc_plot.max())
cf = ax0.pcolormesh(X_conc[::5, ::5], Y_conc[::5, ::5], conc_plot[::5, ::5],
                #levels= 20,#np.linspace(0, np.nanmax(conc_plot), 20), 
                norm=norm, 
                cmap='viridis')
ax0.set_title("actin")
ax0.set_aspect('equal')
ax0.set_xticks([])
ax0.set_yticks([])
ax0.arrow(1., 0.8, -0.4, -0.4, 
                head_width=0.2, 
                head_length=0.2, 
                ec='black', fc='saddlebrown')


##### overlay #####
ax0 = fig.add_subplot(gs[0, 2])
norm = colors.PowerNorm(gamma=1.1, vmin=conc_plot.min(), vmax=conc_plot.max())
X_range_conc = np.linspace(X_mesh[:,0].min(), X_mesh[:,0].max(), conc_plot.shape[0])
Y_range_conc = np.linspace(X_mesh[:,1].min(), X_mesh[:,1].max(), conc_plot.shape[1])
X_conc, Y_conc = np.meshgrid(X_range_conc, Y_range_conc)

cf = ax0.pcolormesh(X_conc[::5, ::5], Y_conc[::5, ::5], conc_plot[::5, ::5],
                #levels= 20,#np.linspace(0, np.nanmax(conc_plot), 20), 
                norm=norm, 
                cmap='viridis')

ax0.quiver(X_mesh[::4,0], 
                 X_mesh[::4,1], 
                 vel_data[::4,0], 
                 vel_data[::4,1], 
                 scale = 0.2,
                 width = 0.005, 
                 color = "black")
ax0.set_title("overlay")
ax0.set_aspect('equal')
ax0.set_xticks([])
ax0.set_yticks([])

cax1 = fig.add_subplot(gs[0, 3])
cbar = fig.colorbar(cf, cax=cax1)
cbar.set_label('Actin concentration') 

####### simulations ######

U = u_ts[-1]
V = v_ts[-1]
X2 = X.reshape((nx, ny))
Y2 = Y.reshape((nx, ny))
U2 = U.reshape((nx, ny))
V2 = V.reshape((nx, ny))
p_ext = p_ext_ts[-1].reshape((nx, ny))


ax0 = fig.add_subplot(gs[1, 0])
ax0.quiver(
            X2[::step, ::step], Y2[::step, ::step],
            U2[::step, ::step], V2[::step, ::step],
            scale=10,
            width=0.005,
            color='black'
        )
ax0.set_xticks([])
ax0.set_yticks([])
ax0.set_ylabel("Simulation")

ax0 = fig.add_subplot(gs[1, 1])
cf = ax0.contourf(X2, Y2, np.nanmax(p_ext_save) - p_ext, 
            levels=np.linspace(0, np.nanmax(p_ext_save),  20), 
            cmap='viridis')
ax0.arrow(1.38, 1.38, -0.18, -0.18, 
                head_width=0.075, 
                head_length=0.075, 
                ec='black', fc='saddlebrown')
ax0.set_xticks([])
ax0.set_yticks([])


## overlay 
ax0 = fig.add_subplot(gs[1, 2])
cf = ax0.contourf(X2, Y2, np.nanmax(p_ext_save) - p_ext, 
            levels=np.linspace(0, np.nanmax(p_ext_save),  20), 
            cmap='viridis')
ax0.quiver(
            X2[::step, ::step], Y2[::step, ::step],
            U2[::step, ::step], V2[::step, ::step],
            scale=10,
            width=0.005,
            color='black'
        )
ax0.set_xticks([])
ax0.set_yticks([])
cax1 = fig.add_subplot(gs[1, 3])
cbar = fig.colorbar(cf, cax=cax1)
cbar.set_label('Model stress due to actin') 

#ax0.set_xticks([])
#ax0.set_yticks([])







fig.tight_layout()
fig.savefig("experiments.pdf")

