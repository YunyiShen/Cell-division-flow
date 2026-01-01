import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import matplotlib.gridspec as gridspec
from matplotlib import colors
from scipy.ndimage import zoom
import os
from tqdm import tqdm

from matplotlib.colors import LinearSegmentedColormap

cmap = LinearSegmentedColormap.from_list(
    "my_cmap",
    ["cyan", "magenta"]  # start → end color
)


fig_u, ax_u = plt.subplots(figsize=(8,5))
fig_v, ax_v = plt.subplots(figsize=(8,5))
# Move left y-axis and bottom x-axis to center
ax_u.spines['left'].set_position('center')
ax_u.spines['bottom'].set_position('center')

# Hide the top and right spines
ax_u.spines['right'].set_color('none')
ax_u.spines['top'].set_color('none')

# Set ticks position
ax_u.xaxis.set_ticks_position('bottom')
ax_u.yaxis.set_ticks_position('left')
    
# Move left y-axis and bottom x-axis to center
ax_v.spines['left'].set_position('center')
ax_v.spines['bottom'].set_position('center')

# Hide the top and right spines
ax_v.spines['right'].set_color('none')
ax_v.spines['top'].set_color('none')

# Set ticks position
ax_v.xaxis.set_ticks_position('bottom')
ax_v.yaxis.set_ticks_position('left')
colors = ["#f3a2c6", "#a1b0c5", "#00afb2"]#, "#003854"]

which_biology = "bulk"
dt = 1
N = 101#81
tmax = 600
chi_thr = 0.2
aspect_ratio = 1.0
stress_max = 1000.0
drag_range = [0, 0]
visc_range = [10000, 10000]

sizes = np.linspace(0.1/2, 1./2, num = 20)[3:]

for color, size_range in tqdm(zip(colors, [sizes[0], sizes[len(sizes)//2], sizes[-1]]
                           )):

    
    cell_radius = size_range
    

    res_file = f"./simulations/{which_biology}/modelcell2D_Stokes_maxstress{stress_max}_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_aspectratio{aspect_ratio}_dt{dt}_N{N}_tmax{tmax}.npz"
    if not os.path.exists(res_file):
        print(f"{res_file} does not exist")
        continue
    
    simulation = np.load(res_file)
    #breakpoint()
    u, v, p, stress_ext_save, t = simulation['u'], simulation['v'], simulation['p'], simulation['stress_ext']/1000, simulation['t']
    chi = simulation['chi']
    u[:,chi > 0.5*chi_thr] = np.nan
    v[:, chi > 0.5*chi_thr] = np.nan

    X, Y, N = simulation['x'], simulation['y'], N
    #stress_ext_save[:, chi > 0.2] = np.nan

    #u[chi > 0.8] = np.nan
    nx, ny = N, N
    n_frame = len(t)
    n_plot_time_series = 5
    u_ts = u[::(n_frame//n_plot_time_series)]
    v_ts = v[::(n_frame//n_plot_time_series)]
    stress_ext_ts = stress_ext_save[::(n_frame//n_plot_time_series)]
    #breakpoint()
    t_ts = t[::(n_frame//n_plot_time_series)]

    U = u_ts[-1]
    V = v_ts[-1]
    X2 = X.reshape((nx, ny))
    Y2 = Y.reshape((nx, ny))
    U2 = U.reshape((nx, ny))
    V2 = V.reshape((nx, ny))
    stress_ext = stress_ext_ts[-2].reshape((nx, ny))

    #### three panel plot ###

    fig = plt.figure(figsize=(3*3.5, 3.0))
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 0.03, 1, 0.03])



    # interpolate a bit
    stressfine  = zoom(stress_ext, 6, order=1)   # 4x resolution, linear
    chifine = zoom(chi.reshape(N, N), 6, order=1)

    Xf = zoom(X2, 6, order=1)
    Yf = zoom(Y2, 6, order=1)

    mask = (chifine > 0.1) 
    stress_masked = np.ma.array(stressfine, mask=mask)
    vel_size = np.linalg.norm(np.stack((U2, V2), axis = 0), axis = 0)
    vel_sizef = zoom(vel_size, 6, order = 1)
    vel_masked = np.ma.array(vel_sizef, mask = mask)

    ax0 = fig.add_subplot(gs[0, 0])
    cf = ax0.contourf(Xf, Yf, stress_masked, 
                  corner_mask=True, antialiased=True,
            levels=np.linspace(0, stress_max/1000,  21), 
            cmap='viridis')

    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_axis_off()
    ax0.set_aspect('equal')
    ax0.set_title("stress")


    ## overlay 
    ax0 = fig.add_subplot(gs[0, 1])
    cf = ax0.contourf(Xf, Yf, stress_masked, 
                  corner_mask=True, antialiased=True,
                  vmin=0,
                  vmax=stress_max/1000,
            levels=np.linspace(0, stress_max/1000,  21), 
            cmap='viridis')
    #breakpoint()

    ax0.quiver(
            X2[::4, ::4], Y2[::4, ::4],
            U2[::4, ::4], V2[::4, ::4],
            #scale=10,
            width=0.01,
            color='black' if which_biology == "bulk" else "white"
        )
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_axis_off()

    ax0.set_aspect('equal')
    ax0.set_title("overlay")

    import matplotlib





    cax1 = fig.add_subplot(gs[0, 2])
    #cbar = fig.colorbar(cf, cax=cax1, extend="both")
    norm = matplotlib.colors.Normalize(vmin=0,
                  vmax=stress_max/1000)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])  # required by colorbar
    cbar = fig.colorbar(sm, cax=cax1, ticks=np.linspace(0, stress_max/1000, 6))
    cbar.set_label('Contractile stress by actomyosin (Pa)') 

    #ax0.set_xticks([])
    #ax0.set_yticks([])
    
    ax0 = fig.add_subplot(gs[0, 3])
    cf = ax0.contourf(Xf, Yf, vel_masked * 1000 * 60, 
                  corner_mask=True, antialiased=True,
                  vmin=0,
                  vmax=np.nanmax(vel_masked * 1000 * 60),
            levels=np.linspace(0, np.nanmax(vel_masked * 1000 * 60),  21), 
            #linewidths=0,
            cmap=cmap)
    
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.set_axis_off()

    ax0.set_aspect('equal')
    ax0.set_title("velocity")
    
    cax1 = fig.add_subplot(gs[0, 4])
    #cbar = fig.colorbar(cf, cax=cax1, extend="both")
    norm = matplotlib.colors.Normalize(vmin=0,
                  vmax=np.nanmax(vel_masked * 1000 * 60))
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required by colorbar
    cbar = fig.colorbar(sm, cax=cax1, ticks=np.linspace(0, np.nanmax(vel_masked * 1000 * 60), 6))
    cbar.set_label('Velocity (µm/min)') 

    fig.tight_layout()
    fig.savefig(f"./Figs/flow/modelcell2D_Stokes_{which_biology}_maxstress{stress_max}_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_aspectratio{aspect_ratio}_dt{dt}_N{N}_tmax{tmax}.pdf")


    #breakpoint()


    ######### vel plot as fun of stress ######

    

    


    

    if N%2 == 0:
        u_slice = (U2[N//2, :] + U2[(N//2+1), :])/2 * (1000 * 60)
        v_slice = (V2[:, N//2] + V2[:, (N//2+1)])/2 * (1000 * 60)
    else:
        u_slice = (U2[(N//2+1), :])/1 * (1000 * 60)
        v_slice = (V2[:, (N//2+1)])/1 * (1000 * 60)
    x_slice = X2[0,:]
    y_slice = Y2[:,0]
    
    ax_u.plot(x_slice-cell_radius, u_slice, label = f"{cell_radius} mm", color = color, linewidth = 4)
    ax_v.plot(y_slice-cell_radius, v_slice, label = f"{cell_radius} mm", color = color, linewidth = 4)

ax_u.legend()
ax_u.set_xlabel('x (mm)')
ax_u.xaxis.set_label_coords(0.9, 0.45)
ax_u.set_ylabel('u (µm/min)')
ax_u.yaxis.set_label_coords(0.4, 0.9)
    
#ax.set_ylim([-2, 2])
fig_u.tight_layout()
fig_u.show()
fig_u.savefig(f"./Figs/{which_biology}_vel_middle_Stokes_maxstress{1000}_drag{drag_range[0]}-{drag_range[1]}_size_changing_visc{visc_range[0]}-{visc_range[1]}_aspectratio{aspect_ratio}_dt{dt}_N{N}_tmax{tmax}.pdf")
    
#### v plot
ax_v.legend()
ax_v.set_xlabel('y (mm)')
ax_v.xaxis.set_label_coords(0.9, 0.45)
ax_v.set_ylabel('v (µm/min)')
ax_v.yaxis.set_label_coords(0.4, 0.9)
    
#ax.set_ylim([-2, 2])
fig_v.tight_layout()
fig_v.show()
fig_v.savefig(f"./Figs/{which_biology}_vel_yaxis_middle_Stokes_maxstress{1000}_drag{drag_range[0]}-{drag_range[1]}_size_changing_visc{visc_range[0]}-{visc_range[1]}_aspectratio{aspect_ratio}_dt{dt}_N{N}_tmax{tmax}.pdf")
  
    
'''
max velocity as a function of cell size
'''


    
fig, ax = plt.subplots(figsize=(8,5))
velnorm = []
uslice = []
vslice = []

for cell_radius in tqdm(sizes):
    res_file = f"./simulations/{which_biology}/modelcell2D_Stokes_maxstress{stress_max}_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_aspectratio{aspect_ratio}_dt{dt}_N{N}_tmax{tmax}.npz"
    if not os.path.exists(res_file):
        print(f"{res_file} does not exist")
        continue
    
    simulation = np.load(res_file)
    u, v, p, stress_ext_save, t = simulation['u'], simulation['v'], simulation['p'], simulation['stress_ext']/1000, simulation['t']
    chi = simulation['chi']
    u[:,chi > 0.5*chi_thr] = np.nan
    v[:, chi > 0.5*chi_thr] = np.nan
    
    U = u[-1]
    V = v[-1]
    X2 = X.reshape((nx, ny))
    Y2 = Y.reshape((nx, ny))
    U2 = U.reshape((nx, ny))
    V2 = V.reshape((nx, ny))
    
    if N%2 == 0:
        u_slice = (U2[N//2, :] + U2[(N//2+1), :])/2 * (1000 * 60)
        v_slice = (V2[:, N//2] + V2[:, (N//2+1)])/2 * (1000 * 60)
    else:
        u_slice = (U2[(N//2+1), :])/1 * (1000 * 60)
        v_slice = (V2[:, (N//2+1)])/1 * (1000 * 60)
    
    
    uslice.append(np.nanmax(u_slice))
    vslice.append(np.nanmax(v_slice))
    
    velnorm.append(np.nanmax(np.linalg.norm(np.stack((u[-1], v[-1]), axis = 0), axis = 0)) * 1000 * 60)
#breakpoint()  
ax.scatter(sizes, velnorm)
ax.plot(sizes, velnorm, label = "global")

ax.scatter(sizes, uslice)
ax.plot(sizes, uslice, label = "x slice")

ax.scatter(sizes, vslice)
ax.plot(sizes, vslice, label = "y slice")

ax.legend()
ax.set_xlabel('cell radius (mm)')
ax.set_ylabel('maximum velocity (µm/min)')
ax.set_ylim(0, np.max(velnorm)*1.1)


fig.tight_layout()
fig.show()
fig.savefig(f"./Figs/{which_biology}_max_vel_Stokes_maxstress{1000}_drag{drag_range[0]}-{drag_range[1]}_size_changing_visc{visc_range[0]}-{visc_range[1]}_aspectratio{aspect_ratio}_dt{dt}_N{N}_N{N}_tmax{tmax}.pdf")
