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


visc_range = [500, 500]
for drag_range in tqdm([
                           #[200, 500],
                           #[200, 1000], 
                           #[200, 3000],
                           #[500, 3000],  
                           
                           
                            #[2000, 2000],
                            #[3000, 3000],
                            #[5000, 5000],
                            #[2000, 10000],
                            #[3000, 10000],
                            #[5000, 10000]
                            
                            
                            [20000, 20000],
                            [30000, 30000],
                            [50000, 50000],
                            [20000, 100000],
                            [30000, 100000],
                            [50000, 100000]
                            
                           #[500, 5000],
                           #[1000, 5000],
                           
                           ]):

    stress_max = 1000.0
    cell_radius = 0.5
    dt = 0.004
    dx =0.01098901098901099 #0.012345679012345678
    N = 91#81
    tmax = 600
    chi_thr = 0.2

    res_file = f"./simulations/modelcell2D_Stokes_maxstress{stress_max}_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_dt{dt}_dx{dx}_tmax{tmax}.npz"
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
    vel_size = np.linalg.norm(np.stack((U2, V2), axis = 0), axis = 0)
    #breakpoint()
    #### three panel plot ###

    fig = plt.figure(figsize=(3*3.5, 3.0))
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 1, 0.03, 1, 0.03])



    # interpolate a bit
    stressfine  = zoom(stress_ext, 6, order=1)   
    chifine = zoom(chi.reshape(N, N), 6, order=1)
    vel_sizef = zoom(vel_size, 6, order = 1)

    Xf = zoom(X2, 6, order=1)
    Yf = zoom(Y2, 6, order=1)

    mask = (chifine > 0.1) 
    stress_masked = np.ma.array(stressfine, mask=mask)
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
            color='black'
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

    
        ## vel 
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
    
    
    
    
    
    
    
    #ax0.set_xticks([])
    #ax0.set_yticks([])

    fig.tight_layout()
    fig.savefig(f"./Fig_6/modelcell2D_Stokes_maxstress{stress_max}_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_dt{dt}_dx{dx}_N{N}_tmax{tmax}.pdf")


    #breakpoint()


    ######### vel plot as fun of stress ######

    fig, ax = plt.subplots(figsize=(8,5))
    fig_v, ax_v = plt.subplots(figsize=(8,5))

    # Move left y-axis and bottom x-axis to center
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Hide the top and right spines
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Set ticks position
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    # Move left y-axis and bottom x-axis to center
    ax_v.spines['left'].set_position('center')
    ax_v.spines['bottom'].set_position('center')

    # Hide the top and right spines
    ax_v.spines['right'].set_color('none')
    ax_v.spines['top'].set_color('none')

    # Set ticks position
    ax_v.xaxis.set_ticks_position('bottom')
    ax_v.yaxis.set_ticks_position('left')


    for stress_max, color in zip([2000.0, 1000.0, 500.0, 100.0], ["#f3a2c6", "#a1b0c5", "#00afb2", "#003854"]):
        sim_file = f"./simulations/modelcell2D_Stokes_maxstress{stress_max}_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_dt{dt}_dx{dx}_tmax{tmax}.npz"
        if not os.path.exists(sim_file):
            print(f"{sim_file} does not exist")
            continue
        simulation = np.load(sim_file)
        #breakpoint()
        u, v, p, stress_ext_save, t = simulation['u'], simulation['v'], simulation['p'], simulation['stress_ext']/1000, simulation['t']
        chi = simulation['chi']
        u[:,chi > 0.5*chi_thr] = np.nan
        v[:, chi > 0.5*chi_thr] = np.nan

        X, Y, N = simulation['x'], simulation['y'], int(1/dx)
        #stress_ext_save[:, chi > 0.2] = np.nan

        #u[chi > 0.8] = np.nan
        nx, ny = N, N
        n_frame = len(t)
        n_plot_time_series = 5
        u_ts = u[::(n_frame//n_plot_time_series)]
        v_ts = v[::(n_frame//n_plot_time_series)]
        stress_ext_ts = stress_ext_save[::(n_frame//n_plot_time_series)]
        t_ts = t[::(n_frame//n_plot_time_series)]

        U = u_ts[-1]
        V = v_ts[-1]
        X2 = X.reshape((nx, ny))
        Y2 = Y.reshape((nx, ny))
        U2 = U.reshape((nx, ny))
        V2 = V.reshape((nx, ny))
        stress_ext = stress_ext_ts[-1].reshape((nx, ny))

        if N%2 == 0:
            u_slice = (U2[N//2, :] + U2[(N//2+1), :])/2 * (1000 * 60)
            v_slice = (V2[:, N//2] + V2[:, (N//2+1)])/2 * (1000 * 60)
        else:
            u_slice = (U2[(N//2+1), :])/1 * (1000 * 60)
            v_slice = (V2[:, (N//2+1)])/1 * (1000 * 60)
        x_slice = X2[0,:]
        y_slice = Y2[:,0]
    
        ax.plot(x_slice-.5, u_slice, label = f"{stress_max/1000} Pa", color = color, linewidth = 4)
        ax_v.plot(y_slice-.5, v_slice, label = f"{stress_max/1000} Pa", color = color, linewidth = 4)

    ax.legend()
    ax.set_xlabel('x (mm)')
    ax.xaxis.set_label_coords(0.9, 0.45)
    ax.set_ylabel('u (µm/min)')
    ax.yaxis.set_label_coords(0.4, 0.9)
    
    #ax.set_ylim([-2, 2])
    fig.tight_layout()
    fig.show()
    fig.savefig(f"./Fig_6/vel_middle_Stokes_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_dt{dt}_dx{dx}_N{N}_tmax{tmax}.pdf")
    
    #### v plot
    ax_v.legend()
    ax_v.set_xlabel('y (mm)')
    ax_v.xaxis.set_label_coords(0.9, 0.45)
    ax_v.set_ylabel('v (µm/min)')
    ax_v.yaxis.set_label_coords(0.4, 0.9)
    
    #ax.set_ylim([-2, 2])
    fig_v.tight_layout()
    fig_v.show()
    fig_v.savefig(f"./Fig_6/vel_yaxis_middle_Stokes_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_dt{dt}_dx{dx}_N{N}_tmax{tmax}.pdf")
  
    
    ######### v 
    
    
    
