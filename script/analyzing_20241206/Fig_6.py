import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import matplotlib.gridspec as gridspec
from matplotlib import colors
from scipy.ndimage import zoom
import os
from tqdm import tqdm

visc_range = [3000, 3000]
for drag_range in tqdm([
                           #[200, 500],
                           #[200, 1000], 
                           #[200, 3000],
                           #[500, 3000],  
                           
                           
                           [2000, 2000],
                            [3000, 3000],
                            [5000, 5000],
                            [2000, 10000],
                            [3000, 10000],
                            [5000, 10000]

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

    res_file = f"./simulations/modelcell2D_maxstress{stress_max}_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_dt{dt}_dx{dx}_tmax{tmax}.npz"
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

    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.03])



    # interpolate a bit
    stressfine  = zoom(stress_ext, 6, order=1)   # 4x resolution, linear
    chifine = zoom(chi.reshape(N, N), 6, order=1)

    Xf = zoom(X2, 6, order=1)
    Yf = zoom(Y2, 6, order=1)

    mask = (chifine > 0.1) 
    stress_masked = np.ma.array(stressfine, mask=mask)

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

    #ax0.set_xticks([])
    #ax0.set_yticks([])

    fig.tight_layout()
    fig.savefig(f"./Fig_6/modelcell2D_maxstress{stress_max}_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_dt{dt}_dx{dx}_N{N}_tmax{tmax}.pdf")


    #breakpoint()


    ######### vel plot as fun of stress ######

    fig, ax = plt.subplots(figsize=(8,5))

    # Move left y-axis and bottom x-axis to center
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Hide the top and right spines
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Set ticks position
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


    for stress_max, color in zip([2000.0, 1000.0, 500.0, 100.0], ["orange", "black", "red", "green"]):
        sim_file = f"./simulations/modelcell2D_maxstress{stress_max}_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_dt{dt}_dx{dx}_tmax{tmax}.npz"
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
        else:
            u_slice = (U2[(N//2+1), :])/1 * (1000 * 60)
        x_slice = X2[0,:]
    
        ax.plot(x_slice-.5, u_slice, label = f"{stress_max/1000} Pa", color = color, linewidth = 2)

    ax.legend()
    ax.set_xlabel('x (mm)')
    ax.xaxis.set_label_coords(0.9, 0.45)
    ax.set_ylabel('u (µm/min)')
    ax.yaxis.set_label_coords(0.4, 0.9)
    #ax.set_ylim([-2, 2])
    fig.tight_layout()
    fig.show()
    fig.savefig(f"./Fig_6/vel_middle_drag{drag_range[0]}-{drag_range[1]}_size{cell_radius}_visc{visc_range[0]}-{visc_range[1]}_dt{dt}_dx{dx}_N{N}_tmax{tmax}.pdf")
