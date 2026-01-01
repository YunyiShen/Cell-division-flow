import pyvista as pv
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from cytoFD.forward.solinoidal_interpolating import solinoidal_interpolating, simple_interpolate


def manual_seed(n = 6, zs = [0.2, 0.25], rs = [0.125]):
    theta = np.linspace(0, 2*3.1415926, num = n+1)
    x, y = np.cos(theta), np.sin(theta)
    x = np.concatenate([x * r for r in rs])
    y = np.concatenate([y * r for r in rs])
    seeds = []
    for z in zs:
    
        seeds.append(np.concatenate((np.column_stack((x, y, x*0 + z)), np.column_stack((x, y, x*0 - z)))) + 0.5)
    seeds = np.concatenate(seeds)
    inside = ((seeds[:,0]-0.5)**2 + (seeds[:,1]-0.5)**2 + (seeds[:,2]-0.5)**2) < 0.5**2
    seeds = seeds[inside]
    return pv.PolyData(seeds)
    
    
    
    
    



def plot_yz_slice_quiver(u, v, w, stress, x, y, z, N = 36, slice_x=None, stride=1, filename="yz_slice.pdf"):
    """
    Plot a Y-Z cross section of pressure and velocity using matplotlib quiver.
    """
    # Reshape flat arrays to (N, N, N)
    u = u.reshape((N, N, N))
    v = v.reshape((N, N, N))
    w = w.reshape((N, N, N))
    stress = stress.reshape((N, N, N))
    x = x.reshape((N, N, N))
    y = y.reshape((N, N, N))
    z = z.reshape((N, N, N))

    # Choose x-slice
    if slice_x is None:
        slice_idx = N // 2
    else:
        slice_idx = np.argmin(np.abs(x[:, 0, 0] - slice_x))

    # Extract Y-Z plane
    y_plane = y[:, :, slice_idx]  # shape (nz, ny)
    z_plane = z[:, :, slice_idx]
    u_plane = v[:, :, slice_idx]     # your saved u array should be reshaped similarly
    v_plane = w[:, :, slice_idx]
    p_plane = stress[:, :, slice_idx]

    # Subsample for clarity
    y_sub = y_plane[::stride, ::stride]
    z_sub = z_plane[::stride, ::stride]
    v_sub = v_plane[::stride, ::stride]
    u_sub = u_plane[::stride, ::stride]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    pressure_img = ax.contourf(y_plane, z_plane, 20.-p_plane, levels=50, cmap="coolwarm")
    
    plt.colorbar(pressure_img, ax=ax, label="Pressure")

    ax.quiver(y_sub, z_sub, u_sub, v_sub, color='k')
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    #ax.set_title(f"Y-Z Slice at x ≈ {x[slice_idx,0,0]:.2f}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved to {filename}")


def create_seed_points_in_box(x_range, y_range, z_range, n_seeds_per_dim=10):
    """
    Create seed points uniformly distributed in a 3D box.
    
    Args:
        x_range, y_range, z_range: tuples (min, max) defining box boundaries.
        n_seeds_per_dim: number of seeds along each axis.
    
    Returns:
        pv.PolyData with seed points.
    """
    x_vals = np.linspace(x_range[0], x_range[1], n_seeds_per_dim)
    y_vals = np.linspace(y_range[0], y_range[1], n_seeds_per_dim)
    z_vals = np.linspace(z_range[0], z_range[1], n_seeds_per_dim)

    X_seed, Y_seed, Z_seed = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    seeds = np.column_stack((X_seed.ravel(), Y_seed.ravel(), Z_seed.ravel()))
    inside = ((seeds[:,0]-0.5)**2 + (seeds[:,1]-0.5)**2 + (seeds[:,2]-0.5)**2) < 0.5**2
    outside = ((seeds[:,0]-0.5)**2 + (seeds[:,1]-0.5)**2) > 0.1**2
    seeds = seeds[np.logical_and(inside, outside, np.abs(seeds[:,1]-0.5)>0.2)]
    #breakpoint()

    return pv.PolyData(seeds)

def subsample_streamline_points(streamlines, step=10):
    # Extract points and vectors
    points = streamlines.points
    vectors = streamlines['vectors']  # or appropriate vector array name

    # Pick every 'step' point
    indices = np.arange(0, points.shape[0], step)

    # Create new polydata with subsampled points and vectors
    subsampled = pv.PolyData(points[indices])
    subsampled['vectors'] = vectors[indices]

    return subsampled

def visualize_streamlines(u, v, w, stress, x, y, z, filename="3Dstreamlines.pdf"):
    N = int(round(np.cbrt(len(x))))
    X = x.reshape((N, N, N))
    Y = y.reshape((N, N, N))
    Z = z.reshape((N, N, N))
    U = u.reshape((N, N, N))
    V = v.reshape((N, N, N))
    W = w.reshape((N, N, N))
    Stress_ = stress.reshape((N, N, N))
    
    # Flatten points and vectors
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    vectors = np.column_stack((U.ravel(), V.ravel(), W.ravel()))

    # Create PyVista StructuredGrid
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (N, N, N)
    grid["vectors"] = vectors

    # Define seed points for streamlines: e.g., a line along one face
    '''
    range = 0.1
    starting = (.5-range, .5+range)
    seed_points = create_seed_points_in_box((.5-0.2, .5+0.2), 
                                            (.5-0.2, .5+0.2), 
                                            (.5-0.2, .5+0.2), 
                                            n_seeds_per_dim=4) #pv.PolyData(seeds)
    '''
    seed_points = manual_seed(n = 8, zs = [0.1, 0.15, 0.2], rs = [0.175,  0.25])
    # Generate streamlines starting at seed points
    streamlines = grid.streamlines_from_source(
        seed_points,
        vectors="vectors",
        integrator_type = 45,
        integration_direction='forward',
        max_time=1000.0,
        initial_step_length=0.05,
        max_steps=1000,
        terminal_speed=1e-8,
        interpolator_type = "point",
        #opacity=0.5
    )
    #breakpoint()
    print(seed_points)
    print(streamlines)
    arrows = subsample_streamline_points(streamlines, 100).glyph(
        orient='vectors',     # Use vector data for arrow orientation (streamline tangent)
        scale=False,          # Set to False to keep uniform arrow size (or True to scale arrows)
        factor=0.1,           # Adjust arrow size
        
        geom=pv.Arrow()       # Use arrow glyph geometry

    )
    sphere = pv.Sphere(center=(0.5, 0.5, 0.5), radius=0.5, theta_resolution=30, phi_resolution=30)
    sphere["scalar"] = np.full(sphere.n_points, np.nanmin(stress))
    # pancake 
    structured = pv.StructuredGrid()
    structured.points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    structured.dimensions = X.shape
    structured["stress"] = Stress_.ravel()

    # Plot
    plotter = pv.Plotter()
    plotter.add_mesh(streamlines.tube(radius=0.0025), color="#f3a2c6", label="Streamlines")
    #plotter.add_mesh(arrows, color='red', label='Flow direction arrows')
    contours = structured.contour(isosurfaces=np.linspace(0, np.nanmax(stress), 20))
    contour_actor = plotter.add_mesh(contours, cmap="viridis", clim=[0,np.nanmax(stress)], opacity=0.1, label="Actin stress (Pa)", show_scalar_bar=False)
    plotter.scalar_bar_args = {
        "title": "",
        "vertical": True,
        "position_x": 0.85,
        "position_y": 0.1,
        "width": 0.08,
        "height": 0.75,
        "label_font_size": 12,
    }

    # Then add scalar bar — this will use the current active scalars
    plotter.add_scalar_bar(**plotter.scalar_bar_args)
    
    
    #plotter.add_mesh(sphere, color='lightgray', style='wireframe', opacity=0.2, scalars="scalar", cmap="viridis", clim=[0, 20], show_scalar_bar=False)
    #plotter.show_axes()
    #plotter.add_axes()
    #plotter.show_axes()
    #plotter.show_grid()
    plotter.camera_position = [
        (2.5, 2.5, 2.5),   # camera location
        (0.5, 0.5, 0.5),   # focal point
        (0, 0, 1),         # view-up vector (Z-up)
    ]
    #plotter.camera.parallel_projection = True
    plotter.camera.azimuth = 10
    plotter.camera.elevation = -5
    '''
    plotter.show_grid(xlabel='X', ylabel='Y', zlabel='Z', n_xlabels=3,
        n_ylabels=3,
        n_zlabels=3,
        font_size=50,
    )
    '''
    plotter.show_grid(xlabel='X', ylabel='Y', zlabel='Z', n_xlabels=3,
        n_ylabels=3,
        n_zlabels=3,
        font_size=0,
    )
    
    #plotter.show_grid(False)
    #plotter.show(screenshot=filename)
    plotter.show(auto_close=False)  # Keep plot open for saving
    plotter.screenshot(filename.replace(".pdf", ".png"), scale=2)
    plotter.close()

    # Optional: Convert to PDF
    if filename.endswith(".pdf"):
        from PIL import Image
        im = Image.open(filename.replace(".pdf", ".png"))
        im.save(filename)

visc = [4000, 20000]
refine = 2
simures = np.load(f"./simulations/modelcell3D_Stokes_maxstress1000.0_drag0_size0.5_visc{visc[0]}-{visc[1]}_dt0.05_dx0.03225806451612903_tmax60_interpolated{refine}.npz")
#breakpoint()
'''
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

xf, yf, zf, stressf = simple_interpolate(stress, x, y, z, refine=2)
#breakpoint()
xff, yff, zff, uf, vf, wf, chif,_ = solinoidal_interpolating(x, y, z, u, v, w, chi, refine = 2)

breakpoint()
stressf[chif>chi_thr] = np.nan
uf[chif>0.5*chi_thr] = 0
vf[chif>0.5*chi_thr] = 0
wf[chif>0.5*chi_thr] = 0



stress[chi>chi_thr] = np.nan
#breakpoint()
'''
x, y, z = simures['x'], simures['y'], simures['z']
u, v, w = simures['u'], simures['v'], simures['w']
stress, chi = simures['stress'], simures['chi']
N = simures['N']

chi_thr = 0.35
stress[chi>chi_thr] = np.nan
u[chi>0.5*chi_thr] = 0
v[chi>0.5*chi_thr] = 0
w[chi>0.5*chi_thr] = 0



plot_yz_slice_quiver(u * 1000 * 60, 
                      v * 1000 * 60, 
                      w * 1000 * 60, stress, x, 
                                y, 
                                z, N = N, slice_x=None, stride=1, filename=f"yz_slice_interpolated{refine}.pdf")

visualize_streamlines(u * 1000 * 60, 
                      v * 1000 * 60, 
                      w * 1000 * 60, 
                      stress/1000, x, 
                                y, 
                                z,
                                filename = f"./Fig_6/modelcell3D_Stokes_maxstress1000.0_drag0_size0.5_visc{visc[0]}-{visc[1]}_dt0.05_dx0.03225806451612903_tmax60_interpolated{refine}.pdf"
                                )