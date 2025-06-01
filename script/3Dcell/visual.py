import pyvista as pv
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def plot_yz_slice_quiver(u, v, w, p, x, y, z, N = 36, slice_x=None, stride=1, filename="yz_slice.pdf"):
    """
    Plot a Y-Z cross section of pressure and velocity using matplotlib quiver.
    """
    # Reshape flat arrays to (N, N, N)
    u = u.reshape((N, N, N))
    v = v.reshape((N, N, N))
    w = w.reshape((N, N, N))
    p = p.reshape((N, N, N))
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
    p_plane = p[:, :, slice_idx]

    # Subsample for clarity
    y_sub = y_plane[::stride, ::stride]
    z_sub = z_plane[::stride, ::stride]
    v_sub = v_plane[::stride, ::stride]
    u_sub = u_plane[::stride, ::stride]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    pressure_img = ax.contourf(y_plane, z_plane, 20.-p_plane, levels=50, cmap="coolwarm")
    
    plt.colorbar(pressure_img, ax=ax, label="Pressure")

    ax.quiver(y_sub, z_sub, u_sub, v_sub, color='k', scale=5)
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
    inside = ((seeds[:,0]-1)**2 + (seeds[:,1]-1)**2 + (seeds[:,2]-1)**2) < 1
    seeds = seeds[inside]

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

def visualize_streamlines(u, v, w, p, x, y, z, filename="3Dstreamlines.pdf"):
    N = int(round(np.cbrt(len(x))))
    X = x.reshape((N, N, N))
    Y = y.reshape((N, N, N))
    Z = z.reshape((N, N, N))
    U = u.reshape((N, N, N))
    V = v.reshape((N, N, N))
    W = w.reshape((N, N, N))
    P = p.reshape((N, N, N))

    # Flatten points and vectors
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    vectors = np.column_stack((U.ravel(), V.ravel(), W.ravel()))

    # Create PyVista StructuredGrid
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = (N, N, N)
    grid["vectors"] = vectors

    # Define seed points for streamlines: e.g., a line along one face
    range = 0.45
    starting = (1-range, 1.+range)
    seed_points = create_seed_points_in_box(starting, 
                                            starting, 
                                            starting, 
                                            n_seeds_per_dim=3) #pv.PolyData(seeds)

    # Generate streamlines starting at seed points
    streamlines = grid.streamlines_from_source(
        seed_points,
        vectors="vectors",
        integration_direction='both',
        max_time=100.0,
        initial_step_length=0.5,
        max_steps=1000,
        terminal_speed=1e-2
    )
    arrows = subsample_streamline_points(streamlines, 100).glyph(
        orient='vectors',     # Use vector data for arrow orientation (streamline tangent)
        scale=False,          # Set to False to keep uniform arrow size (or True to scale arrows)
        factor=0.1,           # Adjust arrow size
        
        geom=pv.Arrow()       # Use arrow glyph geometry

    )
    sphere = pv.Sphere(center=(1, 1, 1), radius=1, theta_resolution=30, phi_resolution=30)
    sphere["scalar"] = np.full(sphere.n_points, 20)
    # pancake 
    structured = pv.StructuredGrid()
    structured.points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    structured.dimensions = X.shape
    structured["stress"] = np.nanmax(P) - P.ravel()

    # Plot
    plotter = pv.Plotter()
    plotter.add_mesh(streamlines.tube(radius=0.005), color="blue", label="Streamlines")
    #plotter.add_mesh(arrows, color='red', label='Flow direction arrows')
    contours = structured.contour(isosurfaces=np.linspace(0, 20, 20))
    contour_actor = plotter.add_mesh(contours, cmap="viridis", clim=[0,20], opacity=0.4, label="Actin stress", show_scalar_bar=False)
    plotter.scalar_bar_args = {
        "title": "Actin stress",
        "vertical": True,
        "position_x": 0.85,
        "position_y": 0.1,
        "width": 0.08,
        "height": 0.8,
        "label_font_size": 12,
    }

    # Then add scalar bar — this will use the current active scalars
    plotter.add_scalar_bar(**plotter.scalar_bar_args)
    
    
    plotter.add_mesh(sphere, color='lightgray', style='wireframe', opacity=0.5, scalars="scalar", cmap="viridis", clim=[0, 20], show_scalar_bar=False)
    #plotter.show_axes()
    #plotter.add_axes()
    #plotter.show_axes()
    #plotter.show_grid()
    plotter.show_grid(xlabel='X', ylabel='Y', zlabel='Z')
    #plotter.show(screenshot=filename)
    plotter.show(auto_close=False)  # Keep plot open for saving
    plotter.screenshot(filename.replace(".pdf", ".png"))
    plotter.close()

    # Optional: Convert to PDF
    if filename.endswith(".pdf"):
        from PIL import Image
        im = Image.open(filename.replace(".pdf", ".png"))
        im.save(filename)

simures = np.load("modelcell3Dmax20_gird36_steps200.npz")
#breakpoint()

visualize_streamlines(simures['u'][-1], 
                                simures['v'][-1], 
                                simures['w'][-1], 
                                simures['p_ext'][-1], 
                                simures['x'], 
                                simures['y'], 
                                simures['z'])