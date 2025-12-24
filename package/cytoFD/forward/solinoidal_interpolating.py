import numpy as np
from fipy import Grid3D, CellVariable, DiffusionTerm, FaceVariable
from scipy.interpolate import RegularGridInterpolator
import fipy.tools.numerix as numerix


# ---------- 1) Reconstruct coarse Grid3D from saved cell-center x,y,z ----------

def grid3d_from_cell_centers_xyz(x, y, z, rtol=1e-6, atol=1e-12):
    """
    Reconstruct a uniform FiPy Grid3D from saved flattened cell-center coordinates x,y,z.

    Assumptions:
      - x,y,z are cell-center coordinates (not face coords)
      - grid is uniform Cartesian
      - ordering of x,y,z can be arbitrary (we don't assume reshape order)

    Returns:
      mesh_coarse, axes (xu,yu,zu), and an index mapping to go between
      flattened arrays <-> (nx,ny,nz) arrays in a stable way.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    if not (x.size == y.size == z.size):
        raise ValueError("x,y,z must have same length")

    xu = np.unique(x)
    yu = np.unique(y)
    zu = np.unique(z)
    nx, ny, nz = len(xu), len(yu), len(zu)
    #breakpoint()
    if nx * ny * nz != x.size:
        raise ValueError("Unique counts nx*ny*nz != number of cells; grid may be non-Cartesian or missing points.")

    def _check_uniform_axis(u, name):
        if len(u) < 2:
            return 1.0
        d = np.diff(u)
        d0 = float(np.median(d))
        if not np.allclose(d, d0, rtol=rtol, atol=atol):
            raise ValueError(f"Axis {name} not uniform; min/max spacing: {d.min()} / {d.max()}")
        return d0

    dx = _check_uniform_axis(xu, "x")
    dy = _check_uniform_axis(yu, "y")
    dz = _check_uniform_axis(zu, "z")

    # origin is corner (not center): first center is origin + 0.5*d
    origin = (float(xu.min() - 0.5 * dx),
              float(yu.min() - 0.5 * dy),
              float(zu.min() - 0.5 * dz))

    mesh = Grid3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz) #+ origin

    # Build a stable mapping between the user's arbitrary flat order and (i,j,k) array order.
    # We'll define canonical (i,j,k) index where i corresponds to xu, etc.
    ix = np.searchsorted(xu, x)
    iy = np.searchsorted(yu, y)
    iz = np.searchsorted(zu, z)
    if np.any(xu[ix] != x) or np.any(yu[iy] != y) or np.any(zu[iz] != z):
        raise ValueError("Some (x,y,z) did not match unique axes exactly. Are these really exact cell centers?")

    # Canonical linear index for (nx,ny,nz) with order (i,j,k) -> (i*ny + j)*nz + k
    lin = (ix * ny + iy) * nz + iz

    # `perm_to_canonical`: user_flat[?] -> canonical_flat[lin]
    # We'll use argsort to invert.
    perm_user_to_canonical = np.argsort(lin)  # gives indices in user order that fill canonical order
    # `perm_canonical_to_user`: canonical_flat -> user_flat
    perm_canonical_to_user = np.empty_like(perm_user_to_canonical)
    perm_canonical_to_user[perm_user_to_canonical] = np.arange(x.size)

    return mesh, (xu, yu, zu), perm_user_to_canonical, perm_canonical_to_user


# ---------- 2) Interpolate flattened cell fields coarse -> fine (still flattened) ----------

def interpolate_flat_cell_fields(
    mesh_coarse, axes_xyz, perm_user_to_canonical,
    mesh_fine,
    fields_user_flat,              # dict: {"u":..., "v":..., "w":..., "chi":...} in user's saved flat order
    method="linear",
):
    xu, yu, zu = axes_xyz
    nx, ny, nz = mesh_coarse.shape

    out = {}
    # points to sample on: fine cell centers
    ccf = mesh_fine.cellCenters
    pts = np.vstack([np.asarray(ccf[0]), np.asarray(ccf[1]), np.asarray(ccf[2])]).T  # (nFine,3)

    for name, user_flat in fields_user_flat.items():
        user_flat = np.asarray(user_flat, dtype=float).ravel()
        if user_flat.size != mesh_coarse.numberOfCells:
            raise ValueError(f"{name} size mismatch")

        # reorder into canonical flat order, then reshape (nx,ny,nz)
        canonical_flat = user_flat[perm_user_to_canonical]
        arr3 = canonical_flat.reshape((nx, ny, nz), order="C")

        I = RegularGridInterpolator((xu, yu, zu), arr3, method=method, bounds_error=False, fill_value=None)
        vals = np.nan_to_num(I(pts).astype(float))
        out[name] = vals  # flattened on fine mesh (in fine mesh's cell order)
    return out


# ---------- 3) Masked Hodge projection on fine mesh (flattened in, flattened out) ----------

def masked_hodge_project_flat_fipy(
    mesh,
    u_flat, v_flat, w_flat, chi_flat,
    chi_clip=(0.0, 1.0),
    eps_mask=1e-6,
    zero_outside=True,
    bc="dirichlet_zero",
    anchor_face="left",
    solver=None,
):
    n = mesh.numberOfCells
    for name, arr in [("u", u_flat), ("v", v_flat), ("w", w_flat), ("chi", chi_flat)]:
        arr = np.asarray(arr).ravel()
        if arr.size != n:
            raise ValueError(f"{name} has size {arr.size}, expected {n}")

    u = CellVariable(mesh=mesh, value=u_flat)
    v = CellVariable(mesh=mesh, value=v_flat)
    w = CellVariable(mesh=mesh, value=w_flat)

    chi0, chi1 = chi_clip
    chi = CellVariable(mesh=mesh, value=np.clip(chi_flat, chi0, chi1))
    m0 = 1.0 - chi
    m_solve = m0 + eps_mask  # keep PDE well-posed

    # rhs = div(m0 * U)
    velocity = FaceVariable(mesh=mesh, rank=1)
    velocity[:] = numerix.array([u.arithmeticFaceValue, v.arithmeticFaceValue, w.arithmeticFaceValue])
    #rhs = (m0 * u).grad[0] + (m0 * v).grad[1] + (m0 * w).grad[2]
    div_u_star = velocity.divergence
    phi = CellVariable(mesh=mesh, value=0.0)
    if bc == "dirichlet_zero":
        phi.constrain(0.0, mesh.exteriorFaces)
    elif bc == "neumann_zero":
        face_map = {
            "left": mesh.facesLeft, "right": mesh.facesRight,
            "bottom": mesh.facesBottom, "top": mesh.facesTop,
            "front": mesh.facesFront, "back": mesh.facesBack,
        }
        phi.constrain(0.0, face_map[anchor_face])  # anchor to kill nullspace
    else:
        raise ValueError("bc must be 'dirichlet_zero' or 'neumann_zero'")

    (DiffusionTerm(var=phi) == div_u_star).solve( solver=solver)

    uP = u - phi.grad[0]
    vP = v - phi.grad[1]
    wP = w - phi.grad[2]

    if zero_outside:
        uP = m0 * uP
        vP = m0 * vP
        wP = m0 * wP

    return np.asarray(uP.value), np.asarray(vP.value), np.asarray(wP.value), np.asarray(chi.value), np.asarray(rhs.value)


# ---------- 4) End-to-end convenience wrapper ----------

def solinoidal_interpolating(
    x, y, z,
    u, v, w,
    chi,
    refine=2,
    interp_method="linear",
    bc="dirichlet_zero",
    eps_mask=1e-6,
    zero_outside=True,
):
    """
    Inputs: x,y,z,u,v,w,chi all flattened from FiPy cell variables (same length).
    Returns: mesh_fine and flattened (uP,vP,wP,chi_f,rhs_f) on mesh_fine.
    """
    mesh_c, axes_xyz, perm_u2c, _ = grid3d_from_cell_centers_xyz(x, y, z)

    nx, ny, nz = mesh_c.shape
    dx, dy, dz = float(mesh_c.dx), float(mesh_c.dy), float(mesh_c.dz)
    # preserve origin via same translation trick FiPy uses
    cc = mesh_c.cellCenters
    origin = (float(np.min(np.asarray(cc[0])) - 0.5 * dx),
              float(np.min(np.asarray(cc[1])) - 0.5 * dy),
              float(np.min(np.asarray(cc[2])) - 0.5 * dz))

    mesh_f = Grid3D(nx=refine * nx, ny=refine * ny, nz=refine * nz,
                    dx=dx / refine, dy=dy / refine, dz=dz / refine) #+ origin

    fine = interpolate_flat_cell_fields(
        mesh_c, axes_xyz, perm_u2c,
        mesh_f,
        {"u": u, "v": v, "w": w, "chi": chi},
        method=interp_method,
    )

    uP, vP, wP, chi_f, rhs_f = masked_hodge_project_flat_fipy(
        mesh_f, fine["u"], fine["v"], fine["w"], fine["chi"],
        bc=bc, eps_mask=eps_mask, zero_outside=zero_outside
    )
    return mesh_f.cellCenters[0], mesh_f.cellCenters[1],mesh_f.cellCenters[2], \
           uP, vP, wP, chi_f, rhs_f


def simple_interpolate(
    stress_flat,
    x, y, z,
    refine=2,
    method="linear",
):
    """
    Interpolate a flattened FiPy cell-centered stress field from a coarse uniform grid
    (specified by saved x,y,z cell centers) to a refined Grid3D.

    Returns:
      mesh_f, x_f, y_f, z_f, stress_f   (all flattened arrays for fine cell centers)
    """
    stress_flat = np.asarray(stress_flat, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    if not (stress_flat.size == x.size == y.size == z.size):
        raise ValueError("stress_flat and x,y,z must all have the same length")

    mesh_c, xus, perm_u2c, perm_c2u = grid3d_from_cell_centers_xyz(x, y, z)
    xu, yu, zu = xus
    nx, ny, nz = mesh_c.shape
    dx, dy, dz = float(mesh_c.dx), float(mesh_c.dy), float(mesh_c.dz)

    # Build refined mesh with matching origin
    origin = (float(xu.min() - 0.5 * dx),
              float(yu.min() - 0.5 * dy),
              float(zu.min() - 0.5 * dz))
    
    mesh_f = Grid3D(
        nx=refine * nx, ny=refine * ny, nz=refine * nz,
        dx=dx / refine, dy=dy / refine, dz=dz / refine
    ) #+ origin
    #breakpoint()
    # Reorder stress into canonical (nx,ny,nz), then interpolate
    stress_canonical = stress_flat[perm_u2c].reshape((nx, ny, nz), order="C")
    interp = RegularGridInterpolator((xu, yu, zu), stress_canonical,
                                     method=method, bounds_error=False, fill_value=None)
    
    ccf = mesh_f.cellCenters
    pts = np.vstack([np.asarray(ccf[0]), np.asarray(ccf[1]), np.asarray(ccf[2])]).T
    stress_f = np.nan_to_num(interp(pts).astype(float))

    # Fine x,y,z (flattened)
    x_f = np.asarray(ccf[0]).copy()
    y_f = np.asarray(ccf[1]).copy()
    z_f = np.asarray(ccf[2]).copy()

    return  x_f, y_f, z_f, stress_f