"""Balanced 3D Navier--Green surrogate on a cell-centred Neumann grid.

Spatial arrays use ``(z, y, x)`` and vectors/point coordinates use ``(x, y, z)``.
The public dense wrapper follows the 20^3 reference specification; the sparse-node
entry point also supports the native rectangular grids used by D3D_TO_TS.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


REFERENCE_GAMMA = 0.52977353
REFERENCE_SPACING_XYZ = (1.0 / 20.0,) * 3
_CORNERS_XYZ = np.stack(np.meshgrid(*(np.arange(2, dtype=np.int64),) * 3, indexing="ij"), axis=-1).reshape(-1, 3)


def _coalesce(coords, values, shape_xyz, logical_or=False):
    coords = np.asarray(coords, dtype=np.int64).reshape(-1, 3)
    values = np.asarray(values)
    if len(coords) != len(values):
        raise ValueError(f"Coordinate/value length mismatch: {len(coords)} != {len(values)}")
    if not len(coords):
        return coords, values.reshape((0,) + values.shape[1:])
    shape_xyz = np.asarray(shape_xyz, dtype=np.int64)
    if ((coords < 0) | (coords > shape_xyz[None])).any():
        raise ValueError(f"Nodal coordinate outside [0, {shape_xyz.tolist()}]")
    codes = (coords[:, 0] * (shape_xyz[1] + 1) + coords[:, 1]) * (shape_xyz[2] + 1) + coords[:, 2]
    unique, inverse = np.unique(codes, return_inverse=True)
    output = np.zeros((len(unique),) + values.shape[1:], dtype=bool if logical_or else values.dtype)
    (np.logical_or.at if logical_or else np.add.at)(output, inverse, values.astype(bool, copy=False) if logical_or else values)
    z, quotient = unique % (shape_xyz[2] + 1), unique // (shape_xyz[2] + 1)
    y, x = quotient % (shape_xyz[1] + 1), quotient // (shape_xyz[1] + 1)
    return np.stack([x, y, z], axis=1), output


def _effective_cell_centres(node_coords_xyz, shape_xyz, spacing_xyz):
    coords, shape, spacing = np.asarray(node_coords_xyz, dtype=np.float64), np.asarray(shape_xyz), np.asarray(spacing_xyz)
    return np.where(coords == 0, 0.5 * spacing, np.where(coords == shape[None], (shape - 0.5) * spacing, coords * spacing))


def balance_nodal_forces(shape_xyz, fixed_coords_xyz, fixed_dofs, load_coords_xyz, load_forces_xyz, *, spacing_xyz=None, force_reference=1.0):
    """Project loads off constrained DOFs and add minimum-L2 force/moment reactions."""
    shape_xyz = np.asarray(shape_xyz, dtype=np.int64).reshape(3)
    spacing_xyz = np.full(3, 1.0 / shape_xyz.max()) if spacing_xyz is None else np.asarray(spacing_xyz, dtype=np.float64).reshape(3)
    if (shape_xyz < 3).any() or (spacing_xyz <= 0).any() or float(force_reference) <= 0:
        raise ValueError("Grid dimensions must be >=3 and spacing/force_reference must be positive")
    fixed_coords, fixed_dofs = _coalesce(fixed_coords_xyz, np.asarray(fixed_dofs, dtype=bool).reshape(-1, 3), shape_xyz, logical_or=True)
    keep_fixed = fixed_dofs.any(axis=1)
    fixed_coords, fixed_dofs = fixed_coords[keep_fixed], fixed_dofs[keep_fixed]
    load_coords, load_forces = _coalesce(load_coords_xyz, np.asarray(load_forces_xyz, dtype=np.float64).reshape(-1, 3), shape_xyz)
    if not np.isfinite(load_forces).all():
        raise ValueError("Load values must be finite")
    if len(load_coords) and len(fixed_coords):
        shape_nodes = shape_xyz + 1
        fixed_codes = (fixed_coords[:, 0] * shape_nodes[1] + fixed_coords[:, 1]) * shape_nodes[2] + fixed_coords[:, 2]
        load_codes = (load_coords[:, 0] * shape_nodes[1] + load_coords[:, 1]) * shape_nodes[2] + load_coords[:, 2]
        positions = np.searchsorted(fixed_codes, load_codes)
        hits = (positions < len(fixed_codes)) & (fixed_codes[np.minimum(positions, len(fixed_codes) - 1)] == load_codes)
        load_forces[hits] = np.where(fixed_dofs[positions[hits]], 0.0, load_forces[hits])
    amplitude = float(np.linalg.norm(load_forces.ravel()) / float(force_reference))
    active_load = np.any(load_forces != 0.0, axis=1)
    load_coords, load_forces = load_coords[active_load], load_forces[active_load]
    zero_diagnostics = {"amplitude": amplitude, "rank": None, "force_residual": 0.0, "moment_residual": 0.0, "relative_force_residual": 0.0, "relative_moment_residual": 0.0, "support_origin_xyz": None, "reaction_forces_xyz": np.empty((0, 3)), "reaction_coords_xyz": np.empty((0, 3), dtype=np.int64)}
    if amplitude == 0.0:
        return load_coords, load_forces, zero_diagnostics
    if not len(fixed_coords):
        raise ValueError("A nonzero load requires constrained reaction DOFs")

    load_positions = _effective_cell_centres(load_coords, shape_xyz, spacing_xyz)
    fixed_positions = _effective_cell_centres(fixed_coords, shape_xyz, spacing_xyz)
    origin = fixed_positions.mean(axis=0)
    force = load_forces.sum(axis=0)
    moment = np.cross(load_positions - origin, load_forces).sum(axis=0)
    node_ids, components = np.nonzero(fixed_dofs)
    r = fixed_positions[node_ids] - origin
    columns = np.zeros((6, len(node_ids)), dtype=np.float64)
    columns[components, np.arange(len(node_ids))] = 1.0
    x_dof, y_dof, z_dof = components == 0, components == 1, components == 2
    columns[4, x_dof], columns[5, x_dof] = r[x_dof, 2], -r[x_dof, 1]
    columns[3, y_dof], columns[5, y_dof] = -r[y_dof, 2], r[y_dof, 0]
    columns[3, z_dof], columns[4, z_dof] = r[z_dof, 1], -r[z_dof, 0]
    target = -np.r_[force, moment]
    characteristic_length = float(np.max(shape_xyz * spacing_xyz))
    scaled_columns, scaled_target = columns.copy(), target.copy()
    scaled_columns[3:] /= characteristic_length
    scaled_target[3:] /= characteristic_length
    gram = scaled_columns @ scaled_columns.T
    eigenvalues = np.linalg.eigvalsh(gram)
    borderline = eigenvalues[0] <= 1.0e-10 * eigenvalues[-1]
    singular = np.linalg.svd(scaled_columns, compute_uv=False) if borderline else np.sqrt(np.maximum(eigenvalues[::-1], 0.0))
    tolerance = np.finfo(np.float64).eps * max(scaled_columns.shape) * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    if rank != 6:
        raise ValueError(f"Reaction equilibrium matrix must have rank 6, found {rank}")
    rho = np.linalg.lstsq(scaled_columns, scaled_target, rcond=None)[0] if borderline else scaled_columns.T @ np.linalg.solve(gram, scaled_target)
    residual = columns @ rho - target
    reactions = np.zeros((len(fixed_coords), 3), dtype=np.float64)
    reactions[node_ids, components] = rho
    force_scale = max(float(np.linalg.norm(load_forces, axis=1).sum()), float(np.linalg.norm(reactions, axis=1).sum()), np.finfo(np.float64).tiny)
    reaction_moments = np.cross(fixed_positions - origin, reactions)
    moment_scale = max(float(np.linalg.norm(np.cross(load_positions - origin, load_forces), axis=1).sum()), float(np.linalg.norm(reaction_moments, axis=1).sum()), np.finfo(np.float64).tiny)
    relative_force, relative_moment = float(np.linalg.norm(residual[:3]) / force_scale), float(np.linalg.norm(residual[3:]) / moment_scale)
    if relative_force > 1.0e-10 or relative_moment > 1.0e-10:
        raise RuntimeError(f"Reaction balance failed: relative force={relative_force:.3e}, moment={relative_moment:.3e}")
    active_reaction = np.any(reactions != 0.0, axis=1)
    balanced_coords = np.concatenate([load_coords, fixed_coords[active_reaction]])
    balanced_forces = np.concatenate([load_forces, reactions[active_reaction]])
    balanced_coords, balanced_forces = _coalesce(balanced_coords, balanced_forces, shape_xyz)
    diagnostics = {"amplitude": amplitude, "rank": rank, "force_residual": float(np.linalg.norm(residual[:3])), "moment_residual": float(np.linalg.norm(residual[3:])), "relative_force_residual": relative_force, "relative_moment_residual": relative_moment, "support_origin_xyz": origin, "reaction_forces_xyz": reactions, "reaction_coords_xyz": fixed_coords}
    return balanced_coords, balanced_forces, diagnostics


def _backend(name):
    if name == "numpy":
        from scipy import fft
        return np, fft
    if name == "cupy":
        import cupy as cp
        from cupyx.scipy import fft
        return cp, fft
    raise ValueError(f"Unsupported backend {name!r}; choose 'numpy' or 'cupy'")


@lru_cache(maxsize=32)
def _inverse_denominator(backend, device_id, shape_zyx, spacing_xyz, dtype_name):
    xp, _ = _backend(backend)
    nz, ny, nx = shape_zyx
    dx, dy, dz = spacing_xyz
    dtype = xp.float64 if dtype_name == "float64" else xp.float32
    ell_x = 4.0 * xp.sin(xp.pi * xp.arange(nx, dtype=dtype) / (2.0 * nx)) ** 2 / dx ** 2
    ell_y = 4.0 * xp.sin(xp.pi * xp.arange(ny, dtype=dtype) / (2.0 * ny)) ** 2 / dy ** 2
    ell_z = 4.0 * xp.sin(xp.pi * xp.arange(nz, dtype=dtype) / (2.0 * nz)) ** 2 / dz ** 2
    denominator = ell_z[:, None, None] + ell_y[None, :, None] + ell_x[None, None, :]
    inverse = xp.zeros_like(denominator)
    inverse[denominator > 0] = 1.0 / denominator[denominator > 0]
    return inverse


def _deposit_to_cells(coords_xyz, forces_xyz, shape_xyz, spacing_xyz, xp, dtype):
    shape_xyz = np.asarray(shape_xyz, dtype=np.int64)
    coords = np.asarray(coords_xyz, dtype=np.int64)
    forces = np.asarray(forces_xyz, dtype=np.float64)
    source = np.zeros((int(np.prod(shape_xyz)), 3), dtype=np.float64)
    if not len(coords_xyz):
        deposited = source.reshape(int(shape_xyz[2]), int(shape_xyz[1]), int(shape_xyz[0]), 3)
        return deposited.astype(dtype, copy=False) if xp is np else xp.asarray(deposited, dtype=dtype)
    cells = coords[:, None, :] - _CORNERS_XYZ[None]
    valid = ((cells >= 0) & (cells < shape_xyz[None, None])).all(axis=2)
    counts = valid.sum(axis=1)
    selected = cells[valid]
    weights = np.broadcast_to((1.0 / counts)[:, None], valid.shape)[valid]
    selected_forces = np.broadcast_to(forces[:, None, :], (len(coords_xyz), 8, 3))[valid] * weights[:, None] / float(np.prod(spacing_xyz))
    codes = (selected[:, 2] * int(shape_xyz[1]) + selected[:, 1]) * int(shape_xyz[0]) + selected[:, 0]
    np.add.at(source, codes, selected_forces)
    deposited = source.reshape(int(shape_xyz[2]), int(shape_xyz[1]), int(shape_xyz[0]), 3)
    return deposited.astype(dtype, copy=False) if xp is np else xp.asarray(deposited, dtype=dtype)


def solve_balanced_navier_green_sparse(shape_xyz, fixed_coords_xyz, fixed_dofs, load_coords_xyz, load_forces_xyz, *, spacing_xyz=None, force_reference=1.0, youngs_modulus=1.0, poisson_ratio=0.30, gamma=REFERENCE_GAMMA, magnitude_exponent=0.125, design_mask=None, backend="numpy", compute_dtype="float64", output_dtype="float32", model_clip=True, sparsity_cutoff=None, return_displacement=True):
    """Solve from sparse nodal BC/load records and return ``W``, unbounded ``Y``, and clipped ``Y_model``."""
    shape_xyz = tuple(int(v) for v in shape_xyz)
    spacing_xyz = (1.0 / max(shape_xyz),) * 3 if spacing_xyz is None else tuple(float(v) for v in spacing_xyz)
    if not -1.0 < float(poisson_ratio) < 0.5 or float(youngs_modulus) <= 0 or float(gamma) <= 0 or float(magnitude_exponent) <= 0:
        raise ValueError("Invalid elastic constants or gamma")
    coords, forces, diagnostics = balance_nodal_forces(shape_xyz, fixed_coords_xyz, fixed_dofs, load_coords_xyz, load_forces_xyz, spacing_xyz=spacing_xyz, force_reference=force_reference)
    xp, fft = _backend(backend)
    dtype = xp.float64 if compute_dtype == "float64" else xp.float32
    store_dtype = dtype if output_dtype is None else (xp.float64 if output_dtype == "float64" else xp.float32)
    shape_zyx = tuple(reversed(shape_xyz))
    if diagnostics["amplitude"] == 0.0:
        zero = xp.zeros(shape_zyx, dtype=store_dtype)
        result = {"W": zero.copy(), "Y": zero.copy(), "Y_model": zero.copy(), "Y_cut": zero.copy(), "diagnostics": {**diagnostics, "median_W": 0.0, "unclipped_max": 0.0, "clipped_fraction": 0.0}}
        if return_displacement:
            result["u"] = xp.zeros(shape_zyx + (3,), dtype=store_dtype)
        return result

    source = _deposit_to_cells(coords, forces, shape_xyz, spacing_xyz, xp, dtype)
    device_id = int(xp.cuda.runtime.getDevice()) if backend == "cupy" else -1
    inverse = _inverse_denominator(backend, device_id, shape_zyx, spacing_xyz, compute_dtype)

    def green(field):
        transformed = fft.dctn(field, axes=(0, 1, 2), type=2, norm="ortho")
        transformed[0, 0, 0] = 0.0
        return fft.idctn(transformed * (inverse[..., None] if field.ndim == 4 else inverse), axes=(0, 1, 2), type=2, norm="ortho")

    mu = float(youngs_modulus) / (2.0 * (1.0 + float(poisson_ratio)))
    lam = float(youngs_modulus) * float(poisson_ratio) / ((1.0 + float(poisson_ratio)) * (1.0 - 2.0 * float(poisson_ratio)))
    alpha = (lam + mu) / (lam + 2.0 * mu)
    dx, dy, dz = spacing_xyz
    q = green(source)
    qx_grad, qy_grad, qz_grad = xp.gradient(q[..., 0], dz, dy, dx, edge_order=2), xp.gradient(q[..., 1], dz, dy, dx, edge_order=2), xp.gradient(q[..., 2], dz, dy, dx, edge_order=2)
    phi = green(qx_grad[2] + qy_grad[1] + qz_grad[0])
    phi_grad = xp.gradient(phi, dz, dy, dx, edge_order=2)
    u = (q + alpha * xp.stack([phi_grad[2], phi_grad[1], phi_grad[0]], axis=-1)) / mu
    ux_grad, uy_grad, uz_grad = xp.gradient(u[..., 0], dz, dy, dx, edge_order=2), xp.gradient(u[..., 1], dz, dy, dx, edge_order=2), xp.gradient(u[..., 2], dz, dy, dx, edge_order=2)
    exx, eyy, ezz = ux_grad[2], uy_grad[1], uz_grad[0]
    exy, eyz, exz = 0.5 * (ux_grad[1] + uy_grad[2]), 0.5 * (uy_grad[0] + uz_grad[1]), 0.5 * (ux_grad[0] + uz_grad[2])
    W = xp.maximum(mu * (xp.square(exx) + xp.square(eyy) + xp.square(ezz) + 2.0 * (xp.square(exy) + xp.square(eyz) + xp.square(exz))) + 0.5 * lam * xp.square(exx + eyy + ezz), 0.0)
    mask = xp.ones(shape_zyx, dtype=bool) if design_mask is None else xp.asarray(design_mask, dtype=bool)
    if mask.shape != shape_zyx or not bool(mask.any()):
        raise ValueError(f"design_mask must be nonempty with shape {shape_zyx}")
    median_W = float(xp.median(W[mask]).item())
    if not np.isfinite(median_W) or median_W <= 0.0:
        raise RuntimeError(f"Nonzero balanced source produced invalid median energy {median_W}")
    Y = 0.1 * diagnostics["amplitude"] ** float(magnitude_exponent) * (W / median_W) ** float(gamma)
    Y_model = xp.clip(Y, 0.0, 1.0) if model_clip else Y.copy()
    Y_cut = Y_model.copy() if sparsity_cutoff is None else xp.where(Y_model >= float(sparsity_cutoff), Y_model, 0.0)
    diagnostics.update({"median_W": median_W, "unclipped_max": float(Y.max().item()), "clipped_fraction": float((Y > 1.0).mean().item())})
    result = {"W": W.astype(store_dtype), "Y": Y.astype(store_dtype), "Y_model": Y_model.astype(store_dtype), "Y_cut": Y_cut.astype(store_dtype), "diagnostics": diagnostics}
    if return_displacement:
        result["u"] = u.astype(store_dtype)
    return result


def balanced_navier_green_3d(fixed_node_dofs, node_force, **kwargs):
    """Dense reference wrapper; nodal arrays have shape ``(nz+1, ny+1, nx+1, 3)``."""
    fixed_node_dofs, node_force = np.asarray(fixed_node_dofs), np.asarray(node_force)
    if fixed_node_dofs.shape != node_force.shape or fixed_node_dofs.ndim != 4 or fixed_node_dofs.shape[-1] != 3:
        raise ValueError("fixed_node_dofs and node_force must share shape (nz+1, ny+1, nx+1, 3)")
    fixed_zyx = np.argwhere(np.any(fixed_node_dofs, axis=-1))
    load_zyx = np.argwhere(np.any(node_force != 0.0, axis=-1))
    shape_xyz = tuple(int(v - 1) for v in fixed_node_dofs.shape[2::-1])
    return solve_balanced_navier_green_sparse(shape_xyz, fixed_zyx[:, ::-1], fixed_node_dofs[tuple(fixed_zyx.T)], load_zyx[:, ::-1], node_force[tuple(load_zyx.T)], **kwargs)
