"""Encode native boundary conditions for the dense and sparse generators."""
import math

import numpy as np
import torch

from .bc_vae import make_sparse_tensor, make_spatial_range
from .bng_field import solve_balanced_navier_green_sparse
from direct3d_s2.models.autoencoders.distributions import posterior_mode
from direct3d_s2.models.autoencoders.ss_vae import encode_field
from .dit import cube_geometry, geometry_width

FINE_SIZE, PATCH_SIZE, GRID_SIZE, LATENT_GRID_SIZE, TOKEN_WIDTH = 512, 8, 513, 65, 1024
FIXED_LATENT, LOAD_LATENT, COORD_FOURIER = slice(6, 54), slice(54, 150), slice(150, 360)
FIXED_TOTAL, LOAD_TOTAL = slice(360, 363), slice(363, 369)
FIXED_SUB2, FIXED_SUB4, LOAD_SUB2, LOAD_SUB4 = slice(369, 393), slice(393, 585), slice(585, 633), slice(633, 1017)
COORD_RAW = slice(1021, 1024)
LOCAL_OFFSETS = np.stack(np.meshgrid(*(np.arange(PATCH_SIZE, dtype=np.int32),) * 3, indexing="ij"), axis=-1).reshape(-1, 3)
BNG_SOLVER = dict(force_reference=1.0, youngs_modulus=1.0, poisson_ratio=0.30, gamma=0.52977353, magnitude_exponent=0.125, compute_dtype="float32", output_dtype="float32", model_clip=True, sparsity_cutoff=None, return_displacement=False)


def load_case(path):
    """Load native XYZ boundary conditions and prepare the two grid coordinate systems."""
    with np.load(path, allow_pickle=False) as data:
        dims, fixed, loads, ptr = (np.asarray(data[key]) for key in ("dims", "bcs", "loads", "load_indptr"))
        masks, forces, vf = np.asarray(data["bc_c"]), np.asarray(data["load_v"]), np.asarray(data["vf"])
    return prepare_case(dims, fixed, masks, loads, forces, ptr, vf)


def prepare_case(dims, fixed, masks, loads, forces, ptr, vf):
    """Map native grid nodes and free load components to model coordinates."""
    if dims.shape != (3,) or not np.issubdtype(dims.dtype, np.integer) or np.any((dims < 3) | (dims > 512)):
        raise ValueError("dims must contain three grid sizes between 3 and 512")
    for nodes in (fixed, loads):
        if nodes.ndim != 2 or nodes.shape[1] != 3 or not len(nodes) or not np.issubdtype(nodes.dtype, np.integer):
            raise ValueError("nodes must be nonempty integer arrays with shape (N, 3)")
        if np.any(nodes < 0) or np.any(nodes > dims):
            raise ValueError("node coordinates must lie in [0, dims]")
    if masks.shape != fixed.shape or not np.isin(masks, (0, 1)).all() or not masks.any():
        raise ValueError("bc_c must contain a three-component support mask for each fixed node")
    if forces.shape != loads.shape or not np.isrealobj(forces) or not np.isfinite(forces).all():
        raise ValueError("load_v must contain a finite three-component force for each load node")
    if ptr.ndim != 1 or len(ptr) < 2 or not np.issubdtype(ptr.dtype, np.integer):
        raise ValueError("load_indptr must contain integer load-object boundaries")
    if ptr[0] != 0 or ptr[-1] != len(loads) or np.any(ptr[1:] <= ptr[:-1]):
        raise ValueError("load_indptr must partition every load row into nonempty objects")
    if vf.size != 1 or not 0 < float(vf.item()) < 1:
        raise ValueError("vf must be a target volume fraction in (0, 1)")

    dims, fixed, loads = (x.astype(np.int32, copy=False) for x in (dims, fixed, loads))
    forces = forces.astype(np.float32, copy=True)
    if not np.isfinite(forces).all():
        raise ValueError("load_v must be representable in float32")
    node_ids, inverse = np.unique(np.ravel_multi_index(fixed.T, dims + 1), return_inverse=True)
    fixed_dofs = np.zeros((len(node_ids), 3), dtype=bool)
    np.logical_or.at(fixed_dofs, inverse, masks.astype(bool))
    load_ids = np.ravel_multi_index(loads.T, dims + 1)
    positions = np.searchsorted(node_ids, load_ids).clip(max=len(node_ids)-1)
    shared = node_ids[positions] == load_ids
    forces[shared] = np.where(fixed_dofs[positions[shared]], 0, forces[shared])
    active = np.any(forces != 0, axis=1)
    if not active.any():
        raise ValueError("At least one force component must act on a free degree of freedom")
    counts = np.add.reduceat(active.astype(np.int64), ptr[:-1].astype(np.int64))
    ptr = np.r_[0, np.cumsum(counts[counts > 0])]
    loads, forces = loads[active], forces[active]
    scale = np.abs(forces).max()
    offset = (512 - dims) // 2
    return dict(dims=dims, vf=float(vf.item()), offset=offset, fixed_native=fixed, fixed_values=masks.astype(np.float32),
                load_native=loads, load_values=forces / scale, load_indptr=ptr, fixed_coords=fixed + offset,
                load_coords=loads + offset, bng_eligible=True)


def direction_mask(values, direction, bc_type):
    if bc_type == "fixed":
        return values[:, direction] != 0
    axis = direction if direction < 3 else direction - 3
    return values[:, axis] > 0 if direction < 3 else values[:, axis] < 0


def direction_magnitudes(values, direction, bc_type):
    if bc_type == "fixed":
        return None
    axis = direction if direction < 3 else direction - 3
    return np.abs(values[:, axis])


def cutoff_kernel(device, cutoff=5.0):
    axis = torch.arange(-math.ceil(cutoff), math.ceil(cutoff) + 1, device=device, dtype=torch.int32)
    offsets = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
    distance = offsets.float().pow(2).sum(dim=1).sqrt()
    keep = distance < cutoff
    return offsets[keep], ((cutoff - distance[keep]) / cutoff).float()


def coalesce_field(coords, values):
    keys = (coords[:, 0].long() * GRID_SIZE + coords[:, 1].long()) * GRID_SIZE + coords[:, 2].long()
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    output = torch.zeros(len(unique_keys), device=values.device, dtype=values.dtype)
    output.scatter_reduce_(0, inverse, values, reduce="amax", include_self=False)
    z, q = unique_keys % GRID_SIZE, unique_keys // GRID_SIZE
    y, x = q % GRID_SIZE, q // GRID_SIZE
    return torch.stack([x, y, z], dim=1).int(), output


def make_bc_field(point_coords, magnitudes, bc_type, kernel, device, chunk_size=8192):
    offsets, base = kernel
    point_coords = torch.from_numpy(point_coords).to(device=device, dtype=torch.int32)
    magnitudes = None if magnitudes is None else torch.from_numpy(magnitudes).to(device=device, dtype=torch.float32)
    accumulated_coords, accumulated_values = None, None
    for start in range(0, len(point_coords), chunk_size):
        points = point_coords[start:start + chunk_size]
        coords = points[:, None, :] + offsets[None]
        valid = ((coords >= 0) & (coords < GRID_SIZE)).all(dim=2)
        if bc_type == "fixed":
            values = base[None].expand(len(points), -1)[valid]
        else:
            mag = magnitudes[start:start + chunk_size].clamp(1.0e-6, 1.0)
            alpha = (mag - 1.0e-6) / (1.0 - 1.0e-6)
            values = base[None].pow((3.0 + alpha * (0.3 - 3.0))[:, None])[valid]
        coords, values = coalesce_field(coords[valid], values)
        if accumulated_coords is not None:
            coords, values = coalesce_field(torch.cat([accumulated_coords, coords]), torch.cat([accumulated_values, values]))
        accumulated_coords, accumulated_values = coords, values
    if accumulated_coords is None or not len(accumulated_coords):
        raise RuntimeError(f"{bc_type} source set produced an empty cutoff field")
    return accumulated_coords, accumulated_values


def decode_bc_codes(codes):
    z, q = codes % LATENT_GRID_SIZE, codes // LATENT_GRID_SIZE
    return torch.stack([q // LATENT_GRID_SIZE, q % LATENT_GRID_SIZE, z], dim=1)


@torch.inference_mode()
def encode_bc(row, vae, bc_type, kernel, device):
    """Per active direction: sparse field -> BC VAE posterior mode (bf16 -> fp16); assembled as (N, 16*directions)."""
    num_directions = 3 if bc_type == "fixed" else 6
    coords, values = row[f"{bc_type}_coords"], row[f"{bc_type}_values"]
    fields, coords_chunks, feats_chunks = [], [], []
    for direction in range(num_directions):
        mask = direction_mask(values, direction, bc_type)
        if not np.any(mask):
            continue
        field_coords, field_values = make_bc_field(coords[mask], direction_magnitudes(values[mask], direction, bc_type), bc_type, kernel, device)
        coords_chunks.append(torch.cat([torch.full((len(field_coords), 1), len(fields), device=device, dtype=torch.int32), field_coords], dim=1))
        feats_chunks.append(field_values[:, None])
        fields.append(direction)
    if not fields:
        raise RuntimeError(f"No active {bc_type} direction")
    moments = vae(make_sparse_tensor(torch.cat(feats_chunks).contiguous().float(), torch.cat(coords_chunks).contiguous().int(), stride=1, spatial_range=make_spatial_range(len(fields), GRID_SIZE)))
    latent_coords, latent_mean = moments.C, posterior_mode(moments.F).to(torch.bfloat16)
    directions = torch.tensor(fields, device=device, dtype=torch.long)[latent_coords[:, 0].long()]
    codes = (latent_coords[:, 1].long() * LATENT_GRID_SIZE + latent_coords[:, 2].long()) * LATENT_GRID_SIZE + latent_coords[:, 3].long()
    unique_keys, inverse = torch.unique(codes, sorted=True, return_inverse=True)
    output = torch.zeros(len(unique_keys), num_directions * 16, device=device, dtype=torch.bfloat16)
    for direction in fields:
        mask = directions == direction
        output[inverse[mask], direction * 16:(direction + 1) * 16] = latent_mean[mask]
    return decode_bc_codes(unique_keys).cpu().numpy().astype(np.uint16), output.to(torch.float16).cpu().numpy()


# ---- balanced Navier-Green field (native grid) -> aligned 8^3 blocks -> BNG VAE ----

def sparse_sort_order(coords):
    block, inside = coords // PATCH_SIZE, coords % PATCH_SIZE
    return np.lexsort((inside[:, 2], inside[:, 1], inside[:, 0], block[:, 2], block[:, 1], block[:, 0]))


def bng_field_blocks(row, backend="cupy"):
    """All globally aligned 8^3 blocks intersecting the centered native domain, zero outside."""
    dims = row["dims"]
    result = solve_balanced_navier_green_sparse(dims, row["fixed_native"].astype(np.int64), row["fixed_values"] != 0, row["load_native"].astype(np.int64), row["load_values"], backend=backend, **BNG_SOLVER)
    field = result["Y_model"].transpose(2, 1, 0)
    if backend == "cupy":
        import cupy
        field = cupy.asnumpy(field)
    if not np.isfinite(field).all() or np.any(field < 0) or np.any(field > 1):
        raise FloatingPointError("Nonfinite or out-of-range BNG field")
    offset = row["offset"]
    before = offset % PATCH_SIZE
    after = (-(before + dims)) % PATCH_SIZE
    padded = np.pad(field, tuple((int(before[i]), int(after[i])) for i in range(3)))
    sx, sy, sz = padded.shape
    blocks = padded.reshape(sx // PATCH_SIZE, PATCH_SIZE, sy // PATCH_SIZE, PATCH_SIZE, sz // PATCH_SIZE, PATCH_SIZE).transpose(0, 2, 4, 1, 3, 5)
    coarse = np.stack(np.meshgrid(*(np.arange(size // PATCH_SIZE, dtype=np.int32) for size in padded.shape), indexing="ij"), axis=-1).reshape(-1, 3) + ((offset - before) // PATCH_SIZE)[None].astype(np.int32)
    coords = (coarse[:, None, :] * PATCH_SIZE + LOCAL_OFFSETS[None]).reshape(-1, 3).astype(np.int32)
    values = np.ascontiguousarray(blocks.reshape(-1).astype(np.float32))
    return coords, values, coarse.astype(np.uint16)


@torch.inference_mode()
def encode_bng(row, vae, device, backend="cupy"):
    coords, values, coarse = bng_field_blocks(row, backend)
    latent = encode_field(vae, torch.from_numpy(coords).to(device), torch.from_numpy(values).to(device))
    xyz = latent.coords[:, 1:].cpu().numpy().astype(np.uint16)
    latents = latent.feats.to(torch.bfloat16).to(torch.float16).cpu().numpy()
    order = sparse_sort_order(xyz)
    xyz, latents = np.ascontiguousarray(xyz[order]), np.ascontiguousarray(latents[order])
    if not np.array_equal(xyz, coarse[sparse_sort_order(coarse)]):
        raise RuntimeError("BNG encoder support differs from the aligned field blocks")
    return xyz, latents


def encode_case(row, models, device, bng_backend="cupy"):
    kernel = cutoff_kernel(device)
    row["fixed_bc_coords"], row["fixed_bc_latents"] = encode_bc(row, models.fixed_vae, "fixed", kernel, device)
    row["load_bc_coords"], row["load_bc_latents"] = encode_bc(row, models.load_vae, "load", kernel, device)
    row["bng_coords"], row["bng_latents"] = encode_bng(row, models.bng_vae, device, bng_backend)
    return row


# Packed conditioning tokens.

def occupancy_features(row, support_coords, include_fixed, include_load):
    support_coords = np.asarray(support_coords, dtype=np.int32)
    support_codes = (support_coords[:, 0] * LATENT_GRID_SIZE + support_coords[:, 1]) * LATENT_GRID_SIZE + support_coords[:, 2]
    output = np.zeros((support_coords.shape[0], 657), dtype=np.float32)

    def add_direction(coords, values, direction, bc_type, base):
        mask = direction_mask(values, direction, bc_type)
        if not np.any(mask):
            return
        fine = np.unique(coords[mask], axis=0)
        coarse, local = fine // 8, fine % 8
        codes = (coarse[:, 0] * LATENT_GRID_SIZE + coarse[:, 1]) * LATENT_GRID_SIZE + coarse[:, 2]
        positions = np.searchsorted(support_codes, codes)
        valid = (positions < support_codes.shape[0]) & (support_codes[np.minimum(positions, support_codes.shape[0] - 1)] == codes)
        positions, local = positions[valid], local[valid]
        np.add.at(output[:, base], positions, 1.0 / 512.0)
        sub2 = (local[:, 0] // 4) * 4 + (local[:, 1] // 4) * 2 + local[:, 2] // 4
        sub4 = (local[:, 0] // 2) * 16 + (local[:, 1] // 2) * 4 + local[:, 2] // 2
        np.add.at(output, (positions, base + 1 + sub2), 1.0 / 64.0)
        np.add.at(output, (positions, base + 9 + sub4), 1.0 / 8.0)

    if include_fixed:
        for direction in range(3):
            add_direction(row["fixed_coords"], row["fixed_values"], direction, "fixed", direction * 73)
    if include_load:
        for direction in range(6):
            add_direction(row["load_coords"], row["load_values"], direction, "load", 219 + direction * 73)
    return output


def coordinate_fourier(coords, frequencies):
    phase = 2.0 * math.pi * (coords.float() / 64.0)[:, :, None] * frequencies[None, None]
    return torch.stack([phase.sin(), phase.cos()], dim=-1).reshape(coords.shape[0], -1).to(torch.bfloat16)


def scalar_fourier_token(value, token_type, frequencies, device):
    token = torch.zeros(TOKEN_WIDTH, device=device, dtype=torch.float32)
    token[token_type] = 1.0
    phase = 2.0 * math.pi * float(value) * frequencies
    token[6:1022] = torch.stack([phase.sin(), phase.cos()], dim=1).reshape(-1)
    token[1022:1024] = float(value)
    return token.to(torch.bfloat16)


def normalize_bc(row, prefix, stats, device):
    directions = 3 if prefix == "fixed_bc" else 6
    coords = torch.from_numpy(row[f"{prefix}_coords"].astype(np.int32, copy=False)).to(device=device)
    codes = (coords[:, 0].long() * LATENT_GRID_SIZE + coords[:, 1].long()) * LATENT_GRID_SIZE + coords[:, 2].long()
    blocks = torch.from_numpy(row[f"{prefix}_latents"].astype(np.float32, copy=False)).to(device=device).reshape(-1, directions, 16)
    active = blocks.ne(0).any(dim=-1, keepdim=True)
    group = stats[f"{prefix}_active_pooled"]
    blocks = torch.where(active, (blocks - group["mean"][None, None]) / group["std"][None, None], torch.zeros_like(blocks))
    return codes, blocks.reshape(-1, directions * 16).to(torch.bfloat16)


def make_spatial_tokens(row, fixed, load, coord_frequencies, geometry, geometry_bands, device):
    fixed_codes, fixed_latents = fixed
    load_codes, load_latents = load
    codes = torch.unique(torch.cat([fixed_codes, load_codes]), sorted=True)
    width = geometry_width(geometry_bands)
    if not codes.numel():
        return torch.empty(0, TOKEN_WIDTH, device=device, dtype=torch.bfloat16), torch.empty(0, width, device=device, dtype=torch.bfloat16)
    coords = decode_bc_codes(codes)
    tokens = torch.zeros(codes.shape[0], TOKEN_WIDTH, device=device, dtype=torch.bfloat16)
    tokens[:, 4] = 1.0
    tokens[:, COORD_FOURIER], tokens[:, COORD_RAW] = coordinate_fourier(coords, coord_frequencies), coords.to(torch.bfloat16) / 64.0
    if fixed_codes.numel():
        tokens[torch.searchsorted(codes, fixed_codes), FIXED_LATENT] = fixed_latents
    if load_codes.numel():
        tokens[torch.searchsorted(codes, load_codes), LOAD_LATENT] = load_latents
    occupancy = torch.from_numpy(occupancy_features(row, coords.cpu().numpy(), True, True)).to(device=device, dtype=torch.bfloat16)
    tokens[:, FIXED_TOTAL] = occupancy[:, [0, 73, 146]]
    tokens[:, FIXED_SUB2] = torch.cat([occupancy[:, direction * 73 + 1:direction * 73 + 9] for direction in range(3)], dim=1)
    tokens[:, FIXED_SUB4] = torch.cat([occupancy[:, direction * 73 + 9:direction * 73 + 73] for direction in range(3)], dim=1)
    tokens[:, LOAD_TOTAL] = occupancy[:, [219 + direction * 73 for direction in range(6)]]
    tokens[:, LOAD_SUB2] = torch.cat([occupancy[:, 219 + direction * 73 + 1:219 + direction * 73 + 9] for direction in range(6)], dim=1)
    tokens[:, LOAD_SUB4] = torch.cat([occupancy[:, 219 + direction * 73 + 9:219 + direction * 73 + 73] for direction in range(6)], dim=1)
    if geometry:
        centers = coords.float() / 64.0
        token_geometry = cube_geometry(centers, torch.full_like(centers, 1.0 / 64.0), geometry_bands)
    else:
        token_geometry = torch.empty(codes.shape[0], width, device=device, dtype=torch.bfloat16)
    return tokens, token_geometry


def pack_bng(row, stats, device):
    coords = torch.from_numpy(row["bng_coords"].astype(np.int64)).to(device)
    values = torch.from_numpy(row["bng_latents"].astype(np.float32)).to(device)
    block, local = coords // 4, coords % 4
    codes, inverse = torch.unique((block[:, 0] * 16 + block[:, 1]) * 16 + block[:, 2], sorted=True, return_inverse=True)
    packed = torch.zeros((len(codes), 64, 12), device=device, dtype=torch.bfloat16)
    slots = (local[:, 0] * 4 + local[:, 1]) * 4 + local[:, 2]
    packed[inverse, slots] = ((values - stats["mean"][None]) / stats["std"][None]).to(torch.bfloat16)
    centers = (torch.stack([codes // 256, codes // 16 % 16, codes % 16], dim=-1).float() + 0.5) / 16.0
    tokens = torch.zeros((len(codes), 1024), device=device, dtype=torch.bfloat16)
    tokens[:, 5] = 1
    tokens[:, 6:774] = packed.flatten(1)
    frequencies = torch.logspace(np.log10(0.5), np.log10(32.0), 41, device=device)
    phase = 2 * np.pi * centers[:, :, None] * frequencies[None, None]
    tokens[:, 774:1020] = torch.stack([phase.sin(), phase.cos()], dim=-1).reshape(len(codes), 246).to(torch.bfloat16)
    tokens[:, 1020:1023] = centers.to(torch.bfloat16)
    return tokens, centers


@torch.no_grad()
def make_conditioning(row, stats, cfg, device):
    """Single-case conditional context: [VF, dim_x, dim_y, dim_z] scalar tokens + BC tokens + packed BNG tokens."""
    geometry, bands = bool(cfg["geometry_cross_attention"]), int(cfg["geometry_bands"])
    scalar_frequencies = torch.logspace(math.log10(0.5), math.log10(32.0), 508, device=device)
    coord_frequencies = torch.logspace(math.log10(0.5), math.log10(32.0), 35, device=device)
    scalar_tokens = torch.stack([scalar_fourier_token(row["vf"], 0, scalar_frequencies, device)] + [scalar_fourier_token(row["dims"][axis] / 512.0, axis + 1, scalar_frequencies, device) for axis in range(3)])
    fixed = normalize_bc(row, "fixed_bc", stats, device)
    load = normalize_bc(row, "load_bc", stats, device)
    spatial_tokens, spatial_geometry = make_spatial_tokens(row, fixed, load, coord_frequencies, geometry, bands, device)
    bng_tokens = torch.empty(0, TOKEN_WIDTH, device=device, dtype=torch.bfloat16)
    if cfg["bng_conditioning"] and row["bng_eligible"]:
        bng_tokens, centers = pack_bng(row, stats["bng_active"], device)
        if geometry:
            spatial_geometry = torch.cat([spatial_geometry, cube_geometry(centers, torch.full_like(centers, 1.0 / 16.0), bands)])
    sequence = torch.cat([scalar_tokens, spatial_tokens, bng_tokens], dim=0)
    context = {"tokens": sequence.contiguous(), "global_tokens": scalar_tokens[None].contiguous(), "cu_seqlens": torch.tensor([0, sequence.shape[0]], device=device, dtype=torch.int32), "max_seqlen": int(sequence.shape[0]), "batch_size": 1}
    if geometry:
        context["geometry"] = torch.cat([torch.zeros(4, geometry_width(bands), device=device, dtype=torch.bfloat16), spatial_geometry], dim=0).contiguous()
    return context


def cfg_context(single, count, device):
    """`count` conditional copies followed by `count` unconditional (four zero tokens) entries for classifier-free guidance."""
    length, width = single["tokens"].shape
    lengths = [length] * count + [4] * count
    context = {"tokens": torch.cat([single["tokens"].repeat(count, 1), torch.zeros((4 * count, width), device=device, dtype=torch.bfloat16)]).contiguous(), "global_tokens": torch.cat([single["global_tokens"].repeat(count, 1, 1), torch.zeros((count, 4, width), device=device, dtype=torch.bfloat16)]).contiguous(), "cu_seqlens": torch.tensor([0] + list(np.cumsum(lengths)), device=device, dtype=torch.int32), "max_seqlen": max(lengths), "batch_size": 2 * count}
    if "geometry" in single:
        context["geometry"] = torch.cat([single["geometry"].repeat(count, 1), torch.zeros((4 * count, single["geometry"].shape[1]), device=device, dtype=torch.bfloat16)]).contiguous()
    return context
