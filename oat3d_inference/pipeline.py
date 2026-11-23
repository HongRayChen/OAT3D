"""Sampling: dense coarse support (16^3 latent, CFG 5) -> load-contact repair -> sparse native density (CFG 2.5)."""
import itertools

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from .conditioning import cfg_context, make_conditioning
from direct3d_s2.modules import sparse as sp

CORNERS = np.asarray(list(itertools.product((0, 1), repeat=3)), dtype=np.int64)


@torch.inference_mode()
def flow_sample(model, context, cfg, opts, seeds, guidance_scale, device, coords=None):
    if coords is None:
        latent = torch.cat([torch.randn((1, 8, 16, 16, 16), device=device, dtype=torch.bfloat16, generator=torch.Generator(device=device).manual_seed(seed)) for seed in seeds])
    else:
        latent = torch.randn((len(coords), 16), device=device, dtype=torch.bfloat16, generator=torch.Generator(device=device).manual_seed(seeds[0]))
        cfg_coords = torch.cat([coords, torch.cat([coords[:, :1] + 1, coords[:, 1:]], dim=1)])
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=int(cfg["timestep_count"]), shift=float(opts.shift))
    scheduler.set_timesteps(int(opts.steps), device=device)
    for timestep in scheduler.timesteps:
        x = torch.cat([latent, latent])
        if coords is not None:
            x = sp.SparseTensor(x, cfg_coords, shape=torch.Size([2, 16])).type(torch.bfloat16)
        pred = model(x, timestep.reshape(1).expand(2 * len(seeds)), cond=context)
        cond, uncond = (pred if coords is None else pred.feats).chunk(2)
        latent = scheduler.step(uncond.float() + guidance_scale * (cond.float() - uncond.float()), timestep, latent.float()).prev_sample.to(torch.bfloat16)
    if not bool(torch.isfinite(latent).all()):
        raise RuntimeError("Nonfinite generated latent")
    return latent


@torch.inference_mode()
def dense_densities(models, single, seeds, opts, device):
    latent = flow_sample(models.dense_dit, cfg_context(single, len(seeds), device), models.dense_cfg, opts, seeds, float(opts.dense_cfg), device)
    group = models.stats["dense_density"]
    raw = latent.float() * group["std"][None, :, None, None, None] + group["mean"][None, :, None, None, None]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        decoded = models.dense_decoder(raw).float().sigmoid()[:, 0]
    if not bool(torch.isfinite(decoded).all()):
        raise RuntimeError("Nonfinite dense decoded density")
    return decoded.cpu().numpy()


def load_cells(row):
    nodes = np.unique(row["load_coords"][np.any(row["load_values"] != 0, axis=1)], axis=0).astype(np.int64)
    cells = nodes[:, None] - CORNERS[None]
    valid = ((cells >= row["offset"]) & (cells < row["offset"] + row["dims"])).all(axis=2)
    return cells.clip(0, 511) // 8, valid


def repair_support(density, row):
    """Support = density > 0.3; every active load node gets at least one incident in-domain coarse cell."""
    support = density > 0.3
    before = support.copy()
    coarse, valid = load_cells(row)
    hit = lambda: (support[coarse[..., 0], coarse[..., 1], coarse[..., 2]] & valid).any(axis=1)
    missing = ~hit()
    additions = coarse[missing][valid[missing]]
    if len(additions):
        support[tuple(additions.T)] = True
    if not bool(hit().all()):
        raise RuntimeError("Dense load support repair failed")
    density = density.copy()
    density[support & ~before] = 1
    return density, support, int(np.count_nonzero(support & ~before)), int(missing.sum())


@torch.inference_mode()
def sparse_density(models, single, support, seed, row, opts, device):
    xyz = np.argwhere(support).astype(np.int32)
    coords = torch.cat([torch.zeros((len(xyz), 1), device=device, dtype=torch.int32), torch.as_tensor(xyz, device=device)], dim=1)
    latent = flow_sample(models.sparse_dit, cfg_context(single, 1, device), models.sparse_cfg, opts, [seed], float(opts.sparse_cfg), device, coords=coords)
    group = models.stats["sparse_density"]
    raw = latent.float() * group["std"][None] + group["mean"][None]
    decoded = models.sparse_decoder(sp.SparseTensor(raw, coords, shape=torch.Size([1, 16])))
    if not bool(torch.isfinite(decoded.feats).all()) or len(decoded.coords) != len(xyz) * 512:
        raise RuntimeError("Invalid sparse decoded density")
    lattice = decoded.coords[:, 1:].long()
    if not bool((decoded.coords[:, 0] == 0).all()) or not bool(((lattice >= 0) & (lattice < 512)).all()):
        raise RuntimeError("Sparse decoder batch or lattice bounds invalid")
    coarse_codes = (coords[:, 1].long() * 64 + coords[:, 2]) * 64 + coords[:, 3]
    blocks = lattice // 8
    codes = (blocks[:, 0] * 64 + blocks[:, 1]) * 64 + blocks[:, 2]
    block_ids = torch.searchsorted(coarse_codes, codes)
    if not bool((block_ids < len(xyz)).all()) or not torch.equal(coarse_codes[block_ids], codes):
        raise RuntimeError("Sparse decoder emitted cells outside the repaired coarse support")
    local = lattice - torch.as_tensor(row["offset"], device=device)
    inside = ((local >= 0) & (local < torch.as_tensor(row["dims"], device=device))).all(dim=1)
    native = torch.zeros(tuple(int(x) for x in row["dims"]), device=device, dtype=torch.float32)
    native[tuple(local[inside].T)] = decoded.feats[inside, 0]
    rho = native.half().cpu().numpy()
    if not np.isfinite(rho).all() or np.any(rho < 0) or np.any(rho > 1):
        raise RuntimeError("Sparse density outside finite [0,1]")
    return rho


def fill_load_contacts(density, row):
    """Place each load object on the most supported of its eight incident corner choices."""
    flat = density.reshape(-1)
    targets = []
    for start, stop in zip(row["load_indptr"][:-1], row["load_indptr"][1:]):
        nodes = row["load_native"][start:stop]
        corners, scores = [], []
        for i, corner in enumerate(CORNERS):
            xyz = np.clip(nodes - corner, 0, row["dims"] - 1)
            indices = np.unique(np.ravel_multi_index(xyz.T, density.shape))
            v = flat[indices].astype(np.float64)
            corners.append(indices)
            scores.append((np.count_nonzero(v >= np.float16(.3)), v.sum(), -(1 - v).sum(), -len(v), -i))
        targets.append(corners[max(range(8), key=scores.__getitem__)])
    flat[np.unique(np.concatenate(targets))] = 1
    return density


@torch.inference_mode()
def generate_case(models, row, seeds, options, device, raw=False):
    """Yield native XYZ densities in seed order, reusing the conditioning across candidates."""
    dense = make_conditioning(row, models.stats, models.dense_cfg, device)
    sparse = make_conditioning(row, models.stats, models.sparse_cfg, device)
    for start in range(0, len(seeds), 4):
        batch = seeds[start:start + 4]
        coarse = dense_densities(models, dense, batch, options, device)
        for seed, density in zip(batch, coarse):
            _, support, _, _ = repair_support(density, row)
            fine = sparse_density(models, sparse, support, seed, row, options, device)
            yield seed, fine if raw else fill_load_contacts(fine, row)
