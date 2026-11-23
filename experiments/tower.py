"""Resolution-free inference on six overlapping tower segments."""
import argparse
from pathlib import Path

import numpy as np

DIMS = (128, 60, 60)
OVERLAP = 13
STARTS = np.arange(6) * (DIMS[0] - OVERLAP)
TARGETS = np.array([0.30, 0.27, 0.24, 0.21, 0.18, 0.15], np.float32)
TOWER_DIMS = (int(STARTS[-1]) + DIMS[0], *DIMS[1:])


def segment_boundaries():
    y, z = np.indices((61, 61))
    yz = np.argwhere(np.minimum.reduce([y, z, 60 - y, 60 - z]) <= 5).astype(np.int32)
    fixed = np.column_stack([np.zeros(len(yz), np.int32), yz])
    forces = np.zeros_like(fixed, dtype=np.float32)
    forces[:, 0] = -1 / len(yz)
    return fixed, fixed + [128, 0, 0], forces


def segment_context(row, stats, cfg, device):
    import torch
    from oat3d_inference.conditioning import make_conditioning

    parts = [make_conditioning(dict(row, vf=float(v)), stats, cfg, device) for v in TARGETS]
    lengths = [len(p["tokens"]) for p in parts] + [4] * len(parts)
    context = {"cu_seqlens": torch.tensor([0, *np.cumsum(lengths)], device=device, dtype=torch.int32),
               "max_seqlen": max(lengths), "batch_size": len(lengths)}
    for key in ("tokens", "global_tokens", "geometry"):
        if key in parts[0]:
            value = torch.cat([part[key] for part in parts])
            zeros = (len(parts), 4, value.shape[-1]) if key == "global_tokens" else (4 * len(parts), value.shape[-1])
            context[key] = torch.cat([value, value.new_zeros(zeros)]).contiguous()
    return context


class TowerStage:
    def __init__(self, mode, models, row, device):
        import torch
        from torchsparse.nn import Conv3d

        self.device, self.mode = device, mode
        self.model, self.decoder = getattr(models, mode + "_dit"), getattr(models, mode + "_decoder")
        self.cfg = getattr(models, mode + "_cfg")
        self.context = segment_context(row, models.stats, self.cfg, device)
        group = models.stats[mode + "_density"]
        self.mean, self.std = group["mean"], group["std"]
        self.offset = torch.as_tensor(row["offset"], device=device)
        self.dims = torch.tensor(DIMS, device=device)
        self.index = [(torch.arange(d, device=device) + o) // 8 for d, o in zip(DIMS, self.offset)]
        self.sparse_convs = [m for m in self.decoder.modules() if isinstance(m, Conv3d)]

    def decode(self, latent, coords, coarse=False):
        import torch
        from direct3d_s2.modules import sparse as sp

        if coords is None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                raw = latent.float() * self.std[:, None, None, None] + self.mean[:, None, None, None]
                density = self.decoder(raw[None]).float().sigmoid()[0, 0]
            if coarse:
                return density
            i, j, k = self.index
            return density[i[:, None, None], j[None, :, None], k[None, None, :]]
        # Request inverse kernel maps when computing derivatives with respect to the input latent.
        for conv in self.sparse_convs:
            conv.training = torch.is_grad_enabled()
        raw = latent.float() * self.std + self.mean
        decoded = self.decoder(sp.SparseTensor(raw, coords, shape=torch.Size([1, 16])))
        pos = decoded.coords[:, 1:].long() - self.offset
        valid = ((pos >= 0) & (pos < self.dims)).all(1)
        return decoded.feats.new_zeros(DIMS).index_put(tuple(pos[valid].T), decoded.feats[valid, 0])

    def sample(self, seed, weight, coords=None):
        import torch
        from diffusers import FlowMatchEulerDiscreteScheduler
        from direct3d_s2.modules import sparse as sp

        n = len(STARTS)
        rng = torch.Generator(device=self.device).manual_seed(seed)
        if coords is None:
            initial = torch.randn((1, 8, 16, 16, 16), generator=rng, dtype=torch.bfloat16, device=self.device)
            latent = initial.expand(n, -1, -1, -1, -1).float().contiguous()
            points = [None] * n
        else:
            bank = torch.randn((64, 64, 64, 16), generator=rng, dtype=torch.bfloat16, device=self.device)
            latent = torch.cat([bank[tuple(c[:, 1:].T.long())] for c in coords]).float()
            lengths = [len(c) for c in coords]
            points = coords
            batch = torch.cat([torch.cat([torch.full_like(c[:, :1], i), c[:, 1:]], dim=1) for i, c in enumerate(coords)])
            cfg_coords = torch.cat([batch, torch.cat([batch[:, :1] + n, batch[:, 1:]], dim=1)])
        schedule = FlowMatchEulerDiscreteScheduler(num_train_timesteps=self.cfg["timestep_count"], shift=6.0)
        schedule.set_timesteps(20, device=self.device)
        for step, t in enumerate(schedule.timesteps):
            with torch.no_grad():
                x = torch.cat([latent, latent]).bfloat16()
                if coords is not None:
                    x = sp.SparseTensor(x, cfg_coords, shape=torch.Size([2 * n, 16]))
                prediction = self.model(x, t.expand(2 * n), cond=self.context)
                cond, uncond = (prediction if coords is None else prediction.feats).chunk(2)
                velocity = uncond.float() + (5.0 if coords is None else 2.5) * (cond.float() - uncond.float())
                clean = latent - schedule.sigmas[step] * velocity

            # Accumulate both neighbors' L1 derivatives, retaining at most two decoder graphs.
            gradients = []
            blocks = clean.unbind() if coords is None else clean.split(lengths)
            for i, (block, pts) in enumerate(zip(blocks, points)):
                u = block.detach().requires_grad_()
                field = self.decode(u, pts)
                faces = torch.stack([field[:OVERLAP], field[-OVERLAP:]])
                if i == 0:
                    left = torch.zeros_like(faces[0])
                else:
                    delta = previous_faces[1].detach() - faces[0].detach()
                    right = delta.sign() / ((n - 1) * delta.numel())
                    gradients.append(torch.autograd.grad(previous_faces, previous, torch.stack([left, right]))[0])
                    left = -right
                previous, previous_faces = u, faces
            gradients.append(torch.autograd.grad(previous_faces, previous, torch.stack([left, torch.zeros_like(left)]))[0])
            grad = torch.stack(gradients) if coords is None else torch.cat(gradients)
            if not torch.isfinite(grad).all():
                raise FloatingPointError(f"Nonfinite {self.mode} overlap gradient at step {step + 1}")
            latent = schedule.step(velocity + weight * grad, t, latent).prev_sample.detach()
        with torch.no_grad():
            blocks = latent.unbind() if coords is None else latent.split(lengths)
            return np.stack([self.decode(block, pts, coarse=coords is None).cpu().numpy() for block, pts in zip(blocks, points)])


def assemble(segments):
    tower = np.zeros(TOWER_DIMS, np.float32)
    ramp = np.arange(1, OVERLAP + 1, dtype=np.float32) / (OVERLAP + 1)
    for i, (start, segment) in enumerate(zip(STARTS, segments)):
        coefficient = np.ones(DIMS[0], dtype=np.float32)
        if i:
            coefficient[:OVERLAP] = ramp
        if i + 1 < len(STARTS):
            coefficient[-OVERLAP:] = ramp[::-1]
        tower[start:start + DIMS[0]] += segment * coefficient[:, None, None]
    return tower


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--weight", type=int, choices=(1, 10, 100, 1000), default=100, help="overlap guidance strength")
    parser.add_argument("--seed", type=int, default=6, help="first candidate seed")
    parser.add_argument("--candidates", type=int, default=1, help="number of consecutive seeds")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.seed < 0 or args.candidates < 1 or args.seed + args.candidates > 2**63:
        parser.error("specify a positive candidate count and seeds in [0, 2^63)")

    import cupy
    import torch
    from oat3d_inference.conditioning import encode_case, prepare_case
    from oat3d_inference.loaders import load_models
    from oat3d_inference.pipeline import repair_support

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("a CUDA GPU with bfloat16 support is required")
    torch.cuda.set_device(device.index if device.index is not None else 0)
    cupy.cuda.Device(torch.cuda.current_device()).use()
    if not torch.cuda.is_bf16_supported():
        parser.error("the selected GPU must support bfloat16")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seeds = range(args.seed, args.seed + args.candidates)
    directory = args.out / f"weight_{args.weight}"
    if any((directory / f"seed_{seed:02d}.npz").exists() for seed in seeds):
        parser.error("a requested candidate exists; choose another output directory or seed")
    fixed, loads, forces = segment_boundaries()
    row = prepare_case(np.array(DIMS), fixed, np.ones_like(fixed), loads, forces, np.array([0, len(loads)]), TARGETS[0])
    models = load_models(args.weights, device)
    with torch.no_grad():
        row = encode_case(row, models, device)
    dense, sparse = (TowerStage(mode, models, row, device) for mode in ("dense", "sparse"))
    directory.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        print(f"Generating tower, weight {args.weight}, seed {seed}", flush=True)
        supports = [repair_support(field, row)[1] for field in dense.sample(seed, args.weight)]
        coords = []
        for support in supports:
            xyz = torch.tensor(np.argwhere(support), dtype=torch.int32, device=device)
            coords.append(torch.cat([torch.zeros_like(xyz[:, :1]), xyz], dim=1))
        segments = sparse.sample(seed, args.weight, coords)
        tower = assemble(segments)
        if not np.isfinite(tower).all() or tower.min() < 0 or tower.max() > 1.000001:
            raise RuntimeError("Invalid assembled tower density")
        dest = directory / f"seed_{seed:02d}.npz"
        with dest.open("xb") as f:
            np.savez_compressed(f, density=tower, segments=segments, starts=STARTS, targets=TARGETS,
                                overlap=np.int32(OVERLAP), weight=np.int32(args.weight), seed=np.int64(seed))
        print(dest, flush=True)


if __name__ == "__main__":
    main()
