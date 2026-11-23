"""Generate OAT3D voxel densities from native-grid boundary conditions."""
import argparse
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, nargs="+", required=True, help="one or more boundary-condition NPZ files")
    parser.add_argument("--weights", type=Path, required=True, help="directory containing the OAT3D weights")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--raw", action="store_true", help="save the native decoder density before load-contact filling")
    parser.add_argument("--seed", type=int, default=0, help="first candidate seed (default 0)")
    parser.add_argument("--candidates", type=int, default=1, help="number of consecutive seeds per case")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default cuda:0)")
    args = parser.parse_args()
    if args.seed < 0 or args.candidates < 1 or args.seed + args.candidates > 2**63:
        parser.error("specify a positive candidate count and seeds in [0, 2^63)")
    if len({p.stem for p in args.case}) != len(args.case):
        parser.error("case filenames must have distinct stems")

    import cupy
    import numpy as np
    import torch
    from oat3d_inference.conditioning import encode_case, load_case
    from oat3d_inference.loaders import load_models
    from oat3d_inference.pipeline import generate_case

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("a CUDA GPU with bfloat16 support is required")
    device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
    torch.cuda.set_device(device)
    cupy.cuda.Device(device.index).use()
    if not torch.cuda.is_bf16_supported():
        parser.error(f"{device} does not support bfloat16")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    seeds = range(args.seed, args.seed + args.candidates)
    options = SimpleNamespace(steps=10, shift=6.0, dense_cfg=5.0, sparse_cfg=2.5)
    cases = [(path.stem, load_case(path)) for path in args.case]
    for name, _ in cases:
        if any((args.out / name / f"seed_{seed:02d}.npz").exists() for seed in seeds):
            parser.error(f"output exists for {name}; choose a new output directory or seed")
    models = load_models(args.weights, device)
    for name, row in cases:
        with torch.inference_mode():
            encode_case(row, models, device)
            directory = args.out / name
            directory.mkdir(parents=True, exist_ok=True)
            for seed, density in generate_case(models, row, seeds, options, device, raw=args.raw):
                dest = directory / f"seed_{seed:02d}.npz"
                with dest.open("xb") as output:
                    np.savez_compressed(output, density=density, seed=np.int64(seed))
                print(dest, flush=True)


if __name__ == "__main__":
    main()
