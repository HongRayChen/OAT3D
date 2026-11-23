"""Render native XYZ densities, paired topology comparisons, and the resolution-free tower."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from oat3d_inference.rendering import PANEL_SIZE, scene, topology_mesh

EMPTY = np.empty((0, 3))
EXAMPLES = ("0039a8f1fd0ea232", "00d837ae85c95e22", "008fc167a86ba903", "007a661bae01fa66", "00ff517ecd50cd5f")
RESAMPLE = Image.Resampling.LANCZOS


def load_density(path):
    with np.load(path, allow_pickle=False) as z:
        for key in ("density", "topo"):
            if key in z:
                rho = z[key]
                break
        else:
            raise ValueError(f"{path} must contain a density or topo array")
    if rho.ndim != 3 or not np.isfinite(rho).all() or rho.min() < 0 or rho.max() > 1.00001:
        raise ValueError(f"Expected a finite, native XYZ density in [0, 1] in {path}")
    return rho


def bounds(image):
    y, x = np.where(np.asarray(image.convert("RGB")).min(axis=2) < 246)
    if not len(x):
        raise ValueError("The rendered panel has no visible geometry")
    return (max(0, int(x.min()) - 4), max(0, int(y.min()) - 4),
            min(image.width, int(x.max()) + 5), min(image.height, int(y.max()) + 5))


def shared_crop(images):
    boxes = [bounds(im) for im in images]
    both = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
    return [im.crop(both) for im in images]


def fit(canvas, image, rect):
    x, y, w, h = rect
    scale = min(w / image.width, h / image.height)
    new = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), RESAMPLE)
    canvas.paste(new, (x + (w - new.width) // 2, y + (h - new.height) // 2))


def render(plotter, rho, bc=None):
    loads, forces, fixed = (EMPTY, EMPTY, EMPTY) if bc is None else bc
    mesh = topology_mesh(rho)
    if mesh is None:
        raise ValueError("No voxel exceeds the 0.30 isovalue")
    return Image.fromarray(scene(plotter, mesh, rho.shape, loads, forces, fixed)).convert("RGB")


def comparison(plotter, args):
    from matplotlib import get_data_path

    if len(set(args.case_ids)) != len(args.case_ids):
        raise ValueError("Case IDs must be distinct")
    if any(Path(name).name != name or name in (".", "..", "") for name in args.case_ids):
        raise ValueError("Case IDs must be file stems")
    if args.save_reference_panels and not args.references:
        raise ValueError("--save-reference-panels requires --references")
    left, col, row, margin = 310, 512, 378, 18
    canvas = Image.new("RGB", (left + col * len(args.case_ids) + margin, 2 * row + 2 * margin), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(str(Path(get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"), 45)
    for r, label in enumerate(("OAT3D", "SIMP/PGD")):
        draw.text((left - 18, margin + (r + 0.5) * row), label, fill="#1e2934", font=font, anchor="rm")
    for i, case in enumerate(args.case_ids):
        pred = load_density(args.predictions / case / f"seed_{args.seed:02d}.npz")
        shape = ",".join(str(v) for v in pred.shape)
        if args.references:
            ref = load_density(args.references / case[:2] / f"{case}_topo.npz")
            if pred.shape != ref.shape:
                raise ValueError(f"Grid mismatch for {case}: {pred.shape} and {ref.shape}")
            reference = render(plotter, ref)
            if args.save_reference_panels:
                args.save_reference_panels.mkdir(exist_ok=True, parents=True)
                meta = PngImagePlugin.PngInfo()
                meta.add_text("case_id", case)
                meta.add_text("dimensions_xyz", shape)
                reference.save(args.save_reference_panels / f"{case}.png", pnginfo=meta)
        else:
            ref_path = Path(__file__).resolve().parent / "figures" / "simp-pgd" / f"{case}.png"
            with Image.open(ref_path) as panel:
                if panel.info.get("case_id") != case or panel.info.get("dimensions_xyz") != shape or tuple(plotter.window_size) != panel.size:
                    raise ValueError(f"Reference panel dimensions or case ID do not match {case}")
                reference = panel.convert("RGB")
        pair = shared_crop([render(plotter, pred), reference])
        for r, image in enumerate(pair):
            fit(canvas, image, (left + i * col + 8, margin + r * row + 8, col - 16, row - 16))
    return canvas


def domain_box(plotter, x0, x1, color="#c9d1d8", opacity=0.06, edges=True):
    import pyvista as pv

    box = pv.Box(bounds=(x0, x1, 0, 60, 0, 60))
    plotter.add_mesh(box, color=color, opacity=opacity, lighting=False, render=False)
    if edges:
        plotter.add_mesh(box.extract_all_edges(), color="#8a939b", line_width=1.2 * plotter.window_size[0] / 1600,
                         lighting=False, render=False)


def tower_panels(plotter, path):
    from experiments.tower import DIMS, OVERLAP, STARTS, TOWER_DIMS, segment_boundaries

    density = load_density(path)
    if density.shape != TOWER_DIMS:
        raise ValueError(f"Expected the six-segment tower shape {TOWER_DIMS}")
    fixed, loads, forces = segment_boundaries()
    scene(plotter, None, DIMS, loads, forces, fixed)
    domain_box(plotter, 0, DIMS[0])
    plotter.render()
    single = Image.fromarray(plotter.screenshot()).convert("RGB")
    single = single.crop(bounds(single))

    mesh = topology_mesh(density)
    if mesh is None:
        raise ValueError("Tower has no material above the isovalue")
    shifts = np.column_stack([STARTS, np.zeros((len(STARTS), 2), dtype=int)])
    all_fixed = np.concatenate([fixed + s for s in shifts])
    all_loads = np.concatenate([loads + s for s in shifts])
    scene(plotter, mesh, TOWER_DIMS, all_loads, np.tile(forces, (len(STARTS), 1)), all_fixed,
          geometry_bounds=[mesh.points.min(axis=0), mesh.points.max(axis=0)])
    for name in ("fixed_nodes", "load_nodes", "load_arrows"):
        plotter.remove_actor(name, reset_camera=False, render=False)
    plotter.render()
    structure = Image.fromarray(plotter.screenshot()).convert("RGB")
    plotter.remove_actor("topology", reset_camera=False, render=False)
    for start in STARTS:
        domain_box(plotter, start, start + DIMS[0])
    for start in STARTS[1:]:
        domain_box(plotter, start, start + OVERLAP, "#f2cd17", 0.30, edges=False)
    plotter.render()
    joined = Image.fromarray(plotter.screenshot()).convert("RGB")
    joined, structure = shared_crop([joined, structure])
    canvas = Image.new("RGB", (3300, 1059), "white")
    for panel, rect in zip((single, joined, structure), ((30, 21, 744, 1017), (807, 21, 1215, 1017), (2055, 21, 1215, 1017))):
        fit(canvas, panel, rect)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="mode", required=True)
    single = subs.add_parser("single", help="render a density file with optional supports and force arrows")
    single.add_argument("density", type=Path)
    single.add_argument("--boundary", type=Path)
    panel = subs.add_parser("compare", help="render OAT3D over the matching SIMP/PGD designs")
    panel.add_argument("--predictions", type=Path, required=True, help="generate.py output directory")
    panel.add_argument("--references", type=Path, help="render reference NPZs from a hex-prefix dataset directory")
    panel.add_argument("--save-reference-panels", type=Path, help="also save full-frame reference PNGs when rendering reference NPZs")
    panel.add_argument("--case-ids", nargs="+", default=EXAMPLES)
    panel.add_argument("--seed", type=int, default=0)
    tower = subs.add_parser("tower", help="render the segment boundary conditions, overlaps, and generated tower")
    tower.add_argument("density", type=Path)
    for sub in (single, panel, tower):
        sub.add_argument("--out", type=Path, required=True, help="output PNG path")
    args = parser.parse_args()
    if args.out.suffix.lower() != ".png":
        parser.error("--out must be a PNG path")
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=(1920, 1344), lighting="none")
    try:
        if args.mode == "compare":
            image = comparison(plotter, args)
        elif args.mode == "tower":
            image = tower_panels(plotter, args.density)
        else:
            rho = load_density(args.density)
            bc = None
            if args.boundary:
                with np.load(args.boundary, allow_pickle=False) as z:
                    if tuple(z["dims"]) != rho.shape:
                        raise ValueError("The boundary conditions and density must use the same native grid")
                    bc = z["loads"], z["load_v"], z["bcs"][z["bc_c"].any(axis=1)]
            image = render(plotter, rho, bc).resize(PANEL_SIZE, RESAMPLE)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.out, dpi=(300, 300))
    finally:
        plotter.close()


if __name__ == "__main__":
    main()
