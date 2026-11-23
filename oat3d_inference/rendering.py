"""Native XYZ topology surfaces and the OAT3D paper camera."""
import numpy as np

TOPOLOGY_COLOR, TOPOLOGY_OPACITY = "#edf3f8", 0.30
FIXED_COLOR, LOAD_COLOR, ARROW_COLOR = "#2458db", "#e83131", "#d82127"
FIXED_POINT_SIZE, LOAD_POINT_SIZE = 2.4, 4.0  # Pixels at a 1600-pixel render width.
PANEL_SIZE = (1600, 1120)
CAMERA_DIRECTION = (0.778, 0.438, 0.449)
STUDIO_LIGHTS = ((-0.9, 1.5, 1.8, 0.95), (1.3, 0.3, 1.0, 0.22), (0.4, 1.0, -1.2, 0.42))
ISOVALUE, AMBIENT, DIFFUSE, SPECULAR, SPECULAR_POWER = 0.30, 0.16, 0.80, 0.10, 30
DEPTH_PEELS, OCCLUSION_RATIO = 100, 0.0


def topology_mesh(rho):
    """Build the native XYZ isosurface at density 0.30."""
    import pyvista as pv
    from skimage.measure import marching_cubes

    rho = np.asarray(rho, dtype=np.float32)
    if rho.ndim != 3 or not np.isfinite(rho).all():
        raise ValueError("Density must be a finite native-XYZ 3D array")
    if not np.any(rho > ISOVALUE):
        return None
    vertices, faces, _, _ = marching_cubes(np.pad(rho, 1), level=ISOVALUE, allow_degenerate=False)
    mesh = pv.PolyData(vertices - 0.5, np.column_stack([np.full(len(faces), 3), faces]))
    return mesh.compute_normals(split_vertices=True, feature_angle=55, consistent_normals=True, auto_orient_normals=True)


def scene(plotter, mesh, dims, loads, values, fixed, geometry_bounds=None):
    """Set the lights and orthographic camera, draw the native-grid scene, and return RGB pixels."""
    import pyvista as pv

    dims = np.asarray(dims, dtype=float)
    loads, values, fixed = (np.asarray(a, dtype=float).reshape(-1, 3) for a in (loads, values, fixed))
    if dims.shape != (3,) or np.any(dims <= 0) or loads.shape != values.shape:
        raise ValueError("Expected positive XYZ dimensions and matching load coordinates/vectors")
    if not all(np.isfinite(a).all() for a in (dims, loads, values, fixed)):
        raise ValueError("Geometry and BC arrays must be finite")
    active = np.any(values != 0, axis=1)
    loads, values = loads[active], values[active]
    direction = np.array(CAMERA_DIRECTION)
    direction /= np.linalg.norm(direction)
    right = np.cross([0, 1, 0], direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    bounds = np.array(np.meshgrid(*[(0, d) for d in dims], indexing="ij")).reshape(3, -1).T
    if geometry_bounds is not None:
        bounds = np.concatenate([bounds, np.asarray(geometry_bounds).reshape(-1, 3)])
    if len(fixed):
        bounds = np.concatenate([bounds, fixed])
    if len(loads):
        # Use at most 24 arrows and include their tails in the camera bounds.
        chosen = np.unique(np.linspace(0, len(loads) - 1, min(24, len(loads))).astype(int))
        vectors = values[chosen].astype(float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors *= min(0.09 * max(dims), 0.25 * min(dims)) / np.where(norms == 0, 1, norms)
        bounds = np.concatenate([bounds, loads, loads[chosen] - vectors])
    center = (bounds.min(0) + bounds.max(0)) / 2
    projected = (bounds - center) @ np.array([right, up]).T
    center += ((projected.max(0) + projected.min(0)) / 2) @ np.array([right, up])
    span = max(float(np.ptp(bounds, axis=0).max()), 1.0)
    width, height = plotter.window_size
    was_suppressed = plotter.suppress_rendering
    plotter.suppress_rendering = True
    try:
        plotter.clear()  # clear() also removes lights: always restore the complete rig.
        plotter.disable_anti_aliasing()
        plotter.disable_ssao()
        plotter.ren_win.SetMultiSamples(0)
        plotter.ren_win.SetAlphaBitPlanes(1)
        plotter.set_background("white")
        for horizontal, vertical, forward, intensity in STUDIO_LIGHTS:
            position = center + span * (horizontal * right + vertical * up + forward * direction)
            plotter.add_light(pv.Light(position=position, focal_point=center, intensity=intensity, positional=False))
        if mesh is not None and mesh.n_points:
            plotter.add_mesh(mesh, name="topology", color=TOPOLOGY_COLOR, opacity=TOPOLOGY_OPACITY, smooth_shading=True, ambient=AMBIENT, diffuse=DIFFUSE, specular=SPECULAR, specular_power=SPECULAR_POWER, render=False)
        if len(fixed):
            plotter.add_points(fixed, name="fixed_nodes", color=FIXED_COLOR, point_size=FIXED_POINT_SIZE * width / 1600, render_points_as_spheres=True, lighting=True, opacity=1.0, render=False)
        if len(loads):
            plotter.add_points(loads, name="load_nodes", color=LOAD_COLOR, point_size=LOAD_POINT_SIZE * width / 1600, render_points_as_spheres=True, lighting=True, opacity=1.0, render=False)
            plotter.add_arrows(loads[chosen] - vectors, vectors, name="load_arrows", color=ARROW_COLOR, mag=1, opacity=1.0, render=False)
        plotter.camera_position = [center + span * direction * 3, center, (0, 1, 0)]
        plotter.enable_parallel_projection()
        plotter.camera.parallel_scale = 1.18 * max(np.ptp(projected[:, 1]) / 2, np.ptp(projected[:, 0]) * height / width / 2)
        plotter.reset_camera_clipping_range()
        if not plotter.enable_depth_peeling(number_of_peels=DEPTH_PEELS, occlusion_ratio=OCCLUSION_RATIO):
            raise RuntimeError("Rendering requires a VTK backend with depth peeling")
        plotter.suppress_rendering = False
        plotter.show(auto_close=False, interactive=False)
        image = plotter.screenshot(return_img=True)
        if mesh is not None and mesh.n_points and not plotter.renderer.GetLastRenderingUsedDepthPeeling():
            raise RuntimeError("Depth peeling was enabled but not actually used")
        return image
    finally:
        plotter.suppress_rendering = was_suppressed
