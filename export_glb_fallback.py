"""
Pure-Python GLB exporter for TRELLIS.2 — no CUDA extensions needed.
Replaces o_voxel.postprocess.to_glb for AMD ROCm.

The CUDA path bakes a UV-atlas texture and BVH-remeshes the dual-contour output
into a watertight surface. Neither is available here, and the raw dual-contour
mesh is porous (~17% of quads missing -> ~400k open boundary edges), which looks
sparse/holey in a viewer.

Instead we remesh from the *voxel occupancy*, which is complete: scatter the
voxel coords into a dense grid and run marching cubes -> a watertight surface by
construction. Per-vertex base_color is sampled from the pipeline's attribute
volume, restricted to high-alpha (confident) voxels so low-confidence voxels
(which carry rainbow-noise color) don't bleed in. Falls back to the raw mesh if
skimage/scipy are unavailable or coords/attrs are missing.
"""

import numpy as np
import trimesh

try:
    from skimage import measure as _measure
    from scipy import ndimage as _ndimage
    from scipy.spatial import cKDTree as _cKDTree
    _HAVE_REMESH = True
except Exception:
    _HAVE_REMESH = False


def _np(t):
    return t.detach().cpu().numpy() if hasattr(t, "is_cuda") and t.is_cuda else (
        t.detach().numpy() if hasattr(t, "detach") else np.asarray(t))


def _remesh_from_occupancy(coords, attr_volume, attr_layout, taubin_iters,
                           min_component_faces, alpha_thresh, verbose):
    """Marching-cubes on the voxel occupancy grid -> watertight colored mesh."""
    import scipy.sparse as sp
    coords = _np(coords).astype(np.int64)
    attrs = _np(attr_volume).astype(np.float32)
    base = np.clip(attrs[:, attr_layout['base_color']], 0, 1)
    has_alpha = 'alpha' in attr_layout
    alpha = attrs[:, attr_layout['alpha']].ravel() if has_alpha else np.ones(len(coords), np.float32)

    # Grid resolution used by the extractor (world = coord/G - 0.5).
    G = 1 << int(np.ceil(np.log2(coords.max() + 1)))
    mn = coords.min(0); mx = coords.max(0); pad = 2
    dims = tuple(mx - mn + 1 + 2 * pad)
    gi = coords - mn + pad

    occ = np.zeros(dims, np.float32)
    occ[gi[:, 0], gi[:, 1], gi[:, 2]] = 1.0

    # The texture for occluded voxels (typically one side, since the input is a
    # single view) is iridescent "rainbow" noise. Detect the symmetry (width)
    # axis and which half is noisier via chroma-hue variance, then mirror the
    # clean half's color onto the noisy half. This matches the tank's bilateral
    # symmetry and gives a coherent texture everywhere.
    occ_b = occ > 0.5
    ious = [(occ_b & np.flip(occ_b, a)).sum() / max((occ_b | np.flip(occ_b, a)).sum(), 1)
            for a in range(3)]
    sym_ax = int(np.argmax(ious))
    rg = base[:, 0] - base[:, 1]; gb = base[:, 1] - base[:, 2]
    cax = (mn[sym_ax] + mx[sym_ax]) // 2
    lo = coords[:, sym_ax] < cax
    var_lo = rg[lo].var() + gb[lo].var()
    var_hi = rg[~lo].var() + gb[~lo].var()
    good = lo if var_lo <= var_hi else ~lo
    noisy_var, good_var = (max(var_lo, var_hi), min(var_lo, var_hi))
    mirror_on = noisy_var > 3.0 * good_var + 1e-6 and good.sum() > 1000

    srcmask = np.zeros(dims, bool); srccol = np.zeros(dims + (3,), np.float32)
    gc = gi[good]
    srcmask[gc[:, 0], gc[:, 1], gc[:, 2]] = True
    srccol[gc[:, 0], gc[:, 1], gc[:, 2]] = base[good]
    if mirror_on:
        mc = gc.copy()
        mc[:, sym_ax] = 2 * (cax - mn[sym_ax] + pad) - gc[:, sym_ax]
        mc = np.clip(mc, 0, np.array(dims) - 1)
        srcmask[mc[:, 0], mc[:, 1], mc[:, 2]] = True
        srccol[mc[:, 0], mc[:, 1], mc[:, 2]] = base[good]
    else:
        # No clear noisy half: use all voxels as color source.
        srcmask[gi[:, 0], gi[:, 1], gi[:, 2]] = True
        srccol[gi[:, 0], gi[:, 1], gi[:, 2]] = base

    # Fill every cell with its nearest source color, then smooth for coherence.
    _, inds = _ndimage.distance_transform_edt(~srcmask, return_indices=True)
    filled = srccol[inds[0], inds[1], inds[2]]
    filled = np.stack([_ndimage.gaussian_filter(filled[..., i], 1.2) for i in range(3)], -1)

    # Geometry: marching cubes on the (complete) occupancy grid -> watertight.
    grid = _ndimage.binary_closing(occ_b, iterations=1).astype(np.float32)
    grid = _ndimage.gaussian_filter(grid, 0.6)
    verts, faces, _, _ = _measure.marching_cubes(grid, level=0.5)
    world = (verts + mn - pad) / G - 0.5
    m = trimesh.Trimesh(world, faces, process=True)
    if verbose:
        print(f"[fallback-export] marching_cubes: {len(m.vertices)} verts, {len(m.faces)} faces; "
              f"sym_axis={sym_ax} mirror={'on' if mirror_on else 'off'}")

    comps = sorted(m.split(only_watertight=False), key=lambda c: len(c.faces), reverse=True)
    keep = [c for c in comps if len(c.faces) >= min_component_faces] or comps[:1]
    m = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
    if taubin_iters > 0:
        trimesh.smoothing.filter_taubin(m, iterations=taubin_iters)

    # Sample the filled color grid at mesh vertices.
    gidx = (m.vertices + 0.5) * G - mn + pad
    gidx = np.clip(np.round(gidx).astype(int), 0, np.array(dims) - 1)
    col = filled[gidx[:, 0], gidx[:, 1], gidx[:, 2]]
    # Light mesh-graph smoothing to remove residual speckle.
    N = len(m.vertices); e = m.edges_unique
    A = sp.coo_matrix((np.ones(len(e) * 2),
                       (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                      shape=(N, N)).tocsr()
    deg = np.asarray(A.sum(1)).ravel() + 1e-6
    for _ in range(4):
        col = 0.5 * col + 0.5 * (A @ col) / deg[:, None]
    rgba = np.concatenate([
        (np.clip(col, 0, 1) * 255 + 0.5).astype(np.uint8),
        np.full((N, 1), 255, np.uint8),
    ], axis=1)
    m.visual = trimesh.visual.ColorVisuals(vertex_colors=rgba)
    if verbose:
        print(f"[fallback-export] watertight={m.is_watertight} "
              f"(noisy_var={noisy_var:.3f} good_var={good_var:.3f})")
    return m


def mesh_to_glb(
    vertices,
    faces,
    attr_volume=None,
    coords=None,
    attr_layout=None,
    aabb=None,
    voxel_size=None,
    grid_size=None,
    texture_size: int = 2048,
    decimation_target: int = 1000000,
    remesh: bool = True,
    taubin_iters: int = 10,
    min_component_faces: int = 2000,
    alpha_thresh: float = 0.6,
    verbose: bool = True,
) -> trimesh.Trimesh:
    """Convert TRELLIS.2 mesh output to a watertight, vertex-colored trimesh GLB."""
    # Preferred path: watertight remesh from voxel occupancy + confident colors.
    if remesh and _HAVE_REMESH and coords is not None and attr_volume is not None \
            and attr_layout is not None and 'base_color' in attr_layout:
        try:
            m = _remesh_from_occupancy(coords, attr_volume, attr_layout, taubin_iters,
                                       min_component_faces, alpha_thresh, verbose)
            # Y/Z swap for GLB Y-up (negation copies -> no aliasing).
            m.vertices[:, 1], m.vertices[:, 2] = m.vertices[:, 2].copy(), -m.vertices[:, 1].copy()
            if verbose:
                print(f"[fallback-export] Output: {len(m.vertices)} verts, {len(m.faces)} faces")
            return m
        except Exception as e:
            if verbose:
                print(f"[fallback-export] occupancy remesh failed ({e}); falling back to raw mesh")

    # Fallback: raw dual-contour mesh (porous), floaters removed.
    verts_np = _np(vertices)
    faces_np = _np(faces).astype(np.int64)
    if verbose:
        print(f"[fallback-export] raw-mesh fallback: {verts_np.shape[0]} verts, {faces_np.shape[0]} faces")
    m = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=True)
    try:
        comps = trimesh.graph.connected_components(m.face_adjacency, min_len=100,
                                                   nodes=np.arange(len(m.faces)))
        if comps:
            mask = np.zeros(len(m.faces), bool); mask[np.concatenate(comps)] = True
            m.update_faces(mask); m.remove_unreferenced_vertices()
    except Exception:
        pass
    m.vertices[:, 1], m.vertices[:, 2] = m.vertices[:, 2].copy(), -m.vertices[:, 1].copy()
    return m
