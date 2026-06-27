"""
Pure-Python GLB exporter for TRELLIS.2 — no CUDA extensions needed.
Replaces o_voxel.postprocess.to_glb for AMD ROCm.

Uses trimesh for mesh processing and GLB export.
Skips advanced features (BVH remeshing, nvdiffrast texture baking,
xatlas UV unwrap) in favor of simpler vertex-color or untextured export.
"""

import torch
import numpy as np
import trimesh


def mesh_to_glb(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    attr_volume: torch.Tensor = None,
    coords: torch.Tensor = None,
    attr_layout: dict = None,
    aabb=None,
    voxel_size=None,
    grid_size=None,
    texture_size: int = 2048,
    decimation_target: int = 1000000,
    remesh: bool = False,
    verbose: bool = True,
) -> trimesh.Trimesh:
    """
    Convert TRELLIS.2 mesh output to a trimesh GLB without CUDA extensions.

    Args:
        vertices: (N, 3) on GPU
        faces: (M, 3) on GPU
        attr_volume: (L, C) sparse tensor attrs (optional, for vertex colors)
        coords: (L, 3) coordinates (optional)
        attr_layout: dict of attr name -> slice
        aabb: (2, 3) bounding box
        voxel_size: per-axis voxel size
        grid_size: per-axis grid dimensions
        texture_size: ignored in fallback
        decimation_target: target vertex count
        remesh: ignored in fallback
        verbose: print progress

    Returns:
        trimesh.Trimesh ready for .export('output.glb')
    """
    # Move to CPU
    if vertices.is_cuda:
        verts_np = vertices.detach().cpu().numpy()
    else:
        verts_np = vertices.detach().numpy()

    if faces.is_cuda:
        faces_np = faces.detach().cpu().numpy().astype(np.int64)
    else:
        faces_np = faces.detach().numpy().astype(np.int64)

    if verbose:
        print(f"[fallback-export] Input: {verts_np.shape[0]} verts, {faces_np.shape[0]} faces")

    # Create trimesh object
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=True)

    # Clean up (process=True in constructor handles most cleaning)
    if verbose:
        print("[fallback-export] Cleaning mesh...")
    try:
        mesh.remove_duplicate_faces()
    except AttributeError:
        pass  # removed in trimesh 4.x, process=True handles it
    try:
        mesh.remove_degenerate_faces()
    except AttributeError:
        pass  # removed in trimesh 4.x
    mesh.remove_unreferenced_vertices()

    # Simplify if needed
    if len(mesh.vertices) > decimation_target and decimation_target > 0:
        if verbose:
            print(f"[fallback-export] Simplifying from {len(mesh.vertices)} to ~{decimation_target} verts...")
        try:
            mesh = mesh.simplify_quadric_decimation(decimation_target)
        except Exception as e:
            if verbose:
                print(f"[fallback-export] Simplification failed ({e}), using as-is")

    if verbose:
        print(f"[fallback-export] Output: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    # Swap Y/Z axes for GLB convention
    mesh.vertices[:, 1], mesh.vertices[:, 2] = mesh.vertices[:, 2], -mesh.vertices[:, 1]

    return mesh
