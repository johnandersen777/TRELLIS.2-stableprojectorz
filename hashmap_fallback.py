"""
Pure-PyTorch implementations of o-voxel CUDA hashmap functions.
Used as fallback when o-voxel._C extension is not available.

Optimized: GPU-sorted flat_idx arrays + torch.searchsorted replace
dict-backed hash table. Eliminates Python loops and CPU↔GPU transfers.
~5 CPU min → ~10ms GPU for 689K-voxel mesh extraction.
"""

import torch
from typing import Tuple

SENTINEL = 0xFFFFFFFF

# Cache: id(hashmap_keys) → (hashmap_keys_ref, sorted_keys, sorted_vals)
# ref stored to prevent id reuse — keeps tensor alive, identity check
_sorted_cache = {}


def _flat_index(coords: torch.Tensor, gy: int, gz: int) -> torch.Tensor:
    """flat_idx = x*gy*gz + y*gz + z  (batch always 0 in current pipeline).
    Fully vectorized, stays on GPU."""
    return coords[:, 1].long() * (gy * gz) + coords[:, 2].long() * gz + coords[:, 3].long()


def _build_sorted_table(
    coords: torch.Tensor,
    gy: int,
    gz: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build deduped sorted (flat_idx → value) mapping from coordinates.
    All GPU, no Python loop. Keeps LAST value per key (matches CUDA overwrite)."""
    N = coords.shape[0]
    device = coords.device
    if N == 0:
        return (torch.empty(0, dtype=torch.int64, device=device),
                torch.empty(0, dtype=torch.int64, device=device))

    flat = _flat_index(coords, gy, gz)
    order = flat.sort(stable=True).indices
    s_keys = flat[order]
    s_vals = torch.arange(N, dtype=torch.int64, device=device)[order]

    # Dedup: keep LAST per key → reverse → unique_consecutive → first = last original
    rev_k = s_keys.flip(0)
    rev_v = s_vals.flip(0)
    _, counts = torch.unique_consecutive(rev_k, return_counts=True)
    ends = counts.cumsum(0) - 1
    return rev_k[ends].flip(0).contiguous(), rev_v[ends].flip(0).contiguous()


def _lookup(
    s_keys: torch.Tensor,
    s_vals: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    """Binary search lookup. All GPU."""
    N = query.shape[0]
    device = query.device
    if s_keys.numel() == 0:
        return torch.full((N,), SENTINEL, dtype=torch.int64, device=device)

    idx = torch.searchsorted(s_keys, query).clamp(0, s_keys.shape[0] - 1)
    matched = s_keys[idx] == query
    result = torch.full((N,), SENTINEL, dtype=torch.int64, device=device)
    result[matched] = s_vals[idx[matched]]
    return result


def hashmap_insert_3d_cpu(
    hashmap_keys: torch.Tensor,
    hashmap_vals: torch.Tensor,
    coords: torch.Tensor,
    grid_size_x: int,
    grid_size_y: int,
    grid_size_z: int,
) -> None:
    """Build GPU-sorted flat_idx→value table from coords and cache."""
    if coords.shape[0] == 0:
        return
    sk, sv = _build_sorted_table(coords, grid_size_y, grid_size_z)
    _sorted_cache[id(hashmap_keys)] = (hashmap_keys, sk, sv)


def hashmap_lookup_3d_cpu(
    hashmap_keys: torch.Tensor,
    hashmap_vals: torch.Tensor,
    coords: torch.Tensor,
    grid_size_x: int,
    grid_size_y: int,
    grid_size_z: int,
) -> torch.Tensor:
    """Lookup via GPU binary search on cached sorted arrays."""
    N = coords.shape[0]
    device = coords.device

    cache = _sorted_cache.get(id(hashmap_keys))
    if cache is None:
        return torch.full((N,), SENTINEL, dtype=torch.int64, device=device)

    ref, sk, sv = cache
    if ref is not hashmap_keys or sk.numel() == 0:
        return torch.full((N,), SENTINEL, dtype=torch.int64, device=device)

    return _lookup(sk, sv, _flat_index(coords, grid_size_y, grid_size_z))


def hashmap_insert_3d_cuda(hashmap_keys, hashmap_vals, coords, *grid_size):
    gx, gy, gz = grid_size[0], grid_size[1], grid_size[2]
    hashmap_insert_3d_cpu(hashmap_keys, hashmap_vals, coords, gx, gy, gz)


def hashmap_lookup_3d_cuda(hashmap_keys, hashmap_vals, coords, *grid_size):
    gx, gy, gz = grid_size[0], grid_size[1], grid_size[2]
    return hashmap_lookup_3d_cpu(hashmap_keys, hashmap_vals, coords, gx, gy, gz)


def clear_sorted_cache():
    """Evict all cached sorted arrays (for testing / between pipeline runs)."""
    _sorted_cache.clear()
