"""
Pure-PyTorch sparse convolution fallback — no CUDA extensions needed.
Used when SPARSE_CONV_BACKEND=none (AMD ROCm or CPU).

Implements submanifold sparse convolution (stride=1, no padding)
using PyTorch tensor operations. Slower than CUDA kernels but works
on any PyTorch device.
"""

import math
import torch
import torch.nn as nn
from .. import SparseTensor


def _compute_neighbor_map(coords, spatial_shape, kernel_size, dilation=1):
    """
    Build neighbor index map for each output voxel.

    For each kernel position (kd, kh, kw), finds the input voxel that
    the kernel element would read from.

    Args:
        coords: (N, 4) [batch, x, y, z] int tensor
        spatial_shape: (3,) [X, Y, Z]
        kernel_size: (3,) [Kd, Kh, Kw]
        dilation: (3,) dilation factors

    Returns:
        neighbor_map: (N, V) int32 tensor, where V = kd*kh*kw
            neighbor_map[i, j] = index of input voxel, or -1 if out of bounds
    """
    Kd, Kh, Kw = kernel_size
    dil_d, dil_h, dil_w = dilation
    pad_d = (Kd // 2) * dil_d
    pad_h = (Kh // 2) * dil_h
    pad_w = (Kw // 2) * dil_w

    N = coords.shape[0]
    V = Kd * Kh * Kw

    # Build spatial index → linear index mapping
    max_shape = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
    sp_to_lin = torch.full((max_shape,), -1, dtype=torch.int32, device=coords.device)
    lin_idx = coords[:, 1] * (spatial_shape[1] * spatial_shape[2]) + \
              coords[:, 2] * spatial_shape[2] + \
              coords[:, 3]
    sp_to_lin[lin_idx.long()] = torch.arange(N, dtype=torch.int32, device=coords.device)

    neighbor_map = torch.full((N, V), -1, dtype=torch.int32, device=coords.device)

    v = 0
    for kd in range(Kd):
        off_d = (kd - Kd // 2) * dil_d
        for kh in range(Kh):
            off_h = (kh - Kh // 2) * dil_h
            for kw in range(Kw):
                off_w = (kw - Kw // 2) * dil_w
                neigh_x = coords[:, 1] + off_d
                neigh_y = coords[:, 2] + off_h
                neigh_z = coords[:, 3] + off_w

                # Check bounds
                valid = (neigh_x >= 0) & (neigh_x < spatial_shape[0]) & \
                        (neigh_y >= 0) & (neigh_y < spatial_shape[1]) & \
                        (neigh_z >= 0) & (neigh_z < spatial_shape[2])

                neigh_lin = neigh_x * spatial_shape[1] * spatial_shape[2] + \
                            neigh_y * spatial_shape[2] + neigh_z

                # Lookup linear indices
                neighbor_map[valid, v] = sp_to_lin[neigh_lin[valid].long()]
                v += 1

    return neighbor_map


def sparse_conv3d_init(self, in_channels, out_channels, kernel_size,
                       stride=1, dilation=1, padding=None, bias=True, indice_key=None):
    # Only support submanifold (stride=1, no padding change)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = tuple(kernel_size) if isinstance(kernel_size, (list, tuple)) else (kernel_size,) * 3
    self.stride = tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
    self.dilation = tuple(dilation) if isinstance(dilation, (list, tuple)) else (dilation,) * 3

    # Store weight in flex_gemm layout to match checkpoint format:
    # (Co, Kd, Kh, Kw, Ci) — same as conv_flex_gemm after permute
    self.weight = nn.Parameter(torch.empty(out_channels, *self.kernel_size, in_channels))
    if bias:
        self.bias = nn.Parameter(torch.empty(out_channels))
    else:
        self.register_parameter("bias", None)

    # Initialize
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    if self.bias is not None:
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    self._neighbor_cache = {}


def sparse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    Co, Kd, Kh, Kw, Ci = self.weight.shape
    V = Kd * Kh * Kw

    cache_key = f'conv_none_{Kw}x{Kh}x{Kd}_dil{self.dilation}'
    neighbor_map = self._neighbor_cache.get(cache_key)

    if neighbor_map is None:
        spatial_shape = x.spatial_shape
        if not isinstance(spatial_shape, torch.Tensor):
            spatial_shape = torch.tensor(spatial_shape, device=x.coords.device)
        neighbor_map = _compute_neighbor_map(
            x.coords, spatial_shape,
            self.kernel_size, self.dilation
        )
        self._neighbor_cache[cache_key] = neighbor_map

    N = x.feats.shape[0]
    Ci = self.in_channels
    Co = self.out_channels

    # Gather input features per kernel position
    feats = x.feats  # (N, Ci)
    # Weight: (Co, Kd, Kh, Kw, Ci) → (Co, V*Ci) → transpose → (V*Ci, Co)
    weight_mat = self.weight.reshape(Co, V * Ci).t()  # (V*Ci, Co)

    # im2col: for each output voxel and each kernel position, gather input feats
    # Result: (N, V*Ci)
    im2col = torch.zeros(N, V * Ci, device=feats.device, dtype=feats.dtype)
    for v in range(V):
        valid_mask = neighbor_map[:, v] >= 0
        src_idx = neighbor_map[valid_mask, v].long()
        im2col[valid_mask, v * Ci:(v + 1) * Ci] = feats[src_idx]

    # GEMM
    output = torch.mm(im2col, weight_mat)  # (N, V*Ci) × (V*Ci, Co) → (N, Co)

    if self.bias is not None:
        output += self.bias

    out = x.replace(output)
    return out


def sparse_inverse_conv3d_init(self, *args, **kwargs):
    raise NotImplementedError(
        'SparseInverseConv3d with conv_none is not implemented')


def sparse_inverse_conv3d_forward(self, x: SparseTensor) -> SparseTensor:
    raise NotImplementedError(
        'SparseInverseConv3d with conv_none is not implemented')
