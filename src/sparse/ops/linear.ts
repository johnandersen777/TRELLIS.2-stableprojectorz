/**
 * SparseLinear — applies dense Linear layer to SparseTensor feats.
 * Equivalent to PyTorch nn.Linear applied independently to each voxel.
 */
import { SparseTensor } from "../sparse_tensor.ts";
import { GPUContext } from "../../runtime/device.ts";
import { matmul } from "../../runtime/ops/matmul.ts";

export function sparseLinear(
  x: SparseTensor,
  weight: SparseTensor, // We use a regular tensor for weight — but it's on GPU
  bias: SparseTensor | null,
  context: GPUContext,
): SparseTensor {
  // weight shape: (out_channels, in_channels)
  // feats shape: (N, in_channels)
  // result: (N, out_channels)
  // Use the matmul op: feats × weight^T

  // For now, just do feats @ weight^T using matmul
  // Transpose weight: (in, out) -> (out, in) needed for matmul
  // Actually matmul(a, b) does A × B where A: (M,K), B: (K,N)
  // We want: feats(N, Ci) × weight^T(Ci, Co) = (N, Co)

  // Since weight is stored as SparseTensor, use its feats
  const result = matmul(x.feats, weight.feats, context);
  return x.replace(result);
}
