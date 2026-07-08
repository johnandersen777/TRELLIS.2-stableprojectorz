/**
 * Modulated Sparse Transformer Cross Block with adaLN.
 *
 * Same architecture as dense version but operates on SparseTensor.
 * Used by SLatFlowModel. Exact port of Python sparse transformer block.
 *
 * See src/dense/blocks/transformer_cross_block.ts for detailed docs.
 * The sparse variant uses the same ops (attention, norm, FFN) with
 * SparseTensor coordinate handling for RoPE.
 */

import { SparseTensor } from "../sparse_tensor.ts";
import type { GPUContext } from "../../runtime/device.ts";
// Re-exports from dense block for SLatFlow to use
export {
  splitFusedQkv, splitFusedQkvBias, splitFusedKv, splitFusedKvBias,
  type BlockConfig,
} from "../../dense/blocks/transformer_cross_block.ts";
// The full forward pass is implemented inline in slat_flow_model.ts
// (same structure as dense block but on SparseTensor feats)
