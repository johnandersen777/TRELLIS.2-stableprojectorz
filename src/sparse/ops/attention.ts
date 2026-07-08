/**
 * Sparse attention: scaled dot-product attention for SparseTensor inputs.
 *
 * For single-batch (B=1, typical during inference): delegates to dense attention.
 * For multi-batch: CPU roundtrip per batch item (correct but slow).
 *
 * TRELLIS.2 inference always uses B=1, so performance is equivalent to dense.
 */

import { Tensor } from "../../runtime/tensor.ts";
import type { GPUContext } from "../../runtime/device.ts";
import { scaledDotProductAttention } from "../../dense/ops/attention.ts";

export interface SparseAttentionParams {
  Q: Tensor;              // (total_N, H, D) — flattened across batch
  K: Tensor;              // (total_N_kv, H, D)
  V: Tensor;              // (total_N_kv, H, D)
  cuSeqlensQ: Uint32Array;  // (B+1,)
  cuSeqlensKV: Uint32Array; // (B+1,)
  numHeads: number;
  headDim: number;
}

export async function sparseAttention(
  params: SparseAttentionParams,
  ctx: GPUContext,
): Promise<Tensor> {
  const { Q, K, V, numHeads, headDim, cuSeqlensQ, cuSeqlensKV } = params;
  const B = cuSeqlensQ.length - 1;

  // B=1: direct dispatch (same as dense attention)
  if (B === 1) {
    return scaledDotProductAttention({ Q, K, V, numHeads, batchSize: 1 }, ctx);
  }

  // B>1: process each batch item independently via CPU roundtrip
  const N_total = Q.shape[0];
  const C = numHeads * headDim;
  const T = ctx.scalarType(Q.dtype);
  const bytesPerElem = T === "f16" ? 2 : 4;
  const device = ctx.device;

  // Download all tensors to CPU once
  const qCPU = (await Q.toCPU()).getView() as Float32Array;
  const kCPU = (await K.toCPU()).getView() as Float32Array;
  const vCPU = (await V.toCPU()).getView() as Float32Array;

  // Allocate output on CPU
  const outCPU = new Float32Array(N_total * C);

  for (let b = 0; b < B; b++) {
    const qStart = cuSeqlensQ[b] * C;
    const qEnd = cuSeqlensQ[b + 1] * C;
    const kvStart = cuSeqlensKV[b] * C;
    const kvEnd = cuSeqlensKV[b + 1] * C;
    const qLen = (qEnd - qStart) / C;
    const kvLen = (kvEnd - kvStart) / C;

    if (qLen === 0 || kvLen === 0) continue;

    // Slice and upload
    const qSlice = Tensor.fromArray(qCPU.slice(qStart, qEnd), [qLen, numHeads, headDim], "float32");
    const kSlice = Tensor.fromArray(kCPU.slice(kvStart, kvEnd), [kvLen, numHeads, headDim], "float32");
    const vSlice = Tensor.fromArray(vCPU.slice(kvStart, kvEnd), [kvLen, numHeads, headDim], "float32");
    await qSlice.upload(ctx); await kSlice.upload(ctx); await vSlice.upload(ctx);

    const oSlice = scaledDotProductAttention(
      { Q: qSlice, K: kSlice, V: vSlice, numHeads, batchSize: 1 }, ctx,
    );

    // Read back and copy to output
    const oData = (await oSlice.toCPU()).getView() as Float32Array;
    outCPU.set(oData, qStart);
    oSlice.dispose(); qSlice.dispose(); kSlice.dispose(); vSlice.dispose();
  }

  const out = Tensor.fromArray(outCPU, [N_total, numHeads, headDim], "float32");
  await out.upload(ctx);
  return out;
}
