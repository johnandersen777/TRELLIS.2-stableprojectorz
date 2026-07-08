/**
 * Dense Linear layer: output = x @ W^T + b
 *
 * W is stored as (out_features, in_features) — PyTorch convention.
 * Transposes on CPU, uploads, dispatches matmul, adds bias.
 */

import { Tensor } from "../../runtime/tensor.ts";
import type { GPUContext } from "../../runtime/device.ts";
import { matmul } from "../../runtime/ops/matmul.ts";
import { add } from "./elementwise.ts";

// Cache transposed weights to avoid re-creating GPU buffers every linear() call.
// Keyed by the original weight tensor's GPU buffer identity.
const weightCache = new Map<GPUBuffer, Tensor>();

/** x @ W^T where W is (out_C, in_C). Transposes W on CPU, uploads, returns GPU result. */
export async function linear(
  x: Tensor,
  weight: Tensor,     // (out_C, in_C) — CPU tensor
  bias: Tensor | null, // (out_C,) — CPU tensor, nullable
  ctx: GPUContext,
): Promise<Tensor> {
  const [outC, inC] = [weight.shape[0], weight.shape[1]];

  // If weight is already on GPU, check if we have a cached transposed version
  if (weight.device === "gpu" && weight.gpuBuffer) {
    const cached = weightCache.get(weight.gpuBuffer);
    if (cached && cached.context === ctx) {
      let result = matmul(x, cached, ctx);
      if (bias) {
        let biasGPU: Tensor;
        if (bias.device === "gpu" && bias.context === ctx) {
          biasGPU = bias;
        } else {
          biasGPU = await bias.toGPU(ctx);
        }
        result = add(result, biasGPU, ctx);
        if (bias.device !== "gpu" || bias.context !== ctx) biasGPU.dispose();
      }
      return result;
    }
  }

  // Transpose: (out_C, in_C) → (in_C, out_C)
  const wData = weight.getView() as Float32Array;
  const wT = new Float32Array(inC * outC);
  for (let i = 0; i < inC; i++) {
    for (let j = 0; j < outC; j++) {
      wT[i * outC + j] = wData[j * inC + i];
    }
  }
  const wTTensor = Tensor.fromArray(wT, [inC, outC], "float32");
  const wTGPU = await wTTensor.toGPU(ctx);
  let result = matmul(x, wTGPU, ctx);

  // Cache the transposed weight for future use
  if (weight.device === "gpu" && weight.gpuBuffer) {
    weightCache.set(weight.gpuBuffer, wTGPU);
  } else {
    wTGPU.dispose();
  }

  if (bias) {
    let biasGPU: Tensor;
    if (bias.device === "gpu" && bias.context === ctx) {
      biasGPU = bias;
    } else {
      biasGPU = await bias.toGPU(ctx);
    }
    result = add(result, biasGPU, ctx);
    if (bias.device !== "gpu" || bias.context !== ctx) biasGPU.dispose();
  }
  return result;
}
