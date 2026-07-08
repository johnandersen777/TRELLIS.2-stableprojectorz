/**
 * RMS (Root Mean Square) normalization.
 *
 * rms_norm(x) = x * weight / sqrt(mean(x^2) + eps)
 *
 * Unlike LayerNorm: no mean subtraction, no bias term.
 * Used by qk_rms_norm in SS_Flow and SLatFlow attention blocks.
 *
 * Expects (N, C) layout. In-place.
 */

import { Tensor } from "../../runtime/tensor.ts";
import type { GPUContext } from "../../runtime/device.ts";
import { getOrCreateShader, getOrCreatePipeline } from "../../runtime/shader_cache.ts";

export function rmsNorm(
  x: Tensor,
  weight: Tensor | null,
  eps: number,
  context: GPUContext,
): Tensor {
  const device = context.device;
  const N = x.shape[0];
  const C = x.shape[1] || 1;
  const T = context.scalarType(x.dtype);

  const code = `${context.precisionPrelude()}
@group(0) @binding(0) var<storage, read_write> feats: array<${T}>;
@group(0) @binding(1) var<storage, read> weight: array<${T}>;
@group(0) @binding(2) var<uniform> dims: vec3<u32>; // N, C, eps_f32_bits

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= dims.x) { return; }
  let C = dims.y;
  let eps = bitcast<f32>(dims.z);

  // Compute RMS = sqrt(mean(x^2) + eps)
  var sum_sq: f32 = 0.0;
  for (var c = 0u; c < C; c++) {
    let val = f32(feats[n * C + c]);
    sum_sq += val * val;
  }
  let rms = sqrt(sum_sq / f32(C) + eps);

  // Normalize and scale
  for (var c = 0u; c < C; c++) {
    let norm = f32(feats[n * C + c]) / rms;
    let w = f32(weight[c]);
    feats[n * C + c] = ${T}(norm * w);
  }
}
`;

  const key = `rmsnorm_${T}`;
  const shader = getOrCreateShader(device, key, code);
  const pipeline = getOrCreatePipeline(device, key, shader);

  const epsBits = new Float32Array([eps]);
  const dimsData = new Uint32Array([
    N,
    C,
    new Uint32Array(epsBits.buffer)[0],
    0,
  ]);
  const dimsBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(dimsBuf, 0, dimsData);

  // Create ones buffer if no weight — temp buffer MUST be destroyed after submit
  // to avoid leaking untracked GPU buffers (root cause of mapAsync bug).
  let weightBuf: GPUBuffer;
  let weightBufOwned = false;
  if (weight?.gpuBuffer) {
    weightBuf = weight.gpuBuffer;
  } else {
    const ones = new Float32Array(C);
    ones.fill(1.0);
    weightBuf = device.createBuffer({
      size: C * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(weightBuf, 0, ones);
    weightBufOwned = true;
  }

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: x.gpuBuffer! } },
      { binding: 1, resource: { buffer: weightBuf } },
      { binding: 2, resource: { buffer: dimsBuf } },
    ],
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(N / 256), 1, 1);
  pass.end();
  device.queue.submit([enc.finish()]);

  dimsBuf.destroy();
  if (weightBufOwned) weightBuf.destroy();

  return x;
}
