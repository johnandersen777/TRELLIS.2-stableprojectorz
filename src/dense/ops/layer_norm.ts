/**
 * LayerNorm: (x - mean) / std * weight + bias
 *
 * For LayerNorm32 (elementwise_affine=False): no weight/bias, just normalize.
 * For elementwise_affine=True: includes learnable weight and bias.
 */

import { Tensor } from "../../runtime/tensor.ts";
import type { GPUContext } from "../../runtime/device.ts";
import { getOrCreateShader, getOrCreatePipeline } from "../../runtime/shader_cache.ts";

export function layerNorm(
  x: Tensor,
  weight: Tensor | null,
  bias: Tensor | null,
  eps: number,
  ctx: GPUContext,
): Tensor {
  const device = ctx.device;
  const N = x.shape[0];
  const C = x.shape[x.shape.length - 1]; // normalize across last dim
  const T = ctx.scalarType(x.dtype);

  // Handle multi-dim tensors by flattening to (N, C)
  // Input may be (N, C) or (N, H, D) — always normalize across last dimension
  const stride = x.size / N; // = C for 2D, = H*D for 3D

  const code = `${ctx.precisionPrelude()}
@group(0) @binding(0) var<storage, read_write> feats: array<${T}>;
@group(0) @binding(1) var<storage, read> weight: array<${T}>;
@group(0) @binding(2) var<storage, read> bias: array<${T}>;
@group(0) @binding(3) var<uniform> dims: vec3<u32>; // N, C, eps_f32_bits

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= dims.x) { return; }
  let C = dims.y;
  let eps = bitcast<f32>(dims.z);

  // Compute mean
  var mean: f32 = 0.0;
  for (var c = 0u; c < C; c++) {
    mean += f32(feats[n * C + c]);
  }
  mean = mean / f32(C);

  // Compute variance
  var variance: f32 = 0.0;
  for (var c = 0u; c < C; c++) {
    let diff = f32(feats[n * C + c]) - mean;
    variance += diff * diff;
  }
  variance = variance / f32(C);

  let inv_std = 1.0 / sqrt(variance + eps);

  // Normalize + affine
  for (var c = 0u; c < C; c++) {
    let norm = (f32(feats[n * C + c]) - mean) * inv_std;
    let w = f32(weight[c]);
    let b = f32(bias[c]);
    feats[n * C + c] = ${T}(norm * w + b);
  }
}
`;

  const key = `layernorm_${T}_C${C}`;
  const shader = getOrCreateShader(device, key, code);
  const pipeline = getOrCreatePipeline(device, key, shader);

  const epsBits = new Float32Array([eps]);
  const dimsData = new Uint32Array([N, C, new Uint32Array(epsBits.buffer)[0], 0]);
  const dimsBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(dimsBuf, 0, dimsData);

  // Create ones/zeros if no weight/bias — these temp buffers MUST be destroyed
  // after submit to avoid leaking untracked GPU buffers (root cause of mapAsync bug).
  let wBuf: GPUBuffer;
  let bBuf: GPUBuffer;
  let wBufOwned = false;
  let bBufOwned = false;
  if (weight?.gpuBuffer) {
    wBuf = weight.gpuBuffer;
  } else {
    const ones = new Float32Array(C); ones.fill(1.0);
    wBuf = device.createBuffer({ size: C * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(wBuf, 0, ones);
    wBufOwned = true;
  }
  if (bias?.gpuBuffer) {
    bBuf = bias.gpuBuffer;
  } else {
    const zeros = new Float32Array(C);
    bBuf = device.createBuffer({ size: C * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(bBuf, 0, zeros);
    bBufOwned = true;
  }

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: x.gpuBuffer! } },
      { binding: 1, resource: { buffer: wBuf } },
      { binding: 2, resource: { buffer: bBuf } },
      { binding: 3, resource: { buffer: dimsBuf } },
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
  if (wBufOwned) wBuf.destroy();
  if (bBufOwned) bBuf.destroy();
  return x;
}
