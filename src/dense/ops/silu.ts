/**
 * SiLU (Sigmoid Linear Unit) activation: f(x) = x * sigmoid(x).
 *
 * In-place on a dense (N, C) tensor. Uses 2D dispatch grid for large tensors.
 */

import { Tensor } from "../../runtime/tensor.ts";
import type { GPUContext } from "../../runtime/device.ts";
import { getOrCreateShader, getOrCreatePipeline } from "../../runtime/shader_cache.ts";

export function silu(
  x: Tensor,
  context: GPUContext,
): Tensor {
  const device = context.device;
  const N = x.size;
  const T = context.scalarType(x.dtype);
  const enableF16 = context.supportsFP16 && T === "f16";

  const code = `${context.precisionPrelude()}
@group(0) @binding(0) var<storage, read_write> data: array<${T}>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 16776960u; // 65535 * 256 threads per X row
  if (i >= ${N}u) { return; }
  let val = f32(data[i]);
  let sig = 1.0 / (1.0 + exp(-val));
  data[i] = ${T}(val * sig);
}
`;

  const key = `silu_${T}`;
  const shader = getOrCreateShader(device, key, code);
  const pipeline = getOrCreatePipeline(device, key, shader);

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: x.gpuBuffer! } },
    ],
  });

  const wgTotal = Math.ceil(N / 256);
  const maxPerDim = 65535;
  const wgX = Math.min(wgTotal, maxPerDim);
  const wgY = Math.ceil(wgTotal / wgX);
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(wgX, wgY, 1);
  pass.end();
  device.queue.submit([enc.finish()]);

  return x; // in-place
}
