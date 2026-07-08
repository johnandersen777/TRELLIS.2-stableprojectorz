/**
 * GELU (Gaussian Error Linear Unit) activation.
 *
 * tanh approximation: f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * In-place on a dense (N, C) tensor. Uses 2D dispatch grid for large tensors.
 */

import { Tensor } from "../../runtime/tensor.ts";
import type { GPUContext } from "../../runtime/device.ts";
import { getOrCreateShader, getOrCreatePipeline } from "../../runtime/shader_cache.ts";

export function gelu(
  x: Tensor,
  context: GPUContext,
): Tensor {
  const device = context.device;
  const N = x.size;
  const T = context.scalarType(x.dtype);
  const enableF16 = context.supportsFP16 && T === "f16";

  // Pre-compute constants
  const sqrt2OverPi = Math.sqrt(2.0 / Math.PI);
  const coeff = 0.044715;
  const half = 0.5;

  const code = `${context.precisionPrelude()}
@group(0) @binding(0) var<storage, read_write> data: array<${T}>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 16776960u; // 65535 * 256 threads per X row
  if (i >= ${N}u) { return; }
  let x = f32(data[i]);
  let x3 = x * x * x;
  let inner = ${sqrt2OverPi} * (x + ${coeff} * x3);
  let tanh_val = tanh(inner);
  data[i] = ${T}(${half} * x * (1.0 + tanh_val));
}
`;

  const key = `gelu_${T}`;
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
