/**
 * Element-wise operations for dense tensors.
 *
 * In-place where possible (a = a + b, a = a * b).
 * Broadcast: scalar × tensor, or row-wise broadcast.
 */

import { Tensor } from "../../runtime/tensor.ts";
import type { GPUContext } from "../../runtime/device.ts";
import { getOrCreateShader, getOrCreatePipeline } from "../../runtime/shader_cache.ts";

/** Compute a 2D dispatch grid that fits within per-dimension limits (65,535). */
function dispatch2D(N: number, wgSize: number): [number, number] {
  const maxPerDim = 65535;
  const totalWg = Math.ceil(N / wgSize);
  const wgX = Math.min(totalWg, maxPerDim);
  const wgY = Math.ceil(totalWg / wgX);
  return [wgX, wgY];
}

/** In-place element-wise add: a[i] += b[i] */
export function add(
  a: Tensor,
  b: Tensor,
  context: GPUContext,
): Tensor {
  const device = context.device;
  const N = a.size;
  const T = context.scalarType(a.dtype);

  const code = `${context.precisionPrelude()}
@group(0) @binding(0) var<storage, read_write> a: array<${T}>;
@group(0) @binding(1) var<storage, read> b: array<${T}>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 16776960u; // 65535 * 256 threads per X row
  if (i >= ${N}u) { return; }
  a[i] = ${T}(f32(a[i]) + f32(b[i]));
}
`;

  const key = `add_${T}`;
  const shader = getOrCreateShader(device, key, code);
  const pipeline = getOrCreatePipeline(device, key, shader);

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: a.gpuBuffer! } },
      { binding: 1, resource: { buffer: b.gpuBuffer! } },
    ],
  });

  const [wgX, wgY] = dispatch2D(N, 256);
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(wgX, wgY, 1);
  pass.end();
  device.queue.submit([enc.finish()]);

  return a;
}

/** In-place element-wise multiply: a[i] *= b[i] */
export function mul(
  a: Tensor,
  b: Tensor,
  context: GPUContext,
): Tensor {
  const device = context.device;
  const N = a.size;
  const T = context.scalarType(a.dtype);

  const code = `${context.precisionPrelude()}
@group(0) @binding(0) var<storage, read_write> a: array<${T}>;
@group(0) @binding(1) var<storage, read> b: array<${T}>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 16776960u; // 65535 * 256 threads per X row
  if (i >= ${N}u) { return; }
  a[i] = ${T}(f32(a[i]) * f32(b[i]));
}
`;

  const key = `mul_${T}`;
  const shader = getOrCreateShader(device, key, code);
  const pipeline = getOrCreatePipeline(device, key, shader);

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: a.gpuBuffer! } },
      { binding: 1, resource: { buffer: b.gpuBuffer! } },
    ],
  });

  const [wgX, wgY] = dispatch2D(N, 256);
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(wgX, wgY, 1);
  pass.end();
  device.queue.submit([enc.finish()]);

  return a;
}

/** Row-wise broadcast scale: out[i] = a[i] * scale[row] where scale is [Rows, 1] */
export function scaleRows(
  x: Tensor,
  scale: Tensor,
  rowSize: number,
  context: GPUContext,
): Tensor {
  const device = context.device;
  const N = x.size;
  const T = context.scalarType(x.dtype);
  const C = rowSize;

  const code = `${context.precisionPrelude()}
@group(0) @binding(0) var<storage, read_write> x: array<${T}>;
@group(0) @binding(1) var<storage, read> scale: array<${T}>;
@group(0) @binding(2) var<uniform> dims: vec2<u32>; // N, C

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * 16776960u; // 65535 * 256 threads per X row
  if (i >= dims.x) { return; }
  let row = i / dims.y;
  x[i] = ${T}(f32(x[i]) * f32(scale[row]));
}
`;

  const key = `scale_rows_${T}`;
  const shader = getOrCreateShader(device, key, code);
  const pipeline = getOrCreatePipeline(device, key, shader);

  const dimsData = new Uint32Array([N, C]);
  const dimsBuf = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(dimsBuf, 0, dimsData);

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: x.gpuBuffer! } },
      { binding: 1, resource: { buffer: scale.gpuBuffer! } },
      { binding: 2, resource: { buffer: dimsBuf } },
    ],
  });

  const [wgX, wgY] = dispatch2D(N, 256);
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(wgX, wgY, 1);
  pass.end();
  device.queue.submit([enc.finish()]);

  dimsBuf.destroy();
  return x;
}
