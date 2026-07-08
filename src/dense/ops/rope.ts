/**
 * Rotary Position Embedding (RoPE) for 3D coordinates.
 *
 * Applies rotary embeddings to Q and K tensors before attention.
 * 3D variant: splits head_dim into thirds, applies per-axis rotation.
 *
 * Input: (N, H, D) where N = S_q or S_kv, H = num_heads, D = head_dim
 * Coords: (N, 3) integer coordinates (x, y, z)
 *
 * Pre-computed sin/cos tables passed as uniform buffers.
 */

import { Tensor } from "../../runtime/tensor.ts";
import type { GPUContext } from "../../runtime/device.ts";
import { getOrCreateShader, getOrCreatePipeline } from "../../runtime/shader_cache.ts";

export interface RoPEParams {
  /** Input tensor (N, H, D) */
  x: Tensor;
  /** Coordinates (N, 3) as int32 */
  coords: Tensor;
  /** Number of heads */
  numHeads: number;
  /** Head dimension */
  headDim: number;
  /** Max coordinate value (for frequency computation) */
  maxCoord: number;
  /** Base frequency for RoPE (default 10000) */
  theta?: number;
}

/**
 * Apply RoPE to Q or K tensor in-place.
 *
 * Frequencies: 1 / (theta^(2i/D)) for i in 0..D/6
 * For 3D: each coordinate axis rotates D/3 dimensions.
 */
export function applyRoPE(
  params: RoPEParams,
  context: GPUContext,
): Tensor {
  const { x, coords, numHeads, headDim } = params;
  const maxCoord = params.maxCoord;
  const theta = params.theta ?? 10000.0;
  const device = context.device;
  const N = x.shape[0];
  const T = context.scalarType(x.dtype);

  // Each axis gets D/3 dimensions → D/6 frequency pairs
  const dimPerAxis = Math.floor(headDim / 3);
  const halfDims = Math.floor(dimPerAxis / 2); // pairs per axis

  // Pre-compute sin/cos for each coordinate position
  // For each coord value c in [0, maxCoord] and each freq pair:
  //   freq = 1 / theta^(2*i/halfDims*2)
  //   rot = c / freq
  //   cos_table[c][i] = cos(rot), sin_table[c][i] = sin(rot)
  const maxC = maxCoord + 1;
  const cosTable = new Float32Array(maxC * halfDims * 3);
  const sinTable = new Float32Array(maxC * halfDims * 3);

  for (let axis = 0; axis < 3; axis++) {
    const offset = axis * maxC * halfDims;
    for (let c = 0; c < maxC; c++) {
      for (let i = 0; i < halfDims; i++) {
        const freq = 1.0 / Math.pow(theta, (2 * i) / dimPerAxis);
        const rot = c / freq;
        cosTable[offset + c * halfDims + i] = Math.cos(rot);
        sinTable[offset + c * halfDims + i] = Math.sin(rot);
      }
    }
  }

  const cosBuf = device.createBuffer({
    size: cosTable.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const sinBuf = device.createBuffer({
    size: sinTable.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(cosBuf, 0, cosTable);
  device.queue.writeBuffer(sinBuf, 0, sinTable);

  // Uniform: vec4(N, H, D, halfDims), vec4(maxC, dimPerAxis, 0, 0)
  const dimsData = new Uint32Array([N, numHeads, headDim, halfDims, maxC, dimPerAxis, 0, 0]);
  const dimsBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const dimsBuf2 = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(dimsBuf, 0, dimsData.subarray(0, 4));
  device.queue.writeBuffer(dimsBuf2, 0, dimsData.subarray(4, 8));

  const code = `${context.precisionPrelude()}
@group(0) @binding(0) var<storage, read_write> x: array<${T}>;
@group(0) @binding(1) var<storage, read> coords: array<i32>;
@group(0) @binding(2) var<storage, read> cos_table: array<f32>;
@group(0) @binding(3) var<storage, read> sin_table: array<f32>;
@group(0) @binding(4) var<uniform> dims: vec4<u32>;
@group(0) @binding(5) var<uniform> dims2: vec4<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= dims.x) { return; }

  let H = dims.y;
  let head_dim = dims.z;
  let half_dims = dims.w;
  let max_c = dims2.x;
  let dim_per_axis = dims2.y;

  // Read coordinates for this position
  let cx = coords[idx * 3 + 0];
  let cy = coords[idx * 3 + 1];
  let cz = coords[idx * 3 + 2];

  // Process each head
  for (var h = 0u; h < H; h++) {
    let base = (idx * H + h) * head_dim;

    // Axis 0 (X): dimensions [0, dim_per_axis)
    for (var i = 0u; i < half_dims; i++) {
      let cos_val = cos_table[0u * max_c * half_dims + u32(cx) * half_dims + i];
      let sin_val = sin_table[0u * max_c * half_dims + u32(cx) * half_dims + i];
      let d0 = base + 2 * i;
      let d1 = base + 2 * i + 1;
      let v0 = f32(x[d0]);
      let v1 = f32(x[d1]);
      x[d0] = ${T}(v0 * cos_val - v1 * sin_val);
      x[d1] = ${T}(v0 * sin_val + v1 * cos_val);
    }

    // Axis 1 (Y): dimensions [dim_per_axis, 2*dim_per_axis)
    for (var i = 0u; i < half_dims; i++) {
      let cos_val = cos_table[1u * max_c * half_dims + u32(cy) * half_dims + i];
      let sin_val = sin_table[1u * max_c * half_dims + u32(cy) * half_dims + i];
      let d0 = base + dim_per_axis + 2 * i;
      let d1 = base + dim_per_axis + 2 * i + 1;
      let v0 = f32(x[d0]);
      let v1 = f32(x[d1]);
      x[d0] = ${T}(v0 * cos_val - v1 * sin_val);
      x[d1] = ${T}(v0 * sin_val + v1 * cos_val);
    }

    // Axis 2 (Z): dimensions [2*dim_per_axis, 3*dim_per_axis)
    for (var i = 0u; i < half_dims; i++) {
      let cos_val = cos_table[2u * max_c * half_dims + u32(cz) * half_dims + i];
      let sin_val = sin_table[2u * max_c * half_dims + u32(cz) * half_dims + i];
      let d0 = base + 2 * dim_per_axis + 2 * i;
      let d1 = base + 2 * dim_per_axis + 2 * i + 1;
      let v0 = f32(x[d0]);
      let v1 = f32(x[d1]);
      x[d0] = ${T}(v0 * cos_val - v1 * sin_val);
      x[d1] = ${T}(v0 * sin_val + v1 * cos_val);
    }
  }
}
`;

  const key = `rope3d_${T}_D${headDim}`;
  const shader = getOrCreateShader(device, key, code);
  const pipeline = getOrCreatePipeline(device, key, shader);

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: x.gpuBuffer! } },
      { binding: 1, resource: { buffer: coords.gpuBuffer! } },
      { binding: 2, resource: { buffer: cosBuf } },
      { binding: 3, resource: { buffer: sinBuf } },
      { binding: 4, resource: { buffer: dimsBuf } },
      { binding: 5, resource: { buffer: dimsBuf2 } },
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
  dimsBuf2.destroy();
  cosBuf.destroy();
  sinBuf.destroy();

  return x;
}
