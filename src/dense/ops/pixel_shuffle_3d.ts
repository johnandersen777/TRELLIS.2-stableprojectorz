/**
 * 3D Pixel Shuffle (depth-to-space) WGSL kernel.
 *
 * Rearranges (B, C*r^3, D, H, W) → (B, C, D*r, H*r, W*r) where r=2.
 * Standard operation in Conv3d decoder upsample blocks.
 *
 * One thread per output element. Shared-memory not needed (element-wise permute).
 */

import { Tensor } from "../../runtime/tensor.ts";
import type { GPUContext } from "../../runtime/device.ts";
import { getOrCreateShader, getOrCreatePipeline } from "../../runtime/shader_cache.ts";

export function pixelShuffle3d(
  x: Tensor,
  outChannels: number,
  upscaleFactor: number,
  ctx: GPUContext,
): Tensor {
  const device = ctx.device;
  const T = ctx.scalarType(x.dtype);
  // x shape: (in_C, D, H, W) where in_C = out_C * r^3
  const r = upscaleFactor; // 2
  const r3 = r * r * r; // 8
  const [inC, D, H, W] = [x.shape[0], x.shape[1], x.shape[2], x.shape[3]];
  const outC = outChannels;
  const outD = D * r;
  const outH = H * r;
  const outW = W * r;
  const outSpatial = outD * outH * outW;
  const N = outC * outSpatial;

  const code = `${ctx.precisionPrelude()}
@group(0) @binding(0) var<storage, read> input: array<${T}>;
@group(0) @binding(1) var<storage, read_write> output: array<${T}>;
@group(0) @binding(2) var<uniform> dims: vec4<u32>; // outC, D, H, W, r

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= ${N}u) { return; }

  let outC = dims.x;
  let D = dims.y;
  let H = dims.z;
  let W = dims.w;
  let r = ${r}u;

  // Decode output index → (c_out, d_out, h_out, w_out)
  let outD = D * r;
  let outH = H * r;
  let outW = W * r;
  let spat = outH * outW * outD;
  let c_out = idx / spat;
  let rem = idx % spat;
  let d_out = rem / (outH * outW);
  let rem2 = rem % (outH * outW);
  let h_out = rem2 / outW;
  let w_out = rem2 % outW;

  // Map to input: (c_in, d, h, w)
  let d = d_out / r;
  let h = h_out / r;
  let w = w_out / r;
  let dd = d_out % r;
  let dh = h_out % r;
  let dw = w_out % r;

  let c_offset = dd * r * r + dh * r + dw;
  let c_in = c_out * ${r3}u + c_offset;

  let in_idx = ((c_in * D + d) * H + h) * W + w;
  output[idx] = input[in_idx];
}
`;

  const key = `pixel_shuffle_3d_r${r}_${T}`;
  const shader = getOrCreateShader(device, key, code);
  const pipeline = getOrCreatePipeline(device, key, shader);

  const dimsData = new Uint32Array([outC, D, H, W, r]);
  const dimsBuf = device.createBuffer({
    size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(dimsBuf, 0, dimsData);

  const outBuf = device.createBuffer({
    size: N * (T === "f16" ? 2 : 4),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: x.gpuBuffer! } },
      { binding: 1, resource: { buffer: outBuf } },
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
  return Tensor.fromGpuBuffer(outBuf, [outC, outD, outH, outW], x.dtype, ctx, "ps3d_out");
}
