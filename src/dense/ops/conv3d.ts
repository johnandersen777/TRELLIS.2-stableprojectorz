/**
 * Dense 3D convolution: standard Conv3d for dense volumetric grids.
 *
 * Weight layout: (Co, Ci, Kd, Kh, Kw) — PyTorch standard.
 * Uses SAME padding (output spatial size = input spatial size for K=3, S=1, P=1).
 *
 * For SS_Decoder: 32^3 grid, Ci~Co~32-128, K=3.
 */

import { Tensor } from "../../runtime/tensor.ts";
import type { GPUContext } from "../../runtime/device.ts";
import { getOrCreateShader, getOrCreatePipeline } from "../../runtime/shader_cache.ts";

export interface Conv3dConfig {
  inChannels: number;
  outChannels: number;
  kernelSize: [number, number, number]; // [Kd, Kh, Kw]
  stride: [number, number, number];     // [Sd, Sh, Sw]
  padding: [number, number, number];    // [Pd, Ph, Pw]
  inputShape: [number, number, number]; // [D, H, W]
}

export function makeConv3dShader(config: Conv3dConfig, scalarType: string): string {
  const [Kd, Kh, Kw] = config.kernelSize;
  const [D, H, W] = config.inputShape;
  const Ci = config.inChannels;
  const Co = config.outChannels;
  const T = scalarType;

  // Output spatial dims with SAME padding and stride 1
  const Od = D;
  const Oh = H;
  const Ow = W;
  const pad_d = Math.floor(Kd / 2);
  const pad_h = Math.floor(Kh / 2);
  const pad_w = Math.floor(Kw / 2);

  return `
const K_D = ${Kd}u; const K_H = ${Kh}u; const K_W = ${Kw}u;
const C_IN = ${Ci}u; const C_OUT = ${Co}u;
const IN_D = ${D}u; const IN_H = ${H}u; const IN_W = ${W}u;
const OUT_D = ${Od}u; const OUT_H = ${Oh}u; const OUT_W = ${Ow}u;
const PAD_D = ${pad_d}u; const PAD_H = ${pad_h}u; const PAD_W = ${pad_w}u;
const SPATIAL = OUT_D * OUT_H * OUT_W;

@group(0) @binding(0) var<storage, read> input: array<${T}>;
@group(0) @binding(1) var<storage, read> weight: array<${T}>;  // (Co, Ci, Kd, Kh, Kw)
@group(0) @binding(2) var<storage, read> bias: array<${T}>;     // (Co,)
@group(0) @binding(3) var<storage, read_write> output: array<${T}>;

// Index helpers
fn in_idx(ci: u32, d: u32, h: u32, w: u32) -> u32 {
  return ((ci * IN_D + d) * IN_H + h) * IN_W + w;
}
fn w_idx(co: u32, ci: u32, kd: u32, kh: u32, kw: u32) -> u32 {
  return ((((co * C_IN + ci) * K_D + kd) * K_H + kh) * K_W + kw);
}
fn out_idx(co: u32, d: u32, h: u32, w: u32) -> u32 {
  return ((co * OUT_D + d) * OUT_H + h) * OUT_W + w;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let co = wgid.x;
  if (co >= C_OUT) { return; }

  let bias_val = f32(bias[co]);

  // Each thread handles multiple spatial positions
  for (var idx = lid.x; idx < SPATIAL; idx += 256u) {
    // Decode spatial index → (d, h, w)
    let d = idx / (OUT_H * OUT_W);
    let remainder = idx % (OUT_H * OUT_W);
    let h = remainder / OUT_W;
    let w = remainder % OUT_W;

    var sum: f32 = bias_val;

    // Convolve over input channels and kernel
    for (var ci = 0u; ci < C_IN; ci++) {
      for (var kd = 0u; kd < K_D; kd++) {
        let in_d = d + kd - PAD_D; // d + kd - pad = d (for SAME padding with stride 1)
        // Skip padding checks for interior positions
        // For full SAME padding support: clamp or conditional
        if (in_d < IN_D) {
          for (var kh = 0u; kh < K_H; kh++) {
            let in_h = h + kh - PAD_H;
            if (in_h < IN_H) {
              for (var kw = 0u; kw < K_W; kw++) {
                let in_w = w + kw - PAD_W;
                if (in_w < IN_W) {
                  let w_val = f32(weight[w_idx(co, ci, kd, kh, kw)]);
                  let in_val = f32(input[in_idx(ci, in_d, in_h, in_w)]);
                  sum += w_val * in_val;
                }
              }
            }
          }
        }
      }
    }

    output[out_idx(co, d, h, w)] = ${T}(sum);
  }
}
`;
}

export function conv3dShaderKey(config: Conv3dConfig, scalarType: string): string {
  const [Kd, Kh, Kw] = config.kernelSize;
  return `conv3d_K${Kd}x${Kh}x${Kw}_Ci${config.inChannels}_Co${config.outChannels}_${scalarType}`;
}

/**
 * Dense 3D convolution forward pass.
 *
 * @param x — (B, Ci, D, H, W) input tensor on GPU
 * @param weight — (Co, Ci, Kd, Kh, Kw) weight tensor on GPU
 * @param bias — (Co,) bias tensor on GPU
 * @param config — convolution parameters
 */
export function conv3d(
  x: Tensor,
  weight: Tensor,
  bias: Tensor,
  config: Conv3dConfig,
  context: GPUContext,
): Tensor {
  const device = context.device;
  const T = context.scalarType(x.dtype);
  const [D, H, W] = config.inputShape;
  const Co = config.outChannels;
  const totalSpatial = D * H * W;

  const code = context.precisionPrelude() + makeConv3dShader(config, T);
  const key = conv3dShaderKey(config, T);
  const shader = getOrCreateShader(device, key, code);
  const pipeline = getOrCreatePipeline(device, key, shader);

  // Allocate output: (Co, D, H, W)
  const outSize = Co * totalSpatial * (T === "f16" ? 2 : 4);
  const outBuf = device.createBuffer({
    size: outSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: x.gpuBuffer! } },
      { binding: 1, resource: { buffer: weight.gpuBuffer! } },
      { binding: 2, resource: { buffer: bias.gpuBuffer! } },
      { binding: 3, resource: { buffer: outBuf } },
    ],
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Co, 1, 1); // one workgroup per output channel
  pass.end();
  device.queue.submit([enc.finish()]);

  return Tensor.fromGpuBuffer(outBuf, [Co, D, H, W], x.dtype, context, `conv3d_out`);
}
