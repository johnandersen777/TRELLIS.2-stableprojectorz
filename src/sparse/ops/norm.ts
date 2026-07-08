/**
 * Sparse normalization ops: LayerNorm, GroupNorm.
 * Applies normalization independently to each voxel's feature vector.
 */
import { Tensor } from "../../runtime/tensor.ts";
import { GPUContext } from "../../runtime/device.ts";
import { getOrCreateShader, getOrCreatePipeline } from "../../runtime/shader_cache.ts";

/** In-place LayerNorm on a dense feature tensor (N, C) */
export function layerNorm(
  feats: Tensor,
  weight: Tensor | null,
  bias: Tensor | null,
  eps: number,
  context: GPUContext,
): Tensor {
  const device = context.device;
  const N = feats.shape[0];
  const C = feats.shape[1] || 1;
  const T = context.scalarType(feats.dtype);
  const enableF16 = context.supportsFP16 && T === "f16";

  const code = `${context.precisionPrelude()}
@group(0) @binding(0) var<storage, read_write> feats: array<${T}>;
@group(0) @binding(1) var<storage, read> weight: array<${T}>;
@group(0) @binding(2) var<storage, read> bias: array<${T}>;
@group(0) @binding(3) var<uniform> dims: vec4<u32>; // N, C, eps_f32_bits, 0

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

  const cacheKey = `layernorm_${T}`;
  const shader = getOrCreateShader(device, cacheKey, code);
  const pipeline = getOrCreatePipeline(device, cacheKey, shader);

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

  // Create output buffer (storage, copy results back)
  const outSize = feats.byteLength;
  const outBuf = device.createBuffer({
    size: outSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Copy input to output first
  const copyEnc = device.createCommandEncoder();
  copyEnc.copyBufferToBuffer(
    feats.gpuBuffer!,
    0,
    outBuf,
    0,
    outSize,
  );
  device.queue.submit([copyEnc.finish()]);

  // Create ones/zeros if no weight/bias — temp buffers MUST be destroyed after submit
  // to avoid leaking untracked GPU buffers (root cause of mapAsync bug).
  let wBuf: GPUBuffer;
  let bBuf: GPUBuffer;
  let wBufOwned = false;
  let bBufOwned = false;
  if (weight?.gpuBuffer) {
    wBuf = weight.gpuBuffer;
  } else {
    wBuf = createOnesBuf(device, C, feats.dtype);
    wBufOwned = true;
  }
  if (bias?.gpuBuffer) {
    bBuf = bias.gpuBuffer;
  } else {
    bBuf = createZerosBuf(device, C, feats.dtype);
    bBufOwned = true;
  }

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: outBuf } },
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

  return Tensor.fromGpuBuffer(outBuf, [N, C], feats.dtype, context, "layernorm_out");
}

function createOnesBuf(device: GPUDevice, size: number, _dt: string): GPUBuffer {
  const buf = device.createBuffer({
    size: size * 4,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  new Float32Array(buf.getMappedRange()).fill(1);
  buf.unmap();
  return buf;
}

function createZerosBuf(device: GPUDevice, size: number, _dt: string): GPUBuffer {
  const buf = device.createBuffer({
    size: size * 4,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  new Float32Array(buf.getMappedRange()).fill(0);
  buf.unmap();
  return buf;
}
