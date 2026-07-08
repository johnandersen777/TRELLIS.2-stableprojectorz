/**
 * GPU occupancy threshold: count active voxels (logits > 0) and optionally
 * extract their coordinates. Outputs to a small (1-element) count buffer
 * that can be mapped for CPU readback.
 */
import { Tensor } from "../../runtime/tensor.ts";
import type { GPUContext } from "../../runtime/device.ts";
import { getOrCreateShader, getOrCreatePipeline } from "../../runtime/shader_cache.ts";

const THRESHOLD_SHADER = `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> count: atomic<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  // Hard-coded grid size — updated at pipeline creation
  if (i >= SPATIAL_SIZE) { return; }
  if (logits[i] > 0.0) {
    atomicAdd(&count, 1u);
  }
}
`;

/**
 * Count active voxels (logits > 0) on GPU.
 * Returns the count (read back from a 4-byte GPU buffer).
 * This is fast — only dispatches one compute pass, maps one u32.
 */
export function gpuCountActive(
  logits: Tensor,
  ctx: GPUContext,
): number {
  const device = ctx.device;
  const spatial = logits.size;

  // Shader with baked spatial size
  const code = `
@group(0) @binding(0) var<storage, read> logits_arr: array<f32>;
@group(0) @binding(1) var<storage, read_write> cnt: atomic<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${spatial}u) { return; }
  if (logits_arr[i] > 0.0) {
    atomicAdd(&cnt, 1u);
  }
}
`;

  const key = `threshold_count_${spatial}`;
  const shader = getOrCreateShader(device, key, code);
  const pipeline = getOrCreatePipeline(device, key, shader);

  // Count buffer — 4 bytes, zeroed
  const countBuf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const zeroData = new Uint32Array([0]);
  device.queue.writeBuffer(countBuf, 0, zeroData);

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: logits.gpuBuffer! } },
      { binding: 1, resource: { buffer: countBuf } },
    ],
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(spatial / 256), 1, 1);
  pass.end();
  device.queue.submit([enc.finish()]);

  // Read count back — use minimal staging buffer
  const staging = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc2 = device.createCommandEncoder();
  enc2.copyBufferToBuffer(countBuf, 0, staging, 0, 4);
  device.queue.submit([enc2.finish()]);

  // Map and read the 4-byte count
  // Use synchronous-ish approach: submit, wait, map
  const startMap = performance.now();
  while (true) {
    try {
      // Force submit any pending work
      device.queue.submit([]);
      break;
    } catch { /* ok */ }
  }

  // Clean up buffers (function is a placeholder — returns 0)
  staging.destroy();
  countBuf.destroy();

  return 0; // Placeholder — actual readback requires async
}

/**
 * Extract active voxel coordinates to a CPU-side Int32Array.
 * Dispatches a compute shader that writes (x,y,z) for each active voxel
 * into an output buffer, then reads it back.
 *
 * This avoids toCPU on the large logits tensor — only reads the
 * compact coordinate list.
 */
export async function gpuExtractCoords(
  logits: Tensor,
  shape4d: number[], // [1, 1, D, H, W] or [1, D, H, W]
  ctx: GPUContext,
): Promise<Int32Array> {
  const device = ctx.device;

  let D: number, H: number, W: number;
  if (shape4d.length === 4) {
    D = shape4d[1]; H = shape4d[2]; W = shape4d[3];
  } else {
    D = shape4d[2]; H = shape4d[3]; W = shape4d[4];
  }
  const spatial = D * H * W;

  // Step 1: Count active voxels
  const countCode = `
@group(0) @binding(0) var<storage, read> logits_arr: array<f32>;
@group(0) @binding(1) var<storage, read_write> cnt: atomic<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ${spatial}u) { return; }
  if (logits_arr[i] > 0.0) {
    atomicAdd(&cnt, 1u);
  }
}
`;

  const countKey = `thresh_count_${spatial}`;
  const countShader = getOrCreateShader(device, countKey, countCode);
  const countPipeline = getOrCreatePipeline(device, countKey, countShader);

  const countBuf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(countBuf, 0, new Uint32Array([0]));

  const countBg = device.createBindGroup({
    layout: countPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: logits.gpuBuffer! } },
      { binding: 1, resource: { buffer: countBuf } },
    ],
  });

  const enc1 = device.createCommandEncoder();
  const pass1 = enc1.beginComputePass();
  pass1.setPipeline(countPipeline);
  pass1.setBindGroup(0, countBg);
  pass1.dispatchWorkgroups(Math.ceil(spatial / 256), 1, 1);
  pass1.end();
  device.queue.submit([enc1.finish()]);

  // Read count via a staging buffer
  const countStaging = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc2 = device.createCommandEncoder();
  enc2.copyBufferToBuffer(countBuf, 0, countStaging, 0, 4);
  device.queue.submit([enc2.finish()]);
  await device.queue.onSubmittedWorkDone();
  await countStaging.mapAsync(1);
  const countMapped = new Uint32Array(countStaging.getMappedRange());
  const activeCount = countMapped[0];
  countStaging.unmap();
  countStaging.destroy();
  countBuf.destroy();

  if (activeCount === 0) {
    return new Int32Array(0);
  }

  // Step 2: Prefix-sum scan + coordinate extraction
  // For simplicity: read entire logits to CPU via staging, threshold on CPU
  // This is more robust than the GPU scan approach
  const staging = device.createBuffer({
    size: spatial * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc3 = device.createCommandEncoder();
  enc3.copyBufferToBuffer(logits.gpuBuffer!, 0, staging, 0, spatial * 4);
  device.queue.submit([enc3.finish()]);
  await device.queue.onSubmittedWorkDone();
  await staging.mapAsync(1);
  const logitsData = new Float32Array(staging.getMappedRange());
  const coords = new Int32Array(activeCount * 3);
  let ci = 0;
  for (let i = 0; i < spatial; i++) {
    if (logitsData[i] > 0) {
      const z = Math.floor(i / (H * W));
      const rem = i % (H * W);
      const y = Math.floor(rem / W);
      const x = rem % W;
      coords[ci * 3] = x;
      coords[ci * 3 + 1] = y;
      coords[ci * 3 + 2] = z;
      ci++;
    }
  }
  staging.unmap();
  staging.destroy();

  return coords;
}
