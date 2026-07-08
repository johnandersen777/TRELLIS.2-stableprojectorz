/**
 * Scaled dot-product attention: O = softmax(Q @ K^T / sqrt(D)) @ V
 *
 * Supports self-attention (Q=K=V sequence) and cross-attention (Q vs KV).
 * Each dispatch handles one query position with 256 cooperating threads.
 */

import { Tensor } from "../../runtime/tensor.ts";
import type { GPUContext } from "../../runtime/device.ts";
import { getOrCreateShader, getOrCreatePipeline } from "../../runtime/shader_cache.ts";
import {
  makeAttentionShader,
  attentionShaderKey,
  type AttentionShaderConfig,
} from "./attention_wgsl.ts";

export interface AttentionParams {
  /** Query tensor: shape [B, S_q, H, D] or [S_q, H, D] */
  Q: Tensor;
  /** Key tensor: shape [B, S_kv, H, D] or [S_kv, H, D] */
  K: Tensor;
  /** Value tensor: shape [B, S_kv, H, D] or [S_kv, H, D] */
  V: Tensor;
  /** Number of attention heads */
  numHeads: number;
  /** Batch size (1 for most TRELLIS.2 models) */
  batchSize?: number;
}

/**
 * Compute scaled dot-product attention on GPU.
 *
 * Dispatch: batchSize * numHeads * S_q workgroups.
 * Each workgroup handles one query position, 256 threads cooperate.
 */
export function scaledDotProductAttention(
  params: AttentionParams,
  context: GPUContext,
): Tensor {
  const device = context.device;
  const { Q, K, V, numHeads } = params;
  const B = params.batchSize ?? 1;
  const D = Q.shape[Q.shape.length - 1]; // head_dim
  const H = numHeads;
  // Shape is (S_q, H, D): S_q at dim 0, H at dim 1, D at dim -1
  const S_q = Q.shape[0];
  const S_kv = K.shape[0];

  // Validate shapes
  if (Q.size !== B * S_q * H * D) throw new Error(
    `Q shape mismatch: expected ${B*S_q*H*D} elements, got ${Q.size}`);
  if (K.size !== B * S_kv * H * D) throw new Error(
    `K shape mismatch: expected ${B*S_kv*H*D} elements, got ${K.size}`);
  if (V.size !== B * S_kv * H * D) throw new Error(
    `V shape mismatch: expected ${B*S_kv*H*D} elements, got ${V.size}`);

  const T = context.scalarType(Q.dtype);
  const enableF16 = context.supportsFP16 && T === "f16";

  // Scale = 1/sqrt(D) as f32 bits
  const scale = 1.0 / Math.sqrt(D);
  const scaleBits = new Float32Array([scale]);
  const scaleU32 = new Uint32Array(scaleBits.buffer)[0];

  // Config — compute TILE_K that fits within maxComputeWorkgroupStorageSize.
  // Workgroup storage = TILE_K * D * 4 (kv_sh) + TILE_K * 4 (dots) + 8 (scalars).
  // Must stay under adapter limit (typically 32768 bytes on D3D12).
  const maxWgStorage = context.maxComputeWorkgroupStorageSize;
  const tileK = Math.min(
    256,
    Math.floor((maxWgStorage - 8) / (D * 4 + 4)),
  );
  const workgroupSize = Math.min(256, tileK); // one thread per KV position

  const shaderConfig: AttentionShaderConfig = {
    headDim: D,
    scalarType: T as "f32" | "f16",
    tileK,
    workgroupSize,
  };

  const code = context.precisionPrelude() +
    makeAttentionShader(shaderConfig);
  const key = attentionShaderKey(shaderConfig);
  const shader = getOrCreateShader(device, key, code);
  const pipeline = getOrCreatePipeline(device, key, shader);

  // Allocate output
  const outSize = B * S_q * H * D * (T === "f16" ? 2 : 4);
  const outBuf = device.createBuffer({
    size: outSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Uniform buffer: [S_q, S_kv, H, scaleU32, workgroup_offset=0]
  const paramsData = new Uint32Array([S_q, S_kv, H, scaleU32, 0]);
  const paramsBuf = device.createBuffer({
    size: 32,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuf, 0, paramsData);

  // Bind group
  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: Q.gpuBuffer! } },
      { binding: 1, resource: { buffer: K.gpuBuffer! } },
      { binding: 2, resource: { buffer: V.gpuBuffer! } },
      { binding: 3, resource: { buffer: outBuf } },
      { binding: 4, resource: { buffer: paramsBuf } },
    ],
  });

  // Dispatch all workgroups in a single pass.
  //
  // Previously chunked into 6-12 submits to avoid Windows TDR. However,
  // each submit accumulates state in wgpu-native's internal tracking,
  // eventually corrupting the mapping subsystem (Deno #24798, #22146).
  //
  // A single 49,152-WG dispatch on a modern GPU completes in well under
  // 2 seconds — the TDR limit. The actual compute is ~200 GFLOPs, which at
  // 24 TFLOPS is ~8ms. Memory bandwidth adds time, but total should be
  // <500ms. Chunking was precautionary; removing it reduces wgpu-native
  // pressure and may prevent the mapAsync corruption.
  const totalWorkgroups = B * H * S_q;
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(totalWorkgroups, 1, 1);
  pass.end();
  device.queue.submit([enc.finish()]);

  paramsBuf.destroy();

  return Tensor.fromGpuBuffer(
    outBuf,
    [B, S_q, H, D],
    Q.dtype,
    context,
    `attn_out_${S_q}x${D}`,
  );
}
