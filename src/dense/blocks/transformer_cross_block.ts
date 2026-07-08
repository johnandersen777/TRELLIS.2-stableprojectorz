/**
 * Modulated Transformer Cross Block with adaLN (share_mod variant).
 *
 * Exact port of Python ModulatedTransformerCrossBlock._forward().
 *
 * Weights from safetensors (verified against ss_flow_img_dit_1_3B_64_bf16):
 *   blocks.N.self_attn.to_qkv.weight  — fused QKV (3*C, C) [1536→4608]
 *   blocks.N.self_attn.to_out.weight  — output projection (C, C)
 *   blocks.N.cross_attn.to_kv.weight  — fused KV (2*C, ctx_C) [1024→3072]
 *   blocks.N.cross_attn.to_q.weight   — cross-attn Q (C, C)
 *   blocks.N.mlp.mlp.0.weight         — FFN 0 (mlp_C, C) [8192→1536]
 *   blocks.N.mlp.mlp.2.weight         — FFN 2 (C, mlp_C)
 *   blocks.N.modulation               — learned (6*C,) param (share_mod=True)
 */

import { Tensor } from "../../runtime/tensor.ts";
import type { GPUContext, DType } from "../../runtime/device.ts";
import { linear } from "../ops/linear.ts";
import { add, mul } from "../ops/elementwise.ts";
import { silu } from "../ops/silu.ts";
import { gelu } from "../ops/gelu.ts";
import { rmsNorm } from "../ops/rms_norm.ts";
import { layerNorm } from "../ops/layer_norm.ts";
import { scaledDotProductAttention } from "../ops/attention.ts";
import { applyRoPE } from "../ops/rope.ts";

// ── Types ─────────────────────────────────────────────────────

export interface BlockConfig {
  numHeads: number;
  headDim: number;
  modelChannels: number;
  condChannels: number;
  mlpChannels: number;
  qkRmsNorm: boolean;
  qkRmsNormCross: boolean;
}

/** Split fused QKV weight (3*C, C) into 3 separate (C, C) weights on CPU. */
export function splitFusedQkv(fused: Float32Array, C: number): {
  qWeight: Tensor; kWeight: Tensor; vWeight: Tensor;
} {
  const perChunk = C * C;
  const qData = new Float32Array(fused.buffer, fused.byteOffset, perChunk);
  const kData = new Float32Array(fused.buffer, fused.byteOffset + perChunk * 4, perChunk);
  const vData = new Float32Array(fused.buffer, fused.byteOffset + perChunk * 8, perChunk);
  return {
    qWeight: Tensor.fromArray(new Float32Array(qData), [C, C], "float32"),
    kWeight: Tensor.fromArray(new Float32Array(kData), [C, C], "float32"),
    vWeight: Tensor.fromArray(new Float32Array(vData), [C, C], "float32"),
  };
}

/** Split fused QKV bias (3*C,) into 3 separate (C,) biases. */
export function splitFusedQkvBias(fused: Float32Array, C: number): {
  qBias: Tensor; kBias: Tensor; vBias: Tensor;
} {
  // Float32Array(buffer, byteOffset, elementCount) — byteOffset in bytes, count in elements
  return {
    qBias: Tensor.fromArray(new Float32Array(fused.buffer, fused.byteOffset, C), [C], "float32"),
    kBias: Tensor.fromArray(new Float32Array(fused.buffer, fused.byteOffset + C * 4, C), [C], "float32"),
    vBias: Tensor.fromArray(new Float32Array(fused.buffer, fused.byteOffset + C * 8, C), [C], "float32"),
  };
}

/** Split fused KV weight (2*C, ctx_C) into 2 separate (C, ctx_C) weights. */
export function splitFusedKv(fused: Float32Array, C: number, ctxC: number): {
  kWeight: Tensor; vWeight: Tensor;
} {
  const perChunk = C * ctxC;
  const kData = new Float32Array(fused.buffer, fused.byteOffset, perChunk);
  const vData = new Float32Array(fused.buffer, fused.byteOffset + perChunk * 4, perChunk);
  return {
    kWeight: Tensor.fromArray(new Float32Array(kData), [C, ctxC], "float32"),
    vWeight: Tensor.fromArray(new Float32Array(vData), [C, ctxC], "float32"),
  };
}

/** Split fused KV bias (2*C,) into K bias (C,) and V bias (C,). */
export function splitFusedKvBias(fused: Float32Array, C: number): {
  kBias: Tensor; vBias: Tensor;
} {
  return {
    kBias: Tensor.fromArray(new Float32Array(fused.buffer, fused.byteOffset, C), [C], "float32"),
    vBias: Tensor.fromArray(new Float32Array(fused.buffer, fused.byteOffset + C * 4, C), [C], "float32"),
  };
}

// ── Forward pass ──────────────────────────────────────────────

// Deferred disposal queue: accumulates tensors to dispose safely after
// GPU work completes. WebGPU doesn't allow destroying buffers while the
// GPU is reading from them, so we defer disposal until after
// onSubmittedWorkDone(). The caller owns this queue.
export type DisposeQueue = Tensor[];

export async function transformerCrossBlockForward(
  x: Tensor,
  modEmb: Tensor,
  context: Tensor,
  coords: Int32Array,
  saQWeight: Tensor, saKWeight: Tensor, saVWeight: Tensor,
  saQBias: Tensor | null, saKBias: Tensor | null, saVBias: Tensor | null,
  saOutWeight: Tensor, saOutBias: Tensor | null,
  saQNormWeight: Tensor | null, saKNormWeight: Tensor | null,
  caQWeight: Tensor, caQBias: Tensor | null,
  caKWeight: Tensor, caVWeight: Tensor,
  caKBias: Tensor | null, caVBias: Tensor | null,
  caOutWeight: Tensor, caOutBias: Tensor | null,
  caQNormWeight: Tensor | null, caKNormWeight: Tensor | null,
  norm2Weight: Tensor | null, norm2Bias: Tensor | null,
  ffn0Weight: Tensor, ffn0Bias: Tensor | null,
  ffn2Weight: Tensor, ffn2Bias: Tensor | null,
  modulation: Float32Array,
  config: BlockConfig,
  ctx: GPUContext,
  disposeQueue?: DisposeQueue,
): Promise<Tensor> {
  const device = ctx.device;
  const { numHeads, headDim, modelChannels } = config;
  const C = modelChannels;
  const H = numHeads;
  const D = headDim;
  const N = x.shape[0];
  const T = ctx.scalarType(x.dtype);

  // ── adaLN modulation ──
  // mod_emb (N, C) — all rows identical (timestep broadcast)
  // modulation (6*C,) — learned per-block param
  // mod_base = modulation + mod_emb[0]
  // chunk(6) → shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp

  // Get mod_emb[0] as CPU slice
  const modEmbData = modEmb.device === "cpu"
    ? modEmb.getView() as Float32Array
    : (await modEmb.toCPU()).getView() as Float32Array;
  const mod0 = modEmbData.subarray(0, C); // first row

  // mod_base = modulation + mod0
  const modBase = new Float32Array(6 * C);
  for (let i = 0; i < 6 * C; i++) {
    modBase[i] = modulation[i] + mod0[i % C]; // modulation is (6*C,), mod0 is (C,)
    // Actually: modulation[i] + mod0[...]
    // mod_emb[0] is (C,) — need to add to each chunk separately
  }
  // Correct: modulation[i*C + j] + mod0[j] for chunk i
  for (let chunk = 0; chunk < 6; chunk++) {
    for (let j = 0; j < C; j++) {
      modBase[chunk * C + j] = modulation[chunk * C + j] + mod0[j];
    }
  }
  // 6 modulation vectors, each (C,)
  const shiftSa = modBase.subarray(0, C);
  const scaleSa = modBase.subarray(C, 2 * C);
  const gateSa = modBase.subarray(2 * C, 3 * C);
  const shiftMlp = modBase.subarray(3 * C, 4 * C);
  const scaleMlp = modBase.subarray(4 * C, 5 * C);
  const gateMlp = modBase.subarray(5 * C, 6 * C);

  // Upload modulation vectors to GPU as (N, C) broadcast tensors
  async function broadcastVec(vec: Float32Array): Promise<Tensor> {
    const t = Tensor.fromArray(vec, [1, C], "float32");
    await t.upload(ctx);
    return t;
  }

  // ── 1. Self-attention ──────────────────────────────────
  // norm1(x): LayerNorm32 without affine (no weight/bias)
  let h = layerNorm(x, null, null, 1e-6, ctx);

  // Modulation: h = h * (1 + scale_sa) + shift_sa
  // scale_sa is (C,) → broadcast to (N, C) via reshape (1, C)
  let scale1 = Tensor.fromArray(new Float32Array(C), [1, C], "float32");
  let shift1 = Tensor.fromArray(new Float32Array(C), [1, C], "float32");
  {
    const sv = scale1.getView() as Float32Array;
    const shv = shift1.getView() as Float32Array;
    for (let i = 0; i < C; i++) {
      sv[i] = 1.0 + scaleSa[i]; // (1 + scale) for broadcast
      shv[i] = shiftSa[i];
    }
  }
  await scale1.upload(ctx); await shift1.upload(ctx);
  h = mul(h, scale1, ctx); // broadcasts (1,C) → (N,C)
  h = add(h, shift1, ctx);
  if (disposeQueue) { disposeQueue.push(scale1, shift1); } else { scale1.dispose(); shift1.dispose(); }

  // QKV projection: q = h @ W_q^T + b_q, k = h @ W_k^T + b_k, v = h @ W_v^T + b_v
  let q = await linear(h, saQWeight, saQBias, ctx);
  let k = await linear(h, saKWeight, saKBias, ctx);
  let v = await linear(h, saVWeight, saVBias, ctx);

  // Reshape (N, C) → (N, H, D) for attention
  q = q.reshape([N, H, D]);
  k = k.reshape([N, H, D]);
  v = v.reshape([N, H, D]);

  // RMS norm on Q and K (qk_rms_norm) — applied per-head across head_dim
  // Reshape to (N*H, D), apply norm, reshape back
  if (saQNormWeight) {
    q = q.reshape([N * H, D]);
    q = rmsNorm(q, saQNormWeight, 1e-6, ctx);
    q = q.reshape([N, H, D]);
  }
  if (saKNormWeight) {
    k = k.reshape([N * H, D]);
    k = rmsNorm(k, saKNormWeight, 1e-6, ctx);
    k = k.reshape([N, H, D]);
  }

  // Apply RoPE to Q and K
  if (coords) {
    const coordsTensor = Tensor.fromArray(coords, [N, 3], "int32");
    await coordsTensor.upload(ctx);
    const qPre = q, kPre = k;
    q = applyRoPE({ x: q, coords: coordsTensor, numHeads: H, headDim: D, maxCoord: 31 }, ctx);
    k = applyRoPE({ x: k, coords: coordsTensor, numHeads: H, headDim: D, maxCoord: 31 }, ctx);
    if (disposeQueue) { disposeQueue.push(qPre, kPre, coordsTensor); } else { qPre.dispose(); kPre.dispose(); coordsTensor.dispose(); }
  }

  // Scaled dot-product attention
  let o = scaledDotProductAttention(
    { Q: q, K: k, V: v, numHeads: H, batchSize: 1 },
    ctx,
  );

  // Output projection: (N, H, D) → reshape → (N, C) → linear
  o = o.reshape([N, C]);
  const oPreProj = o;
  o = await linear(o, saOutWeight, saOutBias, ctx);
  if (disposeQueue) disposeQueue.push(oPreProj);

  // Residual with gate: x = x + gate_sa * o
  let gateSaTensor = Tensor.fromArray(new Float32Array(C), [1, C], "float32");
  { const gv = gateSaTensor.getView() as Float32Array; for (let i = 0; i < C; i++) gv[i] = gateSa[i]; }
  await gateSaTensor.upload(ctx);
  o = mul(o, gateSaTensor, ctx);
  x = add(x, o, ctx);
  if (disposeQueue) { disposeQueue.push(gateSaTensor, q, k, v, o); } else { gateSaTensor.dispose(); q.dispose(); k.dispose(); v.dispose(); o.dispose(); }

  // ── 2. Cross-attention ─────────────────────────────────
  // norm2(x) with affine (weight + bias from safetensors blocks.N.norm2.*)
  h = layerNorm(x, norm2Weight, norm2Bias, 1e-6, ctx);

  // Q projection
  q = await linear(h, caQWeight, caQBias, ctx);
  q = q.reshape([N, H, D]);

  // KV projection from context
  k = await linear(context, caKWeight, caKBias, ctx);
  v = await linear(context, caVWeight, caVBias, ctx);
  const ctxLen = context.shape[0];
  k = k.reshape([ctxLen, H, D]);
  v = v.reshape([ctxLen, H, D]);

  if (caQNormWeight) { q = rmsNorm(q, caQNormWeight, 1e-6, ctx); }
  if (caKNormWeight) { k = rmsNorm(k, caKNormWeight, 1e-6, ctx); }

  // Cross-attention
  o = scaledDotProductAttention(
    { Q: q, K: k, V: v, numHeads: H, batchSize: 1 },
    ctx,
  );

  const oPreProjCA = o.reshape([N, C]);
  o = await linear(oPreProjCA, caOutWeight, caOutBias, ctx);
  if (disposeQueue) disposeQueue.push(oPreProjCA);

  // Residual (NO gate on cross-attention in Python)
  x = add(x, o, ctx);
  if (disposeQueue) { disposeQueue.push(q, k, v, o); } else { q.dispose(); k.dispose(); v.dispose(); o.dispose(); }

  // ── 3. Feed-forward ────────────────────────────────────
  // norm3(x) without affine
  h = layerNorm(x, null, null, 1e-6, ctx);

  // Modulation: h = h * (1 + scale_mlp) + shift_mlp
  let scaleM = Tensor.fromArray(new Float32Array(C), [1, C], "float32");
  let shiftM = Tensor.fromArray(new Float32Array(C), [1, C], "float32");
  {
    const sv = scaleM.getView() as Float32Array;
    const shv = shiftM.getView() as Float32Array;
    for (let i = 0; i < C; i++) {
      sv[i] = 1.0 + scaleMlp[i];
      shv[i] = shiftMlp[i];
    }
  }
  await scaleM.upload(ctx); await shiftM.upload(ctx);
  h = mul(h, scaleM, ctx);
  h = add(h, shiftM, ctx);
  if (disposeQueue) { disposeQueue.push(scaleM, shiftM); } else { scaleM.dispose(); shiftM.dispose(); }

  // FFN: Linear → GELU → Linear
  h = await linear(h, ffn0Weight, ffn0Bias, ctx);
  h = gelu(h, ctx);
  const hPreFfn2 = h;
  h = await linear(h, ffn2Weight, ffn2Bias, ctx);
  if (disposeQueue) disposeQueue.push(hPreFfn2);

  // Residual with gate: x = x + gate_mlp * h
  let gateMlpTensor = Tensor.fromArray(new Float32Array(C), [1, C], "float32");
  { const gv = gateMlpTensor.getView() as Float32Array; for (let i = 0; i < C; i++) gv[i] = gateMlp[i]; }
  await gateMlpTensor.upload(ctx);
  h = mul(h, gateMlpTensor, ctx);
  x = add(x, h, ctx);
  if (disposeQueue) { disposeQueue.push(gateMlpTensor, h); } else { gateMlpTensor.dispose(); h.dispose(); }

  // Cleanup modulation tensors — defer to dispose queue
  if (disposeQueue) {
    if (scale1) disposeQueue.push(scale1);
    if (shift1) disposeQueue.push(shift1);
    if (scaleM) disposeQueue.push(scaleM);
    if (shiftM) disposeQueue.push(shiftM);
  } else {
    scale1?.dispose(); shift1?.dispose();
    scaleM?.dispose(); shiftM?.dispose();
  }

  return x;
}
