/**
 * SparseStructureFlowModel — dense 3D DiT for structure generation.
 *
 * Architecture: InputLinear → RoPE → 30× ModulatedTransformerCrossBlock → LN → OutputLinear.
 * Grid: 32³ = 32768 tokens, C=1536, H=12, D=128.
 * Weights: ~640 tensors from ss_flow_img_dit_1_3B_64_bf16.safetensors.
 */

import { Tensor } from "../runtime/tensor.ts";
import type { GPUContext } from "../runtime/device.ts";
import { matmul } from "../runtime/ops/matmul.ts";
import { add, mul } from "../dense/ops/elementwise.ts";
import { linear } from "../dense/ops/linear.ts";
import { rmsNorm } from "../dense/ops/rms_norm.ts";
import { silu } from "../dense/ops/silu.ts";
import { gelu } from "../dense/ops/gelu.ts";
import {
  transformerCrossBlockForward,
  splitFusedQkv,
  splitFusedQkvBias,
  splitFusedKv,
  splitFusedKvBias,
  type BlockConfig,
  type DisposeQueue,
} from "../dense/blocks/transformer_cross_block.ts";
import { scaledDotProductAttention } from "../dense/ops/attention.ts";
import type { LoadedWeights } from "../model/loader.ts";

// ── Config ────────────────────────────────────────────────────

// Module-level weight cache for SS_Flow — persists across forward passes.
// Key: weight name (e.g. "blocks.0.self_attn.to_qkv.weight")
// Value: { uploaded GPU weight tensor, uploaded GPU bias tensor | null, context }
let ssFlowWeightCache: Map<string, {weight: Tensor, bias: Tensor | null, ctx: GPUContext}> | null = null;
let ssFlowWeightCacheCtx: GPUContext | null = null;

export interface SSFlowConfig {
  resolution: number;       // 16 (= grid after VAE encoding, actual input is res)
  inChannels: number;       // 8
  outChannels: number;      // 8
  modelChannels: number;    // 1536
  condChannels: number;     // 1024
  numBlocks: number;        // 30
  numHeads: number;         // 12
  mlpRatio: number;         // 5.3334
  peMode: "rope" | "ape";
  shareMod: boolean;
  qkRmsNorm: boolean;
  qkRmsNormCross: boolean;
}

export function ssFlowConfig(args: Record<string, unknown>): SSFlowConfig {
  return {
    resolution: args.resolution as number,
    inChannels: args.in_channels as number,
    outChannels: args.out_channels as number,
    modelChannels: args.model_channels as number,
    condChannels: args.cond_channels as number,
    numBlocks: args.num_blocks as number,
    numHeads: args.num_heads as number,
    mlpRatio: args.mlp_ratio as number,
    peMode: (args.pe_mode as string ?? "rope") as "rope" | "ape",
    shareMod: (args.share_mod as boolean) ?? true,
    qkRmsNorm: (args.qk_rms_norm as boolean) ?? true,
    qkRmsNormCross: (args.qk_rms_norm_cross as boolean) ?? true,
  };
}

// ── Forward pass ──────────────────────────────────────────────

/**
 * Complete SS_Flow model forward pass.
 *
 * @param x — (B, in_channels, R, R, R) dense input (noise) on GPU
 * @param t — (B,) float timesteps in [0, 1000]
 * @param cond — (B, M, cond_channels) image condition tokens on GPU
 * @param negCond — (B, M, cond_channels) negative cond (zeros) or null on GPU
 * @param weights — loaded model weights from safetensors
 */
export async function ssFlowForward(
  x: Tensor,
  t: Float32Array,
  cond: Tensor,
  negCond: Tensor | null,
  weights: LoadedWeights,
  cfg: SSFlowConfig,
  ctx: GPUContext,
): Promise<Tensor> {
  const device = ctx.device;
  const getTensor = weights.getTensor;
  const {
    inChannels, outChannels, modelChannels, condChannels,
    numBlocks, numHeads, mlpRatio, resolution,
  } = cfg;
  const C = modelChannels;
  const H = numHeads;
  const D = C / H; // 128
  const mlpC = Math.floor(C * mlpRatio); // 8192
  const R = resolution;
  const N = x.shape[0] * R * R * R; // total tokens (B*R^3)

  console.log(`SS_Flow: N=${N}, C=${C}, H=${H}, D=${D}, blocks=${numBlocks}`);

  // Flatten cond from (B, M, cond_C) to (M, cond_C)
  const condFlat = cond.reshape([cond.shape[1], condChannels]);

  // ── 1. Input layer: Linear(x) = x @ W^T + b ──
  let h = x.reshape([N, inChannels]);
  h = await linear(h, getTensor("input_layer.weight"), getTensor("input_layer.bias"), ctx);

  // ── 2. Timestep embedding (CPU) ──
  // Sinusoidal embedding + 2-layer MLP with SiLU
  const tEmb = timestepEmbed(
    t,
    C, // embedding dim = model_channels
    getTensor("t_embedder.mlp.0.weight"),
    getTensor("t_embedder.mlp.0.bias"),
    getTensor("t_embedder.mlp.2.weight"),
    getTensor("t_embedder.mlp.2.bias"),
  );
  // Keep timestep embedding as CPU (1, C) — only used for modulation computation
  const tEmbTensor = Tensor.fromArray(tEmb, [1, C], "float32");
  // Note: NOT uploaded to GPU — transformer block reads CPU data for modulation

  // ── 3. Generate coordinate grid ──
  // (R, R, R) → coords array (N, 3) for RoPE
  const coords = new Int32Array(N * 3);
  let idx = 0;
  for (let z = 0; z < R; z++) {
    for (let y = 0; y < R; y++) {
      for (let x = 0; x < R; x++) {
        coords[idx * 3 + 0] = x;
        coords[idx * 3 + 1] = y;
        coords[idx * 3 + 2] = z;
        idx++;
      }
    }
  }

  // Block config
  const blockConfig: BlockConfig = {
    numHeads: H,
    headDim: D,
    modelChannels: C,
    condChannels: condChannels,
    mlpChannels: mlpC,
    qkRmsNorm: cfg.qkRmsNorm,
    qkRmsNormCross: cfg.qkRmsNormCross,
  };

  // ── 5. Transformer blocks × N ──
  const disposeQueue: DisposeQueue = [];

  // Cache weights across forward passes — getTensor() creates fresh CPU tensors
  // every call, and upload() creates new GPU buffers. Without caching, 22 forward
  // passes × 30 blocks × 28 weights = 18,480 GPU buffer creations, which exhausts
  // wgpu-native's internal state and causes device loss.
  // Static cache: key = weight name, value = uploaded GPU tensor
  const weightCache = ssFlowWeightCache ??= new Map<string, {weight: Tensor, bias: Tensor | null, ctx: GPUContext}>();
  // If context changed (device recycled), clear cache
  if (ssFlowWeightCacheCtx !== ctx) {
    for (const v of weightCache.values()) { v.weight.dispose(); v.bias?.dispose(); }
    weightCache.clear();
    ssFlowWeightCacheCtx = ctx;
  }

  for (let b = 0; b < numBlocks; b++) {
    const p = `blocks.${b}.`;

    // Load weights — use cache for split weights (QKV, KV) and non-split weights
    // to avoid re-creating GPU buffers every forward pass.
    let qWeight: Tensor, kWeight: Tensor, vWeight: Tensor;
    let qBias: Tensor, kBias: Tensor, vBias: Tensor;
    const saQkvKey = p + "self_attn.to_qkv";
    const cachedQkv = weightCache.get(saQkvKey);
    if (cachedQkv) {
      // Use cached split weights — stored as a single concatenated tensor
      // Actually, we store each split separately. Let me use a different approach:
      // store all 6 split tensors in the cache. But the cache only holds {weight, bias}.
      // For QKV, weight = [qWeight, kWeight, vWeight] concatenated? No.
      // Simpler: use separate cache entries for each split.
      qWeight = weightCache.get(saQkvKey + ".q_weight")!.weight;
      kWeight = weightCache.get(saQkvKey + ".k_weight")!.weight;
      vWeight = weightCache.get(saQkvKey + ".v_weight")!.weight;
      qBias = weightCache.get(saQkvKey + ".q_bias")!.weight;
      kBias = weightCache.get(saQkvKey + ".k_bias")!.weight;
      vBias = weightCache.get(saQkvKey + ".v_bias")!.weight;
    } else {
      const saQkvData = getTensor(p + "self_attn.to_qkv.weight").getView() as Float32Array;
      const saQkvBiasData = getTensor(p + "self_attn.to_qkv.bias").getView() as Float32Array;
      const split = splitFusedQkv(saQkvData, C);
      const splitBias = splitFusedQkvBias(saQkvBiasData, C);
      qWeight = split.qWeight; kWeight = split.kWeight; vWeight = split.vWeight;
      qBias = splitBias.qBias; kBias = splitBias.kBias; vBias = splitBias.vBias;
      await qWeight.upload(ctx); await kWeight.upload(ctx); await vWeight.upload(ctx);
      await qBias.upload(ctx); await kBias.upload(ctx); await vBias.upload(ctx);
      weightCache.set(saQkvKey + ".q_weight", {weight: qWeight, bias: null, ctx});
      weightCache.set(saQkvKey + ".k_weight", {weight: kWeight, bias: null, ctx});
      weightCache.set(saQkvKey + ".v_weight", {weight: vWeight, bias: null, ctx});
      weightCache.set(saQkvKey + ".q_bias", {weight: qBias, bias: null, ctx});
      weightCache.set(saQkvKey + ".k_bias", {weight: kBias, bias: null, ctx});
      weightCache.set(saQkvKey + ".v_bias", {weight: vBias, bias: null, ctx});
    }

    let caKWeight: Tensor, caVWeight: Tensor;
    let caKBias: Tensor, caVBias: Tensor;
    const caKvKey = p + "cross_attn.to_kv";
    if (weightCache.has(caKvKey + ".k_weight")) {
      caKWeight = weightCache.get(caKvKey + ".k_weight")!.weight;
      caVWeight = weightCache.get(caKvKey + ".v_weight")!.weight;
      caKBias = weightCache.get(caKvKey + ".k_bias")!.weight;
      caVBias = weightCache.get(caKvKey + ".v_bias")!.weight;
    } else {
      const caKvData = getTensor(p + "cross_attn.to_kv.weight").getView() as Float32Array;
      const caKvBiasRaw = getTensor(p + "cross_attn.to_kv.bias").getView() as Float32Array;
      const splitKv = splitFusedKv(caKvData, C, condChannels);
      const splitKvBias = splitFusedKvBias(caKvBiasRaw, C);
      caKWeight = splitKv.kWeight; caVWeight = splitKv.vWeight;
      caKBias = splitKvBias.kBias; caVBias = splitKvBias.vBias;
      await caKWeight.upload(ctx); await caVWeight.upload(ctx);
      await caKBias.upload(ctx); await caVBias.upload(ctx);
      weightCache.set(caKvKey + ".k_weight", {weight: caKWeight, bias: null, ctx});
      weightCache.set(caKvKey + ".v_weight", {weight: caVWeight, bias: null, ctx});
      weightCache.set(caKvKey + ".k_bias", {weight: caKBias, bias: null, ctx});
      weightCache.set(caKvKey + ".v_bias", {weight: caVBias, bias: null, ctx});
    }

    // Non-split weights — cache by name
    function getOrUpload(name: string): Tensor {
      const key = p + name;
      const cached = weightCache.get(key);
      if (cached) return cached.weight;
      const t = getTensor(key);
      // Note: upload is async but we can't await in a sync function.
      // We'll upload below.
      return t;
    }

    // Load non-split weights — use cache or upload
    let saOutWeight: Tensor, saOutBias: Tensor;
    let saQNorm: Tensor, saKNorm: Tensor;
    let caQWeight: Tensor, caQBias: Tensor;
    let caOutWeight: Tensor, caOutBias: Tensor;
    let caQNorm: Tensor, caKNorm: Tensor;
    let norm2Weight: Tensor, norm2Bias: Tensor;
    let ffn0Weight: Tensor, ffn0Bias: Tensor;
    let ffn2Weight: Tensor, ffn2Bias: Tensor;
    let modulation: Float32Array;

    const nonSplitNames = [
      "self_attn.to_out.weight", "self_attn.to_out.bias",
      "self_attn.q_rms_norm.gamma", "self_attn.k_rms_norm.gamma",
      "cross_attn.to_q.weight", "cross_attn.to_q.bias",
      "cross_attn.to_out.weight", "cross_attn.to_out.bias",
      "cross_attn.q_rms_norm.gamma", "cross_attn.k_rms_norm.gamma",
      "norm2.weight", "norm2.bias",
      "mlp.mlp.0.weight", "mlp.mlp.0.bias",
      "mlp.mlp.2.weight", "mlp.mlp.2.bias",
    ];

    const nonSplitTensors: Tensor[] = [];
    const toUpload: Tensor[] = [];
    for (const name of nonSplitNames) {
      const key = p + name;
      const cached = weightCache.get(key);
      if (cached) {
        nonSplitTensors.push(cached.weight);
      } else {
        const t = getTensor(key);
        nonSplitTensors.push(t);
        toUpload.push(t);
      }
    }
    // Upload new tensors sequentially
    for (const t of toUpload) await t.upload(ctx);
    // Cache them
    for (let i = 0; i < nonSplitNames.length; i++) {
      const key = p + nonSplitNames[i];
      if (!weightCache.has(key)) {
        weightCache.set(key, {weight: nonSplitTensors[i], bias: null, ctx});
      }
    }

    [saOutWeight, saOutBias, saQNorm, saKNorm,
     caQWeight, caQBias, caOutWeight, caOutBias,
     caQNorm, caKNorm, norm2Weight, norm2Bias,
     ffn0Weight, ffn0Bias, ffn2Weight, ffn2Bias] = nonSplitTensors;

    modulation = getTensor(p + "modulation").getView() as Float32Array;

    h = await transformerCrossBlockForward(
      h, tEmbTensor, condFlat, coords,
      qWeight, kWeight, vWeight,
      qBias, kBias, vBias,
      saOutWeight, saOutBias,
      saQNorm, saKNorm,
      caQWeight, caQBias,
      caKWeight, caVWeight, caKBias, caVBias,
      caOutWeight, caOutBias,
      caQNorm, caKNorm,
      norm2Weight, norm2Bias,
      ffn0Weight, ffn0Bias,
      ffn2Weight, ffn2Bias,
      modulation, blockConfig, ctx,
      disposeQueue,
    );

    console.log(`  Block ${b + 1}/${numBlocks} done`);

    // TRELLIS.2 fix: Keep weight GPU buffers across ALL blocks.
    // The old per-block dispose+recreate cycle (28 creates + 28 destroys
    // per block) was causing D3D12 "Parent device is lost" at block 5
    // due to cumulative buffer lifecycle churn in wgpu-core.
    //
    // With the other fixes (no device_poll unwrap, proper error surfacing),
    // 840 concurrent buffers (28 × 30 blocks) is stable, whereas the
    // create/destroy cycle was corrupting internal wgpu-core state.
    //
    // Weight cache persists for entire forward pass. Cleared at end.
  }

  // ── 6. Output norm + linear ──
  h = rmsNorm(h, null, 1e-6, ctx); // in-place
  h = await linear(h, getTensor("out_layer.weight"), getTensor("out_layer.bias"), ctx);

  // Reshape back to (B, out_C, R, R, R)
  h = h.reshape([x.shape[0], outChannels, R, R, R]);

  // TRELLIS.2 fix: Wait for GPU to finish all submitted work BEFORE
  // destroying any buffers. Without this, buffer destruction races with
  // GPU execution, causing "Parent device is lost" in wgpu-core.
  try {
    await ctx.device.queue.onSubmittedWorkDone();
  } catch {
    // onSubmittedWorkDone may fail if device is already in bad state.
    // Continue with cleanup anyway.
  }

  // Cleanup: dispose weight cache (all 840 GPU buffers) — once at end.
  for (const key of weightCache.keys()) {
    weightCache.get(key)?.weight.dispose();
    weightCache.get(key)?.bias?.dispose();
  }
  weightCache.clear();
  ssFlowWeightCache = null;
  ssFlowWeightCacheCtx = null;

  // Cleanup: dispose intermediate tensors accumulated in the dispose queue.
  for (const t of disposeQueue) {
    t.dispose();
  }
  disposeQueue.length = 0;
  tEmbTensor.dispose();

  console.log("SS_Flow forward complete");
  return h;
}

// ── Timestep embedder (CPU) ───────────────────────────────────

function sinusoidalEmbedding(t: Float32Array, dim: number): Float32Array {
  const B = t.length;
  const out = new Float32Array(B * dim);
  const half = dim / 2;
  for (let b = 0; b < B; b++) {
    for (let i = 0; i < half; i++) {
      const freq = 1.0 / Math.pow(10000, (2 * i) / dim);
      const val = t[b] * freq;
      out[b * dim + 2 * i] = Math.sin(val);
      out[b * dim + 2 * i + 1] = Math.cos(val);
    }
  }
  return out;
}

function timestepEmbed(
  t: Float32Array,
  dim: number,
  w0: Tensor, b0: Tensor,
  w2: Tensor, b2: Tensor,
): Float32Array {
  const B = t.length;
  const modelC = b0.shape[0];
  const sinEmb = sinusoidalEmbedding(t, dim); // (B, dim)

  const w0Data = w0.getView() as Float32Array; // (dim, modelC)
  const b0Data = b0.getView() as Float32Array; // (modelC,)
  const w2Data = w2.getView() as Float32Array; // (modelC, modelC)
  const b2Data = b2.getView() as Float32Array; // (modelC,)

  // Layer 0: (B, dim) @ (dim, modelC) + bias → SiLU
  const h0 = new Float32Array(B * modelC);
  for (let b = 0; b < B; b++) {
    for (let j = 0; j < modelC; j++) {
      let sum = b0Data[j];
      for (let i = 0; i < dim; i++) {
        sum += sinEmb[b * dim + i] * w0Data[j * dim + i];
      }
      const sig = 1.0 / (1.0 + Math.exp(-sum));
      h0[b * modelC + j] = sum * sig;
    }
  }

  // Layer 2: (B, modelC) @ (modelC, modelC) + bias → SiLU
  const h2 = new Float32Array(B * modelC);
  for (let b = 0; b < B; b++) {
    for (let j = 0; j < modelC; j++) {
      let sum = b2Data[j];
      for (let i = 0; i < modelC; i++) {
        sum += h0[b * modelC + i] * w2Data[j * modelC + i];
      }
      const sig = 1.0 / (1.0 + Math.exp(-sum));
      h2[b * modelC + j] = sum * sig;
    }
  }

  return h2;
}
