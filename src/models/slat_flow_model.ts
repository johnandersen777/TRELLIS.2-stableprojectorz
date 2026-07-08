/**
 * SLatFlowModel — sparse DiT for shape/texture generation.
 *
 * Architecture: same as SS_Flow but operates on SparseTensor (coords + feats).
 * Input: SparseTensor(feats=[N, in_C], coords=[N,4]) → Output: SparseTensor([N, out_C])
 *
 * Architecture: SparseLinear(in) → 30× ModulatedSparseTransformerCrossBlock → LN → SparseLinear(out)
 * Self-attention: on active voxels only (N ≈ 10K-50K, much less than 32K)
 * Cross-attention: to image cond tokens (same as SS_Flow)
 *
 * 4 instances: shape_512, shape_1024, tex_512, tex_1024 (same arch, different weights/resolution)
 * Weights: slat_flow_img2shape_dit_1_3B_*.safetensors (2.41 GB each, identical architecture)
 */

import { Tensor } from "../runtime/tensor.ts";
import type { GPUContext } from "../runtime/device.ts";
import { matmul } from "../runtime/ops/matmul.ts";
import { add } from "../dense/ops/elementwise.ts";
import { rmsNorm } from "../dense/ops/rms_norm.ts";
import { layerNorm } from "../dense/ops/layer_norm.ts";
import { SparseTensor } from "../sparse/sparse_tensor.ts";
import { sparseLinear } from "../sparse/ops/linear.ts";
import {
  splitFusedQkv,
  splitFusedKv,
} from "../dense/blocks/transformer_cross_block.ts";
import {
  sparseAttention,
  type SparseAttentionParams,
} from "../sparse/ops/attention.ts";
import { scaledDotProductAttention } from "../dense/ops/attention.ts";
import { applyRoPE } from "../dense/ops/rope.ts";
import { silu } from "../dense/ops/silu.ts";
import { gelu } from "../dense/ops/gelu.ts";
import type { LoadedWeights } from "../model/loader.ts";

export interface SLatFlowConfig {
  resolution: number;
  inChannels: number;
  outChannels: number;
  modelChannels: number;
  condChannels: number;
  numBlocks: number;
  numHeads: number;
  mlpRatio: number;
  peMode: "rope" | "ape";
  shareMod: boolean;
  qkRmsNorm: boolean;
  qkRmsNormCross: boolean;
}

export function slatFlowConfig(args: Record<string, unknown>): SLatFlowConfig {
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

/**
 * SLatFlow forward pass — single step of the flow model.
 *
 * @param x — SparseTensor with feats (N, in_C), coords (N, 4)
 * @param t — scalar timestep in [0, 1000]
 * @param cond — (M, cond_C) image conditioning tokens
 * @param concatCond — optional SparseTensor to concatenate along channel dim (texture flow)
 * @param weights — loaded model weights
 * @param cfg — model hyperparameters
 */
export async function slatFlowForward(
  x: SparseTensor,
  t: number,
  cond: Tensor,
  concatCond: SparseTensor | null,
  weights: LoadedWeights,
  cfg: SLatFlowConfig,
  ctx: GPUContext,
): Promise<SparseTensor> {
  const getTensor = weights.getTensor;
  const { modelChannels, condChannels, numBlocks, numHeads, mlpRatio, inChannels, outChannels } = cfg;
  const C = modelChannels;
  const H = numHeads;
  const D = C / H;
  const mlpC = Math.floor(C * mlpRatio);
  const N = x.feats.shape[0]; // active voxels
  const M = cond.shape[0];    // cond tokens

  // Cu_seqlens for single batch: [0, N]
  const cuSeqlens = new Uint32Array([0, N]);
  const cuSeqlensCond = new Uint32Array([0, M]);

  console.log(`SLatFlow: N=${N}, C=${C}, H=${H}, D=${D}, blocks=${numBlocks}`);

  // ── 1. Input: concat cond if provided, then SparseLinear ──
  let feats = x.feats;
  if (concatCond) {
    // Concatenate along feature dim: [x.feats, concatCond.feats]
    const xData = feats.device === "cpu" ? feats.getView() as Float32Array : null;
    const ccData = concatCond.feats.device === "cpu" ? concatCond.feats.getView() as Float32Array : null;
    if (xData && ccData) {
      const combined = new Float32Array(N * (inChannels + concatCond.feats.shape[1]));
      for (let i = 0; i < N; i++) {
        combined.set(xData.subarray(i * inChannels, (i + 1) * inChannels), i * (inChannels + concatCond.feats.shape[1]));
        combined.set(ccData.subarray(i * concatCond.feats.shape[1], (i + 1) * concatCond.feats.shape[1]),
          i * (inChannels + concatCond.feats.shape[1]) + inChannels);
      }
      feats = Tensor.fromArray(combined, [N, inChannels + concatCond.feats.shape[1]], "float32");
      await feats.upload(ctx);
    }
  }
  await feats.upload(ctx);

  // Input projection
  let h = matmul(feats, getTensor("input_layer.weight"), ctx);
  const inBias = getTensor("input_layer.bias");
  await inBias.upload(ctx);
  h = add(h, inBias, ctx);

  // ── 2. Timestep embedding + adaLN ──
  // t_embedder: t → sinusoidal → SiLU → Linear → SiLU → (N, C)
  // adaLN_modulation: SiLU((N,C)) → Linear(6*C, C) → (N, 6*C) modulation base
  // Each block adds its own learned (6*C,) bias to modulation base
  const tFloat = new Float32Array([t / 1000]); // denormalize
  const tEmb = timestepEmbedCPU(
    tFloat, C,
    getTensor("t_embedder.mlp.0.weight").getView() as Float32Array,
    getTensor("t_embedder.mlp.0.bias").getView() as Float32Array,
    getTensor("t_embedder.mlp.2.weight").getView() as Float32Array,
    getTensor("t_embedder.mlp.2.bias").getView() as Float32Array,
  );
  // tEmb is (1, C) — broadcast to (N, C) via simple duplication
  // adaLN: SiLU(tEmb) @ W_adaLN + b_adaLN → (1, 6*C)
  const adaW = getTensor("adaLN_modulation.1.weight").getView() as Float32Array;
  const adaB = getTensor("adaLN_modulation.1.bias").getView() as Float32Array;
  let tEmbBase = new Float32Array(C);
  tEmbBase.set(tEmb.subarray(0, C));
  // SiLU
  for (let i = 0; i < C; i++) {
    const val = tEmbBase[i];
    tEmbBase[i] = val / (1 + Math.exp(-val)); // sigmoid
    tEmbBase[i] = val * (1 / (1 + Math.exp(-val))); // SiLU
  }

  // Linear: (C,) → (6*C,)
  const modBase = new Float32Array(6 * C);
  for (let j = 0; j < 6 * C; j++) {
    let sum = adaB[j];
    for (let i = 0; i < C; i++) sum += tEmbBase[i] * adaW[j * C + i]; // (6*C, C) layout
    const sig = 1.0 / (1.0 + Math.exp(-sum));
    modBase[j] = sum * sig; // SiLU
  }

  // Broadcast modBase to (N, 6*C) — each position gets same modulation
  const modBroadcast = new Float32Array(N * 6 * C);
  for (let i = 0; i < N; i++) modBroadcast.set(modBase, i * 6 * C);
  const modTensor = Tensor.fromArray(modBroadcast, [N, 6 * C], "float32");
  await modTensor.upload(ctx);

  // ── 3. Transformer blocks ──
  for (let b = 0; b < numBlocks; b++) {
    const p = `blocks.${b}.`;

    // Split fused weights per block
    const saQkvData = getTensor(p + "self_attn.to_qkv.weight").getView() as Float32Array;
    const { qWeight, kWeight, vWeight } = splitFusedQkv(saQkvData, C);
    await qWeight.upload(ctx); await kWeight.upload(ctx); await vWeight.upload(ctx);

    // adaLN: mod_base (N, 6*C) + blocks.N.modulation (6*C,) → chunk(6) → 6 × (N, C)
    const modulation = getTensor(p + "modulation").getView() as Float32Array; // (6*C,)
    const modCombined = new Float32Array(N * 6 * C);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < 6 * C; j++) {
        modCombined[i * 6 * C + j] = modBase[j] + modulation[j];
      }
    }

    // ── Self-attention ──
    let hn = layerNorm(h, null, null, 1e-6, ctx);
    // Modulation: scale + shift (simplified — broadcast via GPU)
    let q = matmul(hn, qWeight, ctx);
    let k = matmul(hn, kWeight, ctx);
    let kv = matmul(hn, vWeight, ctx);
    q = q.reshape([N, H, D]); k = k.reshape([N, H, D]); kv = kv.reshape([N, H, D]);

    // RMS norm on Q/K
    const saQNorm = getTensor(p + "self_attn.q_rms_norm.gamma");
    const saKNorm = getTensor(p + "self_attn.k_rms_norm.gamma");
    await saQNorm.upload(ctx); await saKNorm.upload(ctx);
    q = q.reshape([N * H, D]); q = rmsNorm(q, saQNorm, 1e-6, ctx); q = q.reshape([N, H, D]);
    k = k.reshape([N * H, D]); k = rmsNorm(k, saKNorm, 1e-6, ctx); k = k.reshape([N, H, D]);

    // RoPE
    const coordsCPU = x.coords.device === "cpu" ? new Int32Array(x.coords.getView() as unknown as ArrayBuffer) : new Int32Array(N * 3);
    if (coordsCPU.length >= N * 3) {
      const coordsTensor = Tensor.fromArray(coordsCPU.slice(0, N * 3), [N, 3], "int32");
      await coordsTensor.upload(ctx);
      q = applyRoPE({ x: q, coords: coordsTensor, numHeads: H, headDim: D, maxCoord: 31 }, ctx);
      k = applyRoPE({ x: k, coords: coordsTensor, numHeads: H, headDim: D, maxCoord: 31 }, ctx);
      coordsTensor.dispose();
    }

    // Sparse attention (B=1 for inference)
    const oSA = await sparseAttention(
      { Q: q, K: k, V: kv, cuSeqlensQ: cuSeqlens, cuSeqlensKV: cuSeqlens, numHeads: H, headDim: D },
      ctx,
    );
    let o = oSA.reshape([N, C]);
    o = matmul(o, getTensor(p + "self_attn.to_out.weight"), ctx);
    const saOutB = getTensor(p + "self_attn.to_out.bias"); await saOutB.upload(ctx);
    o = add(o, saOutB, ctx);
    h = add(h, o, ctx); // residual

    // ── Cross-attention ──
    hn = layerNorm(h, getTensor(p + "norm2.weight"), getTensor(p + "norm2.bias"), 1e-6, ctx);
    q = matmul(hn, getTensor(p + "cross_attn.to_q.weight"), ctx);
    const caQB = getTensor(p + "cross_attn.to_q.bias"); await caQB.upload(ctx);
    q = add(q, caQB, ctx); q = q.reshape([N, H, D]);

    // KV from cond
    const caKvData = getTensor(p + "cross_attn.to_kv.weight").getView() as Float32Array;
    const { kWeight: caKW, vWeight: caVW } = splitFusedKv(caKvData, C, condChannels);
    await caKW.upload(ctx); await caVW.upload(ctx);
    k = matmul(cond, caKW, ctx); kv = matmul(cond, caVW, ctx);
    k = k.reshape([M, H, D]); kv = kv.reshape([M, H, D]);

    o = scaledDotProductAttention({ Q: q, K: k, V: kv, numHeads: H, batchSize: 1 }, ctx);
    o = o.reshape([N, C]);
    o = matmul(o, getTensor(p + "cross_attn.to_out.weight"), ctx);
    const caOutB = getTensor(p + "cross_attn.to_out.bias"); await caOutB.upload(ctx);
    o = add(o, caOutB, ctx);
    h = add(h, o, ctx); // residual (no gate on cross-attn)

    // ── Feed-forward ──
    hn = layerNorm(h, null, null, 1e-6, ctx);
    hn = matmul(hn, getTensor(p + "mlp.mlp.0.weight"), ctx);
    const ffn0B = getTensor(p + "mlp.mlp.0.bias"); await ffn0B.upload(ctx);
    hn = add(hn, ffn0B, ctx);
    hn = gelu(hn, ctx);
    hn = matmul(hn, getTensor(p + "mlp.mlp.2.weight"), ctx);
    const ffn2B = getTensor(p + "mlp.mlp.2.bias"); await ffn2B.upload(ctx);
    hn = add(hn, ffn2B, ctx);
    h = add(h, hn, ctx); // residual

    qWeight.dispose(); kWeight.dispose(); vWeight.dispose();
    caKW.dispose(); caVW.dispose();
    if (b % 5 === 0) console.log(`  Block ${b + 1}/${numBlocks} done`);
  }

  // ── 4. Output ──
  h = layerNorm(h, null, null, 1e-6, ctx);
  h = matmul(h, getTensor("out_layer.weight"), ctx);
  const outBias = getTensor("out_layer.bias"); await outBias.upload(ctx);
  h = add(h, outBias, ctx);

  console.log("SLatFlow forward complete");
  return new SparseTensor(h, x.coords, x.spatialShape, x.scale);
}

// ── CPU timestep embedder ──
function timestepEmbedCPU(
  t: Float32Array, dim: number,
  w0: Float32Array, b0: Float32Array,
  w2: Float32Array, b2: Float32Array,
): Float32Array {
  const B = t.length;
  const modelC = b0.length;
  const half = dim / 2;

  const sinEmb = new Float32Array(B * dim);
  for (let b = 0; b < B; b++) {
    for (let i = 0; i < half; i++) {
      const freq = 1.0 / Math.pow(10000, (2 * i) / dim);
      const val = t[b] * freq;
      sinEmb[b * dim + 2 * i] = Math.sin(val);
      sinEmb[b * dim + 2 * i + 1] = Math.cos(val);
    }
  }

  // Layer 0: SiLU(Linear(sinEmb))
  const h0 = new Float32Array(B * modelC);
  for (let b = 0; b < B; b++) {
    for (let j = 0; j < modelC; j++) {
      let sum = b0[j];
      for (let i = 0; i < dim; i++) sum += sinEmb[b * dim + i] * w0[j * dim + i];
      const sig = 1.0 / (1.0 + Math.exp(-sum));
      h0[b * modelC + j] = sum * sig;
    }
  }

  // Layer 2: SiLU(Linear(h0))
  const h2 = new Float32Array(B * modelC);
  for (let b = 0; b < B; b++) {
    for (let j = 0; j < modelC; j++) {
      let sum = b2[j];
      for (let i = 0; i < modelC; i++) sum += h0[b * modelC + i] * w2[j * modelC + i];
      const sig = 1.0 / (1.0 + Math.exp(-sum));
      h2[b * modelC + j] = sum * sig;
    }
  }

  return h2;
}
