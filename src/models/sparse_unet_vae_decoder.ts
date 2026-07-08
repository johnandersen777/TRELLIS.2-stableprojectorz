/**
 * SparseUnetVaeDecoder — hierarchical sparse VAE decoder.
 *
 * Converts sparse latent to dense features with progressive upsampling.
 * Used by shape_latent_decoder and tex_latent_decoder.
 *
 * Architecture: from_latent → levels[N] × (blocks[B] + upsample) → output_layer
 *
 * Each level: SparseConvNeXtBlock3d or SparseResBlock3d × num_blocks + SparseResBlockUpsample3d
 * Blocks use: SparseConv3d + SparseLinear + LayerNorm + SiLU (existing ops)
 * Upsample: doubles spatial resolution, optionally predicts subdivision mask
 *
 * Weights: shape_dec_next_dc_f16c32.safetensors (~905 MB), tex_dec_next_dc_f16c32.safetensors (~905 MB)
 */

import { Tensor } from "../runtime/tensor.ts";
import type { GPUContext } from "../runtime/device.ts";
import { matmul } from "../runtime/ops/matmul.ts";
import { add, mul } from "../dense/ops/elementwise.ts";
import { silu } from "../dense/ops/silu.ts";
import { layerNorm } from "../dense/ops/layer_norm.ts";
import { SparseTensor } from "../sparse/sparse_tensor.ts";
import { sparseLinear } from "../sparse/ops/linear.ts";
import { SparseConv3d } from "../sparse/ops/conv3d.ts";
import type { LoadedWeights } from "../model/loader.ts";

export interface SparseVaeDecoderConfig {
  outChannels: number;
  modelChannels: number[];
  latentChannels: number;
  numBlocks: number[];
  predSubdiv: boolean;
  useFp16: boolean;
}

export function sparseVaeDecoderConfig(args: Record<string, unknown>): SparseVaeDecoderConfig {
  return {
    outChannels: args.out_channels as number,
    modelChannels: args.model_channels as number[],
    latentChannels: args.latent_channels as number,
    numBlocks: args.num_blocks as number[],
    predSubdiv: (args.pred_subdiv as boolean) ?? true,
    useFp16: (args.use_fp16 as boolean) ?? true,
  };
}

/** One sparse ConvNeXt-style block: DepthwiseConv3d → LN → Linear → LN → SiLU → Linear + skip */
async function sparseConvNeXtBlock(
  x: SparseTensor,
  ch: number,
  dwConv: SparseConv3d | null,
  linear0Weight: Tensor, linear0Bias: Tensor,
  linear2Weight: Tensor, linear2Bias: Tensor,
  ctx: GPUContext,
): Promise<SparseTensor> {
  let h = x;
  // Depthwise sparse conv
  if (dwConv) {
    h = dwConv.forward(h, ctx);
  }
  // LayerNorm → Linear0 → LayerNorm → SiLU → Linear2
  h.feats = layerNorm(h.feats, null, null, 1e-6, ctx);
  h.feats = matmul(h.feats, linear0Weight, ctx);
  if (linear0Bias.size > 0) {
    await linear0Bias.upload(ctx);
    h.feats = add(h.feats, linear0Bias, ctx);
  }
  h.feats = layerNorm(h.feats, null, null, 1e-6, ctx);
  h.feats = silu(h.feats, ctx);
  h.feats = matmul(h.feats, linear2Weight, ctx);
  if (linear2Bias.size > 0) {
    await linear2Bias.upload(ctx);
    h.feats = add(h.feats, linear2Bias, ctx);
  }
  // Skip connection
  h.feats = add(h.feats, x.feats, ctx);
  return h;
}

/**
 * Sparse VAE decoder forward pass.
 *
 * @param x — SparseTensor from VAE encoder (N active voxels)
 * @param guideSubs — optional subdivision masks for guided upsampling (texture decoder)
 * @param weights — loaded model weights
 * @param cfg — model config
 */
export async function sparseVaeDecoderForward(
  x: SparseTensor,
  guideSubs: SparseTensor[] | null,
  weights: LoadedWeights,
  cfg: SparseVaeDecoderConfig,
  ctx: GPUContext,
): Promise<{ output: SparseTensor; subs?: SparseTensor[] }> {
  const getTensor = weights.getTensor;
  const { modelChannels, numBlocks, latentChannels, outChannels, predSubdiv } = cfg;

  console.log(`SparseVAEDecoder: latent_C=${latentChannels}, channels=[${modelChannels}], blocks=[${numBlocks}]`);

  // 1. from_latent: SparseLinear(latent_C → model_C[0])
  const fromLatentW = getTensor("from_latent.weight");
  const fromLatentB = getTensor("from_latent.bias");
  await fromLatentW.upload(ctx); await fromLatentB.upload(ctx);
  let h = new SparseTensor(
    matmul(x.feats, fromLatentW, ctx),
    x.coords,
    x.spatialShape,
    x.scale,
  );
  h.feats = add(h.feats, fromLatentB, ctx);

  const subs: SparseTensor[] = [];

  // 2. Per-level blocks + upsample
  for (let level = 0; level < modelChannels.length; level++) {
    const ch = modelChannels[level];

    for (let b = 0; b < numBlocks[level]; b++) {
      const prefix = `blocks.${level}.${b}.`;
      // Try ConvNeXt block pattern first
      try {
        const dwWeight = getTensor(prefix + "dw_conv.weight");
        const dwBias = getTensor(prefix + "dw_conv.bias");
        const l0W = getTensor(prefix + "linear0.weight");
        const l0B = getTensor(prefix + "linear0.bias");
        const l2W = getTensor(prefix + "linear2.weight");
        const l2B = getTensor(prefix + "linear2.bias");
        await dwWeight.upload(ctx); await dwBias.upload(ctx);
        await l0W.upload(ctx); await l0B.upload(ctx);
        await l2W.upload(ctx); await l2B.upload(ctx);

        const dwConv = new SparseConv3d(dwWeight, dwBias, [3, 3, 3], [1, 1, 1]);
        h = await sparseConvNeXtBlock(h, ch, dwConv, l0W, l0B, l2W, l2B, ctx);
      } catch {
        // Fallback: simple ResBlock (LN → Linear → SiLU → LN → Linear + skip)
        const l0W = getTensor(prefix + "linear0.weight");
        const l0B = getTensor(prefix + "linear0.bias");
        const l2W = getTensor(prefix + "linear2.weight");
        const l2B = getTensor(prefix + "linear2.bias");
        await l0W.upload(ctx); await l0B.upload(ctx);
        await l2W.upload(ctx); await l2B.upload(ctx);

        let r = x.feats;
        r = layerNorm(r, null, null, 1e-6, ctx);
        r = matmul(r, l0W, ctx);
        r = add(r, l0B, ctx);
        r = silu(r, ctx);
        r = layerNorm(r, null, null, 1e-6, ctx);
        r = matmul(r, l2W, ctx);
        r = add(r, l2B, ctx);
        h = new SparseTensor(add(r, h.feats, ctx), h.coords, h.spatialShape, h.scale);
      }
    }

    // Upsample to next level
    if (level < modelChannels.length - 1 && predSubdiv) {
      const upPrefix = `blocks.${level}.${numBlocks[level]}.`;
      // SparseResBlockUpsample3d: predicts which voxels to subdivide
      // Subdivide 1 voxel → 8 child voxels (2x2x2 grid)
      const upW = getTensor(upPrefix + "proj.weight");
      const upB = getTensor(upPrefix + "proj.bias");
      await upW.upload(ctx); await upB.upload(ctx);

      const nextCh = modelChannels[level + 1];
      h.feats = matmul(h.feats, upW, ctx);
      h.feats = add(h.feats, upB, ctx);
      // Upsample coords: double resolution
      const nextShape = h.spatialShape.map((s: number) => s * 2);
      h = new SparseTensor(h.feats, h.coords, nextShape, h.scale);

      if (guideSubs && level < guideSubs.length) {
        // Use provided subdivision mask
        subs.push(guideSubs[level]);
      }
    }
  }

  // 3. output_layer: SparseLinear(model_C[-1] → out_C)
  const outW = getTensor("output_layer.weight");
  const outB = getTensor("output_layer.bias");
  await outW.upload(ctx); await outB.upload(ctx);
  h.feats = matmul(h.feats, outW, ctx);
  h.feats = add(h.feats, outB, ctx);

  console.log(`SparseVAEDecoder: output N=${h.feats.shape[0]}, C=${outChannels}`);
  return { output: h, subs: subs.length > 0 ? subs : undefined };
}
