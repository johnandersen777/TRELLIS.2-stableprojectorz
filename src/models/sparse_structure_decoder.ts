/**
 * SparseStructureDecoder — dense 3D Conv3d decoder.
 *
 * Converts VAE latent to occupancy grid.
 * Architecture: Input Conv3d → Middle ResBlocks → ResBlocks + Upsample → Output.
 *
 * Config (ss_dec_conv3d_16l8_fp16):
 *   out_channels=1, latent_channels=8, channels=[512, 128, 32]
 *   num_res_blocks=2, num_res_blocks_middle=2
 */

import { Tensor } from "../runtime/tensor.ts";
import type { GPUContext } from "../runtime/device.ts";
import { add } from "../dense/ops/elementwise.ts";
import { silu } from "../dense/ops/silu.ts";
import { layerNorm } from "../dense/ops/layer_norm.ts";
import { conv3d, type Conv3dConfig } from "../dense/ops/conv3d.ts";
import { pixelShuffle3d } from "../dense/ops/pixel_shuffle_3d.ts";
import type { LoadedWeights } from "../model/loader.ts";

export interface SSDecoderConfig {
  outChannels: number;
  latentChannels: number;
  numResBlocks: number;
  channels: number[];
  numResBlocksMiddle: number;
  normType: "layer" | "group";
  useFp16: boolean;
}

export function ssDecoderConfig(args: Record<string, unknown>): SSDecoderConfig {
  return {
    outChannels: args.out_channels as number,
    latentChannels: args.latent_channels as number,
    numResBlocks: args.num_res_blocks as number,
    channels: args.channels as number[],
    numResBlocksMiddle: (args.num_res_blocks_middle as number) ?? 2,
    normType: (args.norm_type as string ?? "layer") as "layer" | "group",
    useFp16: (args.use_fp16 as boolean) ?? true,
  };
}

/** ResBlock3d: Norm → SiLU → Conv3d → Norm → SiLU → Conv3d + skip */
async function resBlock3d(
  x: Tensor,
  ch: number,
  conv1Weight: Tensor, conv1Bias: Tensor,
  conv2Weight: Tensor, conv2Bias: Tensor,
  spatialShape: [number, number, number],
  ctx: GPUContext,
): Promise<Tensor> {
  const convCfg: Conv3dConfig = {
    inChannels: ch, outChannels: ch,
    kernelSize: [3, 3, 3], stride: [1, 1, 1], padding: [1, 1, 1],
    inputShape: spatialShape,
  };
  // layerNorm and silu are in-place: h === x after these calls.
  let h = layerNorm(x, null, null, 1e-6, ctx);
  h = silu(h, ctx);
  const hConv1 = conv3d(h, conv1Weight, conv1Bias, convCfg, ctx);
  // h === x (in-place), keep alive for skip
  h = layerNorm(hConv1, null, null, 1e-6, ctx);
  h = silu(h, ctx);
  const hConv2 = conv3d(h, conv2Weight, conv2Bias, convCfg, ctx);
  // add is in-place on hConv2, x is skip connection
  // Don't dispose — minimize buffer.destroy() calls to avoid mapAsync corruption
  const result = add(hConv2, x, ctx);
  return result;
}

/** UpsampleBlock3d: Conv3d(in_C, out_C*8, K=1) → pixel_shuffle_3d(factor=2) */
async function upsampleBlock3d(
  x: Tensor,
  inC: number, outC: number,
  weight: Tensor, bias: Tensor,
  spatialShape: [number, number, number],
  ctx: GPUContext,
): Promise<Tensor> {
  // pixel_shuffle_3d on GPU: (B, C*8, D, H, W) → (B, C, 2D, 2H, 2W)
  const convCfg: Conv3dConfig = {
    inChannels: inC, outChannels: outC * 8,
    kernelSize: [1, 1, 1], stride: [1, 1, 1], padding: [0, 0, 0],
    inputShape: spatialShape,
  };
  const hConv = conv3d(x, weight, bias, convCfg, ctx);
  const h = pixelShuffle3d(hConv, outC, 2, ctx);
  return h;
}
/** Load weights for a ResBlock3d from safetensors */
interface ResBlockWeights {
  conv1Weight: Tensor; conv1Bias: Tensor;
  conv2Weight: Tensor; conv2Bias: Tensor;
}

function loadResBlockWeights(getTensor: (n: string) => Tensor, prefix: string): ResBlockWeights {
  return {
    conv1Weight: getTensor(prefix + "conv1.weight"),
    conv1Bias: getTensor(prefix + "conv1.bias"),
    conv2Weight: getTensor(prefix + "conv2.weight"),
    conv2Bias: getTensor(prefix + "conv2.bias"),
  };
}

/**
 * Full SS_Decoder forward pass.
 *
 * @param z — (B, latent_C, D, H, W) latent from VAE encoder
 * @returns (B, 1, 4D, 4H, 4W) occupancy logits (4× upsampled from input)
 */
export async function ssDecoderForward(
  z: Tensor,
  weights: LoadedWeights,
  cfg: SSDecoderConfig,
  ctx: GPUContext,
): Promise<Tensor> {
  const getTensor = weights.getTensor;
  const { latentChannels, channels, numResBlocks, numResBlocksMiddle } = cfg;
  const [B, lC, D, H, W] = [z.shape[0], z.shape[1], z.shape[2], z.shape[3], z.shape[4]];

  console.log(`SS_Decoder: input (${lC},${D},${H},${W}), channels=[${channels}], blocks=${numResBlocks}`);

  // 1. Input layer: Conv3d(latent_C → channels[0], 3, pad=1)
  const inW = getTensor("input_layer.weight");
  const inB = getTensor("input_layer.bias");
  await inW.upload(ctx); await inB.upload(ctx);
  let h = conv3d(z, inW, inB, {
    inChannels: latentChannels, outChannels: channels[0],
    kernelSize: [3, 3, 3], stride: [1, 1, 1], padding: [1, 1, 1],
    inputShape: [D, H, W],
  }, ctx);

  // 2. Middle block: ResBlock3d × num_res_blocks_middle
  let curSpatial: [number, number, number] = [D, H, W];
  let curCh = channels[0];
  for (let i = 0; i < numResBlocksMiddle; i++) {
    const rb = loadResBlockWeights(getTensor, `middle_block.${i}.`);
    await rb.conv1Weight.upload(ctx); await rb.conv1Bias.upload(ctx);
    await rb.conv2Weight.upload(ctx); await rb.conv2Bias.upload(ctx);
    h = await resBlock3d(h, curCh, rb.conv1Weight, rb.conv1Bias, rb.conv2Weight, rb.conv2Bias, curSpatial, ctx);
  }

  // 3. Blocks: ResBlock3d × num_res_blocks + Upsample between levels
  let blockIdx = 0;
  for (let level = 0; level < channels.length; level++) {
    const ch = channels[level];

    // ResBlocks at this level
    for (let i = 0; i < numResBlocks; i++) {
      const rb = loadResBlockWeights(getTensor, `blocks.${blockIdx}.`);
      await rb.conv1Weight.upload(ctx); await rb.conv1Bias.upload(ctx);
      await rb.conv2Weight.upload(ctx); await rb.conv2Bias.upload(ctx);
      h = await resBlock3d(h, ch, rb.conv1Weight, rb.conv1Bias, rb.conv2Weight, rb.conv2Bias, curSpatial, ctx);
      blockIdx++;
    }

    // Upsample to next level (if not last)
    if (level < channels.length - 1) {
      const nextCh = channels[level + 1];
      const upW = getTensor(`blocks.${blockIdx}.conv.weight`);
      const upB = getTensor(`blocks.${blockIdx}.conv.bias`);
      await upW.upload(ctx); await upB.upload(ctx);
      h = await upsampleBlock3d(h, ch, nextCh, upW, upB, curSpatial, ctx);
      curSpatial = [curSpatial[0] * 2, curSpatial[1] * 2, curSpatial[2] * 2];
      blockIdx++;
    }
  }

  // 4. Output: LayerNorm → SiLU → Conv3d(channels[-1] → out_channels)
  h = layerNorm(h, null, null, 1e-6, ctx);
  h = silu(h, ctx);
  const outW = getTensor("out_layer.2.weight");
  const outB = getTensor("out_layer.2.bias");
  await outW.upload(ctx); await outB.upload(ctx);
  const hOut = conv3d(h, outW, outB, {
    inChannels: channels[channels.length - 1], outChannels: cfg.outChannels,
    kernelSize: [3, 3, 3], stride: [1, 1, 1], padding: [1, 1, 1],
    inputShape: curSpatial,
  }, ctx);

  console.log(`SS_Decoder: output shape (${cfg.outChannels},${curSpatial[0]},${curSpatial[1]},${curSpatial[2]})`);
  return hOut;
}
