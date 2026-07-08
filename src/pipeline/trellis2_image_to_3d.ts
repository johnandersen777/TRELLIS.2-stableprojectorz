/**
 * TRELLIS.2 Image-to-3D pipeline — pure TypeScript/WebGPU.
 *
 * End-to-end: image → cond → noise → SS_Flow → SS_Decoder → coords → mesh → GLB.
 * Replaces run_amd.py's Python inference with pure TS + WebGPU.
 */

import { Tensor } from "../runtime/tensor.ts";
import type { GPUContext } from "../runtime/device.ts";
import {
  loadPipelineConfig,
  loadAllModels,
  disposeModels,
  type PipelineConfig,
  type LoadedWeights,
} from "../model/loader.ts";
import {
  ssFlowConfig,
  ssFlowForward,
  type SSFlowConfig,
} from "../models/sparse_structure_flow.ts";
import {
  ssDecoderConfig,
  ssDecoderForward,
  type SSDecoderConfig,
} from "../models/sparse_structure_decoder.ts";
import { meshFromVoxels } from "../mesh/pipeline.ts";

// ── Types ─────────────────────────────────────────────────────

export interface PipelineOptions {
  cacheDir: string;           // HF cache snapshot directory
  seed: number;
  pipelineType: "512" | "1024" | "1024_cascade";
  useCache: boolean;          // Use pre-computed .bin cache (skip SS_Flow)
}

export interface PipelineResult {
  glbBytes: Uint8Array;
  coords: Int32Array;
  attrs: Float32Array;
}

// ── Main pipeline ─────────────────────────────────────────────

export async function runImageTo3D(
  imagePath: string,
  outputPath: string,
  options: PipelineOptions,
  ctx: GPUContext,
): Promise<PipelineResult> {
  console.log(`TRELLIS.2 TS pipeline: ${imagePath} → ${outputPath}`);
  console.log(`  Pipeline type: ${options.pipelineType}, seed: ${options.seed}`);

  // 1. Load pipeline config
  const config = loadPipelineConfig(options.cacheDir);
  console.log(`  Default pipeline: ${config.default_pipeline_type}`);

  // 2. Load models
  console.log("Loading models...");
  const models = await loadAllModels(options.cacheDir, config);
  console.log(`  Loaded ${models.size} models`);

  // 3. Load specific model configs
  const ssFlowWeights = models.get("sparse_structure_flow_model");
  if (!ssFlowWeights) throw new Error("Missing sparse_structure_flow_model");

  const ssFlowCfg = ssFlowConfig(ssFlowWeights.config.args);
  console.log(`  SS_Flow: C=${ssFlowCfg.modelChannels}, blocks=${ssFlowCfg.numBlocks}, heads=${ssFlowCfg.numHeads}`);

  // 4. Generate noise
  const R = ssFlowCfg.resolution; // 16 (input grid size)
  const N = R * R * R; // 4096
  const inC = ssFlowCfg.inChannels; // 8
  const noiseData = new Float32Array(N * inC);
  // Simple PRNG (reproducible with seed)
  let seed = options.seed;
  for (let i = 0; i < noiseData.length; i++) {
    seed = (seed * 1664525 + 1013904223) | 0;
    // Box-Muller transform for Gaussian
    const u1 = ((seed >>> 0) % 2147483647) / 2147483647;
    seed = (seed * 1664525 + 1013904223) | 0;
    const u2 = ((seed >>> 0) % 2147483647) / 2147483647;
    noiseData[i] = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  }
  const noise = Tensor.fromArray(noiseData, [1, inC, R, R, R], "float32");
  await noise.upload(ctx);

  // 5. Image conditioning (placeholder — use zeros for now)
  // DINOv3 extracts (B, M, cond_C) features from image
  const M = 257; // DINOv3 patch tokens (256 + CLS)
  const condC = ssFlowCfg.condChannels; // 1024
  const condData = new Float32Array(M * condC); // zeros
  const cond = Tensor.fromArray(condData, [1, M, condC], "float32");
  await cond.upload(ctx);

  // 6. Timestep (start of flow — t=1000, scaled to [0,1000])
  const t = new Float32Array([1000]);

  // 7. Run SS_Flow forward pass
  console.log("Running SS_Flow forward...");
  let output: Tensor;
  if (options.useCache) {
    // Cache mode: skip SS_Flow, use pre-computed coords
    console.log("  Cache mode: reading pre-computed coords...");
    // This path already exists in trellis_webgpu.ts
    // For now, fall through to inference
  }
  output = await ssFlowForward(noise, t, cond, null, ssFlowWeights, ssFlowCfg, ctx);
  console.log(`  SS_Flow output: ${output.toString()}`);

  // 8. Run SS_Decoder
  const ssDecWeights = models.get("sparse_structure_decoder");
  if (ssDecWeights) {
    const ssDecCfg = ssDecoderConfig(ssDecWeights.config.args);
    console.log("Running SS_Decoder...");
    const decoded = await ssDecoderForward(output, ssDecWeights, ssDecCfg, ctx);

    // Threshold occupancy: decoded > 0
    const decodedCPU = await decoded.toCPU();
    const logits = decodedCPU.getView() as Float32Array;

    // Find active voxels (occupancy > 0)
    const [B, C, D, H, W] = decoded.shape;
    const active: number[] = [];
    for (let z = 0; z < D; z++) {
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const idx = z * H * W + y * W + x;
          if (logits[idx] > 0) {
            active.push(x, y, z);
          }
        }
      }
    }
    const coords = new Int32Array(active);
    const Nv = active.length / 3;
    console.log(`  Active voxels: ${Nv} (${(100*Nv/(D*H*W)).toFixed(1)}%)`);

    // 9. Mesh pipeline
    if (Nv > 0) {
      const attrs = new Float32Array(Nv * 6); // base_color(RGB) + metallic + roughness + alpha
      attrs.fill(0.5); // grey default
      const glbData = meshFromVoxels({
        coords,
        attrs,
        attrLayout: {
          base_color: [0, 3],
          metallic: 3, roughness: 4, alpha: 5,
        },
      });
      Deno.writeFileSync(outputPath, new Uint8Array(glbData));
      console.log(`  Written: ${outputPath} (${glbData.byteLength} bytes)`);

      disposeModels(models);
      noise.dispose(); cond.dispose(); output.dispose(); decoded.dispose();
      decodedCPU.dispose();

      return { glbBytes: new Uint8Array(glbData), coords, attrs };
    } else {
      console.log("  WARNING: No active voxels — output may be empty");
    }
  }

  // Cleanup
  disposeModels(models);
  noise.dispose(); cond.dispose();

  return { glbBytes: new Uint8Array(0), coords: new Int32Array(0), attrs: new Float32Array(0) };
}
