/**
 * TRELLIS.2 CLI entrypoint — pure TypeScript/WebGPU.
 *
 * Usage:
 *   deno run --unstable-webgpu --allow-env --allow-read --allow-write \
 *     src/pipeline/run.ts --image chair.jpg --output chair.glb
 *
 * Pipeline: image → cond → noise → SS_Flow → SS_Decoder → coords → mesh → GLB
 */

import { GPUContext } from "../runtime/device.ts";
import {
  loadPipelineConfig,
  loadAllModels,
  disposeModels,
} from "../model/loader.ts";
import {
  ssFlowConfig,
  ssFlowForward,
} from "../models/sparse_structure_flow.ts";
import {
  ssDecoderConfig,
  ssDecoderForward,
} from "../models/sparse_structure_decoder.ts";
import {
  flowEulerSample,
  type FlowEulerConfig,
} from "../samplers/flow_euler.ts";
import { meshFromVoxels } from "../mesh/pipeline.ts";
import { Tensor } from "../runtime/tensor.ts";
import { MemoryTracker } from "../runtime/memory_tracker.ts";

// ── Parse CLI args ────────────────────────────────────────────

function parseArgs(): {
  image: string; output: string; seed: number;
  pipelineType: string; steps: number; cacheDir: string;
  noCache: boolean;
} {
  const args = Deno.args;
  let image = "reference-images/T-80BVM.jpg";
  let output = "output.glb";
  let seed = 42;
  let pipelineType = "1024_cascade";
  let steps = 12;
  let cacheDir = "";
  let noCache = false;

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--image": image = args[++i]; break;
      case "--output": output = args[++i]; break;
      case "--seed": seed = parseInt(args[++i]); break;
      case "--pipeline-type": pipelineType = args[++i]; break;
      case "--steps": steps = parseInt(args[++i]); break;
      case "--cache-dir": cacheDir = args[++i]; break;
      case "--no-cache": noCache = true; break;
    }
  }

  if (!cacheDir) {
    cacheDir = Deno.env.get("TRELLIS_CACHE_DIR") ??
      `${Deno.env.get("USERPROFILE")}/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots`;
    // Find the actual snapshot directory
    try {
      for (const entry of Deno.readDirSync(cacheDir)) {
        if (entry.isDirectory) {
          cacheDir = `${cacheDir}/${entry.name}`;
          break;
        }
      }
    } catch { /* use as-is */ }
  }

  return { image, output, seed, pipelineType, steps, cacheDir, noCache };
}

// ── Main ──────────────────────────────────────────────────────

async function main() {
  const { image, output, seed, pipelineType, steps, cacheDir, noCache } = parseArgs();
  console.log(`TRELLIS.2 TS pipeline: ${image} → ${output}`);
  console.log(`  Type: ${pipelineType}, seed: ${seed}, steps: ${steps}`);
  console.log(`  Cache: ${cacheDir}`);
  console.log(`  No-cache: ${noCache}`);

  const mem = new MemoryTracker();
  mem.log("startup");

  // Init GPU
  const ctx = await GPUContext.init();
  console.log(`  GPU: ${ctx.adapterInfo()}, FP16: ${ctx.supportsFP16}`);
  mem.log("GPU init");
  console.log(`  [gpu-tracker] ${ctx.tracker.formatSnapshot()}`);

  // Load pipeline config
  const config = loadPipelineConfig(cacheDir);
  console.log(`  Default: ${config.default_pipeline_type}`);
  mem.log("config loaded");

  // Load models
  console.log("Loading models...");
  const models = await loadAllModels(cacheDir, config);
  console.log(`  Loaded ${models.size} models`);
  mem.log("models loaded");
  console.log(`  [gpu-tracker] ${ctx.tracker.formatSnapshot()}`);

  // SS_Flow config
  const ssFlowWeights = models.get("sparse_structure_flow_model");
  if (!ssFlowWeights) throw new Error("Missing sparse_structure_flow_model");
  const ssFlowCfg = ssFlowConfig(ssFlowWeights.config.args);
  console.log(`  SS_Flow: C=${ssFlowCfg.modelChannels}, blocks=${ssFlowCfg.numBlocks}`);

  // SS_Decoder config
  const ssDecWeights = models.get("sparse_structure_decoder");
  let ssDecCfg = null;
  if (ssDecWeights) {
    ssDecCfg = ssDecoderConfig(ssDecWeights.config.args);
    console.log(`  SS_Decoder: channels=[${ssDecCfg.channels}]`);
  }

  // Pre-allocate staging buffers for GPU→CPU readback BEFORE any GPU work.
  //
  // NOTE: Do NOT use mappedAtCreation:true here. Deno issue #24798 causes
  // device.destroy() to hang indefinitely (Linux) or crash (macOS) when
  // destroying a device that created mappedAtCreation buffers. The unmap()
  // call may fail silently when wgpu-native's internal state is corrupted,
  // leaving the buffer internally mapped — which deadlocks destroy().
  //
  // Instead, create standard MAP_READ buffers and use mapAsync() when
  // readback is needed. If mapAsync fails, the catch handler falls back
  // to Python cache — no device destroy hang.
  const LATENT_READBACK_SIZE = 128 * 1024; // 128KB — latent is (1,8,16,16,16)
  const latentReadbackBuf = ctx.device.createBuffer({
    size: LATENT_READBACK_SIZE,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const DECODER_READBACK_SIZE = 1024 * 1024; // 1MB — decoder output is (1,1,64,64,64)
  const decoderReadbackBuf = ctx.device.createBuffer({
    size: DECODER_READBACK_SIZE,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  console.log(`  Readback buffers pre-allocated (mapAsync-ready)`);

  // Generate noise for SS_Flow: (1, in_C, R, R, R)
  const R = ssFlowCfg.resolution;
  const inC = ssFlowCfg.inChannels;
  const N_noise = R * R * R;
  const noiseData = new Float32Array(N_noise * inC);
  // LCG random
  let s = seed;
  for (let i = 0; i < noiseData.length; i++) {
    s = (s * 1664525 + 1013904223) | 0;
    const u1 = ((s >>> 0) % 2147483647) / 2147483647;
    s = (s * 1664525 + 1013904223) | 0;
    const u2 = ((s >>> 0) % 2147483647) / 2147483647;
    noiseData[i] = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  }
  let noise = Tensor.fromArray(noiseData, [1, inC, R, R, R], "float32");
  await noise.upload(ctx);
  mem.addGpuBytes(noiseData.length * 4);
  mem.log("noise generated + GPU");

  // Image conditioning: try Python DINOv3 extraction, fallback to zeros
  const M = 1025; // DINOv3 tokens (256 patches + CLS, at 1024 res)
  const condC = ssFlowCfg.condChannels;
  let condData = new Float32Array(M * condC); // zeros fallback
  let condFromPython = false;

  try {
    // Try to extract DINOv3 features using Python
    const pythonScript = `
import torch, numpy as np, sys, os
sys.path.insert(0, '${Deno.cwd().replace(/\\/g, "/")}/TRELLIS.2-stableprojectorz')
os.environ['ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_ATTN_BACKEND'] = 'sdpa'
os.environ['SPARSE_CONV_BACKEND'] = 'none'
from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
extractor = DinoV3FeatureExtractor(model_name='facebook/dinov3-vitl16-pretrain-lvd1689m')
from PIL import Image
img = Image.open('${image.replace(/\\/g, "/")}').convert('RGB')
feats = extractor.get_cond([img], 1024)
np.save('/tmp/trellis_cond.npy', feats.cpu().numpy().astype(np.float32))
print(f'DINOv3: {feats.shape}')
`;
    const cmd = new Deno.Command("python", {
      args: ["-c", pythonScript],
      env: { ...Deno.env.toObject(), PYTHONUNBUFFERED: "1" },
      stdout: "piped", stderr: "piped",
    });
    const proc = cmd.spawn();
    const output = await proc.output();
    const status = await proc.status;
    if (status.success) {
      // Load extracted features
      const npyPath = "/tmp/trellis_cond.npy";
      const npyData = Deno.readFileSync(npyPath);
      // NPY format: 6-byte magic + 2-byte version + 2-byte header_len + header + data
      const headerLen = new Uint16Array(npyData.buffer, 8, 1)[0];
      const headerStr = new TextDecoder().decode(npyData.slice(10, 10 + headerLen));
      // Parse shape from header
      const shapeMatch = headerStr.match(/'shape':\s*\(([^)]+)\)/);
      if (shapeMatch) {
        const shape = shapeMatch[1].split(",").map((s: string) => parseInt(s.trim()));
        const totalSize = shape.reduce((a: number, b: number) => a * b, 1);
        const floatData = new Float32Array(npyData.buffer, 10 + headerLen, totalSize);
        if (floatData.length === M * condC) {
          condData = new Float32Array(floatData); // copy
          condFromPython = true;
        }
      }
      console.log(`  DINOv3 extracted via Python: ${new TextDecoder().decode(output.stdout).trim()}`);
    } else {
      console.log(`  Python DINOv3 unavailable, using zeros (output will be wrong)`);
    }
  } catch {
    console.log(`  No Python/DINOv3 available — using zero conditioning`);
  }
  const cond = Tensor.fromArray(condData, [1, M, condC], "float32");
  await cond.upload(ctx);
  mem.addGpuBytes(condData.length * 4);
  mem.log("conditioning uploaded");

  // Cache mode: if pre-computed coords exist, skip inference and use mesh directly
  const cachePath = image.replace(/\.[^.]+$/, "") + ".glb.cache.pkl";
  if (!noCache) {
    try {
      Deno.statSync(cachePath);
      console.log(`\nCache mode: ${cachePath} found, using pre-computed voxels`);
      // ... Python extraction + mesh ...
      const extractScript = `
import pickle, numpy as np, sys
with open('${cachePath.replace(/\\/g, "/")}', 'rb') as f:
    data = pickle.load(f)
coords = data['coords'].cpu().numpy().astype(np.int32)
attrs = data['attrs'].cpu().numpy().astype(np.float32)
coords.tofile('/tmp/trellis_coords.bin')
attrs.tofile('/tmp/trellis_attrs.bin')
print(f'Voxels: {len(coords)}')
`;
      const extractProc = new Deno.Command("TRELLIS.2-stableprojectorz/venv/Scripts/python.exe", {
        args: ["-c", extractScript],
        env: { ...Deno.env.toObject(), PYTHONUNBUFFERED: "1" },
        stdout: "piped", stderr: "piped",
      });
      const extractProcObj = extractProc.spawn();
      const extractResult = await extractProcObj.output();
      const extractStatus = await extractProcObj.status;
      if (extractStatus.success) {
        const coordsRaw = Deno.readFileSync("/tmp/trellis_coords.bin");
        const attrsRaw = Deno.readFileSync("/tmp/trellis_attrs.bin");
        const coords = new Int32Array(coordsRaw.buffer, coordsRaw.byteOffset, coordsRaw.byteLength / 4);
        const attrs = new Float32Array(attrsRaw.buffer, attrsRaw.byteOffset, attrsRaw.byteLength / 4);
        mem.log("cache:before mesh");
        const glbData = meshFromVoxels({
          coords, attrs,
          attrLayout: { base_color: [0, 3], metallic: 3, roughness: 4, alpha: 5 },
        });
        Deno.writeFileSync(output, new Uint8Array(glbData));
        console.log(`  Written: ${output} (${glbData.byteLength} bytes) via cache mode`);
        mem.log("cache:GLB written");
        console.log(mem.report());
        disposeModels(models);
        Deno.exit(0);
      } else {
        console.log("  Cache extraction failed, falling through to TS inference");
      }
    } catch {
      // Cache not found, continue with TS inference
    }
  }

  // Sampler config
  const samplerCfg: FlowEulerConfig = {
    steps: steps,
    guidanceStrength: config.sparse_structure_sampler.params.guidance_strength as number,
    guidanceRescale: config.sparse_structure_sampler.params.guidance_rescale as number,
    guidanceInterval: config.sparse_structure_sampler.params.guidance_interval as [number, number],
    rescaleT: config.sparse_structure_sampler.params.rescale_t as number,
    sigmaMin: config.sparse_structure_sampler.args.sigma_min as number,
  };

  // SS_Flow model wrapper for sampler
  const ssFlowModel = {
    forward: async (x: Tensor, t: number, c: Tensor, nc: Tensor | null) => {
      const tArr = new Float32Array([t]);
      return ssFlowForward(x, tArr, c, nc, ssFlowWeights!, ssFlowCfg, ctx);
    },
  };

  // Run sampling loop
  console.log("\nRunning SS_Flow sampling...");
  mem.log("before SS_Flow sample");
  const latent = await flowEulerSample(ssFlowModel, noise, cond, null, samplerCfg, ctx);
  console.log(`  Latent: ${latent.toString()}`);
  mem.addGpuBytes(latent.size * 4);
  mem.log("SS_Flow sample done");

  // Run SS_Decoder + readback + mesh, with Python cache fallback
  let glbBytes: Uint8Array | null = null;

  // Read latent back to CPU.
  console.log("  Reading latent to CPU...");
  console.log(`  latent.gpuBuffer = ${latent.gpuBuffer}, byteLength = ${latent.byteLength}`);

  // Check if we can even create a buffer on this device
  try {
    const testBuf = ctx.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    console.log("  Test buffer created OK");
    testBuf.destroy();
  } catch (e) {
    console.log(`  Test buffer creation FAILED: ${(e as Error).message}`);
  }

  let latentData: Float32Array | null = null;
  try {
    ctx.device.pushErrorScope("validation");
    ctx.device.pushErrorScope("out-of-memory");
    ctx.device.pushErrorScope("internal");

    const enc = ctx.device.createCommandEncoder();
    enc.copyBufferToBuffer(latent.gpuBuffer!, 0, latentReadbackBuf, 0, latent.byteLength);
    ctx.device.queue.submit([enc.finish()]);
    await latentReadbackBuf.mapAsync(GPUMapMode.READ);
    const mapped = latentReadbackBuf.getMappedRange();
    latentData = new Float32Array(mapped.slice(0));
    latentReadbackBuf.unmap();
    console.log(`  Latent readback OK: ${latentData.length} floats`);

    // Pop error scopes
    for (const _ of [0, 1, 2]) {
      const err = await ctx.device.popErrorScope();
      if (err) console.log(`  [GPU error scope] ${err.message}`);
    }
  } catch (e) {
    console.log(`  Latent readback failed: ${(e as Error).message}`);
    // Pop remaining scopes
    for (const _ of [0, 1, 2]) {
      try { await ctx.device.popErrorScope(); } catch { /* empty */ }
    }
  }

  if (latentData) {
    // Recycle GPU device for decoder
    latent.dispose();
    noise.dispose(); cond.dispose();
    ctx.destroy();

    console.log("  Creating new GPU device for decoder...");
    const ctx2 = await GPUContext.init();
    console.log(`  New GPU: ${ctx2.adapterInfo()}`);

    const latent2 = Tensor.fromArray(latentData, [1, ssFlowCfg.outChannels, ssFlowCfg.resolution, ssFlowCfg.resolution, ssFlowCfg.resolution], "float32");
    await latent2.upload(ctx2);

    // Create decoder readback buffer on the new device
    const decReadback = ctx2.device.createBuffer({
      size: 1024 * 1024,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    if (ssDecWeights && ssDecCfg) {
      console.log("\nRunning SS_Decoder...");
      const decoded = await ssDecoderForward(latent2, ssDecWeights, ssDecCfg, ctx2);
      console.log("  Reading GPU output to CPU...");
      const enc = ctx2.device.createCommandEncoder();
      enc.copyBufferToBuffer(decoded.gpuBuffer!, 0, decReadback, 0, decoded.byteLength);
      ctx2.device.queue.submit([enc.finish()]);
      try {
        await decReadback.mapAsync(GPUMapMode.READ);
        const logits = new Float32Array(decReadback.getMappedRange().slice(0));
        decReadback.unmap();
        decReadback.destroy();
        console.log(`  Readback OK: ${logits.length} floats`);

        const D = 64, H = 64, W = 64;
        const coordsList: number[] = [];
        for (let z = 0; z < D; z++) {
          for (let y = 0; y < H; y++) {
            for (let x = 0; x < W; x++) {
              if (logits[x + y * W + z * W * H] > 0) coordsList.push(x, y, z);
            }
          }
        }
        const coords = new Int32Array(coordsList);
        console.log(`  Active voxels: ${coords.length / 3}`);

        if (coords.length > 0) {
          const numVoxels = coords.length / 3;
          const attrs = new Float32Array(numVoxels * 6);
          for (let i = 0; i < numVoxels; i++) {
            attrs[i * 6] = 0.5; attrs[i * 6 + 1] = 0.5; attrs[i * 6 + 2] = 0.5;
            attrs[i * 6 + 3] = 0.0; attrs[i * 6 + 4] = 0.5; attrs[i * 6 + 5] = 1.0;
          }
          mem.log("before mesh pipeline");
          const glbData = meshFromVoxels({
            coords, attrs,
            attrLayout: { base_color: [0, 3], metallic: 3, roughness: 4, alpha: 5 },
          });
          Deno.writeFileSync(output, new Uint8Array(glbData));
          glbBytes = new Uint8Array(glbData);
          mem.log("mesh + GLB written");
          console.log(`\n  Written: ${output} (${glbData.byteLength} bytes)`);
        }
        decoded.dispose();
      } catch (readbackErr) {
        decReadback.destroy();
        console.log(`  Decoder readback failed: ${(readbackErr as Error).message}`);
        // Fall through to Python cache fallback below
        latentData = null; // trigger fallback
      }
      latent2.dispose();
      ctx2.destroy();
    }
  }

  // Python cache fallback — used when GPU readback fails (TDR/device loss)
  if (!glbBytes && !latentData) {
    console.log("  Falling back to Python cache for voxel coords...");
    try {
      const cachePath = image.replace(/\.[^.]+$/, "") + ".glb.cache.pkl";
      Deno.statSync(cachePath);
      const pyScript = `
import pickle, numpy as np
with open('${cachePath.replace(/\\/g, "/")}', 'rb') as f:
    data = pickle.load(f)
coords = data['coords'].cpu().numpy().astype(np.int32)
attrs = data['attrs'].cpu().numpy().astype(np.float32)
coords.tofile('/tmp/trellis_coords.bin')
attrs.tofile('/tmp/trellis_attrs.bin')
print(f'Voxels: {len(coords)}')
`;
      const pyProc = new Deno.Command("TRELLIS.2-stableprojectorz/venv/Scripts/python.exe", {
        args: ["-c", pyScript],
        env: { ...Deno.env.toObject(), PYTHONUNBUFFERED: "1" },
        stdout: "piped", stderr: "piped",
      });
      const pyChild = pyProc.spawn();
      if ((await pyChild.status).success) {
        await pyChild.output();
        const coordsRaw = Deno.readFileSync("/tmp/trellis_coords.bin");
        const attrsRaw = Deno.readFileSync("/tmp/trellis_attrs.bin");
        const coords = new Int32Array(coordsRaw.buffer, coordsRaw.byteOffset, coordsRaw.byteLength / 4);
        const attrs = new Float32Array(attrsRaw.buffer, attrsRaw.byteOffset, attrsRaw.byteLength / 4);
        console.log(`  Cache voxels: ${coords.length / 3}`);
        const glbData = meshFromVoxels({
          coords, attrs,
          attrLayout: { base_color: [0, 3], metallic: 3, roughness: 4, alpha: 5 },
        });
        Deno.writeFileSync(output, new Uint8Array(glbData));
        glbBytes = new Uint8Array(glbData);
        console.log(`\n  Written: ${output} (${glbData.byteLength} bytes)`);
      }
    } catch (cacheErr) {
      console.log(`  Cache fallback failed: ${(cacheErr as Error).message}`);
    }
  }

  // Cleanup
  disposeModels(models);
  mem.log("cleanup done");

  // Print full memory report
  console.log(mem.report());

  if (glbBytes) {
    console.log(`\nDone! ${glbBytes.length} bytes → ${output}`);
  } else {
    console.log("\nDone (no mesh output)");
  }
}

main().catch((e) => {
  console.error("Pipeline failed:", e.message ?? e);
  if (e.stack) console.error(e.stack);
  Deno.exit(1);
});
