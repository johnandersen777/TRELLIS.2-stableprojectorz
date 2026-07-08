/**
 * TRELLIS.2 image-to-3D pipeline — Deno TypeScript + WebGPU.
 *
 * Two modes:
 *   1. Cached (default): reads pre-computed coords+attrs from .bin cache → TS mesh → .glb
 *   2. Full inference: --no-cache runs Python inference to populate cache, then TS mesh
 *
 * Cache format: {output}.coords.bin (int32 flat) + {output}.attrs.bin (float32 flat)
 * These are raw ndarray dumps — no pickle, directly readable from TypeScript.
 *
 * Run: deno task run -- <input.jpg> <output.glb> [--pipeline-type 512] [--no-cache]
 */
import { meshFromVoxels } from "./src/mesh/pipeline.ts";
import { GPUContext } from "./src/runtime/device.ts";

function parseArgs(args: string[]) {
  let input = "", output = "", pipelineType = "512", seed = 42;
  let noCache = false;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--pipeline-type" && i + 1 < args.length) pipelineType = args[++i];
    else if (args[i] === "--seed" && i + 1 < args.length) seed = parseInt(args[++i]);
    else if (args[i] === "--no-cache") noCache = true;
    else if (!args[i].startsWith("--")) {
      if (!input) input = args[i];
      else output = args[i];
    }
  }
  if (!input || !output) {
    console.error("Usage: deno task run -- <input.jpg> <output.glb> [--pipeline-type 512] [--no-cache]");
    Deno.exit(1);
  }
  return { input, output, pipelineType, seed, noCache };
}

async function main() {
  const args = parseArgs(Deno.args);
  console.log(`TRELLIS.2 Deno/WebGPU Pipeline`);
  console.log(`  Output: ${args.output}`);
  console.log(`  Pipeline: ${args.pipelineType}, Seed: ${args.seed}`);
  console.log(`  Mode: ${args.noCache ? "full inference" : "cached"}\n`);

  const coordsFile = args.output + ".coords.bin";
  const attrsFile = args.output + ".attrs.bin";

  // ── Step 1: Get coords/attrs (cache or inference) ──
  if (!args.noCache && fileExists(coordsFile) && fileExists(attrsFile)) {
    console.log("[pipeline] Using cached coords/attrs");
  } else {
    console.log("[pipeline] Running Python inference...");
    const pythonPath = "venv/Scripts/python.exe";
    const relInput = `../${args.input}`;
    const relOutput = `../${args.output}`;

    // Inference via run_amd.py (handles CUDA monkey-patches, amp, etc.)
    const proc1 = new Deno.Command(pythonPath, {
      args: [
        "-u", "run_amd.py", relInput, relOutput,
        "--pipeline-type", args.pipelineType,
        "--seed", String(args.seed),
        "--no-cache", // always no-cache for inference — we manage cache ourselves
      ],
      cwd: "TRELLIS.2-stableprojectorz",
      env: {
        ...Deno.env.toObject(),
        HF_HUB_DISABLE_SYMLINKS_WARNING: "1",
        SPARSE_CONV_BACKEND: "none",
        SPARSE_ATTN_BACKEND: "sdpa",
        ATTN_BACKEND: "sdpa",
        PYTORCH_ALLOC_CONF: "expandable_segments:True",
      },
      stdout: "inherit",
      stderr: "inherit",
    });
    const r1 = await proc1.output();
    if (r1.code !== 0) {
      console.error("[pipeline] Python inference failed.");
      Deno.exit(1);
    }

    // Extract coords+attrs from pickle cache, write as raw .bin
    console.log("[pipeline] Extracting coords/attrs to raw binary cache...");
    const extractScript = `
import pickle, numpy as np
with open(r"${(args.output + ".cache.pkl").replace(/\\/g, "\\\\")}", "rb") as f:
    data = pickle.load(f)
coords = data["coords"].numpy().astype("int32")
attrs = data["attrs"].numpy().astype("float32")
coords.tofile(r"${coordsFile.replace(/\\/g, "\\\\")}")
attrs.tofile(r"${attrsFile.replace(/\\/g, "\\\\")}")
print(f"{coords.shape[0]}", flush=True)
`;

    const proc2 = new Deno.Command(pythonPath, {
      args: ["-u", "-c", extractScript],
      cwd: "TRELLIS.2-stableprojectorz",
      env: {
        ...Deno.env.toObject(),
        HF_HUB_DISABLE_SYMLINKS_WARNING: "1",
        SPARSE_CONV_BACKEND: "none",
        SPARSE_ATTN_BACKEND: "sdpa",
        ATTN_BACKEND: "sdpa",
      },
      stdout: "piped",
      stderr: "inherit",
    });
    const r2 = await proc2.output();
    if (r2.code !== 0) {
      console.error("[pipeline] Cache extraction failed.");
      Deno.exit(1);
    }
    const numVoxels = parseInt(new TextDecoder().decode(r2.stdout).trim());
    console.log(`[pipeline] Cached ${numVoxels} voxels to ${coordsFile}`);
  }

  // ── Step 2: TS mesh pipeline ──
  console.log("[pipeline] Running TS mesh pipeline...");
  const coordsRaw = Deno.readFileSync(coordsFile);
  const attrsRaw = Deno.readFileSync(attrsFile);
  const coords = new Int32Array(coordsRaw.buffer, coordsRaw.byteOffset, coordsRaw.byteLength / 4);
  const attrs = new Float32Array(attrsRaw.buffer, attrsRaw.byteOffset, attrsRaw.byteLength / 4);

  const ctx = await GPUContext.init();

  const glb = meshFromVoxels(
    { coords, attrs, attrLayout: { base_color: [0, 3], metallic: 3, roughness: 4, alpha: 5 } },
    { verbose: true },
  );

  Deno.writeFileSync(args.output, new Uint8Array(glb));
  console.log(`[pipeline] Saved ${args.output} (${glb.byteLength} bytes)`);

  ctx.destroy();
  console.log("[pipeline] Done.");
}

function fileExists(path: string): boolean {
  try { Deno.statSync(path); return true; } catch { return false; }
}

main();
