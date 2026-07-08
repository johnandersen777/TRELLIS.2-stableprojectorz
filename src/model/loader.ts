/**
 * Model weight loader and config parser.
 *
 * Reads HuggingFace pipeline.json + per-model .json configs,
 * loads .safetensors into Tensor objects, maps weight names to model params.
 *
 * Weights stay on CPU until explicitly uploaded to GPU via toGPU().
 * This enables sequential model load/unload (only 2-3 models active at a time).
 */

import { Tensor } from "../runtime/tensor.ts";
import { SafetensorsFile, MultiFileLoader, type TensorEntry } from "../runtime/weights.ts";
import type { DType } from "../runtime/device.ts";

// ── Types ─────────────────────────────────────────────────────

export interface ModelConfig {
  name: string;
  args: Record<string, unknown>;
  safetensorsPath: string;
}

export interface SamplerConfig {
  name: string;
  args: Record<string, unknown>;
  params: Record<string, unknown>;
}

export interface PipelineConfig {
  models: Record<string, string>;
  sparse_structure_sampler: SamplerConfig;
  shape_slat_sampler: SamplerConfig;
  shape_slat_normalization: { mean: number[]; std: number[] };
  tex_slat_sampler: SamplerConfig;
  tex_slat_normalization: { mean: number[]; std: number[] };
  image_cond_model: { name: string; args: Record<string, unknown> };
  rembg_model: { name: string; args: Record<string, unknown> };
  default_pipeline_type: string;
}

export interface LoadedWeights {
  config: ModelConfig;
  tensors: Map<string, TensorEntry>;     // metadata
  safetensors: SafetensorsFile;           // for on-demand reads
  /** Read a tensor by name, converting to F32 on CPU */
  getTensor: (name: string) => Tensor;
  /** Read all weight names */
  weightNames: string[];
}

// ── Pipeline config loading ───────────────────────────────────

/** Load pipeline.json from a HF cache directory */
export function loadPipelineConfig(cacheDir: string): PipelineConfig {
  const raw = Deno.readFileSync(`${cacheDir}/pipeline.json`);
  const decoder = new TextDecoder();
  const parsed = JSON.parse(decoder.decode(raw));
  // Handle nested structure: { name, args: { models, samplers, ... } }
  return (parsed.args ?? parsed) as PipelineConfig;
}

/** Load a per-model .json config */
export function loadModelConfig(
  cacheDir: string,
  modelPath: string,
): ModelConfig {
  const base = modelPath.endsWith(".safetensors") ? modelPath.slice(0, -".safetensors".length) : modelPath;
  let jsonPath: string;
  if (base.startsWith("ckpts/")) {
    jsonPath = `${cacheDir}/${base}.json`;
  } else {
    // External HF reference: "org/repo/ckpts/name"
    const parts = base.split("/");
    jsonPath = resolveHfCachePath(parts.slice(0, 2).join("/"), parts.slice(2).join("/"));
    jsonPath = jsonPath.replace(".safetensors", ".json");
  }
  const raw = Deno.readFileSync(jsonPath);
  const decoder = new TextDecoder();
  const parsed = JSON.parse(decoder.decode(raw)) as { name: string; args: Record<string, unknown> };
  return {
    name: parsed.name,
    args: parsed.args,
    safetensorsPath: modelPath.endsWith(".safetensors")
      ? modelPath
      : modelPath + ".safetensors",
  };
}

/** Resolve a model path to its .safetensors path.
 * Handles both local ckpts/ and external HF references. */
function resolveSafetensorsPath(cacheDir: string, modelPath: string): string {
  const base = modelPath.endsWith(".safetensors") ? modelPath.slice(0, -".safetensors".length) : modelPath;
  if (base.startsWith("ckpts/")) {
    return `${cacheDir}/${modelPath}`;
  }
  // External HF reference: "org/repo/ckpts/name" → repo="org/repo", ckpt="ckpts/name"
  const parts = base.split("/");
  const hfRepo = parts.slice(0, 2).join("/"); // org/repo
  const ckptName = parts.slice(2).join("/");  // ckpts/name
  return resolveHfCachePath(hfRepo, ckptName);
}

function resolveHfCachePath(hfRepo: string, ckptName: string): string {
  // Convert "microsoft/TRELLIS-image-large" to cache path
  const cacheKey = `models--${hfRepo.replace("/", "--")}`;
  const hubDir = `${Deno.env.get("HF_HOME") ?? Deno.env.get("HOME") ?? Deno.env.get("USERPROFILE") ?? "."}/.cache/huggingface/hub/${cacheKey}`;
  // Read refs to find the snapshot
  try {
    const refsRaw = Deno.readFileSync(`${hubDir}/refs/main`);
    const refsDecoder = new TextDecoder();
    const snapshot = refsDecoder.decode(refsRaw).trim();
    return `${hubDir}/snapshots/${snapshot}/${ckptName}.safetensors`;
  } catch {
    // Fallback: try to read snapshots directory
    try {
      const snapshotsDir = `${hubDir}/snapshots`;
      for (const entry of Deno.readDirSync(snapshotsDir)) {
        if (entry.isDirectory) {
          return `${snapshotsDir}/${entry.name}/${ckptName}.safetensors`;
        }
      }
    } catch {
      // ignore
    }
    throw new Error(`Cannot resolve HF cache path for ${hfRepo}/${ckptName}`);
  }
}

// ── Weight loading ────────────────────────────────────────────

/** Load a single model's weights from safetensors.
 *  Returns metadata and on-demand tensor reader.
 *  Does NOT load all tensors into memory — only parses header. */
export async function loadModelWeights(
  cacheDir: string,
  modelPath: string,
): Promise<LoadedWeights> {
  const config = loadModelConfig(cacheDir, modelPath);
  const safetensorsPath = resolveSafetensorsPath(cacheDir, config.safetensorsPath);
  const sf = new SafetensorsFile(safetensorsPath);
  sf.open();
  const weightNames = [...sf.listTensors()];

  const tensors = new Map<string, TensorEntry>();
  for (const name of weightNames) {
    const info = sf.getTensorInfo(name);
    if (info) tensors.set(name, info);
  }

  return {
    config,
    tensors,
    safetensors: sf,
    weightNames,
    getTensor: (name: string): Tensor => {
      const data = sf.readTensorF32(name);
      const entry = tensors.get(name);
      if (!entry) throw new Error(`Unknown tensor: ${name}`);
      return Tensor.fromArray(data, entry.shape, mapDtype(entry.dtype));
    },
  };
}

/** Load all models from pipeline config. Returns a map of model key → loaded weights. */
export async function loadAllModels(
  cacheDir: string,
  pipelineConfig: PipelineConfig,
): Promise<Map<string, LoadedWeights>> {
  const models = new Map<string, LoadedWeights>();
  for (const [key, modelPath] of Object.entries(pipelineConfig.models)) {
    console.log(`  Loading ${key}...`);
    const weights = await loadModelWeights(cacheDir, modelPath);
    models.set(key, weights);
  }
  return models;
}

// ── Helpers ───────────────────────────────────────────────────

function mapDtype(dtype: string): DType {
  // readTensorF32 converts BF16/F16 → F32, so all float tensors are f32
  switch (dtype) {
    case "F32": case "F16": case "BF16": return "float32";
    case "I32": case "U32": return "int32";
    default: return "float32";
  }
}

/** Dispose all loaded models, freeing file handles and memory */
export function disposeModels(models: Map<string, LoadedWeights>): void {
  for (const [, loaded] of models) {
    loaded.safetensors.close();
  }
  models.clear();
}
