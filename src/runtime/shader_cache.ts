/**
 * Shared WebGPU shader and pipeline cache.
 *
 * Shader compilation is cheap (~1ms) but repeated createShaderModule calls
 * for the same WGSL source waste GPU driver time. Pipeline creation is
 * expensive (~5-50ms) and should always be cached.
 *
 * Caches are keyed per-device (WeakMap) so device loss drops stale entries.
 * Callers pass a human-readable key (e.g. "matmul_64x64_f16").
 */

// ── Shader cache ──────────────────────────────────────────────

const shaderCache = new WeakMap<GPUDevice, Map<string, GPUShaderModule>>();

export function getCachedShader(
  device: GPUDevice,
  key: string,
): GPUShaderModule | undefined {
  return shaderCache.get(device)?.get(key);
}

export function setCachedShader(
  device: GPUDevice,
  key: string,
  shader: GPUShaderModule,
): void {
  let map = shaderCache.get(device);
  if (!map) {
    map = new Map();
    shaderCache.set(device, map);
  }
  map.set(key, shader);
}

/** Get or create a shader module. */
export function getOrCreateShader(
  device: GPUDevice,
  key: string,
  code: string,
): GPUShaderModule {
  const cached = getCachedShader(device, key);
  if (cached) return cached;
  const shader = device.createShaderModule({ code });
  setCachedShader(device, key, shader);
  return shader;
}

// ── Pipeline cache ────────────────────────────────────────────

const pipelineCache = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();

export function getCachedPipeline(
  device: GPUDevice,
  key: string,
): GPUComputePipeline | undefined {
  return pipelineCache.get(device)?.get(key);
}

export function setCachedPipeline(
  device: GPUDevice,
  key: string,
  pipeline: GPUComputePipeline,
): void {
  let map = pipelineCache.get(device);
  if (!map) {
    map = new Map();
    pipelineCache.set(device, map);
  }
  map.set(key, pipeline);
}

/** Get or create a compute pipeline. */
export function getOrCreatePipeline(
  device: GPUDevice,
  key: string,
  shaderModule: GPUShaderModule,
  entryPoint = "main",
): GPUComputePipeline {
  const cached = getCachedPipeline(device, key);
  if (cached) return cached;
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shaderModule, entryPoint },
  });
  setCachedPipeline(device, key, pipeline);
  return pipeline;
}
