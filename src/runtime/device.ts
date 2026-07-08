/**
 * GPUContext — WebGPU device wrapper + adapter + feature detection.
 * Single entry point for all GPU operations.
 */
import { MemoryTracker } from "./memory.ts";

export type DType = "float32" | "float16" | "int32" | "uint32" | "bool";
export type TensorDevice = "cpu" | "gpu";

export function dtypeToBytes(dtype: DType): number {
  switch (dtype) {
    case "float32": return 4;
    case "float16": return 2;
    case "int32": return 4;
    case "uint32": return 4;
    case "bool": return 1;
  }
}

export function dtypeToWGSL(dtype: DType): string {
  switch (dtype) {
    case "float32": return "f32";
    case "float16": return "f16";
    case "int32": return "i32";
    case "uint32": return "u32";
    case "bool": return "bool";
  }
}

export class GPUContext {
  readonly adapter: GPUAdapter;
  readonly device: GPUDevice;
  readonly tracker: MemoryTracker;
  readonly supportsFP16: boolean;
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
  readonly maxComputeWorkgroupsPerDimension: number;
  readonly maxComputeWorkgroupStorageSize: number;
  readonly maxComputeInvocationsPerWorkgroup: number;

  private constructor(adapter: GPUAdapter, device: GPUDevice) {
    this.adapter = adapter;
    this.device = device;
    this.tracker = new MemoryTracker(device, {
      gpu: Infinity, // Don't limit — GPU driver handles actual VRAM
      cpu: Math.floor(10 * 1024 ** 3),
    }, false); // quiet mode — suppress per-allocation log spam

    this.supportsFP16 = adapter.features.has("shader-f16");
    this.maxBufferSize = Number(device.limits.maxBufferSize);
    this.maxStorageBufferBindingSize = Number(
      device.limits.maxStorageBufferBindingSize,
    );
    this.maxComputeWorkgroupsPerDimension = Number(
      device.limits.maxComputeWorkgroupsPerDimension,
    );
    this.maxComputeWorkgroupStorageSize = Number(
      device.limits.maxComputeWorkgroupStorageSize,
    );
    this.maxComputeInvocationsPerWorkgroup = Number(
      device.limits.maxComputeInvocationsPerWorkgroup,
    );

    console.log(`[GPU] ${this.adapterInfo()}`);
    console.log(`[GPU] limits: maxComputeWorkgroupStorageSize=${this.maxComputeWorkgroupStorageSize}, maxBuffer=${this.maxBufferSize}, maxStorageBuffer=${this.maxStorageBufferBindingSize}`);
  }

  static async init(
    adapterOptions?: GPURequestAdapterOptions,
  ): Promise<GPUContext> {
    const gpu = navigator.gpu;
    if (!gpu) throw new Error("WebGPU not available");

    const adapter = await gpu.requestAdapter(adapterOptions ?? {
      powerPreference: "high-performance",
    });
    if (!adapter) throw new Error("No GPU adapter found");

    const hasShaderF16 = adapter.features.has("shader-f16");
    const requiredFeatures: GPUFeatureName[] = [];
    if (hasShaderF16) requiredFeatures.push("shader-f16");

    const device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxComputeWorkgroupStorageSize: Number(
          adapter.limits.maxComputeWorkgroupStorageSize,
        ),
      },
    });

    return new GPUContext(adapter, device);
  }

  adapterInfo(): string {
    const info = this.adapter.info;
    const vendor = info.vendor;
    const arch = info.architecture;
    const desc = info.description || info.device;
    const fallback = info.isFallbackAdapter ? " (FALLBACK!)" : "";
    const f16 = this.supportsFP16 ? " f16" : "";
    const mem = `maxBuffer=${(this.maxBufferSize / 1024 ** 3).toFixed(1)}GB`;
    return `${vendor} ${desc} arch=${arch}${fallback}${f16} ${mem}`;
  }

  /** WGSL precision prelude — prepend to all shaders */
  precisionPrelude(): string {
    if (this.supportsFP16) {
      return `enable f16;\nalias min16float = f16;\nalias min16float4 = vec4<f16>;\n`;
    }
    return `alias min16float = f32;\nalias min16float4 = vec4<f32>;\n`;
  }

  /** Best scalar type for given DType */
  scalarType(dtype: DType): string {
    if (dtype === "float16" && this.supportsFP16) return "f16";
    return dtypeToWGSL(dtype);
  }

  destroy(): void {
    this.device.destroy();
    this.tracker.releaseAll();
  }
}
