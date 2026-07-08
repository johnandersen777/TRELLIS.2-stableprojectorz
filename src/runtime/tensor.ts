/**
 * Tensor — multi-dimensional array backed by CPU ArrayBuffer or GPU GPUBuffer.
 *
 * Ref-counted lifecycle: retain/release/dispose.
 * Explicit device placement: toGPU() / toCPU().
 * Immutable shape and dtype after creation.
 */
import { type DType, dtypeToBytes, type GPUContext } from "./device.ts";
import type { AllocationHandle } from "./memory.ts";

let nextTensorId = 0;

export class Tensor {
  readonly id: number;
  readonly shape: readonly number[];
  readonly dtype: DType;
  readonly byteLength: number;
  readonly size: number; // numel

  private _device: "cpu" | "gpu";
  private _handle: AllocationHandle | null;
  private _gpuBuffer: GPUBuffer | null; // direct ref, bypasses tracker
  private _refCount: number;
  private _context: GPUContext | null; // non-null when on GPU
  private _cpuData: ArrayBuffer | null; // non-null after toCPU()

  private constructor(
    shape: readonly number[],
    dtype: DType,
    device: "cpu" | "gpu",
    handle: AllocationHandle | null,
    context: GPUContext | null,
    gpuBuffer?: GPUBuffer,
  ) {
    this.id = nextTensorId++;
    this.shape = [...shape];
    this.dtype = dtype;
    this.size = shape.reduce((a, b) => a * b, 1);
    this.byteLength = this.size * dtypeToBytes(dtype);
    this._device = device;
    this._handle = handle;
    this._gpuBuffer = gpuBuffer ?? null;
    this._refCount = 1;
    this._context = context;
    this._cpuData = handle?.arrayBuffer ?? null;
  }

  // ── Factory ──────────────────────────────────────────────

  /** Create a CPU tensor from a typed array */
  static fromArray(
    data: Float32Array | Float64Array | Int32Array | Uint32Array | Int16Array | Uint16Array | Int8Array | Uint8Array,
    shape: number[],
    dtype?: DType,
  ): Tensor {
    const inferredDtype: DType = dtype ?? (data instanceof Float32Array
      ? "float32"
      : data instanceof Uint32Array
      ? "uint32"
      : data instanceof Int32Array
      ? "int32"
      : data instanceof Float64Array
      ? "float32"
      : "float32");

    // Store a copy
    const byteLength = data.byteLength;
    const ab = new ArrayBuffer(byteLength);
    const view = new Uint8Array(ab);
    view.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));

    const t = new Tensor(shape, inferredDtype, "cpu", null, null);
    t._cpuData = ab;
    return t;
  }

  /** Create a CPU tensor from an ArrayBuffer */
  static fromBuffer(
    buffer: ArrayBuffer,
    shape: number[],
    dtype: DType,
  ): Tensor {
    const t = new Tensor(shape, dtype, "cpu", null, null);
    t._cpuData = buffer;
    return t;
  }

  /** Create a GPU tensor backed by a GPUBuffer handle */
  static fromGpuHandle(
    handle: AllocationHandle,
    shape: number[],
    dtype: DType,
    context: GPUContext,
  ): Tensor {
    return new Tensor(shape, dtype, "gpu", handle, context);
  }

  /** Create a GPU tensor from raw GPUBuffer.
   *
   *  Does NOT register with MemoryTracker — holds GPUBuffer reference directly.
   *  This avoids accumulating thousands of tracker handles that corrupt
   *  wgpu-native's internal state (Deno #24798, #22146).
   *
   *  dispose() calls buffer.destroy() directly.
   */
  static fromGpuBuffer(
    buffer: GPUBuffer,
    shape: number[],
    dtype: DType,
    context: GPUContext,
    _label?: string,
  ): Tensor {
    return new Tensor(shape, dtype, "gpu", null, context, buffer);
  }

  /** Create a zero-filled tensor */
  static zeros(
    shape: number[],
    dtype: DType = "float32",
  ): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const bytes = size * dtypeToBytes(dtype);
    const ab = new ArrayBuffer(bytes);
    return new Tensor(shape, dtype, "cpu", null, null);
  }

  // ── Device transfer ──────────────────────────────────────

  /** Move this tensor to GPU, allocating a storage buffer. Returns new Tensor. */
  async toGPU(context: GPUContext): Promise<Tensor> {
    if (this._device === "gpu" && this._context === context) return this;

    const device = context.device;
    const contents = this._device === "cpu" ? this.getCPUData() : undefined;

    const buffer = device.createBuffer({
      size: this.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });

    if (contents) {
      device.queue.writeBuffer(buffer, 0, contents);
    }

    // Store GPUBuffer directly — bypasses MemoryTracker to avoid
    // accumulating handles that corrupt wgpu-native (Deno #24798).
    return new Tensor(this.shape, this.dtype, "gpu", null, context, buffer);
  }

  /** Move this tensor to CPU via staging buffer */
  async toCPU(): Promise<Tensor> {
    if (this._device === "cpu") return this;

    const context = this._context!;
    const device = context.device;

    const srcBuf = this._handle?.gpuBuffer;
    if (!srcBuf) throw new Error("toCPU: no GPU buffer");

    // Use a pre-allocated staging buffer if available (avoids creating new
    // buffers after heavy compute, which triggers wgpu-native mapAsync bug)
    const staging = device.createBuffer({
      size: this.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(srcBuf, 0, staging, 0, this.byteLength);
    device.queue.submit([encoder.finish()]);

    await device.queue.onSubmittedWorkDone();

    let mapped: ArrayBuffer;
    try {
      await staging.mapAsync(GPUMapMode.READ);
      mapped = staging.getMappedRange();
    } catch (e) {
      staging.destroy();
      throw new Error(`toCPU: mapAsync failed for ${this.byteLength} bytes: ${(e as Error).message}`);
    }

    const ab = new ArrayBuffer(this.byteLength);
    new Uint8Array(ab).set(new Uint8Array(mapped));
    staging.unmap();
    staging.destroy();

    const t = new Tensor(this.shape, this.dtype, "cpu", null, null);
    t._cpuData = ab;
    return t;
  }

  /** Upload to GPU in-place, mutating this tensor. Does NOT return a new tensor. */
  async upload(ctx: GPUContext): Promise<void> {
    if (this._device === "gpu" && this._context === ctx) return;
    const gpuCopy = await this.toGPU(ctx);
    this._device = gpuCopy._device;
    this._context = gpuCopy._context;
    this._handle = gpuCopy._handle;
    this._gpuBuffer = gpuCopy._gpuBuffer;
    // Transfer CPU data ownership
    this._cpuData = gpuCopy._cpuData ?? this._cpuData;
  }

  /** Upload multiple tensors in parallel, mutating each in-place. */
  static async uploadAll(tensors: Tensor[], ctx: GPUContext): Promise<void> {
    await Promise.all(tensors.map(t => t.upload(ctx)));
  }

  // ── Accessors ────────────────────────────────────────────

  get device(): "cpu" | "gpu" {
    return this._device;
  }

  get context(): GPUContext | null {
    return this._context;
  }

  get handle(): AllocationHandle | null {
    return this._handle;
  }

  get gpuBuffer(): GPUBuffer | undefined {
    return this._gpuBuffer ?? this._handle?.gpuBuffer;
  }

  /** Get CPU data. For GPU tensors, this will fail — use toCPU() first. */
  getCPUData(): ArrayBuffer {
    if (this._cpuData) return this._cpuData;
    if (this._handle?.arrayBuffer) return this._handle.arrayBuffer;
    // Create an empty buffer matching byteLength (for zero-filled tensors)
    const ab = new ArrayBuffer(this.byteLength);
    return ab;
  }

  /** Get CPU data as a typed view */
  getView(): Float32Array | Int32Array | Uint32Array {
    const ab = this.getCPUData();
    switch (this.dtype) {
      case "float32": return new Float32Array(ab);
      case "int32": return new Int32Array(ab);
      case "uint32": return new Uint32Array(ab);
      default: return new Float32Array(ab);
    }
  }

  // ── Lifecycle ────────────────────────────────────────────

  retain(): void {
    this._refCount++;
  }

  release(): void {
    this._refCount--;
    if (this._refCount <= 0) {
      if (this._gpuBuffer) {
        this._gpuBuffer.destroy();
        this._gpuBuffer = null;
      }
      if (this._handle) {
        this._context?.tracker.release(this._handle);
        this._handle = null;
      }
    }
  }

  dispose(): void {
    if (this._gpuBuffer) {
      this._gpuBuffer.destroy();
      this._gpuBuffer = null;
    }
    if (this._handle) {
      this._context?.tracker.release(this._handle);
      this._handle = null;
    }
    this._refCount = 0;
  }

  [Symbol.dispose](): void {
    this.dispose();
  }

  // ── Utility ──────────────────────────────────────────────

  /** Zero-copy reshape: new shape must have same total element count. */
  reshape(newShape: number[]): Tensor {
    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (newSize !== this.size) {
      throw new Error(
        `reshape: size mismatch ${this.size} → ${newSize} (${this.shape} → ${newShape})`,
      );
    }
    // Copy constructor with new shape, same backing data
    const t = new Tensor(newShape, this.dtype, this._device, this._handle, this._context, this._gpuBuffer ?? undefined);
    t._cpuData = this._cpuData;
    t._refCount = this._refCount;
    this.retain();
    return t;
  }

  /** Create a byte-offset view into the same buffer.
   *  For GPU tensors: offset is byte offset into the GPUBuffer.
   *  For CPU tensors: offset is byte offset into the ArrayBuffer. */
  view(shape: number[], byteOffset: number): Tensor {
    const viewSize = shape.reduce((a, b) => a * b, 1);
    // For GPU tensor views, we lose the original buffer — can't slice GPUBuffer.
    // This is a CPU-only operation for now.
    if (this._device === "gpu") {
      throw new Error("view() not supported on GPU tensors — use CPU first");
    }
    const ab = this.getCPUData();
    const sliced = ab.slice(byteOffset, byteOffset + viewSize * dtypeToBytes(this.dtype));
    const t = new Tensor(shape, this.dtype, "cpu", null, null);
    t._cpuData = sliced;
    return t;
  }

  toString(): string {
    const dev = this._device.toUpperCase();
    return `Tensor(shape=[${this.shape}], dtype=${this.dtype}, device=${dev}, bytes=${this.byteLength})`;
  }
}
