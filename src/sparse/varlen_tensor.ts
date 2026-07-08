/**
 * VarLenTensor — variable-length batch tensor.
 *
 * feats: flat packed tensor (T, C) where T = sum of all batch seq lens.
 * layout: slice for each batch element [start, stop) into feats.
 *
 * Mirrors Python trellis2.modules.sparse.basic.VarLenTensor.
 */
import { Tensor } from "../runtime/tensor.ts";
import type { GPUContext } from "../runtime/device.ts";
import type { BatchLayout } from "./types.ts";
import { layoutFromSeqlens } from "./types.ts";

export class VarLenTensor {
  feats: Tensor;
  layout: BatchLayout[];
  private _cache: Map<string, unknown>;

  constructor(
    feats: Tensor,
    layout: BatchLayout[],
    cache?: Map<string, unknown>,
  ) {
    this.feats = feats;
    this.layout = layout;
    this._cache = cache ?? new Map();
  }

  /** Number of batch elements */
  get batchSize(): number {
    return this.layout.length;
  }

  /** Per-batch sequence lengths */
  get seqlens(): number[] {
    const cached = this._cache.get("seqlens");
    if (cached) return cached as number[];
    const lens = this.layout.map((s) => s.stop - s.start);
    this._cache.set("seqlens", lens);
    return lens;
  }

  /** Cumulative sequence lengths: [0, L1, L1+L2, ...] */
  get cumSeqlens(): number[] {
    const cached = this._cache.get("cumSeqlens");
    if (cached) return cached as number[];
    const cum: number[] = [0];
    for (const len of this.seqlens) {
      cum.push(cum[cum.length - 1] + len);
    }
    this._cache.set("cumSeqlens", cum);
    return cum;
  }

  /** Shape: [batchSize, *feats.shape[1:]] */
  get shape(): number[] {
    return [this.batchSize, ...this.feats.shape.slice(1)];
  }

  get dtype() { return this.feats.dtype; }
  get device() { return this.feats.device; }
  get context(): GPUContext | null { return this.feats.context; }

  // ── Factory ────────────────────────────────────────────

  /** Create from a list of per-batch tensors */
  static fromTensorList(tensors: Tensor[]): VarLenTensor {
    // Concat along dim 0
    // For CPU tensors, merge ArrayBuffers
    const totalLen = tensors.reduce((s, t) => s + t.shape[0], 0);
    const channels = tensors[0].shape.length > 1
      ? tensors[0].shape.slice(1).reduce((a, b) => a * b, 1)
      : 1;

    // Build concatenation
    const layout = layoutFromSeqlens(tensors.map((t) => t.shape[0]));
    const dtype = tensors[0].dtype;

    if (tensors[0].device === "cpu") {
      const bytesPerElem = tensors[0].byteLength / tensors[0].size;
      const totalBytes = totalLen * channels * bytesPerElem;
      const ab = new ArrayBuffer(totalBytes);
      const dst = new Uint8Array(ab);
      let offset = 0;
      for (const t of tensors) {
        const src = new Uint8Array(t.getCPUData());
        dst.set(src, offset);
        offset += src.length;
      }
      const feats = Tensor.fromBuffer(ab, [totalLen, channels], dtype);
      return new VarLenTensor(feats, layout);
    }

    // GPU: for single tensor (B=1, common inference case), wrap directly
    if (tensors.length === 1) {
      const layout = layoutFromSeqlens([tensors[0].shape[0]]);
      return new VarLenTensor(tensors[0], layout);
    }
    // Multi-tensor GPU concat: download, concat on CPU, re-upload
    // Caller must handle async — use static fromTensorListAsync
    throw new Error(
      "GPU VarLenTensor.fromTensorList requires B=1. " +
      "For multi-batch, use VarLenTensor.fromTensorListAsync()",
    );
  }

  /** Async version for GPU tensor concatenation (multi-batch) */
  static async fromTensorListAsync(tensors: Tensor[]): Promise<VarLenTensor> {
    const downloads = await Promise.all(tensors.map(t => t.toCPU()));
    const result = VarLenTensor.fromTensorList(downloads);
    downloads.forEach(t => t.dispose());
    return result;
  }

  // ── Indexing ────────────────────────────────────────────

  /** Get a single batch element */
  getBatch(index: number): Tensor {
    const sl = this.layout[index];
    const feats = this.feats;
    const channels = feats.shape[1] ?? 1;

    if (feats.device === "cpu") {
      const bytesPerElem = feats.byteLength / feats.size;
      const len = sl.stop - sl.start;
      const byteOffset = sl.start * channels * bytesPerElem;
      const byteLength = len * channels * bytesPerElem;
      const src = feats.getCPUData();
      const ab = new ArrayBuffer(byteLength);
      new Uint8Array(ab).set(
        new Uint8Array(src, byteOffset, byteLength),
      );
      return Tensor.fromBuffer(ab, [len, channels], feats.dtype);
    }

    // GPU: download to CPU, slice, return CPU tensor
    // Caller re-uploads if needed. For inference B=1, never called.
    throw new Error(
      "GPU VarLenTensor.getBatch requires CPU roundtrip. " +
      "Use tensor.toCPU() first, then getBatch(). For B=1 inference, this is never needed.",
    );
  }

  // ── Replace ─────────────────────────────────────────────

  /** Replace feats, keeping layout and cache */
  replace(feats: Tensor): VarLenTensor {
    const vt = new VarLenTensor(feats, this.layout);
    vt._cache = this._cache;
    return vt;
  }

  // ── Device transfer ─────────────────────────────────────

  async toGPU(context: GPUContext): Promise<VarLenTensor> {
    const gpuFeats = await this.feats.toGPU(context);
    const vt = new VarLenTensor(gpuFeats, this.layout);
    vt._cache = this._cache;
    return vt;
  }

  async toCPU(): Promise<VarLenTensor> {
    const cpuFeats = await this.feats.toCPU();
    const vt = new VarLenTensor(cpuFeats, this.layout);
    vt._cache = this._cache;
    return vt;
  }

  // ── Utility ─────────────────────────────────────────────

  toString(): string {
    return `VarLenTensor(batch=${this.batchSize}, shape=[${this.shape}], dtype=${this.dtype}, device=${this.device})`;
  }
}
