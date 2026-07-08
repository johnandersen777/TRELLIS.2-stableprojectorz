/**
 * SparseTensor — sparse tensor with coordinates, features, and spatial metadata.
 *
 * Extends VarLenTensor with:
 *   - coords: (T, 4) [batch, x, y, z] int32 tensor
 *   - spatial_shape: (3,) [X, Y, Z] max coordinate bounds
 *   - scale: per-dimension scale fraction
 *   - spatial_cache: scale-keyed dict for neighbor maps, layouts, etc.
 *
 * Mirrors Python trellis2.modules.sparse.basic.SparseTensor.
 */
import { Tensor } from "../runtime/tensor.ts";
import type { GPUContext } from "../runtime/device.ts";
import { VarLenTensor } from "./varlen_tensor.ts";
import type { BatchLayout, Scale } from "./types.ts";
import { SCALE_IDENTITY, scaleToString } from "./types.ts";

export class SparseTensor extends VarLenTensor {
  coords: Tensor; // (T, 4) int32 [batch, x, y, z]
  private _shape: number[] | null;
  private _scale: Scale;
  private _spatial_cache: Record<string, Record<string, unknown>>;

  constructor(
    feats: Tensor,
    coords: Tensor,
    shape?: number[] | null,
    scale?: Scale,
    cache?: Record<string, Record<string, unknown>>,
    layout?: BatchLayout[],
  ) {
    // Compute layout from coords if not provided
    const computedLayout = layout ?? computeLayout(coords);
    super(feats, computedLayout);
    this.coords = coords;
    this._shape = shape ?? null;
    this._scale = scale ?? SCALE_IDENTITY;
    this._spatial_cache = cache ?? {};
  }

  // ── Computed properties ──────────────────────────────────

  /** Spatial shape: [X, Y, Z] = max coordinate + 1 per dimension */
  get spatialShape(): number[] {
    const cached = this.getSpatialCache("shape");
    if (cached) return cached as number[];

    // Compute from coords
    const shape = [0, 0, 0];
    if (this.coords.device === "cpu") {
      const c = this.coords.getView() as Int32Array;
      for (let i = 0; i < c.length; i += 4) {
        shape[0] = Math.max(shape[0], c[i + 1] + 1);
        shape[1] = Math.max(shape[1], c[i + 2] + 1);
        shape[2] = Math.max(shape[2], c[i + 3] + 1);
      }
    }
    this.registerSpatialCache("shape", shape);
    return shape;
  }

  /** Full shape: [batchSize, *feats shape channels...] */
  override get shape(): number[] {
    if (this._shape) return this._shape;
    const s = [this.batchSize, ...this.feats.shape.slice(1)];
    this._shape = s;
    return s;
  }

  /** Per-voxel spatial scale */
  get scale(): Scale {
    return this._scale;
  }

  // ── Spatial cache ────────────────────────────────────────

  getSpatialCache(key: string): unknown | undefined {
    const scaleKey = scaleToString(this._scale);
    return this._spatial_cache[scaleKey]?.[key];
  }

  registerSpatialCache(key: string, value: unknown): void {
    const scaleKey = scaleToString(this._scale);
    if (!this._spatial_cache[scaleKey]) {
      this._spatial_cache[scaleKey] = {};
    }
    this._spatial_cache[scaleKey][key] = value;
  }

  clearSpatialCache(): void {
    this._spatial_cache = {};
  }

  /** Clear GPU spatial cache (matches Python's _spatial_cache.clear()) */
  clearGpuCache(): void {
    for (const scaleKey of Object.keys(this._spatial_cache)) {
      const cache = this._spatial_cache[scaleKey];
      for (const key of Object.keys(cache)) {
        const val = cache[key];
        // If it's a GPU buffer, destroy it
        if (val && typeof val === "object" && "destroy" in val) {
          (val as GPUBuffer).destroy();
        }
        delete cache[key];
      }
    }
  }

  // ── Replace ──────────────────────────────────────────────

  /** Replace feats, keeping coords, layout, scale, and cache */
  override replace(feats: Tensor, coords?: Tensor): SparseTensor {
    return new SparseTensor(
      feats,
      coords ?? this.coords,
      this._shape,
      this._scale,
      this._spatial_cache,
      this.layout,
    );
  }

  // ── Device transfer ─────────────────────────────────────

  override async toGPU(context: GPUContext): Promise<SparseTensor> {
    const gpuFeats = await this.feats.toGPU(context);
    const gpuCoords = await this.coords.toGPU(context);
    return new SparseTensor(
      gpuFeats,
      gpuCoords,
      this._shape,
      this._scale,
      this._spatial_cache,
      this.layout,
    );
  }

  override async toCPU(): Promise<SparseTensor> {
    const cpuFeats = await this.feats.toCPU();
    const cpuCoords = await this.coords.toCPU();
    return new SparseTensor(
      cpuFeats,
      cpuCoords,
      this._shape,
      this._scale,
      this._spatial_cache,
      this.layout,
    );
  }

  // ── Factory ──────────────────────────────────────────────

  // Note: different signature from base class (adds coordsList param)
  // Use SparseTensor.fromFeatsAndCoords() for the full signature
  static override fromTensorList(tensors: Tensor[]): SparseTensor {
    throw new Error("Use SparseTensor.fromFeatsAndCoords(featsList, coordsList)");
  }

  static fromFeatsAndCoords(
    featsList: Tensor[],
    coordsList: Tensor[],
  ): SparseTensor {
    const totalFeats = featsList.reduce((s, t) => s + t.shape[0], 0);
    const channels = featsList[0].shape[1] ?? 1;

    // Concatenate feats
    const featsDtype = featsList[0].dtype;
    const featsBytesPerElem = featsDtype === "float16" ? 2 : 4;
    const totalFeatsBytes = totalFeats * channels * featsBytesPerElem;
    const featsAb = new ArrayBuffer(totalFeatsBytes);
    const featsDst = new Uint8Array(featsAb);
    let featsOff = 0;
    for (const t of featsList) {
      const src = new Uint8Array(t.getCPUData());
      featsDst.set(src, featsOff);
      featsOff += src.length;
    }
    const feats = Tensor.fromBuffer(featsAb, [totalFeats, channels], featsDtype);

    // Concatenate coords (int32, 4 channels per voxel)
    const totalCoordsBytes = totalFeats * 4 * 4; // int32 × 4 fields
    const coordsAb = new ArrayBuffer(totalCoordsBytes);
    const coordsDst = new Uint8Array(coordsAb);
    let coordsOff = 0;
    for (let b = 0; b < coordsList.length; b++) {
      const c = coordsList[b];
      const src = new Uint8Array(c.getCPUData());
      // Offset batch index
      const srcView = new Int32Array(c.getCPUData());
      const dstView = new Int32Array(
        coordsAb,
        coordsOff,
        srcView.length,
      );
      dstView.set(srcView);
      // Fixup batch index
      for (let i = 0; i < srcView.length; i += 4) {
        dstView[i] = b;
      }
      coordsOff += src.length;
    }
    const coords = Tensor.fromBuffer(coordsAb, [totalFeats, 4], "int32");

    return new SparseTensor(feats, coords);
  }

  // ── Utility ──────────────────────────────────────────────

  override toString(): string {
    const ss = this.spatialShape;
    return `SparseTensor(voxels=${this.feats.shape[0]}, feats=[${this.shape}], spatial=[${ss}], dtype=${this.dtype})`;
  }
}

/** Compute batch layout from coords */
function computeLayout(coords: Tensor): BatchLayout[] {
  if (coords.device !== "cpu") {
    // GPU: defer layout computation
    return [{ start: 0, stop: coords.shape[0] }];
  }
  const c = coords.getView() as Int32Array;
  const batchSize = c.length > 0 ? c[c.length - 4] + 1 : 1;

  // Count per batch
  const counts = new Array(batchSize).fill(0);
  for (let i = 0; i < c.length; i += 4) {
    counts[c[i]]++;
  }

  let start = 0;
  const layout: BatchLayout[] = [];
  for (const count of counts) {
    layout.push({ start, stop: start + count });
    start += count;
  }
  return layout;
}
