/**
 * MemoryTracker — bookkeeping-based GPU+CPU memory tracking.
 *
 * WebGPU has NO VRAM query API. We track every allocation manually.
 * Hard limits prevent OOM. LRU eviction frees GPU buffers when pressure hits.
 * Inspired by Python pipeline's low_vram mode + conv_flex_gemm chunking patterns.
 *
 * Real-world data (from live pipeline run):
 *   - Python peak: 23.9 GB CPU + 6.1 GB GPU reserved (2.2 GB allocated)
 *   - Our target:  <10 GB CPU + <4 GB GPU
 *   - CPU RAM is the real bottleneck (31 GB → 0 free in Python). We MUST stream.
 */

export type MemoryDomain = "gpu" | "cpu" | "staging";

export interface AllocationHandle {
  readonly id: number;
  readonly size: number;
  readonly domain: MemoryDomain;
  readonly label: string;
  readonly timestamp: number;
  gpuBuffer?: GPUBuffer;
  arrayBuffer?: ArrayBuffer;
}

export interface MemoryLimits {
  gpu: number; // bytes — default 14 GB (2 GB headroom on 16 GB card)
  cpu: number; // bytes — default 10 GB (conservative, Python used 23 GB)
}

export interface MemorySnapshot {
  gpuUsed: number;
  cpuUsed: number;
  stagingUsed: number;
  gpuLimit: number;
  cpuLimit: number;
  handleCount: number;
  byLabel: Map<string, number>;
}

export interface Evictable {
  readonly handle: AllocationHandle;
  evictToCpu(): Promise<ArrayBuffer>;
  restoreFromCpu(buffer: ArrayBuffer): Promise<void>;
  readonly size: number;
  lastUsed: number;
}

export class WebGpuMemoryError extends Error {
  constructor(
    message: string,
    public readonly snapshot: MemorySnapshot,
    public readonly attemptedSize: number,
    public readonly domain: MemoryDomain,
  ) {
    super(message);
    this.name = "WebGpuMemoryError";
  }
}

function formatBytes(bytes: number): string {
  if (bytes >= 1024 ** 4) return `${(bytes / 1024 ** 4).toFixed(2)} TiB`;
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KiB`;
  return `${bytes} B`;
}

export class MemoryTracker {
  private nextId = 0;
  private handles = new Map<number, AllocationHandle>();

  private gpuAllocated = 0;
  private cpuAllocated = 0;
  private stagingAllocated = 0;

  readonly limits: MemoryLimits;
  private device: GPUDevice;
  private lruList: Evictable[] = [];
  private errorScopeDepth = 0;

  // Watchdog
  private operationTimer: AbortController | null = null;

  verbose: boolean;

  constructor(device: GPUDevice, limits?: Partial<MemoryLimits>, verbose = true) {
    this.device = device;
    this.verbose = verbose;
    this.limits = {
      gpu: limits?.gpu ?? Math.floor(14 * 1024 ** 3), // 14 GB default
      cpu: limits?.cpu ?? Math.floor(10 * 1024 ** 3), // 10 GB — Python used 23 GB!
    };
    this.setupErrorHandler();
  }

  // ── Public API ────────────────────────────────────────────

  wouldExceed(size: number, domain: MemoryDomain): boolean {
    const current = this.getAllocated(domain);
    const limit = this.getLimit(domain);
    return current + size > limit;
  }

  /**
   * Reserve and track memory. Throws WebGpuMemoryError if would exceed limit.
   * For GPU: pushes OOM error scope for catchable detection.
   */
  async reserve(
    size: number,
    domain: MemoryDomain,
    label: string,
    options?: {
      createGpuBuffer?: boolean;
      gpuBufferUsage?: GPUBufferUsageFlags;
      mappedAtCreation?: boolean;
      contents?: ArrayBuffer;
      evictable?: boolean;
    },
  ): Promise<AllocationHandle> {
    // 1. Pre-flight check
    if (this.wouldExceed(size, domain)) {
      if (domain === "gpu" && options?.evictable) {
        await this.evictLru(size);
      }
      if (this.wouldExceed(size, domain)) {
        throw new WebGpuMemoryError(
          `Allocation of ${formatBytes(size)} on ${domain} would exceed limit`,
          this.snapshot(),
          size,
          domain,
        );
      }
    }

    // 2. Check maxBufferSize for GPU
    if (domain === "gpu") {
      const maxBuf = Number(this.device.limits.maxBufferSize);
      if (size > maxBuf) {
        throw new WebGpuMemoryError(
          `Buffer size ${formatBytes(size)} exceeds device maxBufferSize ${formatBytes(maxBuf)}. Split the buffer.`,
          this.snapshot(),
          size,
          domain,
        );
      }
    }

    // 3. Push OOM error scope
    if (domain === "gpu") {
      this.device.pushErrorScope("out-of-memory");
      this.errorScopeDepth++;
    }

    // 4. Create the buffer
    let gpuBuffer: GPUBuffer | undefined;
    let arrayBuffer: ArrayBuffer | undefined;

    if (domain === "gpu" && options?.createGpuBuffer) {
      try {
        const desc: GPUBufferDescriptor = {
          size,
          usage: options.gpuBufferUsage ??
            (GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
          mappedAtCreation: options.mappedAtCreation ?? false,
        };
        gpuBuffer = this.device.createBuffer(desc);

        if (options.contents) {
          this.device.queue.writeBuffer(gpuBuffer, 0, options.contents);
        }
      } catch (e) {
        if (this.errorScopeDepth > 0) {
          await this.device.popErrorScope();
          this.errorScopeDepth--;
        }
        throw e;
      }
    } else if (domain === "staging" && options?.createGpuBuffer) {
      // Staging buffer: GPU buffer with MAP_READ
      try {
        const desc: GPUBufferDescriptor = {
          size,
          usage: options.gpuBufferUsage ??
            (GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
        };
        gpuBuffer = this.device.createBuffer(desc);
      } catch (e) {
        if (this.errorScopeDepth > 0) {
          await this.device.popErrorScope();
          this.errorScopeDepth--;
        }
        throw e;
      }
    } else if (domain === "cpu") {
      try {
        arrayBuffer = new ArrayBuffer(size);
        if (options?.contents) {
          const dst = new Uint8Array(arrayBuffer);
          dst.set(new Uint8Array(options.contents));
        }
      } catch (e) {
        throw new WebGpuMemoryError(
          `Failed to allocate ${formatBytes(size)} CPU ArrayBuffer`,
          this.snapshot(),
          size,
          domain,
        );
      }
    }

    // 5. Pop error scope
    if (domain === "gpu" && this.errorScopeDepth > 0) {
      const error = await this.device.popErrorScope();
      this.errorScopeDepth--;
      if (error) {
        if (gpuBuffer) gpuBuffer.destroy();
        if (options?.evictable) {
          await this.evictLru(size);
          return this.reserve(size, domain, label, options);
        }
        throw new WebGpuMemoryError(
          `GPU OOM allocating ${formatBytes(size)}: ${error.message}`,
          this.snapshot(),
          size,
          domain,
        );
      }
    }

    // 6. Register
    const handle: AllocationHandle = {
      id: this.nextId++,
      size,
      domain,
      label,
      timestamp: Date.now(),
      gpuBuffer,
      arrayBuffer,
    };
    this.handles.set(handle.id, handle);
    this.addToAllocated(domain, size);

    this.logAllocation(handle);
    return handle;
  }

  release(handle: AllocationHandle): void {
    if (!this.handles.has(handle.id)) return;

    if (handle.gpuBuffer) {
      handle.gpuBuffer.destroy();
    }

    this.handles.delete(handle.id);
    this.subtractFromAllocated(handle.domain, handle.size);

    const lruIdx = this.lruList.findIndex((e) => e.handle.id === handle.id);
    if (lruIdx >= 0) this.lruList.splice(lruIdx, 1);
  }

  /** Register an externally-created GPU buffer for tracking. */
  trackExistingBuffer(buffer: GPUBuffer, size: number, label: string): AllocationHandle {
    const handle: AllocationHandle = {
      id: this.nextId++,
      size,
      domain: "gpu",
      label,
      timestamp: Date.now(),
      gpuBuffer: buffer,
    };
    this.handles.set(handle.id, handle);
    this.gpuAllocated += size;
    return handle;
  }

  releaseAll(): void {
    for (const handle of this.handles.values()) {
      if (handle.gpuBuffer) handle.gpuBuffer.destroy();
    }
    this.handles.clear();
    this.gpuAllocated = 0;
    this.cpuAllocated = 0;
    this.stagingAllocated = 0;
    this.lruList = [];
  }

  snapshot(): MemorySnapshot {
    const byLabel = new Map<string, number>();
    for (const h of this.handles.values()) {
      byLabel.set(h.label, (byLabel.get(h.label) ?? 0) + h.size);
    }
    return {
      gpuUsed: this.gpuAllocated,
      cpuUsed: this.cpuAllocated,
      stagingUsed: this.stagingAllocated,
      gpuLimit: this.limits.gpu,
      cpuLimit: this.limits.cpu,
      handleCount: this.handles.size,
      byLabel,
    };
  }

  formatSnapshot(): string {
    const s = this.snapshot();
    const gpuPct = (s.gpuUsed / s.gpuLimit * 100).toFixed(1);
    const cpuPct = (s.cpuUsed / s.cpuLimit * 100).toFixed(1);
    return (
      `[MEM] GPU: ${formatBytes(s.gpuUsed)}/${formatBytes(s.gpuLimit)} (${gpuPct}%) | ` +
      `CPU: ${formatBytes(s.cpuUsed)}/${formatBytes(s.cpuLimit)} (${cpuPct}%) | ` +
      `Staging: ${formatBytes(s.stagingUsed)} | Handles: ${s.handleCount}`
    );
  }

  // ── LRU Eviction ─────────────────────────────────────────

  registerEvictable(evictable: Evictable): void {
    this.lruList.push(evictable);
  }

  touchEvictable(handleId: number): void {
    const entry = this.lruList.find((e) => e.handle.id === handleId);
    if (entry) entry.lastUsed = Date.now();
  }

  async evictLru(neededBytes: number): Promise<number> {
    const sorted = [...this.lruList].sort((a, b) => a.lastUsed - b.lastUsed);
    let freed = 0;

    for (const entry of sorted) {
      if (freed >= neededBytes) break;
      if (entry.handle.domain !== "gpu") continue;

      const evicted = await entry.evictToCpu();
      if (entry.handle.gpuBuffer) {
        entry.handle.gpuBuffer.destroy();
        entry.handle.gpuBuffer = undefined;
      }
      this.gpuAllocated -= entry.size;
      freed += entry.size;
      console.log(
        `[MEM] LRU evicted: ${entry.handle.label} (${formatBytes(entry.size)})`,
      );
    }
    return freed;
  }

  // ── Watchdog ──────────────────────────────────────────────

  startOperation(label: string, timeoutMs: number = 30 * 60 * 1000): AbortSignal {
    if (this.operationTimer) {
      this.operationTimer.abort("concurrent operation started");
    }
    this.operationTimer = new AbortController();

    setTimeout(() => {
      if (!this.operationTimer?.signal.aborted) {
        console.error(
          `[WATCHDOG] Operation "${label}" timed out after ${timeoutMs}ms`,
        );
        this.operationTimer?.abort(`Operation "${label}" timed out`);
      }
    }, timeoutMs);

    return this.operationTimer.signal;
  }

  endOperation(): void {
    this.operationTimer?.abort("completed");
    this.operationTimer = null;
  }

  // ── Progressive Download ──────────────────────────────────

  async downloadBuffer(
    gpuBuffer: GPUBuffer,
    size: number,
    label: string,
  ): Promise<ArrayBuffer> {
    const staging = await this.reserve(size, "staging", `staging:${label}`, {
      createGpuBuffer: true,
      gpuBufferUsage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    try {
      const encoder = this.device.createCommandEncoder();
      encoder.copyBufferToBuffer(gpuBuffer, 0, staging.gpuBuffer!, 0, size);
      this.device.queue.submit([encoder.finish()]);

      await this.device.queue.onSubmittedWorkDone();

      // Push validation error scope to catch mapAsync failures
      this.device.pushErrorScope("validation");
      await staging.gpuBuffer!.mapAsync(GPUMapMode.READ);
      const mapError = await this.device.popErrorScope();
      if (mapError) {
        throw new Error(`mapAsync validation error: ${mapError.message}`);
      }

      const mapped = staging.gpuBuffer!.getMappedRange();
      const result = mapped.slice(0);
      staging.gpuBuffer!.unmap();
      return result;
    } finally {
      this.release(staging);
    }
  }

  // ── Private ───────────────────────────────────────────────

  private setupErrorHandler(): void {
    this.device.addEventListener("uncapturederror", (event) => {
      const err = (event as GPUUncapturedErrorEvent).error;
      console.error(`[MEM] UNCAPTURED ERROR: ${err.message}`);
      console.error(this.formatSnapshot());
    });

    this.device.lost.then((info) => {
      console.error(`[MEM] DEVICE LOST: ${info.reason} — ${info.message}`);
      console.error(this.formatSnapshot());
      this.releaseAll();
    });
  }

  private getAllocated(domain: MemoryDomain): number {
    switch (domain) {
      case "gpu": return this.gpuAllocated;
      case "cpu": return this.cpuAllocated;
      case "staging": return this.stagingAllocated;
    }
  }

  private getLimit(domain: MemoryDomain): number {
    switch (domain) {
      case "gpu": return this.limits.gpu;
      case "cpu": return this.limits.cpu;
      case "staging": return Infinity;
    }
  }

  private addToAllocated(domain: MemoryDomain, size: number): void {
    switch (domain) {
      case "gpu": this.gpuAllocated += size; break;
      case "cpu": this.cpuAllocated += size; break;
      case "staging": this.stagingAllocated += size; break;
    }
  }

  private subtractFromAllocated(domain: MemoryDomain, size: number): void {
    switch (domain) {
      case "gpu": this.gpuAllocated -= size; break;
      case "cpu": this.cpuAllocated -= size; break;
      case "staging": this.stagingAllocated -= size; break;
    }
  }

  private logAllocation(handle: AllocationHandle): void {
    if (!this.verbose) return;
    const dom = handle.domain.toUpperCase();
    const usage = this.getAllocated(handle.domain);
    const limit = this.getLimit(handle.domain);
    console.log(
      `[MEM] ALLOC ${handle.label}: ${formatBytes(handle.size)} ${dom} ` +
        `(total: ${formatBytes(usage)}/${formatBytes(limit)})`,
    );
  }
}
