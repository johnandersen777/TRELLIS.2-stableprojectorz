/**
 * Memory tracker for VRAM/RAM profiling.
 * Hooks into Deno.memoryUsage() + manual GPU buffer tracking.
 */
export class MemoryTracker {
  private gpuBytes = 0;
  private peakGpuBytes = 0;
  private peakRss = 0;
  private peakHeap = 0;
  private labels: string[] = [];
  private snapshots: MemorySnapshot[] = [];

  log(label: string): void {
    const mem = Deno.memoryUsage();
    const rss = mem.rss;
    const heap = mem.heapTotal + mem.external;
    this.peakRss = Math.max(this.peakRss, rss);
    this.peakHeap = Math.max(this.peakHeap, heap);
    this.peakGpuBytes = Math.max(this.peakGpuBytes, this.gpuBytes);
    this.snapshots.push({
      label,
      rss: rss / 1e6,
      heap: heap / 1e6,
      gpu: this.gpuBytes / 1e6,
    });
    console.log(
      `[mem] ${label}: RSS=${(rss / 1e6).toFixed(0)}MB ` +
      `Heap=${(heap / 1e6).toFixed(0)}MB ` +
      `GPU=${(this.gpuBytes / 1e6).toFixed(0)}MB`,
    );
  }

  addGpuBytes(bytes: number): void {
    this.gpuBytes += bytes;
    this.peakGpuBytes = Math.max(this.peakGpuBytes, this.gpuBytes);
  }

  removeGpuBytes(bytes: number): void {
    this.gpuBytes -= bytes;
  }

  addGpuBytesPeak(bytes: number): void {
    // Track peak without accumulating (for temp buffers)
    this.peakGpuBytes = Math.max(this.peakGpuBytes, Math.max(this.gpuBytes + bytes, this.gpuBytes));
  }

  summary(): string {
    return `Peak:RSS=${(this.peakRss / 1e9).toFixed(1)}GB Heap=${(this.peakHeap / 1e9).toFixed(1)}GB GPU=${(this.peakGpuBytes / 1e9).toFixed(2)}GB`;
  }

  report(): string {
    const lines = ["\nMemory Report", "=".repeat(60)];
    for (const s of this.snapshots) {
      lines.push(
        `  ${s.label.padEnd(24)} RSS=${s.rss.toFixed(0).padStart(5)}MB  ` +
        `Heap=${s.heap.toFixed(0).padStart(5)}MB  GPU=${s.gpu.toFixed(0).padStart(5)}MB`,
      );
    }
    lines.push("=".repeat(60));
    lines.push(`  ${this.summary()}`);
    return lines.join("\n");
  }
}

interface MemorySnapshot {
  label: string;
  rss: number;
  heap: number;
  gpu: number;
}

// Singleton for convenience
let globalTracker: MemoryTracker | null = null;

export function getMemoryTracker(): MemoryTracker {
  if (!globalTracker) globalTracker = new MemoryTracker();
  return globalTracker;
}

export function resetMemoryTracker(): void {
  globalTracker = new MemoryTracker();
}

/**
 * Log GPU buffer sizes from a model's tensors.
 */
export function logGpuBuffers(
  tracker: MemoryTracker,
  label: string,
  byteSize: number,
): void {
  tracker.addGpuBytes(byteSize);
  tracker.log(label);
}
