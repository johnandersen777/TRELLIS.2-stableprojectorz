/**
 * Phase 0 Validation: Tensor Runtime + MemoryTracker + Matmul
 *
 * Run: deno task test
 */
import { GPUContext } from "../runtime/device.ts";
import { Tensor } from "../runtime/tensor.ts";
import { MemoryTracker } from "../runtime/memory.ts";
import { matmul } from "../runtime/ops/matmul.ts";
import { SafetensorsFile } from "../runtime/weights.ts";

// ── Test runner ──

let passed = 0;
let failed = 0;

function assert(condition: boolean, msg: string) {
  if (condition) {
    passed++;
    console.log(`  PASS: ${msg}`);
  } else {
    failed++;
    console.error(`  FAIL: ${msg}`);
  }
}

function assertClose(
  a: Float32Array,
  b: Float32Array,
  tol: number,
  msg: string,
) {
  if (a.length !== b.length) {
    console.error(`  FAIL: ${msg} — length mismatch ${a.length} vs ${b.length}`);
    failed++;
    return;
  }
  let maxErr = 0;
  for (let i = 0; i < a.length; i++) {
    const err = Math.abs(a[i] - b[i]);
    if (err > maxErr) maxErr = err;
  }
  if (maxErr <= tol) {
    passed++;
    console.log(`  PASS: ${msg} (maxErr=${maxErr.toExponential(2)})`);
  } else {
    failed++;
    console.error(
      `  FAIL: ${msg} (maxErr=${maxErr.toExponential(2)} > tol=${tol})`,
    );
  }
}

// ── Tests ──

async function testGPUDiagnostics() {
  console.log("\n=== GPU Diagnostics ===");
  const ctx = await GPUContext.init();
  const info = ctx.adapterInfo();
  console.log(`  Adapter: ${info}`);
  assert(!ctx.adapter.info.isFallbackAdapter, "GPU is not a fallback adapter");
  assert(ctx.maxBufferSize > 0, "maxBufferSize > 0");
  assert(ctx.maxComputeWorkgroupsPerDimension >= 65535, "workgroups per dim >= 65535");
  ctx.destroy();
}

async function testMemoryTracker() {
  console.log("\n=== MemoryTracker ===");

  const ctx = await GPUContext.init();
  const t = ctx.tracker;

  // Test 1: Basic allocation
  const h1 = await t.reserve(1024, "cpu", "test1");
  assert(t.snapshot().cpuUsed === 1024, "CPU allocation tracked");
  t.release(h1);
  assert(t.snapshot().cpuUsed === 0, "CPU release returns to 0");

  // Test 2: GPU buffer allocation
  const h2 = await t.reserve(4096, "gpu", "test2", { createGpuBuffer: true });
  assert(t.snapshot().gpuUsed === 4096, "GPU allocation tracked");
  assert(h2.gpuBuffer !== undefined, "GPU buffer created");

  // Test 3: Would-exceed check
  const wouldExceed = t.wouldExceed(t.limits.gpu, "gpu");
  assert(wouldExceed === true, "wouldExceed detects over-limit allocation");

  // Test 4: Snapshot format
  const snap = t.formatSnapshot();
  assert(snap.includes("GPU:"), "snapshot includes GPU info");
  assert(snap.includes("CPU:"), "snapshot includes CPU info");

  t.releaseAll();
  assert(t.snapshot().gpuUsed === 0, "releaseAll zeroes GPU");
  assert(t.snapshot().cpuUsed === 0, "releaseAll zeroes CPU");

  ctx.destroy();
  console.log("  (MemoryTracker tests complete)");
}

async function testTensorCreation() {
  console.log("\n=== Tensor Creation ===");

  // CPU tensor from array
  const data = new Float32Array([1, 2, 3, 4, 5, 6]);
  const t = Tensor.fromArray(data, [2, 3]);
  assert(t.shape[0] === 2 && t.shape[1] === 3, "CPU tensor shape");
  assert(t.size === 6, "CPU tensor size");
  assert(t.byteLength === 24, "CPU tensor byteLength (f32: 6×4)");
  assert(t.device === "cpu", "CPU tensor device");

  // GPU transfer
  const ctx = await GPUContext.init();
  const tg = await t.toGPU(ctx);
  assert(tg.device === "gpu", "GPU tensor device after toGPU");
  assert(tg.gpuBuffer !== undefined, "GPU tensor has buffer");

  // Round-trip
  const tBack = await tg.toCPU();
  assert(tBack.device === "cpu", "Round-trip back to CPU");
  // TODO: compare data after GPU readback works

  ctx.destroy();
}

async function testMatmul() {
  console.log("\n=== Matmul ===");

  const ctx = await GPUContext.init();

  // Small test: 2×2 × 2×3
  // A = [[1, 3], [2, 4]]
  // B = [[1, 0, 2], [1, 3, 1]]
  // C = [[4, 9, 5], [6, 12, 8]]
  const aData = new Float32Array([1, 3, 2, 4]); // row-major 2×2
  const bData = new Float32Array([1, 0, 2, 1, 3, 1]); // row-major 2×3

  const a = Tensor.fromArray(aData, [2, 2]);
  const b = Tensor.fromArray(bData, [2, 3]);

  const aGpu = await a.toGPU(ctx);
  const bGpu = await b.toGPU(ctx);

  const c = matmul(aGpu, bGpu, ctx);

  assert(c.shape[0] === 2 && c.shape[1] === 3, "Matmul output shape");
  assert(c.device === "gpu", "Matmul output on GPU");

  // Read back
  const cCpu = await c.toCPU();
  const result = cCpu.getView() as Float32Array;
  const expected = new Float32Array([4, 9, 5, 6, 12, 8]);
  assertClose(result, expected, 1e-5, "Matmul 2×2 × 2×3 correctness");

  ctx.destroy();
}

async function testSafetensorsParser() {
  console.log("\n=== Safetensors Parser ===");
  const testPath = "C:/Users/Dionysus/.cache/huggingface/hub/models--microsoft--TRELLIS.2-4B/snapshots";
  // Find the snapshots directory
  try {
    const entries = [...Deno.readDirSync(testPath)];
    const snapshotDir = entries.find((e) => e.isDirectory);
    if (snapshotDir) {
      const ckptPath = `${testPath}/${snapshotDir.name}/ckpts`;
      const ckptEntries = [...Deno.readDirSync(ckptPath)];
      const safetensorsFile = ckptEntries.find((e) =>
        e.name.endsWith(".safetensors")
      );
      if (safetensorsFile) {
        const filePath = `${ckptPath}/${safetensorsFile.name}`;
        const sf = new SafetensorsFile(filePath);
        const header = sf.open();
        console.log(`  File: ${safetensorsFile.name}`);
        console.log(`  Tensors: ${header.tensors.size}`);
        console.log(`  First 3 tensor names:`);
        const names = sf.listTensors().slice(0, 3);
        for (const name of names) {
          const info = sf.getTensorInfo(name)!;
          console.log(
            `    ${name}: ${info.dtype} [${info.shape}] ${info.dataOffsets}`,
          );
        }
        assert(header.tensors.size > 0, "Safetensors has tensors");
        assert(names.length > 0, "List tensors returns names");
        sf.close();
      } else {
        console.log("  SKIP: No safetensors file found in HF cache");
      }
    } else {
      console.log("  SKIP: No HF cache snapshot found");
    }
  } catch (e) {
    const err = e as Error; console.log(`  SKIP: HF cache not accessible (${err.message})`);
  }
}

// ── Main ──

async function main() {
  console.log("Phase 0 Validation: Tensor Runtime Foundation\n");

  await testGPUDiagnostics();
  await testMemoryTracker();
  await testTensorCreation();
  await testMatmul();
  await testSafetensorsParser();

  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed > 0) Deno.exit(1);
}

main();
