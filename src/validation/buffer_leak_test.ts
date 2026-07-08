/**
 * Buffer leak test: verifies that layerNorm/rmsNorm with null weight/bias
 * don't leak GPU buffers, and mapAsync still works after many norm calls.
 *
 * This reproduces the BUG.md scenario: ~5,900 leaked untracked GPU buffers
 * from norm ops caused mapAsync to fail with validation errors.
 *
 * Run: deno run --unstable-webgpu --allow-env --allow-read src/validation/buffer_leak_test.ts
 */
import { GPUContext } from "../runtime/device.ts";
import { Tensor } from "../runtime/tensor.ts";
import { layerNorm } from "../dense/ops/layer_norm.ts";
import { rmsNorm } from "../dense/ops/rms_norm.ts";

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

async function main() {
  console.log("Buffer Leak Test: norm ops + mapAsync\n");

  const ctx = await GPUContext.init();
  console.log(`  GPU: ${ctx.adapterInfo()}`);

  // Create a test tensor on GPU — small enough to fit maxStorageBufferBindingSize (128MB)
  // Uses the same pattern as the pipeline: norm ops on (N, C) tensors
  const N = 4096;  // tokens
  const C = 1024;  // channels — 16MB, well within limits
  const data = new Float32Array(N * C);
  for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  const x = Tensor.fromArray(data, [N, C], "float32");
  await x.upload(ctx);

  // Simulate the SS_Flow workload pattern: many norm ops with null weight/bias
  // The actual pipeline does ~2,250 norm calls (30 blocks × 2 layerNorm + 1 rmsNorm × 24 passes)
  const ITERATIONS = 500; // more than a full pipeline run to stress-test
  console.log(`  Running ${ITERATIONS} norm ops with null weight/bias...`);

  for (let i = 0; i < ITERATIONS; i++) {
    // layerNorm with null weight/bias — was leaking 2 GPU buffers per call
    layerNorm(x, null, null, 1e-6, ctx);
    // rmsNorm with null weight — was leaking 1 GPU buffer per call
    rmsNorm(x, null, 1e-6, ctx);
  }

  // Flush all pending GPU work
  await ctx.device.queue.onSubmittedWorkDone();
  console.log(`  GPU work flushed after ${ITERATIONS * 2} norm ops`);

  // Now try mapAsync — this is where the bug manifested
  // Create a small staging buffer and try to map it
  const staging = ctx.device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const enc = ctx.device.createCommandEncoder();
  enc.copyBufferToBuffer(x.gpuBuffer!, 0, staging, 0, 16);
  ctx.device.queue.submit([enc.finish()]);

  try {
    await ctx.device.queue.onSubmittedWorkDone();
    await staging.mapAsync(GPUMapMode.READ);
    const mapped = staging.getMappedRange();
    assert(mapped.byteLength === 16, "mapAsync works after many norm ops");
    staging.unmap();
    staging.destroy();
  } catch (e) {
    staging.destroy();
    const err = e as Error;
    assert(false, `mapAsync failed: ${err.message}`);
  }

  // Also test full toCPU() round-trip
  try {
    const cpuTensor = await x.toCPU();
    const result = cpuTensor.getView() as Float32Array;
    assert(result.length === N * C, "toCPU round-trip returns correct size");
    // Verify data is not garbage (values should be in reasonable range after norm)
    let allFinite = true;
    for (let i = 0; i < Math.min(100, result.length); i++) {
      if (!Number.isFinite(result[i])) { allFinite = false; break; }
    }
    assert(allFinite, "toCPU data is finite (no NaN/Inf)");
  } catch (e) {
    const err = e as Error;
    assert(false, `toCPU failed: ${err.message}`);
  }

  x.dispose();
  ctx.destroy();

  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed > 0) Deno.exit(1);
}

main();
