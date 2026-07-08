import { GPUContext } from "./src/runtime/device.ts";

const ctx = await GPUContext.init();
const device = ctx.device;

// Simulate reserve() pattern: push OOM scope, create buffer, pop scope
for (let i = 0; i < 5000; i++) {
  device.pushErrorScope("out-of-memory");
  const buf = device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buf, 0, new Uint32Array([i]));
  await device.popErrorScope();
  buf.destroy();
}
console.log("5000 push/pop/create/destroy cycles done");

// Now try mapAsync
await device.queue.onSubmittedWorkDone();
const staging = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
device.queue.writeBuffer(staging, 0, new Uint32Array([42]));
await device.queue.onSubmittedWorkDone();
try {
  await staging.mapAsync(GPUMapMode.READ);
  const val = new Uint32Array(staging.getMappedRange())[0];
  console.log(`mapAsync OK, value: ${val}`);
  staging.unmap();
  staging.destroy();
} catch (e) {
  console.error(`mapAsync FAILED: ${(e as Error).message}`);
  staging.destroy();
}

ctx.destroy();
