/**
 * Matrix multiplication op: C = A × B.
 * Dispatches tiled WGSL matmul compute shader.
 */
import { Tensor } from "../tensor.ts";
import { type GPUContext } from "../device.ts";
import { getOrCreateShader, getOrCreatePipeline } from "../shader_cache.ts";
import {
  makeMatmulShader,
  matmulDispatch,
  matmulShaderKey,
  matmulWorkgroupSize,
  type MatmulShaderConfig,
} from "../shaders/wgsl/matmul.ts";

export function matmul(a: Tensor, b: Tensor, context: GPUContext): Tensor {
  if (a.shape.length !== 2 || b.shape.length !== 2) {
    throw new Error(`matmul requires 2D tensors, got ${a.shape} and ${b.shape}`);
  }
  const M = a.shape[0];
  const K = a.shape[1];
  const KN = b.shape[0];
  const N = b.shape[1];

  if (K !== KN) {
    throw new Error(`matmul shape mismatch: A(${M},${K}) × B(${KN},${N})`);
  }

  if (a.device !== "gpu" || b.device !== "gpu") {
    throw new Error("matmul requires GPU tensors — call toGPU() first");
  }

  const device = context.device;

  // Build shader config
  const scalarType = context.supportsFP16 && a.dtype === "float16"
    ? "f16"
    : context.scalarType(a.dtype);
  const config: MatmulShaderConfig = {
    scalarType,
    wgType: scalarType,
    vec4Type: `vec4<${scalarType}>`,
    enableF16: context.supportsFP16 && scalarType === "f16",
  };

  // Create or get pipeline
  const shaderKey = matmulShaderKey(config);
  const code = context.precisionPrelude() + makeMatmulShader(config);
  const shaderModule = getOrCreateShader(device, shaderKey, code);
  const pipeline = getOrCreatePipeline(device, shaderKey, shaderModule);

  // Dimension uniform
  const dimsData = new Uint32Array([M, N, K]);
  const dimsBuffer = device.createBuffer({
    size: 16, // 4 × u32, padded
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(dimsBuffer, 0, dimsData);

  // Output buffer
  const outBytes = M * N * (a.dtype === "float16" ? 2 : 4);
  const outBuffer = device.createBuffer({
    size: outBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: `matmul_out_${M}x${N}`,
  });

  // Bind group
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: a.gpuBuffer! } },
      { binding: 1, resource: { buffer: b.gpuBuffer! } },
      { binding: 2, resource: { buffer: outBuffer } },
      { binding: 3, resource: { buffer: dimsBuffer } },
    ],
  });

  // Dispatch
  const [wgX, wgY, wgZ] = matmulDispatch(M, N);
  const wgSize = matmulWorkgroupSize();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(wgX, wgY, wgZ);
  pass.end();
  device.queue.submit([encoder.finish()]);

  // Return tensor with the output buffer. Caller responsible for disposal.
  dimsBuffer.destroy(); // uniform buffer no longer needed
  return Tensor.fromGpuBuffer(outBuffer, [M, N], a.dtype, context, `matmul_out_${M}x${N}`);
}
