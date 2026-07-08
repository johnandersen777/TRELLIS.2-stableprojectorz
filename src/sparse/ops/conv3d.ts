/**
 * SparseConv3d — submanifold sparse 3D convolution.
 *
 * Ports conv_none.py algorithm: neighbor map + gather + GEMM + bias.
 * Weight layout: (Co, Kd, Kh, Kw, Ci) — matches checkpoint format.
 */
import { Tensor } from "../../runtime/tensor.ts";
import { type GPUContext } from "../../runtime/device.ts";
import { getOrCreateShader, getOrCreatePipeline } from "../../runtime/shader_cache.ts";
import { SparseTensor } from "../sparse_tensor.ts";
import type { ConvParams } from "./conv3d_wgsl.ts";
import {
  makePass0Shader,
  makePass1Shader,
  makePass2Shader,
} from "./conv3d_wgsl.ts";

export class SparseConv3d {
  weight: Tensor; // (Co, Kd, Kh, Kw, Ci)
  bias: Tensor | null; // (Co,)
  kernelSize: [number, number, number];
  dilation: [number, number, number];

  constructor(
    weight: Tensor,
    bias?: Tensor | null,
    kernelSize: [number, number, number] = [3, 3, 3],
    dilation: [number, number, number] = [1, 1, 1],
  ) {
    this.weight = weight;
    this.bias = bias ?? null;
    this.kernelSize = kernelSize;
    this.dilation = dilation;
  }

  /** Number of kernel positions V = Kd * Kh * Kw */
  get V(): number {
    return this.kernelSize[0] * this.kernelSize[1] * this.kernelSize[2];
  }

  forward(x: SparseTensor, context: GPUContext): SparseTensor {
    if (x.device !== "gpu") throw new Error("SparseConv3d requires GPU input");

    const [Kd, Kh, Kw] = this.kernelSize;
    const [dilD, dilH, dilW] = this.dilation;
    const [sx, sy, sz] = x.spatialShape;
    const N = x.feats.shape[0];
    const Ci = this.weight.shape[4]; // (Co, Kd, Kh, Kw, Ci)
    const Co = this.weight.shape[0];
    const V = this.V;

    const device = context.device;

    const params: ConvParams = {
      N, Ci, Co, X: sx, Y: sy, Z: sz, Kd, Kh, Kw, dilD, dilH, dilW,
    };
    const paramsData = new Uint32Array([
      N, Ci, Co, sx, sy, sz, 0,
      Kd, Kh, Kw, dilD, dilH, dilW, 0, 0,
    ]);

    // ── Pass 0+1: Build neighbor map (cached) ──
    const cacheKey =
      `conv_neighbors_${Kw}x${Kh}x${Kd}_dil${dilW}x${dilH}x${dilD}`;
    let neighborMapBuf: GPUBuffer | undefined;

    const cached = x.getSpatialCache(cacheKey) as
      | { buffer: GPUBuffer; size: number }
      | undefined;
    if (cached) {
      neighborMapBuf = cached.buffer;
    } else {
      // Allocate sp_to_lin: sx*sy*sz * 4 bytes (int32)
      const spSize = sx * sy * sz * 4;
      const spBuf = device.createBuffer({
        size: spSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      // Init to -1
      const initData = new Int32Array(sx * sy * sz);
      initData.fill(-1);
      device.queue.writeBuffer(spBuf, 0, initData);

      // Allocate neighbor_map: N*V*4 bytes (int32)
      const nmSize = N * V * 4;
      neighborMapBuf = device.createBuffer({
        size: nmSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const nmInit = new Int32Array(N * V);
      nmInit.fill(-1);
      device.queue.writeBuffer(neighborMapBuf, 0, nmInit);

      // Uniform buffer
      const paramsBuf = device.createBuffer({
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(paramsBuf, 0, paramsData);

      // Pass 0: build sp_to_lin
      const pass0Shader = getOrCreateShader(device, "conv_pass0", makePass0Shader());
      const pass0Pipeline = getOrCreatePipeline(device, "conv_pass0", pass0Shader);
      {
        const bg = device.createBindGroup({
          layout: pass0Pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: x.coords.gpuBuffer! } },
            { binding: 2, resource: { buffer: spBuf } },
          ],
        });
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pass0Pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(N / 256), 1, 1);
        pass.end();
        device.queue.submit([enc.finish()]);
      }

      // Pass 1: fill neighbor map
      const pass1Shader = getOrCreateShader(device, "conv_pass1", makePass1Shader());
      const pass1Pipeline = getOrCreatePipeline(device, "conv_pass1", pass1Shader);
      {
        const bg = device.createBindGroup({
          layout: pass1Pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: paramsBuf } },
            { binding: 1, resource: { buffer: x.coords.gpuBuffer! } },
            { binding: 2, resource: { buffer: spBuf } },
            { binding: 3, resource: { buffer: neighborMapBuf } },
          ],
        });
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pass1Pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(N / 256), 1, 1);
        pass.end();
        device.queue.submit([enc.finish()]);
      }

      // Cache
      x.registerSpatialCache(cacheKey, {
        buffer: neighborMapBuf,
        size: nmSize,
      });

      paramsBuf.destroy();
      spBuf.destroy();
    }

    // ── Pass 2: Fused gather-GEMM ──
    const T = context.supportsFP16 && x.dtype === "float16"
      ? "f16"
      : "f32";
    const enableF16 = context.supportsFP16 && T === "f16";
    const pass2Key = `conv_pass2_${T}`;
    const pass2Shader = getOrCreateShader(device, pass2Key,
      context.precisionPrelude() + makePass2Shader(T, enableF16));
    const pass2Pipeline = getOrCreatePipeline(device, pass2Key, pass2Shader);

    const paramsBuf2 = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(paramsBuf2, 0, paramsData);

    const outSize = N * Co * (x.dtype === "float16" ? 2 : 4);
    const outBuf = device.createBuffer({
      size: outSize,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Create zero bias if none — temp buffer must be destroyed after submit
    let biasBuf: GPUBuffer;
    let biasBufOwned = false;
    if (this.bias?.gpuBuffer) {
      biasBuf = this.bias.gpuBuffer;
    } else {
      biasBuf = createZeroBias(device, Co, T);
      biasBufOwned = true;
    }

    const bg2 = device.createBindGroup({
      layout: pass2Pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: paramsBuf2 } },
        { binding: 1, resource: { buffer: neighborMapBuf } },
        { binding: 2, resource: { buffer: x.feats.gpuBuffer! } },
        { binding: 3, resource: { buffer: this.weight.gpuBuffer! } },
        { binding: 4, resource: { buffer: biasBuf } },
        { binding: 5, resource: { buffer: outBuf } },
      ],
    });

    const tilesN = Math.ceil(N / 16);
    const tilesCo = Math.ceil(Co / 16);
    {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pass2Pipeline);
      pass.setBindGroup(0, bg2);
      pass.dispatchWorkgroups(tilesN, tilesCo, 1);
      pass.end();
      device.queue.submit([enc.finish()]);
    }

    const outTensor = Tensor.fromGpuBuffer(
      outBuf,
      [N, Co],
      x.dtype,
      context,
      `conv_out_${N}x${Co}`,
    );

    paramsBuf2.destroy();
    if (biasBufOwned) biasBuf.destroy();

    return x.replace(outTensor);
  }
}

/** Create zero-filled bias buffer for conv layers without bias */
function createZeroBias(
  device: GPUDevice,
  Co: number,
  T: string,
): GPUBuffer {
  const size = Co * 4; // f32 always for bias (or f16)
  const buf = device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const data = new Float32Array(buf.getMappedRange());
  data.fill(0);
  buf.unmap();
  return buf;
}
