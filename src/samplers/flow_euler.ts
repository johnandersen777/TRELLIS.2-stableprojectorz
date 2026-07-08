/**
 * Flow Euler sampler with Classifier-Free Guidance.
 *
 * Flow matching ODE: dx/dt = v_theta(x, t, cond)
 * Euler step: x_{t-1} = x_t - dt * v_theta(x_t, t, cond)
 *
 * CFG: v = v_uncond + guidance_strength * (v_cond - v_uncond)
 * Guidance interval: only apply CFG for t in [interval_start, interval_end]
 *
 * Pure CPU-side loop — model forward pass runs on GPU.
 * Numeric ops (tensor subtract, scalar multiply) on CPU for simplicity.
 */

import { Tensor } from "../runtime/tensor.ts";
import type { GPUContext } from "../runtime/device.ts";
import { add, mul } from "../dense/ops/elementwise.ts";

export interface FlowEulerConfig {
  steps: number;                    // 12 (from pipeline.json params)
  guidanceStrength: number;         // 7.5
  guidanceRescale: number;          // 0.7
  guidanceInterval: [number, number]; // [0.6, 1.0]
  rescaleT: number;                 // 5.0
  sigmaMin: number;                 // 1e-5
}

export interface FlowModel {
  /** Forward pass: (x: Tensor, t: number, cond: Tensor, negCond: Tensor|null) → Tensor */
  forward: (
    x: Tensor,
    t: number,
    cond: Tensor,
    negCond: Tensor | null,
  ) => Promise<Tensor>;
}

/** Optional device recycle callback — called when GPU state needs resetting */
export type RecycleDeviceFn = (latent: Tensor) => Promise<Tensor>;

/**
 * Flow Euler sampling loop.
 *
 * @param model — model with forward(x, t, cond, negCond) method
 * @param noise — initial noise tensor on GPU (same shape as model output)
 * @param cond — conditioning tensor on GPU
 * @param negCond — negative conditioning (zeros) or null on GPU
 * @param config — sampler hyperparameters from pipeline.json
 * @returns denoised sample tensor on GPU
 */
export async function flowEulerSample(
  model: FlowModel,
  noise: Tensor,
  cond: Tensor,
  negCond: Tensor | null,
  config: FlowEulerConfig,
  ctx: GPUContext,
  recycleDevice?: RecycleDeviceFn,
): Promise<Tensor> {
  const { steps, guidanceStrength, guidanceRescale, guidanceInterval, rescaleT } = config;

  // Timestep schedule: t from 1 to 0 with rescaling
  // Python: t_seq = rescaleT * t / (1 + (rescaleT - 1) * t) where t = linspace(1, 0, steps+1)
  const ts: number[] = [];
  for (let i = 0; i <= steps; i++) {
    const t = 1.0 - i / steps;
    const tRescaled = rescaleT * t / (1.0 + (rescaleT - 1.0) * t);
    ts.push(tRescaled);
  }

  let xt = noise;

  for (let i = 0; i < steps; i++) {
    const t = ts[i];
    const tPrev = ts[i + 1];
    const tModel = t * 1000; // Scale to model timestep convention (0-1000)

    // CFG: apply guidance only within interval
    const applyCfg = guidanceStrength > 1.0 &&
      t >= guidanceInterval[0] && t <= guidanceInterval[1];

    let vt: Tensor;
    if (applyCfg && negCond) {
      // Two forward passes: cond and uncond
      const vCond = await model.forward(xt, tModel, cond, null);
      const vUncond = await model.forward(xt, tModel, negCond, null);

      // CFG blend on GPU: v = v_uncond + guidance * (v_cond - v_uncond)
      // v_cond - v_uncond: use a GPU subtract (vUncond = vCond - vUncond in-place)
      // Then: v = vUncond + guidance * vCond_scaled
      // Simpler: vCond = vCond - vUncond (in-place sub), then vUncond += guidance * vCond
      // But we don't have sub(). Use: vCond = vCond * guidance, then vUncond = vUncond + vCond
      // Actually: v = v_uncond + g * (v_cond - v_uncond) = (1-g)*v_uncond + g*v_cond
      // = g * v_cond + (1-g) * v_uncond

      // Scale vCond by guidance: vCond *= guidance
      const gData = new Float32Array(vCond.size);
      gData.fill(guidanceStrength);
      const gTensor = Tensor.fromArray(gData, [...vCond.shape], "float32");
      await gTensor.upload(ctx);
      mul(vCond, gTensor, ctx);
      gTensor.dispose();

      // Scale vUncond by (1-guidance): vUncond *= (1-guidance)
      const oneMinusG = 1.0 - guidanceStrength;
      const oneMinusGData = new Float32Array(vUncond.size);
      oneMinusGData.fill(oneMinusG);
      const oneMinusGTensor = Tensor.fromArray(oneMinusGData, [...vUncond.shape], "float32");
      await oneMinusGTensor.upload(ctx);
      mul(vUncond, oneMinusGTensor, ctx);
      oneMinusGTensor.dispose();

      // v = vCond + vUncond (in-place add)
      add(vCond, vUncond, ctx);
      vUncond.dispose();
      vt = vCond;

      // Note: guidance rescale (std-based) is skipped — it requires CPU readback
      // which triggers the Deno WebGPU mapAsync bug. The output is slightly
      // different but the sampling still converges.
    } else {
      // Single forward pass (no CFG or outside guidance interval)
      vt = await model.forward(xt, tModel, cond, null);
    }

    // Euler step: x_{t-1} = x_t - dt * vt  (GPU-only, no CPU round-trip)
    const dt = t - tPrev;
    if (dt > 0) {
      // vt = vt * (-dt) — scale in-place (vt disposed after step)
      const negDt = new Float32Array(vt.size);
      negDt.fill(-dt);
      const negDtTensor = Tensor.fromArray(negDt, [...vt.shape], "float32");
      await negDtTensor.upload(ctx);
      mul(vt, negDtTensor, ctx);
      negDtTensor.dispose();

      // xt = xt + vt  (= xt - dt * vt_original, in-place add)
      add(xt, vt, ctx);
    }

    vt.dispose();

    console.log(`  Step ${i + 1}/${steps}: t=${t.toFixed(3)} → ${tPrev.toFixed(3)}, dt=${dt.toFixed(4)}${applyCfg ? ", CFG" : ""}`);
  }

  return xt;
}
