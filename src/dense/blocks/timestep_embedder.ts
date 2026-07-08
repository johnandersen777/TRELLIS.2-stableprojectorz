/**
 * Timestep embedder: sinusoidal embedding + 2-layer MLP with SiLU.
 *
 * Used by all flow models to embed diffusion timesteps.
 * t ∈ [0, 1000] → sin/cos embedding → Linear → SiLU → Linear → SiLU
 */

import { Tensor } from "../../runtime/tensor.ts";
import type { GPUContext } from "../../runtime/device.ts";

/** Compute sinusoidal timestep embedding for scalar timesteps.
 *  Returns CPU tensor with shape (B, dim). Caller must upload to GPU. */
export function sinusoidalEmbedding(
  t: Float32Array,   // (B,) timesteps in [0, 1000]
  dim: number,        // embedding dimension
): Float32Array {
  const B = t.length;
  const out = new Float32Array(B * dim);
  const half = dim / 2;

  // Frequencies: 1 / (10000^(2i/dim)) for i in [0, half)
  for (let b = 0; b < B; b++) {
    for (let i = 0; i < half; i++) {
      const freq = 1.0 / Math.pow(10000, (2 * i) / dim);
      const val = t[b] * freq;
      out[b * dim + 2 * i] = Math.sin(val);
      out[b * dim + 2 * i + 1] = Math.cos(val);
    }
  }
  return out;
}

/**
 * Full timestep embedder forward pass.
 *
 * t: (B,) in [0, 1000]
 * weights: { "0.weight": (dim, model_channels), "0.bias": (model_channels,),
 *             "2.weight": (model_channels, model_channels), "2.bias": (model_channels,) }
 *
 * Returns: (B, model_channels) GPU tensor
 */
export interface TimestepEmbedderWeights {
  mlp0Weight: Float32Array;   // (dim, model_channels)
  mlp0Bias: Float32Array;     // (model_channels,)
  mlp2Weight: Float32Array;   // (model_channels, model_channels)
  mlp2Bias: Float32Array;     // (model_channels,)
}

/** CPU-side timestep embedding (used before model forward pass).
 *  Small enough to compute on CPU. */
export function timestepEmbed(
  t: Float32Array,
  dim: number,
  weights: TimestepEmbedderWeights,
): Float32Array {
  const B = t.length;
  const modelChannels = weights.mlp0Bias.length;

  // 1. Sinusoidal embedding
  const sinEmb = sinusoidalEmbedding(t, dim); // (B, dim)

  // 2. Linear 0: (B, dim) @ (dim, model_channels) + bias
  const h0 = new Float32Array(B * modelChannels);
  for (let b = 0; b < B; b++) {
    for (let j = 0; j < modelChannels; j++) {
      let sum = weights.mlp0Bias[j];
      for (let i = 0; i < dim; i++) {
        sum += sinEmb[b * dim + i] * weights.mlp0Weight[j * dim + i];
      }
      // SiLU
      const sig = 1.0 / (1.0 + Math.exp(-sum));
      h0[b * modelChannels + j] = sum * sig;
    }
  }

  // 3. Linear 2: (B, model_channels) @ (model_channels, model_channels) + bias
  const h2 = new Float32Array(B * modelChannels);
  for (let b = 0; b < B; b++) {
    for (let j = 0; j < modelChannels; j++) {
      let sum = weights.mlp2Bias[j];
      for (let i = 0; i < modelChannels; i++) {
        sum += h0[b * modelChannels + i] * weights.mlp2Weight[j * modelChannels + i];
      }
      // SiLU
      const sig = 1.0 / (1.0 + Math.exp(-sum));
      h2[b * modelChannels + j] = sum * sig;
    }
  }

  return h2;
}
