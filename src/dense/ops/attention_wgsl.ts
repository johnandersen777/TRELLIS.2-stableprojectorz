/**
 * Scaled dot-product attention WGSL shader generator.
 *
 * Design: one workgroup per query position. Thread 0 coordinates;
 * threads 1-255 load K/V tiles into shared memory and compute dot products.
 * Thread 0 handles the sequential softmax logic (find max, sum, rescale).
 * Workers accumulate O += P * V in parallel across D-dimension slices.
 *
 * Dispatch: (B * H * S_q, 1, 1)
 * Workgroup size: min(256, D + 1) — adjusted so one thread per D dimension
 *
 * Bindings:
 *   @binding(0): Q [B * S_q * H * D]
 *   @binding(1): K [B * S_kv * H * D]
 *   @binding(2): V [B * S_kv * H * D]
 *   @binding(3): out [B * S_q * H * D]
 *   @binding(4): params: S_q, S_kv, H, scale_f32_bits, D
 */

export interface AttentionShaderConfig {
  headDim: number;
  scalarType: "f32" | "f16";
  tileK: number;       // rows per tile (max 256 = one per worker thread)
  workgroupSize: number; // typically 256
}

export function makeAttentionShader(config: AttentionShaderConfig): string {
  const D = config.headDim;
  const TK = config.tileK;
  const T = config.scalarType;
  const WS = config.workgroupSize;

  return `
const HEAD_DIM = ${D}u;
const TILE_K = ${TK}u;
const LOG_DIM = ${Math.ceil(Math.log2(D))}u; // for barrier checks

struct Params {
  S_q: u32,
  S_kv: u32,
  num_heads: u32,
  scale_bits: u32,
  workgroup_offset: u32,
}

@group(0) @binding(0) var<storage, read> Q: array<${T}>;
@group(0) @binding(1) var<storage, read> K: array<${T}>;
@group(0) @binding(2) var<storage, read> V: array<${T}>;
@group(0) @binding(3) var<storage, read_write> out: array<${T}>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> kv_sh: array<f32, TILE_K * HEAD_DIM>; // reused for K then V per tile
var<workgroup> dots: array<f32, TILE_K>;
var<workgroup> m_val: f32;     // running max (written by thread 0)
var<workgroup> l_sum: f32;     // running sum exp (written by thread 0)

fn Q_off(b: u32, h: u32, i: u32, d: u32) -> u32 {
  return ((b * params.S_q + i) * params.num_heads + h) * HEAD_DIM + d;
}
fn KV_off(b: u32, h: u32, j: u32, d: u32) -> u32 {
  return ((b * params.S_kv + j) * params.num_heads + h) * HEAD_DIM + d;
}

@compute @workgroup_size(${WS})
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let scale = bitcast<f32>(params.scale_bits);

  // Decode global workgroup index → (batch, head, query_pos)
  // Add workgroup_offset so chunked dispatches map to correct query indices.
  let gid = wgid.x + params.workgroup_offset;
  let total = params.S_q * params.num_heads;
  let b = gid / total;
  let rest = gid % total;
  let h = rest / params.S_q;
  let q_pos = rest % params.S_q;

  // ── Load Q vector for this query ──
  var Q_vec: array<f32, ${D}>;
  for (var d = lid.x; d < HEAD_DIM; d += ${WS}u) {
    Q_vec[d] = f32(Q[Q_off(b, h, q_pos, d)]);
  }
  workgroupBarrier();

  // ── Init accumulators ──
  var O_vec: array<f32, ${D}>;
  for (var d = 0u; d < HEAD_DIM; d++) {
    O_vec[d] = 0.0;
  }

  // Thread 0 initializes shared state
  if (lid.x == 0u) {
    m_val = -1e30;
    l_sum = 0.0;
  }
  workgroupBarrier();

  // ── Pass 1: find global max ──
  let num_tiles = (params.S_kv + TILE_K - 1u) / TILE_K;

  for (var tile = 0u; tile < num_tiles; tile++) {
    let k_start = tile * TILE_K;
    let k_end = min(k_start + TILE_K, params.S_kv);
    let num_k = k_end - k_start;

    // Workers load K tile
    for (var idx = lid.x; idx < num_k * HEAD_DIM; idx += ${WS}u) {
      let kj = idx / HEAD_DIM;
      let d = idx % HEAD_DIM;
      kv_sh[kj * HEAD_DIM + d] = f32(K[KV_off(b, h, k_start + kj, d)]);
    }
    workgroupBarrier();

    // Compute dot products (one per worker, if within num_k)
    if (lid.x < num_k) {
      var dot: f32 = 0.0;
      for (var d = 0u; d < HEAD_DIM; d++) {
        dot += Q_vec[d] * kv_sh[lid.x * HEAD_DIM + d];
      }
      dots[lid.x] = dot * scale;
    }
    workgroupBarrier();

    // Thread 0: find max over this tile
    if (lid.x == 0u) {
      for (var kj = 0u; kj < num_k; kj++) {
        m_val = max(m_val, dots[kj]);
      }
    }
    workgroupBarrier();
  }

  // ── Pass 2: compute softmax + accumulate output ──
  // Uses kv_sh for K, then reloads with V to save workgroup memory.
  let final_max = m_val; // all threads can read via shared memory
  if (lid.x == 0u) { l_sum = 0.0; }
  workgroupBarrier();

  for (var tile = 0u; tile < num_tiles; tile++) {
    let k_start = tile * TILE_K;
    let k_end = min(k_start + TILE_K, params.S_kv);
    let num_k = k_end - k_start;

    // Step 2a: Load K tile into kv_sh, compute P = exp(Q·K * scale - max)
    for (var idx = lid.x; idx < num_k * HEAD_DIM; idx += ${WS}u) {
      let kj = idx / HEAD_DIM;
      let d = idx % HEAD_DIM;
      kv_sh[kj * HEAD_DIM + d] = f32(K[KV_off(b, h, k_start + kj, d)]);
    }
    workgroupBarrier();

    if (lid.x < num_k) {
      var dot: f32 = 0.0;
      for (var d = 0u; d < HEAD_DIM; d++) {
        dot += Q_vec[d] * kv_sh[lid.x * HEAD_DIM + d];
      }
      dots[lid.x] = exp(dot * scale - final_max);
    }
    workgroupBarrier();

    // Thread 0: accumulate l_sum
    if (lid.x == 0u) {
      var tile_sum: f32 = 0.0;
      for (var kj = 0u; kj < num_k; kj++) {
        tile_sum += dots[kj];
      }
      l_sum += tile_sum;
    }
    workgroupBarrier();

    // Step 2b: Reload kv_sh with V tile, accumulate O += P_j * V_j
    for (var idx = lid.x; idx < num_k * HEAD_DIM; idx += ${WS}u) {
      let kj = idx / HEAD_DIM;
      let d = idx % HEAD_DIM;
      kv_sh[kj * HEAD_DIM + d] = f32(V[KV_off(b, h, k_start + kj, d)]);
    }
    workgroupBarrier();

    // Workers: accumulate O += P_j * V_j for their D-slice
    for (var d = lid.x; d < HEAD_DIM; d += ${WS}u) {
      var acc: f32 = 0.0;
      for (var kj = 0u; kj < num_k; kj++) {
        acc += dots[kj] * kv_sh[kj * HEAD_DIM + d];
      }
      O_vec[d] += acc;
    }
    workgroupBarrier();
  }

  // ── Normalize and write output ──
  let inv_l = 1.0 / max(l_sum, 1e-10);
  for (var d = lid.x; d < HEAD_DIM; d += ${WS}u) {
    out[Q_off(b, h, q_pos, d)] = ${T}(O_vec[d] * inv_l);
  }
}
`;
}

export function attentionShaderKey(config: AttentionShaderConfig): string {
  return `sdpa_v3_D${config.headDim}_TK${config.tileK}_${config.scalarType}`;
}
