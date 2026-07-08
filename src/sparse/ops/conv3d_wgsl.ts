/**
 * WGSL shader generators for sparse 3D submanifold convolution.
 *
 * 3-pass design:
 *   Pass 0: Build spatial index (sp_to_lin)
 *   Pass 1: Fill neighbor map (neighbor_map[N×V])
 *   Pass 2: Fused gather-GEMM + bias
 *
 * Neighbor map is cached in SparseTensor.spatial_cache per (kernel, dilation, spatial_shape).
 * Pass 2 runs every conv layer; Pass 0+1 only when spatial config changes.
 */

/** Parameters shared across all conv passes */
export interface ConvParams {
  N: number; // number of active voxels
  Ci: number; // input channels
  Co: number; // output channels
  X: number; // spatial X
  Y: number;
  Z: number;
  Kd: number; // kernel depth
  Kh: number;
  Kw: number;
  dilD: number;
  dilH: number;
  dilW: number;
}

export function convParamsToUniform(params: ConvParams): Uint32Array {
  return new Uint32Array([
    params.N, params.Ci, params.Co,
    params.X, params.Y, params.Z,
    0, // padding
    params.Kd, params.Kh, params.Kw,
    params.dilD, params.dilH, params.dilW,
    0, 0, // padding to 16 bytes
  ]);
}

const V = 27; // 3×3×3 kernel

// ── Pass 0: Build spatial index ──────────────────────────

export function makePass0Shader(): string {
  return `
struct Params {
  N: u32, Ci: u32, Co: u32,
  X: u32, Y: u32, Z: u32,
  _pad0: u32,
  Kd: u32, Kh: u32, Kw: u32,
  dilD: u32, dilH: u32, dilW: u32,
  _pad1: u32, _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> coords: array<vec4<i32>>;  // (N, 4) [batch, x, y, z]
@group(0) @binding(2) var<storage, read_write> sp_to_lin: array<i32>;  // (X*Y*Z) init to -1

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.N) { return; }
  let c = coords[i];
  let x = u32(c.y);
  let y = u32(c.z);
  let z = u32(c.w);
  let lin = x * params.Y * params.Z + y * params.Z + z;
  sp_to_lin[lin] = i32(i);
}
`;
}

// ── Pass 1: Fill neighbor map ─────────────────────────────

export function makePass1Shader(): string {
  return `
struct Params {
  N: u32, Ci: u32, Co: u32,
  X: u32, Y: u32, Z: u32,
  _pad0: u32,
  Kd: u32, Kh: u32, Kw: u32,
  dilD: u32, dilH: u32, dilW: u32,
  _pad1: u32, _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> coords: array<vec4<i32>>;
@group(0) @binding(2) var<storage, read> sp_to_lin: array<i32>;
@group(0) @binding(3) var<storage, read_write> neighbor_map: array<i32>;

fn kernel_offset(v: u32) -> vec3<i32> {
  let hw = params.Kh * params.Kw;
  let kd = i32(v / hw);
  let kh = i32((v % hw) / params.Kw);
  let kw = i32(v % params.Kw);
  return vec3<i32>(
    (kd - i32(params.Kd / 2u)) * i32(params.dilD),
    (kh - i32(params.Kh / 2u)) * i32(params.dilH),
    (kw - i32(params.Kw / 2u)) * i32(params.dilW),
  );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x;
  if (n >= params.N) { return; }

  let c = coords[n];
  let max_x = i32(params.X);
  let max_y = i32(params.Y);
  let max_z = i32(params.Z);

  for (var v = 0u; v < 27u; v++) {
    let off = kernel_offset(v);
    let nx = c.y + off.x;
    let ny = c.z + off.y;
    let nz = c.w + off.z;

    var nbr: i32 = -1;
    if (nx >= 0 && nx < max_x && ny >= 0 && ny < max_y && nz >= 0 && nz < max_z) {
      let lin = u32(nx) * params.Y * params.Z + u32(ny) * params.Z + u32(nz);
      nbr = sp_to_lin[lin];
    }
    neighbor_map[n * 27u + v] = nbr;
  }
}
`;
}

// ── Pass 2: Fused gather-GEMM ─────────────────────────────

export function makePass2Shader(scalarType: string, enableF16: boolean): string {
  const T = scalarType;
  const f16Directive = enableF16 ? "enable f16;\n" : "";

  return `${f16Directive}
// TILE: 16 output voxels × 16 output channels = 256 threads
// CI_CHUNK: 64 input channels per shared-memory tile
const TILE_N: u32 = 16u;
const TILE_CO: u32 = 16u;
const CI_CHUNK: u32 = 64u;

struct Params {
  N: u32, Ci: u32, Co: u32,
  X: u32, Y: u32, Z: u32,
  _pad0: u32,
  Kd: u32, Kh: u32, Kw: u32,
  dilD: u32, dilH: u32, dilW: u32,
  _pad1: u32, _pad2: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> neighbor_map: array<i32>;  // [N × 27]
@group(0) @binding(2) var<storage, read> feats_in: array<${T}>;    // [N × Ci]
@group(0) @binding(3) var<storage, read> weight: array<${T}>;     // [Co × 27 × Ci]
@group(0) @binding(4) var<storage, read> bias: array<${T}>;       // [Co]
@group(0) @binding(5) var<storage, read_write> output: array<${T}>; // [N × Co]

// Shared memory: weight tile for CI_CHUNK input channels × TILE_CO output channels
var<workgroup> weight_tile: array<${T}, CI_CHUNK * TILE_CO>;
// Neighbor indices for TILE_N voxels
var<workgroup> neighbor_tile: array<i32, TILE_N>;

@compute @workgroup_size(TILE_CO, TILE_N, 1)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wgid: vec3<u32>,
) {
  let local_n = lid.y;   // voxel within tile (0..15)
  let local_co = lid.x;  // output channel within tile (0..15)
  let n_global = wgid.x * TILE_N + local_n;
  let co_global = wgid.y * TILE_CO + local_co;

  if (n_global >= params.N || co_global >= params.Co) { return; }

  // Accumulate in register
  var accum: ${T} = bias[co_global];

  // Outer loop over input channels in chunks
  for (var ci_base = 0u; ci_base < params.Ci; ci_base += CI_CHUNK) {
    let ci_remaining = min(CI_CHUNK, params.Ci - ci_base);

    // Loop over 27 kernel positions
    for (var v = 0u; v < 27u; v++) {
      // ── Cooperative load weight tile ──
      // TILE_CO × CI_CHUNK elements, 256 threads → each thread loads a few
      let wg_size = TILE_CO * TILE_N; // 256
      let wt_per_thread = (CI_CHUNK * TILE_CO + wg_size - 1u) / wg_size;
      let thread_flat = local_co * TILE_N + local_n;

      for (var t = 0u; t < wt_per_thread; t++) {
        let flat = thread_flat * wt_per_thread + t;
        if (flat < CI_CHUNK * TILE_CO) {
          let w_co = flat / CI_CHUNK;
          let w_ci = flat % CI_CHUNK;
          if (w_ci < ci_remaining && co_global + w_co < params.Co) {
            let w_idx = (co_global + w_co) * 27u * params.Ci + v * params.Ci + ci_base + w_ci;
            weight_tile[w_co * CI_CHUNK + w_ci] = weight[w_idx];
          } else {
            weight_tile[w_co * CI_CHUNK + w_ci] = ${T}(0.0);
          }
        }
      }

      // Only thread 0 per voxel loads neighbor
      if (local_co == 0u) {
        neighbor_tile[local_n] = neighbor_map[n_global * 27u + v];
      }

      workgroupBarrier();

      // ── Accumulate dot product ──
      let src_n = neighbor_tile[local_n];
      if (src_n >= 0) {
        let src_idx = u32(src_n);
        for (var ci_off = 0u; ci_off < ci_remaining; ci_off++) {
          let ci = ci_base + ci_off;
          let f = feats_in[src_idx * params.Ci + ci];
          let w = weight_tile[local_co * CI_CHUNK + ci_off];
          accum += f * w;
        }
      }

      workgroupBarrier();
    }
  }

  // Write result
  output[n_global * params.Co + co_global] = accum;
}
`;
}
