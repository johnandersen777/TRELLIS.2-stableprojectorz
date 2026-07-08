/**
 * Tiled matrix multiplication WGSL shader generator.
 *
 * Design: 64x64 output tiles, TILE_K=32, 16x16 threads = 256/workgroup.
 * Shared memory: 8 KB (f16) or 16 KB (f32).
 *
 * Supports f16 (via "shader-f16" feature) with f32 fallback.
 * vec4 loads for 128-bit memory transactions (RDNA 4 native).
 *
 * Adapted from ggml-webgpu mul_mat_reg_tile.wgsl patterns.
 */

export interface MatmulShaderConfig {
  /** WGSL scalar type: "f32" or "f16" (or "min16float" alias) */
  scalarType: string;
  /** Workgroup shared memory alias type */
  wgType: string;
  /** vec4 variant */
  vec4Type: string;
  /** Include "enable f16" directive */
  enableF16: boolean;
}

export function matmulShaderKey(config: MatmulShaderConfig): string {
  const t = config.scalarType === "f16" || config.scalarType === "min16float"
    ? "f16"
    : "f32";
  return `matmul_${t}_t64x64_k32`;
}

export function matmulWorkgroupSize(): [number, number, number] {
  return [16, 16, 1]; // 256 threads
}

export function matmulDispatch(
  M: number,
  N: number,
  tileM = 64,
  tileN = 64,
): [number, number, number] {
  const wgM = Math.ceil(M / tileM);
  const wgN = Math.ceil(N / tileN);

  const maxPerDim = 65535;
  const totalTiles = wgM * wgN;

  if (totalTiles <= maxPerDim) {
    return [wgN, wgM, 1];
  }
  // 2D dispatch for very large matrices
  return [Math.min(wgN, maxPerDim), Math.ceil(totalTiles / maxPerDim), 1];
}

export function makeMatmulShader(config: MatmulShaderConfig): string {
  const T = config.scalarType;
  const V4 = config.vec4Type;
  const f16Directive = config.enableF16 ? "enable f16;\n" : "";

  return `${f16Directive}
// ── Tiled MatMul C = A × B ──
// TILE_M=64, TILE_N=64, TILE_K=32
// 16×16 threads = 256/workgroup

const TILE_M: u32 = 64u;
const TILE_N: u32 = 64u;
const TILE_K: u32 = 32u;
const THREADS_M: u32 = 16u;
const THREADS_N: u32 = 16u;

// Each thread computes a 4×4 block of output
const REG_M: u32 = 4u;
const REG_N: u32 = 4u;

// Workgroup shared memory tiles
var<workgroup> tile_A: array<${T}, TILE_M * TILE_K>;
var<workgroup> tile_B: array<${T}, TILE_K * TILE_N>;

// Buffer bindings
@group(0) @binding(0) var<storage, read> mat_A: array<${T}>;
@group(0) @binding(1) var<storage, read> mat_B: array<${T}>;
@group(0) @binding(2) var<storage, read_write> mat_C: array<${T}>;

// Dimensions: M, N, K
struct Dims {
  M: u32,
  N: u32,
  K: u32,
}
@group(0) @binding(3) var<uniform> dims: Dims;

@compute @workgroup_size(THREADS_M, THREADS_N, 1)
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(num_workgroups) num_wg: vec3<u32>,
) {
  let tile_row = wg_id.y * TILE_M;
  let tile_col = wg_id.x * TILE_N;
  let M = dims.M;
  let N = dims.N;
  let K = dims.K;

  // Thread-local row/col within the 64x64 tile
  let thread_row = lid.y * REG_M;
  let thread_col = lid.x * REG_N;

  // Register accumulators: 4×4 block
  var acc: array<array<${T}, REG_N>, REG_M>;
  for (var i = 0u; i < REG_M; i++) {
    for (var j = 0u; j < REG_N; j++) {
      acc[i][j] = ${T}(0.0);
    }
  }

  // Main K loop
  for (var k_base = 0u; k_base < K; k_base += TILE_K) {
    // ── Cooperative load tile_A [TILE_M × TILE_K] ──
    // Each thread loads 4 rows × 2 cols = 8 elements of tile_A
    for (var i = 0u; i < REG_M; i++) {
      let global_row = tile_row + thread_row + i;
      for (var j = 0u; j < 2u; j++) {
        let col = lid.x * 2u + j;
        if (col < TILE_K) {
          let k_idx = k_base + col;
          if (k_idx < K && global_row < M) {
            tile_A[(thread_row + i) * TILE_K + col] = mat_A[global_row * K + k_idx];
          } else {
            tile_A[(thread_row + i) * TILE_K + col] = ${T}(0.0);
          }
        }
      }
    }

    // ── Cooperative load tile_B [TILE_K × TILE_N] ──
    // Each thread loads 2 rows × 4 cols = 8 elements of tile_B
    for (var i = 0u; i < 2u; i++) {
      let row = lid.y * 2u + i;
      if (row < TILE_K) {
        let k_idx = k_base + row;
        for (var j = 0u; j < REG_N; j++) {
          let global_col = tile_col + thread_col + j;
          if (k_idx < K && global_col < N) {
            tile_B[row * TILE_N + (thread_col + j)] = mat_B[k_idx * N + global_col];
          } else {
            tile_B[row * TILE_N + (thread_col + j)] = ${T}(0.0);
          }
        }
      }
    }

    workgroupBarrier();

    // ── Compute over this K-tile ──
    for (var k = 0u; k < TILE_K; k++) {
      for (var i = 0u; i < REG_M; i++) {
        let a_val = tile_A[(thread_row + i) * TILE_K + k];
        for (var j = 0u; j < REG_N; j++) {
          acc[i][j] += a_val * tile_B[k * TILE_N + (thread_col + j)];
        }
      }
    }

    workgroupBarrier();
  }

  // ── Write results ──
  for (var i = 0u; i < REG_M; i++) {
    let global_row = tile_row + thread_row + i;
    if (global_row >= M) { break; }
    for (var j = 0u; j < REG_N; j++) {
      let global_col = tile_col + thread_col + j;
      if (global_col >= N) { break; }
      mat_C[global_row * N + global_col] = acc[i][j];
    }
  }
}
`;
}
