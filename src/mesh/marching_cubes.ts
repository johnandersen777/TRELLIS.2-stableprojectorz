/**
 * Marching Cubes — Lewiner algorithm with sub-case resolution.
 *
 * Uses Lewiner's MC33 tables from scikit-image to produce topologically correct,
 * manifold triangular meshes. Matches Python's skimage.measure.marching_cubes
 * default output (method='lewiner').
 *
 * Edge dedup: global hash map with canonical edge keys.
 */

import {
  CASES, CASESCLASSIC,
  TILING1, TILING2, TILING3_1, TILING3_2,
  TILING4_1, TILING4_2, TILING5,
  TILING6_1_1, TILING6_1_2, TILING6_2,
  TILING7_1, TILING7_2, TILING7_3, TILING7_4_1, TILING7_4_2,
  TILING8, TILING9,
  TILING10_1_1, TILING10_1_1_, TILING10_1_2, TILING10_2, TILING10_2_,
  TILING11,
  TILING12_1_1, TILING12_1_1_, TILING12_1_2, TILING12_2, TILING12_2_,
  TILING13_1, TILING13_1_, TILING13_2, TILING13_2_,
  TILING13_3, TILING13_3_, TILING13_4,
  TILING13_5_1, TILING13_5_2,
  TILING14,
  SUBCONFIG13,
  TEST_VALUES,
} from "./lewiner_tables.ts";

const {
  TEST3, TEST4, TEST6, TEST7, TEST10, TEST12, TEST13,
} = TEST_VALUES;

// ── Standard MC constants ──────────────────────────────────

const EDGE_TABLE = new Uint16Array([
  0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
  0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
  0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
  0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
  0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
  0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
  0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
  0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
  0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
  0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
  0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
  0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
  0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
  0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
  0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
  0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
  0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
  0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
  0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
  0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
  0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
  0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
  0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
  0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
  0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
  0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
  0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
  0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
  0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
  0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
  0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
  0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
]);

const EDGE_CORNERS: [number, number][] = [
  [0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7],
];

const CORNER_OFFSETS: [number, number, number][] = [
  [0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1],
];

// ── Lewiner helpers ────────────────────────────────────────

/** Extract triangles from a tiling table at a given config offset.
 *  Each entry has `triCount * 3` edge indices (0-11 = edge, 12 = interior).
 *  Negative values terminate the list. */
function decodeTiling(table: Int8Array, config: number, count: number): number[] {
  const stride = count * 3;
  const off = config * stride;
  const result: number[] = [];
  for (let i = 0; i < stride; i++) {
    const v = table[off + i];
    if (v < 0) break;
    result.push(v);
  }
  return result;
}

/** Decode tiling with subconfig index (for tables with shape [N, M, K]). */
function decodeTiling2(table: Int8Array, config: number, subconfig: number, variants: number, count: number): number[] {
  const stride = variants * count * 3;
  const off = config * stride + subconfig * count * 3;
  const result: number[] = [];
  for (let i = 0; i < count * 3; i++) {
    const v = table[off + i];
    if (v < 0) break;
    result.push(v);
  }
  return result;
}

/** Face test: resolve ambiguous face configurations.
 *  face value comes from TEST tables. */
function testFace(corners: number[], face: number, level: number): boolean {
  const absFace = Math.abs(face);
  let a: number, b: number, c: number, d: number;
  // Face corner selection
  switch (absFace) {
    case 1: a = corners[0]; b = corners[4]; c = corners[5]; d = corners[1]; break; // front y=0
    case 2: a = corners[1]; b = corners[5]; c = corners[6]; d = corners[2]; break; // right x=1
    case 3: a = corners[2]; b = corners[6]; c = corners[7]; d = corners[3]; break; // back y=1
    case 4: a = corners[3]; b = corners[7]; c = corners[4]; d = corners[0]; break; // left x=0
    case 5: a = corners[0]; b = corners[3]; c = corners[2]; d = corners[1]; break; // bottom z=0
    case 6: a = corners[4]; b = corners[7]; c = corners[6]; d = corners[5]; break; // top z=1
    default: return face >= 0;
  }
  a -= level; b -= level; c -= level; d -= level;
  const ac_bd = a * c - b * d;
  if (Math.abs(ac_bd) < 1e-10) return face >= 0;
  return face * a * ac_bd >= 0;
}

/** Resolve Lewiner triangles for a given cube index and corner values. */
function resolveTriangles(cubeIdx: number, corners: number[], level: number): number[] {
  const caseType = CASES[cubeIdx * 2];
  if (caseType === 0) return [];
  const config = CASES[cubeIdx * 2 + 1];

  switch (caseType) {
    case 1: return decodeTiling(TILING1, config, 1);
    case 2: return decodeTiling(TILING2, config, 2);
    case 3: {
      const tf = testFace(corners, TEST3[config], level);
      return tf ? decodeTiling(TILING3_2, config, 4) : decodeTiling(TILING3_1, config, 2);
    }
    case 4: {
      const tf = testFace(corners, TEST4[config], level);
      return tf ? decodeTiling(TILING4_1, config, 2) : decodeTiling(TILING4_2, config, 6);
    }
    case 5: return decodeTiling(TILING5, config, 3);
    case 6: {
      const t0 = TEST6[config * 3];
      const t1 = TEST6[config * 3 + 1];
      if (testFace(corners, t0, level)) return decodeTiling(TILING6_2, config, 5);
      if (testFace(corners, t1, level)) return decodeTiling(TILING6_1_1, config, 3);
      return decodeTiling(TILING6_1_2, config, 9);
    }
    case 7: {
      const t0 = TEST7[config * 5];
      const t1 = TEST7[config * 5 + 1];
      const t2 = TEST7[config * 5 + 2];
      let sub = 0;
      if (testFace(corners, t0, level)) sub |= 1;
      if (testFace(corners, t1, level)) sub |= 2;
      if (testFace(corners, t2, level)) sub |= 4;
      switch (sub) {
        case 0: return decodeTiling(TILING7_1, config, 3);
        case 1: case 2: case 4: return decodeTiling2(TILING7_2, config, sub - 1, 3, 5);
        case 3: case 5: case 6: return decodeTiling2(TILING7_3, config, sub - 3, 3, 9);
        case 7: {
          const t3 = TEST7[config * 5 + 3];
          return testFace(corners, t3, level)
            ? decodeTiling(TILING7_4_1, config, 5)
            : decodeTiling(TILING7_4_2, config, 9);
        }
      }
      return [];
    }
    case 8: return decodeTiling(TILING8, config, 2);
    case 9: return decodeTiling(TILING9, config, 4);
    case 10: {
      const t0 = TEST10[config * 3];
      const t1 = TEST10[config * 3 + 1];
      const tf0 = testFace(corners, t0, level);
      const tf1 = testFace(corners, t1, level);
      if (tf0 && tf1) return decodeTiling(TILING10_1_1_, config, 4);
      if (tf0 && !tf1) return decodeTiling(TILING10_2, config, 8);
      if (!tf0 && tf1) return decodeTiling(TILING10_2_, config, 8);
      return decodeTiling(TILING10_1_2, config, 8);
    }
    case 11: return decodeTiling(TILING11, config, 4);
    case 12: {
      const t0 = TEST12[config * 4];
      const t1 = TEST12[config * 4 + 1];
      const tf0 = testFace(corners, t0, level);
      const tf1 = testFace(corners, t1, level);
      if (tf0 && tf1) return decodeTiling(TILING12_1_1_, config, 4);
      if (tf0 && !tf1) return decodeTiling(TILING12_2, config, 8);
      if (!tf0 && tf1) return decodeTiling(TILING12_2_, config, 8);
      return decodeTiling(TILING12_1_2, config, 8);
    }
    case 13: {
      // 6 face tests build subconfig
      let sub = 0;
      for (let i = 0; i < 6; i++) {
        const t = TEST13[config * 7 + i];
        if (testFace(corners, t, level)) sub |= (1 << i);
      }
      sub = SUBCONFIG13[sub]; // compress 0-63 → 0-45
      if (sub === 0) return decodeTiling2(TILING13_1, config, 0, 2, 4);
      if (sub <= 6) return decodeTiling2(TILING13_2, config, sub - 1, 6, 6);
      if (sub <= 18) return decodeTiling2(TILING13_3, config, sub - 7, 12, 10);
      if (sub <= 22) return decodeTiling2(TILING13_4, config, sub - 19, 4, 12);
      if (sub <= 26) {
        const t6 = TEST13[config * 7 + 6];
        return testFace(corners, t6, level)
          ? decodeTiling2(TILING13_5_1, config, sub - 23, 4, 6)
          : decodeTiling2(TILING13_5_2, config, sub - 23, 4, 10);
      }
      if (sub <= 38) return decodeTiling2(TILING13_3_, config, sub - 27, 12, 10);
      if (sub <= 44) return decodeTiling2(TILING13_2_, config, sub - 39, 6, 6);
      return decodeTiling2(TILING13_1_, config, 0, 2, 4);
    }
    case 14: return decodeTiling(TILING14, config, 4);
    default: return [];
  }
}

// ── Public API ──────────────────────────────────────────────

export interface MarchingCubesResult {
  vertices: Float32Array;
  faces: Uint32Array;
}

export function marchingCubes(
  grid: Float32Array,
  nx: number,
  ny: number,
  nz: number,
  level = 0.5,
): MarchingCubesResult {
  const vertices: number[] = [];
  const faces: number[] = [];

  // Global dedup: canonical edge key → vertex index
  const edgeMap = new Map<number, number>();
  const nxy = nx * ny;

  function edgeKey(x1: number, y1: number, z1: number, e: number): number {
    const dir = Math.floor(e / 4);
    return (x1 & 0x3FF) | ((y1 & 0x3FF) << 10) | ((z1 & 0x3FF) << 20) | (dir << 30);
  }

  for (let z = 0; z < nz - 1; z++) {
    for (let y = 0; y < ny - 1; y++) {
      for (let x = 0; x < nx - 1; x++) {
        // Read 8 corners
        const i000 = x + y * nx + z * nxy;
        const corners = [
          grid[i000], grid[i000 + 1], grid[i000 + 1 + nx], grid[i000 + nx],
          grid[i000 + nxy], grid[i000 + 1 + nxy], grid[i000 + 1 + nx + nxy], grid[i000 + nx + nxy],
        ];

        // Compute cube index
        let cubeIdx = 0;
        for (let c = 0; c < 8; c++) {
          if (corners[c] < level) cubeIdx |= (1 << c);
        }

        const edgeFlags = EDGE_TABLE[cubeIdx];
        if (edgeFlags === 0) continue;

        // Compute edge vertex positions
        const edgePositions: [number, number, number][] = new Array(12);
        for (let e = 0; e < 12; e++) {
          if (!(edgeFlags & (1 << e))) { edgePositions[e] = [0, 0, 0]; continue; }
          const [c1, c2] = EDGE_CORNERS[e];
          const [ox1, oy1, oz1] = CORNER_OFFSETS[c1];
          const [ox2, oy2, oz2] = CORNER_OFFSETS[c2];
          const v1 = corners[c1], v2 = corners[c2];
          const t = (level - v1) / (v2 - v1);
          edgePositions[e] = [
            x + ox1 + t * (ox2 - ox1),
            y + oy1 + t * (oy2 - oy1),
            z + oz1 + t * (oz2 - oz1),
          ];
        }

        // Map edges to vertices (dedup)
        const edgeToVert = new Map<number, number>();
        for (let e = 0; e < 12; e++) {
          if (!(edgeFlags & (1 << e))) continue;

          const [c1, c2] = EDGE_CORNERS[e];
          const [ox1, oy1, oz1] = CORNER_OFFSETS[c1];
          const [ox2, oy2, oz2] = CORNER_OFFSETS[c2];
          const ex = Math.min(x + ox1, x + ox2);
          const ey = Math.min(y + oy1, y + oy2);
          const ez = Math.min(z + oz1, z + oz2);
          const key = edgeKey(ex, ey, ez, e);

          const existing = edgeMap.get(key);
          if (existing !== undefined) {
            edgeToVert.set(e, existing);
          } else {
            const idx = vertices.length / 3;
            const [px, py, pz] = edgePositions[e];
            vertices.push(px, py, pz);
            edgeToVert.set(e, idx);
            edgeMap.set(key, idx);
          }
        }

        // Resolve triangles using Lewiner sub-case logic
        const triIndices = resolveTriangles(cubeIdx, corners, level);

        // Emit faces
        for (let t = 0; t < triIndices.length; t += 3) {
          const a = edgeToVert.get(triIndices[t]);
          const b = edgeToVert.get(triIndices[t + 1]);
          const c = edgeToVert.get(triIndices[t + 2]);
          if (a !== undefined && b !== undefined && c !== undefined) {
            faces.push(a, b, c);
          }
        }
      }
    }
  }

  return {
    vertices: new Float32Array(vertices),
    faces: new Uint32Array(faces),
  };
}
