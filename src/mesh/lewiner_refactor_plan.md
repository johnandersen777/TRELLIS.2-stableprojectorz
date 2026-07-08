# Lewiner Marching Cubes Refactor Plan

## Overview

Port the Lewiner algorithm (used by scikit-image's `marching_cubes` default) into the existing TypeScript marching cubes implementation. The goal is to produce **byte-identical output** to the Python pipeline by replacing the 256-case Lorensen (Paul Bourke) lookup tables with Lewiner's sub-case resolution tables.

The existing code at `src/mesh/marching_cubes.ts` is a working standard MC implementation. The pipeline at `src/mesh/pipeline.ts` needs no changes -- it only calls `marchingCubes()` and processes the result.

---

## 1. What Stays the Same

### 1.1 Edge interpolation formula
The standard linear interpolation along each cube edge:
```typescript
const t = (level - v1) / (v2 - v1);
edgePositions[e] = [
  x + ox1 + t * (ox2 - ox1),
  y + oy1 + t * (oy2 - oy1),
  z + oz1 + t * (oz2 - oz1),
];
```
Lewiner uses the same edge numbering and the same interpolation formula. No change needed.

### 1.2 Corner value computation
Reading the 8 corner voxel values from the 3D grid:
```typescript
const i000 = x + y * nx + z * nxy;
const corners = [grid[i000], grid[i000+1], grid[i000+1+nx], grid[i000+nx],
                 grid[i000+nxy], grid[i000+1+nxy], grid[i000+1+nx+nxy], grid[i000+nx+nxy]];
```
No change needed.

### 1.3 Grid traversal order
```typescript
for (let z = 0; z < nz - 1; z++)
  for (let y = 0; y < ny - 1; y++)
    for (let x = 0; x < nx - 1; x++)
```
No change needed.

### 1.4 Vertex dedup via hash map (edge vertices)
The `edgeMap` that canonicalizes edge vertices by (min-corner-gcoords, edge-direction) stays. Edge vertices (indices 0-11 in the CASESCLASSIC/TILING tables) always map 1:1 to edge midpoints.

### 1.5 Face compaction + vertex merging (pipeline.ts)
The post-processing in `pipeline.ts` (degenerate face removal, transitive vertex merge, component filtering, GLB export) is algorithm-agnostic. No changes to `pipeline.ts`.

### 1.6 EDGE_CORNERS and CORNER_OFFSETS
Same standard MC edge numbering:
```typescript
EDGE_CORNERS: [0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]
CORNER_OFFSETS: [0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]
```
Both unchanged.

### 1.7 EDGE_TABLE (edge flag lookup)
The `EDGE_TABLE` (12-bit edge-intersection flags per cube index) is still needed. Lewiner's CASESCLASSIC table does NOT replace EDGE_TABLE; it replaces TRI_TABLE. The edge flags determine which edge midpoints need interpolation.

---

## 2. What Changes

### 2.1 TRI_TABLE replaced by CASESCLASSIC
The `Int8Array`-based `triTableCompact` and the decompressed `TRI_TABLE: number[][]` are replaced by `CASESCLASSIC: Uint8Array[4096]` (256 entries x 16 bytes).

**Format per entry:** 16 bytes, each byte is an edge index (0-11) or 255 (0xFF = terminator/skip). Same logical layout as the current compact tri table, but with an important difference: when subconfig resolves to use CASESCLASSIC directly, the edge indices here map 1:1 to the current TRI_TABLE entries.

**Verification step:** For all 256 cube indices where CASES[CubeIdx][1] == 0 (non-ambiguous), the values in CASESCLASSIC must produce the same triangles as the current TRI_TABLE. This must be validated during implementation.

### 2.2 New CASES table
Add `CASES: Uint8Array[512]` (256 entries x 2 bytes). Entry layout:
```
CASES[cubeIdx][0] = config   (which of the 15 base cases: 1-14, or 0 for empty)
CASES[cubeIdx][1] = subconfig (0 = classic/non-ambiguous, >0 = ambiguous sub-case)
```

**Config-to-tiling mapping:**
| config | Sub-cases | TILING tables to use |
|--------|-----------|---------------------|
| 0 | 0 | Empty (no triangles) |
| 1 | 0 | TILING1 (16 entries) |
| 2 | 0 | TILING2 (24 entries) |
| 3 | 24 | TILING3_1, TILING3_2 |
| 4 | 8 | TILING4_1, TILING4_2 |
| 5 | 0 | TILING5 (48 entries) |
| 6 | 48 | TILING6_1_1, TILING6_1_2, TILING6_2 |
| 7 | 16 | TILING7_1, TILING7_2, TILING7_3, TILING7_4_1, TILING7_4_2 |
| 8 | 0 | TILING8 (6 entries) |
| 9 | 0 | TILING9 (8 entries) |
| 10 | 6 | TILING10_1_1, TILING10_1_1_, TILING10_1_2, TILING10_2, TILING10_2_ |
| 11 | 0 | TILING11 (12 entries) |
| 12 | 24 | TILING12_1_1, TILING12_1_1_, TILING12_1_2, TILING12_2, TILING12_2_ |
| 13 | 2 | TILING13_1, TILING13_1_, TILING13_2, TILING13_2_, TILING13_3, TILING13_3_, TILING13_4, TILING13_5_1, TILING13_5_2 |
| 14 | 0 | TILING14 (12 entries) |

### 2.3 Sub-case resolution logic
When `subconfig != 0`, the cube is ambiguous and needs sub-case resolution. This is the core algorithmic change.

**Flow:**
```
cubeIdx → CASES[cubeIdx] → (config, subconfig)
                            ↓
                  subconfig == 0? ──YES──→ Use CASESCLASSIC directly (edge indices)
                            │
                           NO
                            ↓
                   ┌───────────────────────────────┐
                   │ Resolve sub-case:              │
                   │ 1. Compute test values from     │
                   │    corner values (see §3.2)     │
                   │ 2. Query TEST{config} table     │
                   │    to find matching sub-case    │
                   │ 3. Look up triangles from       │
                   │    the corresponding TILING     │
                   │    table entry                  │
                   │ 4. TILING entries use VERTEX    │
                   │    indices (0-11 = edge mid-    │
                   │    points, 12+ = interior)      │
                   └───────────────────────────────┘
```

**Why sub-configs exist:** In the standard 256-case table, ambiguous configurations (242 out of 256 cases) can produce multiple valid triangulations. The standard table picks one arbitrarily, which can cause holes between adjacent cubes. The Lewiner algorithm resolves this by using consistent test points to pick the same sub-case for neighboring cubes.

### 2.4 Interior vertex handling
TILING tables reference vertex indices in range [0, 14]:
- **0-11:** Edge midpoints (computed same as current code)
- **12:** Body center of the cube at isovalue
- **13, 14:** Additional interior points (face centers or edge centers for complex topologies)

For interior vertices (12+), positions must be computed from the 8 corner values. Interior vertex 12 is at the weighted center of the cube, using linear interpolation of the edge intersection points to find the isovalue surface crossing within the cube body.

The interior vertex computation is:
```
Position of vertex 12: computed as the centroid of the edge intersection points
  on all edges that cross the isosurface (weighted by their t values)
  
Position of vertex 13, 14: specific to sub-case configuration
```

### 2.5 Edge numbering (NO CHANGE)
The Lewiner algorithm uses the exact same edge numbering as standard MC (edges 0-11 matching EDGE_CORNERS). No remapping needed.

### 2.6 Dedup for interior vertices
Interior vertices (12+) need their own dedup per-cube (they are specific to each cube and cannot be shared between cubes). A local Map within each cube cell handles dedup:
```
interiorVerts: Map<interiorVertexIndex, globalVertIdx>
```

---

## 3. New Functions Needed

### 3.1 `getLewinerTriangles(corners, cubeIdx, config, subconfig, level): TriangleSpec[]`

```typescript
interface TriangleSpec {
  vertices: number[];  // 3 or more vertex indices (0-11 = edge, 12+ = interior)
}

function getLewinerTriangles(
  corners: Float32Array,    // 8 corner values
  cubeIdx: number,          // 8-bit mask
  config: number,           // from CASES[cubeIdx][0]
  subconfig: number,        // from CASES[cubeIdx][1]
  level: number,            // isovalue
): number[];                // flat array of vertex indices (groups of 3 = triangles)
```

**Algorithm:**
```
if subconfig === 0:
  // Non-ambiguous: use CASESCLASSIC directly
  decode CASESCLASSIC[cubeIdx] → return flat edge index array
  
else:
  // Ambiguous: resolve sub-case
  config → determine which TILING/TEST tables to use
  
  // Compute test values from corner signs
  testBits = computeTestBits(config, corners, level)
  
  // Find matching sub-case in TEST{config} table
  subCaseIdx = findSubCase(config, testBits)  // returns entry index in TILING table
  
  // Look up triangles from TILING{config} at entry subCaseIdx
  triangles = lookupTiling(config, subCaseIdx)  // returns vertex indices 0-14
  
  return triangles
```

### 3.2 `computeTestBits(config, corners, level): number`

```typescript
function computeTestBits(
  config: number,           // case number (3, 4, 6, 7, 10, 12, or 13)
  corners: Float32Array,    // 8 corner values
  level: number,            // isovalue
): number;                  // bitmask of test results
```

For each ambiguous case, specific edge midpoints are used as test positions. The sign of the interpolated value at each test position (above/below isovalue) determines the sub-case.

**Test edge positions per config:**
- Config 3: 3 test positions on specific edges
- Config 4: 3 test positions on specific edges
- Config 6: 3 test positions
- Config 7: 5 test positions
- Config 10: 3 test positions
- Config 12: 4 test positions
- Config 13: 7 test positions

The specific test edges for each config are determined from the CASESCLASSIC table. Each config's ambiguous edges are those where the classic table has multiple valid choices.

The test at each position is:
```
t = (level - v1) / (v2 - v1)
midValue = v1 + t * (v2 - v1)
testBit = midValue >= level ? 1 : 0
```

### 3.3 `findSubCase(config, testBits): number`

```typescript
function findSubCase(
  config: number,     // case number
  testBits: number,   // bitmask of test results
): number;            // entry index into the TILING table for this config
```

Looks up `testBits` in TEST{config} table to find the matching sub-case.

**TEST table format (decoded from base64):**
```typescript
// Each test table maps (testBits) → subCaseIndex
// TEST3: Uint8Array[24] → 24 entries, matching by bits
// TEST4: Uint8Array[8]
// TEST6: Uint8Array[48 * 3] → each entry 3 bytes
// TEST7: Uint8Array[16 * 5] → each entry 5 bytes
// TEST10: Uint8Array[6 * 3]
// TEST12: Uint8Array[24 * 4]
// TEST13: Uint8Array[2 * 7] + SUBCONFIG13: Uint8Array[64]
```

For config 13, the sub-case resolution is two-stage: TEST13 selects a group, then SUBCONFIG13 selects the exact sub-case.

### 3.4 `lookupTiling(config, entryIdx): number[]`

```typescript
function lookupTiling(
  config: number,     // case number 1-14
  entryIdx: number,   // which entry in the TILING table
): number[];          // flat vertex indices (each triple = one triangle)
```

Decodes the appropriate TILING{config} table at the given entry index. Returns the vertex indices (0-14). Each triple forms a triangle.

For complex configs (6, 7, 10, 12, 13), multiple TILING sub-tables exist (e.g., TILING6_1_1, TILING6_1_2, TILING6_2). The correct sub-table depends on additional sign tests from the cube corners. This is handled by `findSubCase` returning enough information to select both the table and entry.

### 3.5 `computeInteriorVertex(vertexIdx, x, y, z, corners, level): [number, number, number]`

```typescript
function computeInteriorVertex(
  vertexIdx: number,              // 12, 13, or 14
  cellX: number, cellY: number, cellZ: number,  // grid cell origin
  corners: Float32Array,          // 8 corner values
  level: number,                  // isovalue
): [number, number, number];      // world position
```

Computes the position of interior vertex 12/13/14 from the cube's corner values.

**Vertex 12 (body center):**
- Position: centroid of the isosurface intersection within the cube body
- Computed by finding the centroid of all edge intersection points in the cube
- Position formula: average of edge intersection positions weighted by their distance from the isovalue
- Fallback: (x+0.5, y+0.5, z+0.5) for degenerate cases

**Vertices 13, 14 (face/edge cusps):**
- Used only in specific complex sub-cases (config 13)
- Position is on a cube face or edge
- Computed from the specific edge intersection using the standard t formula

---

## 4. Data Structures

### 4.1 Lewiner Lookup Tables

All tables decoded from `lewiner_luts.py` (base64). Stored as flat typed arrays to minimize memory footprint.

```typescript
// ── Core tables ──

/** 256 entries × 16 bytes = 4096 bytes. Edge indices (0-11) or 0xFF. */
const CASESCLASSIC: Uint8Array;  // length 4096

/** 256 entries × 2 bytes = 512 bytes. [config, subconfig] per cube index. */
const CASES: Uint8Array;         // length 512

// ── Tiling tables (vertex indices: 0-11 = edge, 12-14 = interior) ──

/** 16 entries × 3 bytes. Config 1. */
const TILING1: Uint8Array;       // length 48

/** 24 entries × 6 bytes (2 triangles). Config 2. */
const TILING2: Uint8Array;       // length 144

/** 24 entries × 6 bytes. Config 3 primary triangles. */
const TILING3_1: Uint8Array;     // length 144

/** 24 entries × 12 bytes (4 triangles). Config 3 complementary triangles. */
const TILING3_2: Uint8Array;     // length 288

/** 8 entries × 6 bytes. Config 4 primary. */
const TILING4_1: Uint8Array;     // length 48

/** 8 entries × 18 bytes (6 triangles). Config 4 complementary. */
const TILING4_2: Uint8Array;     // length 144

/** 48 entries × 9 bytes (3 triangles). Config 5. */
const TILING5: Uint8Array;       // length 432

// ... all remaining TILING tables (see lewiner_luts.py for complete list)

/** 48 entries × 9 bytes. Config 6 primary variant 1. */
const TILING6_1_1: Uint8Array;

/** 48 entries × 27 bytes. Config 6 primary variant 2. */
const TILING6_1_2: Uint8Array;

/** 48 entries × 15 bytes. Config 6 secondary. */
const TILING6_2: Uint8Array;

// ... through TILING14

// ── Test tables (sub-case selection masks) ──

const TEST3: Uint8Array;     // 24 entries × 1 byte = 24 bytes
const TEST4: Uint8Array;     // 8 entries × 1 byte = 8 bytes
const TEST6: Uint8Array;     // 48 entries × 3 bytes = 144 bytes
const TEST7: Uint8Array;     // 16 entries × 5 bytes = 80 bytes
const TEST10: Uint8Array;    // 6 entries × 3 bytes = 18 bytes
const TEST12: Uint8Array;    // 24 entries × 4 bytes = 96 bytes
const TEST13: Uint8Array;    // 2 entries × 7 bytes = 14 bytes

/** 64 entries × 1 byte = 64 bytes. Additional sub-case selection for config 13. */
const SUBCONFIG13: Uint8Array;   // length 64
```

### 4.2 Dispatch Table

```typescript
/** Maps config number → info for sub-case resolution. */
interface TilingInfo {
  config: number;           // 1-14
  hasSubCases: boolean;     // true if config has ambiguous sub-cases
  testTable?: Uint8Array;   // TEST table for this config
  testEntrySize: number;    // bytes per entry in test table
  tilingTables: Uint8Array[];  // TILING table(s) for this config
  tilingEntrySizes: number[];  // bytes per entry per tiling table
}

const TILING_DISPATCH: TilingInfo[] = [
  { config: 0,  hasSubCases: false, testEntrySize: 0, tilingTables: [], tilingEntrySizes: [] },
  { config: 1,  hasSubCases: false, testEntrySize: 0, tilingTables: [TILING1], tilingEntrySizes: [3] },
  { config: 2,  hasSubCases: false, testEntrySize: 0, tilingTables: [TILING2], tilingEntrySizes: [6] },
  { config: 3,  hasSubCases: true,  testTable: TEST3, testEntrySize: 1,
    tilingTables: [TILING3_1, TILING3_2], tilingEntrySizes: [6, 12] },
  // ... etc
];
```

### 4.3 Base64 Decoding

The `lewiner_luts.py` file contains all tables as base64-encoded strings. A helper script or inline decoding function will convert them to TypeScript Uint8Array constants.

```typescript
function decodeBase64Bytes(b64: string): Uint8Array {
  const binary = atob(b64.replace(/\s/g, ''));
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}
```

**Recommendation:** Pre-decode all tables at module load time and store as module-level constants to avoid per-call decoding overhead.

---

## 5. Modified Main Loop

The main `marchingCubes()` function gets a new triangle lookup path. Here is the modified pseudocode with changes highlighted:

```typescript
export function marchingCubes(
  grid: Float32Array,
  nx: number,
  ny: number,
  nz: number,
  level = 0.5,
): MarchingCubesResult {
  const vertices: number[] = [];
  const faces: number[] = [];
  const edgeMap = new Map<number, number>();
  const nxy = nx * ny;

  // edgeKey() unchanged

  for (let z = 0; z < nz - 1; z++) {
    for (let y = 0; y < ny - 1; y++) {
      for (let x = 0; x < nx - 1; x++) {
        // 1. Compute corners (UNCHANGED)
        const corners = [...]; // same 8 corner reads

        // 2. Compute cube index (UNCHANGED)
        let cubeIdx = 0;
        for (let c = 0; c < 8; c++)
          if (corners[c] < level) cubeIdx |= (1 << c);

        // 3. Edge flags (UNCHANGED)
        const edgeFlags = EDGE_TABLE[cubeIdx];
        if (edgeFlags === 0) continue;

        // 4. Get case config from Lewiner tables (NEW)
        const config = CASES[cubeIdx * 2];
        const subconfig = CASES[cubeIdx * 2 + 1];

        // 5. Get triangles (CHANGED: replaces TRI_TABLE lookup)
        const triVertIndices = resolveTriangles(
          corners, cubeIdx, config, subconfig, level,
        );
        // triVertIndices: flat array of vertex indices [v0, v1, v2, v0, v1, v2, ...]
        // where each index is 0-11 (edge) or 12+ (interior)

        // 6. Compute edge positions (UNCHANGED for 0-11)
        const edgePositions = computeEdgePositions(corners, edgeFlags, x, y, z, level);

        // 7. Compute interior vertex positions (NEW, only if triVertIndices contains 12+)
        const interiorPositions: Map<number, [number, number, number]> = new Map();
        for (const vi of triVertIndices) {
          if (vi >= 12 && !interiorPositions.has(vi)) {
            interiorPositions.set(vi, computeInteriorVertex(vi, x, y, z, corners, level));
          }
        }

        // 8. Map edge numbers to global vertex indices (UNCHANGED for 0-11)
        const edgeToVert = new Map<number, number>();
        // ... same hash-map logic for edges 0-11 ...

        // 9. Map interior vertices (NEW)
        const interiorToVert = new Map<number, number>();
        for (const [vi, pos] of interiorPositions) {
          // Interior vertices are unique per-cube (no cross-cube sharing)
          const idx = vertices.length / 3;
          vertices.push(pos[0], pos[1], pos[2]);
          interiorToVert.set(vi, idx);
        }

        // 10. Emit triangles (MODIFIED: resolve which map to use per index)
        for (let t = 0; t < triVertIndices.length; t += 3) {
          const a = resolveVertIndex(triVertIndices[t], edgeToVert, interiorToVert);
          const b = resolveVertIndex(triVertIndices[t + 1], edgeToVert, interiorToVert);
          const c = resolveVertIndex(triVertIndices[t + 2], edgeToVert, interiorToVert);
          faces.push(a, b, c);
        }
      }
    }
  }
  // ...
}

function resolveVertIndex(
  vi: number,
  edgeToVert: Map<number, number>,
  interiorToVert: Map<number, number>,
): number {
  if (vi < 12) return edgeToVert.get(vi)!;
  return interiorToVert.get(vi)!;
}
```

---

## 6. New Function Signatures (Complete)

### `resolveTriangles`

```typescript
/**
 * Returns the complete triangle list for a cube, using Lewiner's sub-case
 * resolution when the configuration is ambiguous.
 *
 * @param corners - 8 scalar values at cube corners
 * @param cubeIdx - 8-bit index (bit c set if corners[c] < level)
 * @param config - case number from CASES table (0-14)
 * @param subconfig - sub-case index from CASES table (0 = classic)
 * @param level - isovalue
 * @returns Flat array of vertex indices: [v0,v1,v2, v0,v1,v2, ...]
 *          Indices 0-11 are edge midpoints, 12+ are interior vertices.
 */
function resolveTriangles(
  corners: Float32Array,
  cubeIdx: number,
  config: number,
  subconfig: number,
  level: number,
): number[]
```

### `computeInteriorVertex`

```typescript
/**
 * Computes position of an interior vertex (index 12-14) within a grid cell.
 *
 * Index 12 = body center (isosurface centroid within the cube)
 * Index 13, 14 = face/edge interior points for config 13 sub-cases
 */
function computeInteriorVertex(
  vertexIdx: number,        // 12, 13, or 14
  cx: number, cy: number, cz: number,  // cell origin in grid coords
  corners: Float32Array,    // 8 corner values
  level: number,            // isovalue
): [number, number, number]
```

### `findSubCaseIndex`

```typescript
/**
 * Determines which sub-case to use by testing the scalar field
 * at specific edge test positions.
 *
 * @param config - case number
 * @param corners - 8 corner values
 * @param level - isovalue
 * @returns entry index into the appropriate TILING table
 */
function findSubCaseIndex(
  config: number,
  corners: Float32Array,
  level: number,
): number
```

### `decodeTrianglesFromClassic`

```typescript
/**
 * Decodes CASESCLASSIC entry: extracts edge indices from 16-byte entry.
 * Each byte is an edge index (0-11) or 0xFF (end of triangle list).
 */
function decodeTrianglesFromClassic(
  classicEntry: Uint8Array,   // 16 bytes (subarray of CASESCLASSIC)
): number[]                   // flat edge indices
```

### `decodeTrianglesFromTiling`

```typescript
/**
 * Decodes a TILING table entry: extracts vertex indices.
 * Each byte is a vertex index (0-14) or 0xFF (end).
 *
 * @param tilingTable - the specific TILING* Uint8Array
 * @param entryIndex - which entry in the table
 * @param entrySize - bytes per entry (varies per TILING table)
 * @returns flat vertex indices
 */
function decodeTrianglesFromTiling(
  tilingTable: Uint8Array,
  entryIndex: number,
  entrySize: number,
): number[]
```

---

## 7. Implementation Phases

### Phase 1: Table Generation (estimated: 1 session)

Create a new file `src/mesh/lewiner_tables.ts` containing all decoded lookup tables.

**Steps:**
1. Copy all base64-encoded table strings from `lewiner_luts.py` into a TypeScript decoder
2. Write `decodeBase64Bytes()` to convert them to Uint8Array constants
3. Build the `TILING_DISPATCH` dispatch table
4. Validate that CASESCLASSIC matches the current TRI_TABLE for all 256 entries where subconfig == 0

**Validation:**
```typescript
// For every cubeIdx where CASES[cubeIdx*2+1] === 0:
//   assert decodeTrianglesFromClassic(CASESCLASSIC, cubeIdx)
//          deep-equals TRI_TABLE[cubeIdx]
```

### Phase 2: Sub-case Resolution Logic (estimated: 2 sessions)

Implement sub-case resolution for all 7 ambiguous config types (3, 4, 6, 7, 10, 12, 13).

**Steps:**
1. Implement `findSubCaseIndex()` for each config type, starting with simpler ones (4, 10) and building up to the most complex (13)
2. Implement `resolveTriangles()` combining classic and tiling paths
3. Implement `computeInteriorVertex()` for vertices 12-14
4. Write unit tests comparing against Python output for specific ambiguous cube configs

**Config 13 is the most complex:**
- 2 sub-cases from TEST13
- Each sub-case has 12 further sub-sub-cases from TILING13_1/2/3/4/5_1/5_2
- SUBCONFIG13 helps determine which sub-sub-case to use
- Interior vertices 13 and 14 are used frequently

### Phase 3: Integration (estimated: 1 session)

Modify `marching_cubes.ts` to use the new Lewiner triangle resolution.

**Steps:**
1. Import tables from `lewiner_tables.ts`
2. Modify the main loop to call `resolveTriangles()` instead of `TRI_TABLE[cubeIdx]`
3. Add interior vertex position computation and dedup
4. Run the full pipeline and compare output with Python GLB

### Phase 4: Validation (estimated: 1 session)

**Quantitative comparison with Python GLB output:**
- Vertex count match (target: 2,247,310)
- Face count match (target: 4,500,564)
- Byte-identical vertex positions (within float tolerance)
- Byte-identical face indices
- GLB structural validity

**Comparison methodology:**
```bash
# Extract vertex/face arrays from both GLBs for diff
./compare_glb.py python_output.glb  # ground truth
deno run --allow-read --allow-write src/validation/mesh_test.ts  # TS output
diff <(xxd python_verts.bin) <(xxd ts_verts.bin)
```

---

## 8. Edge Cases

### 8.1 Empty cubes (cubeIdx == 0 or 255)
Handled by `EDGE_TABLE[cubeIdx] === 0` early return. No change needed.

### 8.2 Flat cubes (all corners exactly at isovalue)
Can cause degenerate triangles. The current code handles this via post-processing in pipeline.ts (degenerate face removal). The Lewiner algorithm produces the same triangles as CASESCLASSIC for non-ambiguous cases, so this is unchanged.

### 8.3 Interior vertices at grid boundaries
Interior vertices (12+) are always within a single cell, so they are unaffected by grid boundaries. Edge vertex dedup (which uses grid neighbors) is unaffected.

### 8.4 Config 13 complexity
Config 13 has the most complex sub-case resolution (2 stages, 5+ tiling tables). It is also the rarest configuration. The implementation should handle it correctly but is the highest risk for bugs.

### 8.5 Case symmetry
Sub-cases are symmetric under cube rotations/flips. The Lewiner tables handle this internally -- the CASES entry's config/subconfig already accounts for the specific cube orientation.

### 8.6 Integer overflow in edgeKey
The existing edgeKey function packs (x, y, z, dir) into a 32-bit integer:
```typescript
return (x1 & 0x3FF) | ((y1 & 0x3FF) << 10) | ((z1 & 0x3FF) << 20) | (dir << 30);
```
This supports grid dimensions up to 1024 in each axis. No change needed.

---

## 9. Testing Strategy

### 9.1 Table correctness tests
```typescript
// Test that CASESCLASSIC matches TRI_TABLE for non-ambiguous cases
for (let i = 0; i < 256; i++) {
  if (CASES[i * 2 + 1] === 0) {
    const classic = decodeClassic(i);
    assertTrianglesEqual(classic, TRI_TABLE[i]);
  }
}
```

### 9.2 Per-case unit tests
```typescript
// For each of the 15 configs, generate a cube with known corner values
// that triggers that config, and verify the triangle count/positions
const testCases = [
  { corners: [...], expectedTriCount: 1 },  // config 1
  { corners: [...], expectedTriCount: 2 },  // config 2
  // ... etc
];
```

### 9.3 Known sub-case tests
```typescript
// Test specific ambiguous cube configs with known correct output
// Source: compare against scikit-image output for the same input
test("sub-case resolution for config 3", () => {
  // corners that produce cubeIdx with config=3, subconfig≠0
  const result = resolveTriangles(corners, cubeIdx, 3, subconfig, 0.5);
  // Verify against known correct triangle list
});
```

### 9.4 Full pipeline comparison
```typescript
// Run the full mesh pipeline on a known input and compare with Python
const input = loadTestInput("T-80BVM");
const result = meshFromVoxels(input);
const expected = loadExpectedOutput("T-80BVM-expected");
assert(result.byteLength === expected.byteLength);
assert(arrayEquals(new Uint8Array(result), new Uint8Array(expected)));
```

### 9.5 Visual regression
Load both GLB files in `model-viewer` or render both to PNG and compare:
- No holes in the Lewiner output
- Smooth surfaces at ambiguous configurations
- Consistent normals (no inverted faces)

---

## 10. Summary of Changes by File

| File | Change |
|------|--------|
| `src/mesh/marching_cubes.ts` | Replace `TRI_TABLE` lookup with Lewiner `resolveTriangles()`. Add interior vertex computation and dedup. Import tables from new file. |
| `src/mesh/lewiner_tables.ts` | **NEW FILE.** All decoded CASESCLASSIC, CASES, TILING*, TEST*, SUBCONFIG13 tables as Uint8Array constants. TILING_DISPATCH table. Base64 decoder helper. |
| `src/mesh/pipeline.ts` | **NO CHANGES.** Pipeline is algorithm-agnostic. |
| `src/mesh/marching_cubes_gpu.ts` | **NO CHANGES.** GPU implementation is dead code (128MB buffer limit). |
| `src/mesh/lewiner_tables_test.ts` | **NEW FILE.** Unit tests for table correctness and sub-case resolution. |
| `lewiner_luts.py` | **NO CHANGES.** Source of truth for table data. Do not modify. |

---

## 11. Key Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Table decoding errors from base64 | Wrong triangles, holes | Phase 1 validation -- verify CASESCLASSIC matches TRI_TABLE for subconfig==0 |
| Interior vertex position wrong | Mesh deformation | Phase 2 unit tests with known corner values |
| Config 13 sub-case logic incorrect | Rare but severe artifacts | Isolate config 13 testing; compare with Python output for the same cube |
| Sub-case selection wrong for config {n} | Surface holes | Per-case unit tests; visual regression test |
| Memory from large tables | ~15 KB of Uint8Array tables | Trivial; tables are a one-time allocation |
| Per-call overhead of sub-case resolution | Performance regression | Only triggered for ambiguous cubes (most cubes are non-ambiguous); pre-compute test edge positions where possible |
