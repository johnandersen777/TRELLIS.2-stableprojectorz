/**
 * Mesh processing pipeline — ported from export_glb_fallback.py.
 *
 * Takes TRELLIS.2 voxel coords + attribute volume, produces watertight vertex-colored GLB.
 * CPU-only, no WebGPU needed.
 */
import { marchingCubes, type MarchingCubesResult } from "./marching_cubes.ts";
import { writeGlb } from "./glb_writer.ts";

export interface AttrLayout {
  base_color: [number, number]; // start, end channel
  metallic?: number;
  roughness?: number;
  alpha?: number;
}

export interface MeshPipelineInput {
  coords: Int32Array; // (N, 3) flat [x0,y0,z0, x1,y1,z1, ...]
  attrs: Float32Array; // (N, numChannels) flat
  attrLayout: AttrLayout;
}

export interface MeshPipelineOptions {
  taubinIters?: number;
  minComponentFaces?: number;
  verbose?: boolean;
  gridScale?: number; // resolution G = next pow2
}

// ── EDT Color Fill (two-pass Manhattan distance transform) ──

function edtColorFill3D(
  colorGrid: Float32Array,
  dims: [number, number, number],
): void {
  const [nx, ny, nz] = dims;
  const size = nx * ny * nz;
  const dist = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const ci = i * 3;
    dist[i] = colorGrid[ci] >= 0 ? 0 : Infinity;
  }

  // Forward pass (x++, y++, z++)
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const idx = x + y * nx + z * nx * ny;
        const ci = idx * 3;
        if (dist[idx] === 0) continue;

        let bestDist = Infinity;
        let bestColR = 0, bestColG = 0, bestColB = 0;

        if (x > 0) {
          const nidx = (x - 1) + y * nx + z * nx * ny;
          const nd = dist[nidx] + 1;
          if (nd < bestDist) {
            bestDist = nd;
            const nci = nidx * 3;
            bestColR = colorGrid[nci]; bestColG = colorGrid[nci + 1]; bestColB = colorGrid[nci + 2];
          }
        }
        if (y > 0) {
          const nidx = x + (y - 1) * nx + z * nx * ny;
          const nd = dist[nidx] + 1;
          if (nd < bestDist) {
            bestDist = nd;
            const nci = nidx * 3;
            bestColR = colorGrid[nci]; bestColG = colorGrid[nci + 1]; bestColB = colorGrid[nci + 2];
          }
        }
        if (z > 0) {
          const nidx = x + y * nx + (z - 1) * nx * ny;
          const nd = dist[nidx] + 1;
          if (nd < bestDist) {
            bestDist = nd;
            const nci = nidx * 3;
            bestColR = colorGrid[nci]; bestColG = colorGrid[nci + 1]; bestColB = colorGrid[nci + 2];
          }
        }

        if (bestDist < Infinity) {
          dist[idx] = bestDist;
          colorGrid[ci] = bestColR;
          colorGrid[ci + 1] = bestColG;
          colorGrid[ci + 2] = bestColB;
        }
      }
    }
  }

  // Backward pass (x--, y--, z--)
  for (let z = nz - 1; z >= 0; z--) {
    for (let y = ny - 1; y >= 0; y--) {
      for (let x = nx - 1; x >= 0; x--) {
        const idx = x + y * nx + z * nx * ny;
        const ci = idx * 3;
        if (dist[idx] === 0) continue;

        let bestDist = dist[idx];
        let bestColR = colorGrid[ci], bestColG = colorGrid[ci + 1], bestColB = colorGrid[ci + 2];

        if (x + 1 < nx) {
          const nidx = (x + 1) + y * nx + z * nx * ny;
          const nd = dist[nidx] + 1;
          if (nd < bestDist) {
            bestDist = nd;
            const nci = nidx * 3;
            bestColR = colorGrid[nci]; bestColG = colorGrid[nci + 1]; bestColB = colorGrid[nci + 2];
          }
        }
        if (y + 1 < ny) {
          const nidx = x + (y + 1) * nx + z * nx * ny;
          const nd = dist[nidx] + 1;
          if (nd < bestDist) {
            bestDist = nd;
            const nci = nidx * 3;
            bestColR = colorGrid[nci]; bestColG = colorGrid[nci + 1]; bestColB = colorGrid[nci + 2];
          }
        }
        if (z + 1 < nz) {
          const nidx = x + y * nx + (z + 1) * nx * ny;
          const nd = dist[nidx] + 1;
          if (nd < bestDist) {
            bestDist = nd;
            const nci = nidx * 3;
            bestColR = colorGrid[nci]; bestColG = colorGrid[nci + 1]; bestColB = colorGrid[nci + 2];
          }
        }

        if (bestDist < dist[idx]) {
          dist[idx] = bestDist;
          colorGrid[ci] = bestColR;
          colorGrid[ci + 1] = bestColG;
          colorGrid[ci + 2] = bestColB;
        }
      }
    }
  }
}

// ── Gaussian filter on color grid (per-channel, separable 3D) ──

function gaussianFilterColorGrid(
  colorGrid: Float32Array,
  dims: [number, number, number],
  sigma: number,
): void {
  const [nx, ny, nz] = dims;
  const size = nx * ny * nz;

  // Deinterleave channels
  const ch0 = new Float32Array(size);
  const ch1 = new Float32Array(size);
  const ch2 = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    const ci = i * 3;
    ch0[i] = colorGrid[ci];
    ch1[i] = colorGrid[ci + 1];
    ch2[i] = colorGrid[ci + 2];
  }

  // Filter each channel
  const f0 = gaussianFilter3D(ch0, dims, sigma);
  const f1 = gaussianFilter3D(ch1, dims, sigma);
  const f2 = gaussianFilter3D(ch2, dims, sigma);

  // Re-interleave (clamp to [0,1])
  for (let i = 0; i < size; i++) {
    const ci = i * 3;
    colorGrid[ci] = Math.max(0, Math.min(1, f0[i]));
    colorGrid[ci + 1] = Math.max(0, Math.min(1, f1[i]));
    colorGrid[ci + 2] = Math.max(0, Math.min(1, f2[i]));
  }
}

// ── Taubin mesh smoothing (λ/μ alternating Laplacian) ──

function taubinSmooth(
  vertices: Float32Array,
  faces: Uint32Array,
  iterations: number,
  lambda: number,
  mu: number,
): Float32Array {
  if (iterations <= 0) return vertices;

  const n = vertices.length / 3;

  // Build per-vertex neighbor lists from triangle mesh
  const neighborSets: Set<number>[] = new Array(n);
  for (let i = 0; i < n; i++) neighborSets[i] = new Set();
  for (let i = 0; i < faces.length; i += 3) {
    const a = faces[i], b = faces[i + 1], c = faces[i + 2];
    neighborSets[a].add(b); neighborSets[a].add(c);
    neighborSets[b].add(a); neighborSets[b].add(c);
    neighborSets[c].add(a); neighborSets[c].add(b);
  }
  const neighbors: number[][] = new Array(n);
  for (let i = 0; i < n; i++) neighbors[i] = Array.from(neighborSets[i]);

  const buf1 = new Float32Array(vertices);
  const buf2 = new Float32Array(vertices.length);

  for (let iter = 0; iter < iterations; iter++) {
    // Shrink pass (λ)
    for (let i = 0; i < n; i++) {
      const nb = neighbors[i];
      if (nb.length === 0) {
        buf2[i * 3] = buf1[i * 3];
        buf2[i * 3 + 1] = buf1[i * 3 + 1];
        buf2[i * 3 + 2] = buf1[i * 3 + 2];
        continue;
      }
      let sx = 0, sy = 0, sz = 0;
      for (let j = 0; j < nb.length; j++) {
        const vi = nb[j] * 3;
        sx += buf1[vi]; sy += buf1[vi + 1]; sz += buf1[vi + 2];
      }
      const inv = 1.0 / nb.length;
      sx *= inv; sy *= inv; sz *= inv;
      buf2[i * 3]     = buf1[i * 3]     + lambda * (sx - buf1[i * 3]);
      buf2[i * 3 + 1] = buf1[i * 3 + 1] + lambda * (sy - buf1[i * 3 + 1]);
      buf2[i * 3 + 2] = buf1[i * 3 + 2] + lambda * (sz - buf1[i * 3 + 2]);
    }
    // Expand pass (μ)
    for (let i = 0; i < n; i++) {
      const nb = neighbors[i];
      if (nb.length === 0) {
        buf1[i * 3] = buf2[i * 3];
        buf1[i * 3 + 1] = buf2[i * 3 + 1];
        buf1[i * 3 + 2] = buf2[i * 3 + 2];
        continue;
      }
      let sx = 0, sy = 0, sz = 0;
      for (let j = 0; j < nb.length; j++) {
        const vi = nb[j] * 3;
        sx += buf2[vi]; sy += buf2[vi + 1]; sz += buf2[vi + 2];
      }
      const inv = 1.0 / nb.length;
      sx *= inv; sy *= inv; sz *= inv;
      buf1[i * 3]     = buf2[i * 3]     + mu * (sx - buf2[i * 3]);
      buf1[i * 3 + 1] = buf2[i * 3 + 1] + mu * (sy - buf2[i * 3 + 1]);
      buf1[i * 3 + 2] = buf2[i * 3 + 2] + mu * (sz - buf2[i * 3 + 2]);
    }
  }

  return buf1;
}

/**
 * Full pipeline: occupancy → marching cubes → color → GLB
 */
export function meshFromVoxels(
  input: MeshPipelineInput,
  options: MeshPipelineOptions = {},
): ArrayBuffer {
  const {
    taubinIters = 0,
    minComponentFaces = 2000,
    verbose = true,
  } = options;

  const { coords, attrs, attrLayout } = input;
  const N = coords.length / 3;

  // ── Extract base_color ──
  const attrStride = 6; // base_color(3) + metallic(1) + roughness(1) + alpha(1)
  const [bcStart, bcEnd] = attrLayout.base_color;
  const baseColor = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    for (let c = bcStart; c < bcEnd; c++) {
      const val = attrs[i * attrStride + c];
      baseColor[i * 3 + (c - bcStart)] = Math.max(0, Math.min(1, val));
    }
  }

  // ── Determine grid size ──
  let maxCoord = 0;
  for (let i = 0; i < coords.length; i++) {
    if (coords[i] > maxCoord) maxCoord = coords[i];
  }
  const gridScale = Math.pow(2, Math.ceil(Math.log2(maxCoord + 1)));
  if (verbose) console.log(`[mesh] Grid scale: ${gridScale}, N=${N}`);

  // ── Compute bounding box ──
  let mnX = Infinity, mnY = Infinity, mnZ = Infinity;
  let mxX = -Infinity, mxY = -Infinity, mxZ = -Infinity;
  for (let i = 0; i < N; i++) {
    const i3 = i * 3;
    if (coords[i3] < mnX) mnX = coords[i3];
    if (coords[i3 + 1] < mnY) mnY = coords[i3 + 1];
    if (coords[i3 + 2] < mnZ) mnZ = coords[i3 + 2];
    if (coords[i3] > mxX) mxX = coords[i3];
    if (coords[i3 + 1] > mxY) mxY = coords[i3 + 1];
    if (coords[i3 + 2] > mxZ) mxZ = coords[i3 + 2];
  }
  const pad = 2;
  const dims: [number, number, number] = [
    mxX - mnX + 1 + 2 * pad,
    mxY - mnY + 1 + 2 * pad,
    mxZ - mnZ + 1 + 2 * pad,
  ];
  if (verbose) {
    console.log(`[mesh] Bounds: [${mnX},${mnY},${mnZ}] → [${mxX},${mxY},${mxZ}], dims=[${dims}]`);
  }

  // ── Build occupancy grid ──
  const occ = new Float32Array(dims[0] * dims[1] * dims[2]);
  for (let i = 0; i < N; i++) {
    const i3 = i * 3;
    const gx = coords[i3] - mnX + pad;
    const gy = coords[i3 + 1] - mnY + pad;
    const gz = coords[i3 + 2] - mnZ + pad;
    occ[gx + gy * dims[0] + gz * dims[0] * dims[1]] = 1;
  }
  if (verbose) console.log(`[mesh] Occupancy grid built: ${dims}`);

  // ── Symmetry mirroring ──
  // Detect bilateral symmetry axis and mirror clean half to noisy half.
  // Ported from export_glb_fallback.py _remesh_from_occupancy.
  const occBool = new Float32Array(occ.length);
  for (let i = 0; i < occ.length; i++) occBool[i] = occ[i] > 0.5 ? 1 : 0;
  const totalOcc = occBool.reduce((a, v) => a + v, 0);

  // Find best symmetry axis by testing flip-IoU along each axis
  let bestAxis = 0, bestIoU = 0;
  for (let axis = 0; axis < 3; axis++) {
    const flipSize = axis === 0 ? dims[0] : axis === 1 ? dims[1] : dims[2];
    let intersect = 0, union = 0;
    for (let z = 0; z < dims[2]; z++) {
      for (let y = 0; y < dims[1]; y++) {
        for (let x = 0; x < dims[0]; x++) {
          const idx = x + y * dims[0] + z * dims[0] * dims[1];
          const orig = occBool[idx];
          let flipXi = x, flipYi = y, flipZi = z;
          if (axis === 0) flipXi = dims[0] - 1 - x;
          else if (axis === 1) flipYi = dims[1] - 1 - y;
          else flipZi = dims[2] - 1 - z;
          const flipIdx = flipXi + flipYi * dims[0] + flipZi * dims[0] * dims[1];
          const flip = occBool[flipIdx];
          if (orig > 0.5 || flip > 0.5) union++;
          if (orig > 0.5 && flip > 0.5) intersect++;
        }
      }
    }
    const iou = intersect / Math.max(union, 1);
    if (iou > bestIoU) { bestIoU = iou; bestAxis = axis; }
  }

  // Determine which half has cleaner colors (lower chroma-hue variance)
  // Python: rg = base[:,0] - base[:,1]; gb = base[:,1] - base[:,2]
  // var = rg.var() + gb.var()
  const cax = Math.floor((bestAxis === 0 ? mnX + mxX : bestAxis === 1 ? mnY + mxY : mnZ + mxZ) / 2);
  let sumRgLo = 0, sumRgLo2 = 0, sumGbLo = 0, sumGbLo2 = 0, nLo = 0;
  let sumRgHi = 0, sumRgHi2 = 0, sumGbHi = 0, sumGbHi2 = 0, nHi = 0;
  for (let i = 0; i < N; i++) {
    const i3 = i * 3;
    const gCoord = bestAxis === 0 ? coords[i3] : bestAxis === 1 ? coords[i3 + 1] : coords[i3 + 2];
    const rg = baseColor[i3] - baseColor[i3 + 1];
    const gb = baseColor[i3 + 1] - baseColor[i3 + 2];
    if (gCoord < cax) {
      sumRgLo += rg; sumRgLo2 += rg * rg;
      sumGbLo += gb; sumGbLo2 += gb * gb;
      nLo++;
    } else {
      sumRgHi += rg; sumRgHi2 += rg * rg;
      sumGbHi += gb; sumGbHi2 += gb * gb;
      nHi++;
    }
  }
  const varLo = nLo > 0 ? (sumRgLo2 / nLo - (sumRgLo / nLo) ** 2) + (sumGbLo2 / nLo - (sumGbLo / nLo) ** 2) : Infinity;
  const varHi = nHi > 0 ? (sumRgHi2 / nHi - (sumRgHi / nHi) ** 2) + (sumGbHi2 / nHi - (sumGbHi / nHi) ** 2) : Infinity;
  const goodIsLo = varLo <= varHi;
  const mirrorOn = (Math.max(varLo, varHi) > 3.0 * Math.min(varLo, varHi) + 1e-6) && (goodIsLo ? nLo : nHi) > 1000;

  if (mirrorOn) {
    if (verbose) console.log(`[mesh] Symmetry mirror colors: axis=${bestAxis}, iou=${bestIoU.toFixed(3)}, mirroring ${goodIsLo ? "lo→hi" : "hi→lo"}`);
  } else {
    if (verbose) console.log(`[mesh] Symmetry mirror skipped (variance ratio too low)`);
  }

  // ── Build color grid (matches Python srccol scatter) ──
  // Python scatters good-half colors → srccol grid, mirrors to other side.
  // Occupancy grid is NEVER modified by symmetry — only color source data.
  const colorGrid = new Float32Array(dims[0] * dims[1] * dims[2] * 3);
  colorGrid.fill(-1); // sentinel: no color assigned

  for (let i = 0; i < N; i++) {
    const i3 = i * 3;
    const gx = coords[i3] - mnX + pad;
    const gy = coords[i3 + 1] - mnY + pad;
    const gz = coords[i3 + 2] - mnZ + pad;
    const gCoord = bestAxis === 0 ? coords[i3] : bestAxis === 1 ? coords[i3 + 1] : coords[i3 + 2];
    const isGood = goodIsLo ? (gCoord < cax) : (gCoord >= cax);

    if (!mirrorOn || isGood) {
      const ci = (gx + gy * dims[0] + gz * dims[0] * dims[1]) * 3;
      colorGrid[ci] = baseColor[i3];
      colorGrid[ci + 1] = baseColor[i3 + 1];
      colorGrid[ci + 2] = baseColor[i3 + 2];
    }
  }

  // Mirror good-half colors to bad-half grid positions (Python: srcmask/srccol mirror)
  if (mirrorOn) {
    let mirroredColors = 0;
    for (let i = 0; i < N; i++) {
      const i3 = i * 3;
      const gCoord = bestAxis === 0 ? coords[i3] : bestAxis === 1 ? coords[i3 + 1] : coords[i3 + 2];
      const isGood = goodIsLo ? (gCoord < cax) : (gCoord >= cax);
      if (!isGood) continue;

      let mx = coords[i3], my = coords[i3 + 1], mz = coords[i3 + 2];
      if (bestAxis === 0) mx = 2 * cax - mx;
      else if (bestAxis === 1) my = 2 * cax - my;
      else mz = 2 * cax - mz;

      const mgx = Math.max(0, Math.min(dims[0] - 1, Math.round(mx - mnX + pad)));
      const mgy = Math.max(0, Math.min(dims[1] - 1, Math.round(my - mnY + pad)));
      const mgz = Math.max(0, Math.min(dims[2] - 1, Math.round(mz - mnZ + pad)));
      const mci = (mgx + mgy * dims[0] + mgz * dims[0] * dims[1]) * 3;

      if (colorGrid[mci] < 0) { // don't overwrite original if both map to same cell
        colorGrid[mci] = baseColor[i3];
        colorGrid[mci + 1] = baseColor[i3 + 1];
        colorGrid[mci + 2] = baseColor[i3 + 2];
        mirroredColors++;
      }
    }
    if (verbose) console.log(`[mesh] Mirrored ${mirroredColors} color cells`);
  }

  // ── EDT fill: propagate colors to every cell (unbounded, replaces bounded NN) ──
  edtColorFill3D(colorGrid, dims);
  if (verbose) console.log("[mesh] EDT color fill done");

  // ── Gaussian filter on color grid (σ=1.2, matches Python srccol smoothing) ──
  gaussianFilterColorGrid(colorGrid, dims, 1.2);
  if (verbose) console.log("[mesh] Color grid gaussian (σ=1.2) done");

  // ── Binary closing (scipy-compatible: 6-neighbor cross) ──
  const closed = binaryClosing3D(occ, dims);
  if (verbose) console.log("[mesh] Binary closing done");

  // ── Gaussian filter ──
  const smoothed = gaussianFilter3D(closed, dims, 0.6);
  if (verbose) console.log("[mesh] Gaussian filter done");

  // ── Marching cubes (CPU, can parallelize with Web Workers) ──
  const mcResult = marchingCubes(smoothed, dims[0], dims[1], dims[2], 0.5);
  if (verbose) console.log(`[mesh] Marching cubes: ${mcResult.vertices.length / 3} verts, ${mcResult.faces.length / 3} faces`);

  // ── Transform to world coordinates ──
  const positions = mcResult.vertices;
  const vertCount = positions.length / 3;
  for (let i = 0; i < vertCount; i++) {
    const i3 = i * 3;
    positions[i3] = (positions[i3] + mnX - pad) / gridScale - 0.5;
    positions[i3 + 1] = (positions[i3 + 1] + mnY - pad) / gridScale - 0.5;
    positions[i3 + 2] = (positions[i3 + 2] + mnZ - pad) / gridScale - 0.5;
  }

  // ── Taubin mesh smoothing: remove voxel staircase artifacts ──
  // λ=0.5 (shrink), μ=-0.53 (expand) — standard Taubin params, band-pass on mesh curvature
  const taubinResult = taubinSmooth(positions, mcResult.faces, taubinIters, 0.5, -0.53);
  for (let i = 0; i < positions.length; i++) positions[i] = taubinResult[i];
  if (verbose) console.log(`[mesh] Taubin smoothing: ${taubinIters} iterations`);

  // ── Sample vertex colors from dense colorGrid (filled by EDT) ──
  const vertexColors = new Uint8Array(vertCount * 3);
  for (let i = 0; i < vertCount; i++) {
    const i3 = i * 3;
    const wx = positions[i3];
    const wy = positions[i3 + 1];
    const wz = positions[i3 + 2];
    const gx = Math.round((wx + 0.5) * gridScale - mnX + pad);
    const gy = Math.round((wy + 0.5) * gridScale - mnY + pad);
    const gz = Math.round((wz + 0.5) * gridScale - mnZ + pad);
    const ci = (Math.max(0, Math.min(dims[0] - 1, gx)) +
      Math.max(0, Math.min(dims[1] - 1, gy)) * dims[0] +
      Math.max(0, Math.min(dims[2] - 1, gz)) * dims[0] * dims[1]) * 3;
    vertexColors[i3] = Math.round(Math.max(0, Math.min(1, colorGrid[ci])) * 255);
    vertexColors[i3 + 1] = Math.round(Math.max(0, Math.min(1, colorGrid[ci + 1])) * 255);
    vertexColors[i3 + 2] = Math.round(Math.max(0, Math.min(1, colorGrid[ci + 2])) * 255);
  }

  // Dump intermediate data for bisect comparison
  const dumpDir = Deno.env.get("TRELLIS_DUMP_DIR");
  if (dumpDir) {
    try { Deno.mkdirSync(dumpDir, { recursive: true }); } catch { /* ok */ }
    Deno.writeFileSync(`${dumpDir}/stage_verts.bin`, new Uint8Array(positions.buffer));
    Deno.writeFileSync(`${dumpDir}/stage_faces.bin`, new Uint8Array(mcResult.faces.buffer));
    Deno.writeFileSync(`${dumpDir}/stage_colors.bin`, vertexColors);
    Deno.writeFileSync(`${dumpDir}/dims.json`, new TextEncoder().encode(JSON.stringify({mnX,mnY,mnZ,mxX,mxY,mxZ,pad,gridScale,dims:Array.from(dims)})));
  }

  // ── Compact faces + merge duplicate vertices (matches skimage remove_degenerate_faces) ──
  const rawFaces = mcResult.faces;
  const rawVerts = positions;
  const rawVertCount = rawVerts.length / 3;

  // Build vertex merge map: for each vertex, find the canonical (lowest-index) equivalent
  const vertMap = new Int32Array(rawVertCount);
  for (let i = 0; i < rawVertCount; i++) vertMap[i] = i;

  // Check each face for degenerate edges (same-position vertices)
  for (let i = 0; i < rawFaces.length; i += 3) {
    const a = rawFaces[i], b = rawFaces[i + 1], c = rawFaces[i + 2];
    const i3a = a * 3, i3b = b * 3, i3c = c * 3;

    // Check if any two vertices are at the same position
    const eps = 1e-10;
    const abSame = Math.abs(rawVerts[i3a] - rawVerts[i3b]) < eps &&
                   Math.abs(rawVerts[i3a + 1] - rawVerts[i3b + 1]) < eps &&
                   Math.abs(rawVerts[i3a + 2] - rawVerts[i3b + 2]) < eps;
    const acSame = Math.abs(rawVerts[i3a] - rawVerts[i3c]) < eps &&
                   Math.abs(rawVerts[i3a + 1] - rawVerts[i3c + 1]) < eps &&
                   Math.abs(rawVerts[i3a + 2] - rawVerts[i3c + 2]) < eps;
    const bcSame = Math.abs(rawVerts[i3b] - rawVerts[i3c]) < eps &&
                   Math.abs(rawVerts[i3b + 1] - rawVerts[i3c + 1]) < eps &&
                   Math.abs(rawVerts[i3b + 2] - rawVerts[i3c + 2]) < eps;

    if (abSame) { const mn = Math.min(vertMap[a], vertMap[b]); vertMap[a] = mn; vertMap[b] = mn; }
    if (acSame) { const mn = Math.min(vertMap[a], vertMap[c]); vertMap[a] = mn; vertMap[c] = mn; }
    if (bcSame) { const mn = Math.min(vertMap[b], vertMap[c]); vertMap[b] = mn; vertMap[c] = mn; }
  }

  // Propagate merges (transitive closure)
  for (let iter = 0; iter < 3; iter++) {
    for (let i = 0; i < vertCount; i++) {
      vertMap[i] = vertMap[vertMap[i]];
    }
  }

  // Build new vertex array and remapping
  const newVertIdx = new Int32Array(vertCount);
  newVertIdx.fill(-1);
  const newVerts: number[] = [];
  for (let i = 0; i < vertCount; i++) {
    if (vertMap[i] === i) {
      newVertIdx[i] = newVerts.length / 3;
      newVerts.push(rawVerts[i * 3], rawVerts[i * 3 + 1], rawVerts[i * 3 + 2]);
    }
  }
  // Map all vertices to their canonical index
  for (let i = 0; i < vertCount; i++) {
    if (newVertIdx[i] < 0) newVertIdx[i] = newVertIdx[vertMap[i]];
  }

  // Build compacted face list with remapped indices, skipping degenerate faces
  const compacted: number[] = [];
  for (let i = 0; i < rawFaces.length; i += 3) {
    const a = newVertIdx[rawFaces[i]];
    const b = newVertIdx[rawFaces[i + 1]];
    const c = newVertIdx[rawFaces[i + 2]];
    if (a !== b && b !== c && a !== c) {
      compacted.push(a, b, c);
    }
  }
  const faces = new Uint32Array(compacted);
  const positions2 = new Float32Array(newVerts);
  if (verbose) console.log(`[mesh] Compacted: ${rawFaces.length/3} faces → ${faces.length/3}, ${vertCount} verts → ${newVerts.length/3}`);

  // Rebuild vertex colors for merged vertices (use canonical vertex's color)
  const vertexColors2 = new Uint8Array(newVerts.length);
  for (let i = 0; i < vertCount; i++) {
    if (vertMap[i] === i) {
      const newIdx = newVertIdx[i] * 3;
      vertexColors2[newIdx] = vertexColors[i * 3];
      vertexColors2[newIdx + 1] = vertexColors[i * 3 + 1];
      vertexColors2[newIdx + 2] = vertexColors[i * 3 + 2];
    }
  }

  // ── Filter small components ──
  if (verbose) console.log(`[mesh] Component filter skipped (${faces.length / 3} faces)`);

  // ── Export GLB ──
  const glb = writeGlb(positions2, faces, vertexColors2, { name: "trellis_output" });
  if (verbose) console.log(`[mesh] GLB written: ${glb.byteLength} bytes`);

  return glb;
}

// ── 3D Binary Closing ──

function binaryClosing3D(grid: Float32Array, dims: [number, number, number]): Float32Array {
  const [nx, ny, nz] = dims;
  // Match scipy: cross-shaped 6-neighbor structuring element
  const dilated = new Float32Array(nx * ny * nz);
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const idx = x + y * nx + z * nx * ny;
        if (grid[idx] < 0.5) continue;
        dilated[idx] = 1;
        if (x > 0) dilated[idx - 1] = 1;
        if (x < nx - 1) dilated[idx + 1] = 1;
        if (y > 0) dilated[idx - nx] = 1;
        if (y < ny - 1) dilated[idx + nx] = 1;
        if (z > 0) dilated[idx - nx * ny] = 1;
        if (z < nz - 1) dilated[idx + nx * ny] = 1;
      }
    }
  }
  const eroded = new Float32Array(nx * ny * nz);
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const idx = x + y * nx + z * nx * ny;
        if (dilated[idx] < 0.5) continue;
        let allFilled = true;
        if (x > 0 && dilated[idx - 1] < 0.5) allFilled = false;
        if (x < nx - 1 && dilated[idx + 1] < 0.5) allFilled = false;
        if (y > 0 && dilated[idx - nx] < 0.5) allFilled = false;
        if (y < ny - 1 && dilated[idx + nx] < 0.5) allFilled = false;
        if (z > 0 && dilated[idx - nx * ny] < 0.5) allFilled = false;
        if (z < nz - 1 && dilated[idx + nx * ny] < 0.5) allFilled = false;
        if (allFilled) eroded[idx] = 1;
      }
    }
  }
  return eroded;
}

// ── 3D Separable Gaussian Filter ──

function gaussianFilter3D(grid: Float32Array, dims: [number, number, number], sigma: number): Float32Array {
  const [nx, ny, nz] = dims;
  // Match scipy: truncate = 4.0 * sigma
  const radius = Math.ceil(sigma * 4.0);
  const kernel: number[] = [];
  for (let i = -radius; i <= radius; i++) {
    kernel.push(Math.exp(-0.5 * (i * i) / (sigma * sigma)));
  }
  const ksum = kernel.reduce((a, b) => a + b, 0);
  for (let i = 0; i < kernel.length; i++) kernel[i] /= ksum;

  // scipy 'reflect' boundary helper: reflect index at array boundary
  function reflect(idx: number, max: number): number {
    if (idx < 0) return -idx - 1;
    if (idx >= max) return 2 * max - idx - 1;
    return idx;
  }

  function separablePass(
    src: Float32Array, dst: Float32Array,
    axis: number, // 0=X, 1=Y, 2=Z
  ): void {
    // For each scanline along `axis`, apply 1D gaussian
    const [dimSize, planeSize] = axis === 0
      ? [nx, ny * nz]
      : axis === 1
      ? [ny, nx * nz]
      : [nz, nx * ny];

    const [outer1, outer2] = axis === 0
      ? [ny, nz]  // vary y and z, x is scanline
      : axis === 1
      ? [nx, nz]  // vary x and z, y is scanline
      : [nx, ny]; // vary x and y, z is scanline

    for (let o2 = 0; o2 < outer2; o2++) {
      for (let o1 = 0; o1 < outer1; o1++) {
        // Compute base offset for the start of this scanline
        const base = axis === 0
          ? o1 * nx + o2 * nx * ny           // (0, o1, o2)
          : axis === 1
          ? o1 + o2 * nx * ny                 // (o1, 0, o2)
          : o1 + o2 * nx;                     // (o1, o2, 0)

        const stride = axis === 0 ? 1 : axis === 1 ? nx : nx * ny;

        for (let i = 0; i < dimSize; i++) {
          let sum = 0;
          let wsum = 0;
          for (let k = 0; k < kernel.length; k++) {
            const si = reflect(i + k - radius, dimSize);
            sum += src[base + si * stride] * kernel[k];
            wsum += kernel[k];
          }
          dst[base + i * stride] = sum / wsum;
        }
      }
    }
  }

  const tmp1 = new Float32Array(nx * ny * nz);
  const tmp2 = new Float32Array(nx * ny * nz);
  const result = new Float32Array(nx * ny * nz);

  separablePass(grid, tmp1, 0);
  separablePass(tmp1, tmp2, 1);
  separablePass(tmp2, result, 2);

  return result;
}

// ── Connected component filter ──

function filterComponents(
  mc: MarchingCubesResult,
  minFaces: number,
): MarchingCubesResult {
  const faceCount = mc.faces.length / 3;
  if (faceCount <= minFaces) return mc;

  // Build face adjacency via shared edges
  const edgeMap = new Map<string, number[]>();
  for (let f = 0; f < faceCount; f++) {
    const f3 = f * 3;
    const a = mc.faces[f3];
    const b = mc.faces[f3 + 1];
    const c = mc.faces[f3 + 2];
    for (const [v1, v2] of [[a, b], [b, c], [c, a]]) {
      const key = v1 < v2 ? `${v1}-${v2}` : `${v2}-${v1}`;
      const list = edgeMap.get(key) ?? [];
      list.push(f);
      edgeMap.set(key, list);
    }
  }

  // BFS to find components
  const visited = new Uint8Array(faceCount);
  const components: number[][] = [];
  for (let f = 0; f < faceCount; f++) {
    if (visited[f]) continue;
    const comp: number[] = [];
    const queue = [f];
    visited[f] = 1;
    while (queue.length > 0) {
      const cf = queue.pop()!;
      comp.push(cf);
      const f3 = cf * 3;
      for (const [v1, v2] of [[mc.faces[f3], mc.faces[f3 + 1]], [mc.faces[f3 + 1], mc.faces[f3 + 2]], [mc.faces[f3 + 2], mc.faces[f3]]]) {
        const key = v1 < v2 ? `${v1}-${v2}` : `${v2}-${v1}`;
        for (const nf of edgeMap.get(key) ?? []) {
          if (!visited[nf]) {
            visited[nf] = 1;
            queue.push(nf);
          }
        }
      }
    }
    components.push(comp);
  }

  // Keep large components
  const keep = components
    .filter((c) => c.length >= minFaces)
    .sort((a, b) => b.length - a.length);

  if (keep.length === 0) return mc;

  // Remap vertex indices
  const faceSet = new Set(keep.flat());
  const oldToNew = new Map<number, number>();
  const newVerts: number[] = [];
  const newFaces: number[] = [];

  for (let f = 0; f < faceCount; f++) {
    if (!faceSet.has(f)) continue;
    const f3 = f * 3;
    for (let j = 0; j < 3; j++) {
      const old = mc.faces[f3 + j];
      let ni = oldToNew.get(old);
      if (ni === undefined) {
        ni = newVerts.length / 3;
        oldToNew.set(old, ni);
        newVerts.push(mc.vertices[old * 3], mc.vertices[old * 3 + 1], mc.vertices[old * 3 + 2]);
      }
      newFaces.push(ni);
    }
  }

  return {
    vertices: new Float32Array(newVerts),
    faces: new Uint32Array(newFaces),
  };
}
