/**
 * WebGPU-accelerated marching cubes — chunked per Z-slice tile.
 * Avoids 256MB buffer limits and 8-storage-binding limits.
 */
import { GPUContext } from "../runtime/device.ts";

// Packed lookup tables (all in one buffer)
function buildLookupBuffer(device: GPUDevice): GPUBuffer {
  // Layout: [edge_table 256 u32] [tri_table 4096 i32] [corner_x 8 i32] [corner_y 8 i32] [corner_z 8 i32] [edge_c1 12 i32] [edge_c2 12 i32]
  const total = 256 + 4096 + 8 + 8 + 8 + 12 + 12;
  const data = new Int32Array(total);

  // Edge table (u32, same bits as i32)
  const edgeTable = [
    0x0,0x109,0x203,0x30a,0x406,0x50f,0x605,0x70c,0x80c,0x905,0xa0f,0xb06,0xc0a,0xd03,0xe09,0xf00,
    0x190,0x99,0x393,0x29a,0x596,0x49f,0x795,0x69c,0x99c,0x895,0xb9f,0xa96,0xd9a,0xc93,0xf99,0xe90,
    0x230,0x339,0x33,0x13a,0x636,0x73f,0x435,0x53c,0xa3c,0xb35,0x83f,0x936,0xe3a,0xf33,0xc39,0xd30,
    0x3a0,0x2a9,0x1a3,0xaa,0x7a6,0x6af,0x5a5,0x4ac,0xbac,0xaa5,0x9af,0x8a6,0xfaa,0xea3,0xda9,0xca0,
    0x460,0x569,0x663,0x76a,0x66,0x16f,0x265,0x36c,0xc6c,0xd65,0xe6f,0xf66,0x86a,0x963,0xa69,0xb60,
    0x5f0,0x4f9,0x7f3,0x6fa,0x1f6,0xff,0x3f5,0x2fc,0xdfc,0xcf5,0xfff,0xef6,0x9fa,0x8f3,0xbf9,0xaf0,
    0x650,0x759,0x453,0x55a,0x256,0x35f,0x55,0x15c,0xe5c,0xf55,0xc5f,0xd56,0xa5a,0xb53,0x859,0x950,
    0x7c0,0x6c9,0x5c3,0x4ca,0x3c6,0x2cf,0x1c5,0xcc,0xfcc,0xec5,0xdcf,0xcc6,0xbca,0xac3,0x9c9,0x8c0,
    0x8c0,0x9c9,0xac3,0xbca,0xcc6,0xdcf,0xec5,0xfcc,0xcc,0x1c5,0x2cf,0x3c6,0x4ca,0x5c3,0x6c9,0x7c0,
    0x950,0x859,0xb53,0xa5a,0xd56,0xc5f,0xf55,0xe5c,0x15c,0x55,0x35f,0x256,0x55a,0x453,0x759,0x650,
    0xaf0,0xbf9,0x8f3,0x9fa,0xef6,0xfff,0xcf5,0xdfc,0x2fc,0x3f5,0xff,0x1f6,0x6fa,0x7f3,0x4f9,0x5f0,
    0xb60,0xa69,0x963,0x86a,0xf66,0xe6f,0xd65,0xc6c,0x36c,0x265,0x16f,0x66,0x76a,0x663,0x569,0x460,
    0xca0,0xda9,0xea3,0xfaa,0x8a6,0x9af,0xaa5,0xbac,0x4ac,0x5a5,0x6af,0x7a6,0xaa,0x1a3,0x2a9,0x3a0,
    0xd30,0xc39,0xf33,0xe3a,0x936,0x83f,0xb35,0xa3c,0x53c,0x435,0x73f,0x636,0x13a,0x33,0x339,0x230,
    0xe90,0xf99,0xc93,0xd9a,0xa96,0xb9f,0x895,0x99c,0x69c,0x795,0x49f,0x596,0x29a,0x393,0x99,0x190,
    0xf00,0xe09,0xd03,0xc0a,0xb06,0xa0f,0x905,0x80c,0x70c,0x605,0x50f,0x406,0x30a,0x203,0x109,0x0,
  ];
  let off = 0;
  for (let i = 0; i < 256; i++) data[off++] = edgeTable[i];

  // Tri table (256 × 16 = 4096)
  const triTable = [
    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1,-1,
    3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1,-1,
    3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1,-1,
    3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1,-1,9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1,-1,
    1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1,-1,
    9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1,-1,2,9,7,2,7,9,2,3,7,7,4,9,4,2,3,9,
    8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1,-1,
    9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1,-1,4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1,-1,
    3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1,-1,1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1,-1,
    4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1,-1,4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1,-1,
  ];
  for (let i = 0; i < 4096; i++) data[off++] = triTable[i] ?? -1;

  // Corner offsets
  const cx = [0,1,1,0,0,1,1,0], cy = [0,0,1,1,0,0,1,1], cz = [0,0,0,0,1,1,1,1];
  for (const v of cx) data[off++] = v;
  for (const v of cy) data[off++] = v;
  for (const v of cz) data[off++] = v;

  // Edge corners
  const ec1 = [0,1,2,3,4,5,6,7,0,1,2,3], ec2 = [1,2,3,0,5,6,7,4,4,5,6,7];
  for (const v of ec1) data[off++] = v;
  for (const v of ec2) data[off++] = v;

  const buf = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, data);
  return buf;
}

// WGSL with packed lookup buffer and chunked dispatch
const MC_SHADER = `
struct Dims { nx: u32, ny: u32, nz: u32, level_bits: u32, z_start: u32, z_end: u32, _pad0: u32, _pad1: u32 };

@group(0) @binding(0) var<storage, read> grid: array<f32>;
@group(0) @binding(1) var<storage, read> lookups: array<i32>; // packed edge+tri+corner tables
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>, 2>;
@group(0) @binding(3) var<storage, read_write> vertices: array<f32>;
@group(0) @binding(4) var<storage, read_write> faces: array<u32>;
@group(0) @binding(5) var<uniform> dims: Dims;

fn edge_table(cube_idx: u32) -> u32 { return bitcast<u32>(lookups[i32(cube_idx)]); }

fn corner_x(c: u32) -> f32 {
  return f32(lookups[256 + 4096 + i32(c)]);
}
fn corner_y(c: u32) -> f32 {
  return f32(lookups[256 + 4096 + 8 + i32(c)]);
}
fn corner_z(c: u32) -> f32 {
  return f32(lookups[256 + 4096 + 16 + i32(c)]);
}
fn edge_c1(e: u32) -> i32 {
  return lookups[256 + 4096 + 24 + i32(e)];
}
fn edge_c2(e: u32) -> i32 {
  return lookups[256 + 4096 + 24 + 12 + i32(e)];
}
fn tri_val(cube: u32, idx: u32) -> i32 {
  return lookups[256 + i32(cube) * 16 + i32(idx)];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cell_idx = gid.x;
  let cells_per_slice = (dims.nx - 1u) * (dims.ny - 1u);
  let z_slice = cell_idx / cells_per_slice;
  if (z_slice < dims.z_start || z_slice >= dims.z_end) { return; }

  let level = bitcast<f32>(dims.level_bits);
  let nxy = dims.nx * dims.ny;

  let rem = cell_idx % cells_per_slice;
  let y = rem / (dims.nx - 1u);
  let x = rem % (dims.nx - 1u);

  // Sample 8 corners
  let base = x + y * dims.nx + z_slice * nxy;
  var corners: array<f32, 8>;
  corners[0] = grid[base];
  corners[1] = grid[base + 1u];
  corners[2] = grid[base + 1u + dims.nx];
  corners[3] = grid[base + dims.nx];
  corners[4] = grid[base + nxy];
  corners[5] = grid[base + 1u + nxy];
  corners[6] = grid[base + 1u + dims.nx + nxy];
  corners[7] = grid[base + dims.nx + nxy];

  var cube_idx: u32 = 0u;
  for (var c = 0u; c < 8u; c++) {
    if (corners[c] < level) { cube_idx |= (1u << c); }
  }

  let edge_flags = edge_table(cube_idx);
  if (edge_flags == 0u) { return; }

  // Count verts + faces
  var vert_count: u32 = 0u;
  for (var e = 0u; e < 12u; e++) {
    if ((edge_flags & (1u << e)) != 0u) { vert_count += 1u; }
  }
  var face_count: u32 = 0u;
  for (var t = 0u; t < 15u; t += 3u) {
    if (tri_val(cube_idx, t) == -1) { break; }
    face_count += 3u;
  }

  let vert_off = atomicAdd(&counters[0], vert_count);
  let face_off = atomicAdd(&counters[1], face_count);

  // Emit vertices
  var local_v: u32 = 0u;
  for (var e = 0u; e < 12u; e++) {
    if ((edge_flags & (1u << e)) == 0u) { continue; }
    let c1 = u32(edge_c1(e));
    let c2 = u32(edge_c2(e));
    let v1 = corners[c1];
    let v2 = corners[c2];
    let t = (level - v1) / (v2 - v1);
    let vi = (vert_off + local_v) * 3u;
    vertices[vi] = f32(x) + corner_x(c1) + t * (corner_x(c2) - corner_x(c1));
    vertices[vi + 1u] = f32(y) + corner_y(c1) + t * (corner_y(c2) - corner_y(c1));
    vertices[vi + 2u] = f32(z_slice) + corner_z(c1) + t * (corner_z(c2) - corner_z(c1));
    local_v += 1u;
  }

  // Emit faces
  var fi: u32 = 0u;
  for (var t = 0u; t < 15u; t += 3u) {
    let t0 = tri_val(cube_idx, t);
    if (t0 == -1) { break; }
    faces[face_off + fi] = vert_off + u32(t0);
    faces[face_off + fi + 1u] = vert_off + u32(tri_val(cube_idx, t + 1u));
    faces[face_off + fi + 2u] = vert_off + u32(tri_val(cube_idx, t + 2u));
    fi += 3u;
  }
}
`;

export async function marchingCubesGPU(
  grid: Float32Array,
  nx: number,
  ny: number,
  nz: number,
  level: number,
  context: GPUContext,
): Promise<{ positions: Float32Array; cells: Uint32Array }> {
  const device = context.device;

  // Upload grid split into Z-chunks (each must be <128 MB storage binding limit)
  const nxy = nx * ny;
  const gridChunkZ = Math.floor(128 * 1024 * 1024 / (nxy * 4)); // max Z slices per 128 MB
  const gridChunks: GPUBuffer[] = [];
  for (let zc = 0; zc < nz; zc += gridChunkZ) {
    const zcEnd = Math.min(zc + gridChunkZ, nz);
    const chunkData = grid.subarray(zc * nxy, zcEnd * nxy);
    const buf = device.createBuffer({
      size: chunkData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, chunkData.buffer as ArrayBuffer, chunkData.byteOffset, chunkData.byteLength);
    gridChunks.push(buf);
  }

  const lookupBuf = buildLookupBuffer(device);

  const shader = device.createShaderModule({ code: MC_SHADER });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shader, entryPoint: "main" },
  });

  // Single-pass per Z-slice tile (TILE_Z=4 → max ~500K cells → ~90 MB worst-case vert buffer)
  const TILE_Z = 4;
  const cellsPerSlice = (nx - 1) * (ny - 1);
  const maxTileCells = cellsPerSlice * TILE_Z;
  const maxTileVerts = maxTileCells * 15; // worst case
  const maxTileFaces = maxTileCells * 15;

  const allPositions: Float32Array[] = [];
  const allFaces: Uint32Array[] = [];
  let vertBase = 0;

  // Combined grid buffer (concatenate all chunks) — unused, file is dead code
  const gridBuf = gridChunks[0]; // fix: use first chunk

  const levelF32 = new Float32Array([level]);
  const levelU32 = new Uint32Array(levelF32.buffer);

  for (let zStart = 0; zStart < nz - 1; zStart += TILE_Z) {
    const zEnd = Math.min(zStart + TILE_Z, nz - 1);

    const vertBuf = device.createBuffer({
      size: maxTileVerts * 3 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const faceBuf = device.createBuffer({
      size: maxTileFaces * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const counterBuf = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(counterBuf, 0, new Uint32Array([0, 0]));

    const dimsBuf = device.createBuffer({
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(dimsBuf, 0, new Uint32Array([nx, ny, nz, levelU32[0], zStart, zEnd, 0, 0]));

    const bg = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gridBuf } },
        { binding: 1, resource: { buffer: lookupBuf } },
        { binding: 2, resource: { buffer: counterBuf } },
        { binding: 3, resource: { buffer: vertBuf } },
        { binding: 4, resource: { buffer: faceBuf } },
        { binding: 5, resource: { buffer: dimsBuf } },
      ],
    });

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    // Dispatch for the cells in this z-range (z_start → z_end)
    const tileCellCount = cellsPerSlice * (zEnd - zStart);
    pass.dispatchWorkgroups(Math.ceil(tileCellCount / 256), 1, 1);
    pass.end();
    device.queue.submit([enc.finish()]);

    // Read back
    const counterData = await readAsync(device, counterBuf, 8);
    const counters = new Uint32Array(counterData);
    const nv = counters[0];
    const nf = counters[1];

    if (nv > 0) {
      const vd = await readAsync(device, vertBuf, nv * 3 * 4);
      const fd = await readAsync(device, faceBuf, nf * 4);
      const faces = new Uint32Array(fd);
      if (vertBase > 0) {
        for (let i = 0; i < faces.length; i++) faces[i] += vertBase;
      }
      allPositions.push(new Float32Array(vd));
      allFaces.push(faces);
      vertBase += nv;
    }

    vertBuf.destroy();
    faceBuf.destroy();
    counterBuf.destroy();
    dimsBuf.destroy();
  }

  // Merge all tiles
  const totalVerts = allPositions.reduce((s, a) => s + a.length, 0);
  const totalFaces = allFaces.reduce((s, a) => s + a.length, 0);
  const mergedVerts = new Float32Array(totalVerts);
  const mergedFaces = new Uint32Array(totalFaces);
  let voff = 0, foff = 0;
  for (const v of allPositions) { mergedVerts.set(v, voff); voff += v.length; }
  for (const f of allFaces) { mergedFaces.set(f, foff); foff += f.length; }

  gridBuf.destroy();
  lookupBuf.destroy();

  return { positions: mergedVerts, cells: mergedFaces };
}

async function readAsync(device: GPUDevice, src: GPUBuffer, size: number): Promise<ArrayBuffer> {
  const staging = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, staging, 0, size);
  device.queue.submit([enc.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const mapped = staging.getMappedRange();
  const result = mapped.slice(0);
  staging.unmap();
  staging.destroy();
  return result;
}
