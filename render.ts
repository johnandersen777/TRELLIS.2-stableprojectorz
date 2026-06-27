#!/usr/bin/env -S deno run --unstable-webgpu --allow-read --allow-write

/**
 * render.ts — GLB → PNG via Deno WebGPU
 *
 * Usage:
 *   deno run --unstable-webgpu --allow-read --allow-write render.ts input.glb [output.png]
 *
 * If output.png is omitted, defaults to input.png (same stem, .png extension).
 *
 * Flags:
 *   --width N       Output image width  (default: 1024)
 *   --height N      Output image height (default: 1024)
 *   --bg R G B A    Background colour   (default: 0.15 0.15 0.15 1.0)
 *
 * Requires: Deno ≥ 2.0 with WebGPU support (--unstable-webgpu).
 *           A GPU with Vulkan, DX12, or Metal backend.
 */

// ============================================================================
// GLB / glTF 2.0 binary parser
// ============================================================================

const GLB_MAGIC = 0x46546C67; // "glTF"
const JSON_CHUNK = 0x4E4F534A; // "JSON"
const BIN_CHUNK  = 0x004E4942; // "BIN\0"

interface GLBHeader {
  magic: number;
  version: number;
  length: number;
}

interface GLBChunk {
  length: number;
  type: number;
  data: Uint8Array;
}

interface ParsedGLB {
  json: Record<string, unknown>;
  bin: Uint8Array | null;
}

function readU32(buf: Uint8Array, off: number): number {
  return new DataView(buf.buffer, buf.byteOffset + off, 4).getUint32(0, true);
}

function parseGLB(raw: Uint8Array): ParsedGLB {
  if (raw.byteLength < 12) throw new Error("GLB: file too small for header");

  const magic   = readU32(raw, 0);
  const version = readU32(raw, 4);
  const length  = readU32(raw, 8);

  if (magic !== GLB_MAGIC) throw new Error(`GLB: bad magic 0x${magic.toString(16)}`);
  if (version !== 2)       throw new Error(`GLB: unsupported version ${version}`);
  if (length !== raw.byteLength) console.warn(`GLB: declared length ${length} ≠ actual ${raw.byteLength}`);

  let json: Record<string, unknown> | null = null;
  let bin: Uint8Array | null = null;
  let off = 12;

  while (off < raw.byteLength) {
    if (off + 8 > raw.byteLength) break;
    const chunkLen  = readU32(raw, off);
    const chunkType = readU32(raw, off + 4);
    off += 8;
    if (off + chunkLen > raw.byteLength) throw new Error("GLB: chunk exceeds file");

    const data = raw.slice(off, off + chunkLen);
    off += chunkLen;

    if (chunkType === JSON_CHUNK) {
      json = JSON.parse(new TextDecoder().decode(data));
    } else if (chunkType === BIN_CHUNK) {
      bin = data;
    }
  }

  if (!json) throw new Error("GLB: missing JSON chunk");
  return { json, bin };
}

// ============================================================================
// glTF accessor helpers
// ============================================================================

const COMPONENT_SIZES: Record<number, number> = {
  5120: 1, // BYTE
  5121: 1, // UNSIGNED_BYTE
  5122: 2, // SHORT
  5123: 2, // UNSIGNED_SHORT
  5125: 4, // UNSIGNED_INT
  5126: 4, // FLOAT
};

const TYPE_COUNTS: Record<string, number> = {
  "SCALAR": 1,
  "VEC2":   2,
  "VEC3":   3,
  "VEC4":   4,
  "MAT2":   4,
  "MAT3":   9,
  "MAT4":   16,
};

function getAccessorData(
  json: Record<string, unknown>,
  bin: Uint8Array | null,
  accessorIdx: number,
): { data: Float32Array | Uint16Array | Uint32Array; count: number; type: string } | null {
  const accessors = json.accessors as Array<Record<string, unknown>> | undefined;
  if (!accessors || accessorIdx >= accessors.length) return null;
  const acc = accessors[accessorIdx];
  const bufferViewIdx = acc.bufferView as number;
  const componentType = acc.componentType as number;
  const count = acc.count as number;
  const type = acc.type as string;
  const byteOffset = (acc.byteOffset as number) ?? 0;

  const bufferViews = json.bufferViews as Array<Record<string, unknown>> | undefined;
  if (!bufferViews || bufferViewIdx >= bufferViews.length) return null;
  const bv = bufferViews[bufferViewIdx];

  const bvOffset = (bv.byteOffset as number) ?? 0;
  const bvStride = (bv.byteStride as number) ?? 0;

  const compSize = COMPONENT_SIZES[componentType];
  if (!compSize) throw new Error(`glTF: unknown component type ${componentType}`);

  const typeCount = TYPE_COUNTS[type];
  if (!typeCount) throw new Error(`glTF: unknown accessor type "${type}"`);

  const elementSize = typeCount * compSize;
  const stride = bvStride || elementSize;

  const totalBytes = stride * (count - 1) + elementSize;

  if (!bin) {
    // No BIN chunk — buffer must be embedded in data URIs (rare); bail for now
    throw new Error("GLB: no BIN chunk and no external buffer support yet");
  }

  const start = bvOffset + byteOffset;
  if (start + totalBytes > bin.byteLength) {
    throw new Error(`GLB: accessor data out of bounds (need ${start + totalBytes}, have ${bin.byteLength})`);
  }

  // If stride == elementSize, we can slice directly; otherwise we must interleave-extract
  if (stride === elementSize || stride === 0) {
    const slice = bin.slice(start, start + elementSize * count);
    if (componentType === 5126) { // FLOAT
      return { data: new Float32Array(slice.buffer, slice.byteOffset, count * typeCount), count, type };
    } else if (componentType === 5123) { // UNSIGNED_SHORT
      return { data: new Uint16Array(slice.buffer, slice.byteOffset, count * typeCount), count, type };
    } else if (componentType === 5125) { // UNSIGNED_INT
      return { data: new Uint32Array(slice.buffer, slice.byteOffset, count * typeCount), count, type };
    }
  }

  // Interleaved — extract
  if (componentType === 5126) { // FLOAT
    const out = new Float32Array(count * typeCount);
    for (let i = 0; i < count; i++) {
      const src = new Float32Array(bin.buffer, bin.byteOffset + start + i * stride, typeCount);
      out.set(src, i * typeCount);
    }
    return { data: out, count, type };
  } else if (componentType === 5123) { // UNSIGNED_SHORT
    const out = new Uint16Array(count * typeCount);
    for (let i = 0; i < count; i++) {
      const src = new Uint16Array(bin.buffer, bin.byteOffset + start + i * stride, typeCount);
      out.set(src, i * typeCount);
    }
    return { data: out, count, type };
  }

  throw new Error(`glTF: unsupported interleaved component type ${componentType}`);
}

interface MeshPrimitive {
  positions: Float32Array;
  normals?: Float32Array;
  indices?: Uint32Array | Uint16Array;
  vertexCount: number;
  indexCount: number;
}

function collectPrimitives(json: Record<string, unknown>, bin: Uint8Array | null): MeshPrimitive[] {
  const primitives: MeshPrimitive[] = [];
  const meshes = json.meshes as Array<Record<string, unknown>> | undefined;
  if (!meshes) throw new Error("glTF: no meshes in file");

  for (const mesh of meshes) {
    const prims = mesh.primitives as Array<Record<string, unknown>> | undefined;
    if (!prims) continue;
    for (const prim of prims) {
      const posAccIdx = (prim.attributes as Record<string, number>)?.["POSITION"];
      if (posAccIdx === undefined) continue;

      const posData = getAccessorData(json, bin, posAccIdx);
      if (!posData || posData.type !== "VEC3") continue;

      const positions = posData.data as Float32Array;
      const vertexCount = posData.count;

      // Normals
      let normals: Float32Array | undefined;
      const nrmAccIdx = (prim.attributes as Record<string, number>)?.["NORMAL"];
      if (nrmAccIdx !== undefined) {
        const nrmData = getAccessorData(json, bin, nrmAccIdx);
        if (nrmData && nrmData.type === "VEC3") normals = nrmData.data as Float32Array;
      }

      // Generate normals if missing
      if (!normals) {
        normals = new Float32Array(vertexCount * 3);
        // Will compute after collecting indices
      }

      // Indices
      let indices: Uint32Array | Uint16Array | undefined;
      const idxAccIdx = prim.indices as number | undefined;
      if (idxAccIdx !== undefined) {
        const idxData = getAccessorData(json, bin, idxAccIdx);
        if (idxData && idxData.type === "SCALAR") {
          indices = idxData.data as Uint32Array | Uint16Array;
        }
      }

      primitives.push({
        positions,
        normals,
        indices,
        vertexCount,
        indexCount: indices ? indices.length : vertexCount,
      });
    }
  }

  if (primitives.length === 0) throw new Error("glTF: no renderable primitives found");
  return primitives;
}

// Generate flat normals from positions + indices
function generateNormals(positions: Float32Array, indices: Uint32Array | Uint16Array | undefined): Float32Array {
  const vc = positions.length / 3;
  const normals = new Float32Array(vc * 3);

  const addNormal = (i0: number, i1: number, i2: number) => {
    const ax = positions[i1 * 3]     - positions[i0 * 3];
    const ay = positions[i1 * 3 + 1] - positions[i0 * 3 + 1];
    const az = positions[i1 * 3 + 2] - positions[i0 * 3 + 2];
    const bx = positions[i2 * 3]     - positions[i0 * 3];
    const by = positions[i2 * 3 + 1] - positions[i0 * 3 + 1];
    const bz = positions[i2 * 3 + 2] - positions[i0 * 3 + 2];
    const nx = ay * bz - az * by;
    const ny = az * bx - ax * bz;
    const nz = ax * by - ay * bx;
    for (const i of [i0, i1, i2]) {
      normals[i * 3]     += nx;
      normals[i * 3 + 1] += ny;
      normals[i * 3 + 2] += nz;
    }
  };

  if (indices) {
    for (let i = 0; i < indices.length; i += 3) {
      addNormal(indices[i], indices[i + 1], indices[i + 2]);
    }
  } else {
    for (let i = 0; i < vc; i += 3) {
      addNormal(i, i + 1, i + 2);
    }
  }

  // Normalize
  for (let i = 0; i < vc; i++) {
    const x = normals[i * 3], y = normals[i * 3 + 1], z = normals[i * 3 + 2];
    const len = Math.sqrt(x * x + y * y + z * z);
    if (len > 1e-10) {
      normals[i * 3] /= len; normals[i * 3 + 1] /= len; normals[i * 3 + 2] /= len;
    }
  }

  return normals;
}

function computeBBox(primitives: MeshPrimitive[]): { min: [number,number,number]; max: [number,number,number] } {
  const min: [number,number,number] = [Infinity, Infinity, Infinity];
  const max: [number,number,number] = [-Infinity, -Infinity, -Infinity];
  for (const p of primitives) {
    for (let i = 0; i < p.positions.length; i += 3) {
      for (let j = 0; j < 3; j++) {
        const v = p.positions[i + j];
        if (v < min[j]) min[j] = v;
        if (v > max[j]) max[j] = v;
      }
    }
  }
  return { min, max };
}

// ============================================================================
// Linear algebra helpers
// ============================================================================

type Vec3 = [number, number, number];
type Mat4 = Float32Array; // 16 elements, column-major

function vec3Sub(a: Vec3, b: Vec3): Vec3 { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function vec3Norm(v: Vec3): Vec3 {
  const len = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  return len > 1e-10 ? [v[0]/len, v[1]/len, v[2]/len] : [0,0,0];
}
function vec3Cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
}
function vec3Dot(a: Vec3, b: Vec3): number { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

function mat4Identity(): Mat4 {
  const m = new Float32Array(16);
  m[0]=m[5]=m[10]=m[15]=1;
  return m;
}

function mat4Perspective(fovY: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1.0 / Math.tan(fovY / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  // WebGPU NDC: Z in [0, 1] (not [-1, 1] like OpenGL)
  m[10] = far / (near - far);
  m[11] = -1;
  m[14] = (far * near) / (near - far);
  return m;
}

function mat4LookAt(eye: Vec3, center: Vec3, up: Vec3): Mat4 {
  const f = vec3Norm(vec3Sub(center, eye));
  const s = vec3Norm(vec3Cross(f, up));
  const u = vec3Cross(s, f);
  const m = new Float32Array(16);
  m[0]=s[0];  m[4]=s[1];  m[8]=s[2];  m[12]=-vec3Dot(s, eye);
  m[1]=u[0];  m[5]=u[1];  m[9]=u[2];  m[13]=-vec3Dot(u, eye);
  m[2]=-f[0]; m[6]=-f[1]; m[10]=-f[2]; m[14]=vec3Dot(f, eye);
  m[15]=1;
  return m;
}

function mat4Mul(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      out[r + c * 4] = a[r] * b[c * 4] + a[r + 4] * b[c * 4 + 1] + a[r + 8] * b[c * 4 + 2] + a[r + 12] * b[c * 4 + 3];
    }
  }
  return out;
}

// ============================================================================
// Minimal PNG encoder (filter byte 0 per scanline, zlib via CompressionStream)
// ============================================================================

async function encodePNG(width: number, height: number, rgba: Uint8Array): Promise<Uint8Array> {
  // Build raw scanlines: filter byte (0=None) + RGBA pixels
  const stride = width * 4;
  const raw = new Uint8Array(height * (1 + stride));
  for (let y = 0; y < height; y++) {
    raw[y * (1 + stride)] = 0; // filter: None
    const srcOff = y * stride;
    const dstOff = y * (1 + stride) + 1;
    // RGBA → PNG expects RGBA as-is; our WebGPU texture is already RGBA8Unorm
    raw.set(rgba.subarray(srcOff, srcOff + stride), dstOff);
  }

  // Deflate
  const deflated = new Uint8Array(
    await new Response(
      new Blob([raw]).stream().pipeThrough(new CompressionStream("deflate")),
    ).arrayBuffer(),
  );

  // zlib wrapper: CMF=0x78, FLG=0x01 (level 0) or 0x9C (level 6 default), then deflate, then adler32
  // Use 0x78 0x9C for default compression
  const cmf = 0x78;
  const flg = 0x9C;
  const adler = adler32(raw);
  const zlib = new Uint8Array(2 + deflated.length + 4);
  zlib[0] = cmf;
  zlib[1] = flg;
  zlib.set(deflated, 2);
  const dv = new DataView(zlib.buffer, zlib.byteOffset, zlib.byteLength);
  dv.setUint32(2 + deflated.length, adler, false); // big-endian for adler32 in zlib

  // Assemble PNG
  const signature = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

  const ihdr = pngChunk("IHDR", buildIHDR(width, height));
  const idat = pngChunk("IDAT", zlib);
  const iend = pngChunk("IEND", new Uint8Array(0));

  const total = signature.length + ihdr.length + idat.length + iend.length;
  const png = new Uint8Array(total);
  let off = 0;
  png.set(signature, off); off += signature.length;
  png.set(ihdr, off);      off += ihdr.length;
  png.set(idat, off);      off += idat.length;
  png.set(iend, off);
  return png;
}

function buildIHDR(w: number, h: number): Uint8Array {
  const buf = new Uint8Array(13);
  const dv = new DataView(buf.buffer);
  dv.setUint32(0, w, false);
  dv.setUint32(4, h, false);
  dv.setUint8(8, 8);  // bit depth
  dv.setUint8(9, 6);  // color type: RGBA
  dv.setUint8(10, 0); // compression
  dv.setUint8(11, 0); // filter
  dv.setUint8(12, 0); // interlace
  return buf;
}

function pngChunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(8 + data.length + 4);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length, false); // length (big-endian)
  out[4] = type.charCodeAt(0);
  out[5] = type.charCodeAt(1);
  out[6] = type.charCodeAt(2);
  out[7] = type.charCodeAt(3);
  out.set(data, 8);
  // CRC over type + data
  const crcData = out.slice(4, 8 + data.length);
  dv.setUint32(8 + data.length, crc32(crcData), false);
  return out;
}

function adler32(data: Uint8Array): number {
  let a = 1, b = 0;
  const len = data.length;
  for (let i = 0; i < len; i++) {
    a = (a + data[i]) % 65521;
    b = (b + a) % 65521;
  }
  return ((b << 16) | a) >>> 0;
}

// Precomputed CRC32 table
function makeCRC32Table(): Uint32Array {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) {
      c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    }
    table[i] = c;
  }
  return table;
}
const CRC32_TABLE = makeCRC32Table();

function crc32(data: Uint8Array): number {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < data.length; i++) {
    c = CRC32_TABLE[(c ^ data[i]) & 0xFF] ^ (c >>> 8);
  }
  return (c ^ 0xFFFFFFFF) >>> 0;
}

// ============================================================================
// WGSL shaders
// ============================================================================

const SHADER_COMMON = /* wgsl */ `
struct Uniforms {
  mvp: mat4x4<f32>,
  model: mat4x4<f32>,
  light_dir: vec3<f32>,
  ambient: vec3<f32>,
  diffuse_color: vec3<f32>,
}
`;

const VERTEX_SHADER = /* wgsl */ `
${SHADER_COMMON}

struct VertexInput {
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
}

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) world_normal: vec3<f32>,
  @location(1) light_dir: vec3<f32>,
  @location(2) ambient: vec3<f32>,
  @location(3) diffuse_color: vec3<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
  var out: VertexOutput;
  out.position = u.mvp * vec4<f32>(in.position, 1.0);
  out.world_normal = normalize((u.model * vec4<f32>(in.normal, 0.0)).xyz);
  out.light_dir = u.light_dir;
  out.ambient = u.ambient;
  out.diffuse_color = u.diffuse_color;
  return out;
}
`;

const FRAGMENT_SHADER = /* wgsl */ `
@fragment
fn fs_main(
  @location(0) world_normal: vec3<f32>,
  @location(1) light_dir: vec3<f32>,
  @location(2) ambient: vec3<f32>,
  @location(3) diffuse_color: vec3<f32>,
) -> @location(0) vec4<f32> {
  let n = normalize(world_normal);
  let l = normalize(light_dir);
  let ndotl = max(dot(n, l), 0.0);
  let wrapped = ndotl * 0.5 + 0.5;
  let color = ambient + wrapped * diffuse_color;
  return vec4<f32>(color, 1.0);
}
`;

// ============================================================================
// WebGPU renderer
// ============================================================================

interface RenderConfig {
  width: number;
  height: number;
  background: [number, number, number, number]; // RGBA
}

async function render(
  primitives: MeshPrimitive[],
  config: RenderConfig,
): Promise<Uint8Array> {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error("WebGPU: no adapter found. Check GPU driver / --unstable-webgpu flag.");

  const device = await adapter.requestDevice({
    requiredLimits: {
      maxTextureDimension2D: Math.max(config.width, config.height),
    },
  });

  // --- Compute bounding box and camera ---
  const bbox = computeBBox(primitives);
  const center: Vec3 = [
    (bbox.min[0] + bbox.max[0]) / 2,
    (bbox.min[1] + bbox.max[1]) / 2,
    (bbox.min[2] + bbox.max[2]) / 2,
  ];
  const extent: Vec3 = [
    bbox.max[0] - bbox.min[0],
    bbox.max[1] - bbox.min[1],
    bbox.max[2] - bbox.min[2],
  ];
  const maxExtent = Math.max(extent[0], extent[1], extent[2], 0.01);
  const dist = maxExtent * 2.5;

  const eye: Vec3 = [center[0] + dist * 0.6, center[1] + dist * 0.5, center[2] + dist];
  const up: Vec3 = [0, 1, 0];
  // If model is primarily XZ-plane, tilt camera up slightly
  const view = mat4LookAt(eye, center, up);
  const proj = mat4Perspective(Math.PI / 4, config.width / config.height, maxExtent * 0.01, maxExtent * 20);
  const vp = mat4Mul(proj, view);
  const model = mat4Identity(); // No extra model transform needed; glTF data is in world space

  const mvp = mat4Mul(vp, model);

  // --- Create vertex data ---
  // Merge all primitives into one draw call
  type GPUPrimitive = {
    positions: Float32Array;
    normals: Float32Array;
    indices?: Uint32Array;
  };

  const gpuPrims: GPUPrimitive[] = primitives.map(p => {
    let n = p.normals;
    if (!n || n.every(v => v === 0)) {
      n = generateNormals(p.positions, p.indices);
    }
    let indices: Uint32Array | undefined;
    if (p.indices) {
      indices = p.indices instanceof Uint32Array ? p.indices : new Uint32Array(p.indices);
    }
    return { positions: p.positions, normals: n, indices };
  });

  // Total vertex/index counts
  let totalVerts = 0, totalIdx = 0;
  for (const p of gpuPrims) {
    totalVerts += p.positions.length / 3;
    totalIdx += p.indices ? p.indices.length : p.positions.length / 3;
  }

  // Build interleaved vertex buffer: [pos.xyz, normal.xyz] per vertex
  const vertexData = new Float32Array(totalVerts * 6);
  const indexData = new Uint32Array(totalIdx);

  let vOff = 0, iOff = 0, vBase = 0;
  for (const p of gpuPrims) {
    const vc = p.positions.length / 3;
    for (let i = 0; i < vc; i++) {
      vertexData[vOff++] = p.positions[i * 3];
      vertexData[vOff++] = p.positions[i * 3 + 1];
      vertexData[vOff++] = p.positions[i * 3 + 2];
      vertexData[vOff++] = p.normals[i * 3];
      vertexData[vOff++] = p.normals[i * 3 + 1];
      vertexData[vOff++] = p.normals[i * 3 + 2];
    }
    if (p.indices) {
      for (let i = 0; i < p.indices.length; i++) {
        indexData[iOff++] = p.indices[i] + vBase;
      }
    } else {
      for (let i = 0; i < vc; i++) {
        indexData[iOff++] = vBase + i;
      }
    }
    vBase += vc;
  }

  // --- GPU buffers ---
  const vb = device.createBuffer({
    size: vertexData.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(vb.getMappedRange()).set(vertexData);
  vb.unmap();

  const ib = device.createBuffer({
    size: indexData.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Uint32Array(ib.getMappedRange()).set(indexData);
  ib.unmap();

  // --- Uniform buffer ---
  // Single buffer: mvp(64B) + model(64B) + light_dir(16B) + ambient(16B) + diffuse(16B) = 176B = 44 floats
  const uniformData = new Float32Array(44);
  uniformData.set(mvp, 0);                          // offset 0: mvp
  uniformData.set(model, 16);                       // offset 64: model (identity)
  uniformData[32] = 0.6; uniformData[33] = 0.8; uniformData[34] = 0.4; // offset 128: light_dir
  uniformData[36] = config.background[0] * 0.3;     // offset 144: ambient
  uniformData[37] = config.background[1] * 0.3;
  uniformData[38] = config.background[2] * 0.3;
  uniformData[40] = 0.82; uniformData[41] = 0.78; uniformData[42] = 0.72; // offset 160: diffuse
  const uniformBuffer = device.createBuffer({
    size: uniformData.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });
  new Float32Array(uniformBuffer.getMappedRange()).set(uniformData);
  uniformBuffer.unmap();

  // --- Bind group ---
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" as const },
      },
    ],
  });

  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
    ],
  });

  // --- Pipeline ---
  const shaderModule = device.createShaderModule({
    code: `
      ${VERTEX_SHADER}
      ${FRAGMENT_SHADER}
    `,
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  // --- Render targets ---
  const colorTexture = device.createTexture({
    size: [config.width, config.height],
    format: "rgba8unorm",
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  const depthTexture = device.createTexture({
    size: [config.width, config.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });

  const renderPipeline = device.createRenderPipeline({
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [{
        arrayStride: 6 * 4, // 6 floats = 24 bytes
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" as GPUVertexFormat },
          { shaderLocation: 1, offset: 12, format: "float32x3" as GPUVertexFormat },
        ],
      }],
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format: "rgba8unorm" }],
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "back",
    },
    depthStencil: {
      format: "depth24plus",
      depthWriteEnabled: true,
      depthCompare: "less",
    },
  });

  // --- Readback buffer ---
  const paddedBytesPerRow = Math.ceil(config.width * 4 / 256) * 256;
  const readbackBuffer = device.createBuffer({
    size: paddedBytesPerRow * config.height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // --- Render ---
  const encoder = device.createCommandEncoder();

  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: colorTexture.createView(),
      clearValue: { r: config.background[0], g: config.background[1], b: config.background[2], a: config.background[3] },
      loadOp: "clear",
      storeOp: "store",
    }],
    depthStencilAttachment: {
      view: depthTexture.createView(),
      depthClearValue: 1.0,
      depthLoadOp: "clear",
      depthStoreOp: "discard",
    },
  });

  pass.setPipeline(renderPipeline);
  pass.setBindGroup(0, bindGroup);
  pass.setVertexBuffer(0, vb);
  pass.setIndexBuffer(ib, "uint32");
  pass.drawIndexed(totalIdx);
  pass.end();

  // Copy color texture → readback buffer
  encoder.copyTextureToBuffer(
    { texture: colorTexture },
    { buffer: readbackBuffer, bytesPerRow: paddedBytesPerRow, rowsPerImage: config.height },
    [config.width, config.height],
  );

  device.queue.submit([encoder.finish()]);

  // --- Read back ---
  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const mapped = new Uint8Array(readbackBuffer.getMappedRange());

  // Copy with unpadding
  const pixels = new Uint8Array(config.width * config.height * 4);
  for (let y = 0; y < config.height; y++) {
    const srcStart = y * paddedBytesPerRow;
    const dstStart = y * config.width * 4;
    pixels.set(mapped.subarray(srcStart, srcStart + config.width * 4), dstStart);
  }

  readbackBuffer.unmap();

  // --- Cleanup ---
  vb.destroy();
  ib.destroy();
  uniformBuffer.destroy();
  colorTexture.destroy();
  depthTexture.destroy();
  readbackBuffer.destroy();
  device.destroy();

  return pixels;
}

// ============================================================================
// CLI entrypoint
// ============================================================================

function parseArgs(args: string[]): {
  input: string;
  output: string;
  width: number;
  height: number;
  bg: [number, number, number, number];
} {
  let input = "";
  let output = "";
  let width = 1024;
  let height = 1024;
  let bg: [number, number, number, number] = [0.15, 0.15, 0.15, 1.0];

  let i = 0;
  while (i < args.length) {
    const a = args[i];
    if (a === "--width" && i + 1 < args.length) {
      width = parseInt(args[++i]);
    } else if (a === "--height" && i + 1 < args.length) {
      height = parseInt(args[++i]);
    } else if (a === "--bg" && i + 4 < args.length) {
      bg = [
        parseFloat(args[++i]),
        parseFloat(args[++i]),
        parseFloat(args[++i]),
        parseFloat(args[++i]),
      ];
    } else if (!input) {
      input = a;
    } else if (!output) {
      output = a;
    } else {
      throw new Error(`Unknown argument: ${a}`);
    }
    i++;
  }

  if (!input) {
    console.error("Usage: deno run --unstable-webgpu --allow-read --allow-write render.ts <input.glb> [output.png] [--width W] [--height H] [--bg R G B A]");
    Deno.exit(1);
  }

  if (!output) {
    output = input.replace(/\.glb$/i, "") + ".png";
  }

  if (isNaN(width) || width < 1 || isNaN(height) || height < 1) {
    throw new Error("Invalid --width or --height");
  }

  return { input, output, width, height, bg };
}

async function main() {
  const cfg = parseArgs(Deno.args);

  console.error(`Loading: ${cfg.input}`);
  const raw = await Deno.readFile(cfg.input);
  const { json, bin } = parseGLB(raw);
  const primitives = collectPrimitives(json, bin);

  if (primitives.length === 0) {
    throw new Error("No renderable geometry found in GLB.");
  }

  let totalTris = 0;
  for (const p of primitives) {
    totalTris += (p.indices ? p.indices.length : p.vertexCount) / 3;
  }
  console.error(`Meshes: ${primitives.length}, triangles: ${totalTris}`);
  console.error(`Rendering: ${cfg.width}x${cfg.height}`);

  const pixels = await render(primitives, {
    width: cfg.width,
    height: cfg.height,
    background: cfg.bg,
  });

  // Flip vertically — WebGPU framebuffer origin is top-left, PNG expects top-left too.
  // But our data is already top-left.  However, WebGPU's copyTextureToBuffer
  // gives top-left origin rows.  PNG stores top-left first row first.
  // No flip needed.  Confirm: WebGPU framebuffer coordinate (0,0) = top-left.
  // So pixels[0..width*4] is the top row.  PNG scanline 0 is also top row.
  // Correct as-is.

  const png = await encodePNG(cfg.width, cfg.height, pixels);

  console.error(`Writing: ${cfg.output} (${png.byteLength} bytes)`);
  await Deno.writeFile(cfg.output, png);
  console.error("Done.");
}

if (import.meta.main) {
  main().catch(err => {
    console.error(`FATAL: ${err.message}`);
    Deno.exit(1);
  });
}
