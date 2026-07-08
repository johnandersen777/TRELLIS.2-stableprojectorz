/**
 * Minimal GLB binary writer for vertex-colored meshes.
 *
 * Produces valid glTF 2.0 .glb files with:
 *   - POSITION (vec3 f32) accessor
 *   - COLOR_0 (vec3 u8 normalized) accessor
 *   - Index accessor (u16 or u32)
 *
 * No external dependencies. ~150 lines.
 */

const GLB_MAGIC = 0x46546C67; // "glTF"
const GLB_VERSION = 2;
const CHUNK_JSON = 0x4E4F534A; // "JSON"
const CHUNK_BIN = 0x004E4942; // "BIN\0"

const FLOAT = 5126;
const UNSIGNED_SHORT = 5123;
const UNSIGNED_INT = 5125;
const UNSIGNED_BYTE = 5121;

function pad4(n: number): number {
  return (n + 3) & ~3;
}

export interface GlbOptions {
  name?: string;
  /** Skip Y/Z swizzle. Default: false (swizzle Z-up→Y-up). Set true if verts already Y-up. */
  noSwizzle?: boolean;
}

export function writeGlb(
  vertices: Float32Array, // flat [x0,y0,z0, x1,y1,z1, ...]
  faces: Uint32Array, // flat [i0,i1,i2, i0,i1,i2, ...]
  colors: Uint8Array, // flat [r0,g0,b0, r1,g1,b1, ...] — will be padded to RGBA
  options?: GlbOptions,
): ArrayBuffer {
  const vertexCount = vertices.length / 3;
  const faceCount = faces.length / 3;
  const useU16 = vertexCount <= 65535;
  const name = options?.name ?? "trellis_mesh";
  const noSwizzle = options?.noSwizzle ?? false;

  // Y/Z swap for GLB Y-up convention (if vertices are Z-up)
  let swizzledVerts: Float32Array;
  if (noSwizzle) {
    swizzledVerts = vertices;
  } else {
    swizzledVerts = new Float32Array(vertices.length);
    for (let i = 0; i < vertexCount; i++) {
      const i3 = i * 3;
      swizzledVerts[i3] = vertices[i3]; // X stays
      swizzledVerts[i3 + 1] = vertices[i3 + 2]; // Y ← Z
      swizzledVerts[i3 + 2] = -vertices[i3 + 1]; // Z ← -Y
    }
  }

  // Pad colors to RGBA (4 bytes per vertex)
  const colorsRGBA = new Uint8Array(vertexCount * 4);
  for (let i = 0; i < vertexCount; i++) {
    const i3 = Math.min(i * 3, colors.length - 3);
    colorsRGBA[i * 4] = colors[i3];
    colorsRGBA[i * 4 + 1] = colors[i3 + 1];
    colorsRGBA[i * 4 + 2] = colors[i3 + 2];
    colorsRGBA[i * 4 + 3] = 255;
  }

  // Convert faces to u16 if possible
  let indexData: ArrayBuffer;
  let indexComponentType: number;
  if (useU16) {
    const u16 = new Uint16Array(faces.length);
    u16.set(faces);
    indexData = u16.buffer as ArrayBuffer;
    indexComponentType = UNSIGNED_SHORT;
  } else {
    indexData = faces.buffer as ArrayBuffer;
    indexComponentType = UNSIGNED_INT;
  }

  // Layout: [indices] [vertices] [colors] — matches trimesh buffer view order
  const vertByteLength = vertexCount * 12; // vec3 f32
  const colorByteLength = vertexCount * 4; // vec4 u8
  const indexByteLength = indexData.byteLength;

  const binByteLength = indexByteLength + vertByteLength + colorByteLength;
  const binByteLengthPadded = pad4(binByteLength);

  // Compute bounding box
  let bx = Infinity, by = Infinity, bz = Infinity;
  let tx = -Infinity, ty = -Infinity, tz = -Infinity;
  for (let i = 0; i < vertexCount; i++) {
    const i3 = i * 3;
    if (swizzledVerts[i3] < bx) bx = swizzledVerts[i3];
    if (swizzledVerts[i3] > tx) tx = swizzledVerts[i3];
    if (swizzledVerts[i3 + 1] < by) by = swizzledVerts[i3 + 1];
    if (swizzledVerts[i3 + 1] > ty) ty = swizzledVerts[i3 + 1];
    if (swizzledVerts[i3 + 2] < bz) bz = swizzledVerts[i3 + 2];
    if (swizzledVerts[i3 + 2] > tz) tz = swizzledVerts[i3 + 2];
  }

  // Compute color bounds
  let crMin = 255, cgMin = 255, cbMin = 255, caMin = 255;
  let crMax = 0, cgMax = 0, cbMax = 0, caMax = 0;
  for (let i = 0; i < vertexCount; i++) {
    const c = colorsRGBA[i * 4]; if (c < crMin) crMin = c; if (c > crMax) crMax = c;
    const g = colorsRGBA[i * 4 + 1]; if (g < cgMin) cgMin = g; if (g > cgMax) cgMax = g;
    const b = colorsRGBA[i * 4 + 2]; if (b < cbMin) cbMin = b; if (b > cbMax) cbMax = b;
    const a = colorsRGBA[i * 4 + 3]; if (a < caMin) caMin = a; if (a > caMax) caMax = a;
  }

  // Match trimesh JSON structure exactly
  const json = {
    scene: 0,
    scenes: [{ nodes: [0] }],
    asset: { version: "2.0", generator: "https://github.com/mikedh/trimesh" },
    accessors: [
      {
        componentType: indexComponentType,
        type: "SCALAR",
        bufferView: 0,
        count: faceCount * 3,
        max: [vertexCount - 1],
        min: [0],
      },
      {
        componentType: FLOAT,
        type: "VEC3",
        byteOffset: 0,
        bufferView: 1,
        count: vertexCount,
        max: [tx, ty, tz],
        min: [bx, by, bz],
      },
      {
        componentType: UNSIGNED_BYTE,
        normalized: true,
        type: "VEC4",
        byteOffset: 0,
        bufferView: 2,
        count: vertexCount,
        max: [crMax, cgMax, cbMax, caMax],
        min: [crMin, cgMin, cbMin, caMin],
      },
    ],
    meshes: [{
      name: "geometry_0",
      extras: {},
      primitives: [{
        attributes: { POSITION: 1, COLOR_0: 2 },
        indices: 0,
        mode: 4,
      }],
    }],
    nodes: [
      { name: "world", children: [1] },
      { name: "geometry_0", mesh: 0 },
    ],
    buffers: [{ byteLength: binByteLengthPadded }],
    bufferViews: [
      { buffer: 0, byteOffset: 0, byteLength: indexByteLength },
      { buffer: 0, byteOffset: indexByteLength, byteLength: vertByteLength },
      { buffer: 0, byteOffset: indexByteLength + vertByteLength, byteLength: colorByteLength },
    ],
  };

  // Build JSON manually to match Python json.dumps format (spaces after colons/commas)
  const j = (v: unknown): string => {
    if (v === null) return 'null';
    if (typeof v === 'boolean') return v ? 'true' : 'false';
    if (typeof v === 'number') {
      if (Number.isInteger(v)) return v.toString();
      return v.toString(); // float — JS stringify is close enough
    }
    if (typeof v === 'string') return JSON.stringify(v);
    if (Array.isArray(v)) return '[' + v.map(j).join(', ') + ']';
    // Object
    const pairs = Object.entries(v as Record<string, unknown>).map(([k, val]) => `${JSON.stringify(k)}: ${j(val)}`);
    return '{' + pairs.join(', ') + '}';
  };
  const jsonStr = j(json);
  const jsonPadded = jsonStr + " ".repeat(pad4(jsonStr.length) - jsonStr.length);
  const jsonBytes = new TextEncoder().encode(jsonPadded);

  // Write GLB
  const totalSize = 12 + 8 + jsonBytes.length + 8 + binByteLengthPadded;
  const buf = new ArrayBuffer(totalSize);
  const dv = new DataView(buf);
  let off = 0;

  // GLB header
  dv.setUint32(off, GLB_MAGIC, true); off += 4;
  dv.setUint32(off, GLB_VERSION, true); off += 4;
  dv.setUint32(off, totalSize, true); off += 4;

  // JSON chunk
  dv.setUint32(off, jsonBytes.length, true); off += 4;
  dv.setUint32(off, CHUNK_JSON, true); off += 4;
  new Uint8Array(buf).set(jsonBytes, off); off += jsonBytes.length;

  // BIN chunk
  dv.setUint32(off, binByteLength, true); off += 4;
  dv.setUint32(off, CHUNK_BIN, true); off += 4;

  // Write binary data: indices first, then vertices, then colors
  const bin = new Uint8Array(buf, off);
  bin.set(new Uint8Array(indexData), 0);
  bin.set(new Uint8Array(swizzledVerts.buffer, swizzledVerts.byteOffset, swizzledVerts.byteLength), indexByteLength);
  bin.set(colorsRGBA, indexByteLength + vertByteLength);

  return buf;
}
