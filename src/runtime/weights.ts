/**
 * Safetensors streaming parser.
 *
 * Format: [8 bytes: header_len LE u64] [JSON header] [raw tensor data, contiguous]
 *
 * Reads ONLY the JSON header eagerly (~1 MB for 4B params).
 * Tensor data read on-demand via file seek + read.
 * This prevents the 23.9 GB RAM spike Python suffers (loading ALL safetensors at init).
 */

export interface TensorEntry {
  dtype: "F32" | "F16" | "BF16" | "I64" | "I32" | "I8" | "U8" | "U32";
  shape: number[];
  dataOffsets: [number, number]; // byte offsets [start, end) relative to data section
  numel: number;
}

export interface SafetensorsHeader {
  metadata: Record<string, string>;
  tensors: Map<string, TensorEntry>;
  dataByteOffset: number; // file offset where raw data starts
}

/**
 * IEEE 754 half-precision (float16) to float32 bit conversion.
 * Handles: normal, subnormal, zero, infinity, NaN.
 */
function f16ToF32Bits(h: number): number {
  const s = (h & 0x8000) << 16;       // sign
  const e = (h >> 10) & 0x1f;          // exponent (5 bits)
  const m = h & 0x03ff;                // mantissa (10 bits)

  if (e === 0) {
    // Zero or subnormal
    if (m === 0) return s;             // ±0
    // Subnormal: normalize the mantissa
    let m2 = m;
    let e2 = -14; // exponent offset for subnormals
    // Find the leading 1 bit
    while ((m2 & 0x0400) === 0) {
      m2 <<= 1;
      e2--;
    }
    m2 &= 0x03ff; // clear the implicit 1
    return s | ((e2 + 127) << 23) | (m2 << 13);
  }

  if (e === 0x1f) {
    // Infinity or NaN
    if (m === 0) return s | 0x7f800000; // ±Infinity
    return s | 0x7fc00000;              // NaN (canonicalize)
  }

  // Normal number
  return s | ((e - 15 + 127) << 23) | (m << 13);
}

export class SafetensorsFile {
  private file: Deno.FsFile | null = null;
  readonly path: string;
  private _header: SafetensorsHeader | null = null;

  constructor(path: string) {
    this.path = path;
  }

  /** Open file and parse header (lightweight — only reads header bytes) */
  open(): SafetensorsHeader {
    this.file = Deno.openSync(this.path, { read: true });

    // Read 8-byte header length
    const sizeBuf = new Uint8Array(8);
    this.file.readSync(sizeBuf);
    const headerLen = Number(new DataView(sizeBuf.buffer).getBigUint64(0, true));

    // Read JSON header
    const headerBuf = new Uint8Array(headerLen);
    this.file.readSync(headerBuf);
    const rawJson = new TextDecoder().decode(headerBuf);
    const parsed = JSON.parse(rawJson);

    const tensors = new Map<string, TensorEntry>();
    for (const [key, value] of Object.entries(parsed)) {
      if (key === "__metadata__") continue;
      const entry = value as Record<string, unknown>;
      const offsets = entry.data_offsets as [number, number];
      const shape = entry.shape as number[];
      const dtype = entry.dtype as TensorEntry["dtype"];
      const numel = shape.reduce((a: number, b: number) => a * b, 1);
      tensors.set(key, { dtype, shape, dataOffsets: offsets, numel });
    }

    this._header = {
      metadata: (parsed as Record<string, unknown>).__metadata__ as Record<string, string> ?? {},
      tensors,
      dataByteOffset: 8 + headerLen,
    };
    return this._header;
  }

  get header(): SafetensorsHeader {
    if (!this._header) throw new Error("File not opened. Call open() first.");
    return this._header;
  }

  get tensorCount(): number {
    return this.header.tensors.size;
  }

  /** List all tensor names */
  listTensors(): string[] {
    return Array.from(this.header.tensors.keys());
  }

  /** Get tensor names matching a prefix (e.g., "blocks.0.") */
  getTensorsByPrefix(prefix: string): string[] {
    return this.listTensors().filter((k) => k.startsWith(prefix));
  }

  /** Get tensor metadata without reading data */
  getTensorInfo(name: string): TensorEntry | undefined {
    return this.header.tensors.get(name);
  }

  /**
   * Read a single tensor's raw data from disk.
   * Returns ArrayBuffer — weight bytes exactly as stored.
   * Caller converts BF16→F32 if needed.
   */
  readTensorRaw(name: string): ArrayBuffer {
    if (!this.file) throw new Error("File not opened");
    const entry = this.header.tensors.get(name);
    if (!entry) throw new Error(`Tensor "${name}" not found`);

    const byteLength = entry.dataOffsets[1] - entry.dataOffsets[0];
    const fileOffset = this.header.dataByteOffset + entry.dataOffsets[0];

    const buf = new Uint8Array(byteLength);
    this.file.seekSync(fileOffset, Deno.SeekMode.Start);
    this.file.readSync(buf);
    return buf.buffer;
  }

  /**
   * Read tensor and convert to Float32Array.
   * BF16→F32: pad lower 16 bits with zeros.
   * F16→F32: kept as Uint16Array raw (caller can convert if needed).
   * F32: direct Float32Array view.
   */
  readTensorF32(name: string): Float32Array {
    const raw = this.readTensorRaw(name);
    const entry = this.header.tensors.get(name)!;

    switch (entry.dtype) {
      case "F32":
        return new Float32Array(raw);
      case "BF16": {
        // BF16: upper 16 bits are the float, lower 16 bits are zero
        const u16 = new Uint16Array(raw);
        const f32 = new Float32Array(u16.length);
        const u32View = new Uint32Array(f32.buffer);
        for (let i = 0; i < u16.length; i++) {
          u32View[i] = (u16[i] << 16);
        }
        return f32;
      }
      case "F16": {
        // F16: proper IEEE 754 half→float conversion
        const u16 = new Uint16Array(raw);
        const f32 = new Float32Array(u16.length);
        const u32View = new Uint32Array(f32.buffer);
        for (let i = 0; i < u16.length; i++) {
          u32View[i] = f16ToF32Bits(u16[i]);
        }
        return f32;
      }
      case "I32":
      case "U32":
      case "I64":
      case "I8":
      case "U8":
        // Return raw bytes as float array (caller knows the dtype)
        return new Float32Array(raw);
      default:
        return new Float32Array(raw);
    }
  }

  /** Close the file handle */
  close(): void {
    this.file?.close();
    this.file = null;
  }
}

/**
 * Multi-file model: opens multiple safetensors files and provides
 * unified tensor lookup across all of them.
 */
export class MultiFileLoader {
  private files: Map<string, SafetensorsFile> = new Map();
  private tensorToFile: Map<string, string> = new Map(); // tensor_name → file_path

  /** Add a safetensors file to the loader (opens + reads header only) */
  addFile(filePath: string, label: string): void {
    const sf = new SafetensorsFile(filePath);
    sf.open();
    this.files.set(label, sf);

    for (const name of sf.listTensors()) {
      this.tensorToFile.set(`${label}:${name}`, label);
    }
  }

  /** Read a tensor by qualified name "label:tensor_name" */
  readTensor(labelAndName: string): ArrayBuffer {
    const colon = labelAndName.indexOf(":");
    const label = labelAndName.substring(0, colon);
    const name = labelAndName.substring(colon + 1);
    const file = this.files.get(label);
    if (!file) throw new Error(`File "${label}" not found`);
    return file.readTensorRaw(name);
  }

  /** Close all files */
  closeAll(): void {
    for (const file of this.files.values()) {
      file.close();
    }
    this.files.clear();
    this.tensorToFile.clear();
  }
}
