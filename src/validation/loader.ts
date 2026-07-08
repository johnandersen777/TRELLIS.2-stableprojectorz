/**
 * Golden data loader for validation tests.
 *
 * Golden data format:
 *   test_data/<case_name>/
 *     metadata.json   — shapes, dtypes, tolerances
 *     input_<name>.bin — raw row-major bytes (numpy .tofile() format)
 *     expected_<name>.bin
 *
 * The .bin format is zero-overhead: no header, no padding,
 * just raw typed array bytes in little-endian row-major order.
 * Same format as existing .coords.bin + .attrs.bin in reference-images/.
 */

import { Tensor } from "../runtime/tensor.ts";
import type { DType } from "../runtime/device.ts";

// ── Types ─────────────────────────────────────────────────────

export interface GoldenEntry {
  file: string;
  shape: number[];
  dtype: string;
}

export interface GoldenCase {
  name: string;
  tolerance: { rtol: number; atol: number };
  nondeterministic: boolean;
  inputs: GoldenEntry[];
  expected: GoldenEntry[];
  metadata?: Record<string, string>;
}

// ── File loading ──────────────────────────────────────────────

/** Read a binary file into the appropriate TypedArray based on dtype. */
export function readGoldenData(
  caseDir: string,
  entry: GoldenEntry,
): Float32Array | Int32Array | Uint32Array {
  const raw = Deno.readFileSync(`${caseDir}/${entry.file}`);
  const bytes = raw.buffer.slice(
    raw.byteOffset,
    raw.byteOffset + raw.byteLength,
  );

  switch (entry.dtype) {
    case "float32":
      return new Float32Array(bytes);
    case "int32":
      return new Int32Array(bytes);
    case "uint32":
      return new Uint32Array(bytes);
    case "float16": {
      // Load as Uint16Array, caller converts to Float32Array
      const u16 = new Uint16Array(bytes);
      const f32 = new Float32Array(u16.length);
      const u32View = new Uint32Array(f32.buffer);
      for (let i = 0; i < u16.length; i++) {
        u32View[i] = (u16[i] << 16); // BF16→F32
      }
      return f32;
    }
    default:
      return new Float32Array(bytes);
  }
}

/** Load a golden test case from test_data/<path>/metadata.json */
export function loadGolden(casePath: string): GoldenCase {
  const text = Deno.readFileSync(`${casePath}/metadata.json`);
  const decoder = new TextDecoder();
  return JSON.parse(decoder.decode(text)) as GoldenCase;
}

/** Map golden dtype string to runtime DType. */
function toDType(dtype: string): DType {
  switch (dtype) {
    case "float32": return "float32";
    case "float16": return "float16";
    case "bfloat16": return "float16"; // bf16 treated as f16 for GPU
    case "int32": return "int32";
    case "uint32": return "uint32";
    default: return "float32";
  }
}

/** Load inputs as CPU Tensors. */
export function loadInputTensors(
  caseDir: string,
  golden: GoldenCase,
): Tensor[] {
  return golden.inputs.map((entry) => {
    const data = readGoldenData(caseDir, entry);
    const shape = [...entry.shape];
    return Tensor.fromArray(data as Float32Array | Int32Array | Uint32Array, shape, toDType(entry.dtype));
  });
}

/** Discover all golden test case directories under a root. */
export function discoverGoldenCases(rootDir: string): string[] {
  const cases: string[] = [];
  try {
    for (const entry of Deno.readDirSync(rootDir)) {
      if (entry.isDirectory) {
        const subDir = `${rootDir}/${entry.name}`;
        try {
          Deno.statSync(`${subDir}/metadata.json`);
          cases.push(subDir);
        } catch {
          // Recurse deeper
          cases.push(...discoverGoldenCases(subDir));
        }
      }
    }
  } catch {
    // rootDir doesn't exist — no golden cases yet
  }
  return cases;
}
