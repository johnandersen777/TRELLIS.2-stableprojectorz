/**
 * Test harness: wraps Deno.test() with golden-data comparison.
 *
 * Usage:
 *   goldenTest({
 *     casePath: "test_data/ops/matmul/small_f32",
 *     opFn: async (a, b) => { ... return c; },
 *   });
 *
 * Auto-discover all cases:
 *   for (const caseDir of discoverGoldenCases("test_data/ops/matmul")) {
 *     goldenTest({ casePath: caseDir, opFn: matmulWrapper });
 *   }
 */

import { Tensor } from "../runtime/tensor.ts";
import {
  loadGolden,
  loadInputTensors,
  readGoldenData,
  type GoldenCase,
} from "./loader.ts";
import { allClose, exactEqual, allCloseSorted, TOLERANCES, type Tolerance } from "./compare.ts";

// ── Re-export for convenience ─────────────────────────────────

export { loadGolden, loadInputTensors, readGoldenData, discoverGoldenCases } from "./loader.ts";
export { allClose, exactEqual, allCloseSorted, TOLERANCES } from "./compare.ts";
export type { CompareResult, Tolerance } from "./compare.ts";
export type { GoldenCase, GoldenEntry } from "./loader.ts";

// ── Test spec ─────────────────────────────────────────────────

export interface RunSpec {
  /** Path to golden test case directory (e.g. "test_data/ops/matmul/small_f32") */
  casePath: string;
  /** Function that runs the op/model under test */
  opFn: (...args: Tensor[]) => Promise<Tensor> | Tensor;
  /** Override tolerance from metadata.json */
  tolerance?: Tolerance;
  /** Override nondeterministic flag */
  nondeterministic?: boolean;
}

// ── Helper ────────────────────────────────────────────────────

function dtypeToTol(dtype: string): Tolerance {
  switch (dtype) {
    case "float32":
      return TOLERANCES.float32;
    case "float16":
      return TOLERANCES.float16;
    case "bfloat16":
      return TOLERANCES.bfloat16;
    case "int32":
    case "uint32":
      return TOLERANCES.int32;
    default:
      return TOLERANCES.float32;
  }
}

function isIntegerDtype(dtype: string): boolean {
  return dtype === "int32" || dtype === "uint32" ||
    dtype === "int16" || dtype === "uint16" ||
    dtype === "int8" || dtype === "uint8";
}

async function getTensorData(t: Tensor): Promise<Float32Array | Int32Array> {
  if (t.device === "gpu") {
    await t.toCPU();
  }
  return t.getView() as Float32Array;
}

// ── Main test wrapper ─────────────────────────────────────────

/**
 * Register a Deno.test() that loads golden data, runs the op,
 * and compares output against expected values.
 */
export function goldenTest(spec: RunSpec): void {
  const name = spec.casePath
    .replace(/^test_data[\/\\]/, "")
    .replace(/\\/g, "/");

  Deno.test(`golden::${name}`, async () => {
    const golden = loadGolden(spec.casePath);
    const tol = spec.tolerance ?? golden.tolerance;
    const isNondeterministic = spec.nondeterministic ??
      golden.nondeterministic;
    const inputs = loadInputTensors(spec.casePath, golden);

    // Run the op
    let actual: Tensor;
    try {
      const result = spec.opFn(...inputs);
      actual = result instanceof Promise ? await result : result;
    } finally {
      // Clean up inputs
      for (const t of inputs) t.dispose();
    }

    const actualData = await getTensorData(actual);

    // Validate against each expected output
    for (let ei = 0; ei < golden.expected.length; ei++) {
      const exp = golden.expected[ei];
      const expectedData = readGoldenData(spec.casePath, exp);
      const expTol = tol ?? dtypeToTol(exp.dtype);

      let result;
      if (isIntegerDtype(exp.dtype)) {
        result = exactEqual(
          actualData as Int32Array,
          expectedData as Int32Array,
        );
      } else if (isNondeterministic) {
        result = allCloseSorted(
          actualData as Float32Array,
          expectedData as Float32Array,
          expTol,
        );
      } else {
        result = allClose(
          actualData as Float32Array,
          expectedData as Float32Array,
          expTol,
        );
      }

      if (!result.pass) {
        const msg =
          `${name} [expected[${ei}] "${exp.file}"] FAIL: ${result.statsLine}`;
        throw new Error(msg);
      }

      console.log(`  PASS ${name} [${exp.file}]: ${result.statsLine}`);
    }

    actual.dispose();
  });
}

/**
 * Run a CPU-side comparison without GPU context.
 * For testing pure-JS functions (mesh pipeline, safetensors parsing, etc.).
 */
export function cpuGoldenTest(
  name: string,
  casePath: string,
  fn: () => { actual: Float32Array | Int32Array; expected: Float32Array | Int32Array; dtype: string },
): void {
  Deno.test(`golden::${name}`, () => {
    const { actual, expected, dtype } = fn();
    const tol = dtypeToTol(dtype);
    let result;
    if (isIntegerDtype(dtype)) {
      result = exactEqual(
        actual as Int32Array,
        expected as Int32Array,
      );
    } else {
      result = allClose(
        actual as Float32Array,
        expected as Float32Array,
        tol,
      );
    }
    if (!result.pass) {
      throw new Error(`${name} FAIL: ${result.statsLine}`);
    }
    console.log(`  PASS ${name}: ${result.statsLine}`);
  });
}
