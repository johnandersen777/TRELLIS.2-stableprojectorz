/**
 * Numeric comparison utilities for validation tests.
 *
 * Follows numpy.allclose semantics: |a - b| <= atol + rtol * |b|
 * Tolerances calibrated per dtype:
 *   fp32: rtol=1e-5, atol=1e-7
 *   fp16: rtol=1e-3, atol=1e-5
 *   bf16: rtol=1e-2, atol=1e-3
 *   int:  exact match required
 */

export interface Tolerance {
  rtol: number;
  atol: number;
}

export const TOLERANCES: Record<string, Tolerance> = {
  float32: { rtol: 1e-5, atol: 1e-7 },
  float16: { rtol: 1e-3, atol: 1e-5 },
  bfloat16: { rtol: 1e-2, atol: 1e-3 },
  int32: { rtol: 0, atol: 0 },
  uint32: { rtol: 0, atol: 0 },
};

export interface CompareResult {
  pass: boolean;
  maxErr: number;
  meanErr: number;
  nFailing: number;
  nTotal: number;
  firstFailureIdx: number;
  firstActual: number;
  firstExpected: number;
  statsLine: string;
}

function failResult(reason: string): CompareResult {
  return {
    pass: false,
    maxErr: Infinity,
    meanErr: Infinity,
    nFailing: 0,
    nTotal: 0,
    firstFailureIdx: -1,
    firstActual: 0,
    firstExpected: 0,
    statsLine: `FAIL: ${reason}`,
  };
}

/** Float comparison with rtol+atol (numpy.allclose semantics). */
export function allClose(
  actual: Float32Array | Float64Array,
  expected: Float32Array | Float64Array,
  tol: Tolerance = TOLERANCES.float32,
): CompareResult {
  if (actual.length !== expected.length) {
    return failResult(
      `length mismatch: actual=${actual.length} expected=${expected.length}`,
    );
  }

  let maxErr = 0;
  let sumErr = 0;
  let nFailing = 0;
  let firstFailureIdx = -1;
  let firstActual = 0;
  let firstExpected = 0;

  for (let i = 0; i < actual.length; i++) {
    const err = Math.abs(actual[i] - expected[i]);
    const tolVal = tol.atol + tol.rtol * Math.abs(expected[i]);
    if (err > maxErr) maxErr = err;
    sumErr += err;
    if (err > tolVal) {
      nFailing++;
      if (firstFailureIdx === -1) {
        firstFailureIdx = i;
        firstActual = actual[i];
        firstExpected = expected[i];
      }
    }
  }

  const pass = nFailing === 0;
  const meanErr = sumErr / actual.length;

  return {
    pass,
    maxErr,
    meanErr,
    nFailing,
    nTotal: actual.length,
    firstFailureIdx,
    firstActual,
    firstExpected,
    statsLine: pass
      ? `maxErr=${maxErr.toExponential(2)} meanErr=${meanErr.toExponential(2)} ${nFailing}/${actual.length} fail`
      : `maxErr=${maxErr.toExponential(2)} > tol ${nFailing}/${actual.length} fail first@[${firstFailureIdx}]: act=${firstActual} exp=${firstExpected}`,
  };
}

/** Integer/bool exact match. */
export function exactEqual(
  actual:
    | Int32Array
    | Uint32Array
    | Int16Array
    | Uint16Array
    | Int8Array
    | Uint8Array,
  expected:
    | Int32Array
    | Uint32Array
    | Int16Array
    | Uint16Array
    | Int8Array
    | Uint8Array,
): CompareResult {
  if (actual.length !== expected.length) {
    return failResult(
      `length mismatch: actual=${actual.length} expected=${expected.length}`,
    );
  }

  let nFailing = 0;
  let firstFailureIdx = -1;
  let firstActual = 0;
  let firstExpected = 0;

  for (let i = 0; i < actual.length; i++) {
    if (actual[i] !== expected[i]) {
      nFailing++;
      if (firstFailureIdx === -1) {
        firstFailureIdx = i;
        firstActual = actual[i];
        firstExpected = expected[i];
      }
    }
  }

  const pass = nFailing === 0;
  return {
    pass,
    maxErr: nFailing > 0 ? 1 : 0,
    meanErr: nFailing / actual.length,
    nFailing,
    nTotal: actual.length,
    firstFailureIdx,
    firstActual,
    firstExpected,
    statsLine: pass
      ? `exact match ${actual.length} elements`
      : `${nFailing}/${actual.length} mismatches first@[${firstFailureIdx}]: act=${firstActual} exp=${firstExpected}`,
  };
}

/**
 * For nondeterministic ops (atomicAdd ordering): sort both arrays
 * element-wise before comparing. Handles cases where the multiset
 * of output values is correct but element ordering varies.
 */
export function allCloseSorted(
  actual: Float32Array,
  expected: Float32Array,
  tol: Tolerance = TOLERANCES.float32,
): CompareResult {
  if (actual.length !== expected.length) {
    return failResult(
      `length mismatch: actual=${actual.length} expected=${expected.length}`,
    );
  }
  const a = new Float32Array(actual).sort();
  const e = new Float32Array(expected).sort();
  return allClose(a, e, tol);
}
