/**
 * Sparse tensor types — mirrors Python VarLenTensor / SparseTensor layout.
 *
 * VarLenTensor: variable-length batch via slice-based layout.
 * SparseTensor: extends VarLenTensor with coordinates + spatial metadata.
 */

/** Slice-based batch layout: each batch element occupies [start, stop) in flat feats */
export interface BatchLayout {
  readonly start: number;
  readonly stop: number;
}

export function layoutFromSeqlens(seqlens: number[]): BatchLayout[] {
  const layout: BatchLayout[] = [];
  let start = 0;
  for (const len of seqlens) {
    layout.push({ start, stop: start + len });
    start += len;
  }
  return layout;
}

export function seqlensFromLayout(layout: BatchLayout[]): number[] {
  return layout.map((s) => s.stop - s.start);
}

/** Spatial scale: represented as (numerator, denominator) fractions per dimension */
export interface Scale {
  readonly x: { num: number; den: number };
  readonly y: { num: number; den: number };
  readonly z: { num: number; den: number };
}

export const SCALE_IDENTITY: Scale = {
  x: { num: 1, den: 1 },
  y: { num: 1, den: 1 },
  z: { num: 1, den: 1 },
};

export function scaleToString(scale: Scale): string {
  return `${scale.x.num}/${scale.x.den},${scale.y.num}/${scale.y.den},${scale.z.num}/${scale.z.den}`;
}
