import type { DatasetSpec } from '../state/types';
import { makeRng, oneHot, type Tensor } from './tensor';

export interface Batch {
  x: Tensor;
  y: Tensor;
  yIndex?: number[];
}

export interface FullData {
  x: Tensor;
  y: Tensor;
  yIndex: number[];
  inputDim: number;
  nClasses: number;
  /** 2D coords for decision-boundary viz when available. */
  points2d?: { x: number; y: number; label: number }[];
}

/**
 * Generate a synthetic dataset for the playground engine.
 *
 * Args:
 *   spec: dataset descriptor
 * Returns:
 *   FullData with features, one-hot targets, and optional 2D points.
 */
export function makeDataset(spec: DatasetSpec): FullData {
  const seed = spec.seed ?? 7;
  const n = spec.nSamples ?? 200;
  const noise = spec.noise ?? 0.05;
  const rng = makeRng(seed);
  const points2d: { x: number; y: number; label: number }[] = [];

  if (spec.id === 'regression') {
    const inputDim = spec.inputDim ?? 1;
    const xData: number[] = [];
    const yData: number[] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let d = 0; d < inputDim; d++) row.push(rng() * 4 - 2);
      const y = row.reduce((a, b) => a + b, 0) + (rng() - 0.5) * noise;
      xData.push(...row);
      yData.push(y);
      if (inputDim >= 2) points2d.push({ x: row[0] ?? 0, y: row[1] ?? 0, label: y > 0 ? 1 : 0 });
    }
    return {
      x: { data: Float64Array.from(xData), shape: [n, inputDim] },
      y: { data: Float64Array.from(yData), shape: [n, 1] },
      yIndex: [],
      inputDim,
      nClasses: 1,
      points2d,
    };
  }

  if (spec.id === 'seq-copy' || spec.id === 'seq-add') {
    const seqLen = spec.seqLen ?? 8;
    const vocab = spec.vocabSize ?? 4;
    const xData: number[] = [];
    const yIndex: number[] = [];
    for (let i = 0; i < n; i++) {
      const tokens: number[] = [];
      for (let t = 0; t < seqLen; t++) tokens.push(Math.floor(rng() * vocab));
      if (spec.id === 'seq-copy') {
        // predict first token
        xData.push(...tokens);
        yIndex.push(tokens[0] ?? 0);
      } else {
        xData.push(...tokens);
        yIndex.push(tokens.reduce((a, b) => a + b, 0) % vocab);
      }
    }
    const y = oneHot(yIndex, vocab);
    return {
      x: { data: Float64Array.from(xData), shape: [n, seqLen] },
      y,
      yIndex,
      inputDim: seqLen,
      nClasses: vocab,
    };
  }

  if (spec.id === 'mnist-toy') {
    // 8x8 flattened “digits” with 3 classes of blob patterns
    const dim = 64;
    const nClasses = spec.nClasses ?? 3;
    const xData: number[] = [];
    const yIndex: number[] = [];
    for (let i = 0; i < n; i++) {
      const label = i % nClasses;
      yIndex.push(label);
      const img = new Array<number>(dim).fill(0);
      const cx = 2 + label * 2;
      const cy = 3 + (label % 2) * 2;
      for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 8; x++) {
          const d = Math.hypot(x - cx, y - cy);
          img[y * 8 + x] = Math.max(0, 1 - d / 3) + rng() * noise;
        }
      }
      xData.push(...img);
    }
    return {
      x: { data: Float64Array.from(xData), shape: [n, dim] },
      y: oneHot(yIndex, nClasses),
      yIndex,
      inputDim: dim,
      nClasses,
    };
  }

  // 2D toy classification
  const nClasses = spec.nClasses ?? (spec.id === 'xor' ? 2 : spec.id === 'spiral' ? 3 : 2);
  const xData: number[] = [];
  const yIndex: number[] = [];
  for (let i = 0; i < n; i++) {
    const label = i % nClasses;
    let px = 0;
    let py = 0;
    if (spec.id === 'xor') {
      const a = rng() > 0.5 ? 1 : 0;
      const b = rng() > 0.5 ? 1 : 0;
      px = a * 1.6 - 0.8 + (rng() - 0.5) * noise;
      py = b * 1.6 - 0.8 + (rng() - 0.5) * noise;
      yIndex.push(a === b ? 0 : 1);
    } else if (spec.id === 'moons') {
      const t = (i % 2) * Math.PI + rng() * 0.5;
      px = Math.cos(t) * 1.2 + (rng() - 0.5) * noise * 2;
      py = Math.sin(t) * 0.7 + (label === 1 ? -0.4 : 0.4) + (rng() - 0.5) * noise * 2;
      yIndex.push(label);
    } else if (spec.id === 'circles') {
      const radius = label === 0 ? 0.5 : 1.2;
      const ang = rng() * Math.PI * 2;
      px = Math.cos(ang) * radius + (rng() - 0.5) * noise * 2;
      py = Math.sin(ang) * radius + (rng() - 0.5) * noise * 2;
      yIndex.push(label);
    } else if (spec.id === 'spiral') {
      const t = (i / n) * 4 * Math.PI + label * 2 + rng() * 0.1;
      const r = (i / n) * 1.5;
      px = r * Math.cos(t) + (rng() - 0.5) * noise;
      py = r * Math.sin(t) + (rng() - 0.5) * noise;
      yIndex.push(label);
    } else {
      // linear
      px = rng() * 4 - 2 + (rng() - 0.5) * noise;
      py = rng() * 4 - 2 + (rng() - 0.5) * noise;
      yIndex.push(px + py > 0 ? 1 : 0);
    }
    xData.push(px, py);
    points2d.push({ x: px, y: py, label: yIndex[yIndex.length - 1] ?? 0 });
  }

  return {
    x: { data: Float64Array.from(xData), shape: [n, 2] },
    y: oneHot(yIndex, nClasses),
    yIndex,
    inputDim: 2,
    nClasses,
    points2d,
  };
}

export function batchSlice(data: FullData, start: number, end: number): Batch {
  const inputDim = data.inputDim;
  const rows = data.x.shape[0] ?? 0;
  const s = Math.max(0, start);
  const e = Math.min(rows, end);
  const xRows = e - s;
  const x = { data: new Float64Array(xRows * inputDim), shape: [xRows, inputDim] };
  for (let i = 0; i < xRows; i++) {
    for (let d = 0; d < inputDim; d++) {
      x.data[i * inputDim + d] = data.x.data[(s + i) * inputDim + d] ?? 0;
    }
  }
  const cols = data.y.shape[1] ?? 1;
  const y = { data: new Float64Array(xRows * cols), shape: [xRows, cols] };
  for (let i = 0; i < xRows; i++) {
    for (let c = 0; c < cols; c++) {
      y.data[i * cols + c] = data.y.data[(s + i) * cols + c] ?? 0;
    }
  }
  return { x, y };
}
