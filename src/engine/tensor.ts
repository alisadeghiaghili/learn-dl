/**
 * Minimal dense tensor helpers for the learn-dl simulation engine.
 *
 * The engine intentionally uses flat Float64Array storage with shape metadata.
 * This is enough for MLP training and light sequence models in the browser.
 */

export interface Tensor {
  data: Float64Array;
  shape: number[];
}

export function tensor(data: number[], shape?: number[]): Tensor {
  const n = data.length;
  const sh = shape ?? [n];
  const expected = sh.reduce((a, b) => a * b, 1);
  if (expected !== n) {
    throw new Error(`tensor shape ${sh.join('x')} does not match data length ${n}`);
  }
  return { data: Float64Array.from(data), shape: sh };
}

export function zeros(shape: number[]): Tensor {
  const n = shape.reduce((a, b) => a * b, 1);
  return { data: new Float64Array(n), shape: [...shape] };
}

export function full(shape: number[], value: number): Tensor {
  const t = zeros(shape);
  t.data.fill(value);
  return t;
}

export function shapeSize(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

export function cloneTensor(t: Tensor): Tensor {
  return { data: Float64Array.from(t.data), shape: [...t.shape] };
}

/** Row-major matmul for 2D tensors: [m,k] @ [k,n] -> [m,n]. */
export function matmul(a: Tensor, b: Tensor): Tensor {
  const [m, k] = [a.shape[0] ?? 0, a.shape[1] ?? 0];
  const [k2, n] = [b.shape[0] ?? 0, b.shape[1] ?? 0];
  if (k !== k2) {
    throw new Error(`matmul shape mismatch: ${a.shape.join('x')} @ ${b.shape.join('x')}`);
  }
  const out = zeros([m, n]);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const aip = a.data[i * k + p] ?? 0;
      if (aip === 0) continue;
      for (let j = 0; j < n; j++) {
        out.data[i * n + j] = (out.data[i * n + j] ?? 0) + aip * (b.data[p * n + j] ?? 0);
      }
    }
  }
  return out;
}

export function transpose2d(t: Tensor): Tensor {
  const [m, n] = [t.shape[0] ?? 0, t.shape[1] ?? 0];
  const out = zeros([n, m]);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      out.data[j * m + i] = t.data[i * n + j] ?? 0;
    }
  }
  return out;
}

export function addBias(t: Tensor, bias: Tensor): Tensor {
  const out = cloneTensor(t);
  const cols = t.shape[1] ?? 1;
  for (let i = 0; i < out.data.length; i++) {
    const b = bias.data[i % cols] ?? 0;
    out.data[i] = (out.data[i] ?? 0) + b;
  }
  return out;
}

export function addInPlace(a: Tensor, b: Tensor): void {
  for (let i = 0; i < a.data.length; i++) {
    a.data[i] = (a.data[i] ?? 0) + (b.data[i] ?? 0);
  }
}

export function scaleInPlace(a: Tensor, s: number): void {
  for (let i = 0; i < a.data.length; i++) {
    a.data[i] = (a.data[i] ?? 0) * s;
  }
}

export function argmaxRows(t: Tensor): number[] {
  const rows = t.shape[0] ?? 0;
  const cols = t.shape[1] ?? 0;
  const out: number[] = [];
  for (let i = 0; i < rows; i++) {
    let best = 0;
    let bestVal = -Infinity;
    for (let j = 0; j < cols; j++) {
      const v = t.data[i * cols + j] ?? -Infinity;
      if (v > bestVal) {
        bestVal = v;
        best = j;
      }
    }
    out.push(best);
  }
  return out;
}

export function oneHot(labels: number[], nClasses: number): Tensor {
  const rows = labels.length;
  const out = zeros([rows, nClasses]);
  for (let i = 0; i < rows; i++) {
    const c = labels[i] ?? 0;
    if (c >= 0 && c < nClasses) {
      out.data[i * nClasses + c] = 1;
    }
  }
  return out;
}

/** Deterministic PRNG (mulberry32). */
export function makeRng(seed: number): () => number {
  let t = seed >>> 0;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
