import type { ActivationName } from '../state/types';

/** Activation forward / backward pairs used by the simulation engine. */

export function activate(name: ActivationName, x: number): number {
  switch (name) {
    case 'relu':
      return x > 0 ? x : 0;
    case 'sigmoid':
      return 1 / (1 + Math.exp(-x));
    case 'tanh':
      return Math.tanh(x);
    case 'gelu':
      return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
    case 'silu':
      return x / (1 + Math.exp(-x));
    case 'linear':
    case 'softmax':
      return x;
    default:
      return x;
  }
}

export function activateDerivative(name: ActivationName, x: number, y: number): number {
  switch (name) {
    case 'relu':
      return x > 0 ? 1 : 0;
    case 'sigmoid':
      return y * (1 - y);
    case 'tanh':
      return 1 - y * y;
    case 'gelu': {
      // approximate derivative
      const c = Math.sqrt(2 / Math.PI);
      const t = Math.tanh(c * (x + 0.044715 * x ** 3));
      return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x);
    }
    case 'silu': {
      const s = 1 / (1 + Math.exp(-x));
      return s + x * s * (1 - s);
    }
    case 'linear':
    case 'softmax':
      return 1;
    default:
      return 1;
  }
}

export function softmaxRow(values: number[]): number[] {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / sum);
}
