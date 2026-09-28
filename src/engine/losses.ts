import type { LossName } from '../state/types';
import { argmaxRows, type Tensor } from './tensor';
import { softmaxRow } from './activations';

export interface LossResult {
  loss: number;
  /** dL/dlogits, same shape as predictions (logits or probabilities). */
  grad: Tensor;
  accuracy: number;
}

/**
 * Compute loss and gradient for a batch.
 *
 * Args:
 *   name: loss function identifier
 *   pred: raw model outputs (logits for CE family)
 *   target: one-hot targets (or integer labels for sparse)
 * Returns:
 *   LossResult with averaged loss, gradient, and accuracy when applicable.
 */
export function computeLoss(
  name: LossName,
  pred: Tensor,
  target: Tensor,
): LossResult {
  const rows = pred.shape[0] ?? 0;
  const cols = pred.shape[1] ?? 0;
  const grad = { data: new Float64Array(pred.data.length), shape: [...pred.shape] };
  let loss = 0;
  let correct = 0;

  if (name === 'mse' || name === 'mae') {
    for (let i = 0; i < pred.data.length; i++) {
      const p = pred.data[i] ?? 0;
      const t = target.data[i] ?? 0;
      const d = p - t;
      if (name === 'mse') {
        loss += d * d;
        grad.data[i] = (2 * d) / pred.data.length;
      } else {
        loss += Math.abs(d);
        grad.data[i] = (Math.sign(d) || 0) / pred.data.length;
      }
    }
    loss /= Math.max(1, pred.data.length);
    return { loss, grad, accuracy: 0 };
  }

  if (name === 'binaryCrossentropy') {
    for (let i = 0; i < pred.data.length; i++) {
      const p = Math.min(1 - 1e-7, Math.max(1e-7, pred.data[i] ?? 0));
      const t = target.data[i] ?? 0;
      loss += -(t * Math.log(p) + (1 - t) * Math.log(1 - p));
      grad.data[i] = (p - t) / Math.max(1, pred.data.length);
    }
    loss /= Math.max(1, pred.data.length);
    for (let i = 0; i < rows; i++) {
      const p = pred.data[i] ?? 0;
      const t = target.data[i] ?? 0;
      if ((p >= 0.5 ? 1 : 0) === (t >= 0.5 ? 1 : 0)) correct++;
    }
    return { loss, grad, accuracy: rows ? correct / rows : 0 };
  }

  // categorical / sparse categorical cross-entropy (softmax)
  const targets: number[][] = [];
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) row.push(target.data[i * cols + j] ?? 0);
    targets.push(row);
  }

  for (let i = 0; i < rows; i++) {
    const logits: number[] = [];
    for (let j = 0; j < cols; j++) logits.push(pred.data[i * cols + j] ?? 0);
    const probs = softmaxRow(logits);
    let classIdx = 0;
    if (name === 'sparseCategoricalCrossentropy') {
      classIdx = Math.round(targets[i]?.[0] ?? 0);
    } else {
      let best = 0;
      let bestVal = -Infinity;
      for (let j = 0; j < cols; j++) {
        const v = targets[i]?.[j] ?? -Infinity;
        if (v > bestVal) {
          bestVal = v;
          best = j;
        }
      }
      classIdx = best;
    }
    const p = Math.min(1 - 1e-7, Math.max(1e-7, probs[classIdx] ?? 1e-7));
    loss += -Math.log(p);
    for (let j = 0; j < cols; j++) {
      const g = ((probs[j] ?? 0) - (j === classIdx ? 1 : 0)) / Math.max(1, rows);
      grad.data[i * cols + j] = g;
    }
  }
  loss /= Math.max(1, rows);
  const predClasses = argmaxRows(pred);
  for (let i = 0; i < rows; i++) {
    let classIdx = 0;
    if (name === 'sparseCategoricalCrossentropy') {
      classIdx = Math.round(targets[i]?.[0] ?? 0);
    } else {
      let best = 0;
      let bestVal = -Infinity;
      for (let j = 0; j < cols; j++) {
        const v = targets[i]?.[j] ?? -Infinity;
        if (v > bestVal) {
          bestVal = v;
          best = j;
        }
      }
      classIdx = best;
    }
    if (predClasses[i] === classIdx) correct++;
  }
  return { loss, grad, accuracy: rows ? correct / rows : 0 };
}

export function predictClasses(pred: Tensor): number[] {
  return argmaxRows(pred);
}
