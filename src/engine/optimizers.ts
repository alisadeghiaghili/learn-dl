import type { OptimizerName } from '../state/types';
import { addInPlace, scaleInPlace, type Tensor, zeros } from './tensor';

export interface OptimizerState {
  name: OptimizerName;
  lr: number;
  t: number;
  m: Map<string, Tensor>;
  v: Map<string, Tensor>;
}

export function createOptimizer(name: OptimizerName, lr: number): OptimizerState {
  return { name, lr, t: 0, m: new Map(), v: new Map() };
}

/**
 * Apply a gradient step to a named parameter tensor.
 *
 * Args:
 *   opt: optimizer state (mutated)
 *   key: parameter id
 *   param: weights (mutated in place)
 *   grad: gradient wrt param
 */
export function applyGrad(
  opt: OptimizerState,
  key: string,
  param: Tensor,
  grad: Tensor,
): void {
  if (opt.name === 'sgd') {
    scaleInPlace(grad, opt.lr);
    // param -= lr * grad  (grad scaled above)
    for (let i = 0; i < param.data.length; i++) {
      param.data[i] = (param.data[i] ?? 0) - (grad.data[i] ?? 0);
    }
    return;
  }

  if (opt.name === 'momentum' || opt.name === 'rmsprop') {
    let state = opt.m.get(key);
    if (!state) {
      state = zeros(param.shape);
      opt.m.set(key, state);
    }
    for (let i = 0; i < param.data.length; i++) {
      const g = grad.data[i] ?? 0;
      const s = state.data[i] ?? 0;
      if (opt.name === 'momentum') {
        const v = 0.9 * s + g;
        state.data[i] = v;
        param.data[i] = (param.data[i] ?? 0) - opt.lr * v;
      } else {
        const v = 0.9 * s + 0.1 * g * g;
        state.data[i] = v;
        param.data[i] = (param.data[i] ?? 0) - (opt.lr * g) / (Math.sqrt(v) + 1e-8);
      }
    }
    return;
  }

  // adam / adamw
  let m = opt.m.get(key);
  let v = opt.v.get(key);
  if (!m) {
    m = zeros(param.shape);
    opt.m.set(key, m);
  }
  if (!v) {
    v = zeros(param.shape);
    opt.v.set(key, v);
  }
  opt.t += 1;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const bias1 = 1 - Math.pow(beta1, opt.t);
  const bias2 = 1 - Math.pow(beta2, opt.t);
  for (let i = 0; i < param.data.length; i++) {
    const g = grad.data[i] ?? 0;
    const mOld = m.data[i] ?? 0;
    const vOld = v.data[i] ?? 0;
    const mNew = beta1 * mOld + (1 - beta1) * g;
    const vNew = beta2 * vOld + (1 - beta2) * g * g;
    m.data[i] = mNew;
    v.data[i] = vNew;
    const mHat = mNew / bias1;
    const vHat = vNew / bias2;
    const update = (opt.lr * mHat) / (Math.sqrt(vHat) + 1e-8);
    if (opt.name === 'adamw') {
      // simplified decoupled decay
      const decay = 0.01;
      param.data[i] = (param.data[i] ?? 0) * (1 - opt.lr * decay) - update;
    } else {
      param.data[i] = (param.data[i] ?? 0) - update;
    }
  }
}

export function zeroLike(t: Tensor): Tensor {
  return zeros(t.shape);
}

export function accumulate(target: Tensor, src: Tensor): void {
  addInPlace(target, src);
}
