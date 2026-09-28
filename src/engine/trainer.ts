import { computeLoss } from './losses';
import { backward, forward, type CompiledModel } from './model';
import { applyGrad, createOptimizer, type OptimizerState } from './optimizers';
import { batchSlice, type FullData } from './datasets';
import { zeros, type Tensor } from './tensor';
import type { CompileSpec, EpochPoint, LayerSpec } from '../state/types';
import { compileModel } from './model';

export interface TrainConfig {
  layers: LayerSpec[];
  compile: CompileSpec;
  batch: number;
  epochs: number;
  inputDim: number;
  outputDim: number;
  seed?: number;
}

export interface TrainStepResult {
  history: EpochPoint[];
  weightNorm: number;
}

/**
 * Run mini-batch SGD/Adam for N epochs on synthetic data.
 *
 * Args:
 *   data: full synthetic dataset
 *   config: architecture + compile + loop hyperparams
 *   startEpoch: epoch counter offset (for continued training)
 * Returns:
 *   per-epoch history and final weight L2 norm for UI.
 */
export function train(
  data: FullData,
  config: TrainConfig,
  startEpoch = 0,
): { model: CompiledModel; result: TrainStepResult; data: FullData } {
  const model = compileModel(
    config.layers,
    config.inputDim || data.inputDim,
    config.outputDim || data.nClasses,
    config.seed ?? 7,
  );
  const opt: OptimizerState = createOptimizer(config.compile.optimizer, config.compile.lr);
  const history: EpochPoint[] = [];
  const n = data.x.shape[0] ?? 0;
  const batch = Math.max(1, config.batch);

  for (let epoch = 0; epoch < config.epochs; epoch++) {
    // simple shuffle via cyclic offset
    let lossSum = 0;
    let accSum = 0;
    let batches = 0;
    const offset = (epoch * 17) % Math.max(1, n);

    for (let start = 0; start < n; start += batch) {
      let s = start + offset;
      if (s >= n) s -= n;
      let e = s + batch;
      let bdata = batchSlice(data, s, Math.min(n, e));
      if (s + batch > n) {
        // wrap-around batch is fine for toy data
        bdata = batchSlice(data, s, n);
      }
      if ((bdata.x.shape[0] ?? 0) === 0) continue;

      const { output, cache } = forward(model, bdata.x, true);
      const { loss, grad, accuracy } = computeLoss(config.compile.loss, output, bdata.y);
      const grads = new Map<string, { param: Tensor; g: Tensor }>();
      backward(model, cache, grad, (key, param, g) => {
        const prev = grads.get(key);
        if (!prev) {
          grads.set(key, { param, g: zeros(g.shape) });
          const entry = grads.get(key);
          if (entry) {
            for (let i = 0; i < g.data.length; i++) entry.g.data[i] = g.data[i] ?? 0;
          }
        } else {
          for (let i = 0; i < g.data.length; i++) {
            prev.g.data[i] = (prev.g.data[i] ?? 0) + (g.data[i] ?? 0);
          }
        }
      });
      for (const [key, { param, g }] of grads) {
        applyGrad(opt, key, param, g);
      }
      lossSum += loss;
      accSum += accuracy;
      batches++;
    }

    const point: EpochPoint = {
      epoch: startEpoch + epoch + 1,
      loss: batches ? lossSum / batches : 0,
      accuracy: batches ? accSum / batches : 0,
    };
    history.push(point);
  }

  return {
    model,
    data,
    result: { history, weightNorm: weightNorm(model) },
  };
}

export function evaluate(model: CompiledModel, data: FullData, lossName: CompileSpec['loss']): {
  loss: number;
  accuracy: number;
} {
  const { output } = forward(model, data.x, false);
  const { loss, accuracy } = computeLoss(lossName, output, data.y);
  return { loss, accuracy };
}

export function weightNorm(model: CompiledModel): number {
  let s = 0;
  for (const layer of model.layers) {
    if (layer.dense) {
      for (const v of layer.dense.W.data) s += v * v;
      for (const v of layer.dense.b.data) s += v * v;
    }
    if (layer.gamma) for (const v of layer.gamma.data) s += v * v;
    if (layer.beta) for (const v of layer.beta.data) s += v * v;
  }
  return Math.sqrt(s);
}

/**
 * Predict probabilities/classes on a 2D grid for decision boundary rendering.
 *
 * Args:
 *   model: trained model
 *   lossName: loss used at compile time (for interpreting outputs)
 * Returns:
 *   classification function
 */
export function makePredictor(model: CompiledModel) {
  return (x: number, y: number): number => {
    const input: Tensor = {
      data: Float64Array.from([x, y]),
      shape: [1, 2],
    };
    const { output } = forward(model, input, false);
    const cols = output.shape[1] ?? 1;
    if (cols === 1) {
      return (output.data[0] ?? 0) >= 0.5 ? 1 : 0;
    }
    let best = 0;
    let bestVal = -Infinity;
    for (let j = 0; j < cols; j++) {
      const v = output.data[j] ?? -Infinity;
      if (v > bestVal) {
        bestVal = v;
        best = j;
      }
    }
    return best;
  };
}
