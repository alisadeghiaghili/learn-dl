import {
  activate,
  activateDerivative,
  softmaxRow,
} from './activations';
import {
  addBias,
  cloneTensor,
  matmul,
  makeRng,
  transpose2d,
  zeros,
  type Tensor,
} from './tensor';
import type { LayerSpec } from '../state/types';

export interface DenseParams {
  W: Tensor;
  b: Tensor;
}

export interface LayerRuntime {
  spec: LayerSpec;
  dense?: DenseParams;
  gamma?: Tensor;
  beta?: Tensor;
}

export interface ForwardCache {
  layerId: string;
  kind: string;
  input: Tensor;
  preAct: Tensor;
  output: Tensor;
  probs?: number[][];
  dropped?: Float64Array;
}

export interface CompiledModel {
  layers: LayerRuntime[];
  inputDim: number;
  outputDim: number;
}

/**
 * Build trainable runtime parameters for a layer list.
 *
 * Args:
 *   specs: architecture
 *   inputDim: feature size (vector models)
 *   outputDim: class count / regression size
 *   seed: init seed
 * Returns:
 *   CompiledModel with allocated weights.
 */
export function compileModel(
  specs: LayerSpec[],
  inputDim: number,
  outputDim: number,
  seed = 7,
): CompiledModel {
  const rng = makeRng(seed);
  const layers: LayerRuntime[] = [];
  let prev = inputDim;

  for (const spec of specs) {
    const id = spec.id ?? `L${layers.length}`;
    const s: LayerSpec = { ...spec, id };

    if (s.kind === 'dense') {
      const units = s.units ?? 1;
      const W = zeros([prev, units]);
      const b = zeros([units]);
      const scale = Math.sqrt(2 / Math.max(1, prev));
      for (let i = 0; i < W.data.length; i++) {
        W.data[i] = (rng() * 2 - 1) * scale;
      }
      layers.push({ spec: s, dense: { W, b } });
      prev = units;
    } else if (s.kind === 'dropout') {
      layers.push({ spec: s });
    } else if (s.kind === 'batchnorm' || s.kind === 'layerNorm') {
      const size = prev;
      const gamma = zeros([size]);
      const beta = zeros([size]);
      gamma.data.fill(1);
      layers.push({ spec: s, gamma, beta });
    } else if (s.kind === 'flatten') {
      layers.push({ spec: s });
    } else if (s.kind === 'embedding') {
      const dim = s.embedDim ?? s.units ?? 8;
      const vocab = s.vocabSize ?? 10;
      const W = zeros([vocab, dim]);
      for (let i = 0; i < W.data.length; i++) W.data[i] = (rng() * 2 - 1) * 0.1;
      layers.push({ spec: s, dense: { W, b: zeros([dim]) } });
      prev = dim;
    } else if (s.kind === 'simpleRNN' || s.kind === 'lstm' || s.kind === 'gru') {
      const units = s.units ?? 8;
      const gates = s.kind === 'lstm' ? 4 : s.kind === 'gru' ? 3 : 1;
      const W = zeros([prev + units, units * gates]);
      const b = zeros([units * gates]);
      const scale = Math.sqrt(2 / Math.max(1, prev + units));
      for (let i = 0; i < W.data.length; i++) W.data[i] = (rng() * 2 - 1) * scale;
      layers.push({ spec: s, dense: { W, b } });
      if (!s.returnSequences) prev = units;
    } else if (s.kind === 'attention' || s.kind === 'multiheadAttention') {
      const units = s.units ?? 16;
      layers.push({ spec: s, dense: { W: zeros([prev, units]), b: zeros([units]) } });
      prev = units;
    } else {
      layers.push({ spec: s });
    }
  }

  return { layers, inputDim, outputDim };
}

function forwardDense(
  input: Tensor,
  params: DenseParams,
  activation: LayerSpec['activation'],
): { preAct: Tensor; output: Tensor } {
  const preAct = addBias(matmul(input, params.W), params.b);
  const output = cloneTensor(preAct);
  const rows = preAct.shape[0] ?? 0;
  const cols = preAct.shape[1] ?? 0;
  if (activation === 'softmax') {
    for (let i = 0; i < rows; i++) {
      const row: number[] = [];
      for (let j = 0; j < cols; j++) row.push(preAct.data[i * cols + j] ?? 0);
      const p = softmaxRow(row);
      for (let j = 0; j < cols; j++) output.data[i * cols + j] = p[j] ?? 0;
    }
  } else if (activation && activation !== 'linear') {
    for (let i = 0; i < output.data.length; i++) {
      output.data[i] = activate(activation, output.data[i] ?? 0);
    }
  }
  return { preAct, output };
}

/**
 * Forward pass for vector-shaped batches [batch, features].
 *
 * RNN/attention layers are simplified over a single feature slice for the
 * playground datasets; architecture and APIs still teach the real constructs.
 */
export function forward(
  model: CompiledModel,
  input: Tensor,
  training: boolean,
  rng: () => number = Math.random,
): { output: Tensor; cache: ForwardCache[] } {
  let x = cloneTensor(input);
  const cache: ForwardCache[] = [];

  for (const layer of model.layers) {
    const id = layer.spec.id ?? 'layer';
    const kind = layer.spec.kind;

    if (kind === 'dense' && layer.dense) {
      const { preAct, output } = forwardDense(x, layer.dense, layer.spec.activation);
      cache.push({ layerId: id, kind, input: x, preAct, output });
      x = output;
    } else if (kind === 'dropout') {
      const rate = training ? (layer.spec.rate ?? 0.5) : 0;
      const out = cloneTensor(x);
      const dropped = new Float64Array(x.data.length);
      if (training && rate > 0) {
        for (let i = 0; i < out.data.length; i++) {
          const keep = rng() >= rate ? 1 : 0;
          dropped[i] = keep;
          out.data[i] = (out.data[i] ?? 0) * keep * (1 / (1 - rate));
        }
      } else {
        dropped.fill(1);
      }
      cache.push({ layerId: id, kind, input: x, preAct: x, output: out, dropped });
      x = out;
    } else if (kind === 'batchnorm' || kind === 'layerNorm') {
      const out = cloneTensor(x);
      const cols = x.shape[1] ?? 1;
      for (let j = 0; j < cols; j++) {
        let mean = 0;
        for (let i = 0; i < (x.shape[0] ?? 0); i++) mean += x.data[i * cols + j] ?? 0;
        mean /= Math.max(1, x.shape[0] ?? 1);
        let varr = 0;
        for (let i = 0; i < (x.shape[0] ?? 0); i++) {
          const d = (x.data[i * cols + j] ?? 0) - mean;
          varr += d * d;
        }
        varr /= Math.max(1, x.shape[0] ?? 1);
        const g = layer.gamma?.data[j] ?? 1;
        const b = layer.beta?.data[j] ?? 0;
        for (let i = 0; i < (x.shape[0] ?? 0); i++) {
          const v = ((x.data[i * cols + j] ?? 0) - mean) / Math.sqrt(varr + 1e-5);
          out.data[i * cols + j] = g * v + b;
        }
      }
      cache.push({ layerId: id, kind, input: x, preAct: x, output: out });
      x = out;
    } else if (kind === 'flatten') {
      cache.push({ layerId: id, kind, input: x, preAct: x, output: x });
    } else if (kind === 'embedding' && layer.dense) {
      // treat as gather of first column rounded as token id for toy sequences
      const rows = x.shape[0] ?? 0;
      const dim = layer.dense.W.shape[1] ?? 1;
      const out = zeros([rows, dim]);
      for (let i = 0; i < rows; i++) {
        const tokenId = Math.max(0, Math.min((layer.spec.vocabSize ?? 1) - 1, Math.round(x.data[i] ?? 0)));
        for (let j = 0; j < dim; j++) {
          out.data[i * dim + j] = layer.dense.W.data[tokenId * dim + j] ?? 0;
        }
      }
      cache.push({ layerId: id, kind, input: x, preAct: x, output: out });
      x = out;
    } else if (
      (kind === 'simpleRNN' || kind === 'lstm' || kind === 'gru') &&
      layer.dense
    ) {
      // Simplified: apply the input affine map and a tanh/relu recurrence-free readout.
      const { preAct, output } = forwardDense(x, layer.dense, 'tanh');
      cache.push({ layerId: id, kind, input: x, preAct, output });
      x = output;
    } else if (kind === 'attention' || kind === 'multiheadAttention' || kind === 'positionalEncoding') {
      if (layer.dense) {
        const { preAct, output } = forwardDense(x, layer.dense, layer.spec.activation ?? 'relu');
        cache.push({ layerId: id, kind, input: x, preAct, output });
        x = output;
      } else {
        cache.push({ layerId: id, kind, input: x, preAct: x, output: x });
      }
    } else {
      cache.push({ layerId: id, kind, input: x, preAct: x, output: x });
    }
  }

  return { output: x, cache };
}

export interface BackwardResult {
  dx: Tensor;
}

/**
 * Backprop through cached forward activations. Updates are applied by caller.
 *
 * Args:
 *   model: compiled parameters
 *   cache: forward caches
 *   grad: dL/doutput
 *   accumulate: if provided, grads are written here keyed by `W:${id}` / `b:${id}`
 */
export function backward(
  model: CompiledModel,
  cache: ForwardCache[],
  grad: Tensor,
  accumulate: (key: string, param: Tensor, g: Tensor) => void,
): void {
  let dx = cloneTensor(grad);

  for (let li = cache.length - 1; li >= 0; li--) {
    const c = cache[li];
    if (!c) continue;
    const layer = model.layers[li];
    if (!layer) continue;

    if (c.kind === 'dense' && layer.dense) {
      const act = layer.spec.activation;
      const rows = c.input.shape[0] ?? 0;
      const outDim = c.output.shape[1] ?? 0;

      // activation backward
      const dPre = cloneTensor(dx);
      if (act === 'softmax') {
        // loss already uses softmax CE; treat incoming grad as dlogits
      } else if (act && act !== 'linear') {
        for (let i = 0; i < dPre.data.length; i++) {
          dPre.data[i] =
            (dPre.data[i] ?? 0) *
            activateDerivative(act, c.preAct.data[i] ?? 0, c.output.data[i] ?? 0);
        }
      }

      // grads for W, b
      const dW = zeros(layer.dense.W.shape);
      const dWt = matmul(transpose2d(c.input), dPre);
      for (let i = 0; i < dW.data.length; i++) dW.data[i] = dWt.data[i] ?? 0;
      const db = zeros(layer.dense.b.shape);
      for (let j = 0; j < outDim; j++) {
        let s = 0;
        for (let i = 0; i < rows; i++) s += dPre.data[i * outDim + j] ?? 0;
        db.data[j] = s;
      }
      accumulate(`W:${layer.spec.id}`, layer.dense.W, dW);
      accumulate(`b:${layer.spec.id}`, layer.dense.b, db);

      const dX = matmul(dPre, transpose2d(layer.dense.W));
      dx = dX;
    } else if (c.kind === 'dropout' && c.dropped) {
      const rate = layer.spec.rate ?? 0.5;
      const scale = rate > 0 && rate < 1 ? 1 / (1 - rate) : 1;
      const out = cloneTensor(dx);
      for (let i = 0; i < out.data.length; i++) {
        out.data[i] = (out.data[i] ?? 0) * (c.dropped[i] ?? 0) * scale;
      }
      dx = out;
    } else if (
      (c.kind === 'batchnorm' || c.kind === 'layerNorm') &&
      layer.gamma &&
      layer.beta
    ) {
      // simplified: pass-through + affine grads
      const cols = c.output.shape[1] ?? 1;
      const dg = zeros(layer.gamma.shape);
      const db = zeros(layer.beta.shape);
      for (let j = 0; j < cols; j++) {
        for (let i = 0; i < (c.output.shape[0] ?? 0); i++) {
          dg.data[j] = (dg.data[j] ?? 0) + (dx.data[i * cols + j] ?? 0) * (c.input.data[i * cols + j] ?? 0);
          db.data[j] = (db.data[j] ?? 0) + (dx.data[i * cols + j] ?? 0);
        }
      }
      accumulate(`g:${layer.spec.id}`, layer.gamma, dg);
      accumulate(`b:${layer.spec.id}`, layer.beta, db);
    } else {
      // pass-through
    }
  }
}
