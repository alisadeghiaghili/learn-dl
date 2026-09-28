/**
 * Canonical experiment and model types for learn-dl.
 *
 * These types are the single source of truth shared by the engine, DSL,
 * codegen, code parser, visuals, and levels.
 */

export type LibraryId = 'pytorch' | 'tensorflow' | 'keras';

export type ActivationName =
  | 'linear'
  | 'relu'
  | 'sigmoid'
  | 'tanh'
  | 'softmax'
  | 'gelu'
  | 'silu';

export type LayerKind =
  | 'input'
  | 'dense'
  | 'dropout'
  | 'batchnorm'
  | 'flatten'
  | 'conv2d'
  | 'maxpool2d'
  | 'embedding'
  | 'lstm'
  | 'gru'
  | 'simpleRNN'
  | 'attention'
  | 'multiheadAttention'
  | 'layerNorm'
  | 'positionalEncoding'
  | 'globalAveragePool';

export interface LayerSpec {
  kind: LayerKind;
  /** Unique within the architecture; auto-assigned if omitted. */
  id?: string;
  /** Dense / embedding / conv filters / RNN units / attention dim. */
  units?: number;
  /** Conv kernel size (square) or attention heads. */
  kernelSize?: number;
  stride?: number;
  padding?: number;
  /** Attention heads for multi-head layers. */
  heads?: number;
  /** Dropout rate. */
  rate?: number;
  activation?: ActivationName;
  /** Dense input features (optional; inferred from dataset). */
  inputDim?: number;
  /** Embedding vocabulary size. */
  vocabSize?: number;
  /** Embedding output dim. */
  embedDim?: number;
  /** Sequence length hint for RNN/attention. */
  seqLen?: number;
  /** Return sequences for stacked RNNs. */
  returnSequences?: boolean;
  useBias?: boolean;
}

export type OptimizerName = 'sgd' | 'momentum' | 'adam' | 'adamw' | 'rmsprop';
export type LossName =
  | 'mse'
  | 'mae'
  | 'binaryCrossentropy'
  | 'categoricalCrossentropy'
  | 'sparseCategoricalCrossentropy';
export type MetricName = 'accuracy' | 'mse' | 'mae';

export type DatasetId =
  | 'xor'
  | 'moons'
  | 'circles'
  | 'spiral'
  | 'linear'
  | 'regression'
  | 'seq-copy'
  | 'seq-add'
  | 'mnist-toy';

export interface DatasetSpec {
  id: DatasetId;
  /** Synthetic generator seed for reproducibility. */
  seed?: number;
  nSamples?: number;
  noise?: number;
  nClasses?: number;
  inputDim?: number;
  seqLen?: number;
  vocabSize?: number;
}

export interface CompileSpec {
  optimizer: OptimizerName;
  lr: number;
  loss: LossName;
  metrics: MetricName[];
}

export interface EpochPoint {
  epoch: number;
  loss: number;
  accuracy?: number;
  valLoss?: number;
  valAccuracy?: number;
}

export interface TrainingSpec {
  epochsTrained: number;
  batch: number;
  history: EpochPoint[];
  /** Floating-point snapshot of trainable params (engine-managed). */
  weightNorm?: number;
}

export interface ExperimentState {
  library: LibraryId;
  dataset: DatasetSpec;
  layers: LayerSpec[];
  compile: CompileSpec;
  training: TrainingSpec;
}

export const DEFAULT_COMPILE: CompileSpec = {
  optimizer: 'adam',
  lr: 0.05,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
};

export const DEFAULT_DATASET: DatasetSpec = {
  id: 'xor',
  seed: 7,
  nSamples: 200,
  noise: 0.05,
  nClasses: 2,
  inputDim: 2,
};

export function emptyExperiment(): ExperimentState {
  return {
    library: 'keras',
    dataset: { ...DEFAULT_DATASET },
    layers: [],
    compile: { ...DEFAULT_COMPILE, metrics: [...DEFAULT_COMPILE.metrics] },
    training: { epochsTrained: 0, batch: 32, history: [] },
  };
}

export function cloneExperiment(state: ExperimentState): ExperimentState {
  return {
    library: state.library,
    dataset: { ...state.dataset },
    layers: state.layers.map((l) => ({ ...l })),
    compile: { ...state.compile, metrics: [...state.compile.metrics] },
    training: {
      ...state.training,
      history: state.training.history.map((h) => ({ ...h })),
    },
  };
}

export function layerLabel(layer: LayerSpec): string {
  switch (layer.kind) {
    case 'dense':
      return `Dense(${layer.units ?? '?'})`;
    case 'dropout':
      return `Dropout(${layer.rate ?? 0.5})`;
    case 'batchnorm':
      return 'BatchNorm';
    case 'layerNorm':
      return 'LayerNorm';
    case 'flatten':
      return 'Flatten';
    case 'conv2d':
      return `Conv2D(${layer.units ?? 8}, ${layer.kernelSize ?? 3})`;
    case 'maxpool2d':
      return `MaxPool2D(${layer.kernelSize ?? 2})`;
    case 'globalAveragePool':
      return 'GlobalAvgPool';
    case 'embedding':
      return `Embedding(${layer.vocabSize ?? '?'}, ${layer.embedDim ?? '?'})`;
    case 'lstm':
      return `LSTM(${layer.units ?? 32})`;
    case 'gru':
      return `GRU(${layer.units ?? 32})`;
    case 'simpleRNN':
      return `SimpleRNN(${layer.units ?? 32})`;
    case 'attention':
      return `Attention(dim=${layer.units ?? 32})`;
    case 'multiheadAttention':
      return `MHA(heads=${layer.heads ?? 4}, dim=${layer.units ?? 32})`;
    case 'positionalEncoding':
      return 'PositionalEncoding';
    case 'input':
      return `Input(${layer.inputDim ?? '?'})`;
    default:
      return layer.kind;
  }
}

export function activationSuffix(layer: LayerSpec): string {
  const a = layer.activation;
  if (!a || a === 'linear') return '';
  return ` + ${a}`;
}

export function countParams(layers: LayerSpec[], inputDim: number): number {
  let prev = inputDim;
  let total = 0;
  for (const layer of layers) {
    switch (layer.kind) {
      case 'dense': {
        const units = layer.units ?? 1;
        total += prev * units + (layer.useBias === false ? 0 : units);
        prev = units;
        break;
      }
      case 'embedding': {
        const vocab = layer.vocabSize ?? 1;
        const dim = layer.embedDim ?? layer.units ?? 8;
        total += vocab * dim;
        prev = dim;
        break;
      }
      case 'lstm':
      case 'gru':
      case 'simpleRNN': {
        const units = layer.units ?? 1;
        // rough gate parameterization
        const gates = layer.kind === 'lstm' ? 4 : layer.kind === 'gru' ? 3 : 1;
        total += gates * (prev * units + units * units + units);
        prev = layer.returnSequences ? prev : units;
        if (layer.returnSequences) prev = units;
        break;
      }
      case 'conv2d': {
        const k = layer.kernelSize ?? 3;
        const out = layer.units ?? 1;
        total += k * k * prev * out + (layer.useBias === false ? 0 : out);
        prev = out;
        break;
      }
      default:
        break;
    }
  }
  return total;
}
