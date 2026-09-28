import type { ExperimentState, LayerSpec } from '../state/types';
import type { CodeBundle, CodegenOptions } from './types';

const ACT_MAP: Record<string, string> = {
  linear: 'linear',
  relu: 'relu',
  sigmoid: 'sigmoid',
  tanh: 'tanh',
  softmax: 'softmax',
  gelu: 'gelu',
  silu: 'silu',
};

const OPT_MAP: Record<string, string> = {
  sgd: 'sgd',
  momentum: 'sgd', // with momentum in compile
  adam: 'adam',
  adamw: 'adamw',
  rmsprop: 'rmsprop',
};

const LOSS_MAP: Record<string, string> = {
  mse: 'mean_squared_error',
  mae: 'mean_absolute_error',
  binaryCrossentropy: 'binary_crossentropy',
  categoricalCrossentropy: 'categorical_crossentropy',
  sparseCategoricalCrossentropy: 'sparse_categorical_crossentropy',
};

function layerLine(layer: LayerSpec, isSequential: boolean): string {
  const units = layer.units ?? 32;
  switch (layer.kind) {
    case 'dense': {
      const act = layer.activation ? `, activation='${ACT_MAP[layer.activation] ?? layer.activation}'` : '';
      if (isSequential && layer.inputDim) {
        return `keras.layers.Dense(${units}${act}, input_shape=(${layer.inputDim},))`;
      }
      return `keras.layers.Dense(${units}${act})`;
    }
    case 'dropout':
      return `keras.layers.Dropout(${layer.rate ?? 0.5})`;
    case 'batchnorm':
      return 'keras.layers.BatchNormalization()';
    case 'layerNorm':
      return 'keras.layers.LayerNormalization()';
    case 'flatten':
      return 'keras.layers.Flatten()';
    case 'conv2d':
      return `keras.layers.Conv2D(${layer.units ?? 8}, ${layer.kernelSize ?? 3}, activation='relu')`;
    case 'maxpool2d':
      return `keras.layers.MaxPooling2D(${layer.kernelSize ?? 2})`;
    case 'globalAveragePool':
      return 'keras.layers.GlobalAveragePooling2D()';
    case 'embedding':
      return `keras.layers.Embedding(${layer.vocabSize ?? 100}, ${layer.embedDim ?? layer.units ?? 16})`;
    case 'lstm':
      return `keras.layers.LSTM(${units}${layer.returnSequences ? ', return_sequences=True' : ''})`;
    case 'gru':
      return `keras.layers.GRU(${units}${layer.returnSequences ? ', return_sequences=True' : ''})`;
    case 'simpleRNN':
      return `keras.layers.SimpleRNN(${units}${layer.returnSequences ? ', return_sequences=True' : ''})`;
    case 'attention':
      return `keras.layers.Attention()`;
    case 'multiheadAttention':
      return `keras.layers.MultiHeadAttention(num_heads=${layer.heads ?? 4}, key_dim=${units})`;
    case 'positionalEncoding':
      return '# PositionalEncoding (custom layer or keras-nlp) here';
    default:
      return `# ${layer.kind}`;
  }
}

/**
 * Generate Keras (tf.keras) source for an experiment.
 */
export function generateKeras(
  state: ExperimentState,
  options: CodegenOptions = {},
): CodeBundle {
  const ind = options.indent ?? '    ';
  const layers = state.layers;
  const hasInput = layers.some((l) => l.kind === 'dense' && l.inputDim);
  const lines: string[] = [];
  lines.push('import tensorflow as tf');
  lines.push('from tensorflow import keras');
  lines.push('import numpy as np');
  lines.push('');
  lines.push('# --- data ---');
  lines.push(`# dataset: ${state.dataset.id}`);
  lines.push(`X, y = load_dataset("${state.dataset.id}")  # toy loader from learn-dl`);
  lines.push('');
  lines.push('# --- model ---');
  lines.push('model = keras.Sequential([');
  const body = layers.map((l, i) => {
    let line = layerLine(l, !hasInput && i === 0);
    if (i === 0 && !hasInput && state.layers[0]?.kind === 'dense') {
      const units = l.units ?? 32;
      const act = l.activation ? `, activation='${ACT_MAP[l.activation] ?? l.activation}'` : '';
      line = `keras.layers.Dense(${units}${act}, input_shape=(${state.dataset.inputDim ?? 2},))`;
    }
    return `${ind}${line},`;
  });
  if (body.length === 0) {
    body.push(`${ind}keras.layers.Dense(32, activation='relu', input_shape=(${state.dataset.inputDim ?? 2},)),`);
    body.push(`${ind}keras.layers.Dense(${state.dataset.nClasses ?? 2}, activation='softmax'),`);
  }
  lines.push(...body);
  lines.push('])');
  lines.push('');
  lines.push('model.compile(');
  const opt =
    state.compile.optimizer === 'momentum'
      ? `keras.optimizers.SGD(learning_rate=${state.compile.lr}, momentum=0.9)`
      : `keras.optimizers.${OPT_MAP[state.compile.optimizer] ?? 'adam'}(learning_rate=${state.compile.lr})`;
  lines.push(`${ind}optimizer=${opt},`);
  lines.push(`${ind}loss='${LOSS_MAP[state.compile.loss] ?? state.compile.loss}',`);
  lines.push(`${ind}metrics=${JSON.stringify(state.compile.metrics)},`);
  lines.push(')');
  lines.push('');
  lines.push('model.summary()');
  lines.push('');
  lines.push('history = model.fit(');
  lines.push(`${ind}X, y,`);
  lines.push(`${ind}epochs=${Math.max(1, state.training.epochsTrained || 10)},`);
  lines.push(`${ind}batch_size=${state.training.batch || 32},`);
  lines.push(`${ind}validation_split=0.2,`);
  lines.push(')');
  if (options.withData !== false) {
    lines.push('');
    lines.push('def load_dataset(name):');
    lines.push(`${ind}"""Synthetic 2D classifier loader matching learn-dl datasets."""`);
    lines.push(`${ind}rng = np.random.default_rng(7)`);
    lines.push(`${ind}# ... map name -> (X, y)`);
    lines.push(`${ind}return rng.normal(size=(200, 2)), np.eye(2)[rng.integers(0, 2, 200)]`);
  }

  return {
    library: 'keras',
    title: 'Keras / tf.keras',
    language: 'python',
    source: lines.join('\n'),
  };
}
