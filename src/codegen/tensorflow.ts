import type { ExperimentState, LayerSpec } from '../state/types';
import type { CodeBundle, CodegenOptions } from './types';
import { generateKeras } from './keras';

/**
 * Generate low-level TensorFlow source (GradientTape training loop).
 * The architecture can also be expressed with tf.keras — we show the
 * tape style so learners see what Keras hides.
 */
export function generateTensorFlow(
  state: ExperimentState,
  options: CodegenOptions = {},
): CodeBundle {
  const layers = state.layers;
  const lines: string[] = [];
  lines.push('import tensorflow as tf');
  lines.push('');
  lines.push('class Net(tf.Module):');
  lines.push('    def __init__(self):');
  lines.push('        super().__init__()');
  if (layers.length === 0) {
    lines.push(
      `        self.w1 = tf.Variable(tf.random.normal([${state.dataset.inputDim ?? 2}, 32]))`,
    );
    lines.push('        self.b1 = tf.Variable(tf.zeros([32]))');
    lines.push(
      `        self.w2 = tf.Variable(tf.random.normal([32, ${state.dataset.nClasses ?? 2}]))`,
    );
    lines.push(`        self.b2 = tf.Variable(tf.zeros([${state.dataset.nClasses ?? 2}]))`);
    lines.push('');
    lines.push('    def __call__(self, x):');
    lines.push('        h = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)');
    lines.push('        return tf.matmul(h, self.w2) + self.b2');
  } else {
    layers.forEach((layer, i) => {
      lines.push(...layerInit(layer, i, state));
    });
    lines.push('');
    lines.push('    def __call__(self, x):');
    layers.forEach((layer, i) => {
      lines.push(...layerCall(layer, i));
    });
  }

  const lr = state.compile.lr;
  lines.push('');
  lines.push('model = Net()');
  lines.push(
    state.compile.optimizer === 'sgd'
      ? `optimizer = tf.keras.optimizers.SGD(learning_rate=${lr})`
      : state.compile.optimizer === 'momentum'
        ? `optimizer = tf.keras.optimizers.SGD(learning_rate=${lr}, momentum=0.9)`
        : state.compile.optimizer === 'rmsprop'
          ? `optimizer = tf.keras.optimizers.RMSprop(learning_rate=${lr})`
          : state.compile.optimizer === 'adamw'
            ? `optimizer = tf.keras.optimizers.AdamW(learning_rate=${lr})`
            : `optimizer = tf.keras.optimizers.Adam(learning_rate=${lr})`,
  );
  lines.push('');
  lines.push('@tf.function');
  lines.push('def train_step(x, y):');
  lines.push('    with tf.GradientTape() as tape:');
  lines.push('        logits = model(x)');
  lines.push('        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, logits))');
  lines.push('    grads = tape.gradient(loss, model.trainable_variables)');
  lines.push('    optimizer.apply_gradients(zip(grads, model.trainable_variables))');
  lines.push('    return loss');
  lines.push('');
  lines.push(`for epoch in range(${Math.max(1, state.training.epochsTrained || 10)}):`);
  lines.push('    loss = train_step(X, y)');
  lines.push("    print(float(loss))");

  if (options.withData !== false) {
    lines.push('');
    lines.push('# Pair with tf.data.Dataset over the learn-dl synthetic sets.');
  }

  // Also expose keras variant note
  const keras = generateKeras(state, options);
  void keras;

  return {
    library: 'tensorflow',
    title: 'TensorFlow (GradientTape)',
    language: 'python',
    source: lines.join('\n'),
  };
}

function layerInit(layer: LayerSpec, i: number, state: ExperimentState): string[] {
  const inDim = layer.inputDim ?? state.dataset.inputDim ?? 2;
  switch (layer.kind) {
    case 'dense':
      return [
        `        self.w${i} = tf.Variable(tf.random.normal([${i === 0 ? inDim : 'prev'}, ${layer.units ?? 32}]))`,
        `        self.b${i} = tf.Variable(tf.zeros([${layer.units ?? 32}]))`,
      ];
    default:
      return [`        # init ${layer.kind} ${i}`];
  }
}

function layerCall(layer: LayerSpec, i: number): string[] {
  switch (layer.kind) {
    case 'dense': {
      const act = layer.activation ?? 'linear';
      const base = `tf.matmul(x, self.w${i}) + self.b${i}`;
      if (act === 'relu') return [`        x = tf.nn.relu(${base})`];
      if (act === 'sigmoid') return [`        x = tf.sigmoid(${base})`];
      if (act === 'tanh') return [`        x = tf.tanh(${base})`];
      return [`        x = ${base}`];
    }
    case 'dropout':
      return [`        x = tf.nn.dropout(x, rate=${layer.rate ?? 0.5})`];
    default:
      return [`        # call ${layer.kind} ${i}`];
  }
}
