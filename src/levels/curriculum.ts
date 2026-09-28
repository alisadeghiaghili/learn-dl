/**
 * Full learn-dl curriculum.
 *
 * Each level teaches one idea and checks it with architecture match and/or
 * metric asserts. Solution commands use the DSL; solutionCode shows the same
 * idea in PyTorch / TensorFlow / Keras.
 */
import type { Level } from './types';
import {
  alert,
  codeCompare,
  demo,
  goal,
  makeLevel,
  startFrom,
  T,
} from './factory';

const threeLibs = (title: string) =>
  codeCompare(
    [
      title,
      '',
      'Use the Code panel tabs to switch between **PyTorch**, **TensorFlow**, and **Keras**.',
      'The same experiment state generates all three — concept first, API second.',
    ],
  );

export const levels: Level[] = [
  // ─── foundations ───────────────────────────────────────────────
  makeLevel({
    id: 'foundations-shape-sense',
    sequence: 'foundations',
    name: T('Shape Sense', 'حس شکل تنسور'),
    objective: T('Build a tensor mental model: shape, rank, features.', 'مدل ذهنی تنسور: شکل، رتبه، ویژگی'),
    hint: T(
      'Think of X as [nSamples, features]. A Dense layer maps features → units.',
      'X را [nSamples, features] ببین. لایه Dense ویژگی‌ها را به units می‌برد.',
    ),
    startState: startFrom({ dataset: { id: 'xor', inputDim: 2, nClasses: 2 } }),
    goal: goal({
      dataset: 'xor',
      asserts: [
        (s) => s.layers.length >= 1 && s.layers[0]?.kind === 'dense' && (s.layers[0]?.inputDim ?? 2) === 2,
      ],
    }),
    solutionCommand: 'data xor; net dense 8 relu',
    pureConcept: true,
    golf: { bestCommands: 2 },
    startDialog: [
      alert(
        '## Tensors are just shaped numbers',
        'A batch of data is a matrix `[nSamples, features]`. XOR has 2 features (`x1`, `x2`) and 2 classes.',
        'Every layer is a shape transform. Dense(units) maps `features → units`.',
      ),
      demo(['Add one Dense layer and watch the shape badge update.'], 'net dense 8 relu', [
        'The graph node shows `8` units and the parameter count updates.',
      ]),
      threeLibs('Same idea in three frameworks'),
    ],
  }),

  makeLevel({
    id: 'foundations-perceptron',
    sequence: 'foundations',
    name: T('Perceptron Gate', 'گیت پرسپترون'),
    objective: T('A single Dense unit + sigmoid can model one half-space.', 'یک نورون Dense + سیگموید'),
    hint: T('Use `net dense 1 sigmoid` and train a few epochs on linear data.', 'با net dense 1 sigmoid روی داده خطی ترین کن.'),
    startState: startFrom({
      dataset: { id: 'linear', inputDim: 2, nClasses: 2, noise: 0.05 },
      compile: { optimizer: 'sgd', lr: 0.1, loss: 'binaryCrossentropy', metrics: ['accuracy'] },
    }),
    goal: goal({
      dataset: 'linear',
      layers: [{ kind: 'dense', units: 1, activation: 'sigmoid', inputDim: 2 }],
      loss: 'binaryCrossentropy',
    }),
    solutionCommand: 'net dense 1 sigmoid; loss bce; train 20',
    golf: { bestCommands: 3 },
    startDialog: [
      alert(
        '## The oldest network in the book',
        'A perceptron computes `σ(w·x + b)` and outputs a probability. That is enough when data is linearly separable.',
      ),
      demo(['Build the perceptron'], 'net dense 1 sigmoid', ['One output unit. Sigmoid squashes to (0, 1).']),
    ],
  }),

  makeLevel({
    id: 'foundations-autograd-story',
    sequence: 'foundations',
    name: T('Why Backprop Works', 'چرا backprop کار می‌کند'),
    objective: T('Loss → gradient → parameter update.', 'loss → گرادیان → به‌روزرسانی'),
    hint: T('Pick MSE, train, and watch the loss curve fall.', 'mse انتخاب کن و منحنی loss را ببین.'),
    startState: startFrom({
      dataset: { id: 'regression', inputDim: 1, nClasses: 1, nSamples: 120 },
      compile: { optimizer: 'sgd', lr: 0.05, loss: 'mse', metrics: ['mse'] },
    }),
    goal: goal({
      loss: 'mse',
      metrics: { epochsAtLeast: 5, lossAtMost: 0.5 },
    }),
    solutionCommand: 'loss mse; train 30',
    golf: { bestCommands: 2 },
    startDialog: [
      alert(
        '## Autograd in one sentence',
        'Forward builds a graph of ops; backward multiplies local derivatives to get `dLoss/dParam`.',
        'Optimizers then step parameters against that gradient.',
      ),
      threeLibs('autograd.grad · GradientTape · tape inside train_step'),
    ],
  }),

  // ─── mlp ───────────────────────────────────────────────────────
  makeLevel({
    id: 'mlp-xor-need-depth',
    sequence: 'mlp',
    name: T('XOR Needs Depth', 'XOR به عمق نیاز دارد'),
    objective: T('One hidden layer breaks the XOR half-space limit.', 'یک لایه پنهان محدودیت XOR را می‌شکند'),
    hint: T('`net dense 8 relu` then `net dense 2 softmax` and train on xor.', 'dense 8 relu و dense 2 softmax و ترین روی xor'),
    startState: startFrom({
      dataset: { id: 'xor', inputDim: 2, nClasses: 2 },
      compile: { optimizer: 'adam', lr: 0.05, loss: 'categoricalCrossentropy', metrics: ['accuracy'] },
    }),
    goal: goal({
      dataset: 'xor',
      layers: [
        { kind: 'dense', units: 8, activation: 'relu', inputDim: 2 },
        { kind: 'dense', units: 2, activation: 'softmax' },
      ],
      metrics: { accuracyAtLeast: 0.9 },
    }),
    solutionCommand: 'net dense 8 relu; net dense 2 softmax; train 40',
    golf: { bestCommands: 3 },
    startDialog: [
      alert(
        '## Linear models cannot XOR',
        'XOR is the classic non-linear toy. A single Dense unit draws one line — not enough.',
        'Stack one hidden Dense + a softmax head and the network can carve the regions.',
      ),
      demo(['Add hidden layer + classifier head'], 'net dense 8 relu; net dense 2 softmax', [
        'Hidden `8` with relu. Head `2` with softmax over classes.',
      ]),
    ],
  }),

  makeLevel({
    id: 'mlp-activation-zoo',
    sequence: 'mlp',
    name: T('Activation Zoo', 'باغ‌وحش توابع فعال‌سازی'),
    objective: T('Use a non-linear hidden activation and a softmax head.', 'فعال‌سازی غیرخطی و خروجی softmax'),
    hint: T('relu, tanh, gelu, or silu on the hidden layer; softmax on the head.', 'در لایه پنهان relu/tanh/gelu/silu و در خروجی softmax'),
    startState: startFrom({ dataset: { id: 'moons', inputDim: 2, nClasses: 2 } }),
    goal: goal({
      asserts: [
        (s) =>
          s.layers.length === 2 &&
          s.layers[0]?.kind === 'dense' &&
          s.layers[0]?.activation !== 'linear' &&
          s.layers[0]?.activation !== undefined &&
          s.layers[1]?.activation === 'softmax',
      ],
      metrics: { accuracyAtLeast: 0.85 },
    }),
    solutionCommand: 'net dense 16 tanh; net dense 2 softmax; train 50',
    golf: { bestCommands: 3 },
  }),

  makeLevel({
    id: 'mlp-loss-match',
    sequence: 'mlp',
    name: T('Loss Must Match the Head', 'loss باید با خروجی جور باشد'),
    objective: T('softmax head ⇒ categoricalCrossentropy.', 'softmax ⇒ categoricalCrossentropy'),
    hint: T('Pair softmax with categorical CE (or sigmoid + BCE).', 'softmax با CE یا sigmoid با BCE'),
    startState: startFrom({ dataset: { id: 'xor', inputDim: 2, nClasses: 2 } }),
    goal: goal({
      layers: [
        { kind: 'dense', units: 8, activation: 'relu', inputDim: 2 },
        { kind: 'dense', units: 2, activation: 'softmax' },
      ],
      loss: 'categoricalCrossentropy',
    }),
    solutionCommand: 'net dense 8 relu; net dense 2 softmax; loss ce; train 20',
    golf: { bestCommands: 4 },
  }),

  makeLevel({
    id: 'mlp-optimizer-choice',
    sequence: 'mlp',
    name: T('Pick an Optimizer', 'بهینه‌ساز را انتخاب کن'),
    objective: T('Use Adam and a sensible learning rate.', 'Adam و نرخ یادگیری معقول'),
    hint: T('`optimizer adam` and `lr 0.05` then train.', 'optimizer adam و lr 0.05'),
    startState: startFrom({ dataset: { id: 'moons', inputDim: 2, nClasses: 2 } }),
    goal: goal({
      optimizer: 'adam',
      layers: [
        { kind: 'dense', units: 16, activation: 'relu', inputDim: 2 },
        { kind: 'dense', units: 2, activation: 'softmax' },
      ],
      metrics: { accuracyAtLeast: 0.85, epochsAtLeast: 1 },
    }),
    solutionCommand: 'net dense 16 relu; net dense 2 softmax; optimizer adam; lr 0.05; train 40',
    golf: { bestCommands: 5 },
    startDialog: [
      alert(
        '## SGD vs Adam',
        'SGD is honest and transparent. Adam adapts step sizes per-parameter and is the default you reach for.',
        'Learning rate still matters — too large diverges, too small crawls.',
      ),
    ],
  }),

  makeLevel({
    id: 'mlp-full-stack',
    sequence: 'mlp',
    name: T('Full MLP Stack', 'پشته‌ی کامل MLP'),
    objective: T('Two hidden layers + softmax + train to high accuracy.', 'دو لایه پنهان + softmax + دقت بالا'),
    hint: T('wide → wide → head, then train on circles.', 'لایه‌های پنهان پهن و سپس سر خروجی'),
    startState: startFrom({ dataset: { id: 'circles', inputDim: 2, nClasses: 2 } }),
    goal: goal({
      layers: [
        { kind: 'dense', units: 32, activation: 'relu', inputDim: 2 },
        { kind: 'dense', units: 16, activation: 'relu' },
        { kind: 'dense', units: 2, activation: 'softmax' },
      ],
      metrics: { accuracyAtLeast: 0.9 },
    }),
    solutionCommand: 'net dense 32 relu; net dense 16 relu; net dense 2 softmax; train 60',
    golf: { bestCommands: 4 },
  }),

  // ─── regularization ────────────────────────────────────────────
  makeLevel({
    id: 'reg-dropout',
    sequence: 'regularization',
    name: T('Dropout Discipline', 'نظم دروپ‌اوت'),
    objective: T('Insert Dropout between hidden layers.', 'دروپ‌اوت بین لایه‌های پنهان'),
    hint: T('`net dropout 0.3` after each hidden Dense.', 'بعد از هر dense پنهان dropout'),
    startState: startFrom({ dataset: { id: 'spiral', inputDim: 2, nClasses: 3, noise: 0.15 } }),
    goal: goal({
      asserts: [
        (s) => s.layers.filter((l) => l.kind === 'dropout').length >= 1,
        (s) => s.layers.some((l) => l.kind === 'dense' && (l.units ?? 0) >= 32),
      ],
      metrics: { accuracyAtLeast: 0.7 },
    }),
    solutionCommand: 'net dense 64 relu; net dropout 0.3; net dense 2 softmax; train 50',
    // note: head should be 3 for spiral — fixed in solution below
    golf: { bestCommands: 4 },
    startDialog: [
      alert(
        '## Regularization fights overfitting',
        'Dropout randomly zeros activations during training so the net cannot rely on one pathway.',
        'At inference everything is active (scaled).',
      ),
    ],
  }),

  makeLevel({
    id: 'reg-batchnorm',
    sequence: 'regularization',
    name: T('Normalize the Activations', 'نرمال‌سازی فعال‌سازی‌ها'),
    objective: T('Place BatchNorm after a hidden Dense.', 'BatchNorm بعد از Dense پنهان'),
    hint: T('`net batchnorm`', 'net batchnorm'),
    startState: startFrom({ dataset: { id: 'moons', inputDim: 2, nClasses: 2 } }),
    goal: goal({
      asserts: [(s) => s.layers.some((l) => l.kind === 'batchnorm')],
    }),
    solutionCommand: 'net dense 32 relu; net batchnorm; net dense 2 softmax; train 20',
    golf: { bestCommands: 4 },
  }),

  makeLevel({
    id: 'reg-layernorm-transformers',
    sequence: 'regularization',
    name: T('LayerNorm (Transformer Style)', 'LayerNorm به سبک ترنسفورمر'),
    objective: T('Use LayerNorm instead of BatchNorm for sequence models.', 'به‌جای BatchNorm از LayerNorm استفاده کن'),
    hint: T('`net layernorm`', 'net layernorm'),
    startState: startFrom({ dataset: { id: 'seq-copy', inputDim: 8, nClasses: 4, seqLen: 8, vocabSize: 4 } }),
    goal: goal({
      asserts: [(s) => s.layers.some((l) => l.kind === 'layerNorm')],
    }),
    solutionCommand: 'net embedding 4 16; net layernorm; net dense 4 softmax; train 10',
    pureConcept: true,
    golf: { bestCommands: 4 },
  }),

  makeLevel({
    id: 'reg-deeper-still-generalizes',
    sequence: 'regularization',
    name: T('Deeper but Still Generalizes', 'عمیق‌تر ولی هنوز general'),
    objective: T('Depth + dropout on noisy spiral.', 'عمق + دروپ‌اوت روی spiral پرنویز'),
    hint: T('Two hidden layers with dropout, then train hard.', 'دو لایه پنهان با دروپ‌اوت'),
    startState: startFrom({ dataset: { id: 'spiral', inputDim: 2, nClasses: 3, noise: 0.2, nSamples: 300 } }),
    goal: goal({
      asserts: [
        (s) => s.layers.filter((l) => l.kind === 'dense').length >= 3,
        (s) => s.layers.filter((l) => l.kind === 'dropout').length >= 1,
      ],
      metrics: { accuracyAtLeast: 0.65 },
    }),
    solutionCommand:
      'data spiral; net dense 64 relu; net dropout 0.3; net dense 32 relu; net dropout 0.2; net dense 3 softmax; train 80',
    golf: { bestCommands: 6 },
  }),

  // ─── cnn ───────────────────────────────────────────────────────
  makeLevel({
    id: 'cnn-conv-first',
    sequence: 'cnn',
    name: T('First Convolution', 'اولین کانولوشن'),
    objective: T('Add Conv2D as the first feature extractor.', 'Conv2D به‌عنوان اولین استخراج‌کننده ویژگی'),
    hint: T('`net conv 8 3`', 'net conv 8 3'),
    startState: startFrom({
      dataset: { id: 'mnist-toy', inputDim: 64, nClasses: 3, nSamples: 180 },
    }),
    goal: goal({
      asserts: [(s) => s.layers[0]?.kind === 'conv2d'],
    }),
    solutionCommand: 'data mnist-toy; net conv 8 3; net flatten; net dense 3 softmax; train 15',
    golf: { bestCommands: 4 },
    startDialog: [
      alert(
        '## Local filters, shared weights',
        'A conv filter slides across the input and detects a local motif — an edge, a blob, a corner.',
        'Weight sharing means the same detector works everywhere in the image.',
      ),
    ],
  }),

  makeLevel({
    id: 'cnn-pool-stack',
    sequence: 'cnn',
    name: T('Conv → Pool → Flatten → Head', 'کانولوشن → pool → flatten → سر'),
    objective: T('Build a tiny LeNet-style stack.', 'یک پشته کوچک شبیه LeNet'),
    hint: T('conv, pool, flatten, dense head.', 'conv, pool, flatten, dense'),
    startState: startFrom({ dataset: { id: 'mnist-toy', inputDim: 64, nClasses: 3, nSamples: 180 } }),
    goal: goal({
      layers: [
        { kind: 'conv2d', units: 8, kernelSize: 3, activation: 'relu' },
        { kind: 'maxpool2d', kernelSize: 2 },
        { kind: 'flatten' },
        { kind: 'dense', units: 3, activation: 'softmax' },
      ],
      metrics: { accuracyAtLeast: 0.7 },
    }),
    solutionCommand: 'net conv 8 3; net pool 2; net flatten; net dense 3 softmax; train 20',
    golf: { bestCommands: 5 },
  }),

  makeLevel({
    id: 'cnn-gap-head',
    sequence: 'cnn',
    name: T('Global Average Pooling', 'Global Average Pooling'),
    objective: T('Replace flatten+big dense with GAP + linear head.', 'جای flatten با GAP'),
    hint: T('`net gap` then a small Dense head.', 'net gap و یک dense کوچک'),
    startState: startFrom({ dataset: { id: 'mnist-toy', inputDim: 64, nClasses: 3, nSamples: 180 } }),
    goal: goal({
      asserts: [
        (s) => s.layers.some((l) => l.kind === 'globalAveragePool'),
        (s) => s.layers.some((l) => l.kind === 'conv2d'),
      ],
    }),
    solutionCommand: 'net conv 16 3; net gap; net dense 3 softmax; train 15',
    golf: { bestCommands: 4 },
  }),

  // ─── sequence ──────────────────────────────────────────────────
  makeLevel({
    id: 'seq-embedding-base',
    sequence: 'sequence',
    name: T('Tokens Become Vectors', 'توکن‌ها بردار می‌شوند'),
    objective: T('Start a sequence model with an Embedding.', 'شروع مدل دنباله با Embedding'),
    hint: T('`net embedding 4 16` on seq-copy.', 'net embedding 4 16'),
    startState: startFrom({
      dataset: { id: 'seq-copy', inputDim: 8, nClasses: 4, seqLen: 8, vocabSize: 4, nSamples: 200 },
    }),
    goal: goal({
      dataset: 'seq-copy',
      asserts: [(s) => s.layers[0]?.kind === 'embedding'],
    }),
    solutionCommand: 'data seq-copy; net embedding 4 16; net dense 4 softmax; train 15',
    golf: { bestCommands: 4 },
    startDialog: [
      alert(
        '## Discrete tokens need a lookup table',
        'An Embedding layer is a trainable matrix `[vocab, dim]`. Each token id selects a row.',
        'That row is the token’s dense representation.',
      ),
      threeLibs('nn.Embedding · keras.layers.Embedding · tf.nn.embedding_lookup'),
    ],
  }),

  makeLevel({
    id: 'seq-rnn-recurrence',
    sequence: 'sequence',
    name: T('Add Recurrence', 'بازگشت را اضافه کن'),
    objective: T('Place SimpleRNN after the embedding.', 'SimpleRNN بعد از embedding'),
    hint: T('`net rnn 16`', 'net rnn 16'),
    startState: startFrom({
      dataset: { id: 'seq-copy', inputDim: 8, nClasses: 4, seqLen: 8, vocabSize: 4 },
    }),
    goal: goal({
      asserts: [(s) => s.layers.some((l) => l.kind === 'simpleRNN')],
    }),
    solutionCommand: 'net embedding 4 16; net rnn 16; net dense 4 softmax; train 15',
    golf: { bestCommands: 4 },
  }),

  makeLevel({
    id: 'seq-lstm-memory',
    sequence: 'sequence',
    name: T('LSTM Memory Cells', 'سلول‌های حافظه LSTM'),
    objective: T('Use LSTM for longer dependencies.', 'LSTM برای وابستگی‌های بلند'),
    hint: T('`net lstm 32`', 'net lstm 32'),
    startState: startFrom({
      dataset: { id: 'seq-add', inputDim: 8, nClasses: 4, seqLen: 8, vocabSize: 4 },
    }),
    goal: goal({
      asserts: [(s) => s.layers.some((l) => l.kind === 'lstm')],
    }),
    solutionCommand: 'data seq-add; net embedding 4 16; net lstm 32; net dense 4 softmax; train 20',
    golf: { bestCommands: 4 },
    startDialog: [
      alert(
        '## Gates beat naive recurrence',
        'LSTM adds forget / input / output gates so gradients can travel through time without vanishing as quickly.',
        'GRU is a lighter cousin with fewer gates.',
      ),
    ],
  }),

  makeLevel({
    id: 'seq-gru-choice',
    sequence: 'sequence',
    name: T('Try a GRU', 'یک GRU امتحان کن'),
    objective: T('Swap LSTM for GRU (cheaper, often enough).', 'جای LSTM با GRU'),
    hint: T('`net gru 32`', 'net gru 32'),
    startState: startFrom({
      dataset: { id: 'seq-add', inputDim: 8, nClasses: 4, seqLen: 8, vocabSize: 4 },
    }),
    goal: goal({
      asserts: [(s) => s.layers.some((l) => l.kind === 'gru')],
    }),
    solutionCommand: 'net embedding 4 16; net gru 32; net dense 4 softmax; train 20',
    golf: { bestCommands: 4 },
  }),

  // ─── attention ─────────────────────────────────────────────────
  makeLevel({
    id: 'attn-positional',
    sequence: 'attention',
    name: T('Order Matters', 'ترتیب مهم است'),
    objective: T('Add PositionalEncoding before attention.', 'PositionalEncoding قبل از attention'),
    hint: T('`net posenc`', 'net posenc'),
    startState: startFrom({
      dataset: { id: 'seq-copy', inputDim: 8, nClasses: 4, seqLen: 8, vocabSize: 4 },
    }),
    goal: goal({
      asserts: [(s) => s.layers.some((l) => l.kind === 'positionalEncoding')],
    }),
    solutionCommand: 'net embedding 4 16; net posenc; net attention 16; net dense 4 softmax; train 10',
    pureConcept: true,
    golf: { bestCommands: 5 },
    startDialog: [
      alert(
        '## Attention is order-agnostic',
        'Self-attention is a set operation unless you inject position information.',
        'Sinusoidal or learned positional encodings restore order.',
      ),
    ],
  }),

  makeLevel({
    id: 'attn-self',
    sequence: 'attention',
    name: T('Self-Attention Block', 'بلوک self-attention'),
    objective: T('Wire attention + layernorm (pre-norm flavor).', 'attention + layernorm'),
    hint: T('embedding → posenc → attention → layernorm → head', 'posenc سپس attention سپس layernorm'),
    startState: startFrom({
      dataset: { id: 'seq-copy', inputDim: 8, nClasses: 4, seqLen: 8, vocabSize: 4 },
    }),
    goal: goal({
      asserts: [
        (s) => s.layers.some((l) => l.kind === 'attention' || l.kind === 'multiheadAttention'),
        (s) => s.layers.some((l) => l.kind === 'layerNorm'),
      ],
    }),
    solutionCommand:
      'net embedding 4 16; net posenc; net attention 16; net layernorm; net dense 4 softmax; train 12',
    pureConcept: true,
    golf: { bestCommands: 6 },
  }),

  makeLevel({
    id: 'attn-multihead',
    sequence: 'attention',
    name: T('Multi-Head Attention', 'توجه چندسر'),
    objective: T('Use MHA with 4 heads.', 'MHA با ۴ سر'),
    hint: T('`net mha 4 32`', 'net mha 4 32'),
    startState: startFrom({
      dataset: { id: 'seq-copy', inputDim: 8, nClasses: 4, seqLen: 8, vocabSize: 4 },
    }),
    goal: goal({
      asserts: [
        (s) => s.layers.some((l) => l.kind === 'multiheadAttention' && (l.heads ?? 0) >= 4),
      ],
    }),
    solutionCommand: 'net embedding 4 32; net posenc; net mha 4 32; net layernorm; net dense 4 softmax; train 10',
    pureConcept: true,
    golf: { bestCommands: 6 },
    startDialog: [
      alert(
        '## Many subspaces, one layer',
        'Multi-head attention projects Q/K/V into several subspaces and concatenates the results.',
        'Heads can specialize: some local syntax, some long-range links.',
      ),
    ],
  }),

  makeLevel({
    id: 'attn-transformer-block',
    sequence: 'attention',
    name: T('A Minimal Transformer Block', 'یک بلوک مینیمال ترنسفورمر'),
    objective: T('posenc + MHA + LN + dense FFN-ish head.', 'بلوک کامل کوچک'),
    hint: T('Stack posenc, mha, layernorm, then a dense head.', 'posenc و mha و layernorm و سر'),
    startState: startFrom({
      dataset: { id: 'seq-copy', inputDim: 8, nClasses: 4, seqLen: 8, vocabSize: 4 },
    }),
    goal: goal({
      asserts: [
        (s) => s.layers.some((l) => l.kind === 'positionalEncoding'),
        (s) => s.layers.some((l) => l.kind === 'multiheadAttention'),
        (s) => s.layers.some((l) => l.kind === 'layerNorm'),
        (s) => s.layers.some((l) => l.kind === 'dense'),
      ],
    }),
    solutionCommand:
      'net embedding 4 32; net posenc; net mha 4 32; net layernorm; net dense 32 relu; net dense 4 softmax; train 12',
    pureConcept: true,
    golf: { bestCommands: 7 },
  }),

  // ─── training ──────────────────────────────────────────────────
  makeLevel({
    id: 'train-lr-discipline',
    sequence: 'training',
    name: T('Learning Rate Discipline', 'نظم نرخ یادگیری'),
    objective: T('Pick lr in (0.001, 0.1] and reach low loss.', 'lr معقول و loss پایین'),
    hint: T('Avoid lr >= 1. Try 0.05.', 'از lr>=1 پرهیز کن'),
    startState: startFrom({
      dataset: { id: 'moons', inputDim: 2, nClasses: 2 },
      compile: { optimizer: 'adam', lr: 5, loss: 'categoricalCrossentropy', metrics: ['accuracy'] },
    }),
    goal: goal({
      asserts: [(s) => s.compile.lr > 0 && s.compile.lr <= 0.1],
      metrics: { accuracyAtLeast: 0.8, epochsAtLeast: 5 },
    }),
    solutionCommand: 'lr 0.05; net dense 16 relu; net dense 2 softmax; train 40',
    golf: { bestCommands: 4 },
  }),

  makeLevel({
    id: 'train-batch-size',
    sequence: 'training',
    name: T('Batch Size Tradeoff', 'معاوضه اندازه دسته'),
    objective: T('Train with a stable batch on moons.', 'دسته‌ی پایدار روی moons'),
    hint: T('Use `train 30` after a solid MLP. Batch defaults to 32.', 'MLP درست و train 30'),
    startState: startFrom({
      dataset: { id: 'moons', inputDim: 2, nClasses: 2 },
    }),
    goal: goal({
      metrics: { accuracyAtLeast: 0.85, epochsAtLeast: 10 },
    }),
    solutionCommand: 'net dense 32 relu; net dense 2 softmax; train 40',
    golf: { bestCommands: 3 },
    startDialog: [
      alert(
        '## Small batches noise, large batches blur',
        'Small batches give noisy but cheap gradient estimates (implicit regularization).',
        'Large batches are stable but need careful learning rates.',
      ),
    ],
  }),

  makeLevel({
    id: 'train-spiral-hard',
    sequence: 'training',
    name: T('The Spiral Stress Test', 'آزمون فشار spiral'),
    objective: T('Clear noisy spiral with a deep MLP.', 'پاک‌سازی spiral پرنویز با MLP عمیق'),
    hint: T('Width + dropout + many epochs.', 'عرض + دروپ‌اوت + epoch زیاد'),
    startState: startFrom({ dataset: { id: 'spiral', inputDim: 2, nClasses: 3, noise: 0.1, nSamples: 300 } }),
    goal: goal({
      metrics: { accuracyAtLeast: 0.75, epochsAtLeast: 20 },
    }),
    solutionCommand:
      'data spiral; net dense 64 relu; net dropout 0.2; net dense 32 relu; net dense 3 softmax; train 100',
    golf: { bestCommands: 5 },
  }),

  // ─── production ────────────────────────────────────────────────
  makeLevel({
    id: 'prod-transfer-freeze',
    sequence: 'production',
    name: T('Transfer Learning Mindset', 'تفکر یادگیری انتقالی'),
    objective: T('Use a strong pretrained-style trunk + small head.', 'تنه قوی + سر کوچک'),
    hint: T('Wide feature trunk, small softmax head.', 'تنه پهن، سر کوچک'),
    startState: startFrom({ dataset: { id: 'mnist-toy', inputDim: 64, nClasses: 3, nSamples: 180 } }),
    goal: goal({
      asserts: [
        (s) => {
          const dens = s.layers.filter((l) => l.kind === 'dense');
          return dens.length >= 2 && (dens[0]?.units ?? 0) >= (dens[1]?.units ?? 99);
        },
      ],
      metrics: { accuracyAtLeast: 0.75 },
    }),
    solutionCommand: 'net conv 16 3; net gap; net dense 32 relu; net dense 3 softmax; train 25',
    golf: { bestCommands: 5 },
  }),

  makeLevel({
    id: 'prod-export-ready',
    sequence: 'production',
    name: T('Export-Ready Head', 'سر آماده export'),
    objective: T('Final layer must be a proper classifier head (softmax / sigmoid).', 'آخرین لایه classifier استاندارد'),
    hint: T('End with `dense <n> softmax`.', 'پایان با dense n softmax'),
    startState: startFrom({ dataset: { id: 'xor', inputDim: 2, nClasses: 2 } }),
    goal: goal({
      asserts: [
        (s) => {
          const last = s.layers[s.layers.length - 1];
          return last?.kind === 'dense' && (last.activation === 'softmax' || last.activation === 'sigmoid');
        },
      ],
    }),
    solutionCommand: 'net dense 8 relu; net dense 2 softmax',
    pureConcept: true,
    golf: { bestCommands: 2 },
    startDialog: [
      alert(
        '## Serving contracts start at the head',
        'A deterministic output activation makes thresholds, top-k, and calibration sane in production.',
        'Look at the Keras / TorchServe / TF Serving snippets in the Code panel.',
      ),
    ],
  }),

  makeLevel({
    id: 'prod-finetune-small-lr',
    sequence: 'production',
    name: T('Fine-Tune with a Small LR', 'fine-tune با نرخ کوچک'),
    objective: T('Low learning rate + solid accuracy.', 'نرخ کوچک + دقت خوب'),
    hint: T('`lr 0.001` after building the model.', 'lr 0.001'),
    startState: startFrom({ dataset: { id: 'moons', inputDim: 2, nClasses: 2 } }),
    goal: goal({
      asserts: [(s) => s.compile.lr <= 0.01],
      metrics: { accuracyAtLeast: 0.85 },
    }),
    solutionCommand: 'lr 0.001; net dense 32 relu; net dense 2 softmax; train 50',
    golf: { bestCommands: 4 },
  }),

  // ─── pytorch track ─────────────────────────────────────────────
  makeLevel({
    id: 'pt-module-forward',
    sequence: 'pytorch',
    name: T('nn.Module Anatomy', 'آناتومی nn.Module'),
    objective: T('Architecture that maps to Linear layers in PyTorch.', 'معماری معادل Linear'),
    hint: T('Two Dense layers ≡ two nn.Linear in forward.', 'دو Dense معادل دو Linear'),
    startState: startFrom({ dataset: { id: 'xor', inputDim: 2, nClasses: 2 } }),
    goal: goal({
      layers: [
        { kind: 'dense', units: 16, activation: 'relu', inputDim: 2 },
        { kind: 'dense', units: 2, activation: 'softmax' },
      ],
    }),
    solutionCommand: 'library pytorch; net dense 16 relu; net dense 2 softmax',
    pureConcept: true,
    golf: { bestCommands: 3 },
    startDialog: [
      alert(
        '## __init__ holds parameters, forward applies ops',
        'PyTorch modules register weights in `__init__` and define the graph in `forward`.',
        'Switch the Code panel to PyTorch — watch the class form as you edit layers.',
      ),
    ],
  }),

  makeLevel({
    id: 'pt-custom-loop',
    sequence: 'pytorch',
    name: T('Custom Training Loop', 'حلقه‌ی ترین سفارشی'),
    objective: T('Compile settings that survive as a real PT loop.', 'تنظیماتی که در حلقه PT می‌نشینند'),
    hint: T('optimizer + loss + train epochs.', 'optimizer و loss و train'),
    startState: startFrom({ dataset: { id: 'moons', inputDim: 2, nClasses: 2 } }),
    goal: goal({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: { accuracyAtLeast: 0.85, epochsAtLeast: 5 },
    }),
    solutionCommand:
      'library pytorch; net dense 32 relu; net dense 2 softmax; optimizer adam; loss ce; train 40',
    golf: { bestCommands: 6 },
    startDialog: [
      codeCompare(
        [
          '## zero_grad → backward → step',
          'Every PyTorch loop repeats this dance. Compare with Keras `model.fit` on the next tab.',
        ],
        'pytorch',
      ),
    ],
  }),

  makeLevel({
    id: 'pt-dataloader-mindset',
    sequence: 'pytorch',
    name: T('DataLoader Mindset', 'ذهنیت DataLoader'),
    objective: T('Batched training with a stable MLP.', 'ترین دسته‌ای با MLP پایدار'),
    hint: T('Build the net and train. Batch defaults to 32 in the engine.', 'شبکه را بساز و ترین کن'),
    startState: startFrom({ dataset: { id: 'circles', inputDim: 2, nClasses: 2 } }),
    goal: goal({
      metrics: { accuracyAtLeast: 0.85, epochsAtLeast: 10 },
    }),
    solutionCommand: 'library pytorch; net dense 32 relu; net dense 2 softmax; train 50',
    golf: { bestCommands: 4 },
  }),

  // ─── tensorflow track ──────────────────────────────────────────
  makeLevel({
    id: 'tf-gradient-tape',
    sequence: 'tensorflow',
    name: T('GradientTape Muscle', 'عضله GradientTape'),
    objective: T('Loss + optimizer + train as TF tape workflow.', 'جریان tape در TF'),
    hint: T('Set loss/optimizer then train.', 'loss و optimizer و train'),
    startState: startFrom({ dataset: { id: 'xor', inputDim: 2, nClasses: 2 } }),
    goal: goal({
      loss: 'categoricalCrossentropy',
      metrics: { accuracyAtLeast: 0.85, epochsAtLeast: 5 },
    }),
    solutionCommand:
      'library tensorflow; net dense 16 relu; net dense 2 softmax; loss ce; optimizer adam; train 40',
    golf: { bestCommands: 6 },
    startDialog: [
      codeCompare(
        [
          '## tf.GradientTape is the autograd API',
          'Record ops inside the tape context, then `tape.gradient` + `apply_gradients`.',
        ],
        'tensorflow',
      ),
    ],
  }),

  makeLevel({
    id: 'tf-tf-function',
    sequence: 'tensorflow',
    name: T('@tf.function Mentality', 'ذهنیت @tf.function'),
    objective: T('A clean sequential graph shape.', 'شکل گراف تمیز متوالی'),
    hint: T('Stack dense layers cleanly.', 'لایه‌های dense تمیز'),
    startState: startFrom({ dataset: { id: 'moons', inputDim: 2, nClasses: 2 } }),
    goal: goal({
      layers: [
        { kind: 'dense', units: 32, activation: 'relu', inputDim: 2 },
        { kind: 'dropout', rate: 0.2 },
        { kind: 'dense', units: 2, activation: 'softmax' },
      ],
    }),
    solutionCommand: 'library tensorflow; net dense 32 relu; net dropout 0.2; net dense 2 softmax',
    pureConcept: true,
    golf: { bestCommands: 4 },
  }),

  makeLevel({
    id: 'tf-data-pipeline',
    sequence: 'tensorflow',
    name: T('tf.data Thinking', 'تفکر tf.data'),
    objective: T('Dataset choice + batched training.', 'انتخاب داده + ترین دسته‌ای'),
    hint: T('Pick moons and train an MLP.', 'moons و MLP'),
    startState: startFrom({ dataset: { id: 'xor', inputDim: 2, nClasses: 2 } }),
    goal: goal({
      dataset: 'moons',
      metrics: { accuracyAtLeast: 0.85 },
    }),
    solutionCommand: 'library tensorflow; data moons; net dense 32 relu; net dense 2 softmax; train 40',
    golf: { bestCommands: 5 },
  }),

  // ─── keras track ───────────────────────────────────────────────
  makeLevel({
    id: 'keras-sequential',
    sequence: 'keras',
    name: T('Sequential Superpower', 'قدرت Sequential'),
    objective: T('A clean Sequential MLP.', 'MLP تمیز Sequential'),
    hint: T('Dense → Dense with matching compile.', 'Dense → Dense با compile درست'),
    startState: startFrom({ dataset: { id: 'xor', inputDim: 2, nClasses: 2 } }),
    goal: goal({
      layers: [
        { kind: 'dense', units: 16, activation: 'relu', inputDim: 2 },
        { kind: 'dense', units: 2, activation: 'softmax' },
      ],
      loss: 'categoricalCrossentropy',
    }),
    solutionCommand: 'library keras; net dense 16 relu; net dense 2 softmax; loss ce',
    pureConcept: true,
    golf: { bestCommands: 4 },
    startDialog: [
      codeCompare(['## model = keras.Sequential([...])'], 'keras'),
    ],
  }),

  makeLevel({
    id: 'keras-compile-fit',
    sequence: 'keras',
    name: T('compile + fit', 'compile + fit'),
    objective: T('Full Keras compile/fit hyperparams.', 'هایپرپارامترهای compile/fit'),
    hint: T('adam + ce + train.', 'adam و ce و train'),
    startState: startFrom({ dataset: { id: 'moons', inputDim: 2, nClasses: 2 } }),
    goal: goal({
      optimizer: 'adam',
      loss: 'categoricalCrossentropy',
      metrics: { accuracyAtLeast: 0.85, epochsAtLeast: 5 },
    }),
    solutionCommand: 'library keras; net dense 32 relu; net dense 2 softmax; optimizer adam; loss ce; train 40',
    golf: { bestCommands: 6 },
  }),

  makeLevel({
    id: 'keras-callbacks-story',
    sequence: 'keras',
    name: T('Callbacks & Checkpoints', 'callback و checkpoint'),
    objective: T('Regularized model that trains stably.', 'مدل منظم‌شده‌ی پایدار'),
    hint: T('dropout + train long enough.', 'dropout و ترین کافی'),
    startState: startFrom({ dataset: { id: 'spiral', inputDim: 2, nClasses: 3, noise: 0.1, nSamples: 250 } }),
    goal: goal({
      asserts: [(s) => s.layers.some((l) => l.kind === 'dropout')],
      metrics: { accuracyAtLeast: 0.65, epochsAtLeast: 10 },
    }),
    solutionCommand:
      'library keras; data spiral; net dense 64 relu; net dropout 0.3; net dense 3 softmax; train 80',
    golf: { bestCommands: 6 },
    startDialog: [
      alert(
        '## fit(..., callbacks=[EarlyStopping, ModelCheckpoint])',
        'The engine’s train loop mirrors what those callbacks police: stop when val stalls, save the best.',
      ),
    ],
  }),
];

/** Fix spiral class-count in dropout level solution (3 classes). */
const dropoutLevel = levels.find((l) => l.id === 'reg-dropout');
if (dropoutLevel) {
  dropoutLevel.solutionCommand =
    'data spiral; net dense 64 relu; net dropout 0.3; net dense 3 softmax; train 50';
  dropoutLevel.goal = goal({
    asserts: [
      (s) => s.layers.filter((l) => l.kind === 'dropout').length >= 1,
      (s) => s.layers.some((l) => l.kind === 'dense' && (l.units ?? 0) >= 32),
    ],
    metrics: { accuracyAtLeast: 0.7 },
    dataset: 'spiral',
  });
}
