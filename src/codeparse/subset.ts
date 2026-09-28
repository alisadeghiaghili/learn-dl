import type { ActivationName, ExperimentState, LayerSpec, LibraryId } from '../state/types';
import { cloneExperiment, emptyExperiment } from '../state/types';

/**
 * Pragmatic Python subset parser.
 *
 * Recognizes common Sequential / nn.Module / Dense / train patterns from the
 * three libraries and maps them into ExperimentState. This is intentionally
 * incomplete — it teaches the happy-path APIs and keeps the three surfaces in
 * sync for the patterns used in levels.
 */

export interface ParseResult {
  state: ExperimentState;
  warnings: string[];
}

const LAYER_CALL_RE =
  /(?:keras\.layers\.|tf\.keras\.layers\.|layers\.|nn\.)([A-Za-z0-9_]+)\s*\(([^)]*)\)/g;
const NN_CLASS_RE = /self\.(\w+)\s*=\s*nn\.([A-Za-z0-9_]+)\s*\(([^)]*)\)/g;

function parseArgs(argStr: string): Record<string, string> {
  const out: Record<string, string> = {};
  // positional capture
  const parts = splitArgs(argStr);
  const keywords = ['activation', 'units', 'rate', 'num_heads', 'key_dim', 'filters',
    'kernel_size', 'return_sequences', 'vocab_size', 'embedding_dim', 'input_shape',
    'learning_rate', 'momentum'];
  parts.forEach((raw, idx) => {
    const eq = raw.indexOf('=');
    if (eq > 0 && keywords.includes(raw.slice(0, eq).trim())) {
      out[raw.slice(0, eq).trim()] = raw.slice(eq + 1).trim().replace(/^['"]|['"]$/g, '');
      return;
    }
    out[`arg${idx}`] = raw.trim().replace(/^['"]|['"]$/g, '');
  });
  return out;
}

function splitArgs(s: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let cur = '';
  for (const ch of s) {
    if (ch === '(') depth++;
    if (ch === ')') depth--;
    if (ch === ',' && depth === 0) {
      if (cur.trim()) out.push(cur.trim());
      cur = '';
    } else {
      cur += ch;
    }
  }
  if (cur.trim()) out.push(cur.trim());
  return out;
}

function mapActivation(a?: string): ActivationName | undefined {
  if (!a) return undefined;
  const v = a.toLowerCase();
  if (v === 'relu' || v === 'sigmoid' || v === 'tanh' || v === 'softmax' || v === 'gelu' || v === 'silu') {
    return v;
  }
  if (v === 'linear') return 'linear';
  return undefined;
}

function kerasLayerToSpec(name: string, args: Record<string, string>, inputDim?: number): LayerSpec | null {
  const units = Number(args.units ?? args.arg0 ?? args.filters ?? 32);
  switch (name) {
    case 'Dense':
      return {
        kind: 'dense',
        units: Number(args.units ?? args.arg0 ?? 32),
        activation: mapActivation(args.activation),
        inputDim,
      };
    case 'Dropout':
      return { kind: 'dropout', rate: Number(args.rate ?? args.arg0 ?? 0.5) };
    case 'BatchNormalization':
      return { kind: 'batchnorm' };
    case 'LayerNormalization':
      return { kind: 'layerNorm' };
    case 'Flatten':
      return { kind: 'flatten' };
    case 'Conv2D':
      return {
        kind: 'conv2d',
        units: Number(args.filters ?? args.arg0 ?? 8),
        kernelSize: Number(args.kernel_size ?? args.arg1 ?? 3),
        activation: 'relu',
      };
    case 'MaxPooling2D':
      return { kind: 'maxpool2d', kernelSize: Number(args.pool_size ?? args.arg0 ?? 2) };
    case 'GlobalAveragePooling2D':
      return { kind: 'globalAveragePool' };
    case 'Embedding':
      return {
        kind: 'embedding',
        vocabSize: Number(args.vocab_size ?? args.arg0 ?? 100),
        embedDim: Number(args.embedding_dim ?? args.arg1 ?? 16),
      };
    case 'LSTM':
      return {
        kind: 'lstm',
        units: Number(args.units ?? args.arg0 ?? 32),
        returnSequences: args.return_sequences === 'True' || args.return_sequences === 'true',
      };
    case 'GRU':
      return {
        kind: 'gru',
        units: Number(args.units ?? args.arg0 ?? 32),
        returnSequences: args.return_sequences === 'True' || args.return_sequences === 'true',
      };
    case 'SimpleRNN':
      return {
        kind: 'simpleRNN',
        units: Number(args.units ?? args.arg0 ?? 32),
        returnSequences: args.return_sequences === 'True' || args.return_sequences === 'true',
      };
    case 'Attention':
      return { kind: 'attention', units };
    case 'MultiHeadAttention':
      return {
        kind: 'multiheadAttention',
        heads: Number(args.num_heads ?? 4),
        units: Number(args.key_dim ?? 32),
      };
    default:
      return null;
  }
}

function nnLayerToSpec(name: string, args: Record<string, string>): LayerSpec | null {
  const a0 = Number(args.arg0 ?? args.units ?? 0);
  const a1 = Number(args.arg1 ?? 0);
  switch (name) {
    case 'Linear':
      return { kind: 'dense', units: a1 || a0, inputDim: a1 ? a0 : undefined };
    case 'Dropout':
      return { kind: 'dropout', rate: a0 || 0.5 };
    case 'BatchNorm1d':
    case 'BatchNorm2d':
      return { kind: 'batchnorm' };
    case 'LayerNorm':
      return { kind: 'layerNorm' };
    case 'Flatten':
      return { kind: 'flatten' };
    case 'Conv2d':
      return { kind: 'conv2d', units: a1 || 8, kernelSize: 3 };
    case 'MaxPool2d':
      return { kind: 'maxpool2d', kernelSize: a0 || 2 };
    case 'Embedding':
      return { kind: 'embedding', vocabSize: a0 || 100, embedDim: a1 || 16 };
    case 'LSTM':
      return { kind: 'lstm', units: a1 || 32 };
    case 'GRU':
      return { kind: 'gru', units: a1 || 32 };
    case 'RNN':
      return { kind: 'simpleRNN', units: a1 || 32 };
    case 'MultiheadAttention':
      return {
        kind: 'multiheadAttention',
        units: Number(args.embed_dim ?? 32),
        heads: Number(args.num_heads ?? 4),
      };
    default:
      return null;
  }
}

/**
 * Parse a Python snippet into ExperimentState.
 *
 * Args:
 *   source: Python-like source (PyTorch / TF / Keras patterns)
 *   base: optional starting state (defaults to empty experiment)
 * Returns:
 *   ParseResult with merged state and human-readable warnings.
 *
 * Raises:
 *   (never) — unparseable lines become warnings.
 */
export function parsePythonSubset(source: string, base?: ExperimentState): ParseResult {
  const state = cloneExperiment(base ?? emptyExperiment());
  const warnings: string[] = [];
  const layers: LayerSpec[] = [];
  const text = source;

  // library detect
  if (/import torch|from torch/.test(text)) state.library = 'pytorch';
  else if (/from tensorflow import keras|from tensorflow\.|import tensorflow/.test(text) && !/keras\.Sequential|model\.compile/.test(text)) {
    state.library = 'tensorflow';
  } else if (/keras\.|from tensorflow import keras/.test(text)) {
    state.library = 'keras';
  }

  // optimizer
  const optMatch =
    text.match(/optimi[sz]ers?\.([A-Za-z0-9_]+)\s*\(/) ||
    text.match(/torch\.optim\.([A-Za-z0-9_]+)\s*\(/) ||
    text.match(/optimizer\s*=\s*['"]([a-z]+)['"]/);
  if (optMatch?.[1]) {
    const o = optMatch[1].toLowerCase();
    if (o.includes('adamw')) state.compile.optimizer = 'adamw';
    else if (o.includes('adam')) state.compile.optimizer = 'adam';
    else if (o.includes('rmsprop')) state.compile.optimizer = 'rmsprop';
    else if (o.includes('sgd')) state.compile.optimizer = 'sgd';
  }
  const lrMatch = text.match(/learning_rate\s*=\s*([0-9.eE+-]+)/) || text.match(/lr\s*=\s*([0-9.eE+-]+)/);
  if (lrMatch?.[1]) state.compile.lr = Number(lrMatch[1]);

  // loss
  const lossMatch =
    text.match(/loss\s*=\s*['"]([a-z_]+)['"]/) ||
    text.match(/([A-Za-z]*CrossEntropy[A-Za-z]*|MSELoss|L1Loss|BCELoss|BCEWithLogitsLoss)/);
  if (lossMatch?.[1]) {
    const l = lossMatch[1];
    if (/sparse/i.test(l)) state.compile.loss = 'sparseCategoricalCrossentropy';
    else if (/CrossEntropy|categorical/i.test(l)) state.compile.loss = 'categoricalCrossentropy';
    else if (/BCE/i.test(l)) state.compile.loss = 'binaryCrossentropy';
    else if (/MSE/i.test(l)) state.compile.loss = 'mse';
    else if (/L1|mae/i.test(l)) state.compile.loss = 'mae';
    else state.compile.loss = 'categoricalCrossentropy';
  }

  // epochs / batch
  const ep = text.match(/epochs\s*=\s*(\d+)/);
  if (ep?.[1]) state.training.epochsTrained = Number(ep[1]);
  const bs = text.match(/batch_size\s*=\s*(\d+)/) || text.match(/batch_size=(\d+)/);
  if (bs?.[1]) state.training.batch = Number(bs[1]);

  // dataset hint
  const ds = text.match(/load_dataset\(\s*['"]([a-z0-9-]+)['"]/i) || text.match(/dataset:\s*([a-z0-9-]+)/i);
  if (ds?.[1]) {
    const id = ds[1] as ExperimentState['dataset']['id'];
    state.dataset = { ...state.dataset, id };
  }

  // layers from keras-style
  let m: RegExpExecArray | null;
  LAYER_CALL_RE.lastIndex = 0;
  const kerasy = /keras\.layers\.|tf\.keras\.layers\.|layers\.Dense|Sequential/.test(text);
  if (kerasy) {
    while ((m = LAYER_CALL_RE.exec(text)) !== null) {
      const spec = kerasLayerToSpec(m[1] ?? '', parseArgs(m[2] ?? ''));
      if (spec) layers.push(spec);
    }
  }

  // layers from nn.Module
  NN_CLASS_RE.lastIndex = 0;
  while ((m = NN_CLASS_RE.exec(text)) !== null) {
    const spec = nnLayerToSpec(m[2] ?? '', parseArgs(m[3] ?? ''));
    if (spec) layers.push(spec);
  }

  // Sequential([...]) multi-line Dense list
  const seq = text.match(/Sequential\(\[([\s\S]*?)\]\)/);
  if (seq?.[1] && layers.length === 0) {
    const inner = seq[1];
    LAYER_CALL_RE.lastIndex = 0;
    while ((m = LAYER_CALL_RE.exec(inner)) !== null) {
      const spec = kerasLayerToSpec(m[1] ?? '', parseArgs(m[2] ?? ''));
      if (spec) layers.push(spec);
    }
  }

  if (layers.length > 0) {
    // assign ids and inputDim on first dense
    let sawInput = false;
    state.layers = layers.map((l, i) => {
      const id = `L${i}`;
      if (l.kind === 'dense' && !sawInput) {
        sawInput = true;
        return { ...l, id, inputDim: l.inputDim ?? state.dataset.inputDim ?? 2 };
      }
      return { ...l, id };
    });
  } else {
    warnings.push('No recognizable layers found — architecture unchanged.');
  }

  if (!text.trim()) {
    warnings.push('Empty source.');
  }

  return { state, warnings };
}

export function detectLibrary(source: string): LibraryId {
  if (/import torch|from torch/.test(source)) return 'pytorch';
  if (/Sequential|keras\.layers/.test(source)) return 'keras';
  return 'tensorflow';
}
