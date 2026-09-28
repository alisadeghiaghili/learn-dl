import type {
  ActivationName,
  CompileSpec,
  DatasetId,
  ExperimentState,
  LayerSpec,
  LibraryId,
  LossName,
  MetricName,
  OptimizerName,
} from '../state/types';
import { cloneExperiment, emptyExperiment } from '../state/types';

export interface CommandResult {
  ok: boolean;
  message: string;
  state: ExperimentState;
  /** Side-effect commands (not state mutations). */
  action?: 'levels' | 'hint' | 'goal' | 'solution' | 'reset' | 'undo' | 'help' | 'clear' | 'build' | 'demo';
  payload?: unknown;
}

export interface CommandContext {
  state: ExperimentState;
  undoStack: ExperimentState[];
  levelId?: string;
  disabled?: Record<string, true>;
}

function fail(message: string, state: ExperimentState): CommandResult {
  return { ok: false, message, state };
}

function ok(message: string, state: ExperimentState, action?: CommandResult['action'], payload?: unknown): CommandResult {
  return { ok: true, message, state, action, payload };
}

function parseActivation(s: string | undefined): ActivationName | undefined {
  if (!s) return undefined;
  const v = s.toLowerCase();
  const allowed: ActivationName[] = ['linear', 'relu', 'sigmoid', 'tanh', 'softmax', 'gelu', 'silu'];
  return (allowed as string[]).includes(v) ? (v as ActivationName) : undefined;
}

/**
 * Execute one DSL command against the experiment state.
 *
 * Args:
 *   raw: user command line
 *   ctx: current state + undo stack + level guards
 * Returns:
 *   CommandResult with new state and message
 */
export function executeCommand(raw: string, ctx: CommandContext): CommandResult {
  const line = raw.trim();
  if (!line) return ok('', ctx.state);

  const [head, ...rest] = line.split(/\s+/);
  const cmd = (head ?? '').toLowerCase();
  const args = rest;

  if (ctx.disabled?.[cmd] || ctx.disabled?.[line]) {
    return fail(`Command disabled in this level: ${cmd}`, ctx.state);
  }

  // allow chaining like LGB with semicolons
  if (line.includes(';')) {
    let state = ctx.state;
    const undoStack = [...ctx.undoStack];
    const messages: string[] = [];
    for (const part of line.split(';')) {
      const r = executeCommand(part, { ...ctx, state, undoStack });
      state = r.state;
      undoStack.push(ctx.state);
      messages.push(r.message);
      if (!r.ok) return { ...r, message: messages.filter(Boolean).join(' | ') };
      if (r.action) return { ...r, message: messages.filter(Boolean).join(' | ') };
    }
    return ok(messages.filter(Boolean).join(' | '), state);
  }

  switch (cmd) {
    case 'help':
    case '?':
      return ok(HELP_TEXT, ctx.state, 'help');

    case 'levels':
      return ok('Opening level browser…', ctx.state, 'levels');

    case 'hint':
      return ok('Hint requested…', ctx.state, 'hint');

    case 'show': {
      const what = args[0];
      if (what === 'goal') return ok('Showing goal…', ctx.state, 'goal');
      if (what === 'solution') return ok('Solution requested…', ctx.state, 'solution');
      return fail('Usage: show goal | show solution', ctx.state);
    }

    case 'objective':
    case 'startDialog':
      return ok('Objective dialog…', ctx.state, 'demo');

    case 'reset': {
      const next = emptyExperiment();
      next.library = ctx.state.library;
      return ok('Experiment reset.', next, 'reset');
    }

    case 'undo': {
      const prev = ctx.undoStack[ctx.undoStack.length - 1];
      if (!prev) return fail('Nothing to undo.', ctx.state);
      return ok('Undid last command.', prev, 'undo');
    }

    case 'clear':
      return ok('', ctx.state, 'clear');

    case 'build':
    case 'builder':
      return ok('Opening visual builder…', ctx.state, 'build');

    case 'library':
    case 'use': {
      const lib = args[0]?.toLowerCase();
      if (lib !== 'pytorch' && lib !== 'tensorflow' && lib !== 'keras') {
        return fail('Usage: library pytorch|tensorflow|keras', ctx.state);
      }
      const state = cloneExperiment(ctx.state);
      state.library = lib as LibraryId;
      return ok(`Code panels set to ${lib}.`, state);
    }

    case 'data':
    case 'dataset': {
      const id = args[0] as DatasetId | undefined;
      const valid: DatasetId[] = ['xor', 'moons', 'circles', 'spiral', 'linear', 'regression', 'seq-copy', 'seq-add', 'mnist-toy'];
      if (!id || !valid.includes(id)) {
        return fail(`Usage: data <${valid.join('|')}>`, ctx.state);
      }
      const state = cloneExperiment(ctx.state);
      state.dataset = {
        ...state.dataset,
        id,
        inputDim: id.startsWith('seq') ? (state.dataset.seqLen ?? 8) : id === 'mnist-toy' ? 64 : id === 'regression' ? 1 : 2,
        nClasses: id === 'spiral' ? 3 : id === 'mnist-toy' ? 3 : 2,
        seqLen: id.startsWith('seq') ? (state.dataset.seqLen ?? 8) : undefined,
        vocabSize: id.startsWith('seq') ? (state.dataset.vocabSize ?? 4) : undefined,
      };
      state.training = { epochsTrained: 0, batch: state.training.batch, history: [] };
      return ok(`Dataset set to ${id}.`, state);
    }

    case 'net':
    case 'layer':
    case 'add': {
      const r = addLayer(args, ctx.state);
      return r;
    }

    case 'clearnet':
    case 'clearNet': {
      const state = cloneExperiment(ctx.state);
      state.layers = [];
      state.training = { epochsTrained: 0, batch: state.training.batch, history: [] };
      return ok('Architecture cleared.', state);
    }

    case 'compile': {
      return compileCommand(args, ctx.state);
    }

    case 'train':
    case 'fit': {
      const epochs = Number(args[0] ?? 1);
      if (!Number.isFinite(epochs) || epochs < 1) return fail('Usage: train <epochs>', ctx.state);
      // training itself is executed by the app side-effect (needs engine);
      // here we only stage the request into training.epochsTrained target.
      return ok(`Train requested for ${epochs} epoch(s).`, cloneExperiment(ctx.state), 'demo', {
        trainEpochs: epochs,
      });
    }

    case 'lr': {
      const lr = Number(args[0]);
      if (!Number.isFinite(lr) || lr <= 0) return fail('Usage: lr <number>', ctx.state);
      const state = cloneExperiment(ctx.state);
      state.compile.lr = lr;
      return ok(`learning rate = ${lr}`, state);
    }

    case 'optimizer': {
      const name = args[0]?.toLowerCase();
      const valid: OptimizerName[] = ['sgd', 'momentum', 'adam', 'adamw', 'rmsprop'];
      if (!name || !valid.includes(name as OptimizerName)) {
        return fail(`Usage: optimizer <${valid.join('|')}>`, ctx.state);
      }
      const state = cloneExperiment(ctx.state);
      state.compile.optimizer = name as OptimizerName;
      return ok(`optimizer = ${name}`, state);
    }

    case 'loss': {
      const name = args[0];
      const map: Record<string, LossName> = {
        mse: 'mse',
        mae: 'mae',
        bce: 'binaryCrossentropy',
        ce: 'categoricalCrossentropy',
        sparsece: 'sparseCategoricalCrossentropy',
        binarycrossentropy: 'binaryCrossentropy',
        categoricalcrossentropy: 'categoricalCrossentropy',
      };
      const key = name?.toLowerCase() ?? '';
      const loss = map[key];
      if (!loss) return fail('Usage: loss <mse|mae|bce|ce|sparsece>', ctx.state);
      const state = cloneExperiment(ctx.state);
      state.compile.loss = loss;
      return ok(`loss = ${loss}`, state);
    }

    case 'metric': {
      const name = args[0]?.toLowerCase();
      const valid: MetricName[] = ['accuracy', 'mse', 'mae'];
      if (!name || !valid.includes(name as MetricName)) {
        return fail(`Usage: metric <${valid.join('|')}>`, ctx.state);
      }
      const state = cloneExperiment(ctx.state);
      if (!state.compile.metrics.includes(name as MetricName)) {
        state.compile.metrics = [...state.compile.metrics, name as MetricName];
      }
      return ok(`metrics = ${state.compile.metrics.join(', ')}`, state);
    }

    case 'status':
    case 'info':
    case 'summary':
      return ok(describeState(ctx.state), ctx.state);

    case 'demo': {
      return ok('Demo…', ctx.state, 'demo', { demo: args[0] });
    }

    default:
      return fail(`Unknown command '${cmd}'. Type 'help'.`, ctx.state);
  }
}

function addLayer(args: string[], state: ExperimentState): CommandResult {
  const kind = args[0]?.toLowerCase();
  if (!kind) return fail('Usage: net <dense|dropout|batchnorm|flatten|conv|pool|lstm|gru|rnn|attention|embedding> [params]', state);

  const next = cloneExperiment(state);
  const layer: LayerSpec = { kind: 'dense' };

  switch (kind) {
    case 'dense':
    case 'linear':
    case 'fc': {
      layer.kind = 'dense';
      layer.units = Number(args[1] ?? 32);
      if (!Number.isFinite(layer.units) || layer.units < 1) return fail('net dense <units> [activation]', state);
      layer.activation = parseActivation(args[2]) ?? 'relu';
      if (next.layers.length === 0) {
        layer.inputDim = state.dataset.inputDim ?? 2;
      }
      break;
    }
    case 'dropout': {
      layer.kind = 'dropout';
      layer.rate = Number(args[1] ?? 0.5);
      if (layer.rate < 0 || layer.rate >= 1) return fail('dropout rate must be in [0, 1)', state);
      break;
    }
    case 'batchnorm':
    case 'bn': {
      layer.kind = 'batchnorm';
      break;
    }
    case 'layernorm':
    case 'ln': {
      layer.kind = 'layerNorm';
      break;
    }
    case 'flatten': {
      layer.kind = 'flatten';
      break;
    }
    case 'conv':
    case 'conv2d': {
      layer.kind = 'conv2d';
      layer.units = Number(args[1] ?? 8);
      layer.kernelSize = Number(args[2] ?? 3);
      layer.activation = 'relu';
      break;
    }
    case 'pool':
    case 'maxpool':
    case 'maxpool2d': {
      layer.kind = 'maxpool2d';
      layer.kernelSize = Number(args[1] ?? 2);
      break;
    }
    case 'gap':
    case 'globalpool': {
      layer.kind = 'globalAveragePool';
      break;
    }
    case 'embedding':
    case 'emb': {
      layer.kind = 'embedding';
      layer.vocabSize = Number(args[1] ?? 100);
      layer.embedDim = Number(args[2] ?? 16);
      break;
    }
    case 'lstm': {
      layer.kind = 'lstm';
      layer.units = Number(args[1] ?? 32);
      layer.returnSequences = args.includes('return_sequences');
      break;
    }
    case 'gru': {
      layer.kind = 'gru';
      layer.units = Number(args[1] ?? 32);
      layer.returnSequences = args.includes('return_sequences');
      break;
    }
    case 'rnn':
    case 'simplernn': {
      layer.kind = 'simpleRNN';
      layer.units = Number(args[1] ?? 32);
      layer.returnSequences = args.includes('return_sequences');
      break;
    }
    case 'attention':
    case 'attn': {
      layer.kind = 'attention';
      layer.units = Number(args[1] ?? 32);
      break;
    }
    case 'mha':
    case 'multihead': {
      layer.kind = 'multiheadAttention';
      layer.heads = Number(args[1] ?? 4);
      layer.units = Number(args[2] ?? 32);
      break;
    }
    case 'posenc':
    case 'positionalembedding': {
      layer.kind = 'positionalEncoding';
      break;
    }
    default:
      return fail(`Unknown layer kind '${kind}'.`, state);
  }

  layer.id = `L${next.layers.length}`;
  next.layers = [...next.layers, layer];
  next.training = { epochsTrained: 0, batch: next.training.batch, history: [] };
  return ok(`Added ${layer.kind} ${layer.units ?? ''}`.trim(), next);
}

function compileCommand(args: string[], state: ExperimentState): CommandResult {
  // compile <optimizer> <lr> <loss>
  const [opt, lr, loss] = args;
  const next = cloneExperiment(state);
  const compile: CompileSpec = { ...next.compile };
  if (opt) {
    const valid: OptimizerName[] = ['sgd', 'momentum', 'adam', 'adamw', 'rmsprop'];
    if (!valid.includes(opt.toLowerCase() as OptimizerName)) {
      return fail(`compile optimizer must be ${valid.join('|')}`, state);
    }
    compile.optimizer = opt.toLowerCase() as OptimizerName;
  }
  if (lr !== undefined) {
    const n = Number(lr);
    if (!Number.isFinite(n) || n <= 0) return fail('compile lr must be > 0', state);
    compile.lr = n;
  }
  if (loss) {
    const map: Record<string, LossName> = {
      mse: 'mse',
      mae: 'mae',
      bce: 'binaryCrossentropy',
      ce: 'categoricalCrossentropy',
      sparsece: 'sparseCategoricalCrossentropy',
    };
    const l = map[loss.toLowerCase()];
    if (!l) return fail('compile loss must be mse|mae|bce|ce|sparsece', state);
    compile.loss = l;
  }
  next.compile = compile;
  return ok(
    `compile ${compile.optimizer} lr=${compile.lr} loss=${compile.loss}`,
    next,
  );
}

export function describeState(state: ExperimentState): string {
  const lines: string[] = [];
  lines.push(`library: ${state.library}`);
  lines.push(`dataset: ${state.dataset.id}  (n≈${state.dataset.nSamples ?? 200}, in=${state.dataset.inputDim ?? 2}, classes=${state.dataset.nClasses ?? 2})`);
  lines.push(`compile: ${state.compile.optimizer}  lr=${state.compile.lr}  loss=${state.compile.loss}`);
  lines.push(`layers (${state.layers.length}):`);
  state.layers.forEach((l, i) => {
    lines.push(`  ${i + 1}. ${l.kind} ${JSON.stringify({ ...l, kind: undefined, id: undefined })}`);
  });
  const last = state.training.history[state.training.history.length - 1];
  if (last) {
    lines.push(`trained: ${state.training.epochsTrained} epochs  loss=${last.loss.toFixed(4)}  acc=${(last.accuracy ?? 0).toFixed(3)}`);
  } else {
    lines.push('trained: (not yet)');
  }
  return lines.join('\n');
}

export const HELP_TEXT = `learn-dl commands
  net dense <units> [act]     add Dense layer (relu|sigmoid|tanh|softmax|gelu|silu)
  net dropout <rate>          add Dropout
  net batchnorm | layernorm   normalization
  net flatten | gap           reshape / pool
  net conv <f> [k] | net pool [k]
  net embedding <vocab> <dim>
  net lstm|gru|rnn <units> [return_sequences]
  net attention <dim> | net mha <heads> <dim>
  net posenc                  positional encoding
  clearnet                    clear architecture
  data xor|moons|circles|spiral|linear|regression|seq-copy|seq-add|mnist-toy
  compile [opt] [lr] [loss]   optimizer / lr / loss
  optimizer sgd|momentum|adam|adamw|rmsprop
  lr <number>
  loss mse|mae|bce|ce|sparsece
  train <epochs>              run training
  library pytorch|tensorflow|keras
  levels | hint | show goal | show solution
  build                       open visual builder
  status | info               show experiment
  undo | reset | clear | help`;
