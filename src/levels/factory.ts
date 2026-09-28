import {
  emptyExperiment,
  type ExperimentState,
  type LayerSpec,
} from '../state/types';
import type { DialogView, GoalSpec, I18nText, Level } from './types';

export function T(en_US: string, fa?: string): I18nText {
  return fa ? { en_US, fa } : { en_US };
}

export function startFrom(
  patch: Partial<ExperimentState> & { layers?: LayerSpec[] } = {},
): ExperimentState {
  const base = emptyExperiment();
  return {
    ...base,
    ...patch,
    dataset: { ...base.dataset, ...(patch.dataset ?? {}) },
    compile: { ...base.compile, ...(patch.compile ?? {}), metrics: patch.compile?.metrics ?? base.compile.metrics },
    training: {
      ...base.training,
      ...(patch.training ?? {}),
      history: [],
      epochsTrained: 0,
    },
    layers: patch.layers ?? [],
  };
}

export function makeLevel(level: Level): Level {
  return level;
}

export function alert(...markdowns: string[]): DialogView {
  return { type: 'ModalAlert', options: { markdowns } };
}

export function demo(
  beforeMarkdowns: string[],
  command: string,
  afterMarkdowns: string[],
): DialogView {
  return { type: 'DemoView', options: { beforeMarkdowns, command, afterMarkdowns } };
}

export function codeCompare(
  markdowns: string[],
  highlight?: 'pytorch' | 'tensorflow' | 'keras',
): DialogView {
  return { type: 'CodeCompare', options: { markdowns, highlight } };
}

export function goal(patch: GoalSpec): GoalSpec {
  return patch;
}

export function dense(
  units: number,
  activation?: LayerSpec['activation'],
  inputDim?: number,
): LayerSpec {
  return { kind: 'dense', units, activation: activation ?? 'relu', inputDim };
}

export function dropout(rate: number): LayerSpec {
  return { kind: 'dropout', rate };
}

export function bn(): LayerSpec {
  return { kind: 'batchnorm' };
}

export function ln(): LayerSpec {
  return { kind: 'layerNorm' };
}

export function conv(filters: number, kernelSize = 3): LayerSpec {
  return { kind: 'conv2d', units: filters, kernelSize, activation: 'relu' };
}

export function pool(kernelSize = 2): LayerSpec {
  return { kind: 'maxpool2d', kernelSize };
}

export function flatten(): LayerSpec {
  return { kind: 'flatten' };
}

export function gap(): LayerSpec {
  return { kind: 'globalAveragePool' };
}

export function lstm(units: number, returnSequences = false): LayerSpec {
  return { kind: 'lstm', units, returnSequences };
}

export function gru(units: number, returnSequences = false): LayerSpec {
  return { kind: 'gru', units, returnSequences };
}

export function rnn(units: number, returnSequences = false): LayerSpec {
  return { kind: 'simpleRNN', units, returnSequences };
}

export function embedding(vocabSize: number, embedDim: number): LayerSpec {
  return { kind: 'embedding', vocabSize, embedDim };
}

export function attention(units = 32): LayerSpec {
  return { kind: 'attention', units };
}

export function mha(heads = 4, units = 32): LayerSpec {
  return { kind: 'multiheadAttention', heads, units };
}

export function posenc(): LayerSpec {
  return { kind: 'positionalEncoding' };
}
