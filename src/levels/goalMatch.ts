import type { ExperimentState, LayerSpec } from '../state/types';
import type { GoalSpec } from './types';

function normalizeLayer(l: LayerSpec | undefined): Record<string, unknown> {
  if (!l) return {};
  const { id: _id, ...rest } = l;
  const out: Record<string, unknown> = { kind: rest.kind };
  for (const [k, v] of Object.entries(rest)) {
    if (v !== undefined) out[k] = v;
  }
  return out;
}

export function layersEqual(a: LayerSpec[], b: LayerSpec[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const la = normalizeLayer(a[i]);
    const lb = normalizeLayer(b[i]);
    if (JSON.stringify(la) !== JSON.stringify(lb)) return false;
  }
  return true;
}

/**
 * Check whether the student's experiment satisfies a level goal.
 *
 * Args:
 *   goal: level goal spec
 *   state: current experiment
 * Returns:
 *   true when all present conditions hold
 */
export function goalSatisfied(goal: GoalSpec, state: ExperimentState): boolean {
  if (goal.layers && !layersEqual(goal.layers, state.layers)) return false;
  if (goal.loss && state.compile.loss !== goal.loss) return false;
  if (goal.optimizer && state.compile.optimizer !== goal.optimizer) return false;
  if (goal.dataset && state.dataset.id !== goal.dataset) return false;

  if (goal.metrics) {
    const last = state.training.history[state.training.history.length - 1];
    if (goal.metrics.epochsAtLeast !== undefined) {
      if (state.training.epochsTrained < goal.metrics.epochsAtLeast) return false;
    }
    if (goal.metrics.accuracyAtLeast !== undefined) {
      if (!last || (last.accuracy ?? 0) < goal.metrics.accuracyAtLeast) return false;
    }
    if (goal.metrics.lossAtMost !== undefined) {
      if (!last || last.loss > goal.metrics.lossAtMost) return false;
    }
  }

  if (goal.asserts) {
    for (const fn of goal.asserts) {
      if (!fn(state)) return false;
    }
  }
  return true;
}

export function describeGoal(goal: GoalSpec): string {
  const lines: string[] = [];
  if (goal.dataset) lines.push(`dataset = ${goal.dataset}`);
  if (goal.layers) {
    lines.push('architecture:');
    goal.layers.forEach((l, i) => {
      lines.push(`  ${i + 1}. ${l.kind} ${l.units ?? ''} ${l.activation ?? ''}`.replace(/\s+/g, ' ').trim());
    });
  }
  if (goal.loss) lines.push(`loss = ${goal.loss}`);
  if (goal.optimizer) lines.push(`optimizer = ${goal.optimizer}`);
  if (goal.metrics) {
    if (goal.metrics.accuracyAtLeast !== undefined) {
      lines.push(`accuracy >= ${goal.metrics.accuracyAtLeast}`);
    }
    if (goal.metrics.lossAtMost !== undefined) {
      lines.push(`loss <= ${goal.metrics.lossAtMost}`);
    }
    if (goal.metrics.epochsAtLeast !== undefined) {
      lines.push(`epochs >= ${goal.metrics.epochsAtLeast}`);
    }
  }
  if (goal.asserts?.length) lines.push(`(${goal.asserts.length} custom assert(s))`);
  return lines.join('\n') || '(any state matches)';
}
