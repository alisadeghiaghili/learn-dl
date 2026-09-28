import { levels } from './curriculum';
import { sequenceInfo } from './sequenceInfo';
import type { Level } from './types';

export { levels, sequenceInfo };
export type { Level, SequenceInfo, GoalSpec, DialogView, I18nText } from './types';
export { goalSatisfied, describeGoal, layersEqual } from './goalMatch';

export function getLevel(id: string): Level | undefined {
  return levels.find((l) => l.id === id);
}

export function levelsBySequence(sequenceId: string): Level[] {
  return levels.filter((l) => l.sequence === sequenceId);
}

export function nextLevel(id: string): Level | undefined {
  const i = levels.findIndex((l) => l.id === id);
  return i >= 0 ? levels[i + 1] : undefined;
}

export function sequenceProgress(
  sequenceId: string,
  solved: Record<string, true>,
): { solved: number; total: number } {
  const list = levelsBySequence(sequenceId);
  return {
    solved: list.filter((l) => solved[l.id]).length,
    total: list.length,
  };
}
