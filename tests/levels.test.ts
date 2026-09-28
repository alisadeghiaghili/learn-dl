import { describe, expect, it } from 'vitest';
import { goalSatisfied, layersEqual } from '../src/levels/goalMatch';
import { levels, levelsBySequence, sequenceInfo } from '../src/levels';
import { emptyExperiment } from '../src/state/types';

describe('curriculum', () => {
  it('has levels in every sequence', () => {
    for (const seq of sequenceInfo) {
      expect(levelsBySequence(seq.id).length).toBeGreaterThanOrEqual(3);
    }
  });

  it('each level has a solution and unique id', () => {
    const ids = new Set<string>();
    for (const level of levels) {
      expect(level.solutionCommand.length).toBeGreaterThan(0);
      expect(ids.has(level.id)).toBe(false);
      ids.add(level.id);
    }
  });
});

describe('goalMatch', () => {
  it('compares layers ignoring ids', () => {
    expect(
      layersEqual(
        [{ kind: 'dense', units: 2, activation: 'relu', id: 'a' }],
        [{ kind: 'dense', units: 2, activation: 'relu', id: 'b' }],
      ),
    ).toBe(true);
  });

  it('checks metric thresholds', () => {
    const state = emptyExperiment();
    state.training.history.push({ epoch: 1, loss: 0.1, accuracy: 0.9 });
    state.training.epochsTrained = 1;
    expect(
      goalSatisfied({ metrics: { accuracyAtLeast: 0.85, lossAtMost: 0.5 } }, state),
    ).toBe(true);
    expect(goalSatisfied({ metrics: { accuracyAtLeast: 0.99 } }, state)).toBe(false);
  });
});
