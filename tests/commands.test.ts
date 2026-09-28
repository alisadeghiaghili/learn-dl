import { describe, expect, it } from 'vitest';
import { executeCommand } from '../src/commands/dsl';
import { emptyExperiment } from '../src/state/types';

describe('dsl', () => {
  it('adds dense layers and records shapes', () => {
    let state = emptyExperiment();
    let r = executeCommand('net dense 8 relu', { state, undoStack: [] });
    expect(r.ok).toBe(true);
    state = r.state;
    expect(state.layers).toHaveLength(1);
    expect(state.layers[0]?.units).toBe(8);
    r = executeCommand('net dense 2 softmax', { state, undoStack: [] });
    state = r.state;
    expect(state.layers).toHaveLength(2);
    expect(state.layers[1]?.activation).toBe('softmax');
  });

  it('chains commands with semicolons', () => {
    const r = executeCommand('net dense 16 relu; net dense 2 softmax; optimizer adam', {
      state: emptyExperiment(),
      undoStack: [],
    });
    expect(r.ok).toBe(true);
    expect(r.state.layers).toHaveLength(2);
    expect(r.state.compile.optimizer).toBe('adam');
  });

  it('rejects unknown commands', () => {
    const r = executeCommand('frobnicate 3', { state: emptyExperiment(), undoStack: [] });
    expect(r.ok).toBe(false);
  });

  it('sets dataset and loss', () => {
    let r = executeCommand('data moons', { state: emptyExperiment(), undoStack: [] });
    expect(r.state.dataset.id).toBe('moons');
    r = executeCommand('loss bce', { state: r.state, undoStack: [] });
    expect(r.state.compile.loss).toBe('binaryCrossentropy');
  });

  it('respects disabled commands in levels', () => {
    const r = executeCommand('reset', {
      state: emptyExperiment(),
      undoStack: [],
      disabled: { reset: true },
    });
    expect(r.ok).toBe(false);
  });
});
