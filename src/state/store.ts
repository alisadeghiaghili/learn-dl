import { cloneExperiment, emptyExperiment, type ExperimentState, type LibraryId } from './types';

export type Listener = (state: ExperimentState) => void;

export interface AppStore {
  getState(): ExperimentState;
  setState(next: ExperimentState): void;
  update(fn: (s: ExperimentState) => ExperimentState): void;
  reset(): void;
  pushUndo(): void;
  undo(): void;
  subscribe(fn: Listener): () => void;
  setLibrary(lib: LibraryId): void;
}

/**
 * Tiny observable store for the experiment state.
 */
export function createStore(initial?: ExperimentState): AppStore {
  let state = cloneExperiment(initial ?? emptyExperiment());
  let undoStack: ExperimentState[] = [];
  const listeners = new Set<Listener>();

  const emit = () => {
    for (const fn of listeners) fn(state);
  };

  return {
    getState: () => state,
    setState(next) {
      state = cloneExperiment(next);
      emit();
    },
    update(fn) {
      state = fn(state);
      emit();
    },
    reset() {
      const lib = state.library;
      state = emptyExperiment();
      state.library = lib;
      emit();
    },
    pushUndo() {
      undoStack.push(cloneExperiment(state));
      if (undoStack.length > 50) undoStack.shift();
    },
    undo() {
      const prev = undoStack.pop();
      if (prev) {
        state = prev;
        emit();
      }
    },
    subscribe(fn) {
      listeners.add(fn);
      return () => listeners.delete(fn);
    },
    setLibrary(lib) {
      state = { ...cloneExperiment(state), library: lib };
      emit();
    },
  };
}
