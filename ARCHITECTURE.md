# Architecture — learn-dl

Interactive deep learning visualizer, sandbox, and leveled tutorial inspired by
[learnGitBranching](https://github.com/pcottle/learnGitBranching).

## Product mapping (LGB → learn-dl)

| LGB | learn-dl | Notes |
|-----|----------|-------|
| Git tree state | `ExperimentState` | architecture, dataset, compile, training history |
| Git commands | DSL (`net` / `data` / `compile` / `train` / …) | mutates state with visual feedback |
| Commit tree canvas | Network graph + curves + decision boundary | live update |
| Levels (`startTree` / `goalTreeString`) | `startState` / `goal` matchers + asserts | structural + metric goals |
| `solutionCommand` | `solutionCommand` + `solutionCode` (3 libs) | |
| TreeCompare | `goalMatch.ts` | architecture deep-equal, asserts, metric thresholds |
| Sandbox | Sandbox mode | free play |
| Level builder | Visual builder + export level JSON | |
| i18n strings | `src/i18n` | en_US + fa |

## Runtime model

**JS simulation + three code panels.** Training runs in a small TypeScript engine
(MLP + selected CNN/RNN primitives on toy datasets). Real TensorFlow / PyTorch / Keras
code is **generated and shown**; a Python **subset parser** maps common patterns back
into `ExperimentState` so editing code updates the sandbox. We do not execute CPython
in the browser.

```
┌─────────────┐   ┌──────────────┐   ┌──────────────────┐
│ DSL terminal│   │ VisualBuilder│   │ Python subset    │
└──────┬──────┘   └──────┬───────┘   │ parser (3 libs)  │
       │                 │           └────────┬─────────┘
       └────────────┬────┴────────────────────┘
                    ▼
            Command / State API
                    ▼
             ExperimentState
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   Engine       GoalMatch     Codegen
   (train)      (levels)      (pt/tf/keras)
       │            │            │
       ▼            ▼            ▼
   Visuals      Level UI      CodePanels
```

## Packages (under `src/`)

| Path | Responsibility |
|------|----------------|
| `engine/` | pure DL simulation — tensors, layers, losses, optimizers, datasets, trainer |
| `state/` | `ExperimentState` types + immutable-ish store |
| `commands/` | DSL grammar, parser, command handlers, undo stack |
| `codeparse/` | Python subset → state (Sequential / Module-style) |
| `codegen/` | state → PyTorch / TensorFlow / Keras source |
| `viz/` | canvas/SVG visuals |
| `levels/` | level schema, sequences, definitions |
| `ui/` | React shell (terminal, panels, builder, level chrome) |
| `i18n/` | locale strings |

Dependency rule: `engine` and `state` have **no** React imports. `codegen` and
`codeparse` depend on `state` only. `ui` sits on top. Tests target `engine`,
`commands`, `codegen`, `levels`.

## ExperimentState (canonical)

```ts
ExperimentState {
  library: 'pytorch' | 'tensorflow' | 'keras'   // preferred codegen view
  dataset: DatasetSpec                          // xor | moons | circles | spiral | seq-copy | …
  layers: LayerSpec[]                           // ordered architecture
  compile: { optimizer, lr, loss, metrics[] }
  training: { epochsTrained, history[], weights? }
}
```

## Level schema

```ts
Level {
  id, sequence, name: i18n, hint: i18n
  startState: ExperimentState
  goal: {
    layers?: LayerSpec[]            // exact architecture match (order-sensitive)
    asserts?: (s: ExperimentState) => boolean
    metrics?: { accuracyAtLeast?, lossAtMost?, epochsAtLeast? }
  }
  solutionCommand: string           // DSL
  solutionCode: { pytorch, tensorflow, keras }
  startDialog?: DialogView[]        // ModalAlert | Demo | CodeCompare
  disabledCommands?: Record<string, true>
  golf?: { bestCommands: number }
}
```

## Curriculum map

| Sequence | Concepts | Library track |
|----------|----------|---------------|
| `foundations` | tensors, shapes, broadcasting, autograd, perceptron | all three side-by-side |
| `mlp` | dense, activations, loss, optimizers, train loop, batching | all three |
| `regularization` | dropout, batchnorm, weight decay, early stop, val split | all three |
| `cnn` | conv, pool, flatten, LeNet-style stack | all three |
| `sequence` | RNN/LSTM/GRU, seq labeling, encoder-decoder ideas | all three |
| `attention` | attention, multi-head, transformer block, positional enc | all three |
| `training` | LR schedules, init, mixed precision concept, grad clip | all three |
| `production` | transfer learning, fine-tune, export/inference | all three |
| `pytorch` | Module, autograd, DataLoader, custom loops | PyTorch depth |
| `tensorflow` | tf.function, Keras subclass, tf.data, distribution | TF depth |
| `keras` | Sequential/Functional, callbacks, preprocessing layers | Keras depth |

## Command surface (all three live)

1. **DSL terminal** — authoritative mutator for levels/sandbox.
2. **Python editor** — subset parser → state; codegen reverse path for common APIs.
3. **Visual builder** — form/graph edits → state (same store as 1).

## Explicit non-goals (v0)

- Full GPU / large-model training
- Arbitrary Python execution
- Exact bitwise parity with real TF/PT training
- Production export of real SavedModel / TorchScript binaries

The engine teaches **mechanics and APIs**, not cluster-scale numerics.
