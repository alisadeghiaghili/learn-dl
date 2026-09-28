import type { ActivationName, ExperimentState, LayerKind, LayerSpec } from '../state/types';

export interface VisualBuilderProps {
  state: ExperimentState;
  onChange: (next: ExperimentState) => void;
}

const KINDS: LayerKind[] = [
  'dense',
  'dropout',
  'batchnorm',
  'layerNorm',
  'flatten',
  'conv2d',
  'maxpool2d',
  'globalAveragePool',
  'embedding',
  'lstm',
  'gru',
  'simpleRNN',
  'attention',
  'multiheadAttention',
  'positionalEncoding',
];

const ACTS: ActivationName[] = ['linear', 'relu', 'sigmoid', 'tanh', 'softmax', 'gelu', 'silu'];

/**
 * Form-based architecture builder — third command surface.
 */
export function VisualBuilder({ state, onChange }: VisualBuilderProps) {
  const layers = state.layers;

  const updateLayers = (next: LayerSpec[]) => {
    onChange({
      ...state,
      layers: next.map((l, i) => ({ ...l, id: `L${i}` })),
      training: { ...state.training, epochsTrained: 0, history: [] },
    });
  };

  const add = (kind: LayerKind) => {
    const base: LayerSpec = { kind };
    if (kind === 'dense') {
      base.units = 32;
      base.activation = 'relu';
      if (layers.length === 0) base.inputDim = state.dataset.inputDim ?? 2;
    }
    if (kind === 'dropout') base.rate = 0.3;
    if (kind === 'conv2d') {
      base.units = 8;
      base.kernelSize = 3;
      base.activation = 'relu';
    }
    if (kind === 'maxpool2d') base.kernelSize = 2;
    if (kind === 'lstm' || kind === 'gru' || kind === 'simpleRNN') base.units = 32;
    if (kind === 'embedding') {
      base.vocabSize = 100;
      base.embedDim = 16;
    }
    if (kind === 'attention' || kind === 'multiheadAttention') {
      base.units = 32;
      base.heads = 4;
    }
    updateLayers([...layers, base]);
  };

  const patch = (idx: number, p: Partial<LayerSpec>) => {
    updateLayers(layers.map((l, i) => (i === idx ? { ...l, ...p } : l)));
  };

  const remove = (idx: number) => {
    updateLayers(layers.filter((_, i) => i !== idx));
  };

  const move = (idx: number, dir: -1 | 1) => {
    const j = idx + dir;
    if (j < 0 || j >= layers.length) return;
    const next = [...layers];
    const a = next[idx];
    const b = next[j];
    if (!a || !b) return;
    next[idx] = b;
    next[j] = a;
    updateLayers(next);
  };

  return (
    <div className="builder">
      <div className="builder-row">
        <label>
          Dataset
          <select
            value={state.dataset.id}
            onChange={(e) =>
              onChange({
                ...state,
                dataset: { ...state.dataset, id: e.target.value as ExperimentState['dataset']['id'] },
                training: { ...state.training, epochsTrained: 0, history: [] },
              })
            }
          >
            {['xor', 'moons', 'circles', 'spiral', 'linear', 'regression', 'seq-copy', 'seq-add', 'mnist-toy'].map(
              (d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ),
            )}
          </select>
        </label>
        <label>
          Optimizer
          <select
            value={state.compile.optimizer}
            onChange={(e) =>
              onChange({
                ...state,
                compile: { ...state.compile, optimizer: e.target.value as ExperimentState['compile']['optimizer'] },
              })
            }
          >
            {['sgd', 'momentum', 'adam', 'adamw', 'rmsprop'].map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
          </select>
        </label>
        <label>
          LR
          <input
            type="number"
            step="0.001"
            value={state.compile.lr}
            onChange={(e) =>
              onChange({ ...state, compile: { ...state.compile, lr: Number(e.target.value) } })
            }
          />
        </label>
        <label>
          Loss
          <select
            value={state.compile.loss}
            onChange={(e) =>
              onChange({
                ...state,
                compile: { ...state.compile, loss: e.target.value as ExperimentState['compile']['loss'] },
              })
            }
          >
            {['mse', 'mae', 'binaryCrossentropy', 'categoricalCrossentropy', 'sparseCategoricalCrossentropy'].map(
              (l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ),
            )}
          </select>
        </label>
      </div>

      <div className="builder-layer-list">
        {layers.map((layer, idx) => (
          <div key={layer.id ?? idx} className="builder-layer">
            <span className="builder-layer-kind">{layer.kind}</span>
            {layer.kind === 'dense' && (
              <>
                <label>
                  units
                  <input
                    type="number"
                    value={layer.units ?? 32}
                    onChange={(e) => patch(idx, { units: Number(e.target.value) })}
                  />
                </label>
                <label>
                  act
                  <select
                    value={layer.activation ?? 'relu'}
                    onChange={(e) => patch(idx, { activation: e.target.value as ActivationName })}
                  >
                    {ACTS.map((a) => (
                      <option key={a} value={a}>
                        {a}
                      </option>
                    ))}
                  </select>
                </label>
              </>
            )}
            {layer.kind === 'dropout' && (
              <label>
                rate
                <input
                  type="number"
                  step="0.05"
                  min={0}
                  max={0.95}
                  value={layer.rate ?? 0.5}
                  onChange={(e) => patch(idx, { rate: Number(e.target.value) })}
                />
              </label>
            )}
            {(layer.kind === 'lstm' || layer.kind === 'gru' || layer.kind === 'simpleRNN' || layer.kind === 'attention') && (
              <label>
                units
                <input
                  type="number"
                  value={layer.units ?? 32}
                  onChange={(e) => patch(idx, { units: Number(e.target.value) })}
                />
              </label>
            )}
            {layer.kind === 'multiheadAttention' && (
              <>
                <label>
                  heads
                  <input
                    type="number"
                    value={layer.heads ?? 4}
                    onChange={(e) => patch(idx, { heads: Number(e.target.value) })}
                  />
                </label>
                <label>
                  dim
                  <input
                    type="number"
                    value={layer.units ?? 32}
                    onChange={(e) => patch(idx, { units: Number(e.target.value) })}
                  />
                </label>
              </>
            )}
            <div className="builder-layer-actions">
              <button className="btn ghost" onClick={() => move(idx, -1)} aria-label="Move up">
                ↑
              </button>
              <button className="btn ghost" onClick={() => move(idx, 1)} aria-label="Move down">
                ↓
              </button>
              <button className="btn danger ghost" onClick={() => remove(idx)} aria-label="Remove layer">
                ×
              </button>
            </div>
          </div>
        ))}
        {layers.length === 0 && <p className="muted">No layers yet — add one below.</p>}
      </div>

      <div className="builder-add">
        {KINDS.map((k) => (
          <button key={k} className="btn ghost" onClick={() => add(k)}>
            + {k}
          </button>
        ))}
      </div>
    </div>
  );
}
