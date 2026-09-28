import type { ExperimentState } from '../state/types';
import { countParams, layerLabel, activationSuffix } from '../state/types';

export interface NetworkGraphProps {
  state: ExperimentState;
  width?: number;
  height?: number;
}

/**
 * Architecture graph: input → layers → head, with shape badges.
 */
export function NetworkGraph({ state, width = 520, height = 360 }: NetworkGraphProps) {
  const layers = state.layers;
  const inputDim = state.dataset.inputDim ?? 2;
  const nodes = [
    { label: `Input(${inputDim})`, sub: 'features', kind: 'input' },
    ...layers.map((l) => ({
      label: layerLabel(l) + activationSuffix(l),
      sub: l.kind,
      kind: l.kind,
    })),
    { label: 'Output', sub: `${state.dataset.nClasses ?? 2} classes`, kind: 'output' },
  ];

  const n = Math.max(nodes.length, 1);
  const padY = 28;
  const gap = Math.min(56, (height - padY * 2) / n);
  const startY = (height - gap * (n - 1)) / 2;
  const cx = width * 0.28;

  const colors: Record<string, string> = {
    input: '#22D3EE',
    dense: '#34D399',
    dropout: '#FBBF24',
    batchnorm: '#A78BFA',
    layerNorm: '#A78BFA',
    conv2d: '#38BDF8',
    maxpool2d: '#60A5FA',
    globalAveragePool: '#60A5FA',
    flatten: '#8B9BB4',
    embedding: '#F472B6',
    lstm: '#FB923C',
    gru: '#FB923C',
    simpleRNN: '#FB923C',
    attention: '#E879F9',
    multiheadAttention: '#E879F9',
    positionalEncoding: '#C084FC',
    output: '#F87171',
  };

  return (
    <svg width={width} height={height} className="viz-network" role="img" aria-label="Network architecture">
      <defs>
        <marker id="arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L8,4 L0,8" fill="none" stroke="#8B9BB4" strokeWidth="1" />
        </marker>
      </defs>
      {nodes.map((node, i) => {
        const y = startY + i * gap;
        const prevY = i === 0 ? y : startY + (i - 1) * gap;
        const h = 44;
        const w = 280;
        const x = cx - w / 2;
        const c = colors[node.kind] ?? '#8B9BB4';
        return (
          <g key={`${node.label}-${i}`}>
            {i > 0 && (
              <line
                x1={cx}
                y1={prevY + h}
                x2={cx}
                y2={y - 2}
                stroke="#8B9BB4"
                strokeWidth="1"
                markerEnd="url(#arrow)"
              />
            )}
            <rect x={x} y={y} width={w} height={h} rx={8} fill="#1C2538" stroke={c} strokeWidth="1" />
            <text
              x={cx}
              y={y + 18}
              textAnchor="middle"
              fill="#E8EEF7"
              fontSize="13"
              fontFamily="JetBrains Mono, ui-monospace, monospace"
            >
              {node.label}
            </text>
            <text
              x={cx}
              y={y + 34}
              textAnchor="middle"
              fill="#8B9BB4"
              fontSize="11"
              fontFamily="JetBrains Mono, ui-monospace, monospace"
            >
              {node.sub}
            </text>
          </g>
        );
      })}
      <text
        x={width - 12}
        y={18}
        textAnchor="end"
        fill="#8B9BB4"
        fontSize="11"
        fontFamily="JetBrains Mono, ui-monospace, monospace"
      >
        params ≈ {countParams(state.layers, inputDim)}
      </text>
    </svg>
  );
}
