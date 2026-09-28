import type { EpochPoint } from '../state/types';

export interface TrainCurvesProps {
  history: EpochPoint[];
  width?: number;
  height?: number;
}

/**
 * Loss and accuracy curves across epochs.
 */
export function TrainCurves({ history, width = 420, height = 220 }: TrainCurvesProps) {
  const pad = { l: 40, r: 16, t: 16, b: 28 };
  const w = width - pad.l - pad.r;
  const h = height - pad.t - pad.b;
  const pts = history;
  const maxEpoch = Math.max(1, pts[pts.length - 1]?.epoch ?? 1);
  const losses = pts.map((p) => p.loss);
  const maxLoss = Math.max(0.1, ...losses);

  const xAt = (epoch: number) => pad.l + ((epoch - 1) / Math.max(1, maxEpoch - 1 || 1)) * w;
  const yLoss = (v: number) => pad.t + h - (v / maxLoss) * h;
  const yAcc = (v: number) => pad.t + h - v * h;

  const lossPath = pts
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xAt(p.epoch).toFixed(1)} ${yLoss(p.loss).toFixed(1)}`)
    .join(' ');
  const accPath = pts
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${xAt(p.epoch).toFixed(1)} ${yAcc(p.accuracy ?? 0).toFixed(1)}`)
    .join(' ');

  return (
    <svg width={width} height={height} className="viz-curves" role="img" aria-label="Training curves">
      <rect x={0} y={0} width={width} height={height} fill="#0B1220" />
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + h} stroke="#1C2538" strokeWidth="1" />
      <line x1={pad.l} y1={pad.t + h} x2={pad.l + w} y2={pad.t + h} stroke="#1C2538" strokeWidth="1" />
      <text x={pad.l} y={12} fill="#22D3EE" fontSize="11" fontFamily="JetBrains Mono, monospace">
        loss
      </text>
      <text x={pad.l + 48} y={12} fill="#34D399" fontSize="11" fontFamily="JetBrains Mono, monospace">
        accuracy
      </text>
      <text x={width - 8} y={height - 8} textAnchor="end" fill="#8B9BB4" fontSize="10" fontFamily="JetBrains Mono, monospace">
        epoch → {maxEpoch}
      </text>
      {pts.length > 0 ? (
        <>
          <path d={lossPath} fill="none" stroke="#22D3EE" strokeWidth="2" />
          <path d={accPath} fill="none" stroke="#34D399" strokeWidth="2" />
        </>
      ) : (
        <text x={width / 2} y={height / 2} textAnchor="middle" fill="#8B9BB4" fontSize="12" fontFamily="Inter, sans-serif">
          train to draw curves
        </text>
      )}
    </svg>
  );
}
