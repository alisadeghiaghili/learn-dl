import { useEffect, useRef } from 'react';
import type { FullData } from '../engine/datasets';
import type { CompiledModel } from '../engine/model';
import { makePredictor } from '../engine/trainer';

export interface DecisionBoundaryProps {
  model: CompiledModel | null;
  data: FullData | null;
  width?: number;
  height?: number;
}

/**
 * 2D decision field under the toy classification points.
 */
export function DecisionBoundary({
  model,
  data,
  width = 420,
  height = 220,
}: DecisionBoundaryProps) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const grid = 48;
    const cellW = width / grid;
    const cellH = height / grid;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#0B1220';
    ctx.fillRect(0, 0, width, height);

    const toWorld = (px: number, py: number): [number, number] => {
      const x = (px / width) * 4 - 2;
      const y = (py / height) * 4 - 2;
      return [x, y];
    };

    if (model && data?.points2d) {
      const predict = makePredictor(model);
      const nClasses = data.nClasses || 2;
      const palette = ['rgba(52,211,153,0.18)', 'rgba(248,113,113,0.18)', 'rgba(56,189,248,0.18)', 'rgba(251,191,36,0.18)'];
      for (let i = 0; i < grid; i++) {
        for (let j = 0; j < grid; j++) {
          const [x, y] = toWorld((i + 0.5) * cellW, (j + 0.5) * cellH);
          const cls = predict(x, y) % nClasses;
          ctx.fillStyle = palette[cls] ?? palette[0] ?? '#333';
          ctx.fillRect(i * cellW, j * cellH, cellW + 1, cellH + 1);
        }
      }
      for (const p of data.points2d) {
        const px = ((p.x + 2) / 4) * width;
        const py = ((p.y + 2) / 4) * height;
        ctx.beginPath();
        ctx.arc(px, py, 3, 0, Math.PI * 2);
        ctx.fillStyle = p.label === 0 ? '#E8EEF7' : p.label === 1 ? '#FBBF24' : '#22D3EE';
        ctx.fill();
      }
    } else if (data?.points2d) {
      for (const p of data.points2d) {
        const px = ((p.x + 2) / 4) * width;
        const py = ((p.y + 2) / 4) * height;
        ctx.beginPath();
        ctx.arc(px, py, 3, 0, Math.PI * 2);
        ctx.fillStyle = p.label === 0 ? '#E8EEF7' : p.label === 1 ? '#FBBF24' : '#22D3EE';
        ctx.fill();
      }
    } else {
      ctx.fillStyle = '#8B9BB4';
      ctx.font = '12px Inter, sans-serif';
      ctx.fillText('2D datasets only', 16, height / 2);
    }
  }, [model, data, width, height]);

  return (
    <canvas
      ref={ref}
      width={width}
      height={height}
      className="viz-boundary"
      aria-label="Decision boundary"
    />
  );
}
