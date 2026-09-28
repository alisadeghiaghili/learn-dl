import { describe, expect, it } from 'vitest';
import { makeDataset } from '../src/engine/datasets';
import { forward } from '../src/engine/model';
import { train } from '../src/engine/trainer';
import { matmul, tensor, zeros } from '../src/engine/tensor';
import { computeLoss } from '../src/engine/losses';

describe('tensor', () => {
  it('multiplies matrices', () => {
    const a = tensor([1, 2, 3, 4], [2, 2]);
    const b = tensor([5, 6, 7, 8], [2, 2]);
    const c = matmul(a, b);
    expect(c.shape).toEqual([2, 2]);
    expect(Array.from(c.data)).toEqual([19, 22, 43, 50]);
  });

  it('zeros allocate full size', () => {
    expect(zeros([2, 3]).data.length).toBe(6);
  });
});

describe('loss', () => {
  it('computes CE and accuracy on correct logits', () => {
    const pred = tensor([4, 0, 0, 4], [2, 2]);
    const target = tensor([1, 0, 0, 1], [2, 2]);
    const { loss, accuracy } = computeLoss('categoricalCrossentropy', pred, target);
    expect(loss).toBeLessThan(0.2);
    expect(accuracy).toBe(1);
  });
});

describe('trainer', () => {
  it('learns XOR with a small MLP', () => {
    const data = makeDataset({ id: 'xor', nSamples: 200, noise: 0.0, seed: 7, nClasses: 2, inputDim: 2 });
    const { result, model } = train(data, {
      layers: [
        { kind: 'dense', units: 8, activation: 'relu', inputDim: 2 },
        { kind: 'dense', units: 2, activation: 'softmax' },
      ],
      compile: { optimizer: 'adam', lr: 0.05, loss: 'categoricalCrossentropy', metrics: ['accuracy'] },
      batch: 32,
      epochs: 80,
      inputDim: 2,
      outputDim: 2,
      seed: 7,
    });
    const last = result.history[result.history.length - 1];
    expect(last?.loss).toBeLessThan(0.4);
    expect(last?.accuracy ?? 0).toBeGreaterThan(0.85);
    const { output } = forward(model, data.x, false);
    expect(output.shape[1]).toBe(2);
  });
});
