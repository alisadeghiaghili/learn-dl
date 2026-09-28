import { describe, expect, it } from 'vitest';
import { generateAllCode } from '../src/codegen';
import { parsePythonSubset } from '../src/codeparse/subset';
import { emptyExperiment } from '../src/state/types';

describe('codegen', () => {
  it('emits three libraries', () => {
    const state = emptyExperiment();
    state.layers = [
      { kind: 'dense', units: 16, activation: 'relu', inputDim: 2 },
      { kind: 'dense', units: 2, activation: 'softmax' },
    ];
    const all = generateAllCode(state);
    expect(all.pytorch.source).toContain('nn.Linear');
    expect(all.tensorflow.source).toContain('GradientTape');
    expect(all.keras.source).toContain('keras.Sequential');
    expect(all.keras.source).toContain('model.compile');
  });
});

describe('codeparse', () => {
  it('parses keras sequential layers', () => {
    const src = `
import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(2,)),
    keras.layers.Dense(2, activation='softmax'),
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=12, batch_size=32)
`;
    const { state } = parsePythonSubset(src);
    expect(state.library).toBe('keras');
    expect(state.layers.length).toBeGreaterThanOrEqual(2);
    expect(state.layers[0]?.kind).toBe('dense');
    expect(state.compile.optimizer).toBe('adam');
    expect(state.training.epochsTrained).toBe(12);
  });

  it('parses nn.Module linears', () => {
    const src = `
import torch
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
`;
    const { state } = parsePythonSubset(src);
    expect(state.library).toBe('pytorch');
    expect(state.layers.length).toBe(2);
    expect(state.compile.optimizer).toBe('adam');
  });
});
