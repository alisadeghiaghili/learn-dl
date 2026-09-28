import type { ExperimentState, LayerSpec } from '../state/types';
import type { CodeBundle, CodegenOptions } from './types';

const ACT_FN: Record<string, string> = {
  relu: 'torch.relu',
  sigmoid: 'torch.sigmoid',
  tanh: 'torch.tanh',
  gelu: 'torch.nn.functional.gelu',
  silu: 'torch.nn.functional.silu',
  softmax: 'torch.softmax',
  linear: '',
};

function moduleLayers(layers: LayerSpec[]): string[] {
  const lines: string[] = [];
  let prev = 0;
  for (const [i, layer] of layers.entries()) {
    switch (layer.kind) {
      case 'dense': {
        if (prev === 0 && layer.inputDim) prev = layer.inputDim;
        const units = layer.units ?? 32;
        lines.push(`        self.fc${i} = nn.Linear(${prev || 2}, ${units})`);
        if (layer.activation && layer.activation !== 'linear' && layer.activation !== 'softmax') {
          // applied in forward
        }
        prev = units;
        break;
      }
      case 'dropout':
        lines.push(`        self.drop${i} = nn.Dropout(${layer.rate ?? 0.5})`);
        break;
      case 'batchnorm':
        lines.push(`        self.bn${i} = nn.BatchNorm1d(${prev || 2})`);
        break;
      case 'layerNorm':
        lines.push(`        self.ln${i} = nn.LayerNorm(${prev || 2})`);
        break;
      case 'conv2d':
        lines.push(
          `        self.conv${i} = nn.Conv2d(3, ${layer.units ?? 8}, ${layer.kernelSize ?? 3})`,
        );
        prev = layer.units ?? 8;
        break;
      case 'maxpool2d':
        lines.push(`        self.pool${i} = nn.MaxPool2d(${layer.kernelSize ?? 2})`);
        break;
      case 'flatten':
        lines.push(`        self.flat${i} = nn.Flatten()`);
        break;
      case 'embedding':
        lines.push(
          `        self.emb${i} = nn.Embedding(${layer.vocabSize ?? 100}, ${layer.embedDim ?? layer.units ?? 16})`,
        );
        prev = layer.embedDim ?? layer.units ?? 16;
        break;
      case 'lstm':
        lines.push(
          `        self.lstm${i} = nn.LSTM(${prev || 16}, ${layer.units ?? 32}, batch_first=True)`,
        );
        prev = layer.returnSequences ? (layer.units ?? 32) : (layer.units ?? 32);
        break;
      case 'gru':
        lines.push(
          `        self.gru${i} = nn.GRU(${prev || 16}, ${layer.units ?? 32}, batch_first=True)`,
        );
        prev = layer.units ?? 32;
        break;
      case 'simpleRNN':
        lines.push(
          `        self.rnn${i} = nn.RNN(${prev || 16}, ${layer.units ?? 32}, batch_first=True)`,
        );
        prev = layer.units ?? 32;
        break;
      case 'attention':
      case 'multiheadAttention':
        lines.push(
          `        self.attn${i} = nn.MultiheadAttention(embed_dim=${layer.units ?? 32}, num_heads=${layer.heads ?? 4}, batch_first=True)`,
        );
        prev = layer.units ?? 32;
        break;
      default:
        lines.push(`        # ${layer.kind} ${i}`);
    }
  }
  return lines;
}

function forwardLines(layers: LayerSpec[]): string[] {
  const lines: string[] = [];
  for (const [i, layer] of layers.entries()) {
    if (layer.kind === 'dense') {
      lines.push(`        x = self.fc${i}(x)`);
      const act = layer.activation && layer.activation !== 'linear' ? ACT_FN[layer.activation] : '';
      if (act && layer.activation !== 'softmax') lines.push(`        x = ${act}(x)`);
    } else if (layer.kind === 'dropout') {
      lines.push(`        x = self.drop${i}(x)`);
    } else if (layer.kind === 'batchnorm') {
      lines.push(`        x = self.bn${i}(x)`);
    } else if (layer.kind === 'layerNorm') {
      lines.push(`        x = self.ln${i}(x)`);
    } else if (layer.kind === 'conv2d') {
      lines.push(`        x = torch.relu(self.conv${i}(x))`);
    } else if (layer.kind === 'maxpool2d') {
      lines.push(`        x = self.pool${i}(x)`);
    } else if (layer.kind === 'flatten') {
      lines.push(`        x = self.flat${i}(x)`);
    } else if (layer.kind === 'embedding') {
      lines.push(`        x = self.emb${i}(x)`);
    } else if (layer.kind === 'lstm') {
      lines.push(`        x, _ = self.lstm${i}(x)`);
    } else if (layer.kind === 'gru') {
      lines.push(`        x, _ = self.gru${i}(x)`);
    } else if (layer.kind === 'simpleRNN') {
      lines.push(`        x, _ = self.rnn${i}(x)`);
    } else if (layer.kind === 'attention' || layer.kind === 'multiheadAttention') {
      lines.push(`        x, _ = self.attn${i}(x, x, x)`);
    }
  }
  return lines;
}

const LOSS_FN: Record<string, string> = {
  mse: 'nn.MSELoss()',
  mae: 'nn.L1Loss()',
  binaryCrossentropy: 'nn.BCELoss()',
  categoricalCrossentropy: 'nn.CrossEntropyLoss()',
  sparseCategoricalCrossentropy: 'nn.CrossEntropyLoss()',
};

/**
 * Generate idiomatic PyTorch source (nn.Module + training loop).
 */
export function generatePyTorch(
  state: ExperimentState,
  options: CodegenOptions = {},
): CodeBundle {
  const layers = state.layers;
  const decls = moduleLayers(layers);
  const fwd = forwardLines(layers);
  const opt =
    state.compile.optimizer === 'sgd'
      ? `torch.optim.SGD(model.parameters(), lr=${state.compile.lr})`
      : state.compile.optimizer === 'momentum'
        ? `torch.optim.SGD(model.parameters(), lr=${state.compile.lr}, momentum=0.9)`
        : state.compile.optimizer === 'rmsprop'
          ? `torch.optim.RMSprop(model.parameters(), lr=${state.compile.lr})`
          : state.compile.optimizer === 'adamw'
            ? `torch.optim.AdamW(model.parameters(), lr=${state.compile.lr})`
            : `torch.optim.Adam(model.parameters(), lr=${state.compile.lr})`;

  const lines: string[] = [];
  lines.push('import torch');
  lines.push('import torch.nn as nn');
  lines.push('from torch.utils.data import DataLoader, TensorDataset');
  lines.push('');
  lines.push('class Net(nn.Module):');
  lines.push('    def __init__(self):');
  lines.push('        super().__init__()');
  if (decls.length === 0) {
    lines.push(`        self.fc1 = nn.Linear(${state.dataset.inputDim ?? 2}, 32)`);
    lines.push(`        self.fc2 = nn.Linear(32, ${state.dataset.nClasses ?? 2})`);
  } else {
    lines.push(...decls);
  }
  lines.push('');
  lines.push('    def forward(self, x):');
  if (fwd.length === 0) {
    lines.push('        x = torch.relu(self.fc1(x))');
    lines.push('        x = self.fc2(x)');
  } else {
    lines.push(...fwd);
  }
  lines.push('        return x');
  lines.push('');
  lines.push('model = Net()');
  lines.push(`criterion = ${LOSS_FN[state.compile.loss] ?? 'nn.CrossEntropyLoss()'}`);
  lines.push(`optimizer = ${opt}`);
  lines.push('');
  lines.push('dataset = TensorDataset(X_train, y_train)');
  lines.push(`loader = DataLoader(dataset, batch_size=${state.training.batch || 32}, shuffle=True)`);
  lines.push('');
  lines.push('for epoch in range(1, ' + Math.max(1, state.training.epochsTrained || 10) + ' + 1):');
  lines.push('    model.train()');
  lines.push('    total_loss = 0.0');
  lines.push('    for xb, yb in loader:');
  lines.push('        optimizer.zero_grad()');
  lines.push('        pred = model(xb)');
  lines.push('        loss = criterion(pred, yb)');
  lines.push('        loss.backward()');
  lines.push('        optimizer.step()');
  lines.push('        total_loss += loss.item()');
  lines.push("    print(f'epoch {epoch} loss {total_loss:.4f}')");

  if (options.withData !== false) {
    lines.push('');
    lines.push('# Load synthetic learn-dl dataset into X_train, y_train first.');
  }

  return {
    library: 'pytorch',
    title: 'PyTorch',
    language: 'python',
    source: lines.join('\n'),
  };
}
