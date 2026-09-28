# learn-dl

Interactive deep learning visualization, sandbox, and leveled tutorial.

Inspired by [learnGitBranching](https://github.com/pcottle/learnGitBranching): type
commands, watch the system change, clear challenges. Here the system is a neural
network experiment — and the same concept is shown in **PyTorch**, **TensorFlow**,
and **Keras**.

## What you get

- **Sandbox** with a live network graph, training curves, and 2D decision boundary
- **Levels** across the full DL curriculum (tensors → transformers → production)
- **Three command surfaces**: DSL terminal, Python editor (subset), visual builder
- **Three library panels** generated from the same experiment state
- **Command golf** and goal matching like LGB

## Quick start

```bash
npm install
npm run dev
```

Open the printed local URL. Type `help` in the terminal, or `levels` to start learning.

```bash
npm test          # unit tests
npm run typecheck
npm run build
```

## Concepts covered

Tensors and shapes · autograd · perceptron · MLP · activations · losses · optimizers ·
training loops · batching · dropout · batch norm · weight decay · early stopping ·
CNN / conv / pool · RNN / LSTM / GRU · attention · transformers · transfer learning ·
export & inference · and library-specific depth for PyTorch, TensorFlow, and Keras.

## Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md). Design tokens and layout live in
[DESIGN.md](./DESIGN.md).

## License

MIT
