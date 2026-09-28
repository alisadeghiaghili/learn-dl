# Design System — learn-dl

## Style anchor

Educational lab notebook meets interactive terminal: the structure and challenge loop of
`learnGitBranching`, the scientific clarity of TensorFlow Playground, and the calm density
of Observable. Not a marketing site. Not a course LMS. A sandbox you live in.

## Palette

| Token | Hex | Use |
|-------|-----|-----|
| bg | `#0B1220` | app canvas |
| surface | `#151C2C` | panels, terminal chrome |
| surface-2 | `#1C2538` | nested cards, code editors |
| paper | `#F4F1EA` | dialog sheets, goal paper |
| ink | `#E8EEF7` | primary text on dark |
| ink-muted | `#8B9BB4` | secondary, comments |
| ink-paper | `#1A1F2B` | text on paper |
| accent | `#22D3EE` | primary action, active state |
| success | `#34D399` | solved, accuracy up |
| warn | `#FBBF24` | overfit / warning |
| danger | `#F87171` | error, diverging loss |
| pt | `#EE4C2C` | PyTorch |
| tf | `#FF6F00` | TensorFlow |
| keras | `#E11D48` | Keras |

Three library colors stay distinct from UI accent so tabs never collide with actions.

## Typography

| Role | Family | Size / weight |
|------|--------|---------------|
| Display | `Space Grotesk`, `Segoe UI`, system-ui | 28–36px / 600 |
| UI body | `Inter`, `Segoe UI`, system-ui | 14–15px / 400 |
| Mono / code / terminal | `JetBrains Mono`, `Cascadia Code`, ui-monospace | 13px / 400 |
| Labels / axis | mono | 11px / 500, uppercase tracking 0.04em |

Scale contrast is intentional: one large display line per screen (level title or sandbox
wordmark), then dense mono data.

## Layout system

- Desktop-first 3-zone shell (LGB lineage):
  - **Top bar**: logo, mode (Sandbox / Levels), library switcher, progress
  - **Main stage**: network graph + train curves + decision boundary (canvas)
  - **Bottom dock**: DSL terminal (collapsible to 120px / 320px)
  - **Right rail**: tabs — Goal | Code (PT / TF / Keras) | Builder
- Spacing rhythm: 4 / 8 / 12 / 16 / 24 / 32. Dense by default (lab, not landing).
- Max content width for dialogs: 720px paper card.
- Corner radius: 4 (chips), 8 (panels), 12 (dialogs).

## Signature moments

1. **Command → architecture**: typing `net dense 32 relu` draws the layer into the graph
   with shape badges updating live.
2. **`train 10`**: loss/accuracy curves draw left→right; decision boundary morphs under
   the points — the "aha" of gradient descent.
3. **Three-library sync**: any DSL or builder change rewrites the PyTorch / TensorFlow /
   Keras panels side by side. Concept → three real APIs in one glance.

## Interaction rules

- Terminal is always available (Ctrl+` focus).
- `levels` opens the sequence browser; `hint` / `show goal` / `show solution` work inside levels.
- Undo / reset mirror LGB muscle memory (`undo`, `reset`).
- Command golf: count of training-irrelevant vs solution length displayed at solve time.

## Visual language

- Flat fills, 0.5–1px strokes, no gradients on chrome.
- Network nodes: rounded rect, height 44–56, shape badge under label.
- Curves: 2px stroke, accent for loss, success for accuracy.
- Decision boundary: low-opacity field under 2D points.

## Content language

All files, code, comments, and Markdown in English. UI strings ship en_US first; fa
locale is first-class in the i18n map (RTL-ready layout hooks).
