import { useState } from 'react';
import type { DialogView } from '../levels/types';

export interface DialogProps {
  views: DialogView[];
  onClose: () => void;
  onCommand?: (cmd: string) => void;
}

/**
 * Multi-step intro dialog (ModalAlert / DemoView / CodeCompare).
 */
export function Dialog({ views, onClose, onCommand }: DialogProps) {
  const [i, setI] = useState(0);
  const view = views[i];
  if (!view) return null;
  const isLast = i >= views.length - 1;

  const renderMarkdowns = (mds: string[]) =>
    mds.map((line, idx) =>
      line.startsWith('## ') ? (
        <h3 key={idx}>{line.slice(3)}</h3>
      ) : line === '' ? (
        <br key={idx} />
      ) : (
        <p key={idx}>{line}</p>
      ),
    );

  return (
    <div className="dialog-backdrop" role="dialog" aria-modal="true">
      <div className="dialog-paper">
        {view.type === 'ModalAlert' && renderMarkdowns(view.options.markdowns)}
        {view.type === 'DemoView' && (
          <>
            {renderMarkdowns(view.options.beforeMarkdowns)}
            <button
              className="btn accent"
              onClick={() => onCommand?.(view.options.command)}
            >
              Run: {view.options.command}
            </button>
            {renderMarkdowns(view.options.afterMarkdowns)}
          </>
        )}
        {view.type === 'CodeCompare' && (
          <>
            {renderMarkdowns(view.options.markdowns)}
            {view.options.highlight && (
              <p className="muted">Focus panel: {view.options.highlight}</p>
            )}
          </>
        )}
        <div className="dialog-actions">
          <span className="muted">
            {i + 1} / {views.length}
          </span>
          <div>
            {i > 0 && (
              <button className="btn ghost" onClick={() => setI(i - 1)}>
                Back
              </button>
            )}
            {!isLast ? (
              <button className="btn accent" onClick={() => setI(i + 1)}>
                Next
              </button>
            ) : (
              <button className="btn accent" onClick={onClose}>
                Start
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
